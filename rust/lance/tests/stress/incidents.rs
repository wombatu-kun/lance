// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Incidents: storage faults, cancellation points and hold points, armed for
//! one step.
//!
//! [`StressWrapper`] puts [`StressStore`] in front of the dataset's object
//! store, and [`StressHooks`] receives the progress of index builds and
//! writes. Both ask the shared [`IncidentState`] what to do with each call.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::Write as _;
use std::ops::Range;
use std::path::Path as FsPath;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, future};
use lance::dataset::progress::WriteFragmentProgress;
use lance::index::IndexBuildProgress;
use lance_io::object_store::WrappingObjectStore;
use lance_table::format::Fragment;
use object_store::list::PaginatedListStore;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as OSResult,
};
use tokio::sync::{Notify, watch};

use crate::engine::Journal;
use crate::knobs::Key;

/// The kind of a storage call. Reads of whole objects and of ranges are both
/// `Get`, a metadata-only read is `Head`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Put,
    PutMultipart,
    Get,
    Head,
    Delete,
    List,
    Copy,
    Rename,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Put => "put",
            Self::PutMultipart => "put_multipart",
            Self::Get => "get",
            Self::Head => "head",
            Self::Delete => "delete",
            Self::List => "list",
            Self::Copy => "copy",
            Self::Rename => "rename",
        };
        f.write_str(name)
    }
}

/// The part of the dataset a call touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Area {
    Data,
    Indices,
    Deletions,
    Transactions,
    Manifests,
    VersionHint,
    Other,
}

impl fmt::Display for Area {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Data => "data",
            Self::Indices => "indices",
            Self::Deletions => "deletions",
            Self::Transactions => "transactions",
            Self::Manifests => "manifests",
            Self::VersionHint => "version_hint",
            Self::Other => "other",
        };
        f.write_str(name)
    }
}

/// When a cancelled storage call is dropped relative to the request itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelWhen {
    /// The request is never sent.
    Before,
    /// The request is in flight; on a local store the blocking write it
    /// started finishes after the drop. An in-memory store has no flight, so
    /// there this is the same as `After`.
    During,
    /// The request completed, but its caller never resumes.
    After,
}

impl fmt::Display for CancelWhen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Before => "before",
            Self::During => "during",
            Self::After => "after",
        };
        f.write_str(name)
    }
}

/// A slow call: the `nth` matching call waits `millis` before its request or,
/// with `after`, once the request has completed and before its caller sees the
/// response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delay {
    pub method: Method,
    pub area: Area,
    pub nth: u64,
    pub millis: u64,
    pub after: bool,
}

/// A call that never answers: the `nth` matching call, before its request or,
/// with `after`, once the request has completed. The operation gives up on it
/// `grace_ms` later, as an application whose deadline has passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HangPoint {
    pub method: Method,
    pub area: Area,
    pub nth: u64,
    pub after: bool,
    pub grace_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Incident {
    /// The `nth` matching call fails, and with `sticky` every later one too.
    /// With `after_call` the call goes through and only its response is lost.
    /// With `blind`, reads of manifests fail from then on as well, so that a
    /// failed commit cannot find out whether its manifest landed.
    Fail {
        method: Method,
        area: Area,
        nth: u64,
        after_call: bool,
        sticky: bool,
        blind: bool,
    },
    /// The operation is cancelled at the `nth` matching storage call.
    CancelCall {
        method: Method,
        area: Area,
        nth: u64,
        when: CancelWhen,
    },
    /// The operation is cancelled when an index build stage starts or ends.
    CancelStage { stage: &'static str, at_start: bool },
    /// The operation is cancelled when the `nth` fragment write begins or
    /// completes.
    CancelFragment { nth: u64, at_begin: bool },
    /// The operation waits for its gate at the `nth` matching storage call,
    /// before the request or, with `after`, once the request has completed.
    HoldCall {
        method: Method,
        area: Area,
        nth: u64,
        after: bool,
    },
    /// The operation waits for its gate when an index build stage starts or ends.
    HoldStage { stage: &'static str, at_start: bool },
    /// The operation waits for its gate when the `nth` fragment write begins or
    /// completes.
    HoldFragment { nth: u64, at_begin: bool },
    /// A slow store: every one of `delays` slows a call, and `hang` stops one
    /// for good, so that the operation is given up on. It fires when the hang
    /// does, or without one when a delay does.
    Slow {
        delays: Vec<Delay>,
        hang: Option<HangPoint>,
    },
}

impl Incident {
    /// The incident without its position, for coverage counts.
    pub fn kind(&self) -> String {
        match self {
            Self::Fail {
                method,
                area,
                after_call,
                sticky,
                blind,
                ..
            } => format!(
                "fail:{method}:{area}:{}:{}{}",
                if *after_call {
                    "response_lost"
                } else {
                    "request_lost"
                },
                if *sticky { "sticky" } else { "once" },
                if *blind { ":blind" } else { "" }
            ),
            // Which call matters beyond the first: the second manifest of a
            // compaction comes after its reservation has landed.
            Self::CancelCall {
                method,
                area,
                when,
                nth,
            } if *nth > 1 => format!("cancel:{method}:{area}:{when}:call{nth}"),
            Self::CancelCall {
                method, area, when, ..
            } => format!("cancel:{method}:{area}:{when}"),
            Self::CancelStage { stage, at_start } => format!(
                "cancel_stage:{stage}:{}",
                if *at_start { "start" } else { "complete" }
            ),
            Self::CancelFragment { at_begin, .. } => format!(
                "cancel_fragment:{}",
                if *at_begin { "begin" } else { "complete" }
            ),
            Self::HoldCall {
                method,
                area,
                nth,
                after,
            } => format!(
                "hold:{method}:{area}:{}{}",
                if *after { "after" } else { "before" },
                if *nth > 1 {
                    format!(":call{nth}")
                } else {
                    String::new()
                }
            ),
            Self::HoldStage { stage, at_start } => format!(
                "hold_stage:{stage}:{}",
                if *at_start { "start" } else { "complete" }
            ),
            Self::HoldFragment { at_begin, .. } => format!(
                "hold_fragment:{}",
                if *at_begin { "begin" } else { "complete" }
            ),
            // The delays only slow an operation down; a hang decides its end.
            Self::Slow {
                hang: Some(hang), ..
            } => format!(
                "slow:hang:{}:{}:{}",
                hang.method,
                hang.area,
                if hang.after { "after" } else { "before" }
            ),
            Self::Slow { hang: None, .. } => "slow:delays".to_string(),
        }
    }
}

impl fmt::Display for Incident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fail { nth, .. }
            | Self::CancelCall { nth, .. }
            | Self::CancelFragment { nth, .. }
            | Self::HoldCall { nth, .. }
            | Self::HoldFragment { nth, .. } => write!(f, "{}:n={nth}", self.kind()),
            Self::CancelStage { .. } | Self::HoldStage { .. } => f.write_str(&self.kind()),
            Self::Slow { delays, hang } => {
                let side = |after: bool| if after { "after" } else { "before" };
                let mut calls: Vec<String> = delays
                    .iter()
                    .map(|delay| {
                        format!(
                            "delay:{}:{}:n={}:{}ms:{}",
                            delay.method,
                            delay.area,
                            delay.nth,
                            delay.millis,
                            side(delay.after)
                        )
                    })
                    .collect();
                if let Some(hang) = hang {
                    calls.push(format!(
                        "hang:{}:{}:n={}:{}:grace={}ms",
                        hang.method,
                        hang.area,
                        hang.nth,
                        side(hang.after),
                        hang.grace_ms
                    ));
                }
                write!(f, "slow({})", calls.join(","))
            }
        }
    }
}

/// What a storage call does under the armed incident.
enum Action {
    Proceed,
    FailBefore(u64),
    FailAfter(u64),
    Cancel(CancelWhen, Cancellation),
    Hold {
        gate: Gate,
        after: bool,
    },
    Delay {
        millis: u64,
        after: bool,
        watch: Option<Watch>,
    },
    Hang(Hang),
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proceed => f.write_str("proceed"),
            Self::FailBefore(fault) => write!(f, "fail_before#{fault}"),
            Self::FailAfter(fault) => write!(f, "fail_after#{fault}"),
            Self::Cancel(when, _) => write!(f, "cancel_{when}"),
            Self::Hold { after: false, .. } => f.write_str("hold_before"),
            Self::Hold { after: true, .. } => f.write_str("hold_after"),
            Self::Delay {
                millis,
                after: false,
                ..
            } => write!(f, "delay_before#{millis}ms"),
            Self::Delay {
                millis,
                after: true,
                ..
            } => write!(f, "delay_after#{millis}ms"),
            Self::Hang(hang) if hang.after => f.write_str("hang_after"),
            Self::Hang(_) => f.write_str("hang_before"),
        }
    }
}

/// A call that never answers. It fires the cancellation once its grace has
/// passed, which drops the operation; the call itself ends only when its future
/// is dropped or the step lets it go, so one in a task the operation detached
/// outlives the operation. It counts from the moment it meets the store wrapper,
/// its own request included.
struct Hang {
    cancellation: Cancellation,
    grace: Duration,
    after: bool,
    _live: Live,
    released: watch::Receiver<bool>,
}

impl Hang {
    async fn wait(mut self) {
        tokio::time::sleep(self.grace).await;
        self.cancellation.fire();
        // An error only means that the step dropped its sender, which lets it go too.
        let _ = self.released.wait_for(|released| *released).await;
    }
}

/// How long after its operation a slow call may still end. A task the operation
/// aborted finishes the poll it is in, so a call that ends a moment after the
/// operation may be that last poll's rather than a task's left behind.
pub const LATE_AFTER: Key<Duration> = Key::new("late_after");
/// What `LATE_AFTER` is by default, and in a scenario that slows no call.
pub const DEFAULT_LATE_AFTER: Duration = Duration::from_millis(50);

/// What the slow and hung calls of a step count themselves in, which a call in
/// the operation's own futures leaves when the operation is dropped: only a task
/// the operation left behind keeps a call counted once it is over.
#[derive(Clone)]
struct Watch {
    /// How long after the operation a call that ends counts as late.
    late_after: Duration,
    /// Hung reads of data and index files, which Lance's I/O scheduler runs as
    /// tasks of their own that a dropped scan does not stop: one only holds its
    /// connection.
    scan_reads: Arc<AtomicU64>,
    /// Every other slow or hung call still in flight.
    calls: Arc<AtomicU64>,
    /// Slow calls, but scan reads, that ended well after their operation was over.
    late: Arc<AtomicU64>,
    /// When the operation of the step ended.
    ended: Arc<Mutex<Option<Instant>>>,
}

impl Watch {
    fn new(late_after: Duration) -> Self {
        Self {
            late_after,
            scan_reads: Arc::default(),
            calls: Arc::default(),
            late: Arc::default(),
            ended: Arc::default(),
        }
    }

    /// Runs a slow call, counted while in flight, and as late when it ends well
    /// after its operation.
    async fn track<T>(&self, call: impl Future<Output = T>) -> T {
        let _live = Live::count(&self.calls);
        let result = call.await;
        let ended = *self.ended.lock().unwrap();
        if ended.is_some_and(|ended| ended.elapsed() > self.late_after) {
            self.late.fetch_add(1, Ordering::SeqCst);
        }
        result
    }
}

/// What a step's slow and hung calls left behind once its operation was over.
pub struct LeftBehind {
    pub scan_reads: u64,
    pub calls: u64,
    pub late: u64,
}

/// One slow or hung call, counted while its future lives.
struct Live(Arc<AtomicU64>);

impl Live {
    fn count(live: &Arc<AtomicU64>) -> Self {
        live.fetch_add(1, Ordering::SeqCst);
        Self(live.clone())
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The line that tells the parent a SIGKILL was planned.
pub const CRASH_FIRED: &str = "crash_fired";

/// Sends this process the SIGKILL that nothing can catch or clean up after.
pub fn kill_this_process() -> ! {
    // SAFETY: `kill` only sends a signal, here to this very process.
    unsafe { libc::kill(libc::getpid(), libc::SIGKILL) };
    loop {
        std::thread::park();
    }
}

const KILLER_WAITS: u8 = 0;
const KILLER_GOES: u8 = 1;
const KILLER_STANDS_DOWN: u8 = 2;

/// Turns a cancellation point into the death of the process.
#[derive(Clone)]
pub struct Death {
    journal: Arc<Journal>,
    /// What the thread that kills during a request is told, if there is one.
    killer: Arc<AtomicU8>,
}

impl Death {
    /// With `in_flight`, a thread stands ready to kill that long after a request
    /// starts. It is running already and spins, because a small write is over in
    /// a time shorter than a thread takes to start or to wake up.
    pub fn new(journal: Arc<Journal>, in_flight: Option<Duration>) -> Self {
        let death = Self {
            journal,
            killer: Arc::new(AtomicU8::new(KILLER_WAITS)),
        };
        if let Some(in_flight) = in_flight {
            let killer = death.clone();
            std::thread::spawn(move || killer.wait_and_kill(in_flight));
        }
        death
    }

    fn wait_and_kill(&self, in_flight: Duration) {
        loop {
            match self.killer.load(Ordering::Acquire) {
                KILLER_GOES => break,
                KILLER_STANDS_DOWN => return,
                _ => std::thread::yield_now(),
            }
        }
        let started = Instant::now();
        while started.elapsed() < in_flight {
            std::hint::spin_loop();
        }
        self.now();
    }

    fn now(&self) -> ! {
        self.journal.line(CRASH_FIRED);
        kill_this_process()
    }

    fn while_the_request_runs(&self) {
        self.killer.store(KILLER_GOES, Ordering::Release);
    }

    /// The step is over. A killer that was told to go has a kill to deliver.
    fn stand_down(&self) {
        self.killer
            .compare_exchange(
                KILLER_WAITS,
                KILLER_STANDS_DOWN,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok();
    }
}

/// The signals of one armed step: `cancel` tells the step to drop its
/// operation, `released` lets a call that was held back go on once the step
/// is over. A held call can sit in a task the operation detached, which a
/// drop does not reach, so holding it forever would leak that task. With
/// `death`, the point kills the process instead of cancelling the operation.
#[derive(Clone)]
struct Cancellation {
    cancel: Arc<Notify>,
    released: watch::Receiver<bool>,
    death: Option<Death>,
}

impl Cancellation {
    fn fire(&self) {
        match &self.death {
            Some(death) => death.now(),
            None => self.cancel.notify_one(),
        }
    }

    fn fire_during_the_request(&self) {
        match &self.death {
            Some(death) => death.while_the_request_runs(),
            None => self.cancel.notify_one(),
        }
    }

    async fn hold(mut self) {
        self.fire();
        // An error only means the step is over, which releases the call too.
        let _ = self.released.wait_for(|released| *released).await;
    }
}

/// What a hold point waits for: the end of another operation, or its arrival at
/// a point of its own. A limit makes the plan's order a wish rather than a
/// promise, so that a planned wait that cannot end costs time and shows in the
/// count of timeouts, while only a gate without a limit can hang a run.
#[derive(Clone)]
pub struct Gate {
    opened: watch::Receiver<bool>,
    limit: Option<Duration>,
    timeouts: Arc<AtomicU64>,
    /// How many times the state that armed it found it still closed: a point can
    /// fire after the gate has opened, when nothing is held up.
    waits: Arc<AtomicU64>,
    /// Told that the holder got to its point, before it waits.
    arrival: Option<Arc<watch::Sender<bool>>>,
}

impl Gate {
    pub fn new(limit: Option<Duration>, timeouts: Arc<AtomicU64>) -> (watch::Sender<bool>, Self) {
        let (open, opened) = watch::channel(false);
        (
            open,
            Self {
                opened,
                limit,
                timeouts,
                waits: Arc::default(),
                arrival: None,
            },
        )
    }

    /// The same gate, which tells `arrival` when its holder gets to the point.
    pub fn announcing(mut self, arrival: Arc<watch::Sender<bool>>) -> Self {
        self.arrival = Some(arrival);
        self
    }

    /// Waits until the gate opens or its limit passes.
    pub async fn pass(&self) {
        if let Some(arrival) = &self.arrival {
            arrival.send_replace(true);
        }
        let mut opened = self.opened.clone();
        if !*opened.borrow() {
            self.waits.fetch_add(1, Ordering::SeqCst);
        }
        let wait = opened.wait_for(|opened| *opened);
        // An error only means that the sender is gone, so nothing will come.
        match self.limit {
            Some(limit) => {
                if tokio::time::timeout(limit, wait).await.is_err() {
                    self.timeouts.fetch_add(1, Ordering::SeqCst);
                }
            }
            None => {
                let _ = wait.await;
            }
        }
    }
}

struct Armed {
    incident: Incident,
    cancellation: Cancellation,
    gate: Option<Gate>,
    matched: u64,
    fired: u64,
    /// A `blind` incident has fired: reads of manifests fail.
    blinded: bool,
    /// For `Slow`: the matching calls that each delay and the hang have seen.
    delays_matched: Vec<u64>,
    hang_matched: u64,
    watch: Watch,
    /// What a hang waits for: not the step's release, which comes as soon as the
    /// operation ends.
    hang_released: watch::Receiver<bool>,
}

impl Armed {
    /// The action for a call, and its number among the calls the incident
    /// targets, if it is one of them.
    fn on_call(&mut self, method: Method, area: Area) -> (Action, Option<u64>) {
        if self.blinded && matches!(method, Method::Get | Method::Head) && area == Area::Manifests {
            self.fired += 1;
            return (Action::FailBefore(self.fired), None);
        }
        match self.incident {
            Incident::Fail {
                method: target_method,
                area: target_area,
                nth,
                after_call,
                sticky,
                blind,
            } if target_method == method && target_area == area => {
                self.matched += 1;
                let action = if self.matched == nth || (sticky && self.matched > nth) {
                    self.fired += 1;
                    self.blinded = blind;
                    if after_call {
                        Action::FailAfter(self.fired)
                    } else {
                        Action::FailBefore(self.fired)
                    }
                } else {
                    Action::Proceed
                };
                (action, Some(self.matched))
            }
            Incident::CancelCall {
                method: target_method,
                area: target_area,
                nth,
                when,
            } if target_method == method && target_area == area => {
                self.matched += 1;
                let action = if self.matched == nth {
                    self.fired += 1;
                    Action::Cancel(when, self.cancellation.clone())
                } else {
                    Action::Proceed
                };
                (action, Some(self.matched))
            }
            Incident::HoldCall {
                method: target_method,
                area: target_area,
                nth,
                after,
            } if target_method == method && target_area == area => {
                self.matched += 1;
                let action = match &self.gate {
                    Some(gate) if self.matched == nth => {
                        self.fired += 1;
                        Action::Hold {
                            gate: gate.clone(),
                            after,
                        }
                    }
                    _ => Action::Proceed,
                };
                (action, Some(self.matched))
            }
            // Every target counts every call it matches, so that two of them on
            // the same kind of call agree on its number. A hung call is not slow.
            Incident::Slow {
                ref delays,
                hang: hang_point,
            } => {
                let mut action = Action::Proceed;
                let mut matched = None;
                if let Some(point) = hang_point
                    && point.method == method
                    && point.area == area
                {
                    self.hang_matched += 1;
                    matched = Some(self.hang_matched);
                    if self.hang_matched == point.nth {
                        self.fired += 1;
                        action = Action::Hang(Hang {
                            cancellation: self.cancellation.clone(),
                            grace: Duration::from_millis(point.grace_ms),
                            after: point.after,
                            _live: Live::count(match scan_read(method, area) {
                                true => &self.watch.scan_reads,
                                false => &self.watch.calls,
                            }),
                            released: self.hang_released.clone(),
                        });
                    }
                }
                for (delay, seen) in delays.iter().zip(self.delays_matched.iter_mut()) {
                    if delay.method == method && delay.area == area {
                        *seen += 1;
                        matched = Some(*seen);
                        if *seen == delay.nth && matches!(action, Action::Proceed) {
                            // With a hang planned, it alone counts: the delays only
                            // slow the way to it.
                            if hang_point.is_none() {
                                self.fired += 1;
                            }
                            action = Action::Delay {
                                millis: delay.millis,
                                after: delay.after,
                                watch: (!scan_read(method, area)).then(|| self.watch.clone()),
                            };
                        }
                    }
                }
                (action, matched)
            }
            _ => (Action::Proceed, None),
        }
    }

    fn gate_at_stage(&mut self, stage: &str, start: bool) -> Option<Gate> {
        match self.incident {
            Incident::HoldStage {
                stage: target,
                at_start,
            } if target == stage && at_start == start && self.fired == 0 => {
                self.fired += 1;
                self.gate.clone()
            }
            _ => None,
        }
    }

    fn gate_at_fragment(&mut self, begin: bool) -> Option<Gate> {
        match self.incident {
            Incident::HoldFragment { nth, at_begin } if at_begin == begin => {
                self.matched += 1;
                if self.matched == nth {
                    self.fired += 1;
                    self.gate.clone()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn on_stage(&mut self, stage: &str, start: bool) -> Option<Cancellation> {
        match self.incident {
            Incident::CancelStage {
                stage: target,
                at_start,
            } if target == stage && at_start == start && self.fired == 0 => {
                self.fired += 1;
                Some(self.cancellation.clone())
            }
            _ => None,
        }
    }

    fn on_fragment(&mut self, begin: bool) -> Option<Cancellation> {
        match self.incident {
            Incident::CancelFragment { nth, at_begin } if at_begin == begin => {
                self.matched += 1;
                (self.matched == nth).then(|| {
                    self.fired += 1;
                    self.cancellation.clone()
                })
            }
            _ => None,
        }
    }
}

/// The incident armed for the current step, shared by the store wrapper and
/// the progress hooks, plus the log of every call they see.
pub struct IncidentState {
    /// The object path of the dataset root, to tell its directories apart.
    root: String,
    calls: Arc<Mutex<File>>,
    started: Instant,
    /// What the run is doing: a step number, its retry, or the final checks.
    phase: Mutex<String>,
    armed: Mutex<Option<Armed>>,
    release: Mutex<Option<watch::Sender<bool>>>,
    /// What the slow and hung calls count themselves in, and what lets the hung
    /// ones go: only the step, once it has counted them, since a disarm comes as
    /// soon as the operation ends.
    watch: Watch,
    hang_release: Mutex<Option<watch::Sender<bool>>>,
    /// Writes that reached the store since `count_writes`.
    writes: AtomicU64,
    counting_writes: AtomicBool,
    /// The logical clock of a round, and when it last saw this state's manifest
    /// land. A `concurrency` round compares these between its operations.
    clock: Arc<AtomicU64>,
    last_commit: AtomicU64,
    /// Manifest writes that found the version already taken by another writer.
    lost_races: AtomicU64,
    gate_waits: Arc<AtomicU64>,
}

impl fmt::Debug for IncidentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncidentState")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl IncidentState {
    /// A slow call that ends more than `late_after` after its operation is late.
    pub fn new(root: String, calls_log: &FsPath, started: Instant, late_after: Duration) -> Self {
        Self {
            root,
            // A seed's processes share the log, one after the other.
            calls: Arc::new(Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(calls_log)
                    .expect("open the call log"),
            )),
            started,
            phase: Mutex::new("setup".to_string()),
            armed: Mutex::new(None),
            release: Mutex::new(None),
            watch: Watch::new(late_after),
            hang_release: Mutex::new(None),
            writes: AtomicU64::new(0),
            counting_writes: AtomicBool::new(false),
            clock: Arc::new(AtomicU64::new(0)),
            last_commit: AtomicU64::new(0),
            lost_races: AtomicU64::new(0),
            gate_waits: Arc::default(),
        }
    }

    /// A state of its own for one writer or reader of a round. The call log stays
    /// the one that was opened before the baseline of open descriptors.
    pub fn sibling(&self, clock: Arc<AtomicU64>) -> Self {
        Self {
            root: self.root.clone(),
            calls: self.calls.clone(),
            started: self.started,
            phase: Mutex::new(self.phase()),
            armed: Mutex::new(None),
            release: Mutex::new(None),
            watch: Watch::new(self.watch.late_after),
            hang_release: Mutex::new(None),
            writes: AtomicU64::new(0),
            counting_writes: AtomicBool::new(false),
            clock,
            last_commit: AtomicU64::new(0),
            lost_races: AtomicU64::new(0),
            gate_waits: Arc::default(),
        }
    }

    /// When this state last saw a manifest land, on the clock of the round.
    pub fn last_commit(&self) -> u64 {
        self.last_commit.load(Ordering::SeqCst)
    }

    pub fn take_gate_waits(&self) -> u64 {
        self.gate_waits.swap(0, Ordering::SeqCst)
    }

    pub fn take_lost_races(&self) -> u64 {
        self.lost_races.swap(0, Ordering::SeqCst)
    }

    fn note_manifest_put(&self, result: &OSResult<PutResult>, location: &Path) {
        match result {
            Ok(_) => {
                let tick = self.clock.fetch_add(1, Ordering::SeqCst) + 1;
                self.last_commit.store(tick, Ordering::SeqCst);
            }
            Err(object_store::Error::AlreadyExists { .. }) => {
                self.lost_races.fetch_add(1, Ordering::SeqCst);
                self.log(format_args!("lost_manifest_race {location}"));
            }
            Err(_) => {}
        }
    }

    pub fn set_phase(&self, phase: impl Into<String>) {
        *self.phase.lock().unwrap() = phase.into();
    }

    pub fn phase(&self) -> String {
        self.phase.lock().unwrap().clone()
    }

    /// Arms `incident` and returns the signal that cancels the operation it
    /// is armed for. Without an incident the signal never fires. With `death`,
    /// the incident kills the process rather than cancelling the operation.
    pub fn arm(&self, incident: Option<Incident>, death: Option<Death>) -> Arc<Notify> {
        let cancel = Arc::new(Notify::new());
        let (release, released) = watch::channel(false);
        *self.release.lock().unwrap() = Some(release);
        let (hang_release, hang_released) = watch::channel(false);
        if matches!(incident, Some(Incident::Slow { hang: Some(_), .. })) {
            *self.hang_release.lock().unwrap() = Some(hang_release);
        }
        let delays = match &incident {
            Some(Incident::Slow { delays, .. }) => delays.len(),
            _ => 0,
        };
        *self.armed.lock().unwrap() = incident.map(|incident| Armed {
            incident,
            cancellation: Cancellation {
                cancel: cancel.clone(),
                released,
                death,
            },
            gate: None,
            matched: 0,
            fired: 0,
            blinded: false,
            delays_matched: vec![0; delays],
            hang_matched: 0,
            watch: self.watch.clone(),
            hang_released,
        });
        cancel
    }

    /// Marks the operation of a step begun: none of its calls is late yet.
    pub fn begin_operation(&self) {
        *self.watch.ended.lock().unwrap() = None;
        self.watch.late.store(0, Ordering::SeqCst);
    }

    /// Marks the operation over: a slow call that ends well after this was left
    /// behind by it.
    pub fn end_operation(&self) {
        *self.watch.ended.lock().unwrap() = Some(Instant::now());
    }

    /// The slow and hung calls still in flight but scan reads.
    pub fn calls_in_flight(&self) -> u64 {
        self.watch.calls.load(Ordering::SeqCst)
    }

    /// The hung scan reads still in flight.
    pub fn scan_reads_in_flight(&self) -> u64 {
        self.watch.scan_reads.load(Ordering::SeqCst)
    }

    /// Counts the writes that reach the store from now on, whatever is armed.
    pub fn count_writes(&self) {
        self.writes.store(0, Ordering::SeqCst);
        self.counting_writes.store(true, Ordering::SeqCst);
    }

    /// Stops counting writes, and tells how many came while it counted. A write
    /// that raced the stop is left over in the count, so a step that never
    /// counted gets none of it.
    pub fn writes_counted(&self) -> u64 {
        let counting = self.counting_writes.swap(false, Ordering::SeqCst);
        let writes = self.writes.swap(0, Ordering::SeqCst);
        if counting { writes } else { 0 }
    }

    /// What the operation left behind. A task it aborted drops its calls a
    /// moment after the abort, so this waits up to `limit` for them.
    pub async fn settle_left_behind(&self, limit: Duration) -> LeftBehind {
        let started = Instant::now();
        loop {
            let scan_reads = self.watch.scan_reads.load(Ordering::SeqCst);
            let calls = self.watch.calls.load(Ordering::SeqCst);
            if (scan_reads == 0 && calls == 0) || started.elapsed() >= limit {
                return LeftBehind {
                    scan_reads,
                    calls,
                    late: self.watch.late.swap(0, Ordering::SeqCst),
                };
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Lets every hung call end, with an error.
    pub fn release_hung_calls(&self) {
        if let Some(release) = self.hang_release.lock().unwrap().take() {
            release.send_replace(true);
        }
    }

    /// Arms a hold point: the operation waits at it until `gate` lets it go.
    pub fn arm_hold(&self, point: Incident, mut gate: Gate) {
        gate.waits = self.gate_waits.clone();
        self.arm(Some(point), None);
        if let Some(armed) = self.armed.lock().unwrap().as_mut() {
            armed.gate = Some(gate);
        }
    }

    /// Disarms the incident, releases every call it held back, and returns
    /// how many times it fired.
    pub fn disarm(&self) -> u64 {
        let armed = self.armed.lock().unwrap().take();
        if let Some(death) = armed
            .as_ref()
            .and_then(|armed| armed.cancellation.death.as_ref())
        {
            death.stand_down();
        }
        let fired = armed.map_or(0, |armed| armed.fired);
        if let Some(release) = self.release.lock().unwrap().take() {
            release.send_replace(true);
        }
        fired
    }

    fn area(&self, location: &Path) -> Area {
        let Some(rest) = location.as_ref().strip_prefix(self.root.as_str()) else {
            return Area::Other;
        };
        let rest = rest.trim_start_matches('/');
        match rest.split('/').next().unwrap_or_default() {
            "data" => Area::Data,
            "_indices" => Area::Indices,
            "_deletions" => Area::Deletions,
            "_transactions" => Area::Transactions,
            "_versions" if rest.ends_with("latest_version_hint.json") => Area::VersionHint,
            "_versions" => Area::Manifests,
            _ => Area::Other,
        }
    }

    fn log(&self, event: fmt::Arguments<'_>) {
        let line = format!(
            "t={} phase={} {event}\n",
            self.started.elapsed().as_millis(),
            self.phase.lock().unwrap()
        );
        self.calls
            .lock()
            .unwrap()
            .write_all(line.as_bytes())
            .expect("write the call log");
    }

    fn on_call(&self, method: Method, location: &Path) -> Action {
        let area = self.area(location);
        if writes(method) && self.counting_writes.load(Ordering::SeqCst) {
            self.writes.fetch_add(1, Ordering::SeqCst);
        }
        let (action, matched) = match self.armed.lock().unwrap().as_mut() {
            Some(armed) => armed.on_call(method, area),
            None => (Action::Proceed, None),
        };
        match matched {
            Some(matched) => self.log(format_args!(
                "{method} {area} {location} {action} match={matched}"
            )),
            None => self.log(format_args!("{method} {area} {location} {action}")),
        }
        action
    }

    async fn on_stage(&self, stage: &str, start: bool) -> lance_core::Result<()> {
        let edge = if start { "start" } else { "complete" };
        let (held, gate) = match self.armed.lock().unwrap().as_mut() {
            Some(armed) => (
                armed.on_stage(stage, start),
                armed.gate_at_stage(stage, start),
            ),
            None => (None, None),
        };
        self.log(format_args!(
            "stage_{edge} {stage}{}{}",
            if held.is_some() { " cancel" } else { "" },
            if gate.is_some() { " hold" } else { "" }
        ));
        if let Some(gate) = gate {
            gate.pass().await;
        }
        match held {
            Some(cancellation) => {
                cancellation.hold().await;
                Err(released_error())
            }
            None => Ok(()),
        }
    }

    async fn on_fragment(&self, fragment: &Fragment, begin: bool) -> lance_core::Result<()> {
        let edge = if begin { "begin" } else { "complete" };
        let (held, gate) = match self.armed.lock().unwrap().as_mut() {
            Some(armed) => (armed.on_fragment(begin), armed.gate_at_fragment(begin)),
            None => (None, None),
        };
        self.log(format_args!(
            "fragment_{edge} {}{}{}",
            fragment.id,
            if held.is_some() { " cancel" } else { "" },
            if gate.is_some() { " hold" } else { "" }
        ));
        if let Some(gate) = gate {
            gate.pass().await;
        }
        match held {
            Some(cancellation) => {
                cancellation.hold().await;
                Err(released_error())
            }
            None => Ok(()),
        }
    }
}

/// A read of a data or index file: the only call Lance leaves behind by design,
/// as its I/O scheduler runs each one as a task of its own.
fn scan_read(method: Method, area: Area) -> bool {
    method == Method::Get && matches!(area, Area::Data | Area::Indices)
}

/// A call that changes what the store holds.
fn writes(method: Method) -> bool {
    matches!(
        method,
        Method::Put | Method::PutMultipart | Method::Delete | Method::Copy | Method::Rename
    )
}

fn released_error() -> lance_core::Error {
    lance_core::Error::io("stress: a call held back for a cancellation was released")
}

fn injected(fault: u64, method: Method, location: &Path, when: &str) -> object_store::Error {
    object_store::Error::Generic {
        store: "stress",
        source: format!("injected fault #{fault}: {method} {location} failed {when}").into(),
    }
}

/// Puts a [`StressStore`] in front of every object store the dataset opens.
#[derive(Debug)]
pub struct StressWrapper {
    state: Arc<IncidentState>,
}

impl StressWrapper {
    pub fn new(state: Arc<IncidentState>) -> Self {
        Self { state }
    }
}

impl WrappingObjectStore for StressWrapper {
    fn wrap(&self, _store_prefix: &str, original: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(StressStore {
            inner: original,
            state: self.state.clone(),
        })
    }

    // Every request has to meet the armed incident, so listings give up the
    // pagination pushdown.
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

#[derive(Debug)]
pub struct StressStore {
    inner: Arc<dyn ObjectStore>,
    state: Arc<IncidentState>,
}

impl fmt::Display for StressStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StressStore({})", self.inner)
    }
}

impl StressStore {
    async fn run<T>(
        &self,
        method: Method,
        location: &Path,
        call: impl Future<Output = OSResult<T>>,
    ) -> OSResult<T> {
        let action = self.state.on_call(method, location);
        apply(action, method, location, call).await
    }
}

/// Runs `call` under `action`. `call` is lazy, so a call that fails or is
/// held before the request never reaches the inner store.
async fn apply<T>(
    action: Action,
    method: Method,
    location: &Path,
    call: impl Future<Output = OSResult<T>>,
) -> OSResult<T> {
    match action {
        Action::Proceed => call.await,
        Action::FailBefore(fault) => Err(injected(fault, method, location, "before the request")),
        Action::FailAfter(fault) => {
            drop(call.await);
            Err(injected(
                fault,
                method,
                location,
                "after the request, losing its response",
            ))
        }
        Action::Cancel(CancelWhen::Before, cancellation) => {
            cancellation.hold().await;
            Err(injected(
                0,
                method,
                location,
                "after a cancellation released it",
            ))
        }
        // An in-memory store answers without ever yielding, so each arm
        // yields once after the call, as a real store's response would,
        // and the step drops the operation before it sees the result.
        Action::Cancel(CancelWhen::During, cancellation) => {
            cancellation.fire_during_the_request();
            let result = call.await;
            tokio::task::yield_now().await;
            result
        }
        Action::Cancel(CancelWhen::After, cancellation) => {
            let result = call.await;
            cancellation.fire();
            tokio::task::yield_now().await;
            result
        }
        Action::Hold { gate, after: false } => {
            gate.pass().await;
            call.await
        }
        Action::Hold { gate, after: true } => {
            let result = call.await;
            gate.pass().await;
            result
        }
        Action::Delay {
            millis,
            after,
            watch,
        } => {
            let slowed = async move {
                let pause = Duration::from_millis(millis);
                if after {
                    let result = call.await;
                    tokio::time::sleep(pause).await;
                    result
                } else {
                    tokio::time::sleep(pause).await;
                    call.await
                }
            };
            match watch {
                Some(watch) => watch.track(slowed).await,
                None => slowed.await,
            }
        }
        Action::Hang(hang) => {
            if hang.after {
                drop(call.await);
            }
            hang.wait().await;
            Err(injected(0, method, location, "after a hang was let go"))
        }
    }
}

#[async_trait]
impl ObjectStore for StressStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        let action = self.state.on_call(Method::Put, location);
        if let Action::Cancel(CancelWhen::During, cancellation) = &action {
            // The request runs as a task of its own: dropping the operation
            // does not stop a request in flight, and the runtime counts the
            // task until it lands, so the step's quiet phase waits for it.
            let inner = self.inner.clone();
            let target = location.clone();
            let request = tokio::spawn(async move { inner.put_opts(&target, payload, opts).await });
            cancellation.fire_during_the_request();
            tokio::task::yield_now().await;
            return request
                .await
                .map_err(|error| object_store::Error::Generic {
                    store: "stress",
                    source: Box::new(error),
                })?;
        }
        let is_manifest = self.state.area(location) == Area::Manifests;
        let call = async {
            let result = self.inner.put_opts(location, payload, opts).await;
            if is_manifest {
                self.state.note_manifest_put(&result, location);
            }
            result
        };
        apply(action, Method::Put, location, call).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.run(
            Method::PutMultipart,
            location,
            self.inner.put_multipart_opts(location, opts),
        )
        .await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        let method = if options.head {
            Method::Head
        } else {
            Method::Get
        };
        self.run(method, location, self.inner.get_opts(location, options))
            .await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.run(
            Method::Get,
            location,
            self.inner.get_ranges(location, ranges),
        )
        .await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        let state = self.state.clone();
        // `delete` passes one location at a time, so a cancellation after a
        // deletion fires with the next result of the inner store.
        let after_deletion: Arc<Mutex<Vec<Cancellation>>> = Arc::default();
        let pending = after_deletion.clone();
        let checked = locations
            .and_then(move |location| {
                let action = state.on_call(Method::Delete, &location);
                let pending = pending.clone();
                async move {
                    match action {
                        Action::FailBefore(fault) | Action::FailAfter(fault) => Err(injected(
                            fault,
                            Method::Delete,
                            &location,
                            "before the request",
                        )),
                        Action::Cancel(CancelWhen::Before, cancellation) => {
                            cancellation.hold().await;
                            Err(injected(
                                0,
                                Method::Delete,
                                &location,
                                "after a cancellation released it",
                            ))
                        }
                        Action::Cancel(CancelWhen::During, cancellation) => {
                            cancellation.fire_during_the_request();
                            Ok(location)
                        }
                        Action::Cancel(CancelWhen::After, cancellation) => {
                            pending.lock().unwrap().push(cancellation);
                            Ok(location)
                        }
                        Action::Proceed | Action::Hold { .. } => Ok(location),
                        // The plan slows or hangs a deletion only before its request.
                        Action::Delay { millis, watch, .. } => {
                            let pause = tokio::time::sleep(Duration::from_millis(millis));
                            match watch {
                                Some(watch) => watch.track(pause).await,
                                None => pause.await,
                            }
                            Ok(location)
                        }
                        Action::Hang(hang) => {
                            hang.wait().await;
                            Err(injected(
                                0,
                                Method::Delete,
                                &location,
                                "after a hang was let go",
                            ))
                        }
                    }
                }
            })
            .boxed();
        self.inner
            .delete_stream(checked)
            .then(move |result| {
                let due: Vec<Cancellation> = after_deletion.lock().unwrap().drain(..).collect();
                async move {
                    if !due.is_empty() {
                        due.iter().for_each(Cancellation::fire);
                        tokio::task::yield_now().await;
                    }
                    result
                }
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let location = prefix.cloned().unwrap_or_default();
        match self.state.on_call(Method::List, &location) {
            Action::FailBefore(fault) | Action::FailAfter(fault) => {
                futures::stream::once(future::ready(Err(injected(
                    fault,
                    Method::List,
                    &location,
                    "before the request",
                ))))
                .boxed()
            }
            Action::Hold { gate, .. } => {
                let inner = self.inner.clone();
                let prefix = prefix.cloned();
                futures::stream::once(async move {
                    gate.pass().await;
                    inner.list(prefix.as_ref())
                })
                .flatten()
                .boxed()
            }
            Action::Cancel(CancelWhen::Before, cancellation) => futures::stream::once(async move {
                cancellation.hold().await;
                Err(injected(
                    0,
                    Method::List,
                    &location,
                    "after a cancellation released it",
                ))
            })
            .boxed(),
            Action::Cancel(CancelWhen::During, cancellation) => {
                cancellation.fire_during_the_request();
                self.inner.list(prefix)
            }
            Action::Cancel(CancelWhen::After, cancellation) => {
                let end = futures::stream::once(async move {
                    cancellation.fire();
                    tokio::task::yield_now().await;
                })
                .filter_map(|()| future::ready(None));
                self.inner.list(prefix).chain(end).boxed()
            }
            Action::Proceed => self.inner.list(prefix),
            Action::Delay {
                millis,
                after,
                watch,
            } => {
                let pause = async move {
                    let pause = tokio::time::sleep(Duration::from_millis(millis));
                    match watch {
                        Some(watch) => watch.track(pause).await,
                        None => pause.await,
                    }
                };
                if after {
                    let end = futures::stream::once(pause).filter_map(|()| future::ready(None));
                    self.inner.list(prefix).chain(end).boxed()
                } else {
                    let inner = self.inner.clone();
                    let prefix = prefix.cloned();
                    futures::stream::once(async move {
                        pause.await;
                        inner.list(prefix.as_ref())
                    })
                    .flatten()
                    .boxed()
                }
            }
            Action::Hang(hang) => {
                let after = hang.after;
                let hung = futures::stream::once(async move {
                    hang.wait().await;
                    Err(injected(
                        0,
                        Method::List,
                        &location,
                        "after a hang was let go",
                    ))
                });
                if after {
                    self.inner.list(prefix).chain(hung).boxed()
                } else {
                    hung.boxed()
                }
            }
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let location = prefix.cloned().unwrap_or_default();
        self.run(
            Method::List,
            &location,
            self.inner.list_with_delimiter(prefix),
        )
        .await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.run(Method::Copy, from, self.inner.copy_opts(from, to, opts))
            .await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        self.run(Method::Rename, from, self.inner.rename_opts(from, to, opts))
            .await
    }
}

/// Progress hooks that cancel an operation at a named point.
#[derive(Debug)]
pub struct StressHooks {
    state: Arc<IncidentState>,
}

impl StressHooks {
    pub fn new(state: Arc<IncidentState>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl IndexBuildProgress for StressHooks {
    async fn stage_start(
        &self,
        stage: &str,
        _total: Option<u64>,
        _unit: &str,
    ) -> lance_core::Result<()> {
        self.state.on_stage(stage, true).await
    }

    async fn stage_progress(&self, _stage: &str, _completed: u64) -> lance_core::Result<()> {
        Ok(())
    }

    async fn stage_complete(&self, stage: &str) -> lance_core::Result<()> {
        self.state.on_stage(stage, false).await
    }
}

#[async_trait]
impl WriteFragmentProgress for StressHooks {
    async fn begin(&self, fragment: &Fragment) -> lance_core::Result<()> {
        self.state.on_fragment(fragment, true).await
    }

    async fn complete(&self, fragment: &Fragment) -> lance_core::Result<()> {
        self.state.on_fragment(fragment, false).await
    }
}
