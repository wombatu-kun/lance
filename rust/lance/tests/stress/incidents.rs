// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Incidents: storage faults and cancellation points, armed for one step.
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
use std::sync::atomic::{AtomicU8, Ordering};
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

use crate::scenario::Journal;

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
        }
    }
}

impl fmt::Display for Incident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fail { nth, .. }
            | Self::CancelCall { nth, .. }
            | Self::CancelFragment { nth, .. } => write!(f, "{}:n={nth}", self.kind()),
            Self::CancelStage { .. } => f.write_str(&self.kind()),
        }
    }
}

/// What a storage call does under the armed incident.
enum Action {
    Proceed,
    FailBefore(u64),
    FailAfter(u64),
    Cancel(CancelWhen, Cancellation),
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proceed => f.write_str("proceed"),
            Self::FailBefore(fault) => write!(f, "fail_before#{fault}"),
            Self::FailAfter(fault) => write!(f, "fail_after#{fault}"),
            Self::Cancel(when, _) => write!(f, "cancel_{when}"),
        }
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

struct Armed {
    incident: Incident,
    cancellation: Cancellation,
    matched: u64,
    fired: u64,
    /// A `blind` incident has fired: reads of manifests fail.
    blinded: bool,
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
            _ => (Action::Proceed, None),
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
    calls: Mutex<File>,
    started: Instant,
    /// What the run is doing: a step number, its retry, or the final checks.
    phase: Mutex<String>,
    armed: Mutex<Option<Armed>>,
    release: Mutex<Option<watch::Sender<bool>>>,
}

impl fmt::Debug for IncidentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncidentState")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl IncidentState {
    pub fn new(root: String, calls_log: &FsPath, started: Instant) -> Self {
        Self {
            root,
            // A seed's processes share the log, one after the other.
            calls: Mutex::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(calls_log)
                    .expect("open the call log"),
            ),
            started,
            phase: Mutex::new("setup".to_string()),
            armed: Mutex::new(None),
            release: Mutex::new(None),
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
        *self.armed.lock().unwrap() = incident.map(|incident| Armed {
            incident,
            cancellation: Cancellation {
                cancel: cancel.clone(),
                released,
                death,
            },
            matched: 0,
            fired: 0,
            blinded: false,
        });
        cancel
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
        let held = match self.armed.lock().unwrap().as_mut() {
            Some(armed) => armed.on_stage(stage, start),
            None => None,
        };
        self.log(format_args!(
            "stage_{edge} {stage}{}",
            if held.is_some() { " cancel" } else { "" }
        ));
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
        let held = match self.armed.lock().unwrap().as_mut() {
            Some(armed) => armed.on_fragment(begin),
            None => None,
        };
        self.log(format_args!(
            "fragment_{edge} {}{}",
            fragment.id,
            if held.is_some() { " cancel" } else { "" }
        ));
        match held {
            Some(cancellation) => {
                cancellation.hold().await;
                Err(released_error())
            }
            None => Ok(()),
        }
    }
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
        apply(
            action,
            Method::Put,
            location,
            self.inner.put_opts(location, payload, opts),
        )
        .await
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
        let checked = locations
            .and_then(move |location| {
                let result = match state.on_call(Method::Delete, &location) {
                    Action::FailBefore(fault) | Action::FailAfter(fault) => Err(injected(
                        fault,
                        Method::Delete,
                        &location,
                        "before the request",
                    )),
                    _ => Ok(location),
                };
                future::ready(result)
            })
            .boxed();
        self.inner.delete_stream(checked)
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
            _ => self.inner.list(prefix),
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
