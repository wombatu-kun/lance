// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The `concurrency` scenario: writers and readers at once on one dataset.
//!
//! A step is a round: workers run their operations in parallel on handles of
//! their own, readers check snapshots meanwhile, and once it is quiet every
//! version the round made is opened and checked.
//!
//! A row belongs to the class `id % (workers + 2)`, whatever fragment it lies
//! in, and a worker only changes the rows of its own class. So what a worker's
//! rows hold depends on its own operations alone, and every snapshot must show
//! them as of some prefix of those operations, while the operations still fight
//! over the same fragments. Two more classes are shared. All workers upsert, update
//! and delete the same keys of the first, so an operation must apply to all of them
//! or to none. All workers add 1 to the whole of the second, a value computed from
//! what was read, so that a lost increment shows exactly.
//!
//! Which rebases happen is not left to timing alone. Some rounds force a pair of
//! `PAIRS`, the others committing over what the first worker landed, and round 1
//! commits writes prepared before a cleanup took every version they were based
//! on, see `HOLES`.

mod history;
mod plan;
mod rounds;
mod workers;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use async_trait::async_trait;

use crate::engine::data::{SMALL, Scale};
use crate::engine::ops::Model;
use crate::engine::rows::{full_batch, schema, source_of};
use crate::engine::run::Run;
use crate::engine::steps::StepCheck;
use crate::engine::{
    Body, ContentRules, Failure, Header, Journal, PlanInput, PlanRng, StoreChoice,
};
use crate::harness::{CALLS_LOG, STEPS_LOG, SeedReport, field};
use crate::incidents::{Area, Incident, Method};
use crate::knobs::{self, Class, Key, Knob, Knobs, STEPS};
use crate::scenario::{Runtime, Scenario, SelfTest, Settings, nothing_more};
use history::LineageTrack;
use plan::{
    Hold, PlannedOp, PlannedReader, ReaderKind, Round, RoundOp, RoundsPlan, RowsModel,
    focusable_ops, plan_rounds,
};

pub const SCENARIO: &dyn Scenario = &Concurrency;

/// An update lands as two commits, each with half of its rows.
const TORN_SNAPSHOT: SelfTest = SelfTest {
    name: "torn_snapshot",
    expected: &["torn_snapshot"],
    reason: None,
    verify: nothing_more,
};

/// A commit restores the version the round started from.
const LOST_WRITE: SelfTest = SelfTest {
    name: "lost_write",
    expected: &["lost_write"],
    reason: None,
    verify: nothing_more,
};

/// A reader that should see the latest version reads the first of the round.
const STALE_READ: SelfTest = SelfTest {
    name: "stale_read",
    expected: &["stale_read"],
    reason: None,
    verify: nothing_more,
};

/// A reader's pinned version turns into another between two reads.
const SNAPSHOT_CHANGED: SelfTest = SelfTest {
    name: "snapshot_changed",
    expected: &["snapshot_changed"],
    reason: None,
    verify: nothing_more,
};

/// A row is appended a second time.
const DUPLICATE_KEY: SelfTest = SelfTest {
    name: "duplicate_key",
    expected: &["duplicate_key"],
    reason: None,
    verify: nothing_more,
};

/// The history of a row's lineage is not what the rows say.
const LINEAGE: SelfTest = SelfTest {
    name: "lineage",
    expected: &["lineage"],
    reason: None,
    verify: nothing_more,
};

/// A hold point waits, with no limit, for what cannot happen, while a reader
/// keeps reading: its reads must not count as progress.
const GATE_HANG: SelfTest = SelfTest {
    name: "gate_hang",
    expected: &["hang"],
    reason: None,
    verify: ended_by_the_stall,
};

self_tests! {
    torn_snapshot: TORN_SNAPSHOT,
    lost_write: LOST_WRITE,
    stale_read: STALE_READ,
    snapshot_changed: SNAPSHOT_CHANGED,
    duplicate_key: DUPLICATE_KEY,
    lineage: LINEAGE,
    gate_hang: GATE_HANG,
}

/// Reads have a log of their own: the watchdog counts the growth of the step log
/// as progress, and a busy reader must not hide a hung writer.
const READS_LOG: &str = "reads.log";
/// How long a commit may keep retrying: a hold point can eat the default 30 s.
const COMMIT_RETRY_TIMEOUT_VARIABLE: &str = "LANCE_COMMIT_RETRY_TIMEOUT_SECS";
const COMMIT_RETRY_TIMEOUT_SECS: &str = "120";

/// The order of a plan is a wish: a wait that cannot end costs this much and
/// counts as `gate_timeout`, while a real hang stays the watchdog's. Ranks rule
/// out a circle of holds, but a shared `Session` can close one: its metadata cache
/// loads a file once, so a hold inside that load stalls whoever reads that file,
/// perhaps the very operation the hold waits for.
const GATE_LIMIT: Key<Duration> = Key::new("gate_limit");
/// How many workers a round runs.
const WORKERS: Key<RangeInclusive<u64>> = Key::new("workers");
/// The share of operations that hold at a point until another one is over.
const HOLD: Key<f64> = Key::new("hold");
/// The share of operations whose handle moves to the latest version first.
const CATCH_UP: Key<f64> = Key::new("catch_up");
/// How many times a worker tries an operation among the others. After that it
/// waits for them to finish and tries alone, when it must succeed.
const PARALLEL_ATTEMPTS: usize = 3;
const MIN_CLASS_ROWS: usize = 20;
const CONTENDED_EXISTING_KEYS: RangeInclusive<u64> = 8..=16;
const CONTENDED_NEW_KEYS: RangeInclusive<u64> = 4..=12;
const CONTENDED_KINDS: [&str; 3] = ["contended_upsert", "contended_update", "contended_delete"];
/// An operation's number is `round * NUMBERS_PER_ROUND + rank + 1`.
const NUMBERS_PER_ROUND: i64 = 100;

struct Concurrency;

impl Scenario for Concurrency {
    fn name(&self) -> &'static str {
        "concurrency"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["concurrency"]
    }

    fn focusable_ops(&self) -> Vec<&'static str> {
        focusable_ops()
    }

    /// The forced pairs come from the third round on.
    fn least_focus_steps(&self, _focused: &[&str]) -> u64 {
        3
    }

    fn knobs(&self) -> Vec<Knob> {
        let mut knobs = knobs::harness(16);
        knobs.extend([
            knobs::steps(),
            WORKERS.range(Class::Plan, 2..=4, 2..=8, "how many workers a round runs"),
            HOLD.share(
                Class::Plan,
                0.6,
                0.0..=1.0,
                "the share of operations that hold until another one is over",
            ),
            CATCH_UP.share(
                Class::Plan,
                0.7,
                0.0..=1.0,
                "the share of operations on the latest version",
            ),
            // A hold longer than this eats the 120 s a commit may retry for, and
            // its try gives up after it committed.
            // No shorter: in a round of a pair, the second operation waits at its
            // commit for the first to end, and one that gives up first misses the
            // pair.
            GATE_LIMIT.duration(
                Class::Plan,
                Duration::from_secs(10),
                Duration::from_secs(10)..=Duration::from_secs(30),
                "how long a hold waits before it gives up",
            ),
        ]);
        knobs
    }

    /// A hold's wait goes into the first try of a commit, which sets the unit of
    /// Lance's backoff: 10 s there can become a silent sleep of over a minute, and
    /// a round of two minutes. Workers past the default four make more such
    /// commits in a round.
    fn settings(&self, _self_test: Option<&SelfTest>, knobs: &Knobs) -> Settings {
        let steps = STEPS.get(knobs);
        let gate_limit = GATE_LIMIT.get(knobs);
        let crowd = WORKERS.get(knobs).end().div_ceil(4) as u32;
        Settings {
            steps,
            stall: Duration::from_secs(150).max(gate_limit * 15),
            deadline: Duration::from_secs(300)
                + Duration::from_secs(120).max(gate_limit * 12) * crowd * steps as u32,
            least_deadline: None,
            child_env: vec![(
                COMMIT_RETRY_TIMEOUT_VARIABLE,
                Some(COMMIT_RETRY_TIMEOUT_SECS.to_string()),
            )],
            cgroup: false,
        }
    }

    fn self_tests(&self) -> &'static [&'static SelfTest] {
        SELF_TESTS
    }

    fn checks(&self) -> &'static [&'static dyn StepCheck] {
        &[]
    }

    /// A round's writers and readers are tasks, which only race with threads to
    /// run on: as many as workers, and 4 at least.
    fn runtime(&self, knobs: &Knobs) -> Runtime {
        Runtime {
            worker_threads: 4.max(*WORKERS.get(knobs).end() as usize),
            blocking_keep_alive: None,
            leaves_tasks_on_failure: true,
        }
    }

    fn store(&self, rng: &mut PlanRng) -> StoreChoice {
        match rng.range(0..=2) {
            0 => StoreChoice::Local,
            1 => StoreChoice::FileObjectStore,
            _ => StoreChoice::SharedMemory,
        }
    }

    fn scale(&self) -> Scale {
        SMALL
    }

    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body> {
        let header = input.header;
        let mut rounds = plan_rounds(
            rng,
            input.seed,
            header.store,
            &header.scale,
            header.initial_rows,
            input.steps,
            input.with_incidents,
            &header.focus,
            input.knobs,
        );
        if let Some(test) = input.self_test {
            rounds.for_self_test(test, header.initial_rows);
        }
        Box::new(rounds)
    }
}

/// The stall, not the deadline, must have ended it, with the reader going on
/// long after the last line of the step log.
fn ended_by_the_stall(report: &SeedReport) {
    assert!(
        report.detail.contains("no progress for"),
        "the watchdog must have seen no progress, but: {}",
        report.detail
    );
    let last_tick = |name: &str| {
        fs::read_to_string(report.dir.join(name))
            .unwrap_or_default()
            .lines()
            .rev()
            .find_map(|line| field(line, "t").and_then(|tick| tick.parse::<u64>().ok()))
            .unwrap_or(0)
    };
    assert!(
        last_tick(READS_LOG) >= last_tick(STEPS_LOG) + 10_000,
        "a reader must have kept reading during the hang, see {}",
        report.dir.display()
    );
    let calls = fs::read_to_string(report.dir.join(CALLS_LOG)).unwrap_or_default();
    assert!(
        calls.contains("hold_before"),
        "the hang must come from the hold point, see {}",
        report.dir.display()
    );
}

#[async_trait(?Send)]
impl Body for RoundsPlan {
    fn render(&self) -> String {
        let mut text = format!(
            "workers={} stable_row_ids={} shared_session={} declared_key={}\n",
            self.workers, self.stable_row_ids, self.shared_session, self.declared_key,
        );
        for (index, round) in self.rounds.iter().enumerate() {
            writeln!(
                text,
                "round={} theme={} keys={:?}",
                index + 1,
                round.theme.unwrap_or("none"),
                round.keys
            )
            .unwrap();
            for op in &round.ops {
                let hold = op.hold.as_ref().map_or_else(
                    || "none".to_string(),
                    |hold| {
                        let arrival = if hold.arrival { ":arrival" } else { "" };
                        format!("{}:until={}{arrival}", hold.point, hold.until)
                    },
                );
                let start_after = op
                    .start_after
                    .map_or_else(|| "none".to_string(), |rank| rank.to_string());
                writeln!(
                    text,
                    "  rank={} worker={} number={} catch_up={} op={} hold={hold} start_after={start_after} staged={}",
                    op.rank, op.worker, op.number, op.catch_up, op.op, op.staged
                )
                .unwrap();
            }
            for reader in &round.readers {
                let hold = reader.hold.as_ref().map_or_else(
                    || "none".to_string(),
                    |hold| {
                        let arrival = if hold.arrival { ":arrival" } else { "" };
                        format!("{}:until={}{arrival}", hold.point, hold.until)
                    },
                );
                let start_after = reader
                    .start_after
                    .map_or_else(|| "none".to_string(), |rank| rank.to_string());
                writeln!(
                    text,
                    "  reader={} hold={hold} start_after={start_after}",
                    reader.kind.name()
                )
                .unwrap();
            }
        }
        text
    }

    fn steps(&self) -> usize {
        self.rounds.len()
    }

    /// The rounds check their history themselves.
    fn content(&self) -> ContentRules {
        ContentRules {
            versions_stay_listed: false,
            indices_cover_the_model: false,
            digests: false,
            hidden_files_fail: false,
        }
    }

    fn stable_row_ids(&self) -> bool {
        self.stable_row_ids
    }

    fn initial_source(&self, header: &Header) -> Box<dyn RecordBatchReader + Send> {
        let ids: Vec<i64> = (0..header.initial_rows).collect();
        Box::new(source_of(full_batch(
            &self.schema(),
            header.data_seed,
            &header.scale,
            &ids,
            0,
        )))
    }

    /// `gen` of every row, as the checks of the content read it.
    fn initial_model(&self, header: &Header) -> Model {
        Model {
            stamps: Some(RowsModel::new(header.initial_rows, self.workers).all_rows()),
            stamps_ever: Some(BTreeSet::new()),
            ..Model::new(header.initial_rows)
        }
    }

    fn journals(&self) -> &'static [&'static str] {
        &[READS_LOG]
    }

    async fn run(&self, run: &mut Run) -> Result<(), Failure> {
        let mut state = self.start(run.header.initial_rows, run.journal_of(READS_LOG));
        Box::pin(run.run_rounds(self, &mut state)).await
    }
}

impl RoundsPlan {
    /// One round of two whole-class updates, whose effect is known: the defect a
    /// self-test injects must stand out, and a random plan need not contain what
    /// that defect acts on.
    fn for_self_test(&mut self, test: &SelfTest, initial_rows: i64) {
        let update = |worker: usize, rank: usize| PlannedOp {
            rank,
            worker,
            number: NUMBERS_PER_ROUND + rank as i64 + 1,
            catch_up: true,
            op: RoundOp::Update {
                lo: 0,
                hi: initial_rows,
            },
            hold: None,
            start_after: None,
            staged: false,
        };
        let mut ops = vec![update(0, 0), update(1, 1)];
        // It waits for its own end, which a gate without a limit never sees.
        if *test == GATE_HANG {
            ops[1].hold = Some(Hold {
                point: Incident::HoldCall {
                    method: Method::Put,
                    area: Area::Manifests,
                    nth: 1,
                    after: false,
                },
                until: 1,
                arrival: false,
            });
        }
        let kind = match *test == SNAPSHOT_CHANGED {
            true => ReaderKind::Pinned,
            false => ReaderKind::FreshOpen,
        };
        self.stable_row_ids |= *test == LINEAGE;
        self.rounds = vec![Round {
            theme: Some("self_test"),
            keys: Vec::new(),
            ops,
            readers: vec![PlannedReader {
                kind,
                hold: None,
                start_after: None,
            }],
            hole: None,
        }];
    }

    fn start(&self, initial_rows: i64, reads: Arc<Journal>) -> RoundsState {
        RoundsState {
            rows: RowsModel::new(initial_rows, self.workers),
            digests: BTreeMap::new(),
            lineage: LineageTrack::default(),
            reads,
        }
    }

    fn schema(&self) -> SchemaRef {
        schema(self.declared_key)
    }
}

/// What `concurrency` knows between its rounds, besides the model of the run.
pub struct RoundsState {
    rows: RowsModel,
    /// What each version held when the checks first read it.
    digests: BTreeMap<u64, u64>,
    lineage: LineageTrack,
    reads: Arc<Journal>,
}
