// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The step engine: a plan of single operations, each meeting its incident or limit,
//! and the checks after each.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::{self};
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::FutureExt;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use object_store::GetOptions;
use roaring::RoaringBitmap;
use tokio::sync::Notify;

use crate::harness::{CALLS_LOG, STEPS_LOG, SeedReport, field};
use crate::incidents::{
    Area, CRASH_FIRED, CancelWhen, Death, Delay, HangPoint, Incident, IncidentState, Method,
    kill_this_process,
};
use crate::leaks::{Snapshot, TmpWatch, files_below, heap_peak_bytes, start_heap_peak};
use crate::limits::OwnCgroup;
use crate::scenario::{Runtime, Scenario, SelfTest, Settings, nothing_more};

use super::checks::{
    file_sizes, file_times, files_by_version, is_commit_conflict,
    is_merge_insert_pool_share_defect, is_resources_exhausted, listed_versions, paths,
};
use super::data::Scale;
use super::ops::{
    CommitShape, IndexKind, Limit, Model, Op, OpOutput, Outcome, POOL_VARIABLE, op_future,
};
use super::run::Run;
use super::{
    Body, ContentRules, Failure, Header, Plan, PlanInput, PlanRng, StoreChoice, fail, one_line,
    panic_message,
};

pub const HANG: SelfTest = SelfTest {
    name: "hang",
    expected: &["hang"],
    reason: None,
    verify: nothing_more,
};

pub const PANIC: SelfTest = SelfTest {
    name: "panic",
    expected: &["panic"],
    reason: None,
    verify: nothing_more,
};

pub const TASK_LEAK: SelfTest = SelfTest {
    name: "task_leak",
    expected: &["task_leak"],
    reason: None,
    verify: nothing_more,
};

pub const TMP_LEAK: SelfTest = SelfTest {
    name: "tmp_leak",
    expected: &["tmp_leak"],
    reason: None,
    verify: nothing_more,
};

pub const MODEL_MISMATCH: SelfTest = SelfTest {
    name: "model_mismatch",
    expected: &["content"],
    reason: None,
    verify: nothing_more,
};

pub const COLUMN_MISMATCH: SelfTest = SelfTest {
    name: "column_mismatch",
    expected: &["content"],
    reason: None,
    verify: nothing_more,
};

/// A SIGKILL nobody planned: the seed must end there.
pub const FOREIGN_KILL: SelfTest = SelfTest {
    name: "foreign_kill",
    expected: &["crash"],
    reason: None,
    verify: |report| {
        assert!(
            !report.journal.contains("generation=2 "),
            "a death nobody planned must end the seed, see {}",
            report.dir.display()
        );
    },
};

/// A file a manifest references goes away while the process is dead.
pub const KILL_THEN_MISSING_FILE: SelfTest = SelfTest {
    name: "kill_then_missing_file",
    // Which check meets the missing file first depends on the file.
    expected: &["validate", "missing_files", "unreadable"],
    reason: None,
    verify: |report| {
        assert!(
            report.journal.contains("generation=2 "),
            "the death was planned, so a later process must look at the dataset, see {}",
            report.dir.display()
        );
    },
};

/// A step goes over its limit: the seed goes on and ends well.
pub const OOM_RECOVERY: SelfTest = SelfTest {
    name: "oom_recovery",
    expected: &[],
    reason: None,
    verify: |report| {
        assert!(
            report.journal.contains("generation=2 ")
                && report.journal.contains("observation=killed:oom_kill")
                && report.journal.contains("outcome=killed"),
            "the step must have died of its limit and been concluded later, see {}",
            report.dir.display()
        );
    },
};

/// A step hangs under a limit.
pub const HANG_UNDER_LIMIT: SelfTest = SelfTest {
    name: "hang_under_limit",
    expected: &["hang"],
    reason: None,
    verify: left_the_cgroup_as_it_was,
};

/// A task the operation left behind holds a call that never answers.
pub const HUNG_CALL: SelfTest = SelfTest {
    name: "hung_call",
    expected: &["hung_call"],
    reason: Some("hung, 1 still wait and 0 ended after"),
    verify: nothing_more,
};

/// A slow call in a task the operation left behind ends after the operation.
pub const LATE_CALL: SelfTest = SelfTest {
    name: "late_call",
    expected: &["hung_call"],
    reason: Some("hung, 0 still wait and 1 ended after"),
    verify: nothing_more,
};

/// A task the operation left behind writes once its hung read is let go.
pub const WRITE_AFTER_RELEASE: SelfTest = SelfTest {
    name: "write_after_release",
    expected: &["hung_call"],
    reason: Some("writes reached the store after the hung reads"),
    verify: nothing_more,
};

fn left_the_cgroup_as_it_was(report: &SeedReport) {
    let cgroup = fs::read_to_string(report.dir.join("cgroup.txt")).unwrap_or_default();
    // Taken before the kill, so the child is still there to see.
    assert!(
        cgroup
            .split("== cgroup.procs\n")
            .nth(1)
            .is_some_and(|procs| !procs.trim().is_empty()),
        "a hang under a limit must leave the cgroup as it was in {}",
        report.dir.display()
    );
}

/// What a scenario of the step engine decides for its plans.
pub trait StepPlanner: Scenario {
    fn rules(&self) -> &'static StepRules;

    /// What the steps that no focus takes run.
    fn op_mix(&self) -> &'static OpMix;

    /// What a focus on `create_index` builds, and on `optimize_indices` before
    /// there is an index. A random build takes its kind from the roll, see `OpMix`.
    fn index_kinds(&self) -> &'static [IndexKind];

    /// The operation of a step that no focus takes, if it decides one before
    /// the draw of the mix.
    fn opening_op(&self, _rng: &mut PlanRng, _index: usize) -> Option<Op> {
        None
    }

    /// Every step catches the handle up first.
    fn always_resync(&self) -> bool {
        false
    }

    /// How long a request in flight runs before a planned death.
    fn in_flight_us(&self, _rng: &mut PlanRng) -> u64 {
        0
    }

    /// The limit of a step, given the model with its operation. Drawn like the
    /// incident.
    fn limit(&self, _rng: &mut PlanRng, _op: &Op, _model: &Model) -> Option<Limit> {
        None
    }

    /// The incident of a step, drawn at every step, even where a focus, a
    /// self-test or a calibration drops it: dropping one moves no other draw.
    fn incident(
        &self,
        rng: &mut PlanRng,
        store: StoreChoice,
        scale: &Scale,
        op: &Op,
    ) -> Option<Incident>;
}

/// What a random step runs, by a roll of 0 to 99. The ranges take it in the
/// order of the fields; a roll none takes builds an index from 55 on (a BTree, an
/// IVF_FLAT, then an inverted index), and appends below.
pub struct OpMix {
    /// Unless a delete would leave fewer rows than the checks need.
    pub deletes: Range<u64>,
    pub cleanups: Range<u64>,
    /// Once there is an index.
    pub optimizations: Range<u64>,
    /// Taken from the appends.
    pub others: &'static [(Range<u64>, Other)],
}

pub enum Other {
    Compact,
    /// The BTree index on the key, which a merge insert uses.
    IdIndex,
    Scan,
    MergeInsert,
}

/// How a scenario's steps go: what its trouble does, and which checks its steps
/// take.
pub struct StepRules {
    /// The process dies where another scenario cancels, and the verdicts count
    /// its incidents as `kill`.
    pub dies: bool,
    /// What a step observes when its incident fired and it still returned Ok.
    pub absorbed: &'static str,
    /// An incident that fired explains an error. A death injects none, and a
    /// slow or hung call none either.
    pub incidents_explain_errors: bool,
    /// An error on a stale handle must be a lost conflict. A slow store injects
    /// no error at all, so there it may lose nothing else.
    pub stale_errors_are_conflicts: bool,
    /// The operations share one session, see `let_go_of_the_session`.
    pub keeps_session: bool,
    /// A new index or segment is followed by a check of every version a cleanup
    /// keeps.
    pub checks_kept_versions: bool,
}

/// How the operation of a step ended, for the checks after it.
struct StepEnd {
    outcome: Outcome,
    /// The latest version before the operation.
    before: u64,
    /// How many times its incident or limit bit.
    fired: u64,
    /// The version a fresh open landed on right after a cancellation.
    right_after: Option<u64>,
}

/// A check that a scenario adds to the steps of the step engine, with what it
/// needs from the parent and from the child.
pub trait StepCheck: Sync {
    /// The variables a child gets, or must not inherit when `None`.
    fn child_env(&self, self_test: Option<&SelfTest>) -> Vec<(&'static str, Option<String>)>;

    /// Changes the plan of a self-test of its own, once the step planner has
    /// drawn it.
    fn prepare(&self, steps: &mut [Step], self_test: Option<&SelfTest>);

    /// The logs it keeps in the directory of a seed, which a death can tear.
    fn logs(&self) -> &'static [&'static str];

    /// Starts it in a child, before the dataset is written.
    fn start(
        &self,
        dir: &Path,
        started: Instant,
        generation: u64,
        state: Arc<IncidentState>,
    ) -> Result<(), String>;

    /// Checks a step right after its verdict, whichever process concludes it.
    fn after_step(
        &self,
        run: &Run,
        n: usize,
        step: &Step,
        outcome: &Outcome,
        committed: bool,
    ) -> Result<(), Failure>;
}

/// How the seeds of a scenario of the step engine run, for plans of `steps`.
pub fn settings(steps: u64) -> Settings {
    Settings {
        seeds: 16,
        steps,
        stall_secs: 60,
        deadline_secs: 120 + 60 * steps,
        least_deadline: None,
        child_env: Vec::new(),
        replay_env: Vec::new(),
        cgroup: false,
    }
}

pub const RUNTIME: Runtime = Runtime {
    worker_threads: 2,
    blocking_keep_alive: None,
    leaves_tasks_on_failure: false,
};

/// A focus takes every other step. An index optimization has work from the
/// fourth step on, after an index build and an append.
pub fn least_focus_steps(focused: &[&str]) -> u64 {
    if focused == ["optimize_indices"] {
        4
    } else {
        2
    }
}

/// A plan of single operations, and the rules its steps go by.
pub struct StepsBody {
    pub steps: Vec<Step>,
    pub rules: &'static StepRules,
}

#[async_trait(?Send)]
impl Body for StepsBody {
    fn render(&self) -> String {
        render_steps(&self.steps)
    }

    fn steps(&self) -> usize {
        self.steps.len()
    }

    fn content(&self) -> ContentRules {
        ContentRules {
            versions_stay_listed: true,
            indices_cover_the_model: true,
            digests: true,
            hidden_files_fail: false,
        }
    }

    async fn run(&self, run: &mut Run) -> Result<(), Failure> {
        Box::pin(run.resume(&self.steps, self.rules)).await?;
        Box::pin(run.run_steps(&self.steps, self.rules)).await
    }
}

/// The steps of a scenario of the step engine. Every operation ends up committed
/// once, by its first try or by the retry, so the plan tracks the model on its
/// own. A focus makes every other step one of its operations and leaves the
/// trouble to those steps alone, while the steps between build the history they
/// need.
pub fn plan(planner: &dyn StepPlanner, rng: &mut PlanRng, input: &PlanInput<'_>) -> StepsBody {
    let PlanInput {
        steps,
        header,
        with_incidents,
        self_test,
        ..
    } = *input;
    let (store, scale, focus) = (header.store, &header.scale, &header.focus);
    let mut model = Model::new(header.initial_rows);
    let mut planned = Vec::with_capacity(steps);
    for index in 0..steps {
        let focused = (!focus.is_empty() && index % 2 == 1)
            .then(|| focused_op(rng, planner, scale, &model, focus))
            .flatten();
        let troubled = focus.is_empty() || focused.is_some();
        let op = match focused {
            Some(op) => op,
            None => match planner.opening_op(rng, index) {
                Some(op) => op,
                // An optimization of the focus needs work: an index, and fragments
                // it does not cover. Other steps go on as they would, short of
                // taking that work away from the next step of the focus.
                None if focus.contains("optimize_indices")
                    && index % 2 == 0
                    && !model.unindexed =>
                {
                    match model.indices.is_empty() {
                        true => Op::CreateIndex(*pick_from(rng, planner.index_kinds())),
                        false => Op::Append {
                            first_id: model.next_id,
                            rows: added_rows(rng, scale),
                        },
                    }
                }
                None => {
                    let op = random_op(rng, planner.op_mix(), scale, &model);
                    let takes_the_work = focus.contains("optimize_indices")
                        && index % 2 == 0
                        && model.unindexed
                        && {
                            let mut after = model.clone();
                            after.apply(&op);
                            !after.unindexed
                        };
                    match takes_the_work {
                        true => Op::Append {
                            first_id: model.next_id,
                            rows: added_rows(rng, scale),
                        },
                        false => op,
                    }
                }
            },
        };
        model.apply(&op);
        let incident = planner.incident(rng, store, scale, &op);
        // A stale handle is covered by the scenarios of its own; a read
        // or a merge on one would only report what it was left with.
        let resync = rng.chance(0.7) || matches!(op, Op::Delete { .. }) || planner.always_resync();
        let in_flight_us = planner.in_flight_us(rng);
        let limit = planner.limit(rng, &op, &model);
        planned.push(Step {
            op,
            incident: incident.filter(|_| with_incidents && troubled),
            resync,
            in_flight_us,
            limit: limit.filter(|_| with_incidents && troubled),
        });
    }
    for check in planner.checks() {
        check.prepare(&mut planned, self_test);
    }
    StepsBody {
        steps: planned,
        rules: planner.rules(),
    }
}

pub fn render_steps(steps: &[Step]) -> String {
    let mut text = String::new();
    for (index, step) in steps.iter().enumerate() {
        let incident = step
            .incident
            .as_ref()
            .map_or_else(|| "none".to_string(), Incident::to_string);
        let limit = step
            .limit
            .as_ref()
            .map_or_else(|| "none".to_string(), Limit::to_string);
        writeln!(
            text,
            "step={} op={} incident={incident} limit={limit} resync={} in_flight_us={}",
            index + 1,
            step.op,
            step.resync,
            step.in_flight_us
        )
        .unwrap();
    }
    text
}

/// The pool an operation of this scenario must make do with: it may spill, but
/// from this size up, running out of memory is a regression. A sort needs room
/// for a batch, and with less a clean error is all it can do.
fn pool_floor_kib(op: &Op) -> u64 {
    match op {
        // The merge of a spilled source held 949 KiB under a pool of 1 and of 2 MiB,
        // too much for the sort of the target to fit its batch beside it.
        Op::MergeInsert { .. } => 4096,
        // A sorted batch of `category` takes 192 KiB, and a third of the pool is
        // reserved for the merge of the spilled runs.
        _ => 512,
    }
}

#[derive(Clone, Debug)]
pub struct Step {
    pub op: Op,
    pub incident: Option<Incident>,
    /// Whether the handle catches up with the latest version first. Without
    /// it, an operation after a commit the handle never saw has to resolve a
    /// conflict. A delete always catches up: Lance evaluates its predicate on
    /// the handle's snapshot and rebases it over an append without evaluating
    /// it again, but evaluates it again after a conflict with another delete,
    /// so on a stale handle the rows it removes depend on what it raced with.
    pub resync: bool,
    /// How long a request in flight runs before a planned death.
    pub in_flight_us: u64,
    pub limit: Option<Limit>,
}

fn random_op(rng: &mut PlanRng, mix: &OpMix, scale: &Scale, model: &Model) -> Op {
    let roll = rng.range(0..=99);
    if mix.deletes.contains(&roll) && can_delete(scale, model) {
        return delete_op(rng, scale, model);
    }
    if mix.cleanups.contains(&roll) {
        return cleanup_op(rng);
    }
    if mix.optimizations.contains(&roll) && !model.indices.is_empty() {
        return Op::OptimizeIndices;
    }
    match roll {
        55..70 => Op::CreateIndex(IndexKind::BTree),
        70..85 => Op::CreateIndex(IndexKind::IvfFlat),
        85..100 => Op::CreateIndex(IndexKind::Inverted),
        _ => match mix.others.iter().find(|(rolls, _)| rolls.contains(&roll)) {
            Some((_, Other::Compact)) => Op::Compact,
            Some((_, Other::IdIndex)) => Op::CreateIndex(IndexKind::BTreeId),
            Some((_, Other::Scan)) => Op::Scan,
            Some((_, Other::MergeInsert)) => merge_insert_op(rng, scale, model),
            None => Op::Append {
                first_id: model.next_id,
                rows: added_rows(rng, scale),
            },
        },
    }
}

/// One of the operations of a focus that this step can run, if any.
fn focused_op(
    rng: &mut PlanRng,
    planner: &dyn StepPlanner,
    scale: &Scale,
    model: &Model,
    focus: &BTreeSet<&'static str>,
) -> Option<Op> {
    let candidates: Vec<&str> = planner
        .focusable_ops()
        .into_iter()
        .filter(|op| {
            focus.contains(op)
                && match *op {
                    "delete" => can_delete(scale, model),
                    // An optimization with nothing to do meets no trouble.
                    "optimize_indices" => model.unindexed,
                    _ => true,
                }
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    Some(match *pick_from(rng, &candidates) {
        "append" => Op::Append {
            first_id: model.next_id,
            rows: added_rows(rng, scale),
        },
        "delete" => delete_op(rng, scale, model),
        "create_index" => Op::CreateIndex(*pick_from(rng, planner.index_kinds())),
        "optimize_indices" => Op::OptimizeIndices,
        "merge_insert" => merge_insert_op(rng, scale, model),
        "compact" => Op::Compact,
        "scan" => Op::Scan,
        "cleanup" => cleanup_op(rng),
        other => unreachable!("{} plans no {other}", planner.name()),
    })
}

pub fn pick_from<'a, T>(rng: &mut PlanRng, values: &'a [T]) -> &'a T {
    &values[rng.range(0..=values.len() as u64 - 1) as usize]
}

pub fn added_rows(rng: &mut PlanRng, scale: &Scale) -> i64 {
    rng.range(scale.added_rows.0..=scale.added_rows.1) as i64
}

/// Whether a delete leaves the rows the checks need.
fn can_delete(scale: &Scale, model: &Model) -> bool {
    model.live.len() > scale.min_live_rows + 1
}

/// A run of up to a tenth of the live rows.
fn delete_op(rng: &mut PlanRng, scale: &Scale, model: &Model) -> Op {
    let live: Vec<i64> = model.live.iter().copied().collect();
    let max_len = (live.len() / 10)
        .min(live.len() - scale.min_live_rows)
        .max(1);
    let len = rng.range(1..=max_len as u64) as usize;
    let start = rng.range(0..=(live.len() - len) as u64) as usize;
    Op::Delete {
        lo: live[start],
        hi: live[start + len - 1] + 1,
    }
}

fn cleanup_op(rng: &mut PlanRng) -> Op {
    Op::Cleanup {
        retain: rng.range(1..=3) as usize,
        delete_unverified: rng.chance(0.5),
    }
}

/// Half of its ids exist, the other half are new.
fn merge_insert_op(rng: &mut PlanRng, scale: &Scale, model: &Model) -> Op {
    let rows = added_rows(rng, scale);
    Op::MergeInsert {
        first_id: model.next_id - rows / 2,
        rows,
    }
}

fn cancel_call(method: Method, area: Area, nth: u64, when: CancelWhen) -> Incident {
    Incident::CancelCall {
        method,
        area,
        nth,
        when,
    }
}

/// How long a step waits for the hung calls of a task its operation aborted.
const HUNG_SETTLE: Duration = Duration::from_secs(2);

/// The calls each operation of `storage_faults` and `slow_storage` makes:
/// (method, area, the highest call to target).
pub fn storage_calls(op: &Op, scale: &Scale) -> Vec<(Method, Area, u64)> {
    match *op {
        Op::Append { rows, .. } => vec![
            // One data file per fragment.
            (
                Method::Put,
                Area::Data,
                (rows as u64).div_ceil(scale.rows_per_file as u64),
            ),
            (Method::Put, Area::Transactions, 1),
            (Method::Put, Area::Manifests, 1),
            (Method::List, Area::Manifests, 1),
        ],
        Op::Delete { .. } => vec![
            (Method::Get, Area::Data, 3),
            (Method::Put, Area::Deletions, 1),
            (Method::Put, Area::Transactions, 1),
            (Method::Put, Area::Manifests, 1),
            (Method::List, Area::Manifests, 1),
        ],
        Op::CreateIndex(_) => vec![
            (Method::Get, Area::Data, 3),
            (Method::Put, Area::Indices, 2),
            (Method::Put, Area::Transactions, 1),
            (Method::Put, Area::Manifests, 1),
            (Method::List, Area::Manifests, 1),
        ],
        // It reads the segments it merges with.
        Op::OptimizeIndices => vec![
            (Method::Get, Area::Data, 3),
            (Method::Get, Area::Indices, 2),
            (Method::Put, Area::Indices, 2),
            (Method::Put, Area::Transactions, 1),
            (Method::Put, Area::Manifests, 1),
            (Method::List, Area::Manifests, 1),
        ],
        // It lists the manifests for its policy, for itself and with the
        // rest of the dataset, reads what it keeps, and goes on past a
        // failed deletion.
        Op::Cleanup { .. } => vec![
            (Method::List, Area::Manifests, 3),
            (Method::Get, Area::Manifests, 2),
            (Method::List, Area::Data, 1),
            (Method::List, Area::Indices, 1),
            (Method::Delete, Area::Data, 1),
            (Method::Delete, Area::Deletions, 1),
            (Method::Delete, Area::Indices, 1),
            (Method::Delete, Area::Transactions, 2),
            (Method::Delete, Area::Manifests, 2),
        ],
        Op::MergeInsert { .. } | Op::Compact | Op::Scan => {
            unreachable!("{op} is not part of this scenario")
        }
    }
}

/// Where a cancellation or a death meets an operation: a call of its commit, a
/// stage of an index build, a fragment it writes, or a write of its files.
pub fn cancel_point(
    rng: &mut PlanRng,
    store: StoreChoice,
    scale: &Scale,
    op: &Op,
) -> Option<Incident> {
    if !rng.chance(0.85) {
        return None;
    }
    // The manifest write passes the wrapper on every store. Only a local
    // store keeps a request in flight, in its blocking pool.
    let flights: &[CancelWhen] = match store {
        StoreChoice::Local | StoreChoice::FileObjectStore => {
            &[CancelWhen::Before, CancelWhen::During, CancelWhen::After]
        }
        StoreChoice::SharedMemory => &[CancelWhen::Before, CancelWhen::After],
    };
    if let Op::Cleanup { .. } = op {
        return Some(cleanup_point(rng, store, flights));
    }
    let mut points: Vec<Incident> = flights
        .iter()
        .map(|when| cancel_call(Method::Put, Area::Manifests, 1, *when))
        .collect();
    match *op {
        Op::Append { rows, .. } => {
            let fragments = (rows as u64).div_ceil(scale.rows_per_file as u64);
            for nth in 1..=fragments {
                for at_begin in [true, false] {
                    points.push(Incident::CancelFragment { nth, at_begin });
                }
            }
        }
        Op::CreateIndex(index) => {
            for &stage in index.stages() {
                for at_start in [true, false] {
                    points.push(Incident::CancelStage { stage, at_start });
                }
            }
        }
        // Its second manifest: between the reservation of fragment ids
        // and the rewrite.
        Op::Compact => {
            for when in flights {
                points.push(cancel_call(Method::Put, Area::Manifests, 2, *when));
            }
        }
        Op::Delete { .. } | Op::MergeInsert { .. } | Op::Scan | Op::OptimizeIndices => {}
        Op::Cleanup { .. } => unreachable!("a cleanup has points of its own"),
    }
    // A local store writes data, index, transaction and deletion files
    // around the wrapper, so only the other stores see those.
    if store != StoreChoice::Local {
        let written = match op {
            Op::Append { .. } | Op::MergeInsert { .. } | Op::Compact | Op::Scan => Area::Data,
            Op::Delete { .. } => Area::Deletions,
            Op::CreateIndex(_) | Op::OptimizeIndices => Area::Indices,
            Op::Cleanup { .. } => unreachable!("a cleanup writes nothing"),
        };
        for area in [written, Area::Transactions] {
            for when in flights {
                points.push(cancel_call(Method::Put, area, 1, *when));
            }
        }
        if !matches!(op, Op::Append { .. }) {
            points.push(cancel_call(Method::Get, Area::Data, 1, CancelWhen::Before));
        }
    }
    Some(points.swap_remove(rng.range(0..=points.len() as u64 - 1) as usize))
}

/// Where a cancellation or a death meets a cleanup: at a listing of manifests,
/// the one for its policy or its own, at its first read of one, or at a
/// deletion, which can leave a version it expires without some of its files.
/// All of these pass the wrapper on every store. A shared-memory store deletes
/// within the call, and a cleanup runs its deletions side by side, so a
/// cancellation there would land only once the others ran: on that store a
/// cancellation meets a cleanup before its deletions.
fn cleanup_point(rng: &mut PlanRng, store: StoreChoice, flights: &[CancelWhen]) -> Incident {
    let mut points = vec![
        cancel_call(Method::List, Area::Manifests, 1, CancelWhen::Before),
        cancel_call(Method::List, Area::Manifests, 2, CancelWhen::Before),
        cancel_call(Method::Get, Area::Manifests, 1, CancelWhen::Before),
    ];
    if store != StoreChoice::SharedMemory {
        for (area, deletions) in CLEANUP_DELETIONS {
            for nth in 1..=deletions {
                for when in flights {
                    points.push(cancel_call(Method::Delete, area, nth, *when));
                }
            }
        }
    }
    points.swap_remove(rng.range(0..=points.len() as u64 - 1) as usize)
}

/// What a cleanup deletes, and how far into its deletions of each kind an
/// incident may go. Every version has a manifest and a transaction file, while
/// a data, deletion or index file goes only once nothing kept refers to it, or
/// when an operation that failed left it behind.
const CLEANUP_DELETIONS: [(Area, u64); 5] = [
    (Area::Data, 1),
    (Area::Deletions, 1),
    (Area::Indices, 1),
    (Area::Transactions, 2),
    (Area::Manifests, 2),
];

/// What a cleanup must remove and what it must leave, from the files the
/// dataset holds before it starts.
pub struct CleanupFiles {
    /// Files that only versions below its boundary reference.
    expired: BTreeSet<String>,
    /// Files that no listed version references. A write in progress may own
    /// one, so only a cleanup allowed to delete unverified files takes it.
    unreferenced: BTreeSet<String>,
}

impl CleanupFiles {
    async fn before(verifier: &Dataset, n: usize, boundary: u64) -> Result<Self, Failure> {
        let tracked = files_by_version(verifier.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let all = file_times(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        Ok(Self {
            expired: tracked
                .iter()
                .filter(|(_, versions)| versions.iter().all(|version| *version < boundary))
                .map(|(path, _)| path.clone())
                .collect(),
            unreferenced: all
                .into_keys()
                .filter(|path| {
                    !tracked.contains_key(path) && !path.ends_with("latest_version_hint.json")
                })
                .collect(),
        })
    }
}

/// The area of the deletions that an error the wrapper injected failed. Lance
/// counts each failed deletion and goes on.
fn failed_deletions(incident: Option<&Incident>, fired: u64) -> Option<Area> {
    match incident? {
        Incident::Fail {
            method: Method::Delete,
            area,
            ..
        }
        | Incident::CancelCall {
            method: Method::Delete,
            area,
            when: CancelWhen::Before,
            ..
        } if fired > 0 => Some(*area),
        _ => None,
    }
}

/// Whether an error the wrapper injected failed a read of index files: Lance
/// skips an index it fails to open and still returns Ok.
fn failed_an_index_read(incident: Option<&Incident>, fired: u64) -> bool {
    fired > 0
        && matches!(
            incident,
            Some(
                Incident::Fail {
                    method: Method::Get,
                    area: Area::Indices,
                    ..
                } | Incident::CancelCall {
                    method: Method::Get,
                    area: Area::Indices,
                    when: CancelWhen::Before,
                    ..
                }
            )
        )
}

/// A limit in force on a step.
enum ArmedLimit {
    /// Its value after the kernel's rounding, and the count of hits when set.
    Cgroup {
        hits: u64,
        bytes: u64,
    },
    Pool(TmpWatch),
}

/// The last progress hook a step reached, from the call log.
fn last_progress_mark(calls_log: &Path, n: usize) -> String {
    let phase = format!("phase={n}");
    let calls = fs::read_to_string(calls_log).unwrap_or_default();
    calls
        .lines()
        .rev()
        .filter(|line| line.split(' ').any(|token| token == phase))
        .find_map(|line| {
            let mut tokens = line.split(' ').skip(2);
            let (event, name) = (tokens.next()?, tokens.next()?);
            match event {
                "stage_start" | "stage_complete" => Some(format!("{event}:{name}")),
                "fragment_begin" | "fragment_complete" => Some(event.to_string()),
                _ => None,
            }
        })
        .unwrap_or_else(|| "before_any_mark".to_string())
}

/// Where the journal of the earlier processes of a seed says it stopped.
#[derive(Debug, PartialEq, Eq)]
struct ResumePoint {
    /// Steps that ran to their end; they stand at the head of the plan.
    finished: usize,
    /// The step whose operation started and did not end, and the latest
    /// version before it. Nothing commits before the operation starts, so a
    /// step that got no further than its start counts as not started.
    interrupted: Option<(usize, u64)>,
    /// What the operation of that step reported, if it got that far.
    ended: Option<OpEnd>,
}

/// The `op_end` line of a step.
#[derive(Debug, PartialEq, Eq)]
struct OpEnd {
    outcome: String,
    fired: u64,
    status_unknown: bool,
    fragments_removed: usize,
    failed_deletes: u64,
    message: String,
}

impl OpEnd {
    fn from_line(line: &str) -> Self {
        let number = |name: &str| field(line, name).and_then(|value| value.parse().ok());
        Self {
            outcome: field(line, "outcome").unwrap_or_default().to_string(),
            fired: number("fired").unwrap_or(0),
            status_unknown: field(line, "status_unknown") == Some("true"),
            fragments_removed: number("fragments_removed").unwrap_or(0) as usize,
            failed_deletes: number("failed_deletes").unwrap_or(0),
            message: line
                .split_once(" detail=")
                .map_or_else(String::new, |(_, detail)| detail.to_string()),
        }
    }

    fn outcome(&self) -> Outcome {
        match self.outcome.as_str() {
            "ok" => Outcome::Ok(OpOutput {
                fragments_removed: self.fragments_removed,
                scanned: None,
                failed_deletes: self.failed_deletes,
            }),
            "err" => Outcome::Failed {
                message: self.message.clone(),
                status_unknown: self.status_unknown,
            },
            _ => Outcome::Killed,
        }
    }
}

impl ResumePoint {
    fn from_journal(journal: &str) -> Self {
        let mut point = Self {
            finished: 0,
            interrupted: None,
            ended: None,
        };
        for line in journal.lines() {
            let Some(n) = field(line, "step").and_then(|n| n.parse().ok()) else {
                continue;
            };
            // A detail is free text.
            let head = line.split(" detail=").next().unwrap_or(line);
            let word = |word: &str| head.split(' ').any(|token| token == word);
            if word("done") {
                point.finished = n;
                point.interrupted = None;
                point.ended = None;
            } else if word("armed")
                && let Some(before) = field(line, "before").and_then(|v| v.parse().ok())
            {
                point.interrupted = Some((n, before));
                point.ended = None;
            } else if word("op_end") && line.contains(" detail=") {
                // A line the death tore lacks its last field, and then the
                // step ended with the death.
                point.ended = Some(OpEnd::from_line(line));
            }
        }
        point
    }
}

/// A plan of eight steps, as a seed of `planner` draws it.
pub fn plan_of(
    planner: &dyn StepPlanner,
    seed: u64,
    focus: &BTreeSet<&'static str>,
) -> (Header, StepsBody) {
    let (mut rng, header) = Plan::draw_header(planner, seed, focus);
    let input = PlanInput {
        seed,
        steps: 8,
        header: &header,
        with_incidents: true,
        self_test: None,
    };
    let body = plan(planner, &mut rng, &input);
    (header, body)
}

/// The plans of a scenario draw every operation a focus can take, and optimize
/// indices only once there is one: no step drops an index.
pub fn check_draws(planner: &dyn StepPlanner) {
    let everything = BTreeSet::new();
    let mut drawn = BTreeSet::new();
    for seed in 1..=48 {
        let mut indexed = false;
        for step in plan_of(planner, seed, &everything).1.steps {
            assert!(
                indexed || !matches!(step.op, Op::OptimizeIndices),
                "{} seed {seed}: an optimization without an index",
                planner.name()
            );
            indexed |= matches!(step.op, Op::CreateIndex(_));
            drawn.insert(step.op.focus_name());
        }
    }
    let focusable: BTreeSet<&str> = planner.focusable_ops().into_iter().collect();
    assert_eq!(drawn, focusable, "{}", planner.name());
}

/// Every other step of a focused plan runs an operation of the focus, and only
/// those steps meet trouble.
pub fn check_focus(planner: &dyn StepPlanner, seeds: u64) {
    for op in planner.focusable_ops() {
        let focus = BTreeSet::from([op]);
        let mut focused = 0;
        let mut troubled = 0;
        for seed in 1..=seeds {
            let (header, body) = plan_of(planner, seed, &focus);
            assert!(
                header
                    .render(planner, seed, 8)
                    .lines()
                    .next()
                    .is_some_and(|header| header.ends_with(&format!(" focus={op}"))),
                "the focus must be part of the plan and its fingerprint"
            );
            for (index, step) in body.steps.iter().enumerate() {
                let on_focus = index % 2 == 1 && step.op.focus_name() == op;
                if step.incident.is_some() || step.limit.is_some() {
                    assert!(
                        on_focus,
                        "{} seed {seed} step {}: trouble on {} with a focus on {op}",
                        planner.name(),
                        index + 1,
                        step.op
                    );
                    troubled += 1;
                }
                focused += usize::from(on_focus);
            }
        }
        // Four steps of eight are the focus's, unless a delete has no rows to spare,
        // or an optimization has nothing to do yet: at step 2, an index has just
        // been built.
        assert!(
            focused as u64 >= seeds * 3,
            "{}: {focused} steps of {op} in {seeds} seeds",
            planner.name()
        );
        assert!(troubled > 0, "{}: no trouble on {op}", planner.name());
    }
}

/// The incident a step's verdict counts under.
fn incident_label(step: &Step, rules: &StepRules) -> String {
    if let Some(limit) = &step.limit {
        return format!("memory:{}:{limit}", step.op.kind());
    }
    let kind = step
        .incident
        .as_ref()
        .map_or_else(|| "none".to_string(), Incident::kind);
    match rules.dies {
        true => kind.replacen("cancel", "kill", 1),
        false => kind,
    }
}

impl Run {
    /// Picks up the seed after an earlier process died, and first of all looks
    /// at what it left: the dataset must read as if the death was a clean end
    /// of the step it interrupted.
    pub async fn resume(&mut self, steps: &[Step], rules: &StepRules) -> Result<(), Failure> {
        if self.generation == 1 {
            return Ok(());
        }
        let log = fs::read_to_string(self.seed_dir().join(STEPS_LOG))
            .map_err(|error| fail("harness", 0, format_args!("read the step log: {error}")))?;
        let point = ResumePoint::from_journal(&log);
        for step in &steps[..point.finished] {
            self.model.apply(&step.op);
        }
        for line in log.lines() {
            if let Some(boundary) = field(line, "cleanup_boundary").and_then(|b| b.parse().ok()) {
                self.expired_below = self.expired_below.max(boundary);
            }
            // The last line of a dead process can be torn.
            if let (Some(version), Some(ids)) = (
                field(line, "digest_of").and_then(|version| version.parse().ok()),
                field(line, "ids").filter(|ids| {
                    ids.split_once(':')
                        .is_some_and(|(_, hash)| hash.len() == 16)
                }),
            ) {
                self.digests.insert(version, ids.to_string());
            }
            if field(line, "observation") == Some("expired_version_broken")
                && let Some(versions) = field(line, "versions")
            {
                self.broken_reported.extend(
                    versions
                        .split(',')
                        .filter_map(|version| version.parse::<u64>().ok()),
                );
            }
        }
        self.next_step = point.finished;
        let Some((n, before)) = point.interrupted else {
            self.journal.line(format_args!(
                "resume after={} interrupted=none",
                point.finished
            ));
            self.observe_what_the_death_left(point.finished).await?;
            return Box::pin(self.measure_baseline(point.finished)).await;
        };
        let step = &steps[n - 1];
        self.state.set_phase(format!("{n}.resume"));
        self.journal.line(format_args!(
            "resume after={} interrupted={n} before={before}",
            point.finished
        ));
        self.observe(
            n,
            &format!(
                "killed_at:{}:{}",
                step.op.kind(),
                last_progress_mark(&self.seed_dir().join(CALLS_LOG), n)
            ),
        );
        self.observe_what_the_death_left(n).await?;
        if self.self_test == Some(&KILL_THEN_MISSING_FILE) {
            self.remove_a_data_file(n).await?;
        }
        // The read that starts the pools needs the model of this very moment.
        let finished_model = self.model.clone();
        let latest = self.latest_version(n).await?;
        if latest > before {
            self.model.apply(&step.op);
        }
        Box::pin(self.measure_baseline(n)).await?;
        self.model = finished_model;
        self.state.set_phase(n.to_string());
        // A death can come after the operation ended, and what the operation
        // reported still has to hold.
        let (outcome, fired) = match &point.ended {
            Some(ended) => (ended.outcome(), ended.fired),
            None => (Outcome::Killed, 1),
        };
        let end = StepEnd {
            outcome,
            before,
            fired,
            right_after: None,
        };
        Box::pin(self.conclude(n, step, end, rules)).await?;
        // Steps count from 1, their places in the plan from 0.
        self.next_step = n;
        Ok(())
    }

    /// Temporary files stay behind a death, since nothing runs to remove them;
    /// that is recorded and cleared, so that the leak checks start anew. Files
    /// of the dataset stay until a cleanup, which the end of the seed checks.
    async fn observe_what_the_death_left(&self, n: usize) -> Result<(), Failure> {
        let snapshot = Snapshot::take(&self.tmp);
        if !snapshot.tmp_files.is_empty() {
            let bytes: u64 = snapshot
                .tmp_files
                .iter()
                .filter_map(|path| fs::metadata(path).ok())
                .map(|meta| meta.len())
                .sum();
            self.journal.line(format_args!(
                "step={n} observation=tmp_after_kill files={} bytes={bytes}",
                snapshot.tmp_files.len()
            ));
        }
        for entry in fs::read_dir(&self.tmp)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
        {
            let path = entry.path();
            let removed = if path.is_dir() {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_file(&path)
            };
            removed.map_err(|error| {
                fail(
                    "harness",
                    n,
                    format_args!("clear {}: {error}", path.display()),
                )
            })?;
        }
        let verifier = self.open_verifier(n).await?;
        let all = file_sizes(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let tracked = paths(verifier.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let (mut orphans, mut bytes) = (0, 0);
        for (path, size) in &all {
            if !tracked.contains(path) && !path.ends_with("latest_version_hint.json") {
                orphans += 1;
                bytes += size;
            }
        }
        if orphans > 0 {
            self.journal.line(format_args!(
                "step={n} observation=orphans_after_kill files={orphans} bytes={bytes}"
            ));
        }
        self.observe_hidden_files(n, "hidden_after_kill", &all);
        Ok(())
    }

    async fn remove_a_data_file(&self, n: usize) -> Result<(), Failure> {
        let tracked = files_by_version(self.open_verifier(n).await?.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        // One that a kept version needs: an expired one may lack files already.
        let (file, _) = tracked
            .iter()
            .find(|(path, versions)| {
                path.starts_with("data/")
                    && versions
                        .last()
                        .is_some_and(|newest| *newest >= self.expired_below)
            })
            .expect("a dataset has data files");
        let path = self.seed_dir().join("dataset").join(file);
        fs::remove_file(&path).expect("remove a data file");
        self.journal
            .line(format_args!("self_test removed {}", path.display()));
        Ok(())
    }

    pub async fn run_steps(&mut self, steps: &[Step], rules: &StepRules) -> Result<(), Failure> {
        for (index, step) in steps.iter().enumerate().skip(self.next_step) {
            Box::pin(self.run_step(index + 1, step, rules)).await?;
        }
        Ok(())
    }

    pub async fn run_step(
        &mut self,
        n: usize,
        step: &Step,
        rules: &StepRules,
    ) -> Result<(), Failure> {
        self.state.set_phase(n.to_string());
        self.journal.line(format_args!(
            "step={n} start op={} incident={} resync={}",
            step.op,
            step.incident
                .as_ref()
                .map_or_else(|| "none".to_string(), Incident::to_string),
            step.resync
        ));
        if step.resync {
            self.catch_up(n).await?;
        }
        let before = self.latest_version(n).await?;
        if let Op::Cleanup { retain, .. } = step.op {
            self.note_cleanup_boundary(n, retain).await?;
        }
        self.read_fragments =
            (self.handle_stale && matches!(step.op, Op::OptimizeIndices)).then(|| {
                self.ds
                    .fragments()
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect()
            });
        let armed = self.arm_limit(n, step, before)?;
        let death = rules.dies.then(|| {
            let in_flight = matches!(
                step.incident,
                Some(Incident::CancelCall {
                    when: CancelWhen::During,
                    ..
                })
            );
            Death::new(
                self.journal.clone(),
                in_flight.then(|| Duration::from_micros(step.in_flight_us)),
            )
        });
        // Before `arm`: a call the step slows must never see the end of the step before.
        self.state.begin_operation();
        let cancel = self.state.arm(step.incident.clone(), death);
        if n == 2 && self.generation == 1 {
            self.die_if_a_self_test_says_so().await?;
        }
        let heap_start = start_heap_peak();
        let outcome = self.execute(step.op, step.limit, Some(cancel)).await;
        self.state.end_operation();
        let heap_growth_kib = (heap_peak_bytes() - heap_start) >> 10;
        let mut fired = self.state.disarm();
        // Lifted before the line that says the operation ended: a death before
        // that line counts as a death during the operation, and no check runs
        // under the limit.
        fired += Box::pin(self.lift_limit(n, step, armed, &outcome)).await?;
        let (status_unknown, output) = match &outcome {
            Outcome::Failed { status_unknown, .. } => (*status_unknown, OpOutput::default()),
            Outcome::Ok(output) => (false, *output),
            _ => (false, OpOutput::default()),
        };
        self.journal.line(format_args!(
            "step={n} op_end outcome={} fired={fired} status_unknown={status_unknown} fragments_removed={} failed_deletes={} heap_growth_kib={heap_growth_kib} detail={}",
            outcome.name(),
            output.fragments_removed,
            output.failed_deletes,
            outcome.detail()
        ));
        if n == 2
            && let Some(test) = self.self_test
        {
            self.inject(test).await;
        }
        if let Outcome::Panicked(message) = &outcome {
            return Err(fail("panic", n, message));
        }
        // A single read: a manifest still landing could make several disagree.
        let right_after = if matches!(outcome, Outcome::Cancelled) {
            Some(self.opened_version(n).await?)
        } else {
            None
        };
        let end = StepEnd {
            outcome,
            before,
            fired,
            right_after,
        };
        Box::pin(self.conclude(n, step, end, rules)).await
    }

    /// Everything after the operation of step `n`, whichever process runs it:
    /// what the outcome says about the dataset, the model, the checks, the
    /// retry and the leaks.
    async fn conclude(
        &mut self,
        n: usize,
        step: &Step,
        end: StepEnd,
        rules: &StepRules,
    ) -> Result<(), Failure> {
        let StepEnd {
            outcome,
            before,
            fired,
            right_after,
        } = end;
        let incident = incident_label(step, rules);
        let shape = step.op.shape();
        let handle = self.ds.version().version;
        let ran_stale = self.handle_stale;
        // Dropping the operation dropped every slow or hung call it awaited
        // itself: one still in flight, or one that ended well after the operation
        // was over, sits in a task the operation left behind, and a write there could
        // land after the caller gave up. Only a scan read may: Lance's I/O
        // scheduler runs each one as a task of its own, which holds its connection
        // until the store's own timeout.
        let left = self.state.settle_left_behind(HUNG_SETTLE).await;
        // Lance retries a read once it is let go, and a task that waited for it
        // goes on: from then on nothing of the operation may write. A hung read
        // kept the wait above going for the whole of `HUNG_SETTLE`, so the last
        // poll of a task the operation aborted is over by now.
        if left.scan_reads > 0 {
            self.state.count_writes();
        }
        self.state.release_hung_calls();
        if left.calls + left.late > 0 {
            return Err(fail(
                "hung_call",
                n,
                format_args!(
                    "of the calls the step slowed or hung, {} still wait and {} ended after {} was over, in tasks it left behind",
                    left.calls, left.late, step.op
                ),
            ));
        }
        if left.scan_reads > 0 {
            self.observe(n, "hung_read_left_behind");
        }
        self.let_go_of_the_session(n, Some(&step.op), rules.keeps_session)
            .await?;
        self.quiet_down(n).await;
        let writes = self.state.writes_counted();
        if writes > 0 {
            return Err(fail(
                "hung_call",
                n,
                format_args!(
                    "{writes} writes reached the store after the hung reads of {} were let go, from a task it left behind",
                    step.op
                ),
            ));
        }
        if let (Op::Cleanup { .. }, Outcome::Cancelled) = (step.op, &outcome) {
            self.settle_files().await;
        }
        let after = self.latest_version(n).await?;
        if after < before || after > before + shape.max_delta() {
            return Err(fail(
                "version_jump",
                n,
                format_args!(
                    "{} moved the version from {before} to {after}, by at most {} it should",
                    step.op,
                    shape.max_delta()
                ),
            ));
        }
        let committed = after > before;
        self.journal.line(format_args!(
            "step={n} verdict incident={incident} fired={fired} outcome={} committed={committed} before={before} after={after}",
            outcome.name()
        ));
        for check in self.scenario.checks() {
            check.after_step(self, n, step, &outcome, committed)?;
        }
        let lost = match (&outcome, shape) {
            (Outcome::Ok(_), CommitShape::Exactly1) => !committed,
            // The reservation and the rewrite, or nothing at all.
            (Outcome::Ok(output), CommitShape::UpTo2) => {
                (output.fragments_removed > 0) != (after == before + 2)
            }
            _ => false,
        };
        if lost {
            return Err(fail(
                "lost_commit",
                n,
                format_args!(
                    "{} returned Ok, but the latest version moved from {before} to {after}",
                    step.op
                ),
            ));
        }
        // A limit that only changes how a read goes has nothing to absorb.
        let says_nothing = matches!(step.limit, Some(Limit::IoBuffer { .. }));
        if outcome.is_ok() && fired > 0 && !says_nothing {
            self.observe(n, rules.absorbed);
        }
        match &outcome {
            // A caller that sees an error retries, and a retried append
            // duplicates its rows. Only an honest "status unknown" may land.
            Outcome::Failed {
                message,
                status_unknown: false,
            } if committed && after == before + shape.max_delta() => {
                return Err(fail(
                    "error_after_commit",
                    n,
                    format_args!(
                        "{} returned an error, but its commit landed: {message}",
                        step.op
                    ),
                ));
            }
            Outcome::Failed {
                status_unknown: true,
                ..
            } => {
                self.observe(
                    n,
                    if committed {
                        "status_unknown_landed"
                    } else {
                        "status_unknown_lost"
                    },
                );
            }
            Outcome::Cancelled if committed => {
                self.observe(n, "cancel_after_commit");
                if right_after == Some(before) {
                    self.observe(n, "late_commit");
                }
            }
            Outcome::Killed if committed => self.observe(n, "killed_after_commit"),
            _ => {}
        }
        if let Outcome::Failed { message, .. } = &outcome {
            let explained = match step.limit {
                // The counter of hits also grows when reclaim succeeds, so the
                // error itself has to say that memory ran out.
                Some(Limit::Cgroup { .. }) => {
                    fired > 0
                        && (message.contains("(os error 12)")
                            || message.contains("Cannot allocate memory"))
                }
                // Enough for the work to spill, so an error here is a regression.
                Some(Limit::Pool { kib }) => {
                    let defect = is_merge_insert_pool_share_defect(&step.op, message);
                    if defect {
                        self.observe(n, "known_defect:merge_insert_pool_share");
                    }
                    defect || (kib < pool_floor_kib(&step.op) && is_resources_exhausted(message))
                }
                // Neither promises an error: both only change how the work is done.
                Some(Limit::FtsMemory { .. } | Limit::IoBuffer { .. }) => false,
                None => fired > 0 && rules.incidents_explain_errors,
            };
            if !explained {
                // A stale handle may legitimately lose a conflict.
                let excused =
                    ran_stale && (!rules.stale_errors_are_conflicts || is_commit_conflict(message));
                if !excused {
                    return Err(fail(
                        "error_without_fault",
                        n,
                        format_args!("{} failed with nothing to explain it: {message}", step.op),
                    ));
                }
                self.observe(n, "error_on_stale_handle");
            }
        }
        if (step.incident.is_some() || step.limit.is_some()) && fired == 0 {
            self.observe(n, "not_reached");
        }
        if let Outcome::Ok(output) = &outcome {
            // One that commits nothing leaves its handle where it was, stale or not.
            if committed {
                self.check_handle(n, &step.op, handle, after)?;
                self.handle_stale = false;
            }
            self.check_scan(n, output)?;
        } else if committed && !matches!(outcome, Outcome::Killed) {
            self.handle_stale = true;
        }
        if committed {
            self.model.apply(&step.op);
        }
        Box::pin(self.check_content(n)).await?;
        if let Op::Cleanup { .. } = step.op {
            let boundary = self.boundary_of_step(n)?;
            Box::pin(self.check_cleanup(n, step, false, &outcome, fired, boundary)).await?;
        }
        if let (Op::OptimizeIndices, Outcome::Ok(_)) = (step.op, &outcome) {
            let read_failed = failed_an_index_read(step.incident.as_ref(), fired);
            let read_fragments = self.read_fragments.take();
            Box::pin(self.check_optimized(n, read_failed, read_fragments.as_ref())).await?;
        }
        let retries = match shape {
            CommitShape::Exactly1 => !committed,
            CommitShape::UpTo1 | CommitShape::UpTo2 | CommitShape::Never => !outcome.is_ok(),
        };
        if retries {
            self.state.set_phase(format!("{n}.retry"));
            self.catch_up(n).await?;
            let boundary = match step.op {
                Op::Cleanup { retain, .. } => self.note_cleanup_boundary(n, retain).await?,
                _ => 0,
            };
            let retry = self.execute(step.op, None, None).await;
            let handle = self.ds.version().version;
            let retried = self.latest_version(n).await?;
            let right_move = match (shape, &retry) {
                (CommitShape::Exactly1, _) => retried == after + 1,
                // Whether the first try committed or not, there may be nothing left.
                (CommitShape::UpTo1, _) => (after..=after + 1).contains(&retried),
                (CommitShape::UpTo2, Outcome::Ok(output)) if output.fragments_removed > 0 => {
                    retried == after + 2
                }
                (CommitShape::UpTo2 | CommitShape::Never, _) => retried == after,
            };
            if !retry.is_ok() || !right_move {
                return Err(fail(
                    "retry_failed",
                    n,
                    format_args!(
                        "the retry of {} ended {} {}, and the version went from {after} to {retried}",
                        step.op,
                        retry.name(),
                        retry.detail()
                    ),
                ));
            }
            self.check_handle(n, &step.op, handle, retried)?;
            self.journal
                .line(format_args!("step={n} retry outcome=ok version={retried}"));
            self.model.apply(&step.op);
            if let Outcome::Ok(output) = &retry {
                self.check_scan(n, output)?;
            }
            Box::pin(self.check_content(n)).await?;
            if let Op::Cleanup { .. } = step.op {
                Box::pin(self.check_cleanup(n, step, true, &retry, 0, boundary)).await?;
            }
            if let Op::OptimizeIndices = step.op {
                Box::pin(self.check_optimized(n, false, None)).await?;
            }
        }
        // A new index or segment must leave the files of the ones the versions
        // before it hold, which a cleanup may not look at before they expire.
        if matches!(step.op, Op::OptimizeIndices | Op::CreateIndex(_))
            && rules.checks_kept_versions
            && self.latest_version(n).await? > before
        {
            Box::pin(self.check_kept_versions(n)).await?;
        }
        self.let_go_of_the_session(n, Some(&step.op), rules.keeps_session)
            .await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)?;
        // From here on a later process does not touch this step again.
        self.journal.line(format_args!("step={n} done"));
        Ok(())
    }

    /// A scan must have returned every live row once.
    fn check_scan(&self, n: usize, output: &OpOutput) -> Result<(), Failure> {
        let Some((count, sum)) = output.scanned else {
            return Ok(());
        };
        let expected = self
            .model
            .live
            .iter()
            .fold(0i64, |sum, id| sum.wrapping_add(*id));
        if count != self.model.live.len() || sum != expected {
            return Err(fail(
                "content",
                n,
                format_args!(
                    "a scan returned {count} rows with ids summing to {sum}, where the model has {} rows summing to {expected}",
                    self.model.live.len()
                ),
            ));
        }
        Ok(())
    }

    /// Writes the line a later process resumes from, then, on a step with a
    /// limit, what the budget comes down to.
    fn arm_limit(&self, n: usize, step: &Step, before: u64) -> Result<Option<ArmedLimit>, Failure> {
        let harness = |error: String| fail("harness", n, error);
        if let Some(Limit::Pool { kib }) = step.limit {
            self.journal
                .line(format_args!("step={n} armed before={before}"));
            // SAFETY: the step starts from quiet, with no task left and the
            // pools idle, so no other thread reads the environment now.
            unsafe { std::env::set_var(POOL_VARIABLE, (kib << 10).to_string()) };
            return Ok(Some(ArmedLimit::Pool(TmpWatch::start(&self.tmp))));
        }
        let (Some(Limit::Cgroup { budget_kib }), Some(cgroup)) = (step.limit, &self.cgroup) else {
            // Only a run in one process, for a debugger, has no cgroup.
            let skipped = matches!(step.limit, Some(Limit::Cgroup { .. }));
            self.journal.line(format_args!(
                "step={n} armed before={before} limit_skipped={skipped}"
            ));
            return Ok(None);
        };
        cgroup.shrink(&self.seed_dir().join("dataset"));
        let usage = cgroup.usage();
        // SAFETY: `sysconf` only reads a system setting.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let limit = (usage.current + budget_kib * 1024) & !(page - 1);
        let hits = cgroup.limit_hits();
        self.journal.line(format_args!(
            "step={n} armed before={before} limit={limit} current={} anon={} file={}",
            usage.current, usage.anon, usage.file
        ));
        let bytes = cgroup.set_limit(limit).map_err(harness)?;
        if bytes != limit {
            return Err(harness(format!(
                "memory.max reads {bytes} after writing {limit}"
            )));
        }
        Ok(Some(ArmedLimit::Cgroup { hits, bytes }))
    }

    /// Lifts the limit of a step, and returns how many times the step hit it.
    async fn lift_limit(
        &self,
        n: usize,
        step: &Step,
        armed: Option<ArmedLimit>,
        outcome: &Outcome,
    ) -> Result<u64, Failure> {
        match armed {
            Some(ArmedLimit::Cgroup { hits, bytes }) => {
                let cgroup = self
                    .cgroup
                    .as_ref()
                    .ok_or_else(|| fail("harness", n, "a limit without a cgroup"))?;
                cgroup
                    .lift_limit()
                    .map_err(|error| fail("harness", n, error))?;
                let hits = cgroup.limit_hits() - hits;
                self.journal
                    .line(format_args!("step={n} limit_end bytes={bytes} hits={hits}"));
                return Ok(hits);
            }
            Some(ArmedLimit::Pool(watch)) => {
                let spilled = watch.stop();
                // SAFETY: the operation is over, and what it left behind reads
                // no variable of the pool, which only a new operation looks up.
                unsafe { std::env::remove_var(POOL_VARIABLE) };
                self.journal
                    .line(format_args!("step={n} limit_end tmp_peak_bytes={spilled}"));
                if spilled > 0 {
                    self.observe(n, &format!("spill:{}", step.op.kind()));
                }
                let exhausted = matches!(
                    outcome,
                    Outcome::Failed { message, .. } if is_resources_exhausted(message)
                );
                return Ok(u64::from(spilled > 0 || exhausted));
            }
            None => {}
        }
        match (step.limit, outcome) {
            (Some(Limit::IoBuffer { .. }), _) => Ok(1),
            (Some(Limit::FtsMemory { .. }), Outcome::Ok(_)) => {
                let partitions = self.text_index_partitions(n).await?;
                self.journal
                    .line(format_args!("step={n} limit_end partitions={partitions}"));
                Ok(u64::from(partitions > 1))
            }
            _ => Ok(0),
        }
    }

    /// The partitions of the inverted index: a builder that goes over its memory
    /// limit writes one and starts the next.
    async fn text_index_partitions(&self, n: usize) -> Result<usize, Failure> {
        let indices = self
            .ds
            .load_indices()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let parts: BTreeSet<&str> = indices
            .iter()
            .filter(|index| index.name == IndexKind::Inverted.name())
            .filter_map(|index| index.files.as_ref())
            .flatten()
            .filter_map(|file| file.path.strip_suffix("_tokens.lance"))
            .collect();
        Ok(parts.len())
    }

    async fn die_if_a_self_test_says_so(&self) -> Result<(), Failure> {
        let limited = |room: u64| -> Result<&OwnCgroup, Failure> {
            let cgroup = self
                .cgroup
                .as_ref()
                .ok_or_else(|| fail("harness", 2, "this self-test needs a cgroup"))?;
            cgroup
                .set_limit(cgroup.usage().current + room)
                .map_err(|error| fail("harness", 2, error))?;
            Ok(cgroup)
        };
        let Some(test) = self.self_test else {
            return Ok(());
        };
        if *test == FOREIGN_KILL {
            kill_this_process()
        } else if *test == KILL_THEN_MISSING_FILE {
            self.journal.line(CRASH_FIRED);
            kill_this_process()
        } else if *test == OOM_RECOVERY {
            limited(16 << 20)?;
            // Writing every page of it is what goes over the limit.
            std::hint::black_box(vec![1u8; 1 << 30]);
            Err(fail("harness", 2, "1 GiB fit into a budget of 16 MiB"))
        } else if *test == HANG_UNDER_LIMIT {
            limited(16 << 20)?;
            std::future::pending::<()>().await;
            Ok(())
        } else {
            Ok(())
        }
    }

    /// Runs `op` on the handle that meets the incidents. When `cancel` fires,
    /// the operation is dropped at whatever await it has reached.
    async fn execute(
        &mut self,
        op: Op,
        limit: Option<Limit>,
        cancel: Option<Arc<Notify>>,
    ) -> Outcome {
        let cancel = cancel.unwrap_or_default();
        let operation = AssertUnwindSafe(op_future(
            &mut self.ds,
            &self.hooks,
            self.header.data_seed,
            self.header.scale,
            op,
            limit,
        ))
        .catch_unwind();
        let result = tokio::select! {
            biased;
            () = cancel.notified() => return Outcome::Cancelled,
            result = operation => result,
        };
        match result {
            Ok(Ok(output)) => Outcome::Ok(output),
            Ok(Err(error)) => Outcome::Failed {
                status_unknown: error.is_commit_status_unknown(),
                message: one_line(&error.to_string()),
            },
            Err(panic) => Outcome::Panicked(panic_message(panic.as_ref())),
        }
    }

    /// Injects the defect of a self-test that acts after the operation of step 2.
    /// Others act before it, after the final cleanup, or in a later process.
    async fn inject(&mut self, test: &SelfTest) {
        self.journal.line(format_args!("self_test={}", test.name));
        if *test == HANG {
            std::future::pending::<()>().await;
        } else if *test == PANIC {
            tokio::spawn(async { panic!("stress self-test: a panic in a detached task") });
        } else if *test == TASK_LEAK {
            tokio::spawn(std::future::pending::<()>());
        } else if *test == TMP_LEAK {
            fs::write(self.tmp.join("self-test-leak"), b"leak").expect("leak a file");
        } else if *test == HUNG_CALL {
            // A write in a task that nothing drops meets a call that never answers.
            // What the step itself left behind goes first, so that the wait
            // below can only end on this write.
            self.state.release_hung_calls();
            let _ = self.state.settle_left_behind(HUNG_SETTLE).await;
            let path = self.ds.data_dir().join("self-test-hung.lance");
            let store = self
                .ds
                .object_store(None)
                .await
                .expect("the store of the dataset");
            self.state.arm(
                Some(Incident::Slow {
                    delays: Vec::new(),
                    hang: Some(HangPoint {
                        method: Method::Put,
                        area: Area::Data,
                        nth: 1,
                        after: false,
                        grace_ms: 0,
                    }),
                }),
                None,
            );
            tokio::spawn(async move {
                let _ = store.put(&path, b"stress").await;
            });
            while self.state.calls_in_flight() == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            self.state.disarm();
        } else if *test == LATE_CALL {
            // A slow read in a task that nothing drops ends after the operation.
            // Arming again leaves the end of the step's operation in place.
            self.state.release_hung_calls();
            let _ = self.state.settle_left_behind(HUNG_SETTLE).await;
            let manifest = self.ds.manifest_location().path.clone();
            let store = self
                .ds
                .object_store(None)
                .await
                .expect("the store of the dataset");
            self.state.arm(
                Some(Incident::Slow {
                    delays: vec![Delay {
                        method: Method::Get,
                        area: Area::Manifests,
                        nth: 1,
                        millis: 300,
                        after: false,
                    }],
                    hang: None,
                }),
                None,
            );
            let slow_read = tokio::spawn(async move {
                let _ = store.inner.get_opts(&manifest, GetOptions::default()).await;
            });
            // On a busy host the read may be over before this looks.
            while self.state.calls_in_flight() == 0 && !slow_read.is_finished() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            self.state.disarm();
        } else if *test == WRITE_AFTER_RELEASE {
            // A task that nothing drops waits for a hung read of a data file, and
            // writes once the read is let go.
            self.state.release_hung_calls();
            let _ = self.state.settle_left_behind(HUNG_SETTLE).await;
            let read = self.ds.data_dir().join("self-test-read.lance");
            let written = self.ds.data_dir().join("self-test-written.lance");
            let store = self
                .ds
                .object_store(None)
                .await
                .expect("the store of the dataset");
            self.state.arm(
                Some(Incident::Slow {
                    delays: Vec::new(),
                    hang: Some(HangPoint {
                        method: Method::Get,
                        area: Area::Data,
                        nth: 1,
                        after: false,
                        grace_ms: 0,
                    }),
                }),
                None,
            );
            tokio::spawn(async move {
                // The read ends with an error once it is let go.
                let _ = store.inner.get_opts(&read, GetOptions::default()).await;
                let _ = store.put(&written, b"stress").await;
            });
            while self.state.scan_reads_in_flight() == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            self.state.disarm();
        } else if *test == MODEL_MISMATCH {
            self.model.live.pop_first();
        } else if *test == COLUMN_MISMATCH {
            self.oracle_seed ^= 1;
        }
    }

    /// A cleanup dropped half way leaves the deletions it started running in the
    /// blocking pool of a local store, which tokio does not count. The checks
    /// wait until the files of the dataset stop changing: they read every
    /// manifest they list, and one could go in between.
    async fn settle_files(&self) {
        let dataset = self.seed_dir().join("dataset");
        let files = || {
            let mut files = files_below(&dataset);
            files.sort();
            files
        };
        let mut last = files();
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let now = files();
            if now == last {
                return;
            }
            last = now;
        }
    }

    /// The boundary of the cleanup of step `n`, from the journal, which a death
    /// leaves behind too.
    fn boundary_of_step(&self, n: usize) -> Result<u64, Failure> {
        let journal = fs::read_to_string(self.seed_dir().join(STEPS_LOG)).unwrap_or_default();
        journal
            .lines()
            .filter(|line| field(line, "step") == Some(n.to_string().as_str()))
            .filter_map(|line| field(line, "cleanup_boundary")?.parse().ok())
            .next_back()
            .ok_or_else(|| fail("harness", n, "a cleanup without its boundary"))
    }

    /// The first version a cleanup keeping `retain` versions leaves, as
    /// `retain_n_versions` documents it, though never above the version its
    /// handle reads, from which on Lance keeps every version. Written down before
    /// the cleanup starts, so that a later process knows it too; the files it
    /// must remove and leave only this process knows.
    async fn note_cleanup_boundary(&mut self, n: usize, retain: usize) -> Result<u64, Failure> {
        let verifier = self.open_verifier(n).await?;
        let listed: Vec<u64> = listed_versions(&verifier, n).await?.into_iter().collect();
        let boundary = listed[listed.len().saturating_sub(retain)].min(self.ds.version().version);
        self.journal
            .line(format_args!("step={n} cleanup_boundary={boundary}"));
        self.expired_below = self.expired_below.max(boundary);
        self.cleanup_files = Some(CleanupFiles::before(&verifier, n, boundary).await?);
        Ok(boundary)
    }

    /// What a cleanup leaves, however it ended: the versions it keeps open and
    /// read as they did, with all their files, and unless it may delete
    /// unverified files, the files no version referenced stay. One that returned
    /// Ok leaves no version below its boundary but those whose manifest it
    /// failed to delete, and with no deletion failed and no incident fired, none
    /// of the files only those versions referenced. Only an injected error or a
    /// cgroup limit fails a deletion.
    async fn check_cleanup(
        &mut self,
        n: usize,
        step: &Step,
        // The retry runs without the incident and the limit of the step.
        retry: bool,
        outcome: &Outcome,
        fired: u64,
        boundary: u64,
    ) -> Result<(), Failure> {
        let Op::Cleanup {
            delete_unverified, ..
        } = step.op
        else {
            unreachable!("a cleanup check after {}", step.op);
        };
        let incident = step.incident.as_ref().filter(|_| !retry);
        let mut swept = false;
        if let Outcome::Ok(output) = outcome {
            let may_stay = match failed_deletions(incident, fired) {
                Some(area) => {
                    if output.failed_deletes != fired {
                        return Err(fail(
                            "cleanup",
                            n,
                            format_args!(
                                "the cleanup counted {} failed deletions where {fired} were made to fail",
                                output.failed_deletes
                            ),
                        ));
                    }
                    // A manifest it failed to delete keeps its version listed.
                    if area == Area::Manifests { fired } else { 0 }
                }
                // The kernel may refuse an unlink for lack of memory.
                None if !retry && step.limit.is_some() && fired > 0 => {
                    if output.failed_deletes > 0 {
                        self.observe(n, "failed_deletes_under_limit");
                    }
                    output.failed_deletes
                }
                None if output.failed_deletes > 0 => {
                    return Err(fail(
                        "cleanup",
                        n,
                        format_args!(
                            "the cleanup failed {} deletions with nothing to explain them",
                            output.failed_deletes
                        ),
                    ));
                }
                None => 0,
            };
            self.check_expired_gone(n, boundary, may_stay).await?;
            // An expired manifest it failed to read leaves its files unverified.
            // A slow call leaves no manifest unread, only a failed one does.
            swept = output.failed_deletes == 0
                && (incident.is_none()
                    || fired == 0
                    || matches!(incident, Some(Incident::Slow { .. })));
        }
        // A process that died during the cleanup took what it knew with it.
        if let Some(files) = self.cleanup_files.take() {
            let left = file_times(self.open_verifier(n).await?.all_files().await)
                .await
                .map_err(|error| fail("unreadable", n, error))?;
            let taken: Vec<&String> = files
                .unreferenced
                .iter()
                .filter(|path| !left.contains_key(*path))
                .collect();
            if !delete_unverified && !taken.is_empty() {
                return Err(fail(
                    "cleanup",
                    n,
                    format_args!(
                        "the cleanup deleted files no version referenced, which a write in progress may own: {taken:?}"
                    ),
                ));
            }
            let kept: Vec<&String> = files
                .expired
                .iter()
                .filter(|path| left.contains_key(*path))
                .collect();
            if swept && !kept.is_empty() {
                return Err(fail(
                    "cleanup",
                    n,
                    format_args!(
                        "the cleanup returned Ok, but kept files that only versions below its boundary {boundary} referenced: {kept:?}"
                    ),
                ));
            }
        }
        Box::pin(self.check_kept_versions(n)).await
    }

    /// An index optimization that returned Ok leaves no fragment out of an index,
    /// and on a stale handle none of those its version held that are still
    /// there. Lance skips an index it fails to open and still returns Ok, so
    /// after an injected error reading index files, one left out is only
    /// reported.
    async fn check_optimized(
        &self,
        n: usize,
        read_failed: bool,
        read_fragments: Option<&RoaringBitmap>,
    ) -> Result<(), Failure> {
        let ds = self.open_verifier(n).await?;
        let indices = ds
            .load_indices()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let live: RoaringBitmap = ds
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect();
        let due = match read_fragments {
            Some(read) => &live & read,
            None => live,
        };
        let mut left_out = Vec::new();
        for kind in &self.model.indices {
            let covered = indices
                .iter()
                .filter(|index| index.name == kind.name())
                .filter_map(|index| index.fragment_bitmap.as_ref())
                .fold(RoaringBitmap::new(), |covered, bitmap| covered | bitmap);
            let missed = &due - &covered;
            if !missed.is_empty() {
                let missed: Vec<u32> = missed.iter().collect();
                left_out.push(format!("{} misses fragments {missed:?}", kind.name()));
            }
        }
        if left_out.is_empty() {
            return Ok(());
        }
        if read_failed {
            self.journal.line(format_args!(
                "step={n} observation=optimize_left_unindexed detail={}",
                left_out.join("; ")
            ));
            return Ok(());
        }
        Err(fail(
            "lost_commit",
            n,
            format_args!("optimize_indices returned Ok, but {}", left_out.join("; ")),
        ))
    }
}
