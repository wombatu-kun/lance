// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The workers and readers of a round, on handles of their own.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use lance::Dataset;
use lance::Error;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::index::DatasetIndexRemapperOptions;
use lance::dataset::optimize::{
    CompactionOptions, RewriteResult, commit_compaction, plan_compaction,
};
use lance::dataset::scanner::RowAddrTreeMap;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::{CommitBuilder, UncommittedMergeInsert, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::session::Session;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::object_store::ObjectStoreParams;
use tokio::sync::{Barrier, watch};
use tokio::task::JoinHandle;

use super::history::{Contended, Oracle, RoundModel};
use super::plan::{PlannedOp, PlannedReader, ReaderKind, Round, RoundOp};
use super::{DUPLICATE_KEY, PARALLEL_ATTEMPTS, SNAPSHOT_CHANGED, STALE_READ, TORN_SNAPSHOT};
use crate::engine::data::Scale;
use crate::engine::ops::{IndexKind, Op, OpOutput, op_future};
use crate::engine::rows::{
    Attempted, ClassRows, RowCounts, class_range, delete, full_batch, id_run, merge_report,
    source_of, stamp_job, stamp_source, update, upsert, upsert_job,
};
use crate::engine::{Failure, Journal, fail, one_line};
use crate::incidents::{Gate, IncidentState, StressHooks, StressWrapper};
use crate::scenario::SelfTest;

/// When things happened in a round.
pub struct Clocks {
    tick: Arc<AtomicU64>,
    started: Vec<AtomicUsize>,
    completed: Vec<AtomicUsize>,
    contended_started: Mutex<BTreeSet<i64>>,
    contended_completed: Mutex<BTreeSet<i64>>,
    increments_started: AtomicI64,
    increments_completed: AtomicI64,
    /// The newest version any operation has returned with.
    newest_returned: AtomicU64,
    pub spans: Vec<Span>,
}

/// An operation on the logical clock: zero means not yet.
#[derive(Default)]
pub struct Span {
    pub begin: AtomicU64,
    pub commit: AtomicU64,
    /// The version of that commit.
    pub version: AtomicU64,
    pub end: AtomicU64,
    /// The worker stopped trying among the others. What it did alone later met
    /// nobody, and the others' commits in between found it idle.
    pub gave_up: AtomicU64,
}

/// A view of how far the operations have come.
pub struct ClockView {
    pub ops: Vec<usize>,
    pub contended: BTreeSet<i64>,
    pub increments: i64,
    pub version: u64,
}

impl Clocks {
    pub fn new(workers: usize, ops: usize) -> Self {
        Self {
            tick: Arc::new(AtomicU64::new(0)),
            started: (0..workers).map(|_| AtomicUsize::new(0)).collect(),
            completed: (0..workers).map(|_| AtomicUsize::new(0)).collect(),
            contended_started: Mutex::new(BTreeSet::new()),
            contended_completed: Mutex::new(BTreeSet::new()),
            increments_started: AtomicI64::new(0),
            increments_completed: AtomicI64::new(0),
            newest_returned: AtomicU64::new(0),
            spans: (0..ops).map(|_| Span::default()).collect(),
        }
    }

    fn now(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn begin(&self, op: &PlannedOp, position: usize) {
        self.started[op.worker].fetch_max(position + 1, Ordering::SeqCst);
        let span = &self.spans[op.rank];
        if span.begin.load(Ordering::SeqCst) != 0 {
            return;
        }
        if op.op.is_contended() {
            self.contended_started.lock().unwrap().insert(op.number);
        }
        if op.op == RoundOp::CounterUpdate {
            self.increments_started.fetch_add(1, Ordering::SeqCst);
        }
        span.begin.store(self.now(), Ordering::SeqCst);
    }

    fn complete(&self, op: &PlannedOp, position: usize, commit: u64, version: u64) {
        if op.op.is_contended() {
            self.contended_completed.lock().unwrap().insert(op.number);
        }
        if op.op == RoundOp::CounterUpdate {
            self.increments_completed.fetch_add(1, Ordering::SeqCst);
        }
        self.newest_returned.fetch_max(version, Ordering::SeqCst);
        self.completed[op.worker].fetch_max(position + 1, Ordering::SeqCst);
        let span = &self.spans[op.rank];
        // What the worker's state last saw land may be an earlier operation's.
        if commit > span.begin.load(Ordering::SeqCst) {
            span.commit.store(commit, Ordering::SeqCst);
            span.version.store(version, Ordering::SeqCst);
        }
        span.end.store(self.now(), Ordering::SeqCst);
    }

    fn give_up(&self, op: &PlannedOp) {
        self.spans[op.rank]
            .gave_up
            .store(self.now(), Ordering::SeqCst);
    }

    fn started(&self) -> ClockView {
        ClockView {
            ops: self
                .started
                .iter()
                .map(|ops| ops.load(Ordering::SeqCst))
                .collect(),
            contended: self.contended_started.lock().unwrap().clone(),
            increments: self.increments_started.load(Ordering::SeqCst),
            version: self.newest_returned.load(Ordering::SeqCst),
        }
    }

    fn completed(&self) -> ClockView {
        ClockView {
            ops: self
                .completed
                .iter()
                .map(|ops| ops.load(Ordering::SeqCst))
                .collect(),
            contended: self.contended_completed.lock().unwrap().clone(),
            increments: self.increments_completed.load(Ordering::SeqCst),
            version: self.newest_returned.load(Ordering::SeqCst),
        }
    }
}

/// Everything the tasks of a round share.
pub struct RoundContext {
    pub n: usize,
    pub uri: String,
    pub oracle: Oracle,
    /// The seed rows are written from, which a self-test can tell from the
    /// oracle's.
    pub data_seed: u64,
    pub scale: Scale,
    pub schema: SchemaRef,
    pub workers: usize,
    pub stride: i64,
    pub round: Round,
    pub model: RoundModel,
    pub clocks: Clocks,
    pub gates: Vec<Gate>,
    pub open: Vec<watch::Sender<bool>>,
    /// Per rank, that the operation got to its hold point; `arrivals` wait for it.
    pub arrived: Vec<Arc<watch::Sender<bool>>>,
    pub arrivals: Vec<Gate>,
    pub gate_timeouts: Arc<AtomicU64>,
    pub start: Barrier,
    pub writers_done: watch::Sender<bool>,
    pub journal: Arc<Journal>,
    pub reads: Arc<Journal>,
    pub state: Arc<IncidentState>,
    pub session: Option<Arc<Session>>,
    pub before: u64,
    /// The oldest version there was when the round began.
    pub oldest: u64,
    pub digests: BTreeMap<u64, u64>,
    pub min_commits: AtomicU64,
    pub max_commits: AtomicU64,
    /// The newest version any cleanup of this round was allowed to remove from.
    pub cleanup_boundary: AtomicU64,
    /// No row held a key of the round when it began.
    pub only_new_keys: bool,
    /// The index on the key was there when the round began.
    pub indexed_before: bool,
    /// A reader found that a file of the oldest version is gone.
    pub oldest_gone: AtomicBool,
    /// The version and digest of what each worker's handle showed.
    pub handle_reads: Mutex<Vec<(u64, u64)>>,
    /// The fragments each version's transaction touched, by version.
    pub touched: Mutex<BTreeMap<u64, BTreeSet<u64>>>,
    /// What the shared upserts left, as the history check found it.
    pub history_contended: Mutex<ClassRows>,
    pub serial: bool,
    pub self_test: Option<&'static SelfTest>,
}

impl RoundContext {
    async fn open_handle(
        &self,
        wrapper: Option<Arc<StressWrapper>>,
        version: Option<u64>,
    ) -> lance::Result<Dataset> {
        let mut builder = DatasetBuilder::from_uri(&self.uri);
        if let Some(wrapper) = wrapper {
            builder = builder.with_store_params(ObjectStoreParams {
                object_store_wrapper: Some(wrapper),
                ..Default::default()
            });
        }
        if let Some(session) = &self.session {
            builder = builder.with_session(session.clone());
        }
        if let Some(version) = version {
            builder = builder.with_version(version);
        }
        Box::pin(builder.load()).await
    }

    pub async fn worker_handle(&self, worker: usize) -> Result<WorkerHandle, Failure> {
        let state = Arc::new(self.state.sibling(self.clocks.tick.clone()));
        state.set_phase(format!("{}.worker{worker}", self.n));
        let wrapper = Arc::new(StressWrapper::new(state.clone()));
        let ds = self
            .open_handle(Some(wrapper), None)
            .await
            .map_err(|error| fail("unreadable", self.n, error))?;
        Ok(WorkerHandle {
            hooks: Arc::new(StressHooks::new(state.clone())),
            state,
            ds,
        })
    }

    /// A commit that the plan does not know of.
    pub fn note_an_extra_commit(&self) {
        self.min_commits.fetch_add(1, Ordering::SeqCst);
        self.max_commits.fetch_add(1, Ordering::SeqCst);
    }

    fn open_gate(&self, rank: usize) {
        self.open[rank].send_replace(true);
    }

    pub fn observe(&self, observation: impl fmt::Display) {
        self.journal
            .line(format_args!("step={} observation={observation}", self.n));
    }

    /// The cleanup took two versions after the first at least: the write that the
    /// staged commits never saw is among them.
    fn hole_made(&self) -> bool {
        self.cleanup_boundary.load(Ordering::SeqCst) > self.before + 1
    }

    /// What became of a staged commit over the hole: `refused` is the fix of
    /// <https://github.com/lance-format/lance/issues/10603>, and for an insert of
    /// declared keys of <https://github.com/lance-format/lance/issues/10659>;
    /// `landed` only right for what changes no fragment and inserts no declared
    /// key, an index build withdrawing the coverage the hidden rewrite made stale
    /// (<https://github.com/lance-format/lance/issues/10660>); `masked` a missing
    /// file that came first, and `conflict` any other conflict.
    fn observe_hole(&self, op: &PlannedOp, result: &lance::Result<Attempted>) {
        let hidden = self.round.hole.unwrap_or("none");
        let outcome = match result {
            Ok(_) => "landed",
            Err(error) if is_hole_refusal(error) => "refused",
            Err(Error::RetryableCommitConflict { .. }) => "conflict",
            Err(error) if is_gone(error) => "masked",
            Err(_) => "error",
        };
        self.observe(format_args!("hole_{outcome}:{}:{hidden}", op.op.kind()));
    }
}

pub struct WorkerHandle {
    ds: Dataset,
    state: Arc<IncidentState>,
    hooks: Arc<StressHooks>,
}

pub struct WorkerEnd {
    pub worker: usize,
    pub handle: WorkerHandle,
    /// The first of its operations that has not gone through.
    pub next: usize,
}

pub fn touched_fragments(operation: &Operation) -> BTreeSet<u64> {
    match operation {
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            ..
        } => updated_fragments
            .iter()
            .map(|fragment| fragment.id)
            .chain(deleted_fragment_ids.iter().copied())
            .collect(),
        Operation::Update {
            updated_fragments,
            removed_fragment_ids,
            ..
        } => updated_fragments
            .iter()
            .map(|fragment| fragment.id)
            .chain(removed_fragment_ids.iter().copied())
            .collect(),
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .flat_map(|group| group.old_fragments.iter().map(|fragment| fragment.id))
            .collect(),
        _ => BTreeSet::new(),
    }
}

/// One try of an operation.
fn attempt<'a>(
    context: &'a RoundContext,
    handle: &'a mut WorkerHandle,
    op: &PlannedOp,
) -> BoxFuture<'a, lance::Result<Attempted>> {
    let number = op.number;
    let stride = context.stride;
    let worker = op.worker;
    let hooks = handle.hooks.clone();
    let ds = &mut handle.ds;
    let full_source = move |ids: &[i64]| -> Box<dyn RecordBatchReader + Send> {
        Box::new(source_of(full_batch(
            &context.schema,
            context.data_seed,
            &context.scale,
            ids,
            number,
        )))
    };
    let keys_predicate = move || {
        let keys: Vec<String> = context.round.keys.iter().map(i64::to_string).collect();
        format!("id IN ({})", keys.join(","))
    };
    match op.op {
        RoundOp::Append { first_id, rows } => Box::pin(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: context.scale.rows_per_file,
                progress: hooks,
                ..Default::default()
            };
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            Box::pin(ds.append(full_source(&ids), Some(params))).await?;
            Ok(Attempted::default())
        }),
        RoundOp::Delete { lo, hi } => {
            Box::pin(async move { delete(ds, class_range(stride, worker, lo, hi)).await })
        }
        RoundOp::ContendedDelete => Box::pin(async move { delete(ds, keys_predicate()).await }),
        RoundOp::Update { lo, hi } => Box::pin(async move {
            let predicate = class_range(stride, worker, lo, hi);
            if context.self_test == Some(&TORN_SNAPSHOT) {
                let middle = (lo + hi) / 2;
                let first_half = format!("{predicate} AND id < {middle}");
                let first = update(ds, &first_half, &number.to_string()).await?;
                context.note_an_extra_commit();
                let second_half = format!("{predicate} AND id >= {middle}");
                let mut second = update(ds, &second_half, &number.to_string()).await?;
                if let (Some(first), Some(second)) = (first.counts, second.counts.as_mut()) {
                    second.updated += first.updated;
                }
                return Ok(second);
            }
            update(ds, &predicate, &number.to_string()).await
        }),
        RoundOp::ContendedUpdate => {
            Box::pin(async move { update(ds, &keys_predicate(), &number.to_string()).await })
        }
        RoundOp::CounterUpdate => Box::pin(async move {
            let predicate = format!("id % {stride} = {}", context.workers + 1);
            update(ds, &predicate, "gen + 1").await
        }),
        RoundOp::MergeInsert { first_id, rows } => Box::pin(async move {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            upsert(ds, full_source(&ids)).await
        }),
        RoundOp::ContendedUpsert => {
            Box::pin(async move { upsert(ds, full_source(&context.round.keys)).await })
        }
        RoundOp::MergeInsertColumns { first_id, rows } => Box::pin(async move {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            let source = stamp_source(&context.schema, ids, number);
            let (merged, stats) = Box::pin(stamp_job(ds)?.execute_reader(source)).await?;
            *ds = merged.as_ref().clone();
            Ok(merge_report(&stats))
        }),
        RoundOp::CreateIndex(kind) => {
            let build = op_future(
                ds,
                &hooks,
                context.data_seed,
                context.scale,
                Op::CreateIndex(kind),
                None,
            );
            Box::pin(async move {
                Ok(Attempted {
                    output: build.await?,
                    ..Default::default()
                })
            })
        }
        RoundOp::OptimizeIndices => Box::pin(async move {
            Box::pin(ds.optimize_indices(&OptimizeOptions::new().progress(hooks))).await?;
            Ok(Attempted::default())
        }),
        RoundOp::Compact => {
            let compaction = op_future(
                ds,
                &hooks,
                context.data_seed,
                context.scale,
                Op::Compact,
                None,
            );
            Box::pin(async move {
                Ok(Attempted {
                    output: compaction.await?,
                    ..Default::default()
                })
            })
        }
        RoundOp::Reserve => Box::pin(async move {
            let transaction = Transaction::new(
                ds.version().version,
                Operation::ReserveFragments { num_fragments: 1 },
                None,
            );
            *ds = Box::pin(CommitBuilder::new(Arc::new(ds.clone())).execute(transaction)).await?;
            Ok(Attempted::default())
        }),
        RoundOp::Cleanup { retain } => Box::pin(async move {
            let policy = CleanupPolicyBuilder::default()
                .retain_n_versions(ds, retain)
                .await?
                .error_if_tagged_old_versions(false)
                .build();
            if let Some(boundary) = policy.before_version {
                context
                    .cleanup_boundary
                    .fetch_max(boundary, Ordering::SeqCst);
            }
            let removed = Box::pin(ds.cleanup_with_policy(policy)).await?;
            context.journal.line(format_args!(
                "step={} cleanup old_versions={} data_files={} index_files={} transaction_files={} deletion_files={}",
                context.n,
                removed.old_versions,
                removed.data_files_removed,
                removed.index_files_removed,
                removed.transaction_files_removed,
                removed.deletion_files_removed
            ));
            Ok(Attempted::default())
        }),
    }
}

/// A first try prepared without a commit.
pub enum Staged {
    Transaction {
        transaction: Box<Transaction>,
        affected_rows: Option<RowAddrTreeMap>,
        counts: Option<RowCounts>,
    },
    Compaction(Vec<RewriteResult>),
}

/// Prepares the first try of an operation on the version its handle stands on,
/// as a distributed writer does. `None` if there is nothing to commit.
pub async fn stage(
    context: &RoundContext,
    handle: &WorkerHandle,
    op: &PlannedOp,
) -> lance::Result<Option<Staged>> {
    let stride = context.stride;
    let ds = &handle.ds;
    let full_source = |ids: &[i64]| -> Box<dyn RecordBatchReader + Send> {
        Box::new(source_of(full_batch(
            &context.schema,
            context.data_seed,
            &context.scale,
            ids,
            op.number,
        )))
    };
    let merged = |uncommitted: UncommittedMergeInsert| Staged::Transaction {
        transaction: Box::new(uncommitted.transaction),
        affected_rows: uncommitted.affected_rows,
        counts: merge_report(&uncommitted.stats).counts,
    };
    let staged = match op.op {
        RoundOp::Delete { lo, hi } => {
            let predicate = class_range(stride, op.worker, lo, hi);
            let deleted = DeleteBuilder::new(Arc::new(ds.clone()), predicate)
                .execute_uncommitted()
                .await?;
            Staged::Transaction {
                transaction: Box::new(deleted.transaction),
                affected_rows: deleted.affected_rows,
                counts: Some(RowCounts {
                    deleted: deleted.num_deleted_rows,
                    ..Default::default()
                }),
            }
        }
        RoundOp::MergeInsert { first_id, rows } => {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            merged(Box::pin(upsert_job(ds)?.execute_uncommitted(full_source(&ids))).await?)
        }
        RoundOp::ContendedUpsert => {
            let source = full_source(&context.round.keys);
            merged(Box::pin(upsert_job(ds)?.execute_uncommitted(source)).await?)
        }
        RoundOp::MergeInsertColumns { first_id, rows } => {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            let source = stamp_source(&context.schema, ids, op.number);
            merged(Box::pin(stamp_job(ds)?.execute_uncommitted(source)).await?)
        }
        RoundOp::Compact => {
            let options = CompactionOptions::default();
            let plan = plan_compaction(ds, &options).await?;
            let mut rewritten = Vec::new();
            for task in plan.compaction_tasks() {
                rewritten.push(Box::pin(task.execute(ds)).await?);
            }
            if rewritten.is_empty() {
                return Ok(None);
            }
            Staged::Compaction(rewritten)
        }
        RoundOp::CreateIndex(
            kind @ (IndexKind::BTree | IndexKind::BTreeId | IndexKind::BTreeStamp),
        ) => {
            let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
            let mut building = ds.clone();
            let index = Box::pin(
                building
                    .create_index_builder(&[kind.column()], IndexType::BTree, &params)
                    .name(kind.name().to_string())
                    .replace(true)
                    .execute_uncommitted(),
            )
            .await?;
            let removed_indices = ds
                .load_indices()
                .await?
                .iter()
                .filter(|old| old.name == kind.name())
                .cloned()
                .collect();
            let operation = Operation::CreateIndex {
                new_indices: vec![index],
                removed_indices,
            };
            Staged::Transaction {
                transaction: Box::new(Transaction::new(ds.version().version, operation, None)),
                affected_rows: None,
                counts: None,
            }
        }
        _ => unreachable!("only the operations of `HOLES` are staged, not {}", op.op),
    };
    Ok(Some(staged))
}

/// Commits a staged try from the handle, which still stands on the version the
/// try was prepared on.
async fn commit_staged(handle: &mut WorkerHandle, staged: Staged) -> lance::Result<Attempted> {
    match staged {
        Staged::Transaction {
            transaction,
            affected_rows,
            counts,
        } => {
            let mut commit = CommitBuilder::new(Arc::new(handle.ds.clone()));
            if let Some(affected_rows) = affected_rows {
                commit = commit.with_affected_rows(affected_rows);
            }
            handle.ds = Box::pin(commit.execute(*transaction)).await?;
            Ok(Attempted {
                counts,
                ..Default::default()
            })
        }
        Staged::Compaction(rewritten) => {
            let metrics = Box::pin(commit_compaction(
                &mut handle.ds,
                rewritten,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &CompactionOptions::default(),
            ))
            .await?;
            Ok(Attempted {
                output: OpOutput {
                    fragments_removed: metrics.fragments_removed,
                    ..Default::default()
                },
                ..Default::default()
            })
        }
    }
}

/// Runs an operation, with the tries it gets, and reads back what it wrote.
/// Returns false if it did not get through and has to wait for the others.
/// Alone it must get through.
pub async fn run_op(
    context: &RoundContext,
    handle: &mut WorkerHandle,
    op: &PlannedOp,
    position: usize,
    alone: bool,
    mut staged: Option<Staged>,
) -> Result<bool, Failure> {
    let n = context.n;
    let worker = op.worker;
    let rank = op.rank;
    if context.serial && !alone && rank > 0 {
        wait_for_rank(context, rank - 1).await;
    }
    handle.state.set_phase(format!("{n}.w{worker}.r{rank}"));
    // A staged operation with nothing staged goes the usual way.
    if op.catch_up || alone || (op.staged && staged.is_none()) {
        Box::pin(handle.ds.checkout_latest())
            .await
            .map_err(|error| fail("unreadable", n, error))?;
    }
    context.clocks.begin(op, position);
    let lower = context.clocks.completed();
    let tries = if alone { 1 } else { PARALLEL_ATTEMPTS };
    for attempt_number in 1..=tries {
        // One at a time, what an arrival waits for only starts after it ends.
        let hold = op
            .hold
            .as_ref()
            .filter(|hold| attempt_number == 1 && !alone && !(context.serial && hold.arrival));
        if let Some(hold) = hold {
            let gate = if hold.arrival {
                &context.arrivals[hold.until]
            } else {
                &context.gates[hold.until]
            };
            handle.state.arm_hold(
                hold.point.clone(),
                gate.clone().announcing(context.arrived[rank].clone()),
            );
        }
        let version_before = handle.ds.version().version;
        let staged_try = staged.take();
        let is_staged = staged_try.is_some();
        let result = match staged_try {
            Some(staged) => {
                // Worker 0 gave up before its cleanup, or the cleanup took too
                // little: the round would test nothing it is meant to.
                if !context.hole_made() {
                    return Err(fail(
                        "harness",
                        n,
                        format_args!(
                            "{} of worker {worker} was staged over a hole the round did not make",
                            op.op
                        ),
                    ));
                }
                let result = commit_staged(handle, staged).await;
                context.observe_hole(op, &result);
                result
            }
            None => attempt(context, handle, op).await,
        };
        let fired = handle.state.disarm();
        let waited = handle.state.take_gate_waits();
        // An operation that found nothing to do never got to the point, whichever
        // of the racers that is, so it says nothing about the point. A cleanup
        // commits nothing, but always lists the versions an arrival holds it at.
        let committed = handle.ds.version().version != version_before;
        if let Some(hold) = hold.filter(|hold| fired > 0 || committed || hold.arrival) {
            context.journal.line(format_args!(
                "step={n} hold=hold:{}:{} worker={worker} rank={rank} until={} fired={fired} waited={waited}",
                op.op.kind(),
                hold.point.kind(),
                hold.until
            ));
        }
        for _ in 0..handle.state.take_lost_races() {
            context.observe(format_args!("lost_manifest_race:{}", op.op.kind()));
        }
        let (min, max) = op
            .op
            .commits(result.as_ref().ok().map(|attempted| &attempted.output));
        context.min_commits.fetch_add(min, Ordering::SeqCst);
        context.max_commits.fetch_add(max, Ordering::SeqCst);
        match result {
            Ok(attempted) => {
                context.clocks.complete(
                    op,
                    position,
                    handle.state.last_commit(),
                    handle.ds.version().version,
                );
                context.open_gate(rank);
                context.journal.line(format_args!(
                    "step={n} worker={worker} rank={rank} op_end op={} attempt={attempt_number} alone={alone} outcome=ok version={}",
                    op.op,
                    handle.ds.version().version
                ));
                if attempted.tries_inside > 1 {
                    context.observe(format_args!("internal_retry:{}", op.op.kind()));
                }
                if let Some(counts) = &attempted.counts {
                    context.model.check_counts(op, position, counts, n)?;
                }
                if context.self_test == Some(&DUPLICATE_KEY) && rank == 0 {
                    let id = worker as i64;
                    let source = source_of(full_batch(
                        &context.schema,
                        context.data_seed,
                        &context.scale,
                        &[id],
                        op.number,
                    ));
                    Box::pin(handle.ds.append(source, None))
                        .await
                        .map_err(|error| fail("harness", n, error))?;
                    context.note_an_extra_commit();
                }
                check_own_read(
                    context,
                    handle,
                    op,
                    position,
                    version_before,
                    &lower,
                    attempted.counts.as_ref(),
                )
                .await?;
                return Ok(true);
            }
            Err(error) => {
                let message = one_line(&error.to_string());
                // Refusing a commit over the hole is the point of the hole. Any other
                // error of a staged commit is judged as any other operation's.
                let legal = op.op.is_legal_error(&error, &context.round, rank)
                    || (is_staged && is_hole_refusal(&error));
                context.journal.line(format_args!(
                    "step={n} worker={worker} rank={rank} op_end op={} attempt={attempt_number} alone={alone} outcome={} detail={message}",
                    op.op,
                    if legal { "legal_error" } else { "error" }
                ));
                if alone {
                    return Err(fail(
                        "retry_failed",
                        n,
                        format_args!("alone, {} of worker {worker} ended with: {message}", op.op),
                    ));
                }
                if !legal {
                    return Err(fail(
                        "error_without_fault",
                        n,
                        format_args!(
                            "{} of worker {worker} failed with nothing to explain it: {message}",
                            op.op
                        ),
                    ));
                }
                context.observe(format_args!("legal_error:{}", op.op.kind()));
                // Others that wait for its end wait for its first try, unless they
                // come one at a time.
                if !context.serial {
                    context.open_gate(rank);
                }
                Box::pin(handle.ds.checkout_latest())
                    .await
                    .map_err(|error| fail("unreadable", n, error))?;
            }
        }
    }
    context.clocks.give_up(op);
    context.open_gate(rank);
    Ok(false)
}

/// What a handle shows right after its own operation: its own write, and if the
/// operation committed, everything that finished before it began. The history
/// check then compares what the handle showed with the version it stood on.
async fn check_own_read(
    context: &RoundContext,
    handle: &WorkerHandle,
    op: &PlannedOp,
    position: usize,
    version_before: u64,
    lower: &ClockView,
    counts: Option<&RowCounts>,
) -> Result<(), Failure> {
    let n = context.n;
    let worker = op.worker;
    let what = format!("worker {worker} after {}", op.op);
    let image = context
        .oracle
        .image(&handle.ds, n, &what, false)
        .await?
        .map_err(|error| fail("unreadable", n, error))?;
    context
        .handle_reads
        .lock()
        .unwrap()
        .push((image.version, image.digest()));
    let upper = context.clocks.started();
    let placed = context.model.place(&image, n, &what)?;
    if !placed.prefixes[worker].contains(&(position + 1)) {
        return Err(fail(
            "torn_snapshot",
            n,
            format_args!(
                "{what}: its handle shows its rows as of {:?}, after operation {}",
                placed.prefixes[worker],
                position + 1
            ),
        ));
    }
    if handle.ds.version().version == version_before {
        return Ok(());
    }
    // An update or a delete that found no key has no effect to show, and an upsert
    // may have landed on top of its empty commit.
    let own_effect = match op.op {
        RoundOp::ContendedUpsert => Some(Contended::Applied(op.number)),
        RoundOp::ContendedUpdate if counts.is_some_and(|counts| counts.updated > 0) => {
            Some(Contended::Applied(op.number))
        }
        RoundOp::ContendedDelete if counts.is_some_and(|counts| counts.deleted > 0) => {
            Some(Contended::Deleted)
        }
        _ => None,
    };
    if let Some(expected) = own_effect
        && placed.contended != expected
    {
        return Err(fail(
            "lost_write",
            n,
            format_args!(
                "{what}: the commit of operation {} shows {:?}, not {expected:?}",
                op.number, placed.contended
            ),
        ));
    }
    if op.op == RoundOp::CounterUpdate && placed.increments == 0 {
        return Err(fail(
            "lost_write",
            n,
            format_args!("{what}: the commit does not show the increment"),
        ));
    }
    context
        .model
        .check_bounds(image.version, &placed, lower, &upper, n, &what)
}

pub async fn run_worker(
    context: Arc<RoundContext>,
    worker: usize,
    mut handle: WorkerHandle,
) -> Result<WorkerEnd, Failure> {
    let ops = context.round.ops_of(worker);
    // Prepared on the version the round starts from, before anyone writes.
    let staged = match ops.first() {
        Some(op) if op.staged => Some(stage(&context, &handle, op).await),
        _ => None,
    };
    context.start.wait().await;
    let mut staged = match staged {
        Some(Ok(staged)) => {
            if staged.is_none() {
                let hidden = context.round.hole.unwrap_or("none");
                context.observe(format_args!("hole_nothing:{}:{hidden}", ops[0].op.kind()));
            }
            staged
        }
        Some(Err(error)) => {
            return Err(fail(
                "error_without_fault",
                context.n,
                format_args!("staging {} of worker {worker}: {error}", ops[0].op),
            ));
        }
        None => None,
    };
    for (position, op) in ops.iter().enumerate() {
        if let Some(rank) = op.start_after {
            wait_for_rank(&context, rank).await;
        }
        if !run_op(&context, &mut handle, op, position, false, staged.take()).await? {
            for later in &ops[position + 1..] {
                context.open_gate(later.rank);
            }
            return Ok(WorkerEnd {
                worker,
                handle,
                next: position,
            });
        }
    }
    Ok(WorkerEnd {
        worker,
        next: ops.len(),
        handle,
    })
}

/// Waits, with no limit, until the operation of this rank is over. A hang here is
/// the watchdog's to find, as in the serial mode.
async fn wait_for_rank(context: &RoundContext, rank: usize) {
    let mut opened = context.open[rank].subscribe();
    let _ = opened.wait_for(|opened| *opened).await;
}

/// One read of the latest version. Returns its version and digest.
async fn read_latest(
    context: &RoundContext,
    ds: &Dataset,
    lower: &ClockView,
    what: &str,
    previous: Option<u64>,
) -> Result<(u64, u64), Failure> {
    let n = context.n;
    let image = context
        .oracle
        .image(ds, n, what, true)
        .await?
        .map_err(|error| fail("unreadable", n, format_args!("{what}: {error}")))?;
    if let Some(previous) = previous
        && image.version < previous
    {
        return Err(fail(
            "stale_read",
            n,
            format_args!(
                "{what}: the latest version went back from {previous} to {}",
                image.version
            ),
        ));
    }
    let upper = context.clocks.started();
    let placed = context.model.place(&image, n, what)?;
    context
        .model
        .check_bounds(image.version, &placed, lower, &upper, n, what)?;
    Box::pin(context.oracle.check_indices(ds, n, &image, true)).await?;
    let digest = image.digest();
    context.reads.line(format_args!(
        "step={n} what={} version={} digest={digest:016x} rows={} prefixes={:?} contended={:?} increments={}",
        what.replace(' ', "_"),
        image.version,
        image.rows.len(),
        placed.prefixes,
        placed.contended,
        placed.increments
    ));
    Ok((image.version, digest))
}

/// A manifest a cleanup took is `DatasetNotFound`, a file it took `NotFound`.
/// Lance's refusal of a commit over versions that a cleanup removed, in the words
/// of `check_missing_versions`.
fn is_hole_refusal(error: &Error) -> bool {
    matches!(error, Error::RetryableCommitConflict { .. })
        && error.to_string().contains("were cleaned up")
}

fn is_gone(error: &Error) -> bool {
    error.is_not_found() || matches!(error, Error::DatasetNotFound { .. })
}

pub async fn run_reader(
    context: Arc<RoundContext>,
    index: usize,
    plan: PlannedReader,
) -> Result<Vec<(u64, u64)>, Failure> {
    let n = context.n;
    let what = format!("reader {index} {}", plan.kind.name());
    let state = plan.hold.as_ref().map(|_| {
        let state = Arc::new(context.state.sibling(context.clocks.tick.clone()));
        state.set_phase(format!("{n}.reader{index}"));
        state
    });
    let wrapper = state
        .as_ref()
        .map(|state| Arc::new(StressWrapper::new(state.clone())));
    let unreadable = |error: Error| fail("unreadable", n, format_args!("{what}: {error}"));
    let pin_to = match plan.kind {
        ReaderKind::FreshOpen | ReaderKind::CheckoutLatest => None,
        ReaderKind::Pinned => Some(context.before),
        ReaderKind::OldSnapshot => Some(context.oldest),
    };
    let handle = match plan.kind {
        ReaderKind::FreshOpen => None,
        _ => Some(
            context
                .open_handle(wrapper.clone(), pin_to)
                .await
                .map_err(unreadable)?,
        ),
    };
    let mut writers_done = context.writers_done.subscribe();
    context.start.wait().await;
    // Armed only now: opening the handle must not wait for a round not yet begun.
    if let (Some(state), Some(hold)) = (&state, &plan.hold) {
        state.arm_hold(hold.point.clone(), context.gates[hold.until].clone());
    }
    if let Some(rank) = plan.start_after {
        wait_for_rank(&context, rank).await;
    }
    let mut reads = Vec::new();
    match (plan.kind, handle) {
        (ReaderKind::FreshOpen, _) => loop {
            let last = *writers_done.borrow();
            let lower = context.clocks.completed();
            let stale = last && context.self_test == Some(&STALE_READ);
            let ds = context
                .open_handle(wrapper.clone(), stale.then_some(context.before))
                .await
                .map_err(unreadable)?;
            let previous = reads.last().map(|(version, _)| *version);
            reads.push(read_latest(&context, &ds, &lower, &what, previous).await?);
            if last {
                break;
            }
        },
        (ReaderKind::CheckoutLatest, Some(mut ds)) => loop {
            let last = *writers_done.borrow();
            let lower = context.clocks.completed();
            Box::pin(ds.checkout_latest()).await.map_err(unreadable)?;
            let previous = reads.last().map(|(version, _)| *version);
            reads.push(read_latest(&context, &ds, &lower, &what, previous).await?);
            if last {
                break;
            }
        },
        (_, Some(ds)) => {
            let expected = pin_to.and_then(|version| context.digests.get(&version).copied());
            for pass in 0..2 {
                if pass == 1 {
                    let _ = writers_done.wait_for(|done| *done).await;
                }
                let moved = if pass == 1 && context.self_test == Some(&SNAPSHOT_CHANGED) {
                    Some(context.open_handle(None, None).await.map_err(unreadable)?)
                } else {
                    None
                };
                match context
                    .oracle
                    .image(moved.as_ref().unwrap_or(&ds), n, &what, true)
                    .await?
                {
                    Ok(image) => {
                        let digest = image.digest();
                        context.reads.line(format_args!(
                            "step={n} what={} pass={pass} version={} digest={digest:016x} rows={}",
                            what.replace(' ', "_"),
                            image.version,
                            image.rows.len()
                        ));
                        if let Some(expected) = expected
                            && expected != digest
                        {
                            return Err(fail(
                                "snapshot_changed",
                                n,
                                format_args!(
                                    "{what}: version {} read as {digest:016x}, and before as {expected:016x}",
                                    image.version
                                ),
                            ));
                        }
                        // The version a round starts from stays, indices included.
                        if plan.kind == ReaderKind::Pinned {
                            let pinned = moved.as_ref().unwrap_or(&ds);
                            Box::pin(context.oracle.check_indices(pinned, n, &image, true)).await?;
                        }
                        reads.push((image.version, digest));
                    }
                    // Without a cleanup in the round, the version it could see at the
                    // start must stay; `check_history` holds a cleanup to its boundary.
                    Err(error)
                        if plan.kind == ReaderKind::OldSnapshot
                            && context.round.has_cleanup()
                            && is_gone(&error) =>
                    {
                        context.oldest_gone.store(true, Ordering::SeqCst);
                        context.observe("old_snapshot_gone");
                        context.reads.line(format_args!(
                            "step={n} what={} pass={pass} gone detail={}",
                            what.replace(' ', "_"),
                            one_line(&error.to_string())
                        ));
                    }
                    Err(error) => return Err(unreadable(error)),
                }
            }
        }
        _ => unreachable!("every kind but a fresh open has a handle"),
    }
    if let (Some(state), Some(hold)) = (&state, &plan.hold) {
        let fired = state.disarm();
        let waited = state.take_gate_waits();
        context.journal.line(format_args!(
            "step={n} hold=hold:reader_{}:{} reader={index} until={} fired={fired} waited={waited}",
            plan.kind.name(),
            hold.point.kind(),
            hold.until
        ));
    }
    Ok(reads)
}

pub async fn joined<T>(task: JoinHandle<Result<T, Failure>>, n: usize) -> Result<T, Failure> {
    task.await.unwrap_or_else(|error| {
        Err(fail(
            "panic",
            n,
            format_args!("a task of the round: {error}"),
        ))
    })
}
