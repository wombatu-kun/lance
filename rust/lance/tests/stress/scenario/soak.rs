// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The `soak` scenario: a long-lived service on one dataset.
//!
//! Workers keep one handle each for the whole run, all of them on one `Session`
//! with small caches, and go round the work of a service: appends, merge inserts,
//! updates and deletes, compaction, index builds and optimizations, cleanups that
//! keep a few versions, scans and index queries. A row belongs to the class
//! `id % workers`, and a worker only changes its own class, so whatever version its
//! handle is on, its rows are what its own operations left, which its scans and
//! queries check. The operations of a worker follow from the seed and from its rows
//! alone: every run of a seed takes a prefix of the same sequence, while how far it
//! gets and how the workers interleave depend on timing.
//!
//! A worker waits a little between its operations, as a service waits for work:
//! with none, a compaction loses the race for its commit to the writes that land
//! meanwhile most of the time, and every loss leaves the files it wrote behind.
//!
//! A run takes `SAMPLES` samples at even intervals. For each one the workers finish
//! what they are doing and wait, and once the runtime is quiet the sample measures
//! the process, the caches of the session and the dataset, so that what an
//! operation holds while it runs does not count. Then a cleanup that may delete
//! unverified files runs, as a service would run one with nothing in flight, and
//! must leave no file that no version references, and the versions, files and bytes
//! of the dataset are measured after it, so that they count what the versions it
//! keeps hold rather than what lost races left; every `CHECK_EVERY` samples the
//! dataset is checked against the rows of every class too. A cache may never hold
//! more than its capacity, and each series must level off: the median of its last
//! third of the samples may exceed that of the middle third by no more than the
//! tolerance of the series. The first third warms the caches up.
//!
//! Lance's own tests build `lance-io` with `test-util`, under which a store keeps a
//! record of every request it served until someone takes its statistics. A sample
//! takes them, from the store the session shares and from the handle of the
//! checks, before it reads the heap: a build without the feature keeps counts
//! alone.

use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::Write as _;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{Float32Array, RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use lance::Dataset;
use lance::Error;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::{WriteMode, WriteParams};
use lance::session::Session;
use lance_core::cache::CacheStats;
use lance_index::scalar::FullTextSearchQuery;
use lance_io::object_store::ObjectStore;
use tokio::sync::{Mutex as TurnLock, RwLock};
use tokio::task::{JoinError, JoinSet};

use super::concurrency::{
    ClassRows, class_range, delete, full_batch, id_run, schema, source_of, update, upsert,
};
use super::{
    Answers, CATEGORIES, Failure, IVF_PARTITIONS, IndexKind, Journal, NEIGHBORS, Op, PlanRng, Run,
    Scale, category, check_neighbors, compare_ids, fail, file_sizes, fingerprint, int64_column,
    is_commit_conflict, listed_versions, op_future, pick_from, soak_secs, tokens, vector,
};
use crate::harness::SelfTest;
use crate::incidents::{IncidentState, StressHooks};
use crate::leaks::{live_heap_bytes, malloc_bytes, settle, trimmed_anon_kib};

pub const SAMPLES: usize = 24;
const CHECK_EVERY: usize = 4;
const MIDDLE: Range<usize> = 8..16;
const LAST: Range<usize> = 16..24;
pub const DEFAULT_SECS: u64 = 180;
pub const SELF_TEST_SECS: u64 = 24;
/// The samples of a seed, one row each, in its directory.
pub const SERIES_FILE: &str = "soak.csv";
/// How many operations of each worker a plan shows.
const PLANNED_OPS: usize = 48;
/// How many its fingerprint covers, those past the ones shown as a digest: a
/// class reaches the edges of its size only after a while.
const DIGEST_OPS: usize = 1024;
/// Caches this small are full by the end of the first third, so that from then on
/// a cache that does not evict shows.
const INDEX_CACHE_BYTES: usize = 2 << 20;
const METADATA_CACHE_BYTES: usize = 1 << 20;
/// How many times an operation is tried that keeps losing the race for a commit.
const TRIES: usize = 4;
/// The most a worker waits between two operations, per worker: the commits of the
/// run land at about the same pace whatever the number of workers.
const THINK_MS_PER_WORKER: u64 = 100;
/// How long a sample waits for the tasks of finished operations to end.
const SAMPLE_QUIET: Duration = Duration::from_secs(5);
/// How long the blocking pool of a soak keeps an idle thread, and how long the
/// counts of a sample must hold, which is longer.
pub const BLOCKING_KEEP_ALIVE: Duration = Duration::from_millis(500);
const SAMPLE_STILL: Duration = Duration::from_millis(600);
/// What the self-test leaks at every sample.
pub const SELF_TEST_LEAK: usize = 1 << 20;
/// Tells the operations of the workers apart in `gen`; above the most workers.
const NUMBER_STRIDE: i64 = 8;
const INDICES: [IndexKind; 4] = [
    IndexKind::BTree,
    IndexKind::BTreeId,
    IndexKind::IvfFlat,
    IndexKind::Inverted,
];

/// What a sample measures, as the columns of `SERIES_FILE` name it, and how far
/// the median of the last third may rise above that of the middle third.
const SERIES: [(&str, i64); 8] = [
    // What Rust code holds, the series that matters.
    ("heap_bytes", 2 << 20),
    // What malloc has handed out: the heap, and what C code holds besides it. It
    // moves by a third more than the heap, for chunk headers and the caches of each
    // thread, which a tolerance of the heap's would make the tighter of the two.
    ("malloc_bytes", 3 << 20),
    ("fds", 4),
    ("threads", 8),
    // Readers that the caches of the session hold keep tasks of their own: the
    // median of a third has risen by up to 4 on its own.
    ("tasks", 12),
    // What the versions a cleanup keeps hold, measured after it.
    ("versions", 16),
    ("files", 100),
    ("dataset_bytes", 2 << 20),
];

#[derive(Clone, Copy, Debug)]
enum SoakOp {
    Append {
        first_id: i64,
        rows: i64,
    },
    /// Updates the rows with these ids that exist, the newest of the class, and
    /// inserts the others.
    MergeInsert {
        first_id: i64,
        rows: i64,
    },
    /// Sets `gen` on the rows of the class in a range of ids.
    Update {
        lo: i64,
        hi: i64,
    },
    /// Deletes the rows of the class in a range of ids, the oldest ones.
    Delete {
        lo: i64,
        hi: i64,
    },
    /// Reads the class.
    Scan,
    /// Reads rows of the class through an index, or through a scan while there is
    /// none.
    Query(IndexKind),
    CreateIndex(IndexKind),
    OptimizeIndices,
    Compact,
    /// Removes the versions before the last `retain`, but none a handle is on.
    Cleanup {
        retain: usize,
    },
}

impl SoakOp {
    fn kind(&self) -> &'static str {
        match self {
            Self::Append { .. } => "append",
            Self::MergeInsert { .. } => "merge_insert",
            Self::Update { .. } => "update",
            Self::Delete { .. } => "delete",
            Self::Scan => "scan",
            Self::Query(_) => "query",
            Self::CreateIndex(_) => "create_index",
            Self::OptimizeIndices => "optimize_indices",
            Self::Compact => "compact",
            Self::Cleanup { .. } => "cleanup",
        }
    }

    fn changes_rows(&self) -> bool {
        matches!(
            self,
            Self::Append { .. }
                | Self::MergeInsert { .. }
                | Self::Update { .. }
                | Self::Delete { .. }
        )
    }
}

impl fmt::Display for SoakOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Append { first_id, rows } | Self::MergeInsert { first_id, rows } => {
                write!(f, "{}:first={first_id},rows={rows}", self.kind())
            }
            Self::Update { lo, hi } | Self::Delete { lo, hi } => {
                write!(f, "{}:ids={lo}..{hi}", self.kind())
            }
            Self::Query(kind) | Self::CreateIndex(kind) => {
                write!(f, "{}:{}", self.kind(), kind.name())
            }
            Self::Cleanup { retain } => write!(f, "cleanup:retain={retain}"),
            Self::Scan | Self::OptimizeIndices | Self::Compact => f.write_str(self.kind()),
        }
    }
}

/// The operations of one worker, drawn as it goes: each depends on the rows the
/// earlier ones left in its class.
struct OpSource {
    rng: PlanRng,
    worker: usize,
    stride: i64,
    /// Its class: each id with the `gen` it holds. Deletes take the oldest ids and
    /// writes add new ones above the newest, so the ids form one run.
    rows: ClassRows,
    next_id: i64,
    /// The class stays between these sizes, around the one it starts with.
    low: usize,
    high: usize,
    drawn: u64,
}

/// An operation as a worker runs it.
struct Drawn {
    seq: u64,
    /// The `gen` it writes.
    number: i64,
    /// Whether the handle moves to the latest version first.
    catch_up: bool,
    /// How long the worker waits after it.
    think: Duration,
    op: SoakOp,
}

impl OpSource {
    fn new(seed: u64, worker: usize, workers: usize, initial_rows: i64) -> Self {
        let stride = workers as i64;
        let rows: ClassRows = (worker as i64..initial_rows)
            .step_by(workers)
            .map(|id| (id, 0))
            .collect();
        let next_id = rows
            .keys()
            .next_back()
            .map_or(worker as i64, |last| last + stride);
        let size = rows.len();
        Self {
            rng: PlanRng(seed),
            worker,
            stride,
            rows,
            next_id,
            low: size * 3 / 4,
            high: size * 5 / 4,
            drawn: 0,
        }
    }

    fn id_at(&self, position: usize) -> i64 {
        *self
            .rows
            .keys()
            .nth(position)
            .expect("a position within the class")
    }

    /// The next operation, already applied to the rows: a worker runs it until it
    /// lands, or fails the seed.
    fn next(&mut self) -> Drawn {
        self.drawn += 1;
        let number = self.drawn as i64 * NUMBER_STRIDE + self.worker as i64 + 1;
        let catch_up = self.rng.chance(0.7);
        let think =
            Duration::from_millis(self.rng.range(0..=THINK_MS_PER_WORKER * self.stride as u64));
        // Never more than the class holds: it stays above `low` less a run.
        let run = self.rng.range(10..=40) as usize;
        let roll = self.rng.range(0..=99);
        let size = self.rows.len();
        let delete_oldest = |source: &Self| SoakOp::Delete {
            lo: source.id_at(0),
            hi: source.id_at(run - 1) + 1,
        };
        let append = |source: &Self| SoakOp::Append {
            first_id: source.next_id,
            rows: run as i64,
        };
        let op = if size > self.high {
            delete_oldest(self)
        } else if size < self.low {
            append(self)
        } else {
            match roll {
                0..=11 => append(self),
                12..=23 => SoakOp::MergeInsert {
                    first_id: self.id_at(size - run / 2),
                    rows: run as i64,
                },
                24..=37 => {
                    let start = self.rng.range(0..=(size - run) as u64) as usize;
                    SoakOp::Update {
                        lo: self.id_at(start),
                        hi: self.id_at(start + run - 1) + 1,
                    }
                }
                38..=51 => delete_oldest(self),
                52..=59 => SoakOp::Scan,
                60..=71 => SoakOp::Query(*pick_from(&mut self.rng, &INDICES)),
                72..=77 => SoakOp::CreateIndex(*pick_from(&mut self.rng, &INDICES)),
                78..=83 => SoakOp::OptimizeIndices,
                84..=91 => SoakOp::Compact,
                _ => SoakOp::Cleanup {
                    retain: self.rng.range(1..=3) as usize,
                },
            }
        };
        self.apply(&op, number);
        Drawn {
            seq: self.drawn,
            number,
            catch_up,
            think,
            op,
        }
    }

    fn apply(&mut self, op: &SoakOp, number: i64) {
        match *op {
            SoakOp::Append { first_id, rows } | SoakOp::MergeInsert { first_id, rows } => {
                self.rows
                    .extend(id_run(first_id, rows, self.stride).map(|id| (id, number)));
                self.next_id = self.next_id.max(first_id + rows * self.stride);
            }
            SoakOp::Update { lo, hi } => {
                for (_, stamp) in self.rows.range_mut(lo..hi) {
                    *stamp = number;
                }
            }
            SoakOp::Delete { lo, hi } => self.rows.retain(|id, _| !(lo..hi).contains(id)),
            _ => {}
        }
    }
}

/// What a seed decides about its run.
#[derive(Clone, Debug)]
pub(super) struct SoakPlan {
    secs: u64,
    workers: usize,
    pub(super) stable_row_ids: bool,
    initial_rows: i64,
    /// Where the operations of each worker start from.
    worker_seeds: Vec<u64>,
}

pub(super) fn plan(rng: &mut PlanRng, initial_rows: i64) -> SoakPlan {
    let workers = rng.range(2..=4) as usize;
    SoakPlan {
        secs: soak_secs(),
        workers,
        stable_row_ids: rng.chance(0.5),
        initial_rows,
        worker_seeds: (0..workers).map(|_| rng.next()).collect(),
    }
}

impl SoakPlan {
    fn interval(&self) -> Duration {
        Duration::from_secs(self.secs) / SAMPLES as u32
    }

    fn source(&self, worker: usize) -> OpSource {
        OpSource::new(
            self.worker_seeds[worker],
            worker,
            self.workers,
            self.initial_rows,
        )
    }

    pub(super) fn render(&self) -> String {
        let mut text = format!(
            "soak secs={} interval_ms={} workers={} stable_row_ids={} index_cache_bytes={INDEX_CACHE_BYTES} metadata_cache_bytes={METADATA_CACHE_BYTES}\n",
            self.secs,
            self.interval().as_millis(),
            self.workers,
            self.stable_row_ids
        );
        for worker in 0..self.workers {
            let mut source = self.source(worker);
            let mut rest = String::new();
            for seq in 1..=DIGEST_OPS {
                let drawn = source.next();
                let lines = if seq <= PLANNED_OPS {
                    &mut text
                } else {
                    &mut rest
                };
                writeln!(
                    lines,
                    "  worker={worker} seq={} catch_up={} think_ms={} op={}",
                    drawn.seq,
                    drawn.catch_up,
                    drawn.think.as_millis(),
                    drawn.op
                )
                .unwrap();
            }
            writeln!(
                text,
                "  worker={worker} seq={}..={DIGEST_OPS} digest={}",
                PLANNED_OPS + 1,
                fingerprint(&rest)
            )
            .unwrap();
        }
        text
    }
}

pub(super) fn initial_source(
    data_seed: u64,
    initial_rows: i64,
    scale: Scale,
) -> impl RecordBatchReader + Send + 'static {
    let ids: Vec<i64> = (0..initial_rows).collect();
    source_of(full_batch(&schema(false), data_seed, &scale, &ids, 0))
}

/// What the workers and the samples share.
struct Shared {
    session: Arc<Session>,
    schema: SchemaRef,
    data_seed: u64,
    scale: Scale,
    stride: i64,
    /// An operation runs under the read lock. A sample takes the write lock, which
    /// waits for the operations in flight and holds back the next ones.
    pause: RwLock<()>,
    stop: AtomicBool,
    /// Set by a worker that fails, before it lets go of the pause.
    failed: AtomicBool,
    /// Cleanups take turns, as a service would schedule them.
    cleanup: TurnLock<()>,
    /// The operations of each worker, with the rows of its class.
    sources: Vec<Mutex<OpSource>>,
    /// The version each worker's handle is on.
    handles: Vec<AtomicU64>,
    /// The newest version a cleanup was allowed to remove the versions before.
    expired_below: AtomicU64,
    indices: Mutex<BTreeSet<IndexKind>>,
    ops: AtomicU64,
    /// The sample being collected, which a failure gives as its step.
    sample: AtomicUsize,
    journal: Arc<Journal>,
    /// Progress hooks for the operations that take them, logging nowhere: the
    /// calls of a long run would fill the call log.
    hooks: Arc<StressHooks>,
    /// The store every handle of the session shares, whose requests a sample
    /// counts and lets go of.
    store: Arc<ObjectStore>,
}

impl Shared {
    /// The oldest version a worker's handle is on. A cleanup keeps it and every
    /// later one: a write on that handle commits a version after it, and must not
    /// take the number of one a cleanup removed
    /// (<https://github.com/lance-format/lance/issues/3718>).
    fn oldest_handle(&self) -> u64 {
        self.handles
            .iter()
            .map(|version| version.load(Ordering::SeqCst))
            .min()
            .unwrap_or(0)
    }

    fn rows(&self, worker: usize) -> ClassRows {
        self.sources[worker].lock().unwrap().rows.clone()
    }

    fn all_rows(&self) -> ClassRows {
        (0..self.sources.len())
            .flat_map(|worker| self.rows(worker))
            .collect()
    }
}

/// Either Lance failed the operation, which may be retried, or a check of what it
/// read failed.
enum OpError {
    Lance(Error),
    Wrong(Failure),
}

impl From<Error> for OpError {
    fn from(error: Error) -> Self {
        Self::Lance(error)
    }
}

/// Whether an error says that a commit lost to another one, which only a retry on
/// a newer version can win.
fn is_conflict(error: &Error) -> bool {
    matches!(
        error,
        Error::CommitConflict { .. }
            | Error::RetryableCommitConflict { .. }
            | Error::IncompatibleTransaction { .. }
            | Error::TooMuchWriteContention { .. }
    ) || is_commit_conflict(&error.to_string())
}

/// Marks the run failed when dropped: a worker forgets it once its operation has
/// landed.
struct MarkFailed<'a>(&'a AtomicBool);

impl Drop for MarkFailed<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

async fn run_worker(shared: Arc<Shared>, worker: usize, mut ds: Dataset) -> Result<(), Failure> {
    loop {
        let running = shared.pause.read().await;
        if shared.stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        // Dropped before `running` on an error or a panic: the task ends only after
        // it lets go of the pause, and a sample that takes the pause meanwhile must
        // not check the rows the failed operation was to leave.
        let failing = MarkFailed(&shared.failed);
        let think = Box::pin(run_next(&shared, worker, &mut ds)).await?;
        std::mem::forget(failing);
        // Outside the lock, so that a sample need not wait for it.
        drop(running);
        tokio::time::sleep(think).await;
    }
}

/// Draws the next operation of a worker and runs it; how long the worker waits
/// after it.
async fn run_next(shared: &Shared, worker: usize, ds: &mut Dataset) -> Result<Duration, Failure> {
    let drawn = shared.sources[worker].lock().unwrap().next();
    let n = shared.sample.load(Ordering::SeqCst);
    if drawn.catch_up {
        Box::pin(ds.checkout_latest())
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        shared.handles[worker].store(ds.version().version, Ordering::SeqCst);
    }
    shared.journal.line(format_args!(
        "step={n} soak worker={worker} seq={} op={} begin",
        drawn.seq, drawn.op
    ));
    let started = Instant::now();
    let (outcome, tries) = Box::pin(run_op(shared, worker, ds, &drawn, n)).await?;
    shared.handles[worker].store(ds.version().version, Ordering::SeqCst);
    shared.ops.fetch_add(1, Ordering::SeqCst);
    shared.journal.line(format_args!(
        "step={n} soak worker={worker} seq={} op={} outcome={outcome} tries={tries} version={} ms={}",
        drawn.seq,
        drawn.op,
        ds.version().version,
        started.elapsed().as_millis()
    ));
    Ok(drawn.think)
}

/// Runs an operation until it lands. One that keeps losing the race for a commit
/// is tried again on the latest version, and maintenance that still loses is left
/// for later, as a service would; a write must land.
async fn run_op(
    shared: &Shared,
    worker: usize,
    ds: &mut Dataset,
    drawn: &Drawn,
    n: usize,
) -> Result<(&'static str, usize), Failure> {
    for tries in 1..=TRIES {
        let error = match Box::pin(attempt(shared, worker, ds, drawn, n)).await {
            Ok(()) => return Ok(("ok", tries)),
            Err(OpError::Wrong(failure)) => return Err(failure),
            Err(OpError::Lance(error)) => error,
        };
        if !is_conflict(&error) {
            return Err(fail(
                "error_without_fault",
                n,
                format_args!("worker {worker}: {} failed: {error}", drawn.op),
            ));
        }
        shared.journal.line(format_args!(
            "step={n} observation=soak_retry:{}",
            drawn.op.kind()
        ));
        if tries == TRIES {
            if drawn.op.changes_rows() {
                return Err(fail(
                    "error_without_fault",
                    n,
                    format_args!(
                        "worker {worker}: {} lost the race for a commit {TRIES} times: {error}",
                        drawn.op
                    ),
                ));
            }
            shared.journal.line(format_args!(
                "step={n} observation=soak_gave_up:{}",
                drawn.op.kind()
            ));
            return Ok(("gave_up", tries));
        }
        Box::pin(ds.checkout_latest())
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        shared.handles[worker].store(ds.version().version, Ordering::SeqCst);
    }
    unreachable!("the last try returns")
}

async fn attempt(
    shared: &Shared,
    worker: usize,
    ds: &mut Dataset,
    drawn: &Drawn,
    n: usize,
) -> Result<(), OpError> {
    let stride = shared.stride;
    let source = |first_id: i64, rows: i64| -> Box<dyn RecordBatchReader + Send> {
        let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
        Box::new(source_of(full_batch(
            &shared.schema,
            shared.data_seed,
            &shared.scale,
            &ids,
            drawn.number,
        )))
    };
    let hooks = &shared.hooks;
    match drawn.op {
        SoakOp::Append { first_id, rows } => {
            let params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: shared.scale.rows_per_file,
                ..Default::default()
            };
            Box::pin(ds.append(source(first_id, rows), Some(params))).await?;
        }
        SoakOp::MergeInsert { first_id, rows } => {
            Box::pin(upsert(ds, source(first_id, rows))).await?;
        }
        SoakOp::Update { lo, hi } => {
            let predicate = class_range(stride, worker, lo, hi);
            Box::pin(update(ds, &predicate, &drawn.number.to_string())).await?;
        }
        SoakOp::Delete { lo, hi } => {
            Box::pin(delete(ds, class_range(stride, worker, lo, hi))).await?;
        }
        SoakOp::Scan => {
            Box::pin(check_class(ds, shared, worker, &shared.rows(worker), n)).await?;
        }
        SoakOp::Query(kind) => {
            let rows = shared.rows(worker);
            Box::pin(query_class(ds, shared, worker, kind, &rows, drawn.seq, n)).await?;
        }
        SoakOp::CreateIndex(kind) => {
            let op = Op::CreateIndex(kind);
            op_future(ds, hooks, shared.data_seed, shared.scale, op, None).await?;
            shared.indices.lock().unwrap().insert(kind);
        }
        SoakOp::OptimizeIndices => {
            let op = Op::OptimizeIndices;
            op_future(ds, hooks, shared.data_seed, shared.scale, op, None).await?;
        }
        SoakOp::Compact => {
            op_future(ds, hooks, shared.data_seed, shared.scale, Op::Compact, None).await?;
        }
        SoakOp::Cleanup { retain } => {
            let _turn = shared.cleanup.lock().await;
            let mut policy = CleanupPolicyBuilder::default()
                .retain_n_versions(ds, retain)
                .await?
                .error_if_tagged_old_versions(false)
                .build();
            let boundary = policy
                .before_version
                .unwrap_or(0)
                .min(shared.oldest_handle());
            policy.before_version = Some(boundary);
            // Noted before it starts: one that fails half way may have removed some.
            shared.expired_below.fetch_max(boundary, Ordering::SeqCst);
            let removed = Box::pin(ds.cleanup_with_policy(policy)).await?;
            shared.journal.line(format_args!(
                "step={n} soak worker={worker} cleanup before={boundary} old_versions={} data_files={} index_files={} transaction_files={} deletion_files={} failed_deletes={}",
                removed.old_versions,
                removed.data_files_removed,
                removed.index_files_removed,
                removed.transaction_files_removed,
                removed.deletion_files_removed,
                removed.failed_deletes
            ));
            // Nothing is injected, so nothing explains a failed deletion.
            if removed.failed_deletes > 0 {
                return Err(OpError::Wrong(fail(
                    "cleanup",
                    n,
                    format_args!(
                        "worker {worker}: a cleanup failed {} deletions",
                        removed.failed_deletes
                    ),
                )));
            }
        }
    }
    Ok(())
}

/// The rows of a worker's class on its handle must be what its operations left,
/// whatever version the handle is on.
async fn check_class(
    ds: &Dataset,
    shared: &Shared,
    worker: usize,
    rows: &ClassRows,
    n: usize,
) -> Result<(), OpError> {
    let mut scan = ds.scan();
    scan.filter(&format!("id % {} = {worker}", shared.stride))?;
    scan.project(&["id", "gen"])?;
    let batch = scan.try_into_batch().await?;
    let found = stamps_of(&batch);
    if &found != rows || found.len() != batch.num_rows() {
        return Err(OpError::Wrong(fail(
            "content",
            n,
            format_args!(
                "worker {worker} reads {} rows of its class on version {}: {}",
                batch.num_rows(),
                ds.version().version,
                describe_stamps(rows, &found)
            ),
        )));
    }
    Ok(())
}

/// An index query on a worker's handle. Only the rows of its class in the answer
/// can be told, and they must be what its operations left.
async fn query_class(
    ds: &Dataset,
    shared: &Shared,
    worker: usize,
    kind: IndexKind,
    rows: &ClassRows,
    salt: u64,
    n: usize,
) -> Result<(), OpError> {
    let stride = shared.stride;
    let class: BTreeSet<i64> = rows.keys().copied().collect();
    let mut scan = ds.scan();
    let (what, expected): (String, BTreeSet<i64>) = match kind {
        IndexKind::BTree => {
            let value = (salt % CATEGORIES as u64) as i32;
            let filter = format!("category = {value}");
            scan.filter(&filter)?;
            let expected = class
                .iter()
                .copied()
                .filter(|id| category(*id) == value)
                .collect();
            (filter, expected)
        }
        IndexKind::BTreeId => {
            let (first, last) = (class.first().copied(), class.last().copied());
            let (first, last) = (first.unwrap_or(0), last.unwrap_or(0));
            let third = salt as i64 % 3;
            let span = last - first + 1;
            let lo = first + span * third / 3;
            let hi = first + span * (third + 1) / 3;
            let filter = format!("id >= {lo} AND id < {hi}");
            scan.filter(&filter)?;
            (filter, class.range(lo..hi).copied().collect())
        }
        IndexKind::Inverted => {
            let token = salt % shared.scale.vocabulary;
            let term = format!("tok{token:04}");
            let query = FullTextSearchQuery::new(term.clone())
                .with_column("text".to_string())?
                .limit(None);
            scan.full_text_search(query)?;
            let expected = class
                .iter()
                .copied()
                .filter(|id| tokens(*id, &shared.scale).any(|found| found == token))
                .collect();
            (format!("a search for {term}"), expected)
        }
        IndexKind::IvfFlat => {
            // The neighbors among the class alone: the filter goes first.
            let query = vector(!shared.data_seed, salt as i64);
            scan.nearest("vec", &Float32Array::from(query.to_vec()), NEIGHBORS)?;
            scan.minimum_nprobes(IVF_PARTITIONS);
            scan.prefilter(true);
            scan.filter(&format!("id % {stride} = {worker}"))?;
            scan.project(&["id"])?;
            let batch = scan.try_into_batch().await?;
            let answers = Answers {
                live: &class,
                next_id: 0,
                stamps: None,
                stamps_ever: None,
                few_stamps: false,
                scale: shared.scale,
                data_seed: shared.data_seed,
                check_plan: false,
                salt: 0,
            };
            let label = format!("worker {worker}: random query {salt}");
            return check_neighbors(n, &label, &query, &int64_column(&batch, "id"), &answers)
                .map_err(OpError::Wrong);
        }
        IndexKind::BTreeStamp => unreachable!("a soak builds no index on gen"),
    };
    scan.project(&["id"])?;
    let batch = scan.try_into_batch().await?;
    let own: Vec<i64> = int64_column(&batch, "id")
        .into_iter()
        .filter(|id| id % stride == worker as i64)
        .collect();
    compare_ids(n, &format!("worker {worker}: {what}"), &own, &expected).map_err(OpError::Wrong)
}

fn stamps_of(batch: &RecordBatch) -> ClassRows {
    int64_column(batch, "id")
        .into_iter()
        .zip(int64_column(batch, "gen"))
        .collect()
}

/// Where two sets of rows differ, the first five of each kind.
fn describe_stamps(expected: &ClassRows, found: &ClassRows) -> String {
    let wrong: Vec<String> = expected
        .iter()
        .filter(|(id, stamp)| found.get(id) != Some(stamp))
        .take(5)
        .map(|(id, stamp)| format!("{id} holds {:?} for {stamp}", found.get(id)))
        .collect();
    let unexpected: Vec<i64> = found
        .keys()
        .filter(|id| !expected.contains_key(id))
        .take(5)
        .copied()
        .collect();
    format!(
        "{} distinct ids where {} are expected; wrong or missing {wrong:?}, unexpected {unexpected:?}",
        found.len(),
        expected.len()
    )
}

/// What one sample measured.
struct Sample {
    t_ms: u128,
    ops: u64,
    latest: u64,
    /// How long the runtime took to go quiet.
    quiet_ms: u128,
    /// In the order of `SERIES`.
    series: [i64; SERIES.len()],
    index_cache: CacheStats,
    metadata_cache: CacheStats,
    /// What the workers asked of the store since the sample before: reads, bytes
    /// read, writes, bytes written.
    io: [u64; 4],
    /// The anonymous part of the resident set, for whoever reads the samples. No
    /// verdict reads it: glibc keeps the top of every thread's arena, so it wanders
    /// by tens of MiB with nothing held.
    rss_anon_kib: u64,
}

impl Sample {
    fn header() -> String {
        let series: Vec<&str> = SERIES.iter().map(|(name, _)| *name).collect();
        let mut header = format!(
            "sample,t_ms,ops,latest,quiet_ms,{},rss_anon_kib",
            series.join(",")
        );
        for cache in ["index_cache", "metadata_cache"] {
            for stat in ["bytes", "entries", "hits", "misses"] {
                write!(header, ",{cache}_{stat}").unwrap();
            }
        }
        header.push_str(",read_iops,read_bytes,write_iops,written_bytes");
        header
    }

    fn row(&self, n: usize) -> String {
        let mut row = format!(
            "{n},{},{},{},{}",
            self.t_ms, self.ops, self.latest, self.quiet_ms
        );
        for value in self.series {
            write!(row, ",{value}").unwrap();
        }
        write!(row, ",{}", self.rss_anon_kib).unwrap();
        for stats in [&self.index_cache, &self.metadata_cache] {
            write!(
                row,
                ",{},{},{},{}",
                stats.size_bytes, stats.num_entries, stats.hits, stats.misses
            )
            .unwrap();
        }
        for count in self.io {
            write!(row, ",{count}").unwrap();
        }
        row
    }
}

/// What a worker's task ended with.
fn worker_end(ended: Result<Result<(), Failure>, JoinError>, n: usize) -> Result<(), Failure> {
    ended.unwrap_or_else(|error| Err(fail("panic", n, format_args!("a worker's task: {error}"))))
}

impl Run {
    pub(super) async fn run_soak(&mut self) -> Result<(), Failure> {
        let plan = self.plan.soak.clone().expect("a soak plan");
        let started = Instant::now();
        let session = Arc::new(Session::new(
            INDEX_CACHE_BYTES,
            METADATA_CACHE_BYTES,
            Default::default(),
        ));
        let mut handles = Vec::with_capacity(plan.workers);
        for _ in 0..plan.workers {
            let ds = Box::pin(
                DatasetBuilder::from_uri(&self.uri)
                    .with_session(session.clone())
                    .load(),
            )
            .await
            .map_err(|error| fail("unreadable", 1, error))?;
            handles.push(ds);
        }
        let store = handles[0]
            .object_store(None)
            .await
            .map_err(|error| fail("unreadable", 1, error))?;
        let sink = Arc::new(IncidentState::new(
            self.root.clone(),
            Path::new("/dev/null"),
            started,
        ));
        let shared = Arc::new(Shared {
            session,
            schema: schema(false),
            data_seed: self.plan.data_seed,
            scale: self.plan.scale,
            stride: plan.workers as i64,
            pause: RwLock::new(()),
            stop: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            cleanup: TurnLock::new(()),
            sources: (0..plan.workers)
                .map(|worker| Mutex::new(plan.source(worker)))
                .collect(),
            handles: handles
                .iter()
                .map(|ds| AtomicU64::new(ds.version().version))
                .collect(),
            expired_below: AtomicU64::new(self.expired_below),
            indices: Mutex::new(BTreeSet::new()),
            ops: AtomicU64::new(0),
            sample: AtomicUsize::new(1),
            journal: self.journal.clone(),
            hooks: Arc::new(StressHooks::new(sink)),
            store,
        });
        let mut workers = JoinSet::new();
        for (worker, ds) in handles.into_iter().enumerate() {
            workers.spawn(run_worker(shared.clone(), worker, ds));
        }
        let result = Box::pin(self.take_samples(&shared, &plan, &mut workers, started)).await;
        shared.stop.store(true, Ordering::SeqCst);
        let samples = match result {
            Ok(samples) => samples,
            Err(failure) => {
                // The others stop at their next operation, or with the process:
                // aborting them would cancel their scans, which panics.
                workers.detach_all();
                return Err(failure);
            }
        };
        while let Some(ended) = workers.join_next().await {
            worker_end(ended, SAMPLES)?;
        }
        self.check_series(&samples)?;
        self.sync_model(&shared);
        Ok(())
    }

    async fn take_samples(
        &mut self,
        shared: &Shared,
        plan: &SoakPlan,
        workers: &mut JoinSet<Result<(), Failure>>,
        started: Instant,
    ) -> Result<Vec<Sample>, Failure> {
        let path = self.seed_dir().join(SERIES_FILE);
        let harness =
            |error: std::io::Error| fail("harness", 0, format_args!("{SERIES_FILE}: {error}"));
        let mut series = File::create(&path).map_err(harness)?;
        writeln!(series, "{}", Sample::header()).map_err(harness)?;
        let mut samples = Vec::with_capacity(SAMPLES);
        for index in 0..SAMPLES {
            let n = index + 1;
            let due = started + plan.interval() * n as u32;
            tokio::select! {
                _ = tokio::time::sleep_until(due.into()) => {}
                Some(ended) = workers.join_next() => {
                    worker_end(ended, n)?;
                    return Err(fail("harness", n, "a worker stopped before the end of the run"));
                }
            }
            shared.sample.store(n, Ordering::SeqCst);
            let pause = shared.pause.write().await;
            if shared.failed.load(Ordering::SeqCst) {
                // Its task ends right after the pause it let go of.
                if let Some(ended) = workers.join_next().await {
                    worker_end(ended, n)?;
                }
                return Err(fail(
                    "harness",
                    n,
                    "a worker failed without a failure to report",
                ));
            }
            if let Some(ended) = workers.try_join_next() {
                worker_end(ended, n)?;
                return Err(fail(
                    "harness",
                    n,
                    "a worker stopped before the end of the run",
                ));
            }
            let mut sample = self.sample(shared, n, started).await?;
            Box::pin(self.clean_up(shared, n)).await?;
            self.measure_dataset(&mut sample, n).await?;
            writeln!(series, "{}", sample.row(n)).map_err(harness)?;
            self.journal.line(format_args!(
                "step={n} soak sample {}",
                Sample::header()
                    .split(',')
                    .zip(sample.row(n).split(','))
                    .skip(1)
                    .map(|(name, value)| format!("{name}={value}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
            self.check_capacity(&sample, n)?;
            samples.push(sample);
            if n % CHECK_EVERY == 0 {
                self.sync_model(shared);
                Box::pin(self.check_content(n)).await?;
                self.check_stamps(shared, n).await?;
            }
            drop(pause);
        }
        Ok(samples)
    }

    /// Measures the process once the operations that ended have let go of what
    /// they held. The dataset is measured later, after a cleanup.
    async fn sample(&self, shared: &Shared, n: usize, started: Instant) -> Result<Sample, Failure> {
        if self.self_test == Some(SelfTest::SoakGrowth) {
            Box::leak(vec![1u8; SELF_TEST_LEAK].into_boxed_slice());
        }
        let timeout = SAMPLE_QUIET.min(self.quiet);
        let (snapshot, waited) = settle(&self.tmp, SAMPLE_STILL, timeout).await;
        if waited >= timeout {
            // Measured all the same, with what was still starting or ending.
            self.observe(n, "soak_unsettled");
        }
        // The counts alone: under `test-util` the statistics hold every request
        // of the interval, which must be gone before the heap is read.
        let io = {
            let stats = shared.store.io_tracker().incremental_stats();
            [
                stats.read_iops,
                stats.read_bytes,
                stats.write_iops,
                stats.written_bytes,
            ]
        };
        self.ds
            .object_store(None)
            .await
            .map_err(|error| fail("unreadable", n, error))?
            .io_tracker()
            .incremental_stats();
        // Before anything else here takes memory.
        let heap = live_heap_bytes();
        let malloc = malloc_bytes();
        let index_cache = shared.session.index_cache_stats().await;
        let metadata_cache = shared.session.metadata_cache_stats().await;
        let mut series = [0; SERIES.len()];
        series[..5].copy_from_slice(&[
            heap,
            malloc,
            snapshot.fds as i64,
            snapshot.threads as i64,
            snapshot.tasks as i64,
        ]);
        Ok(Sample {
            t_ms: started.elapsed().as_millis(),
            ops: shared.ops.load(Ordering::SeqCst),
            latest: 0,
            quiet_ms: waited.as_millis(),
            series,
            index_cache,
            metadata_cache,
            io,
            rss_anon_kib: trimmed_anon_kib(),
        })
    }

    /// The versions, files and bytes of the dataset, the last three series.
    async fn measure_dataset(&self, sample: &mut Sample, n: usize) -> Result<(), Failure> {
        let verifier = self.open_verifier(n).await?;
        let versions = listed_versions(&verifier, n).await?.len();
        let files = file_sizes(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        sample.latest = verifier.version().version;
        sample.series[5..].copy_from_slice(&[
            versions as i64,
            files.len() as i64,
            files.values().sum(),
        ]);
        Ok(())
    }

    fn check_capacity(&self, sample: &Sample, n: usize) -> Result<(), Failure> {
        let caches = [
            ("index", &sample.index_cache, INDEX_CACHE_BYTES),
            ("metadata", &sample.metadata_cache, METADATA_CACHE_BYTES),
        ];
        for (name, stats, capacity) in caches {
            if stats.size_bytes > capacity {
                return Err(fail(
                    "soak_growth",
                    n,
                    format_args!(
                        "the {name} cache of the session holds {} bytes in {} entries, over its capacity of {capacity}",
                        stats.size_bytes, stats.num_entries
                    ),
                ));
            }
        }
        Ok(())
    }

    /// With every worker waiting, a cleanup that may delete unverified files: what
    /// it leaves must be what the versions it keeps reference.
    async fn clean_up(&mut self, shared: &Shared, n: usize) -> Result<(), Failure> {
        self.state.set_phase(n.to_string());
        self.catch_up(n).await?;
        let boundary = shared.oldest_handle();
        shared.expired_below.fetch_max(boundary, Ordering::SeqCst);
        let mut policy = CleanupPolicyBuilder::default()
            .delete_unverified(true)
            .error_if_tagged_old_versions(false)
            .build();
        policy.before_version = Some(boundary);
        let removed = Box::pin(self.ds.cleanup_with_policy(policy))
            .await
            .map_err(|error| fail("cleanup", n, error))?;
        self.journal.line(format_args!(
            "step={n} soak sample_cleanup before={boundary} old_versions={} data_files={} index_files={} transaction_files={} deletion_files={} failed_deletes={}",
            removed.old_versions,
            removed.data_files_removed,
            removed.index_files_removed,
            removed.transaction_files_removed,
            removed.deletion_files_removed,
            removed.failed_deletes
        ));
        if removed.failed_deletes > 0 {
            return Err(fail(
                "cleanup",
                n,
                format_args!(
                    "a cleanup with nothing in flight failed {} deletions",
                    removed.failed_deletes
                ),
            ));
        }
        self.check_expired_gone(n, boundary, 0).await?;
        self.check_orphans(n).await
    }

    /// The model of the checks that read every row: the rows of all the classes.
    fn sync_model(&mut self, shared: &Shared) {
        let live: BTreeSet<i64> = shared.all_rows().into_keys().collect();
        self.model.next_id = live.last().map_or(0, |last| last + 1);
        self.model.live = live;
        self.model.indices = shared.indices.lock().unwrap().clone();
        self.expired_below = shared.expired_below.load(Ordering::SeqCst);
    }

    /// `gen` of every row: an update or a merge that landed on the wrong rows shows
    /// here.
    async fn check_stamps(&self, shared: &Shared, n: usize) -> Result<(), Failure> {
        let unreadable = |error: Error| fail("unreadable", n, error);
        let expected = shared.all_rows();
        let verifier = self.open_verifier(n).await?;
        let mut scan = verifier.scan();
        scan.project(&["id", "gen"]).map_err(unreadable)?;
        let batch = scan.try_into_batch().await.map_err(unreadable)?;
        let found = stamps_of(&batch);
        if found != expected {
            return Err(fail(
                "content",
                n,
                format_args!(
                    "version {}: {}",
                    verifier.version().version,
                    describe_stamps(&expected, &found)
                ),
            ));
        }
        Ok(())
    }

    /// Each series must have levelled off by the middle third of the run.
    fn check_series(&self, samples: &[Sample]) -> Result<(), Failure> {
        let median = |range: Range<usize>, column: usize| {
            let mut values: Vec<i64> = samples[range]
                .iter()
                .map(|sample| sample.series[column])
                .collect();
            values.sort_unstable();
            values[values.len() / 2]
        };
        let mut grown = Vec::new();
        for (column, (name, tolerance)) in SERIES.iter().enumerate() {
            let (middle, last) = (median(MIDDLE, column), median(LAST, column));
            self.journal.line(format_args!(
                "step={SAMPLES} soak series={name} middle={middle} last={last} tolerance={tolerance}"
            ));
            if last - middle > *tolerance {
                grown.push(format!(
                    "{name} from {middle} to {last}, over its tolerance of {tolerance}"
                ));
            }
        }
        if !grown.is_empty() {
            return Err(fail(
                "soak_growth",
                SAMPLES,
                format_args!(
                    "from the middle third of the run to the last, the median of {}; the samples are in {SERIES_FILE}",
                    grown.join(", ")
                ),
            ));
        }
        Ok(())
    }
}
