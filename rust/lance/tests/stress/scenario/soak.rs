// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The `soak` scenario: a long-lived service on one dataset.
//!
//! Workers keep one handle each, all of them on one `Session` with small caches,
//! and go round the work of a service: appends, merge inserts, updates and
//! deletes, compaction, index optimizations, cleanups that keep a few versions,
//! reopening the dataset, scans and index queries. A row belongs to the class
//! `id % workers`, and a worker only changes its own class, so whatever version its
//! handle is on, its rows are what its own operations left, which its scans and
//! queries check. The operations of a worker follow from the seed and from its rows
//! alone: every run of a seed takes a prefix of the same sequence, while how far it
//! gets and how the workers interleave depend on timing.
//!
//! The state of the dataset lasts as long as the run, as a service's would. The
//! older half of the rows a class starts with is its core, which no operation
//! deletes: updates move its rows to new fragments and merge inserts rewrite their
//! `gen` column in place, so fragments that live for the whole run collect
//! deletions and dead columns for compaction to clear. The other rows of a class
//! turn over: appends and merge inserts add rows above the newest, and deletes take
//! the oldest. The indices are built once, before the workers start, and from then
//! on only optimized. Lance gives a vector index a new segment for new rows unless
//! told to merge, which it leaves to the caller, so some optimizations merge every
//! segment.
//!
//! A worker waits a little between its operations, as a service waits for work:
//! with none, a compaction loses the race for its commit to the writes that land
//! meanwhile most of the time, and every loss leaves the files it wrote behind.
//! Maintenance that keeps losing is left for later, as a service would leave it.
//! The samples count how often each kind was drawn and gave up, but nothing judges
//! the counts: the workers that run maintenance also make the load it races with,
//! so a compaction made 2 s slower loses four tries in five and still lands half
//! of its draws.
//!
//! A run takes at least `MIN_SAMPLES` samples, at even intervals of at most
//! `MAX_SAMPLE_INTERVAL`, so the checks of a sample run in proportion to its
//! length. For each one the workers finish what they are doing and wait,
//! maintenance after the try it is on, and once the runtime is quiet the sample
//! measures the process, the caches of the session and the dataset, so that what
//! an operation holds while it runs does not count.
//! Then a cleanup that may delete unverified files runs, as a service would run one
//! with nothing in flight, and must leave no file that no version references, and
//! the dataset is measured after it, so that its series count what the versions
//! the cleanup keeps hold rather than what lost races left; every `CHECK_EVERY`
//! samples the dataset is checked against the rows of every class too. A cache may
//! never hold more than its capacity, and each series must level off: the median
//! of its last third of the samples may exceed that of the middle third by no more
//! than the tolerance of the series. The first third warms the caches up.
//!
//! A compaction or an index optimization that does not merge only raises the level
//! the series settle at: an update moves rows out of their fragment, and a commit
//! drops a fragment with no rows left and a segment that covers no fragment unless
//! it is the last of its index, so with a bounded number of rows little piles up
//! for long. After the last sample,
//! with nothing in flight, merging the segments of every index and then compacting
//! must leave one fragment, which one segment of each index covers.
//!
//! Lance's own tests build `lance-io` with `test-util`, under which a store keeps a
//! record of every request it served until someone takes its statistics. A sample
//! takes them, from the store the session shares and from the handle of the
//! checks, before it reads the heap: a build without the feature keeps counts
//! alone.

use std::collections::{BTreeMap, BTreeSet};
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
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::session::Session;
use lance_core::cache::CacheStats;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::FullTextSearchQuery;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use tokio::sync::{Mutex as TurnLock, RwLock, RwLockReadGuard};
use tokio::task::{JoinError, JoinSet};

use super::concurrency::{
    ClassRows, class_range, delete, full_batch, id_run, schema, source_of, stamp_job, stamp_source,
    update, upsert,
};
use super::{
    ALL_PARTITIONS, Answers, CATEGORIES, Failure, IndexKind, Journal, NEIGHBORS, Op, PlanRng, Run,
    Scale, category, check_neighbors, compare_ids, fail, file_sizes, fingerprint, int64_column,
    is_commit_conflict, listed_versions, op_future, pick_from, soak_secs, tokens, vector,
};
use crate::harness::SelfTest;
use crate::incidents::{IncidentState, StressHooks};
use crate::leaks::{live_heap_bytes, malloc_bytes, settle, trimmed_anon_kib};

/// The fewest samples a run takes, and the longest it waits between two: a long
/// run takes more samples rather than longer intervals.
pub const MIN_SAMPLES: usize = 24;
const MAX_SAMPLE_INTERVAL: Duration = Duration::from_millis(7500);
const CHECK_EVERY: usize = 4;
/// Takes `MIN_SAMPLES` samples `MAX_SAMPLE_INTERVAL` apart.
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
/// Outside these percentages of the rows it starts with, a class only appends or
/// only deletes until it is back within them, and an operation drawn within them
/// may take it beyond them by the rows of one operation. The older part of those
/// rows, its core, is never deleted.
const CLASS_LOW_PERCENT: usize = 75;
const CLASS_HIGH_PERCENT: usize = 125;
const CORE_PERCENT: usize = 50;

/// What a worker draws while its class is within its bounds.
#[derive(Clone, Copy)]
enum Draw {
    Append,
    MergeInsert,
    MergeInsertColumns,
    Update,
    Delete,
    Scan,
    Query,
    Reopen,
    OptimizeIndices,
    MergeIndices,
    Compact,
    Cleanup,
}

/// How often a worker draws each operation, in hundredths. A quarter of the index
/// optimizations merge every segment, which Lance leaves to the caller.
const DRAWS: [(Draw, u64); 12] = [
    (Draw::Append, 12),
    (Draw::MergeInsert, 6),
    (Draw::MergeInsertColumns, 6),
    (Draw::Update, 14),
    (Draw::Delete, 14),
    (Draw::Scan, 8),
    (Draw::Query, 12),
    (Draw::Reopen, 4),
    (Draw::OptimizeIndices, 6),
    (Draw::MergeIndices, 2),
    (Draw::Compact, 8),
    (Draw::Cleanup, 8),
];

const _: () = {
    let mut total = 0;
    let mut at = 0;
    while at < DRAWS.len() {
        total += DRAWS[at].1;
        at += 1;
    }
    assert!(total == 100, "the weights of DRAWS must add up to 100");
};

/// The operation a roll from 0 to 99 picks.
fn pick_draw(roll: u64) -> Draw {
    let mut left = roll;
    for (draw, weight) in DRAWS {
        if left < weight {
            return draw;
        }
        left -= weight;
    }
    unreachable!("the weights of DRAWS add up to 100")
}

/// What a sample measures, as the columns of `SERIES_FILE` name it, and how far
/// the median of the last third may rise above that of the middle third.
const SERIES: [(&str, i64); 11] = [
    // What Rust code holds, the series that matters.
    ("heap_bytes", 2 << 20),
    // What malloc has handed out: the heap, and what C code holds besides it. It
    // moves by a third more than the heap, for chunk headers and the caches of each
    // thread, which a tolerance of the heap's would make the tighter of the two.
    ("malloc_bytes", 3 << 20),
    ("fds", 4),
    ("threads", 8),
    // Readers that the caches of the session hold keep tasks of their own: in a
    // run of 180 s the median of a third moves by up to 8 either way on its own.
    ("tasks", 16),
    // What the versions a cleanup keeps hold, measured after it.
    ("versions", 16),
    ("files", 100),
    ("dataset_bytes", 2 << 20),
    // The latest version, after the cleanup too: what maintenance keeps in check.
    ("fragments", 24),
    ("index_segments", 8),
    // Its size swings by up to 8 KB within a run on its own.
    ("manifest_bytes", 16 << 10),
];

/// The maintenance whose draws and give-ups the samples count. A cleanup commits
/// no version, so it has no race to lose.
const MAINTENANCE: [SoakOp; 3] = [
    SoakOp::Compact,
    SoakOp::OptimizeIndices,
    SoakOp::MergeIndices,
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
    /// Sets `gen` on a run of core rows with a merge insert that rewrites the
    /// column in place, so the rows stay in their fragments.
    MergeInsertColumns {
        first_id: i64,
        rows: i64,
    },
    /// Sets `gen` on the rows of the class in a range of ids.
    Update {
        lo: i64,
        hi: i64,
    },
    /// Deletes the rows of the class in a range of ids, the oldest ones past its
    /// core.
    Delete {
        lo: i64,
        hi: i64,
    },
    /// Reads the class.
    Scan,
    /// Reads rows of the class through an index.
    Query(IndexKind),
    /// Drops the handle and opens the dataset again on the same session.
    Reopen,
    /// Indexes the rows no index covers.
    OptimizeIndices,
    /// Indexes the rows no index covers and merges the segments of each index
    /// into one.
    MergeIndices,
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
            Self::MergeInsertColumns { .. } => "merge_insert_columns",
            Self::Update { .. } => "update",
            Self::Delete { .. } => "delete",
            Self::Scan => "scan",
            Self::Query(_) => "query",
            Self::Reopen => "reopen",
            Self::OptimizeIndices => "optimize_indices",
            Self::MergeIndices => "merge_indices",
            Self::Compact => "compact",
            Self::Cleanup { .. } => "cleanup",
        }
    }

    fn changes_rows(&self) -> bool {
        matches!(
            self,
            Self::Append { .. }
                | Self::MergeInsert { .. }
                | Self::MergeInsertColumns { .. }
                | Self::Update { .. }
                | Self::Delete { .. }
        )
    }
}

impl fmt::Display for SoakOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Append { first_id, rows }
            | Self::MergeInsert { first_id, rows }
            | Self::MergeInsertColumns { first_id, rows } => {
                write!(f, "{}:first={first_id},rows={rows}", self.kind())
            }
            Self::Update { lo, hi } | Self::Delete { lo, hi } => {
                write!(f, "{}:ids={lo}..{hi}", self.kind())
            }
            Self::Query(kind) => write!(f, "{}:{}", self.kind(), kind.name()),
            Self::Cleanup { retain } => write!(f, "cleanup:retain={retain}"),
            Self::Scan
            | Self::Reopen
            | Self::OptimizeIndices
            | Self::MergeIndices
            | Self::Compact => f.write_str(self.kind()),
        }
    }
}

/// The operations of one worker, drawn as it goes: each depends on the rows the
/// earlier ones left in its class.
struct OpSource {
    rng: PlanRng,
    worker: usize,
    stride: i64,
    /// Its class: each id with the `gen` it holds. The oldest `core` ids are its
    /// core, which no operation deletes; deletes take the oldest ids past it and
    /// writes add new ones above the newest, so the ids form two runs.
    rows: ClassRows,
    core: usize,
    next_id: i64,
    /// Below the one size the class only grows and above the other it only
    /// shrinks; an operation drawn between them may take it beyond them by its
    /// rows.
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
            core: size * CORE_PERCENT / 100,
            next_id,
            low: size * CLASS_LOW_PERCENT / 100,
            high: size * CLASS_HIGH_PERCENT / 100,
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
        let run = self.rng.range(10..=40) as usize;
        let roll = self.rng.range(0..=99);
        let size = self.rows.len();
        // Within its bounds a class holds at least a quarter of the rows it starts
        // with past its core, which may be fewer than a run.
        let past_core = size - self.core;
        let delete_oldest = |source: &Self| {
            let count = run.min(past_core);
            SoakOp::Delete {
                lo: source.id_at(source.core),
                hi: source.id_at(source.core + count - 1) + 1,
            }
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
            match pick_draw(roll) {
                Draw::Append => append(self),
                Draw::MergeInsert => SoakOp::MergeInsert {
                    first_id: self.id_at(size - run / 2),
                    rows: run as i64,
                },
                Draw::MergeInsertColumns => {
                    let rows = run.min(self.core);
                    let start = self.rng.range(0..=(self.core - rows) as u64) as usize;
                    SoakOp::MergeInsertColumns {
                        first_id: self.id_at(start),
                        rows: rows as i64,
                    }
                }
                Draw::Update => {
                    let start = self.rng.range(0..=(size - run) as u64) as usize;
                    SoakOp::Update {
                        lo: self.id_at(start),
                        hi: self.id_at(start + run - 1) + 1,
                    }
                }
                Draw::Delete => delete_oldest(self),
                Draw::Scan => SoakOp::Scan,
                Draw::Query => SoakOp::Query(*pick_from(&mut self.rng, &INDICES)),
                Draw::Reopen => SoakOp::Reopen,
                Draw::OptimizeIndices => SoakOp::OptimizeIndices,
                Draw::MergeIndices => SoakOp::MergeIndices,
                Draw::Compact => SoakOp::Compact,
                Draw::Cleanup => SoakOp::Cleanup {
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
            SoakOp::MergeInsertColumns { first_id, rows } => {
                for id in id_run(first_id, rows, self.stride) {
                    *self.rows.get_mut(&id).expect("a row of the core") = number;
                }
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
    samples: usize,
    workers: usize,
    pub(super) stable_row_ids: bool,
    initial_rows: i64,
    /// Where the operations of each worker start from.
    worker_seeds: Vec<u64>,
}

pub(super) fn plan(rng: &mut PlanRng, initial_rows: i64) -> SoakPlan {
    let workers = rng.range(2..=4) as usize;
    let secs = soak_secs();
    let samples = (secs * 1000)
        .div_ceil(MAX_SAMPLE_INTERVAL.as_millis() as u64)
        .max(MIN_SAMPLES as u64) as usize;
    SoakPlan {
        secs,
        samples,
        workers,
        stable_row_ids: rng.chance(0.5),
        initial_rows,
        worker_seeds: (0..workers).map(|_| rng.next()).collect(),
    }
}

impl SoakPlan {
    fn interval(&self) -> Duration {
        Duration::from_secs(self.secs) / self.samples as u32
    }

    pub(super) fn samples(&self) -> usize {
        self.samples
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
            "soak secs={} samples={} interval_ms={} workers={} stable_row_ids={} index_cache_bytes={INDEX_CACHE_BYTES} metadata_cache_bytes={METADATA_CACHE_BYTES}\n",
            self.secs,
            self.samples,
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
    /// Where a worker opens the dataset again.
    uri: String,
    schema: SchemaRef,
    data_seed: u64,
    scale: Scale,
    stride: i64,
    /// An operation runs under the read lock, maintenance one try at a time. A
    /// sample takes the write lock, which waits for what holds it and holds back
    /// the rest.
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
    ops: AtomicU64,
    /// For each kind of `MAINTENANCE`, how many times it was drawn and how many of
    /// those gave up.
    maintenance: [[AtomicU64; 2]; MAINTENANCE.len()],
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
        let mut running = Some(shared.pause.read().await);
        if shared.stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        // Dropped before `running` on an error or a panic: the task ends only after
        // it lets go of the pause, and a sample that takes the pause meanwhile must
        // not check the rows the failed operation was to leave.
        let failing = MarkFailed(&shared.failed);
        let think = Box::pin(run_next(&shared, worker, &mut ds, &mut running)).await?;
        std::mem::forget(failing);
        // Outside the lock, so that a sample need not wait for it.
        drop(running);
        tokio::time::sleep(think).await;
    }
}

/// Draws the next operation of a worker and runs it under `running`, its hold on
/// the pause; how long the worker waits after it.
async fn run_next<'s>(
    shared: &'s Shared,
    worker: usize,
    ds: &mut Dataset,
    running: &mut Option<RwLockReadGuard<'s, ()>>,
) -> Result<Duration, Failure> {
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
    let (outcome, tries) = Box::pin(run_op(shared, worker, ds, &drawn, n, running)).await?;
    shared.handles[worker].store(ds.version().version, Ordering::SeqCst);
    shared.ops.fetch_add(1, Ordering::SeqCst);
    if let Some(slot) = MAINTENANCE
        .iter()
        .position(|op| op.kind() == drawn.op.kind())
    {
        let [draws, gave_up] = &shared.maintenance[slot];
        draws.fetch_add(1, Ordering::SeqCst);
        if outcome == "gave_up" {
            gave_up.fetch_add(1, Ordering::SeqCst);
        }
    }
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
async fn run_op<'s>(
    shared: &'s Shared,
    worker: usize,
    ds: &mut Dataset,
    drawn: &Drawn,
    mut n: usize,
    running: &mut Option<RwLockReadGuard<'s, ()>>,
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
        // Maintenance changes no row, so a sample may run between its tries: one
        // that waited for them all would stop the other workers and hand the next
        // try a store with nothing in flight.
        if !drawn.op.changes_rows() {
            running.take();
            *running = Some(shared.pause.read().await);
            // That sample may have been the last one, or have failed the run.
            if shared.stop.load(Ordering::SeqCst) {
                return Ok(("stopped", tries));
            }
            n = shared.sample.load(Ordering::SeqCst);
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
        SoakOp::MergeInsertColumns { first_id, rows } => {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            let columns = stamp_source(&shared.schema, ids, drawn.number);
            let (merged, _) = Box::pin(stamp_job(ds)?.execute_reader(columns)).await?;
            *ds = merged.as_ref().clone();
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
        SoakOp::Reopen => {
            *ds = Box::pin(
                DatasetBuilder::from_uri(&shared.uri)
                    .with_session(shared.session.clone())
                    .load(),
            )
            .await?;
        }
        SoakOp::OptimizeIndices => {
            let op = Op::OptimizeIndices;
            op_future(ds, hooks, shared.data_seed, shared.scale, op, None).await?;
        }
        SoakOp::MergeIndices => {
            Box::pin(merge_indices(ds, hooks)).await?;
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

/// Indexes the rows no index covers and merges the segments of each index into
/// one: Lance merges the segments of a vector index only when told to.
async fn merge_indices(ds: &mut Dataset, hooks: &Arc<StressHooks>) -> lance::Result<()> {
    let segments = ds.load_indices().await?.len();
    let options = OptimizeOptions::merge(segments.max(1)).progress(hooks.clone());
    Box::pin(ds.optimize_indices(&options)).await
}

/// How many segments each index of a soak has.
async fn index_segments(ds: &Dataset) -> lance::Result<BTreeMap<&'static str, usize>> {
    let indices = ds.load_indices().await?;
    Ok(INDICES
        .iter()
        .map(|kind| {
            let name = kind.name();
            let segments = indices.iter().filter(|index| index.name == name).count();
            (name, segments)
        })
        .collect())
}

/// The most partitions a segment of the vector index has. Lance keeps the last
/// segment of an index once it covers no fragment, with no statistics, which count
/// as none. Read segment by segment, which writes nothing: the statistics of a
/// whole index commit a version when its segments overlap.
async fn ivf_partitions(ds: &Dataset, n: usize) -> Result<u64, Failure> {
    let name = IndexKind::IvfFlat.name();
    let unreadable = |error: Error| fail("unreadable", n, error);
    let segments: Vec<_> = ds
        .load_indices()
        .await
        .map_err(unreadable)?
        .iter()
        .filter(|index| index.name == name)
        .map(|index| index.uuid)
        .collect();
    let statistics = ds
        .index_segment_statistics(name, &segments)
        .await
        .map_err(unreadable)?;
    let mut most = 0;
    for segment in &statistics {
        let uncovered = segment
            .statistics
            .as_object()
            .is_some_and(|fields| fields.is_empty());
        let partitions = match segment.statistics.get("num_partitions") {
            Some(count) => count.as_u64(),
            None if uncovered => Some(0),
            None => None,
        };
        most = most.max(partitions.ok_or_else(|| {
            fail(
                "harness",
                n,
                format_args!(
                    "no partition count in the statistics of segment {} of {name}: {}",
                    segment.index_uuid, segment.statistics
                ),
            )
        })?);
    }
    Ok(most)
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
            scan.minimum_nprobes(ALL_PARTITIONS);
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
    /// The most partitions a segment of the vector index has.
    ivf_partitions: u64,
    /// What `Shared::maintenance` counted up to the sample.
    maintenance: [[u64; 2]; MAINTENANCE.len()],
}

impl Sample {
    fn header() -> String {
        let series: Vec<&str> = SERIES.iter().map(|(name, _)| *name).collect();
        let mut header = format!(
            "sample,t_ms,ops,latest,quiet_ms,{},rss_anon_kib,ivf_partitions",
            series.join(",")
        );
        for cache in ["index_cache", "metadata_cache"] {
            for stat in ["bytes", "entries", "hits", "misses"] {
                write!(header, ",{cache}_{stat}").unwrap();
            }
        }
        header.push_str(",read_iops,read_bytes,write_iops,written_bytes");
        for op in MAINTENANCE {
            let kind = op.kind();
            write!(header, ",{kind}_draws,{kind}_gave_up").unwrap();
        }
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
        write!(row, ",{},{}", self.rss_anon_kib, self.ivf_partitions).unwrap();
        for stats in [&self.index_cache, &self.metadata_cache] {
            write!(
                row,
                ",{},{},{},{}",
                stats.size_bytes, stats.num_entries, stats.hits, stats.misses
            )
            .unwrap();
        }
        for count in self
            .io
            .into_iter()
            .chain(self.maintenance.into_iter().flatten())
        {
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
        // Without caches: under the default session, the catch-up before every
        // sample would keep the manifest of each version it lands on, up to 1 GiB.
        self.ds = Box::pin(
            DatasetBuilder::from_uri(&self.uri)
                .with_session(Arc::new(Session::new(0, 0, Default::default())))
                .with_store_params(ObjectStoreParams {
                    object_store_wrapper: Some(self.wrapper.clone()),
                    ..Default::default()
                })
                .load(),
        )
        .await
        .map_err(|error| fail("unreadable", 1, error))?;
        let started = Instant::now();
        let session = Arc::new(Session::new(
            INDEX_CACHE_BYTES,
            METADATA_CACHE_BYTES,
            Default::default(),
        ));
        let sink = Arc::new(IncidentState::new(
            self.root.clone(),
            Path::new("/dev/null"),
            started,
        ));
        let hooks = Arc::new(StressHooks::new(sink));
        let mut handles = Vec::with_capacity(plan.workers);
        for worker in 0..plan.workers {
            let mut ds = Box::pin(
                DatasetBuilder::from_uri(&self.uri)
                    .with_session(session.clone())
                    .load(),
            )
            .await
            .map_err(|error| fail("unreadable", 1, error))?;
            // Built once: from then on a service only optimizes its indices.
            if worker == 0 {
                for kind in INDICES {
                    let (data_seed, scale) = (self.plan.data_seed, self.plan.scale);
                    let op = Op::CreateIndex(kind);
                    op_future(&mut ds, &hooks, data_seed, scale, op, None)
                        .await
                        .map_err(|error| {
                            let index = kind.name();
                            fail(
                                "error_without_fault",
                                1,
                                format_args!("building {index}: {error}"),
                            )
                        })?;
                }
            }
            handles.push(ds);
        }
        self.model.indices = INDICES.into_iter().collect();
        let store = handles[0]
            .object_store(None)
            .await
            .map_err(|error| fail("unreadable", 1, error))?;
        let shared = Arc::new(Shared {
            session,
            uri: self.uri.clone(),
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
            ops: AtomicU64::new(0),
            maintenance: Default::default(),
            sample: AtomicUsize::new(1),
            journal: self.journal.clone(),
            hooks,
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
            if let Err(failure) = worker_end(ended, plan.samples) {
                workers.detach_all();
                return Err(failure);
            }
        }
        let left = Box::pin(self.check_maintenance(&shared, plan.samples)).await?;
        self.check_series(&samples, left)?;
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
        let mut samples = Vec::with_capacity(plan.samples);
        for index in 0..plan.samples {
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
            ivf_partitions: 0,
            maintenance: shared.maintenance.each_ref().map(|[draws, gave_up]| {
                [draws.load(Ordering::SeqCst), gave_up.load(Ordering::SeqCst)]
            }),
        })
    }

    /// The versions, files and bytes of the dataset, and the fragments, index
    /// segments and manifest size of its latest version: the series after the
    /// five of the process. Besides, the most partitions a segment of its vector
    /// index has, which must stay within what a query probes.
    async fn measure_dataset(&self, sample: &mut Sample, n: usize) -> Result<(), Failure> {
        let verifier = self.open_verifier(n).await?;
        let versions = listed_versions(&verifier, n).await?.len();
        let files = file_sizes(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let segments = index_segments(&verifier)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        // The listing names a file from the root of the dataset.
        let manifest = &verifier.manifest_location().path;
        let manifest_file = format!("_versions/{}", manifest.filename().unwrap_or_default());
        let manifest_bytes = *files.get(&manifest_file).ok_or_else(|| {
            fail(
                "harness",
                n,
                format_args!("the dataset lists no {manifest_file} for its manifest {manifest}"),
            )
        })?;
        sample.ivf_partitions = ivf_partitions(&verifier, n).await?;
        if sample.ivf_partitions > ALL_PARTITIONS as u64 {
            return Err(fail(
                "harness",
                n,
                format_args!(
                    "a segment of {} has {} partitions, more than the {ALL_PARTITIONS} a query probes",
                    IndexKind::IvfFlat.name(),
                    sample.ivf_partitions
                ),
            ));
        }
        sample.latest = verifier.version().version;
        sample.series[5..].copy_from_slice(&[
            versions as i64,
            files.len() as i64,
            files.values().sum(),
            verifier.get_fragments().len() as i64,
            segments.values().sum::<usize>() as i64,
            manifest_bytes,
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

    /// Each series must have levelled off by the middle third of the run, and the
    /// maintenance at its end must have left what `check_maintenance` expects;
    /// `left` is what it left otherwise.
    fn check_series(&self, samples: &[Sample], left: Option<String>) -> Result<(), Failure> {
        let n = samples.len();
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
            let middle = median(n / 3..2 * n / 3, column);
            let last = median(2 * n / 3..n, column);
            self.journal.line(format_args!(
                "step={n} soak series={name} middle={middle} last={last} tolerance={tolerance}"
            ));
            if last - middle > *tolerance {
                grown.push(format!(
                    "{name} from {middle} to {last}, over its tolerance of {tolerance}"
                ));
            }
        }
        // What was drawn in the last third: the counts of the sample before it end
        // where those of the last third begin.
        let (before, end) = (&samples[2 * n / 3 - 1], &samples[n - 1]);
        for (slot, op) in MAINTENANCE.iter().enumerate() {
            let [draws_before, gave_up_before] = before.maintenance[slot];
            let [draws, gave_up] = end.maintenance[slot];
            self.journal.line(format_args!(
                "step={n} soak last_third={} draws={} gave_up={}",
                op.kind(),
                draws - draws_before,
                gave_up - gave_up_before
            ));
        }
        let mut reasons = Vec::new();
        if !grown.is_empty() {
            reasons.push(format!(
                "from the middle third of the run to the last, the median of {}; the samples are in {SERIES_FILE}",
                grown.join(", ")
            ));
        }
        reasons.extend(left);
        if !reasons.is_empty() {
            return Err(fail("soak_growth", n, reasons.join("; ")));
        }
        Ok(())
    }

    /// With nothing in flight, merging the segments of every index and then
    /// compacting must leave one fragment, which the one segment of each index
    /// covers: every fragment holds far fewer rows than a compaction puts in one,
    /// and once the indices are merged every fragment has the same coverage, so
    /// nothing keeps two apart. What it left otherwise; either way the rows and
    /// the indices must read right afterwards.
    ///
    /// It runs on a handle of its own on the session of the workers: an index in
    /// the cache of a session keeps a task of its reader until the session goes,
    /// which that of the checks' handle never does before the run ends.
    async fn check_maintenance(
        &mut self,
        shared: &Shared,
        n: usize,
    ) -> Result<Option<String>, Failure> {
        let failed = |error: Error| fail("error_without_fault", n, error);
        let mut ds = Box::pin(
            DatasetBuilder::from_uri(&self.uri)
                .with_session(shared.session.clone())
                .load(),
        )
        .await
        .map_err(|error| fail("unreadable", n, error))?;
        let fragments_before = ds.get_fragments().len();
        let segments_before = index_segments(&ds).await.map_err(failed)?;
        Box::pin(merge_indices(&mut ds, &shared.hooks))
            .await
            .map_err(failed)?;
        Box::pin(compact_files(&mut ds, CompactionOptions::default(), None))
            .await
            .map_err(failed)?;
        let fragments: Vec<u32> = ds
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect();
        let indices = ds.load_indices().await.map_err(failed)?;
        let segments = index_segments(&ds).await.map_err(failed)?;
        drop(ds);
        self.journal.line(format_args!(
            "step={n} soak maintenance fragments={fragments_before}->{} segments={segments_before:?}->{segments:?}",
            fragments.len()
        ));
        let mut left = Vec::new();
        if fragments.len() != 1 {
            left.push(format!("{} fragments", fragments.len()));
        }
        for (name, count) in &segments {
            let covering = indices.iter().filter(|index| {
                index.name == *name
                    && index
                        .fragment_bitmap
                        .as_ref()
                        .is_some_and(|bitmap| fragments.iter().all(|id| bitmap.contains(*id)))
            });
            if *count != 1 {
                left.push(format!("{count} segments of {name}"));
            } else if covering.count() != 1 {
                left.push(format!(
                    "a segment of {name} that does not cover every fragment"
                ));
            }
        }
        self.sync_model(shared);
        Box::pin(self.check_content(n)).await?;
        self.check_stamps(shared, n).await?;
        Ok((!left.is_empty()).then(|| {
            format!(
                "with nothing in flight, merging the index segments and compacting left {}",
                left.join(", ")
            )
        }))
    }
}
