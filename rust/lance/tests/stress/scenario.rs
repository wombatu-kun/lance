// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The child side: a seed's plan, its operations, and the checks after each.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::ops::{Range, RangeInclusive};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, TryStreamExt};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use lance::dataset::{MergeInsertBuilder, WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance::index::{DatasetIndexExt, IndexBuildProgress};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, FullTextSearchQuery, ScalarIndexParams};
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, PutOptions, PutPayload};
use roaring::RoaringBitmap;
use tokio::sync::Notify;

use crate::harness::{
    CALLS_LOG, ENV_CALIBRATE, ENV_CHILD, ENV_DIR, ENV_GENERATION, ENV_PARENT_PID, ENV_PLAN,
    ENV_QUIET_SECS, ENV_SEED, ENV_SELF_TEST, ENV_SOAK_SECS, ENV_STANDALONE, ENV_STEPS, MEMORY_LOG,
    PLAN_LOG, READS_LOG, STEPS_LOG, ScenarioKind, SelfTest, field,
};
use crate::incidents::{
    Area, CRASH_FIRED, CancelWhen, Death, Delay, HangPoint, Incident, IncidentState, Method,
    StressHooks, StressWrapper, kill_this_process,
};
use crate::leaks::{
    Snapshot, TmpWatch, files_below, heap_peak_bytes, live_heap_bytes, quiesce, start_heap_peak,
};
use crate::limits::{OwnCgroup, log_memory_events};
use crate::select::Selection;

mod concurrency;
mod soak;

pub use soak::{
    SELF_TEST_LEAK as SOAK_SELF_TEST_LEAK, SELF_TEST_SECS as SOAK_SELF_TEST_SECS,
    SERIES_FILE as SOAK_CSV,
};

const POOL_VARIABLE: &str = "LANCE_MEM_POOL_SIZE";
/// One cycle of `heap_growth`, repeated until a leak has had time to show. The
/// first cycles fill global caches and pools: the heap grows by about 7 KiB per
/// cycle, then stops by the eighth and stays within 1 KiB.
const HEAP_CYCLE_OPS: usize = 9;
const HEAP_CYCLES: usize = 16;
const HEAP_EARLY_CYCLES: Range<usize> = 8..11;
const HEAP_LATE_CYCLES: Range<usize> = 13..16;
/// The heap of a healthy run varies by about 1 KiB between these cycles.
const HEAP_GROWTH_LIMIT: i64 = 16 << 10;
const VECTOR_DIM: i32 = 32;
const IVF_PARTITIONS: usize = 2;
const NEIGHBORS: usize = 10;
const CATEGORIES: i32 = 7;
/// The rows of one generated batch: the reader builds them only when asked, so
/// the source of a write holds almost no memory of its own.
const CHUNK_ROWS: i64 = 1024;

/// How much data a scenario works on.
#[derive(Clone, Copy, Debug)]
struct Scale {
    rows_per_file: usize,
    initial_rows: (u64, u64),
    added_rows: (u64, u64),
    min_live_rows: usize,
    doc_tokens: u64,
    vocabulary: u64,
}

const SMALL: Scale = Scale {
    rows_per_file: 200,
    initial_rows: (600, 1200),
    added_rows: (100, 400),
    min_live_rows: 300,
    doc_tokens: 3,
    vocabulary: 16,
};

/// Enough data for what a build retains to show, in cycles short enough to repeat.
const MEDIUM: Scale = Scale {
    rows_per_file: 500,
    initial_rows: (5000, 8000),
    added_rows: (500, 1000),
    min_live_rows: 1000,
    doc_tokens: 8,
    vocabulary: 256,
};

/// Enough data for the memory of an operation to stand out from the process.
const LARGE: Scale = Scale {
    rows_per_file: 2000,
    initial_rows: (20_000, 40_000),
    added_rows: (2000, 8000),
    min_live_rows: 10_000,
    doc_tokens: 16,
    vocabulary: 4096,
};

/// A calibration run plans no limit, and reports what each operation uses.
fn calibrating() -> bool {
    std::env::var(ENV_CALIBRATE).is_ok_and(|value| value == "1")
}

/// Runs the seed this process was started for, if any, and exits non-zero
/// when it fails. The verdict is the last line of the step log.
pub fn run_child_if_requested() {
    let Ok(kind) = std::env::var(ENV_CHILD) else {
        return;
    };
    let standalone = std::env::var(ENV_STANDALONE).is_ok_and(|value| value == "1");
    if !standalone && !tie_to_parent() {
        return;
    }
    let started = Instant::now();
    let kind =
        ScenarioKind::from_name(&kind).unwrap_or_else(|| panic!("unknown scenario {kind:?}"));
    let seed: u64 = env(ENV_SEED);
    let dir = PathBuf::from(env::<String>(ENV_DIR));
    let steps: usize = env(ENV_STEPS);
    let quiet = Duration::from_secs(env(ENV_QUIET_SECS));
    let generation: u64 = env(ENV_GENERATION);
    let self_test = std::env::var(ENV_SELF_TEST).ok().map(|name| {
        SelfTest::from_name(&name).unwrap_or_else(|| panic!("unknown self-test {name:?}"))
    });

    let setting = ChildSetting {
        kind,
        quiet,
        generation,
        self_test,
        started,
    };
    let journal = Arc::new(Journal::open(&dir.join(STEPS_LOG), started));
    // A self-test arms no incident, so that only the injected defect can fail it,
    // and plans as if nothing were selected.
    let focus = match self_test {
        Some(_) => BTreeSet::new(),
        None => Selection::from_env().focus().clone(),
    };
    let mut plan = Plan::generate(
        kind,
        seed,
        steps,
        self_test.is_none() && !calibrating(),
        &focus,
    );
    // The ids of later steps stay free, so this cannot clash with them.
    if self_test == Some(SelfTest::MemoryEvents) {
        plan.steps[0].op = Op::CreateIndex(IndexKind::BTree);
    }
    if let (Some(test), Some(rounds)) = (self_test, plan.rounds.as_mut()) {
        rounds.for_self_test(test, plan.initial_rows);
    }
    let rendered = plan.render(kind, seed, steps);
    fs::write(dir.join(PLAN_LOG), &rendered).expect("write the plan");
    let print = fingerprint(&rendered);
    let result = match std::env::var(ENV_PLAN) {
        Ok(expected) if expected != print => Err(fail(
            "plan_changed",
            0,
            format_args!(
                "this build generates plan {print} for seed {seed}, while the run being replayed had {expected}"
            ),
        )),
        _ => run_on_runtime(setting, plan, &dir, journal.clone()),
    };
    if let Err(failure) = result {
        journal.line(format_args!(
            "result=fail class={} step={} detail={}",
            failure.class, failure.step, failure.detail
        ));
        std::process::exit(1);
    }
    journal.line("result=pass");
}

/// What a child process knows about itself beyond its plan.
#[derive(Clone, Copy)]
struct ChildSetting {
    kind: ScenarioKind,
    quiet: Duration,
    /// Which process of the seed this is, the first one being 1.
    generation: u64,
    self_test: Option<SelfTest>,
    started: Instant,
}

fn run_on_runtime(
    setting: ChildSetting,
    plan: Plan,
    dir: &Path,
    journal: Arc<Journal>,
) -> Result<(), Failure> {
    // A round's writers and readers are tasks, which only race with threads to
    // run on.
    let threads = match setting.kind {
        ScenarioKind::Concurrency | ScenarioKind::Soak => 4,
        _ => 2,
    };
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(threads).enable_all();
    // A soak counts threads: the blocking pool must not keep the ones of a peak
    // that is over, as it would for 10 s while any work keeps coming.
    if setting.kind == ScenarioKind::Soak {
        builder.thread_keep_alive(soak::BLOCKING_KEEP_ALIVE);
    }
    let runtime = builder.build().expect("build the runtime");
    // A panic outside an operation, in a check for example, must still end in
    // a verdict. Dropping the runtime while unwinding would wait, without a
    // limit, for any thread that panic has wedged.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(async {
            let mut run = Box::pin(Run::setup(setting, plan, dir, journal)).await?;
            Box::pin(run.resume()).await?;
            Box::pin(run.run_steps()).await?;
            Box::pin(run.finish()).await
        })
    }));
    // A failed round or soak leaves its tasks running, and the process is about to
    // exit. Shutting the runtime down would cancel their scans, and a scan cancelled
    // that way panics (`filtered_read.rs`), which would hide the failure.
    if matches!(setting.kind, ScenarioKind::Concurrency | ScenarioKind::Soak)
        && matches!(result, Ok(Err(_)))
    {
        std::mem::forget(runtime);
    } else {
        // A self-test leaves a task behind on purpose; do not wait for it.
        runtime.shutdown_timeout(Duration::from_secs(5));
    }
    result.unwrap_or_else(|panic| Err(fail("panic", 0, panic_message(panic.as_ref()))))
}

fn env<T: std::str::FromStr>(name: &str) -> T {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
    value
        .parse()
        .unwrap_or_else(|_| panic!("{name} has a bad value {value:?}"))
}

/// Ties this process to the parent that started it. Returns false when the
/// environment names no such parent, for example when `LANCE_STRESS_CHILD` was
/// left in a shell, so a normal test run is not disturbed.
fn tie_to_parent() -> bool {
    let notice = |message: &str| {
        // The workspace denies the printing macros.
        let line = format!("stress child: {message}; not running a seed\n");
        std::io::stderr().write_all(line.as_bytes()).ok();
    };
    let Some(parent) = std::env::var(ENV_PARENT_PID)
        .ok()
        .and_then(|pid| pid.parse::<u32>().ok())
    else {
        notice("no parent process is named");
        return false;
    };
    // SAFETY: `prctl` with these options only changes attributes of this process.
    unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong) };
    // Checked after the death signal is set, so a parent that exits in between
    // is not missed.
    if std::os::unix::process::parent_id() != parent {
        notice("the named parent is not this process's parent");
        return false;
    }
    // Lets the parent's gdb attach to a hung child under
    // `kernel.yama.ptrace_scope = 1`, which only allows tracing descendants.
    // SAFETY: as above.
    unsafe { libc::prctl(libc::PR_SET_PTRACER, parent as libc::c_ulong) };
    true
}

/// The step log. Every line goes out in a single write, so the parent sees
/// progress at once, though a SIGKILL can still tear one at a page boundary.
pub struct Journal {
    file: Mutex<File>,
    started: Instant,
}

impl Journal {
    fn open(path: &Path, started: Instant) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open the step log");
        Self {
            file: Mutex::new(file),
            started,
        }
    }

    pub fn line(&self, line: impl fmt::Display) {
        let text = format!("t={} {line}\n", self.started.elapsed().as_millis());
        self.file
            .lock()
            .unwrap()
            .write_all(text.as_bytes())
            .expect("write the step log");
    }
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix64(seed: u64) -> u64 {
    mix(seed.wrapping_add(GOLDEN_GAMMA))
}

/// The plan's random source. A splitmix64 sequence of its own keeps a seed's
/// plan the same whatever version of `rand` the workspace locks.
struct PlanRng(u64);

impl PlanRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN_GAMMA);
        mix(self.0)
    }

    /// Close enough to uniform for the small ranges a plan draws from.
    fn range(&mut self, range: RangeInclusive<u64>) -> u64 {
        range.start() + self.next() % (range.end() - range.start() + 1)
    }

    fn chance(&mut self, probability: f64) -> bool {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64) < probability
    }
}

// Every column is a function of the row id, so the model only keeps ids.

fn category(id: i64) -> i32 {
    (splitmix64(id as u64 ^ 0xCA7E_9087) % CATEGORIES as u64) as i32
}

fn tokens(id: i64, scale: &Scale) -> impl Iterator<Item = u64> {
    let (doc_tokens, vocabulary) = (scale.doc_tokens, scale.vocabulary);
    (0..doc_tokens).map(move |slot| splitmix64(((id as u64) << 8) | slot) % vocabulary)
}

fn text(id: i64, scale: &Scale) -> String {
    tokens(id, scale)
        .map(|token| format!("tok{token:04}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn vector(data_seed: u64, id: i64) -> [f32; VECTOR_DIM as usize] {
    std::array::from_fn(|lane| {
        let bits =
            splitmix64(data_seed ^ splitmix64((id as u64) * VECTOR_DIM as u64 + lane as u64));
        (bits >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    })
}

fn squared_l2(a: &[f32; VECTOR_DIM as usize], b: &[f32; VECTOR_DIM as usize]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn schema() -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", DataType::Int32, false),
        Field::new("vec", DataType::FixedSizeList(item, VECTOR_DIM), false),
        Field::new("text", DataType::Utf8, false),
    ]))
}

fn rows(data_seed: u64, ids: Range<i64>, scale: Scale) -> impl RecordBatchReader + Send + 'static {
    let end = ids.end;
    let chunks = ids.step_by(CHUNK_ROWS as usize).map(move |start| {
        Ok(batch_of_rows(
            data_seed,
            start..(start + CHUNK_ROWS).min(end),
            &scale,
        ))
    });
    RecordBatchIterator::new(chunks, schema())
}

fn batch_of_rows(data_seed: u64, ids: Range<i64>, scale: &Scale) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let vectors = Float32Array::from_iter_values(ids.iter().flat_map(|id| vector(data_seed, *id)));
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ids.clone())),
        Arc::new(Int32Array::from_iter_values(
            ids.iter().map(|id| category(*id)),
        )),
        Arc::new(
            FixedSizeListArray::try_new(item, VECTOR_DIM, Arc::new(vectors), None)
                .expect("build the vectors"),
        ),
        Arc::new(StringArray::from_iter_values(
            ids.iter().map(|id| text(*id, scale)),
        )),
    ];
    RecordBatch::try_new(schema(), columns).expect("build a batch")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum IndexKind {
    BTree,
    /// On the join key of a merge insert, which then takes its indexed path.
    BTreeId,
    /// On a column that changes, which an update in place takes fragments away from.
    BTreeStamp,
    IvfFlat,
    Inverted,
}

impl IndexKind {
    const ALL: [Self; 5] = [
        Self::BTree,
        Self::BTreeId,
        Self::BTreeStamp,
        Self::IvfFlat,
        Self::Inverted,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::BTree => "idx_category",
            Self::BTreeId => "idx_id",
            Self::BTreeStamp => "idx_gen",
            Self::IvfFlat => "idx_vec",
            Self::Inverted => "idx_text",
        }
    }

    fn column(self) -> &'static str {
        match self {
            Self::BTree => "category",
            Self::BTreeId => "id",
            Self::BTreeStamp => "gen",
            Self::IvfFlat => "vec",
            Self::Inverted => "text",
        }
    }

    /// The stages the build reports to its progress hooks.
    fn stages(self) -> &'static [&'static str] {
        match self {
            Self::BTree | Self::BTreeId | Self::BTreeStamp => &["load_data"],
            Self::IvfFlat => &[
                "train_ivf",
                "train_quantizer",
                "shuffle",
                "merge_partitions",
            ],
            Self::Inverted => &[
                "load_data",
                "tokenize_docs",
                "copy_partitions",
                "write_metadata",
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Append {
        first_id: i64,
        rows: i64,
    },
    Delete {
        lo: i64,
        hi: i64,
    },
    CreateIndex(IndexKind),
    /// Updates the rows with these ids that exist and inserts the others.
    /// Columns are functions of the id, so a lost update stays invisible,
    /// while a lost or doubled row does not.
    MergeInsert {
        first_id: i64,
        rows: i64,
    },
    Compact,
    Scan,
    /// Removes the versions before the last `retain` and the files only they use.
    Cleanup {
        retain: usize,
        delete_unverified: bool,
    },
    /// Indexes the fragments the indices do not cover yet.
    OptimizeIndices,
}

/// How an operation moves the latest version when it succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitShape {
    Exactly1,
    /// An index optimization may find nothing to do.
    UpTo1,
    /// A compaction commits a reservation of fragment ids before its rewrite,
    /// and may find nothing to do.
    UpTo2,
    Never,
}

impl CommitShape {
    fn max_delta(self) -> u64 {
        match self {
            Self::Exactly1 | Self::UpTo1 => 1,
            Self::UpTo2 => 2,
            Self::Never => 0,
        }
    }
}

impl Op {
    fn shape(&self) -> CommitShape {
        match self {
            Self::OptimizeIndices => CommitShape::UpTo1,
            Self::Compact => CommitShape::UpTo2,
            Self::Scan | Self::Cleanup { .. } => CommitShape::Never,
            _ => CommitShape::Exactly1,
        }
    }

    /// The operation without its arguments, for coverage counts.
    fn kind(&self) -> String {
        match self {
            Self::CreateIndex(kind) => format!("create_index_{}", kind.name()),
            _ => self.focus_name().to_string(),
        }
    }

    /// The operation as `LANCE_STRESS_ONLY` names it: one name for every index.
    fn focus_name(&self) -> &'static str {
        match self {
            Self::Append { .. } => "append",
            Self::Delete { .. } => "delete",
            Self::CreateIndex(_) => "create_index",
            Self::MergeInsert { .. } => "merge_insert",
            Self::Compact => "compact",
            Self::Scan => "scan",
            Self::Cleanup { .. } => "cleanup",
            Self::OptimizeIndices => "optimize_indices",
        }
    }
}

/// The operations a scenario can put its trouble on, as `LANCE_STRESS_ONLY` names
/// them: what its plans draw, and in `concurrency` what its forced pairs run.
pub fn focusable_ops(kind: ScenarioKind) -> Vec<&'static str> {
    match kind {
        ScenarioKind::StorageFaults | ScenarioKind::Cancellation | ScenarioKind::SlowStorage => {
            vec![
                "append",
                "delete",
                "create_index",
                "optimize_indices",
                "cleanup",
            ]
        }
        ScenarioKind::ProcessKill => vec![
            "append",
            "delete",
            "create_index",
            "optimize_indices",
            "merge_insert",
            "compact",
            "cleanup",
        ],
        ScenarioKind::MemoryLimits => vec![
            "append",
            "delete",
            "create_index",
            "optimize_indices",
            "merge_insert",
            "compact",
            "scan",
            "cleanup",
        ],
        // Their cycles are their own.
        ScenarioKind::HeapGrowth | ScenarioKind::Soak => Vec::new(),
        ScenarioKind::Concurrency => concurrency::focusable_ops(),
    }
}

/// The indices a scenario builds.
fn index_kinds(kind: ScenarioKind) -> &'static [IndexKind] {
    match kind {
        ScenarioKind::ProcessKill | ScenarioKind::MemoryLimits => &[
            IndexKind::BTree,
            IndexKind::BTreeId,
            IndexKind::IvfFlat,
            IndexKind::Inverted,
        ],
        _ => &[IndexKind::BTree, IndexKind::IvfFlat, IndexKind::Inverted],
    }
}

/// The memory limit on a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Limit {
    /// `memory.max` at what the cgroup holds plus this many KiB. Going over it
    /// ends with the OOM killer.
    Cgroup { budget_kib: u64 },
    /// `LANCE_MEM_POOL_SIZE`: going over it means spilling to disk or a clean
    /// error. Sorts for BTree training and the indexed path of a merge insert
    /// honor it.
    Pool { kib: u64 },
    /// `InvertedIndexParams::memory_limit_mb`: going over it means more
    /// partitions in the index.
    FtsMemory { mib: u64 },
    /// `Scanner::io_buffer_size`: the read budget of a scan.
    IoBuffer { kib: u64 },
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cgroup { budget_kib } => write!(f, "cgroup:{budget_kib}"),
            Self::Pool { kib } => write!(f, "pool:{kib}"),
            Self::FtsMemory { mib } => write!(f, "fts_memory:{mib}"),
            Self::IoBuffer { kib } => write!(f, "io_buffer:{kib}"),
        }
    }
}

/// The budgets a seed picks from for an operation: an eighth, a quarter, a half,
/// one and two times the heap a calibration run saw it take on the data of this
/// scenario (`LANCE_STRESS_CALIBRATE=1`). The table is fixed, so that a seed
/// decides its budgets whatever the machine does.
fn cgroup_budgets_kib(op: &Op) -> &'static [u64] {
    match op {
        Op::Compact => &[3072, 6144, 12288, 24576, 49152],
        Op::CreateIndex(IndexKind::IvfFlat) => &[1792, 3584, 7168, 14336, 28672],
        Op::OptimizeIndices => &[768, 1536, 3072, 6144, 12288],
        // Its heap comes back from the allocator, and the cgroup sees little
        // more than the manifests it reads: one cleanup in 60 reached even these.
        Op::Cleanup { .. } => &[16, 32, 64, 128, 256],
        Op::MergeInsert { .. } => &[800, 1600, 3200, 6400, 12800],
        // Most of what these hold is page cache, which the limit evicts, or a
        // few rows to sort.
        Op::Delete { .. } | Op::Scan | Op::CreateIndex(IndexKind::BTree | IndexKind::BTreeId) => {
            &[80, 160, 320, 640, 1280, 2560, 5120]
        }
        _ => &[640, 1280, 2560, 5120, 10240],
    }
}

fn pick(rng: &mut PlanRng, values: &[u64]) -> u64 {
    values[rng.range(0..=values.len() as u64 - 1) as usize]
}

/// The limit on a step. Half of the steps a limit of Lance's own applies to get
/// that one instead of the kernel's. Pool sizes follow what is sorted, not the
/// heap of the operation: a BTree sorts 12 bytes per row, around 0.5 MB here.
fn random_limit(rng: &mut PlanRng, op: &Op, model: &Model) -> Limit {
    let own = rng.chance(0.5);
    match op {
        Op::CreateIndex(IndexKind::BTree | IndexKind::BTreeId) if own => Limit::Pool {
            kib: pick(rng, &[32, 64, 128, 256, 512, 1024]),
        },
        // Without an index on the key, the join does not use the pool at all, and
        // the kernel's limit gets plenty of merges without one.
        Op::MergeInsert { .. }
            if model.indices.contains(&IndexKind::BTreeId) && (own || rng.chance(0.6)) =>
        {
            Limit::Pool {
                kib: pick(rng, &[512, 1024, 2048, 4096, 8192, 16384]),
            }
        }
        Op::CreateIndex(IndexKind::Inverted) if own => Limit::FtsMemory {
            mib: pick(rng, &[1, 2, 4]),
        },
        Op::Scan if own => Limit::IoBuffer {
            kib: pick(rng, &[4, 64, 1024, 16384]),
        },
        _ => Limit::Cgroup {
            budget_kib: pick(rng, cgroup_budgets_kib(op)),
        },
    }
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

/// The size of a `heap_growth` run.
pub fn heap_growth_steps() -> u64 {
    (HEAP_CYCLES * HEAP_CYCLE_OPS) as u64
}

/// How long a `soak` run lasts. A sample needs a second at least.
pub fn soak_secs() -> u64 {
    let secs = match std::env::var(ENV_SOAK_SECS) {
        Ok(value) => value.parse().unwrap_or_else(|_| {
            panic!("{ENV_SOAK_SECS} must be an unsigned integer, got {value:?}")
        }),
        Err(_) => soak::DEFAULT_SECS,
    };
    assert!(
        secs >= soak::SAMPLES as u64,
        "{ENV_SOAK_SECS}={secs} leaves less than a second between two of its {} samples",
        soak::SAMPLES
    );
    secs
}

/// The same cycle over and over, on a steady number of rows: whatever grows is
/// not the data.
fn heap_growth_plan(rng: &mut PlanRng, scale: &Scale, initial_rows: i64) -> Vec<Step> {
    let added = rng.range(scale.added_rows.0..=scale.added_rows.1) as i64;
    let mut model = Model::new(initial_rows);
    let mut oldest = 0;
    let mut steps = Vec::with_capacity(HEAP_CYCLES * HEAP_CYCLE_OPS);
    for _ in 0..HEAP_CYCLES {
        let before = model.live.len();
        let append = Op::Append {
            first_id: model.next_id,
            rows: added,
        };
        model.apply(&append);
        let merge = Op::MergeInsert {
            first_id: model.next_id - added / 2,
            rows: added,
        };
        model.apply(&merge);
        let grown = (model.live.len() - before) as i64;
        let delete = Op::Delete {
            lo: oldest,
            hi: oldest + grown,
        };
        model.apply(&delete);
        oldest += grown;
        for op in [
            // First, so that the merge takes its indexed path.
            Op::CreateIndex(IndexKind::BTreeId),
            append,
            merge,
            delete,
            Op::CreateIndex(IndexKind::BTree),
            Op::CreateIndex(IndexKind::IvfFlat),
            Op::CreateIndex(IndexKind::Inverted),
            Op::Compact,
            Op::Scan,
        ] {
            steps.push(Step {
                op,
                incident: None,
                resync: true,
                in_flight_us: 0,
                limit: None,
            });
        }
    }
    steps
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Append { first_id, rows } => {
                write!(f, "append:ids={first_id}..{}", first_id + rows)
            }
            Self::Delete { lo, hi } => write!(f, "delete:ids={lo}..{hi}"),
            Self::CreateIndex(kind) => write!(f, "create_index:{}", kind.name()),
            Self::MergeInsert { first_id, rows } => {
                write!(f, "merge_insert:ids={first_id}..{}", first_id + rows)
            }
            Self::Compact => f.write_str("compact"),
            Self::Scan => f.write_str("scan"),
            Self::Cleanup {
                retain,
                delete_unverified,
            } => write!(
                f,
                "cleanup:retain={retain},delete_unverified={delete_unverified}"
            ),
            Self::OptimizeIndices => f.write_str("optimize_indices"),
        }
    }
}

/// What the dataset must hold: the live row ids and the indices.
#[derive(Clone, Debug)]
struct Model {
    live: BTreeSet<i64>,
    next_id: i64,
    indices: BTreeSet<IndexKind>,
    /// An index misses fragments that rows were written to since it was built,
    /// which gives an index optimization work.
    unindexed: bool,
}

impl Model {
    fn new(rows: i64) -> Self {
        Self {
            live: (0..rows).collect(),
            next_id: rows,
            indices: BTreeSet::new(),
            unindexed: false,
        }
    }

    fn apply(&mut self, op: &Op) {
        match *op {
            Op::Append { first_id, rows } => {
                self.live.extend(first_id..first_id + rows);
                self.next_id = first_id + rows;
                self.unindexed = !self.indices.is_empty();
            }
            Op::Delete { lo, hi } => self.live.retain(|id| !(lo..hi).contains(id)),
            Op::CreateIndex(kind) => {
                self.indices.insert(kind);
                // The other indices may still miss fragments.
                if self.indices.len() == 1 {
                    self.unindexed = false;
                }
            }
            Op::MergeInsert { first_id, rows } => {
                self.live.extend(first_id..first_id + rows);
                self.next_id = self.next_id.max(first_id + rows);
                self.unindexed = !self.indices.is_empty();
            }
            Op::OptimizeIndices => self.unindexed = false,
            Op::Compact | Op::Scan | Op::Cleanup { .. } => {}
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StoreChoice {
    Local,
    SharedMemory,
    /// The same files as `Local`, but every request passes the object store,
    /// so an incident reaches data, index and transaction files too.
    FileObjectStore,
}

#[derive(Clone, Debug)]
struct Step {
    op: Op,
    incident: Option<Incident>,
    /// Whether the handle catches up with the latest version first. Without
    /// it, an operation after a commit the handle never saw has to resolve a
    /// conflict. A delete always catches up: Lance evaluates its predicate on
    /// the handle's snapshot and rebases it over an append without evaluating
    /// it again, but evaluates it again after a conflict with another delete,
    /// so on a stale handle the rows it removes depend on what it raced with.
    resync: bool,
    /// How long a request in flight runs before a planned death.
    in_flight_us: u64,
    limit: Option<Limit>,
}

/// Everything a seed decides. Every operation ends up committed once, by its
/// first try or by the retry, so the plan can track the model on its own.
struct Plan {
    store: StoreChoice,
    scale: Scale,
    data_seed: u64,
    initial_rows: i64,
    steps: Vec<Step>,
    /// What `concurrency` runs instead of `steps`.
    rounds: Option<concurrency::RoundsPlan>,
    /// What `soak` runs instead of `steps`.
    soak: Option<soak::SoakPlan>,
    /// The operations that `LANCE_STRESS_ONLY` puts the trouble on.
    focus: BTreeSet<&'static str>,
}

impl Plan {
    /// A focus makes every other step one of its operations and leaves the trouble
    /// to those steps alone, while the steps between build the history they need;
    /// in `concurrency` it picks the forced pairs instead.
    fn generate(
        kind: ScenarioKind,
        seed: u64,
        steps: usize,
        with_incidents: bool,
        focus: &BTreeSet<&'static str>,
    ) -> Self {
        // Only what the scenario can run: a list for several scenarios plans each
        // of them as a list of its own operations would.
        let focus: BTreeSet<&'static str> = focusable_ops(kind)
            .into_iter()
            .filter(|op| focus.contains(op))
            .collect();
        let focus = &focus;
        let mut rng = PlanRng(seed);
        let store = match kind {
            ScenarioKind::StorageFaults => StoreChoice::SharedMemory,
            ScenarioKind::Cancellation if rng.chance(0.5) => StoreChoice::Local,
            ScenarioKind::Cancellation => StoreChoice::SharedMemory,
            ScenarioKind::ProcessKill if rng.chance(0.5) => StoreChoice::Local,
            ScenarioKind::ProcessKill => StoreChoice::FileObjectStore,
            ScenarioKind::MemoryLimits | ScenarioKind::HeapGrowth => StoreChoice::Local,
            ScenarioKind::SlowStorage if rng.chance(0.5) => StoreChoice::SharedMemory,
            ScenarioKind::SlowStorage => StoreChoice::FileObjectStore,
            ScenarioKind::Concurrency => match rng.range(0..=2) {
                0 => StoreChoice::Local,
                1 => StoreChoice::FileObjectStore,
                _ => StoreChoice::SharedMemory,
            },
            // An in-memory store keeps every file in the heap, with the whole 5 MiB
            // upload buffer it was written from, so the heap would measure the files.
            ScenarioKind::Soak if rng.chance(0.5) => StoreChoice::Local,
            ScenarioKind::Soak => StoreChoice::FileObjectStore,
        };
        let scale = match kind {
            ScenarioKind::MemoryLimits => LARGE,
            ScenarioKind::HeapGrowth => MEDIUM,
            _ => SMALL,
        };
        let data_seed = rng.next();
        let initial_rows = rng.range(scale.initial_rows.0..=scale.initial_rows.1) as i64;
        if kind == ScenarioKind::HeapGrowth {
            return Self {
                store,
                scale,
                data_seed,
                initial_rows,
                steps: heap_growth_plan(&mut rng, &scale, initial_rows),
                rounds: None,
                soak: None,
                focus: BTreeSet::new(),
            };
        }
        if kind == ScenarioKind::Soak {
            return Self {
                store,
                scale,
                data_seed,
                initial_rows,
                steps: Vec::new(),
                rounds: None,
                soak: Some(soak::plan(&mut rng, initial_rows)),
                focus: BTreeSet::new(),
            };
        }
        if kind == ScenarioKind::Concurrency {
            let rounds = concurrency::plan_rounds(
                &mut rng,
                seed,
                store,
                &scale,
                initial_rows,
                steps,
                with_incidents,
                focus,
            );
            return Self {
                store,
                scale,
                data_seed,
                initial_rows,
                steps: Vec::new(),
                rounds: Some(rounds),
                soak: None,
                focus: focus.clone(),
            };
        }
        let mut model = Model::new(initial_rows);
        let mut planned = Vec::with_capacity(steps);
        for index in 0..steps {
            let focused = (!focus.is_empty() && index % 2 == 1)
                .then(|| focused_op(&mut rng, kind, &scale, &model, focus))
                .flatten();
            let troubled = focus.is_empty() || focused.is_some();
            let op = match focused {
                Some(op) => op,
                // A merge insert honors the pool only with an index on its key.
                None if kind == ScenarioKind::MemoryLimits && index == 0 && rng.chance(0.5) => {
                    Op::CreateIndex(IndexKind::BTreeId)
                }
                // An optimization of the focus needs work: an index, and fragments
                // it does not cover. Other steps go on as they would, short of
                // taking that work away from the next step of the focus.
                None if focus.contains("optimize_indices")
                    && index % 2 == 0
                    && !model.unindexed =>
                {
                    match model.indices.is_empty() {
                        true => Op::CreateIndex(*pick_from(&mut rng, index_kinds(kind))),
                        false => Op::Append {
                            first_id: model.next_id,
                            rows: added_rows(&mut rng, &scale),
                        },
                    }
                }
                None => {
                    let op = random_op(&mut rng, kind, &scale, &model);
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
                            rows: added_rows(&mut rng, &scale),
                        },
                        false => op,
                    }
                }
            };
            model.apply(&op);
            let incident = random_incident(&mut rng, kind, store, &scale, &op);
            // A stale handle is covered by the scenarios of its own; a read
            // or a merge on one would only report what it was left with.
            let resync = rng.chance(0.7)
                || matches!(op, Op::Delete { .. })
                || matches!(kind, ScenarioKind::ProcessKill | ScenarioKind::MemoryLimits);
            let in_flight_us = match kind {
                ScenarioKind::ProcessKill => rng.range(0..=300),
                _ => 0,
            };
            let limit = match kind {
                ScenarioKind::MemoryLimits if rng.chance(0.85) => {
                    Some(random_limit(&mut rng, &op, &model))
                }
                _ => None,
            };
            planned.push(Step {
                op,
                incident: incident.filter(|_| with_incidents && troubled),
                resync,
                in_flight_us,
                limit: limit.filter(|_| with_incidents && troubled),
            });
        }
        Self {
            store,
            scale,
            data_seed,
            initial_rows,
            steps: planned,
            rounds: None,
            soak: None,
            focus: focus.clone(),
        }
    }

    fn render(&self, kind: ScenarioKind, seed: u64, steps: usize) -> String {
        let mut text = format!(
            "scenario={} seed={seed} steps={steps} store={:?} scale={:?} data_seed={} initial_rows={}\n",
            kind.name(),
            self.store,
            self.scale,
            self.data_seed,
            self.initial_rows
        );
        if !self.focus.is_empty() {
            let focus: Vec<&str> = self.focus.iter().copied().collect();
            text.insert_str(text.len() - 1, &format!(" focus={}", focus.join(",")));
        }
        for (index, step) in self.steps.iter().enumerate() {
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
        if let Some(rounds) = &self.rounds {
            text.push_str(&rounds.render());
        }
        if let Some(soak) = &self.soak {
            text.push_str(&soak.render());
        }
        text
    }
}

/// FNV-1a of a rendered plan, in hex.
fn fingerprint(rendered: &str) -> String {
    let hash = rendered
        .bytes()
        .fold(0xCBF2_9CE4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
        });
    format!("{hash:016x}")
}

/// The fingerprint of the plan a seed generates, for a replay command.
pub fn plan_fingerprint(kind: ScenarioKind, seed: u64, steps: usize) -> String {
    let plan = Plan::generate(
        kind,
        seed,
        steps,
        !calibrating(),
        Selection::from_env().focus(),
    );
    fingerprint(&plan.render(kind, seed, steps))
}

fn random_op(rng: &mut PlanRng, kind: ScenarioKind, scale: &Scale, model: &Model) -> Op {
    let roll = rng.range(0..=99);
    let deletes = match kind {
        ScenarioKind::MemoryLimits => 35..45,
        _ => 35..55,
    };
    if deletes.contains(&roll) && can_delete(scale, model) {
        return delete_op(rng, scale, model);
    }
    // Taken from the appends, and in `memory_limits` from the scans.
    let (cleanups, optimizations) = match kind {
        ScenarioKind::ProcessKill => (20..25, 25..30),
        ScenarioKind::MemoryLimits => (25..28, 28..32),
        _ => (20..26, 26..32),
    };
    if cleanups.contains(&roll) {
        return cleanup_op(rng);
    }
    if optimizations.contains(&roll) && !model.indices.is_empty() {
        return Op::OptimizeIndices;
    }
    match (kind, roll) {
        (_, 55..70) => Op::CreateIndex(IndexKind::BTree),
        (_, 70..85) => Op::CreateIndex(IndexKind::IvfFlat),
        (_, 85..100) => Op::CreateIndex(IndexKind::Inverted),
        (ScenarioKind::ProcessKill | ScenarioKind::MemoryLimits, 0..6) => Op::Compact,
        (ScenarioKind::ProcessKill | ScenarioKind::MemoryLimits, 6..10) => {
            Op::CreateIndex(IndexKind::BTreeId)
        }
        (ScenarioKind::MemoryLimits, 25..35) => Op::Scan,
        (ScenarioKind::ProcessKill | ScenarioKind::MemoryLimits, 10..20) => {
            merge_insert_op(rng, scale, model)
        }
        _ => Op::Append {
            first_id: model.next_id,
            rows: added_rows(rng, scale),
        },
    }
}

/// One of the operations of a focus that this step can run, if any.
fn focused_op(
    rng: &mut PlanRng,
    kind: ScenarioKind,
    scale: &Scale,
    model: &Model,
    focus: &BTreeSet<&'static str>,
) -> Option<Op> {
    let candidates: Vec<&str> = focusable_ops(kind)
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
        "create_index" => Op::CreateIndex(*pick_from(rng, index_kinds(kind))),
        "optimize_indices" => Op::OptimizeIndices,
        "merge_insert" => merge_insert_op(rng, scale, model),
        "compact" => Op::Compact,
        "scan" => Op::Scan,
        "cleanup" => cleanup_op(rng),
        other => unreachable!("{} plans no {other}", kind.name()),
    })
}

fn pick_from<'a, T>(rng: &mut PlanRng, values: &'a [T]) -> &'a T {
    &values[rng.range(0..=values.len() as u64 - 1) as usize]
}

fn added_rows(rng: &mut PlanRng, scale: &Scale) -> i64 {
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

/// How long `slow_storage` slows a call: from a busy store to a struggling one.
const DELAYS_MS: [u64; 6] = [10, 30, 100, 300, 1000, 3000];

/// The calls each operation of `storage_faults` and `slow_storage` makes:
/// (method, area, the highest call to target).
fn storage_calls(op: &Op, scale: &Scale) -> Vec<(Method, Area, u64)> {
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

/// The calls `slow_storage` slows: those of `storage_faults`, as `store` makes
/// them. A store that lists in order finds the latest version by listing; another
/// one through the hint and the manifests it names, and a commit rewrites the
/// hint there (`uses_version_hint`). A cleanup lists on both.
fn slow_calls(op: &Op, scale: &Scale, store: StoreChoice) -> Vec<(Method, Area, u64)> {
    let mut calls = storage_calls(op, scale);
    if store == StoreChoice::FileObjectStore && !matches!(op, Op::Cleanup { .. }) {
        let commits = calls
            .iter()
            .any(|&(method, area, _)| method == Method::Put && area == Area::Manifests);
        calls.retain(|&(method, area, _)| (method, area) != (Method::List, Area::Manifests));
        calls.push((Method::Get, Area::VersionHint, 1));
        calls.push((Method::Head, Area::Manifests, 2));
        if commits {
            calls.push((Method::Put, Area::VersionHint, 1));
        }
    }
    calls
}

fn random_incident(
    rng: &mut PlanRng,
    kind: ScenarioKind,
    store: StoreChoice,
    scale: &Scale,
    op: &Op,
) -> Option<Incident> {
    match kind {
        ScenarioKind::MemoryLimits
        | ScenarioKind::HeapGrowth
        | ScenarioKind::Concurrency
        | ScenarioKind::Soak => None,
        ScenarioKind::StorageFaults => {
            if !rng.chance(0.75) {
                return None;
            }
            let targets = storage_calls(op, scale);
            let (method, area, max_nth) = targets[rng.range(0..=targets.len() as u64 - 1) as usize];
            let nth = rng.range(1..=max_nth);
            let after_call = method == Method::Put && rng.chance(0.5);
            let sticky = rng.chance(0.3);
            // Only a failed manifest write goes on to verify its outcome.
            let blind = method == Method::Put && area == Area::Manifests && rng.chance(0.5);
            Some(Incident::Fail {
                method,
                area,
                nth,
                after_call,
                sticky,
                blind,
            })
        }
        ScenarioKind::SlowStorage => {
            if !rng.chance(0.8) {
                return None;
            }
            let targets = slow_calls(op, scale, store);
            let mut delays: Vec<Delay> = Vec::new();
            for _ in 0..rng.range(1..=3) {
                let (method, area, max_nth) = *pick_from(rng, &targets);
                let nth = rng.range(1..=max_nth);
                let millis = *pick_from(rng, &DELAYS_MS);
                // The store wrapper slows a listing or a deletion only before
                // its request.
                let after = !matches!(method, Method::List | Method::Delete) && rng.chance(0.5);
                if !delays
                    .iter()
                    .any(|delay| (delay.method, delay.area, delay.nth) == (method, area, nth))
                {
                    delays.push(Delay {
                        method,
                        area,
                        nth,
                        millis,
                        after,
                    });
                }
            }
            let hang = rng.chance(0.35).then(|| {
                let (method, area, max_nth) = *pick_from(rng, &targets);
                HangPoint {
                    method,
                    area,
                    nth: rng.range(1..=max_nth),
                    // Only a write can land and then never answer.
                    after: method == Method::Put && rng.chance(0.5),
                    grace_ms: rng.range(20..=500),
                }
            });
            Some(Incident::Slow { delays, hang })
        }
        ScenarioKind::Cancellation | ScenarioKind::ProcessKill => {
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
                    Op::Append { .. } | Op::MergeInsert { .. } | Op::Compact | Op::Scan => {
                        Area::Data
                    }
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
    }
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

/// What a finished operation reports beyond success.
#[derive(Clone, Copy, Debug, Default)]
struct OpOutput {
    fragments_removed: usize,
    /// The rows a scan returned, and the wrapping sum of their ids.
    scanned: Option<(usize, i64)>,
    /// The files a cleanup failed to delete, and went on without.
    failed_deletes: u64,
}

enum Outcome {
    Ok(OpOutput),
    Failed {
        message: String,
        status_unknown: bool,
    },
    Cancelled,
    /// The process died during the operation; a later process tells.
    Killed,
    Panicked(String),
}

impl Outcome {
    fn name(&self) -> &'static str {
        match self {
            Self::Ok(_) => "ok",
            Self::Failed { .. } => "err",
            Self::Cancelled => "cancelled",
            Self::Killed => "killed",
            Self::Panicked(_) => "panic",
        }
    }

    fn detail(&self) -> &str {
        match self {
            Self::Failed { message, .. } | Self::Panicked(message) => message,
            Self::Ok(_) | Self::Cancelled | Self::Killed => "",
        }
    }

    fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }
}

struct Failure {
    class: &'static str,
    step: usize,
    detail: String,
}

fn fail(class: &'static str, step: usize, detail: impl fmt::Display) -> Failure {
    Failure {
        class,
        step,
        detail: one_line(&detail.to_string()),
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    let message = panic
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_string());
    one_line(&message)
}

fn int64_column(batch: &RecordBatch, name: &str) -> Vec<i64> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no {name} column"))
        .as_primitive::<Int64Type>()
        .values()
        .to_vec()
}

fn describe_difference(expected: &BTreeSet<i64>, actual: &BTreeSet<i64>) -> String {
    let missing: Vec<i64> = expected.difference(actual).take(5).copied().collect();
    let unexpected: Vec<i64> = actual.difference(expected).take(5).copied().collect();
    format!(
        "{} rows where the model has {}; missing {missing:?}, unexpected {unexpected:?} (the first five of each)",
        actual.len(),
        expected.len()
    )
}

/// Checks a query's rows against the model, and that none comes back twice.
fn compare_ids(
    n: usize,
    what: &str,
    found: &[i64],
    expected: &BTreeSet<i64>,
) -> Result<(), Failure> {
    let distinct: BTreeSet<i64> = found.iter().copied().collect();
    if distinct.len() != found.len() {
        return Err(fail(
            "content",
            n,
            format_args!(
                "{what}: {} rows but {} distinct ids",
                found.len(),
                distinct.len()
            ),
        ));
    }
    if &distinct != expected {
        return Err(fail(
            "content",
            n,
            format_args!("{what}: {}", describe_difference(expected, &distinct)),
        ));
    }
    Ok(())
}

async fn paths<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeSet<String>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("path")
                .expect("a path column")
                .as_string::<i32>()
                .iter()
                .flatten()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect())
}

/// The files `Dataset::tracked_files` lists, with the versions that reference
/// each of them.
async fn files_by_version<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeMap<String, BTreeSet<u64>>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut files: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    for batch in &batches {
        let paths = batch
            .column_by_name("path")
            .expect("a path column")
            .as_string::<i32>();
        for (row, version) in int64_column(batch, "version").into_iter().enumerate() {
            files
                .entry(paths.value(row).to_string())
                .or_default()
                .insert(version as u64);
        }
    }
    Ok(files)
}

/// The versions whose manifests a listing finds, without reading them, as
/// `retain_n_versions` does.
async fn listed_versions(ds: &Dataset, n: usize) -> Result<BTreeSet<u64>, Failure> {
    Ok(ds
        .version_refs()
        .await
        .map_err(|error| fail("unreadable", n, error))?
        .iter()
        .map(|version| version.version)
        .collect())
}

/// What a cleanup must remove and what it must leave, from the files the
/// dataset holds before it starts.
struct CleanupFiles {
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

/// The rows of a version by their ids: how many, and a sum that does not
/// depend on their order.
fn ids_digest(ids: impl IntoIterator<Item = i64>) -> String {
    let (count, sum) = ids.into_iter().fold((0usize, 0u64), |(count, sum), id| {
        (count + 1, sum.wrapping_add(mix(id as u64)))
    });
    format!("{count}:{sum:016x}")
}

/// The files `Dataset::all_files` lists, with their modification times in
/// microseconds since the epoch.
async fn file_times<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeMap<String, i64>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut times = BTreeMap::new();
    for batch in &batches {
        let paths = batch
            .column_by_name("path")
            .expect("a path column")
            .as_string::<i32>();
        let modified = batch
            .column_by_name("last_modified")
            .expect("a last_modified column")
            .as_primitive::<TimestampMicrosecondType>();
        for row in 0..batch.num_rows() {
            times.insert(paths.value(row).to_string(), modified.value(row));
        }
    }
    Ok(times)
}

/// The files `Dataset::all_files` lists, with their sizes in bytes.
async fn file_sizes<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeMap<String, i64>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut sizes = BTreeMap::new();
    for batch in &batches {
        let paths = batch
            .column_by_name("path")
            .expect("a path column")
            .as_string::<i32>();
        let bytes = batch
            .column_by_name("size_bytes")
            .expect("a size_bytes column")
            .as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            sizes.insert(paths.value(row).to_string(), bytes.value(row));
        }
    }
    Ok(sizes)
}

/// The operation of a step on the handle that meets the incidents.
fn op_future<'a>(
    ds: &'a mut Dataset,
    hooks: &Arc<StressHooks>,
    data_seed: u64,
    scale: Scale,
    op: Op,
    limit: Option<Limit>,
) -> BoxFuture<'a, lance::Result<OpOutput>> {
    let hooks = hooks.clone();
    match op {
        Op::Append {
            first_id,
            rows: count,
        } => Box::pin(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: scale.rows_per_file,
                progress: hooks,
                ..Default::default()
            };
            let source = rows(data_seed, first_id..first_id + count, scale);
            Box::pin(ds.append(source, Some(params))).await?;
            Ok(OpOutput::default())
        }),
        Op::Delete { lo, hi } => Box::pin(async move {
            let predicate = format!("id >= {lo} AND id < {hi}");
            Box::pin(ds.delete(&predicate)).await?;
            Ok(OpOutput::default())
        }),
        Op::MergeInsert {
            first_id,
            rows: count,
        } => Box::pin(async move {
            let source: Box<dyn RecordBatchReader + Send> =
                Box::new(rows(data_seed, first_id..first_id + count, scale));
            let job = MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])?
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()?;
            let (merged, _) = Box::pin(job.execute_reader(source)).await?;
            *ds = merged.as_ref().clone();
            Ok(OpOutput::default())
        }),
        Op::Compact => Box::pin(async move {
            let metrics = Box::pin(compact_files(ds, CompactionOptions::default(), None)).await?;
            Ok(OpOutput {
                fragments_removed: metrics.fragments_removed,
                ..Default::default()
            })
        }),
        Op::Cleanup {
            retain,
            delete_unverified,
        } => Box::pin(async move {
            let policy = CleanupPolicyBuilder::default()
                .retain_n_versions(ds, retain)
                .await?
                .delete_unverified(delete_unverified)
                .error_if_tagged_old_versions(false)
                .build();
            let removed = Box::pin(ds.cleanup_with_policy(policy)).await?;
            Ok(OpOutput {
                failed_deletes: removed.failed_deletes,
                ..Default::default()
            })
        }),
        Op::OptimizeIndices => Box::pin(async move {
            Box::pin(ds.optimize_indices(&OptimizeOptions::new().progress(hooks))).await?;
            Ok(OpOutput::default())
        }),
        Op::Scan => Box::pin(async move {
            let mut scan = ds.scan();
            if let Some(Limit::IoBuffer { kib }) = limit {
                scan.io_buffer_size(kib << 10);
            }
            let mut stream = scan.try_into_stream().await?;
            let mut count = 0;
            let mut sum = 0i64;
            while let Some(batch) = stream.try_next().await? {
                count += batch.num_rows();
                for id in int64_column(&batch, "id") {
                    sum = sum.wrapping_add(id);
                }
            }
            Ok(OpOutput {
                scanned: Some((count, sum)),
                ..Default::default()
            })
        }),
        Op::CreateIndex(kind) => Box::pin(async move {
            let columns = [kind.column()];
            let progress: Arc<dyn IndexBuildProgress> = hooks;
            let name = kind.name().to_string();
            match kind {
                IndexKind::BTree | IndexKind::BTreeId | IndexKind::BTreeStamp => {
                    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
                    ds.create_index_builder(&columns, IndexType::BTree, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
                IndexKind::IvfFlat => {
                    let params = VectorIndexParams::ivf_flat(IVF_PARTITIONS, MetricType::L2);
                    ds.create_index_builder(&columns, IndexType::IvfFlat, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
                IndexKind::Inverted => {
                    let params = match limit {
                        Some(Limit::FtsMemory { mib }) => {
                            InvertedIndexParams::default().memory_limit_mb(mib)
                        }
                        _ => InvertedIndexParams::default(),
                    };
                    ds.create_index_builder(&columns, IndexType::Inverted, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
            }
            .map(|_| OpOutput::default())
        }),
    }
}

/// What the queries of an index must answer: the rows of one version.
struct Answers<'a> {
    live: &'a BTreeSet<i64>,
    next_id: i64,
    /// The value of `gen` per row, in a scenario that has the column.
    stamps: Option<&'a BTreeMap<i64, i64>>,
    /// Values of `gen` to ask about, besides those the rows hold.
    stamps_ever: Option<&'a BTreeSet<i64>>,
    /// Whether to ask about a few of the values that the rows hold, not all.
    few_stamps: bool,
    scale: Scale,
    data_seed: u64,
    check_plan: bool,
    /// Varies the queries between checks.
    salt: usize,
}

/// Queries an index and compares the answers with what the version holds. The
/// first query of each index can check its plan, since a query that falls back
/// to a scan would answer right whatever the index holds.
async fn check_index(
    ds: &Dataset,
    n: usize,
    kind: IndexKind,
    answers: &Answers<'_>,
) -> Result<(), Failure> {
    let unreadable = |error: lance::Error| fail("unreadable", n, error);
    match kind {
        IndexKind::BTreeStamp => {
            let stamps = answers
                .stamps
                .expect("an index on gen only exists in a scenario that has the column");
            let distinct: BTreeSet<i64> = stamps.values().copied().collect();
            let held: Vec<i64> = distinct.iter().copied().collect();
            let mut values: Vec<i64> = if answers.few_stamps && !held.is_empty() {
                vec![
                    held[0],
                    held[answers.salt % held.len()],
                    held[held.len() - 1],
                ]
            } else {
                held
            };
            // A value that no row holds now, which an index that went stale can
            // still answer, and one that no row ever held.
            values.extend(
                answers
                    .stamps_ever
                    .into_iter()
                    .flatten()
                    .filter(|value| !distinct.contains(value)),
            );
            values.push(-1);
            for (position, value) in values.into_iter().enumerate() {
                let filter = format!("gen = {value}");
                let mut scan = ds.scan();
                scan.filter(&filter).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if position == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("@idx_gen(BTree)") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("{filter} does not use idx_gen: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = stamps
                    .iter()
                    .filter(|(_, stamp)| **stamp == value)
                    .map(|(id, _)| *id)
                    .collect();
                compare_ids(n, &filter, &int64_column(&batch, "id"), &expected)?;
            }
        }
        IndexKind::BTreeId => {
            let last = answers.next_id.max(1);
            for slot in 0..3 {
                let lo = last * slot / 3;
                let hi = last * (slot + 1) / 3;
                let filter = format!("id >= {lo} AND id < {hi}");
                let mut scan = ds.scan();
                scan.filter(&filter).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if slot == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("@idx_id(BTree)") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("{filter} does not use idx_id: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = answers.live.range(lo..hi).copied().collect();
                compare_ids(n, &filter, &int64_column(&batch, "id"), &expected)?;
            }
        }
        IndexKind::BTree => {
            for value in 0..CATEGORIES {
                let filter = format!("category = {value}");
                let mut scan = ds.scan();
                scan.filter(&filter).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if value == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("@idx_category(BTree)") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("{filter} does not use idx_category: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = answers
                    .live
                    .iter()
                    .copied()
                    .filter(|id| category(*id) == value)
                    .collect();
                compare_ids(n, &filter, &int64_column(&batch, "id"), &expected)?;
            }
        }
        IndexKind::IvfFlat => {
            let mut queries: Vec<(String, [f32; VECTOR_DIM as usize])> = (0..3)
                .map(|slot| {
                    let id = (answers.salt * 3 + slot) as i64;
                    (
                        format!("random query {slot}"),
                        vector(!answers.data_seed, id),
                    )
                })
                .collect();
            // A deleted row's own vector: only an index that ignores
            // deletions would answer with that row.
            if let Some(deleted) = (0..answers.next_id).find(|id| !answers.live.contains(id)) {
                queries.push((
                    format!("deleted row {deleted}"),
                    vector(answers.data_seed, deleted),
                ));
            }
            for (position, (label, query)) in queries.iter().enumerate() {
                let mut scan = ds.scan();
                scan.nearest("vec", &Float32Array::from(query.to_vec()), NEIGHBORS)
                    .map_err(unreadable)?;
                // Every partition: IVF_FLAT is then exact.
                scan.minimum_nprobes(IVF_PARTITIONS);
                scan.project(&["id"]).map_err(unreadable)?;
                if position == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("ANNSubIndex: name=idx_vec") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("a nearest neighbor query does not use idx_vec: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                check_neighbors(n, label, query, &int64_column(&batch, "id"), answers)?;
            }
        }
        IndexKind::Inverted => {
            for slot in 0..4 {
                let token = (answers.salt as u64 * 4 + slot) % answers.scale.vocabulary;
                let term = format!("tok{token:04}");
                let query = FullTextSearchQuery::new(term.clone())
                    .with_column("text".to_string())
                    .map_err(unreadable)?
                    .limit(None);
                let mut scan = ds.scan();
                scan.full_text_search(query).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if slot == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan
                        .lines()
                        .any(|line| line.trim_start().starts_with("MatchQuery: column=text"))
                    {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("a search for {term} does not use idx_text: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = answers
                    .live
                    .iter()
                    .copied()
                    .filter(|id| tokens(*id, &answers.scale).any(|found| found == token))
                    .collect();
                compare_ids(
                    n,
                    &format!("a search for {term}"),
                    &int64_column(&batch, "id"),
                    &expected,
                )?;
            }
        }
    }
    Ok(())
}

/// The answer must be the brute-force nearest live rows, up to ties within
/// float rounding at the cut.
fn check_neighbors(
    n: usize,
    label: &str,
    query: &[f32; VECTOR_DIM as usize],
    found: &[i64],
    answers: &Answers<'_>,
) -> Result<(), Failure> {
    let mut ranked: Vec<(f32, i64)> = answers
        .live
        .iter()
        .map(|id| (squared_l2(query, &vector(answers.data_seed, *id)), *id))
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let k = NEIGHBORS.min(ranked.len());
    let cut = ranked[k - 1].0;
    let tolerance = cut * 1e-4 + 1e-6;
    let required: BTreeSet<i64> = ranked
        .iter()
        .take_while(|(distance, _)| *distance < cut - tolerance)
        .map(|(_, id)| *id)
        .collect();
    let allowed: BTreeSet<i64> = ranked
        .iter()
        .take_while(|(distance, _)| *distance <= cut + tolerance)
        .map(|(_, id)| *id)
        .collect();
    let distinct: BTreeSet<i64> = found.iter().copied().collect();
    if found.len() != k
        || distinct.len() != k
        || !required.is_subset(&distinct)
        || !distinct.is_subset(&allowed)
    {
        let expected: Vec<i64> = ranked.iter().take(k).map(|(_, id)| *id).collect();
        return Err(fail(
            "content",
            n,
            format_args!(
                "the nearest neighbors of {label} are {found:?} where brute force finds {expected:?}"
            ),
        ));
    }
    Ok(())
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

fn is_resources_exhausted(message: &str) -> bool {
    message.contains("Resources exhausted")
}

/// Whether an error says that a commit lost to another one.
fn is_commit_conflict(message: &str) -> bool {
    [
        "Commit conflict for version",
        "Retryable commit conflict for version",
        "Incompatible transaction",
    ]
    .iter()
    .any(|conflict| message.contains(conflict))
}

/// A defect of Lance, not a pool too small for the work. The scan of the fragments
/// an index does not cover reaches the sort of an indexed merge insert through a
/// `RepartitionExec`, which counts as a spilling consumer of the fair pool. The sort
/// is then left with less than the two batches of an eighth of the pool each that
/// `HardCapBatchSizeExec` lets through, at any pool size.
fn is_merge_insert_pool_share_defect(op: &Op, message: &str) -> bool {
    matches!(op, Op::MergeInsert { .. })
        && message.contains("RepartitionExec[")
        && message.contains("for ExternalSorter[")
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

struct Run {
    kind: ScenarioKind,
    plan: Plan,
    generation: u64,
    /// The first step this process has to run; `resume` moves it.
    next_step: usize,
    /// The seed the checks recompute vectors from; a self-test changes it.
    oracle_seed: u64,
    uri: String,
    /// The object path of the dataset root.
    root: String,
    tmp: PathBuf,
    journal: Arc<Journal>,
    state: Arc<IncidentState>,
    hooks: Arc<StressHooks>,
    wrapper: Arc<StressWrapper>,
    /// Where this process lives, under a scenario with memory limits.
    cgroup: Option<OwnCgroup>,
    /// The heap held after each cycle of `heap_growth`.
    heap_after_cycle: Vec<i64>,
    /// The handle every operation runs on; reads that check it go through a
    /// fresh handle without the wrapper.
    ds: Dataset,
    /// The latest commit is one `ds` never saw: its operation committed but
    /// did not return Ok.
    handle_stale: bool,
    model: Model,
    /// A cleanup was allowed to remove every version before this one, so such a
    /// version may be gone, or listed without some of its files when a cleanup
    /// stopped half way.
    expired_below: u64,
    /// What each version held when it was the latest, as `ids_digest` puts it.
    digests: BTreeMap<u64, String>,
    /// The expired versions already reported without some of their files.
    broken_reported: BTreeSet<u64>,
    /// What the cleanup about to run must remove and must leave.
    cleanup_files: Option<CleanupFiles>,
    /// The fragments of a stale handle that an index optimization starts on:
    /// those are what it covers.
    read_fragments: Option<RoaringBitmap>,
    /// What `concurrency` knows between its rounds.
    rounds: Option<concurrency::RoundsState>,
    baseline_tasks: usize,
    baseline_fds: usize,
    quiet: Duration,
    self_test: Option<SelfTest>,
}

impl Run {
    async fn setup(
        setting: ChildSetting,
        plan: Plan,
        dir: &Path,
        journal: Arc<Journal>,
    ) -> Result<Self, Failure> {
        let ChildSetting {
            kind,
            quiet,
            generation,
            self_test,
            started,
        } = setting;
        let (uri, root) = match plan.store {
            StoreChoice::Local | StoreChoice::FileObjectStore => {
                let path = dir
                    .join("dataset")
                    .to_str()
                    .expect("a UTF-8 artifacts path")
                    .to_string();
                let root = path.trim_start_matches('/').to_string();
                match plan.store {
                    StoreChoice::Local => (path, root),
                    _ => (format!("file-object-store://{path}"), root),
                }
            }
            StoreChoice::SharedMemory => (
                format!("shared-memory://stress-{}/ds", std::process::id()),
                "ds".to_string(),
            ),
        };
        let state = Arc::new(IncidentState::new(
            root.clone(),
            &dir.join(CALLS_LOG),
            started,
        ));
        let hooks = Arc::new(StressHooks::new(state.clone()));
        if kind.observes_memory() {
            log_memory_events(&dir.join(MEMORY_LOG), started, generation, state.clone())
                .map_err(|error| fail("harness", 0, error))?;
        }
        journal.line(format_args!(
            "setup store={:?} uri={uri} rows={} generation={generation}",
            plan.store, plan.initial_rows
        ));
        let wrapper = Arc::new(StressWrapper::new(state.clone()));
        let store_params = ObjectStoreParams {
            object_store_wrapper: Some(wrapper.clone()),
            ..Default::default()
        };
        let rounds = plan.rounds.as_ref().map(|rounds| {
            rounds.start(
                plan.initial_rows,
                Journal::open(&dir.join(READS_LOG), started),
            )
        });
        let ds = if generation == 1 {
            let params = WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: plan.scale.rows_per_file,
                store_params: Some(store_params),
                progress: hooks.clone(),
                enable_stable_row_ids: plan
                    .rounds
                    .as_ref()
                    .is_some_and(|rounds| rounds.stable_row_ids)
                    || plan.soak.as_ref().is_some_and(|soak| soak.stable_row_ids),
                ..Default::default()
            };
            let initial: Box<dyn RecordBatchReader + Send> = match (&plan.rounds, &plan.soak) {
                (Some(rounds), _) => {
                    Box::new(rounds.initial_source(plan.data_seed, plan.initial_rows, plan.scale))
                }
                (None, Some(_)) => Box::new(soak::initial_source(
                    plan.data_seed,
                    plan.initial_rows,
                    plan.scale,
                )),
                (None, None) => Box::new(rows(plan.data_seed, 0..plan.initial_rows, plan.scale)),
            };
            Box::pin(Dataset::write(initial, uri.as_str(), Some(params))).await
        } else {
            Box::pin(
                DatasetBuilder::from_uri(&uri)
                    .with_store_params(store_params)
                    .load(),
            )
            .await
        }
        // A dead process must not take the dataset with it.
        .map_err(|error| {
            let class = if generation == 1 {
                "setup"
            } else {
                "unreadable"
            };
            fail(class, 0, error)
        })?;
        let mut run = Self {
            kind,
            generation,
            next_step: 0,
            oracle_seed: plan.data_seed,
            model: Model::new(plan.initial_rows),
            plan,
            uri,
            root,
            tmp: dir.join("tmp"),
            journal,
            state,
            hooks,
            wrapper,
            cgroup: OwnCgroup::from_env().map_err(|error| fail("harness", 0, error))?,
            heap_after_cycle: Vec::with_capacity(HEAP_CYCLES),
            ds,
            handle_stale: false,
            expired_below: 0,
            digests: BTreeMap::new(),
            broken_reported: BTreeSet::new(),
            cleanup_files: None,
            read_fragments: None,
            rounds,
            baseline_tasks: 0,
            baseline_fds: 0,
            quiet,
            self_test,
        };
        if generation == 1 {
            Box::pin(run.measure_baseline(0)).await?;
        }
        Ok(run)
    }

    /// What the runtime and the process keep between operations, measured
    /// after a first read, which starts the pools that stay.
    async fn measure_baseline(&mut self, n: usize) -> Result<(), Failure> {
        Box::pin(self.check_content(n)).await?;
        let (snapshot, waited) = quiesce(0, &self.tmp, self.quiet).await;
        self.baseline_tasks = snapshot.tasks;
        self.baseline_fds = snapshot.fds;
        self.journal.line(format_args!(
            "setup baseline waited_ms={} {}",
            waited.as_millis(),
            snapshot.line()
        ));
        self.check_leaks(n, &snapshot)
    }

    /// Picks up the seed after an earlier process died, and first of all looks
    /// at what it left: the dataset must read as if the death was a clean end
    /// of the step it interrupted.
    async fn resume(&mut self) -> Result<(), Failure> {
        if self.generation == 1 {
            return Ok(());
        }
        let log = fs::read_to_string(self.seed_dir().join(STEPS_LOG))
            .map_err(|error| fail("harness", 0, format_args!("read the step log: {error}")))?;
        let point = ResumePoint::from_journal(&log);
        for step in &self.plan.steps[..point.finished] {
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
        let step = self.plan.steps[n - 1].clone();
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
        if self.self_test == Some(SelfTest::KillThenMissingFile) {
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
        Box::pin(self.conclude(n, &step, outcome, before, fired, None)).await?;
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

    /// A local object store writes `name` by way of a file `name#digits`, and
    /// its listings hide those. So a death inside a write can leave a file that
    /// neither a listing nor a cleanup through the store ever sees: only the
    /// file system tells.
    /// Files on the disk that the store does not list, such as a staged upload;
    /// how many there are.
    fn observe_hidden_files(
        &self,
        n: usize,
        observation: &str,
        listed: &BTreeMap<String, i64>,
    ) -> usize {
        let dataset = self.seed_dir().join("dataset");
        let hidden: Vec<PathBuf> = files_below(&dataset)
            .into_iter()
            .filter(|file| {
                file.strip_prefix(&dataset)
                    .ok()
                    .and_then(Path::to_str)
                    .is_some_and(|path| !listed.contains_key(path))
            })
            .collect();
        if hidden.is_empty() {
            return 0;
        }
        let bytes: u64 = hidden
            .iter()
            .filter_map(|file| fs::metadata(file).ok())
            .map(|meta| meta.len())
            .sum();
        self.journal.line(format_args!(
            "step={n} observation={observation} files={} bytes={bytes} detail={hidden:?}",
            hidden.len()
        ));
        hidden.len()
    }

    fn seed_dir(&self) -> &Path {
        self.tmp.parent().expect("the seed's directory")
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

    async fn run_steps(&mut self) -> Result<(), Failure> {
        if self.plan.rounds.is_some() {
            return Box::pin(self.run_rounds()).await;
        }
        if self.plan.soak.is_some() {
            return Box::pin(self.run_soak()).await;
        }
        for index in self.next_step..self.plan.steps.len() {
            let step = self.plan.steps[index].clone();
            Box::pin(self.run_step(index + 1, &step)).await?;
            if self.kind == ScenarioKind::HeapGrowth && (index + 1) % HEAP_CYCLE_OPS == 0 {
                self.end_heap_cycle(index + 1).await;
            }
        }
        if self.kind == ScenarioKind::HeapGrowth {
            self.check_heap_growth()?;
        }
        Ok(())
    }

    /// The handle is new by now, so its session holds nothing of the cycle.
    async fn end_heap_cycle(&mut self, n: usize) {
        if self.self_test == Some(SelfTest::HeapLeak) {
            Box::leak(vec![1u8; 8 << 10].into_boxed_slice());
        }
        quiesce(self.baseline_tasks, &self.tmp, self.quiet).await;
        let live = live_heap_bytes();
        self.heap_after_cycle.push(live);
        self.journal
            .line(format_args!("step={n} heap_cycle live_kib={}", live >> 10));
    }

    fn check_heap_growth(&self) -> Result<(), Failure> {
        let median = |cycles: Range<usize>| {
            let mut values = self.heap_after_cycle[cycles].to_vec();
            values.sort_unstable();
            values[values.len() / 2]
        };
        let early = median(HEAP_EARLY_CYCLES);
        let late = median(HEAP_LATE_CYCLES);
        if late - early > HEAP_GROWTH_LIMIT {
            return Err(fail(
                "heap_growth",
                self.plan.steps.len(),
                format_args!(
                    "the heap grew from {early} to {late} bytes over cycles of the same work: {:?}",
                    self.heap_after_cycle
                ),
            ));
        }
        Ok(())
    }

    /// The incident a step's verdict counts under.
    fn incident_label(&self, step: &Step) -> String {
        if let Some(limit) = &step.limit {
            return format!("memory:{}:{limit}", step.op.kind());
        }
        let kind = step
            .incident
            .as_ref()
            .map_or_else(|| "none".to_string(), Incident::kind);
        match self.kind {
            ScenarioKind::ProcessKill => kind.replacen("cancel", "kill", 1),
            _ => kind,
        }
    }

    async fn run_step(&mut self, n: usize, step: &Step) -> Result<(), Failure> {
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
        let death = (self.kind == ScenarioKind::ProcessKill).then(|| {
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
        Box::pin(self.conclude(n, step, outcome, before, fired, right_after)).await
    }

    /// Everything after the operation of step `n`, whichever process runs it:
    /// what the outcome says about the dataset, the model, the checks, the
    /// retry and the leaks.
    async fn conclude(
        &mut self,
        n: usize,
        step: &Step,
        outcome: Outcome,
        before: u64,
        fired: u64,
        right_after: Option<u64>,
    ) -> Result<(), Failure> {
        let incident = self.incident_label(step);
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
        self.let_go_of_the_session(n, Some(&step.op)).await?;
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
        self.check_memory_events(n, step, &outcome, committed)?;
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
            self.observe(
                n,
                match self.kind {
                    ScenarioKind::StorageFaults => "fault_absorbed",
                    ScenarioKind::Cancellation => "cancel_missed",
                    ScenarioKind::ProcessKill => "kill_missed",
                    ScenarioKind::MemoryLimits | ScenarioKind::HeapGrowth => "limit_absorbed",
                    ScenarioKind::Concurrency => "hold_absorbed",
                    ScenarioKind::SlowStorage => "delay_absorbed",
                    ScenarioKind::Soak => unreachable!("a soak runs no steps"),
                },
            );
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
                // A death injects no error, and a slow or hung call none either.
                None => {
                    fired > 0
                        && !matches!(
                            self.kind,
                            ScenarioKind::ProcessKill | ScenarioKind::SlowStorage
                        )
                }
            };
            if !explained {
                // A stale handle may legitimately lose a conflict. A slow store
                // injects no error at all, so there it may lose nothing else.
                let excused = ran_stale
                    && (self.kind != ScenarioKind::SlowStorage || is_commit_conflict(message));
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
            && !matches!(
                self.kind,
                ScenarioKind::HeapGrowth | ScenarioKind::Concurrency
            )
            && self.latest_version(n).await? > before
        {
            Box::pin(self.check_kept_versions(n)).await?;
        }
        self.let_go_of_the_session(n, Some(&step.op)).await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)?;
        // From here on a later process does not touch this step again.
        self.journal.line(format_args!("step={n} done"));
        Ok(())
    }

    /// What Lance logged about the memory of an index build must agree with
    /// what happened: the build, its stages, the limit it ran under, and that
    /// the end of a build is only ever logged after its commit.
    fn check_memory_events(
        &self,
        n: usize,
        step: &Step,
        outcome: &Outcome,
        committed: bool,
    ) -> Result<(), Failure> {
        if !self.kind.observes_memory() || !matches!(step.op, Op::CreateIndex(_)) {
            return Ok(());
        }
        let wrong = |detail: String| Err(fail("memory_events", n, detail));
        let in_this_step = |line: &&str| field(line, "phase") == Some(n.to_string().as_str());
        let log = fs::read_to_string(self.seed_dir().join(MEMORY_LOG)).unwrap_or_default();
        // The last line of a dead process can be torn.
        let events: Vec<&str> = log
            .lines()
            .filter(|line| line.ends_with(" end=1"))
            .filter(in_this_step)
            .collect();
        if let Some(limit) = self.limit_of_step(n) {
            for line in &events {
                let bytes = |name: &str| field(line, name).and_then(|v| v.parse::<u64>().ok());
                if bytes("cgroup_limit_bytes") != Some(limit) {
                    return wrong(format!("the limit of the step is {limit}, but: {line}"));
                }
                // A kernel can charge past the limit, but not by much.
                if bytes("cgroup_bytes").is_none_or(|current| current > limit + (1 << 20)) {
                    return wrong(format!("the limit of the step is {limit}, but: {line}"));
                }
            }
        }
        let of_kind = |name: &'static str| {
            events
                .iter()
                .filter(move |line| field(line, "event") == Some(name))
        };
        if !committed
            && matches!(outcome, Outcome::Killed | Outcome::Failed { .. })
            && let Some(line) = of_kind("operation_complete").next()
        {
            return wrong(format!("an uncommitted build logged its end: {line}"));
        }
        let calls = fs::read_to_string(self.seed_dir().join(CALLS_LOG)).unwrap_or_default();
        let hooks_saw = |name: &str| -> Vec<String> {
            calls
                .lines()
                .filter(in_this_step)
                .filter_map(|line| {
                    let mut tokens = line.split(' ').skip(2);
                    (tokens.next() == Some(name))
                        .then(|| tokens.next().map(str::to_string))
                        .flatten()
                })
                .collect()
        };
        if !outcome.is_ok() {
            // An event goes out right before its hook, so a build that did not
            // end can only have logged one stage more than the hooks saw.
            for name in ["stage_start", "stage_complete"] {
                let logged: Vec<&str> = of_kind(name)
                    .filter_map(|line| field(line, "stage"))
                    .collect();
                let seen = hooks_saw(name);
                if logged.len() < seen.len()
                    || logged.len() > seen.len() + 1
                    || !logged.iter().zip(&seen).all(|(a, b)| a == b)
                {
                    return wrong(format!(
                        "{name}: the events name {logged:?}, the progress hooks saw {seen:?}"
                    ));
                }
            }
            return Ok(());
        }
        let starts: Vec<&&str> = of_kind("operation_start").collect();
        let ends: Vec<&&str> = of_kind("operation_complete").collect();
        let ([start], [end]) = (starts.as_slice(), ends.as_slice()) else {
            return wrong(format!(
                "a finished build logged {} starts and {} ends",
                starts.len(),
                ends.len()
            ));
        };
        if field(start, "operation_id") != field(end, "operation_id") {
            return wrong(format!("the ends of a build differ: {start} and {end}"));
        }
        let peak = |line: &str| field(line, "peak_bytes").and_then(|peak| peak.parse::<u64>().ok());
        let Some(whole_peak) = peak(end).filter(|peak| *peak > 0) else {
            return wrong(format!("a finished build logged no peak: {end}"));
        };
        // The window of the build holds the window of each stage.
        for line in of_kind("stage_complete") {
            if peak(line).is_none_or(|stage_peak| stage_peak == 0 || stage_peak > whole_peak) {
                return wrong(format!(
                    "the build logged a peak of {whole_peak}, but: {line}"
                ));
            }
        }
        for name in ["stage_start", "stage_complete"] {
            let logged: Vec<&str> = of_kind(name)
                .filter_map(|line| field(line, "stage"))
                .collect();
            let seen = hooks_saw(name);
            if logged != seen {
                return wrong(format!(
                    "{name}: the events name {logged:?}, the progress hooks saw {seen:?}"
                ));
            }
        }
        Ok(())
    }

    /// The limit a step was armed with, from the journal, which a death leaves
    /// behind too.
    fn limit_of_step(&self, n: usize) -> Option<u64> {
        let journal = fs::read_to_string(self.seed_dir().join(STEPS_LOG)).ok()?;
        journal
            .lines()
            .filter(|line| field(line, "step") == Some(n.to_string().as_str()))
            .find(|line| line.split(' ').any(|token| token == "armed"))
            .and_then(|line| field(line, "limit")?.parse().ok())
    }

    /// Replaces the operations' handle with a fresh one, in a session of its
    /// own. An operation that reads through an index leaves it in the cache of
    /// its session, and with it the task of the index store's I/O scheduler
    /// (`ScanScheduler`) until the session goes. Only what outlives the handle
    /// is a leak. The older scenarios keep one session, so that its caches see
    /// everything that follows, but for an index optimization, which reads the
    /// segments it merges through it.
    async fn let_go_of_the_session(&mut self, n: usize, op: Option<&Op>) -> Result<(), Failure> {
        if matches!(
            self.kind,
            ScenarioKind::StorageFaults | ScenarioKind::Cancellation | ScenarioKind::SlowStorage
        ) && !matches!(op, Some(Op::OptimizeIndices))
        {
            return Ok(());
        }
        self.ds = Box::pin(
            DatasetBuilder::from_uri(&self.uri)
                .with_store_params(ObjectStoreParams {
                    object_store_wrapper: Some(self.wrapper.clone()),
                    ..Default::default()
                })
                .load(),
        )
        .await
        .map_err(|error| fail("unreadable", n, error))?;
        self.handle_stale = false;
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
        match self.self_test {
            Some(SelfTest::ForeignKill) => kill_this_process(),
            Some(SelfTest::KillThenMissingFile) => {
                self.journal.line(CRASH_FIRED);
                kill_this_process()
            }
            Some(SelfTest::OomRecovery) => {
                limited(16 << 20)?;
                // Writing every page of it is what goes over the limit.
                std::hint::black_box(vec![1u8; 1 << 30]);
                Err(fail("harness", 2, "1 GiB fit into a budget of 16 MiB"))
            }
            Some(SelfTest::HangUnderLimit) => {
                limited(16 << 20)?;
                std::future::pending::<()>().await;
                Ok(())
            }
            _ => Ok(()),
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
            self.plan.data_seed,
            self.plan.scale,
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

    async fn inject(&mut self, test: SelfTest) {
        self.journal.line(format_args!("self_test={}", test.name()));
        match test {
            SelfTest::Hang => std::future::pending::<()>().await,
            SelfTest::Panic => {
                tokio::spawn(async { panic!("stress self-test: a panic in a detached task") });
            }
            SelfTest::TaskLeak => {
                tokio::spawn(std::future::pending::<()>());
            }
            SelfTest::TmpLeak => {
                fs::write(self.tmp.join("self-test-leak"), b"leak").expect("leak a file");
            }
            // A write in a task that nothing drops meets a call that never answers.
            SelfTest::HungCall => {
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
            }
            // A slow read in a task that nothing drops ends after the operation.
            // Arming again leaves the end of the step's operation in place.
            SelfTest::LateCall => {
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
            }
            // A task that nothing drops waits for a hung read of a data file, and
            // writes once the read is let go.
            SelfTest::WriteAfterRelease => {
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
            }
            SelfTest::ModelMismatch => {
                self.model.live.pop_first();
            }
            SelfTest::ColumnMismatch => self.oracle_seed ^= 1,
            // Planted after the cleanup, which would remove it. The others act
            // before the operation or in a later process.
            SelfTest::Orphan
            | SelfTest::ForeignKill
            | SelfTest::KillThenMissingFile
            | SelfTest::OomRecovery
            | SelfTest::HangUnderLimit
            | SelfTest::MemoryEvents
            | SelfTest::HeapLeak
            | SelfTest::SoakGrowth => {}
            SelfTest::TornSnapshot
            | SelfTest::LostWrite
            | SelfTest::StaleRead
            | SelfTest::SnapshotChanged
            | SelfTest::DuplicateKey
            | SelfTest::Lineage
            | SelfTest::GateHang => unreachable!("{} acts in a round", test.name()),
        }
    }

    async fn catch_up(&mut self, n: usize) -> Result<(), Failure> {
        Box::pin(self.ds.checkout_latest())
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        self.handle_stale = false;
        Ok(())
    }

    fn observe(&self, n: usize, observation: &str) {
        self.journal
            .line(format_args!("step={n} observation={observation}"));
    }

    /// An operation that returned Ok must leave its handle on its own commit.
    fn check_handle(&self, n: usize, op: &Op, handle: u64, version: u64) -> Result<(), Failure> {
        if handle != version {
            return Err(fail(
                "stale_handle",
                n,
                format_args!(
                    "{op} returned Ok, but its handle is at version {handle}, not {version}"
                ),
            ));
        }
        Ok(())
    }

    async fn open_verifier(&self, n: usize) -> Result<Dataset, Failure> {
        Box::pin(Dataset::open(&self.uri))
            .await
            .map_err(|error| fail("unreadable", n, error))
    }

    /// The version a fresh open lands on.
    async fn opened_version(&self, n: usize) -> Result<u64, Failure> {
        Ok(self.open_verifier(n).await?.version().version)
    }

    /// The latest version, as a fresh handle without the wrapper sees it, once
    /// nothing is landing any more: three ways of asking must agree.
    async fn latest_version(&self, n: usize) -> Result<u64, Failure> {
        let verifier = self.open_verifier(n).await?;
        let opened = verifier.version().version;
        let latest = verifier
            .latest_version_id()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let listed = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?
            .iter()
            .map(|version| version.version)
            .max()
            .unwrap_or_default();
        if opened != latest || latest != listed {
            return Err(fail(
                "version_jump",
                n,
                format_args!(
                    "a fresh open lands on version {opened}, latest_version_id says {latest}, and the newest listed manifest is {listed}"
                ),
            ));
        }
        Ok(latest)
    }

    async fn check_content(&mut self, n: usize) -> Result<(), Failure> {
        let unreadable = |error: lance::Error| fail("unreadable", n, error);
        let ds = self.open_verifier(n).await?;
        Box::pin(ds.validate())
            .await
            .map_err(|error| fail("validate", n, error))?;
        self.check_rows(&ds, n).await?;
        let latest = ds.version().version;
        // The rounds of `concurrency` check their history themselves.
        if self.plan.rounds.is_none() {
            let listed = listed_versions(&ds, n).await?;
            let gone: Vec<u64> = (self.expired_below.max(1)..=latest)
                .filter(|version| !listed.contains(version))
                .collect();
            if !gone.is_empty() {
                return Err(fail(
                    "version_jump",
                    n,
                    format_args!(
                        "versions {gone:?} are gone, though cleanups could only remove the versions before {}",
                        self.expired_below
                    ),
                ));
            }
        }
        let all = file_times(ds.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let tracked = files_by_version(ds.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let mut missing = Vec::new();
        let mut broken = BTreeSet::new();
        for (path, versions) in tracked.iter().filter(|(path, _)| !all.contains_key(*path)) {
            // A cleanup deletes the files of the versions it expires before
            // their manifests, so one that stopped half way leaves such a version
            // listed without some of them.
            match versions.last() {
                Some(newest) if *newest >= self.expired_below => {
                    missing.push(format!("{path} of versions {versions:?}"));
                }
                _ => broken.extend(versions),
            }
        }
        if !missing.is_empty() {
            return Err(fail(
                "missing_files",
                n,
                format_args!("manifests reference files that are not there: {missing:?}"),
            ));
        }
        let newly_broken: Vec<String> = broken
            .difference(&self.broken_reported)
            .map(u64::to_string)
            .collect();
        if !newly_broken.is_empty() {
            self.journal.line(format_args!(
                "step={n} observation=expired_version_broken versions={}",
                newly_broken.join(",")
            ));
            self.broken_reported.extend(broken);
        }
        let indices = ds.load_indices().await.map_err(unreadable)?;
        let names: BTreeSet<String> = indices.iter().map(|index| index.name.clone()).collect();
        let expected: BTreeSet<String> = self
            .model
            .indices
            .iter()
            .map(|kind| kind.name().to_string())
            .collect();
        if names != expected {
            return Err(fail(
                "content",
                n,
                format_args!("the indices are {names:?} where the model has {expected:?}"),
            ));
        }
        for kind in &self.model.indices {
            let answers = Answers {
                live: &self.model.live,
                next_id: self.model.next_id,
                stamps: self.rounds.as_ref().map(|rounds| rounds.stamps()),
                stamps_ever: self.rounds.as_ref().map(|rounds| rounds.stamps_ever()),
                few_stamps: false,
                scale: self.plan.scale,
                data_seed: self.oracle_seed,
                // Under concurrency an index can legitimately cover nothing by now:
                // an update in place takes its fragments away from it.
                check_plan: (self.plan.rounds.is_none() && self.plan.soak.is_none())
                    || concurrency::covers_a_fragment(&ds, &indices, kind.name()),
                salt: n,
            };
            Box::pin(check_index(&ds, n, *kind, &answers)).await?;
        }
        // Only for the versions a cleanup keeps: in `heap_growth` and `soak` the
        // map would count as growth of the heap.
        if !matches!(
            self.kind,
            ScenarioKind::HeapGrowth | ScenarioKind::Concurrency | ScenarioKind::Soak
        ) {
            // The rows of the latest version are the model's by now.
            let digest = ids_digest(self.model.live.iter().copied());
            match self.digests.get(&latest) {
                Some(known) if *known != digest => {
                    return Err(fail(
                        "snapshot_changed",
                        n,
                        format_args!(
                            "version {latest} reads as {digest}, and read as {known} before"
                        ),
                    ));
                }
                Some(_) => {}
                None => {
                    self.journal
                        .line(format_args!("step={n} digest_of={latest} ids={digest}"));
                    self.digests.insert(latest, digest);
                }
            }
        }
        Ok(())
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

    /// Only a failed deletion of its manifest leaves an expired version behind.
    async fn check_expired_gone(
        &self,
        n: usize,
        boundary: u64,
        may_stay: u64,
    ) -> Result<(), Failure> {
        let left: Vec<u64> = listed_versions(&self.open_verifier(n).await?, n)
            .await?
            .into_iter()
            .filter(|version| *version < boundary)
            .collect();
        if left.len() as u64 > may_stay {
            return Err(fail(
                "cleanup",
                n,
                format_args!(
                    "the cleanup returned Ok, but versions {left:?} are still there, below its boundary {boundary}, where its failed deletions allow {may_stay}"
                ),
            ));
        }
        Ok(())
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

    /// Every version the cleanups keep opens, reads as it did when it was the
    /// latest, and has the files of its indices, which `check_content` sees
    /// only by listing their directories.
    async fn check_kept_versions(&self, n: usize) -> Result<(), Failure> {
        let verifier = self.open_verifier(n).await?;
        let all = file_times(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let kept = listed_versions(&verifier, n)
            .await?
            .into_iter()
            .filter(|version| *version >= self.expired_below);
        for version in kept {
            let unreadable = |error: lance::Error| {
                fail("unreadable", n, format_args!("version {version}: {error}"))
            };
            let ds = verifier
                .checkout_version(version)
                .await
                .map_err(unreadable)?;
            let mut scan = ds.scan();
            scan.project(&["id"]).map_err(unreadable)?;
            let batch = scan.try_into_batch().await.map_err(unreadable)?;
            let digest = ids_digest(int64_column(&batch, "id"));
            if let Some(known) = self.digests.get(&version)
                && *known != digest
            {
                return Err(fail(
                    "snapshot_changed",
                    n,
                    format_args!(
                        "version {version} reads as {digest}, and read as {known} when it was the latest"
                    ),
                ));
            }
            for index in ds.load_indices().await.map_err(unreadable)?.iter() {
                let Some(files) = &index.files else {
                    continue;
                };
                let missing: Vec<String> = files
                    .iter()
                    .map(|file| format!("_indices/{}/{}", index.uuid, file.path))
                    .filter(|path| !all.contains_key(path))
                    .collect();
                if !missing.is_empty() {
                    return Err(fail(
                        "missing_files",
                        n,
                        format_args!(
                            "version {version} keeps the index {} without its files {missing:?}",
                            index.name
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Every row must be there once, with every column right.
    async fn check_rows(&self, ds: &Dataset, n: usize) -> Result<(), Failure> {
        let unreadable = |error: lance::Error| fail("unreadable", n, error);
        let batch = ds.scan().try_into_batch().await.map_err(unreadable)?;
        let ids = int64_column(&batch, "id");
        compare_ids(n, "a full scan", &ids, &self.model.live)?;
        let count = ds.count_rows(None).await.map_err(unreadable)?;
        if count != self.model.live.len() {
            return Err(fail(
                "content",
                n,
                format_args!(
                    "count_rows says {count} where the model has {}",
                    self.model.live.len()
                ),
            ));
        }
        let categories = batch
            .column_by_name("category")
            .expect("a category column")
            .as_primitive::<Int32Type>();
        let vectors = batch
            .column_by_name("vec")
            .expect("a vec column")
            .as_fixed_size_list();
        let texts = batch
            .column_by_name("text")
            .expect("a text column")
            .as_string::<i32>();
        for (row, id) in ids.iter().enumerate() {
            let wrong = if categories.value(row) != category(*id) {
                Some("category")
            } else if texts.value(row) != text(*id, &self.plan.scale) {
                Some("text")
            } else if vectors
                .value(row)
                .as_primitive::<Float32Type>()
                .values()
                .as_ref()
                != vector(self.oracle_seed, *id).as_slice()
            {
                Some("vec")
            } else {
                None
            };
            if let Some(column) = wrong {
                return Err(fail(
                    "content",
                    n,
                    format_args!("row {id} holds a wrong {column}"),
                ));
            }
        }
        Ok(())
    }

    async fn quiet_down(&self, n: usize) -> Snapshot {
        let (snapshot, waited) = quiesce(self.baseline_tasks, &self.tmp, self.quiet).await;
        self.journal.line(format_args!(
            "step={n} quiet waited_ms={} {}",
            waited.as_millis(),
            snapshot.line()
        ));
        snapshot
    }

    fn check_leaks(&self, n: usize, snapshot: &Snapshot) -> Result<(), Failure> {
        if snapshot.tasks > self.baseline_tasks {
            return Err(fail(
                "task_leak",
                n,
                format_args!(
                    "{} tokio tasks are alive {}s after the step, {} before the first one",
                    snapshot.tasks,
                    self.quiet.as_secs(),
                    self.baseline_tasks
                ),
            ));
        }
        if !snapshot.tmp_files.is_empty() || snapshot.tmp_dirs > 0 {
            return Err(fail(
                "tmp_leak",
                n,
                format_args!(
                    "temporary files left behind: {:?}, and {} directories",
                    snapshot.tmp_files, snapshot.tmp_dirs
                ),
            ));
        }
        if snapshot.fds > self.baseline_fds {
            return Err(fail(
                "fd_leak",
                n,
                format_args!(
                    "{} open descriptors after the step, {} before the first one",
                    snapshot.fds, self.baseline_fds
                ),
            ));
        }
        Ok(())
    }

    /// Cleans up everything but the latest version, then checks that no
    /// unreferenced file is left and the dataset still reads right.
    async fn finish(&mut self) -> Result<(), Failure> {
        let n = self.plan.steps.len()
            + self
                .plan
                .rounds
                .as_ref()
                .map_or(0, |rounds| rounds.rounds.len())
            + self.plan.soak.as_ref().map_or(0, |_| soak::SAMPLES)
            + 1;
        self.state.set_phase("final");
        self.journal.line(format_args!("step={n} final"));
        self.catch_up(n).await?;
        // Every version but the latest may go now.
        self.expired_below = self.ds.version().version;
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&self.ds, 1)
            .await
            .map_err(|error| fail("cleanup", n, error))?
            .delete_unverified(true)
            .error_if_tagged_old_versions(false)
            .build();
        let removed = self
            .ds
            .cleanup_with_policy(policy)
            .await
            .map_err(|error| fail("cleanup", n, error))?;
        self.journal.line(format_args!(
            "step={n} cleanup old_versions={} data_files={} index_files={} transaction_files={} deletion_files={} failed_deletes={}",
            removed.old_versions,
            removed.data_files_removed,
            removed.index_files_removed,
            removed.transaction_files_removed,
            removed.deletion_files_removed,
            removed.failed_deletes
        ));
        // It runs without an incident or a limit, so nothing explains a failure.
        if removed.failed_deletes > 0 {
            return Err(fail(
                "cleanup",
                n,
                format_args!(
                    "the final cleanup failed {} deletions",
                    removed.failed_deletes
                ),
            ));
        }
        self.check_expired_gone(n, self.expired_below, 0).await?;
        if self.self_test == Some(SelfTest::Orphan) {
            let store = self
                .ds
                .object_store(None)
                .await
                .map_err(|error| fail("unreadable", n, error))?;
            let path = ObjectPath::from(format!("{}/data/self-test-orphan.lance", self.root));
            store
                .inner
                .put_opts(
                    &path,
                    PutPayload::from_static(b"orphan"),
                    PutOptions::default(),
                )
                .await
                .expect("plant an orphan");
        }
        self.check_orphans(n).await?;
        Box::pin(self.check_content(n)).await?;
        // The queries of an index need not touch every file of it.
        Box::pin(self.check_kept_versions(n)).await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)
    }

    /// After a cleanup that may delete unverified files, with nothing in flight,
    /// no file is left that no version references.
    async fn check_orphans(&self, n: usize) -> Result<(), Failure> {
        let verifier = self.open_verifier(n).await?;
        let listed = file_sizes(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let hidden = self.observe_hidden_files(n, "hidden_after_cleanup", &listed);
        // Nothing interrupts the writes of a soak, so none may leave a file behind.
        if self.kind == ScenarioKind::Soak && hidden > 0 {
            return Err(fail(
                "orphans",
                n,
                format_args!(
                    "{hidden} files the store does not list, see observation=hidden_after_cleanup"
                ),
            ));
        }
        // As cleanup takes it: the earliest commit time of the versions it keeps,
        // which a clock that stepped back makes other than that of the oldest one.
        let cutoff = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?
            .iter()
            .map(|version| version.timestamp.timestamp_micros())
            .min()
            .unwrap_or(i64::MAX);
        let all = file_times(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let tracked = paths(verifier.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let mut orphans = Vec::new();
        for (path, modified) in &all {
            // A version hint is rewritten in place rather than referenced.
            if tracked.contains(path) || path.ends_with("latest_version_hint.json") {
                continue;
            }
            match path.split('/').next() {
                // Cleanup lists these only up to that time, to the nanosecond
                // (`listing_unmodified_since` in `cleanup.rs`), and these times
                // are in microseconds.
                Some(area @ ("_transactions" | "_deletions")) if *modified >= cutoff => {
                    self.observe(n, &format!("orphan{area}"));
                }
                _ => orphans.push(path.clone()),
            }
        }
        if !orphans.is_empty() {
            return Err(fail(
                "orphans",
                n,
                format_args!("unreferenced files after the cleanup: {orphans:?}"),
            ));
        }
        // A file in a committed index's directory counts as tracked whether
        // the index lists it or not.
        let indices = verifier
            .load_indices()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        for index in indices.iter() {
            let Some(files) = &index.files else {
                continue;
            };
            let directory = format!("_indices/{}/", index.uuid);
            let listed: BTreeSet<String> = files
                .iter()
                .map(|file| format!("{directory}{}", file.path))
                .collect();
            for path in all
                .keys()
                .filter(|path| path.starts_with(&directory) && !listed.contains(*path))
            {
                self.journal.line(format_args!(
                    "step={n} observation=index_dir_extra path={path}"
                ));
            }
        }
        Ok(())
    }
}

#[test]
fn selection_focusable_operations_are_the_ones_plans_draw() {
    let everything = BTreeSet::new();
    for kind in [
        ScenarioKind::StorageFaults,
        ScenarioKind::Cancellation,
        ScenarioKind::ProcessKill,
        ScenarioKind::MemoryLimits,
        ScenarioKind::SlowStorage,
    ] {
        let mut drawn = BTreeSet::new();
        for seed in 1..=48 {
            // No step drops an index.
            let mut indexed = false;
            for step in Plan::generate(kind, seed, 8, true, &everything).steps {
                assert!(
                    indexed || !matches!(step.op, Op::OptimizeIndices),
                    "{} seed {seed}: an optimization without an index",
                    kind.name()
                );
                indexed |= matches!(step.op, Op::CreateIndex(_));
                drawn.insert(step.op.focus_name());
            }
        }
        let focusable: BTreeSet<&str> = focusable_ops(kind).into_iter().collect();
        assert_eq!(drawn, focusable, "{}", kind.name());
    }
}

/// Every other step of a focused plan runs an operation of the focus, and only
/// those steps meet trouble.
fn check_focus(kind: ScenarioKind, seeds: u64) {
    for op in focusable_ops(kind) {
        let focus = BTreeSet::from([op]);
        let mut focused = 0;
        let mut troubled = 0;
        for seed in 1..=seeds {
            let plan = Plan::generate(kind, seed, 8, true, &focus);
            assert!(
                plan.render(kind, seed, 8)
                    .lines()
                    .next()
                    .is_some_and(|header| header.ends_with(&format!(" focus={op}"))),
                "the focus must be part of the plan and its fingerprint"
            );
            for (index, step) in plan.steps.iter().enumerate() {
                let on_focus = index % 2 == 1 && step.op.focus_name() == op;
                if step.incident.is_some() || step.limit.is_some() {
                    assert!(
                        on_focus,
                        "{} seed {seed} step {}: trouble on {} with a focus on {op}",
                        kind.name(),
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
            kind.name()
        );
        assert!(troubled > 0, "{}: no trouble on {op}", kind.name());
    }
}

#[test]
fn selection_focus_on_storage_faults() {
    check_focus(ScenarioKind::StorageFaults, 16);
}

#[test]
fn selection_focus_on_cancellation() {
    check_focus(ScenarioKind::Cancellation, 16);
}

#[test]
fn selection_focus_on_slow_storage() {
    check_focus(ScenarioKind::SlowStorage, 16);
}

/// A slow step targets calls its operation makes, and slows or hangs a listing or a
/// deletion only before its request, the only way the store wrapper can.
#[test]
fn selection_slow_storage_plans_what_the_store_wrapper_can_do() {
    let everything = BTreeSet::new();
    let (mut delays, mut hangs) = (0, 0);
    for seed in 1..=48 {
        let plan = Plan::generate(ScenarioKind::SlowStorage, seed, 8, true, &everything);
        for step in &plan.steps {
            let Some(Incident::Slow { delays: slow, hang }) = &step.incident else {
                continue;
            };
            let calls = slow_calls(&step.op, &plan.scale, plan.store);
            let made = |method: Method, area: Area, nth: u64| {
                calls
                    .iter()
                    .any(|&(m, a, highest)| m == method && a == area && nth <= highest)
            };
            assert!((1..=3).contains(&slow.len()), "seed {seed}: {slow:?}");
            for delay in slow {
                assert!(
                    made(delay.method, delay.area, delay.nth),
                    "seed {seed}: {delay:?}"
                );
                assert!(
                    !delay.after || !matches!(delay.method, Method::List | Method::Delete),
                    "seed {seed}: {delay:?}"
                );
            }
            if let Some(hang) = hang {
                assert!(
                    made(hang.method, hang.area, hang.nth),
                    "seed {seed}: {hang:?}"
                );
                assert!(
                    !hang.after || hang.method == Method::Put,
                    "seed {seed}: {hang:?}"
                );
                hangs += 1;
            }
            delays += slow.len();
        }
    }
    assert!(delays > 0 && hangs > 0, "{delays} delays and {hangs} hangs");
}

#[test]
fn selection_focus_on_process_kill() {
    check_focus(ScenarioKind::ProcessKill, 16);
}

#[test]
fn selection_focus_on_memory_limits() {
    // Its plans hold tens of thousands of rows.
    check_focus(ScenarioKind::MemoryLimits, 8);
}
