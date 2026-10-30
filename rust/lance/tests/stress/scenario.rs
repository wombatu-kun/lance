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
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance::index::{DatasetIndexExt, IndexBuildProgress};
use lance_index::IndexType;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, FullTextSearchQuery, ScalarIndexParams};
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use object_store::path::Path as ObjectPath;
use object_store::{PutOptions, PutPayload};
use tokio::sync::Notify;

use crate::harness::{
    CALLS_LOG, ENV_CHILD, ENV_DIR, ENV_PARENT_PID, ENV_PLAN, ENV_QUIET_SECS, ENV_SEED,
    ENV_SELF_TEST, ENV_STANDALONE, ENV_STEPS, PLAN_LOG, STEPS_LOG, ScenarioKind, SelfTest,
};
use crate::incidents::{
    Area, CancelWhen, Incident, IncidentState, Method, StressHooks, StressWrapper,
};
use crate::leaks::{Snapshot, quiesce};

const ROWS_PER_FILE: usize = 200;
const MIN_LIVE_ROWS: usize = 300;
const VECTOR_DIM: i32 = 8;
const IVF_PARTITIONS: usize = 2;
const NEIGHBORS: usize = 10;
const VOCABULARY: u64 = 16;
const CATEGORIES: i32 = 7;

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
    let self_test = std::env::var(ENV_SELF_TEST).ok().map(|name| {
        SelfTest::from_name(&name).unwrap_or_else(|| panic!("unknown self-test {name:?}"))
    });

    let journal = Arc::new(Journal::open(&dir.join(STEPS_LOG), started));
    // A self-test arms no incident, so that only the injected defect can fail it.
    let plan = Plan::generate(kind, seed, steps, self_test.is_none());
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
        _ => run_on_runtime(kind, plan, &dir, journal.clone(), quiet, self_test, started),
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

fn run_on_runtime(
    kind: ScenarioKind,
    plan: Plan,
    dir: &Path,
    journal: Arc<Journal>,
    quiet: Duration,
    self_test: Option<SelfTest>,
    started: Instant,
) -> Result<(), Failure> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build the runtime");
    // A panic outside an operation, in a check for example, must still end in
    // a verdict. Dropping the runtime while unwinding would wait, without a
    // limit, for any thread that panic has wedged.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(async {
            let mut run = Box::pin(Run::setup(
                kind, plan, dir, journal, quiet, self_test, started,
            ))
            .await?;
            Box::pin(run.run_steps()).await?;
            Box::pin(run.finish()).await
        })
    }));
    // A self-test leaves a task behind on purpose; do not wait for it.
    runtime.shutdown_timeout(Duration::from_secs(5));
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
/// progress at once and a killed child leaves no half line.
struct Journal {
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

    fn line(&self, line: impl fmt::Display) {
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

fn tokens(id: i64) -> [u64; 3] {
    std::array::from_fn(|slot| splitmix64(((id as u64) << 2) | slot as u64) % VOCABULARY)
}

fn text(id: i64) -> String {
    tokens(id).map(|token| format!("tok{token:02}")).join(" ")
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

fn rows(data_seed: u64, ids: Range<i64>) -> impl RecordBatchReader + Send + 'static {
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
            ids.iter().map(|id| text(*id)),
        )),
    ];
    let batch = RecordBatch::try_new(schema(), columns).expect("build a batch");
    RecordBatchIterator::new(vec![Ok(batch)], schema())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum IndexKind {
    BTree,
    IvfFlat,
    Inverted,
}

impl IndexKind {
    fn name(self) -> &'static str {
        match self {
            Self::BTree => "idx_category",
            Self::IvfFlat => "idx_vec",
            Self::Inverted => "idx_text",
        }
    }

    fn column(self) -> &'static str {
        match self {
            Self::BTree => "category",
            Self::IvfFlat => "vec",
            Self::Inverted => "text",
        }
    }

    /// The stages the build reports to its progress hooks.
    fn stages(self) -> &'static [&'static str] {
        match self {
            Self::BTree => &["load_data"],
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
    Append { first_id: i64, rows: i64 },
    Delete { lo: i64, hi: i64 },
    CreateIndex(IndexKind),
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Append { first_id, rows } => {
                write!(f, "append:ids={first_id}..{}", first_id + rows)
            }
            Self::Delete { lo, hi } => write!(f, "delete:ids={lo}..{hi}"),
            Self::CreateIndex(kind) => write!(f, "create_index:{}", kind.name()),
        }
    }
}

/// What the dataset must hold: the live row ids and the indices.
#[derive(Clone, Debug)]
struct Model {
    live: BTreeSet<i64>,
    next_id: i64,
    indices: BTreeSet<IndexKind>,
}

impl Model {
    fn new(rows: i64) -> Self {
        Self {
            live: (0..rows).collect(),
            next_id: rows,
            indices: BTreeSet::new(),
        }
    }

    fn apply(&mut self, op: &Op) {
        match *op {
            Op::Append { first_id, rows } => {
                self.live.extend(first_id..first_id + rows);
                self.next_id = first_id + rows;
            }
            Op::Delete { lo, hi } => self.live.retain(|id| !(lo..hi).contains(id)),
            Op::CreateIndex(kind) => {
                self.indices.insert(kind);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StoreChoice {
    Local,
    SharedMemory,
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
}

/// Everything a seed decides. Every operation ends up committed once, by its
/// first try or by the retry, so the plan can track the model on its own.
struct Plan {
    store: StoreChoice,
    data_seed: u64,
    initial_rows: i64,
    steps: Vec<Step>,
}

impl Plan {
    fn generate(kind: ScenarioKind, seed: u64, steps: usize, with_incidents: bool) -> Self {
        let mut rng = PlanRng(seed);
        let store = match kind {
            ScenarioKind::StorageFaults => StoreChoice::SharedMemory,
            ScenarioKind::Cancellation if rng.chance(0.5) => StoreChoice::Local,
            ScenarioKind::Cancellation => StoreChoice::SharedMemory,
        };
        let data_seed = rng.next();
        let initial_rows = rng.range(600..=1200) as i64;
        let mut model = Model::new(initial_rows);
        let mut planned = Vec::with_capacity(steps);
        for _ in 0..steps {
            let op = random_op(&mut rng, &model);
            model.apply(&op);
            let incident = random_incident(&mut rng, kind, store, &op);
            let resync = rng.chance(0.7) || matches!(op, Op::Delete { .. });
            planned.push(Step {
                op,
                incident: incident.filter(|_| with_incidents),
                resync,
            });
        }
        Self {
            store,
            data_seed,
            initial_rows,
            steps: planned,
        }
    }

    fn render(&self, kind: ScenarioKind, seed: u64, steps: usize) -> String {
        let mut text = format!(
            "scenario={} seed={seed} steps={steps} store={:?} data_seed={} initial_rows={}\n",
            kind.name(),
            self.store,
            self.data_seed,
            self.initial_rows
        );
        for (index, step) in self.steps.iter().enumerate() {
            let incident = step
                .incident
                .as_ref()
                .map_or_else(|| "none".to_string(), Incident::to_string);
            writeln!(
                text,
                "step={} op={} incident={incident} resync={}",
                index + 1,
                step.op,
                step.resync
            )
            .unwrap();
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
    fingerprint(&Plan::generate(kind, seed, steps, true).render(kind, seed, steps))
}

fn random_op(rng: &mut PlanRng, model: &Model) -> Op {
    let roll = rng.range(0..=99);
    if (35..55).contains(&roll) && model.live.len() > MIN_LIVE_ROWS + 1 {
        let live: Vec<i64> = model.live.iter().copied().collect();
        let max_len = (live.len() / 10).min(live.len() - MIN_LIVE_ROWS).max(1);
        let len = rng.range(1..=max_len as u64) as usize;
        let start = rng.range(0..=(live.len() - len) as u64) as usize;
        return Op::Delete {
            lo: live[start],
            hi: live[start + len - 1] + 1,
        };
    }
    match roll {
        55..70 => Op::CreateIndex(IndexKind::BTree),
        70..85 => Op::CreateIndex(IndexKind::IvfFlat),
        85..100 => Op::CreateIndex(IndexKind::Inverted),
        _ => Op::Append {
            first_id: model.next_id,
            rows: rng.range(100..=400) as i64,
        },
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

fn random_incident(
    rng: &mut PlanRng,
    kind: ScenarioKind,
    store: StoreChoice,
    op: &Op,
) -> Option<Incident> {
    match kind {
        ScenarioKind::StorageFaults => {
            if !rng.chance(0.75) {
                return None;
            }
            // (method, area, the highest call to fail): calls each operation makes.
            let targets: &[(Method, Area, u64)] = match *op {
                Op::Append { rows, .. } => &[
                    // One data file per fragment.
                    (
                        Method::Put,
                        Area::Data,
                        (rows as u64).div_ceil(ROWS_PER_FILE as u64),
                    ),
                    (Method::Put, Area::Transactions, 1),
                    (Method::Put, Area::Manifests, 1),
                    (Method::List, Area::Manifests, 1),
                ],
                Op::Delete { .. } => &[
                    (Method::Get, Area::Data, 3),
                    (Method::Put, Area::Deletions, 1),
                    (Method::Put, Area::Transactions, 1),
                    (Method::Put, Area::Manifests, 1),
                    (Method::List, Area::Manifests, 1),
                ],
                Op::CreateIndex(_) => &[
                    (Method::Get, Area::Data, 3),
                    (Method::Put, Area::Indices, 2),
                    (Method::Put, Area::Transactions, 1),
                    (Method::Put, Area::Manifests, 1),
                    (Method::List, Area::Manifests, 1),
                ],
            };
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
        ScenarioKind::Cancellation => {
            if !rng.chance(0.85) {
                return None;
            }
            // The manifest write passes the wrapper on every store. Only a local
            // store keeps a request in flight, in its blocking pool.
            let flights: &[CancelWhen] = match store {
                StoreChoice::Local => &[CancelWhen::Before, CancelWhen::During, CancelWhen::After],
                StoreChoice::SharedMemory => &[CancelWhen::Before, CancelWhen::After],
            };
            let mut points: Vec<Incident> = flights
                .iter()
                .map(|when| cancel_call(Method::Put, Area::Manifests, 1, *when))
                .collect();
            match *op {
                Op::Append { rows, .. } => {
                    let fragments = (rows as u64).div_ceil(ROWS_PER_FILE as u64);
                    for nth in 1..=fragments {
                        points.push(Incident::CancelFragment {
                            nth,
                            at_begin: true,
                        });
                        // A write reports `complete` only for files closed at the
                        // row limit, never for its last one.
                        if nth < fragments || (rows as u64).is_multiple_of(ROWS_PER_FILE as u64) {
                            points.push(Incident::CancelFragment {
                                nth,
                                at_begin: false,
                            });
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
                Op::Delete { .. } => {}
            }
            // A local store writes data, index, transaction and deletion files
            // around the wrapper, so only a shared-memory store sees those.
            if store == StoreChoice::SharedMemory {
                let written = match op {
                    Op::Append { .. } => Area::Data,
                    Op::Delete { .. } => Area::Deletions,
                    Op::CreateIndex(_) => Area::Indices,
                };
                for area in [written, Area::Transactions] {
                    for when in [CancelWhen::Before, CancelWhen::After] {
                        points.push(cancel_call(Method::Put, area, 1, when));
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

enum Outcome {
    Ok,
    Failed {
        message: String,
        status_unknown: bool,
    },
    Cancelled,
    Panicked(String),
}

impl Outcome {
    fn name(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Failed { .. } => "err",
            Self::Cancelled => "cancelled",
            Self::Panicked(_) => "panic",
        }
    }

    fn detail(&self) -> &str {
        match self {
            Self::Failed { message, .. } | Self::Panicked(message) => message,
            Self::Ok | Self::Cancelled => "",
        }
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

/// The operation of a step on the handle that meets the incidents.
fn op_future<'a>(
    ds: &'a mut Dataset,
    hooks: &Arc<StressHooks>,
    data_seed: u64,
    op: Op,
) -> BoxFuture<'a, lance::Result<()>> {
    let hooks = hooks.clone();
    match op {
        Op::Append {
            first_id,
            rows: count,
        } => Box::pin(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: ROWS_PER_FILE,
                progress: hooks,
                ..Default::default()
            };
            Box::pin(ds.append(rows(data_seed, first_id..first_id + count), Some(params))).await
        }),
        Op::Delete { lo, hi } => Box::pin(async move {
            let predicate = format!("id >= {lo} AND id < {hi}");
            Box::pin(ds.delete(&predicate)).await.map(|_| ())
        }),
        Op::CreateIndex(kind) => Box::pin(async move {
            let columns = [kind.column()];
            let progress: Arc<dyn IndexBuildProgress> = hooks;
            let name = kind.name().to_string();
            match kind {
                IndexKind::BTree => {
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
                    let params = InvertedIndexParams::default();
                    ds.create_index_builder(&columns, IndexType::Inverted, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
            }
            .map(|_| ())
        }),
    }
}

struct Run {
    kind: ScenarioKind,
    plan: Plan,
    /// The seed the checks recompute vectors from; a self-test changes it.
    oracle_seed: u64,
    uri: String,
    /// The object path of the dataset root.
    root: String,
    tmp: PathBuf,
    journal: Arc<Journal>,
    state: Arc<IncidentState>,
    hooks: Arc<StressHooks>,
    /// The handle every operation runs on; reads that check it go through a
    /// fresh handle without the wrapper.
    ds: Dataset,
    /// The latest commit is one `ds` never saw: its operation committed but
    /// did not return Ok.
    handle_stale: bool,
    model: Model,
    baseline_tasks: usize,
    baseline_fds: usize,
    quiet: Duration,
    self_test: Option<SelfTest>,
}

impl Run {
    async fn setup(
        kind: ScenarioKind,
        plan: Plan,
        dir: &Path,
        journal: Arc<Journal>,
        quiet: Duration,
        self_test: Option<SelfTest>,
        started: Instant,
    ) -> Result<Self, Failure> {
        let (uri, root) = match plan.store {
            StoreChoice::Local => {
                let uri = dir
                    .join("dataset")
                    .to_str()
                    .expect("a UTF-8 artifacts path")
                    .to_string();
                let root = uri.trim_start_matches('/').to_string();
                (uri, root)
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
        journal.line(format_args!(
            "setup store={:?} uri={uri} rows={}",
            plan.store, plan.initial_rows
        ));
        let params = WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: ROWS_PER_FILE,
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(StressWrapper::new(state.clone()))),
                ..Default::default()
            }),
            progress: hooks.clone(),
            ..Default::default()
        };
        let initial = rows(plan.data_seed, 0..plan.initial_rows);
        let ds = Box::pin(Dataset::write(initial, uri.as_str(), Some(params)))
            .await
            .map_err(|error| fail("setup", 0, error))?;
        let mut run = Self {
            kind,
            oracle_seed: plan.data_seed,
            model: Model::new(plan.initial_rows),
            plan,
            uri,
            root,
            tmp: dir.join("tmp"),
            journal,
            state,
            hooks,
            ds,
            handle_stale: false,
            baseline_tasks: 0,
            baseline_fds: 0,
            quiet,
            self_test,
        };
        Box::pin(run.check_content(0)).await?;
        // What the runtime and the process keep between operations.
        let (snapshot, waited) = quiesce(0, &run.tmp, run.quiet).await;
        run.baseline_tasks = snapshot.tasks;
        run.baseline_fds = snapshot.fds;
        run.journal.line(format_args!(
            "setup baseline waited_ms={} {}",
            waited.as_millis(),
            snapshot.line()
        ));
        run.check_leaks(0, &snapshot)?;
        Ok(run)
    }

    async fn run_steps(&mut self) -> Result<(), Failure> {
        for index in 0..self.plan.steps.len() {
            let step = self.plan.steps[index].clone();
            Box::pin(self.run_step(index + 1, &step)).await?;
        }
        Ok(())
    }

    async fn run_step(&mut self, n: usize, step: &Step) -> Result<(), Failure> {
        let incident = step
            .incident
            .as_ref()
            .map_or_else(|| "none".to_string(), Incident::kind);
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
        let cancel = self.state.arm(step.incident.clone());
        let outcome = self.execute(step.op, Some(cancel)).await;
        let fired = self.state.disarm();
        self.journal.line(format_args!(
            "step={n} op_end outcome={} fired={fired} detail={}",
            outcome.name(),
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
        self.quiet_down(n).await;
        let after = self.latest_version(n).await?;
        if after < before || after > before + 1 {
            return Err(fail(
                "version_jump",
                n,
                format_args!("one operation moved the version from {before} to {after}"),
            ));
        }
        let committed = after == before + 1;
        self.journal.line(format_args!(
            "step={n} verdict incident={incident} fired={fired} outcome={} committed={committed} before={before} after={after}",
            outcome.name()
        ));
        match (&outcome, committed) {
            (Outcome::Ok, false) => {
                return Err(fail(
                    "lost_commit",
                    n,
                    format_args!(
                        "{} returned Ok, but the latest version is still {before}",
                        step.op
                    ),
                ));
            }
            (Outcome::Ok, true) if fired > 0 => self.observe(
                n,
                match self.kind {
                    ScenarioKind::StorageFaults => "fault_absorbed",
                    ScenarioKind::Cancellation => "cancel_missed",
                },
            ),
            // A caller that sees an error retries, and a retried append
            // duplicates its rows. Only an honest "status unknown" may land.
            (
                Outcome::Failed {
                    message,
                    status_unknown: false,
                },
                true,
            ) => {
                return Err(fail(
                    "error_after_commit",
                    n,
                    format_args!(
                        "{} returned an error, but its commit landed: {message}",
                        step.op
                    ),
                ));
            }
            (
                Outcome::Failed {
                    status_unknown: true,
                    ..
                },
                true,
            ) => {
                self.observe(n, "status_unknown_landed");
            }
            (
                Outcome::Failed {
                    status_unknown: true,
                    ..
                },
                false,
            ) => {
                self.observe(n, "status_unknown_lost");
            }
            (Outcome::Cancelled, true) => {
                self.observe(n, "cancel_after_commit");
                if right_after == Some(before) {
                    self.observe(n, "late_commit");
                }
            }
            _ => {}
        }
        if let (Outcome::Failed { message, .. }, 0) = (&outcome, fired) {
            // A stale handle may legitimately lose a conflict.
            if !self.handle_stale {
                return Err(fail(
                    "error_without_fault",
                    n,
                    format_args!("{} failed with no incident firing: {message}", step.op),
                ));
            }
            self.observe(n, "error_on_stale_handle");
        }
        if step.incident.is_some() && fired == 0 {
            self.observe(n, "not_reached");
        }
        if matches!(outcome, Outcome::Ok) {
            self.check_handle(n, &step.op, after)?;
            self.handle_stale = false;
        } else if committed {
            self.handle_stale = true;
        }
        if committed {
            self.model.apply(&step.op);
        }
        Box::pin(self.check_content(n)).await?;
        if !committed {
            self.state.set_phase(format!("{n}.retry"));
            self.catch_up(n).await?;
            let retry = self.execute(step.op, None).await;
            let retried = self.latest_version(n).await?;
            if !matches!(retry, Outcome::Ok) || retried != after + 1 {
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
            self.check_handle(n, &step.op, retried)?;
            self.journal
                .line(format_args!("step={n} retry outcome=ok version={retried}"));
            self.model.apply(&step.op);
            Box::pin(self.check_content(n)).await?;
        }
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)
    }

    /// Runs `op` on the handle that meets the incidents. When `cancel` fires,
    /// the operation is dropped at whatever await it has reached.
    async fn execute(&mut self, op: Op, cancel: Option<Arc<Notify>>) -> Outcome {
        let cancel = cancel.unwrap_or_default();
        let operation = AssertUnwindSafe(op_future(
            &mut self.ds,
            &self.hooks,
            self.plan.data_seed,
            op,
        ))
        .catch_unwind();
        let result = tokio::select! {
            biased;
            () = cancel.notified() => return Outcome::Cancelled,
            result = operation => result,
        };
        match result {
            Ok(Ok(())) => Outcome::Ok,
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
            SelfTest::ModelMismatch => {
                self.model.live.pop_first();
            }
            SelfTest::ColumnMismatch => self.oracle_seed ^= 1,
            // Planted after the cleanup, which would remove it.
            SelfTest::Orphan => {}
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
    fn check_handle(&self, n: usize, op: &Op, version: u64) -> Result<(), Failure> {
        let handle = self.ds.version().version;
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

    async fn check_content(&self, n: usize) -> Result<(), Failure> {
        let unreadable = |error: lance::Error| fail("unreadable", n, error);
        let ds = self.open_verifier(n).await?;
        Box::pin(ds.validate())
            .await
            .map_err(|error| fail("validate", n, error))?;
        self.check_rows(&ds, n).await?;
        let all = file_times(ds.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let tracked = paths(ds.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let missing: Vec<&String> = tracked
            .iter()
            .filter(|path| !all.contains_key(*path))
            .collect();
        if !missing.is_empty() {
            return Err(fail(
                "missing_files",
                n,
                format_args!("manifests reference files that are not there: {missing:?}"),
            ));
        }
        let names: BTreeSet<String> = ds
            .load_indices()
            .await
            .map_err(unreadable)?
            .iter()
            .map(|index| index.name.clone())
            .collect();
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
            Box::pin(self.check_index(&ds, n, *kind)).await?;
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
            } else if texts.value(row) != text(*id) {
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

    /// Queries an index and compares the answers with the model. The first
    /// query of each index checks its plan, since a query that falls back to a
    /// scan would answer right whatever the index holds.
    async fn check_index(&self, ds: &Dataset, n: usize, kind: IndexKind) -> Result<(), Failure> {
        let unreadable = |error: lance::Error| fail("unreadable", n, error);
        match kind {
            IndexKind::BTree => {
                for value in 0..CATEGORIES {
                    let filter = format!("category = {value}");
                    let mut scan = ds.scan();
                    scan.filter(&filter).map_err(unreadable)?;
                    scan.project(&["id"]).map_err(unreadable)?;
                    if value == 0 {
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
                    let expected: BTreeSet<i64> = self
                        .model
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
                        let id = (n * 3 + slot) as i64;
                        (
                            format!("random query {slot}"),
                            vector(!self.oracle_seed, id),
                        )
                    })
                    .collect();
                // A deleted row's own vector: only an index that ignores
                // deletions would answer with that row.
                if let Some(deleted) =
                    (0..self.model.next_id).find(|id| !self.model.live.contains(id))
                {
                    queries.push((
                        format!("deleted row {deleted}"),
                        vector(self.oracle_seed, deleted),
                    ));
                }
                for (position, (label, query)) in queries.iter().enumerate() {
                    let mut scan = ds.scan();
                    scan.nearest("vec", &Float32Array::from(query.to_vec()), NEIGHBORS)
                        .map_err(unreadable)?;
                    // Every partition: IVF_FLAT is then exact.
                    scan.minimum_nprobes(IVF_PARTITIONS);
                    scan.project(&["id"]).map_err(unreadable)?;
                    if position == 0 {
                        let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                        if !plan.contains("ANNSubIndex: name=idx_vec") {
                            return Err(fail(
                                "index_not_used",
                                n,
                                format_args!(
                                    "a nearest neighbor query does not use idx_vec: {plan}"
                                ),
                            ));
                        }
                    }
                    let batch = scan.try_into_batch().await.map_err(unreadable)?;
                    self.check_neighbors(n, label, query, &int64_column(&batch, "id"))?;
                }
            }
            IndexKind::Inverted => {
                for slot in 0..4 {
                    let token = (n as u64 * 4 + slot) % VOCABULARY;
                    let term = format!("tok{token:02}");
                    let query = FullTextSearchQuery::new(term.clone())
                        .with_column("text".to_string())
                        .map_err(unreadable)?
                        .limit(None);
                    let mut scan = ds.scan();
                    scan.full_text_search(query).map_err(unreadable)?;
                    scan.project(&["id"]).map_err(unreadable)?;
                    if slot == 0 {
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
                    let expected: BTreeSet<i64> = self
                        .model
                        .live
                        .iter()
                        .copied()
                        .filter(|id| tokens(*id).contains(&token))
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
        &self,
        n: usize,
        label: &str,
        query: &[f32; VECTOR_DIM as usize],
        found: &[i64],
    ) -> Result<(), Failure> {
        let mut ranked: Vec<(f32, i64)> = self
            .model
            .live
            .iter()
            .map(|id| (squared_l2(query, &vector(self.oracle_seed, *id)), *id))
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
        if !snapshot.tmp_files.is_empty() {
            return Err(fail(
                "tmp_leak",
                n,
                format_args!("temporary files left behind: {:?}", snapshot.tmp_files),
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
        let n = self.plan.steps.len() + 1;
        self.state.set_phase("final");
        self.journal.line(format_args!("step={n} final"));
        self.catch_up(n).await?;
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
            "step={n} cleanup old_versions={} data_files={} index_files={} transaction_files={} deletion_files={}",
            removed.old_versions,
            removed.data_files_removed,
            removed.index_files_removed,
            removed.transaction_files_removed,
            removed.deletion_files_removed
        ));
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
        let verifier = self.open_verifier(n).await?;
        let cutoff = verifier.version().timestamp.timestamp_micros();
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
                // Cleanup lists these only up to the commit time of the
                // manifest it keeps.
                Some(area @ ("_transactions" | "_deletions")) if *modified > cutoff => {
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
        Box::pin(self.check_content(n)).await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)
    }
}
