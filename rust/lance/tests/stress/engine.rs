// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The child side: a seed's plan, the run of it, and the checks after each step.

pub mod checks;
pub mod data;
pub mod ops;
pub mod rows;
pub mod run;
pub mod steps;

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::ops::RangeInclusive;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::RecordBatchReader;
use async_trait::async_trait;

use crate::harness::{
    ENV_CALIBRATE, ENV_CHILD, ENV_DIR, ENV_GENERATION, ENV_PARENT_PID, ENV_PLAN, ENV_QUIET_SECS,
    ENV_SEED, ENV_SELF_TEST, ENV_STANDALONE, ENV_STEPS, PLAN_LOG, STEPS_LOG,
};
use crate::scenario::{self, Scenario, SelfTest};
use crate::select::Selection;

use data::{Scale, rows};
use ops::Model;
use run::Run;

/// A calibration run plans no limit, and reports what each operation uses.
fn calibrating() -> bool {
    std::env::var(ENV_CALIBRATE).is_ok_and(|value| value == "1")
}

/// Runs the seed this process was started for, if any, and exits non-zero
/// when it fails. The verdict is the last line of the step log.
pub fn run_child_if_requested() {
    let Ok(name) = std::env::var(ENV_CHILD) else {
        return;
    };
    let standalone = std::env::var(ENV_STANDALONE).is_ok_and(|value| value == "1");
    if !standalone && !tie_to_parent() {
        return;
    }
    let started = Instant::now();
    let scenario = scenario::by_name(&name).unwrap_or_else(|| panic!("unknown scenario {name:?}"));
    let seed: u64 = env(ENV_SEED);
    let dir = PathBuf::from(env::<String>(ENV_DIR));
    let steps: usize = env(ENV_STEPS);
    let quiet = Duration::from_secs(env(ENV_QUIET_SECS));
    let generation: u64 = env(ENV_GENERATION);
    let self_test = std::env::var(ENV_SELF_TEST).ok().map(|name| {
        scenario
            .self_tests()
            .iter()
            .copied()
            .find(|test| test.name == name)
            .unwrap_or_else(|| panic!("{} has no self-test {name:?}", scenario.name()))
    });

    let setting = ChildSetting {
        scenario,
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
    let plan = Plan::generate(
        scenario,
        seed,
        steps,
        self_test.is_none() && !calibrating(),
        &focus,
        self_test,
    );
    let rendered = plan.render(scenario, seed, steps);
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
pub struct ChildSetting {
    scenario: &'static dyn Scenario,
    quiet: Duration,
    /// Which process of the seed this is, the first one being 1.
    generation: u64,
    self_test: Option<&'static SelfTest>,
    started: Instant,
}

fn run_on_runtime(
    setting: ChildSetting,
    plan: Plan,
    dir: &Path,
    journal: Arc<Journal>,
) -> Result<(), Failure> {
    let wanted = setting.scenario.runtime();
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(wanted.worker_threads).enable_all();
    if let Some(keep_alive) = wanted.blocking_keep_alive {
        builder.thread_keep_alive(keep_alive);
    }
    let runtime = builder.build().expect("build the runtime");
    let Plan { header, body } = plan;
    // A panic outside an operation, in a check for example, must still end in
    // a verdict. Dropping the runtime while unwinding would wait, without a
    // limit, for any thread that panic has wedged.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(async {
            let mut run =
                Box::pin(Run::setup(setting, header, body.as_ref(), dir, journal)).await?;
            body.run(&mut run).await?;
            Box::pin(run.finish(body.steps() + 1)).await
        })
    }));
    // A failed round or soak leaves its tasks running, and the process is about to
    // exit. Shutting the runtime down would cancel their scans, and a scan cancelled
    // that way panics (`filtered_read.rs`), which would hide the failure.
    if wanted.leaves_tasks_on_failure && matches!(result, Ok(Err(_))) {
        std::mem::forget(runtime);
    } else {
        // A self-test leaves a task behind on purpose; do not wait for it.
        runtime.shutdown_timeout(Duration::from_secs(5));
    }
    result.unwrap_or_else(|panic| Err(fail("panic", 0, panic_message(panic.as_ref()))))
}

pub fn env<T: std::str::FromStr>(name: &str) -> T {
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

pub fn splitmix64(seed: u64) -> u64 {
    mix(seed.wrapping_add(GOLDEN_GAMMA))
}

/// The plan's random source. A splitmix64 sequence of its own keeps a seed's
/// plan the same whatever version of `rand` the workspace locks.
pub struct PlanRng(pub u64);

impl PlanRng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN_GAMMA);
        mix(self.0)
    }

    /// Close enough to uniform for the small ranges a plan draws from.
    pub fn range(&mut self, range: RangeInclusive<u64>) -> u64 {
        range.start() + self.next() % (range.end() - range.start() + 1)
    }

    pub fn chance(&mut self, probability: f64) -> bool {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64) < probability
    }
}

// Every column is a function of the row id, so the model only keeps ids.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreChoice {
    Local,
    SharedMemory,
    /// The same files as `Local`, but every request passes the object store,
    /// so an incident reaches data, index and transaction files too.
    FileObjectStore,
}

/// What a seed decides first, whatever its scenario.
pub struct Header {
    pub store: StoreChoice,
    pub scale: Scale,
    pub data_seed: u64,
    pub initial_rows: i64,
    /// The operations that `LANCE_STRESS_ONLY` puts the trouble on.
    pub focus: BTreeSet<&'static str>,
}

impl Header {
    fn render(&self, scenario: &dyn Scenario, seed: u64, steps: usize) -> String {
        let mut text = format!(
            "scenario={} seed={seed} steps={steps} store={:?} scale={:?} data_seed={} initial_rows={}\n",
            scenario.name(),
            self.store,
            self.scale,
            self.data_seed,
            self.initial_rows
        );
        if !self.focus.is_empty() {
            let focus: Vec<&str> = self.focus.iter().copied().collect();
            text.insert_str(text.len() - 1, &format!(" focus={}", focus.join(",")));
        }
        text
    }
}

/// Everything a seed decides: the header every scenario draws alike, then the
/// body its scenario draws.
pub struct Plan {
    pub header: Header,
    pub body: Box<dyn Body>,
}

impl Plan {
    /// The focus of the header holds the operations of the list that the
    /// scenario can put its trouble on.
    fn generate(
        scenario: &dyn Scenario,
        seed: u64,
        steps: usize,
        with_incidents: bool,
        focus: &BTreeSet<&'static str>,
        self_test: Option<&'static SelfTest>,
    ) -> Self {
        let (mut rng, header) = Self::draw_header(scenario, seed, focus);
        let input = PlanInput {
            seed,
            steps,
            header: &header,
            with_incidents,
            self_test,
        };
        let body = scenario.plan(&mut rng, &input);
        Self { header, body }
    }

    /// The header of a seed's plan, and the random source the rest of it draws from.
    pub fn draw_header(
        scenario: &dyn Scenario,
        seed: u64,
        focus: &BTreeSet<&'static str>,
    ) -> (PlanRng, Header) {
        // Only what the scenario can run: a list for several scenarios plans each
        // of them as a list of its own operations would.
        let focus: BTreeSet<&'static str> = scenario
            .focusable_ops()
            .into_iter()
            .filter(|op| focus.contains(op))
            .collect();
        let mut rng = PlanRng(seed);
        let store = scenario.store(&mut rng);
        let scale = scenario.scale();
        let data_seed = rng.next();
        let initial_rows = rng.range(scale.initial_rows.0..=scale.initial_rows.1) as i64;
        let header = Header {
            store,
            scale,
            data_seed,
            initial_rows,
            focus,
        };
        (rng, header)
    }

    fn render(&self, scenario: &dyn Scenario, seed: u64, steps: usize) -> String {
        let mut text = self.header.render(scenario, seed, steps);
        text.push_str(&self.body.render());
        text
    }
}

/// What the plan of a scenario draws from, besides the random source.
pub struct PlanInput<'a> {
    pub seed: u64,
    /// The steps, or rounds, a run asks for.
    pub steps: usize,
    pub header: &'a Header,
    /// Whether the plan arms incidents and limits: not for a self-test or a
    /// calibration.
    pub with_incidents: bool,
    pub self_test: Option<&'static SelfTest>,
}

/// The part of a seed's plan that its scenario draws, and how it runs.
#[async_trait(?Send)]
pub trait Body {
    /// The lines of the plan below its header.
    fn render(&self) -> String;

    /// How many steps, rounds or samples it takes. The final checks are the one
    /// after.
    fn steps(&self) -> usize;

    fn content(&self) -> ContentRules;

    fn stable_row_ids(&self) -> bool {
        false
    }

    /// The rows the dataset starts with.
    fn initial_source(&self, header: &Header) -> Box<dyn RecordBatchReader + Send> {
        Box::new(rows(header.data_seed, 0..header.initial_rows, header.scale))
    }

    fn initial_model(&self, header: &Header) -> Model {
        Model::new(header.initial_rows)
    }

    /// The logs of its own that it writes. The setup opens them before the
    /// baseline, which counts their descriptors, and they stay open to the end.
    fn journals(&self) -> &'static [&'static str] {
        &[]
    }

    /// Runs it on the dataset the setup wrote, after the baseline. A scenario
    /// whose process dies goes on with `Run::resume`.
    async fn run(&self, run: &mut Run) -> Result<(), Failure>;
}

/// What the check of the content, which every scenario shares, holds a seed to.
#[derive(Clone, Copy)]
pub struct ContentRules {
    /// No version goes that no cleanup could remove. The rounds of `concurrency`
    /// check their history themselves.
    pub versions_stay_listed: bool,
    /// Every index answers for the whole model, and a query's plan uses it. Under
    /// concurrency an index can legitimately cover nothing by now: an update in
    /// place takes its fragments away from it.
    pub indices_cover_the_model: bool,
    /// A version reads as it did when it was the latest. Only for the versions a
    /// cleanup keeps: in `heap_growth` and `soak` the map would count as growth
    /// of the heap, and `concurrency` checks its history itself.
    pub digests: bool,
    /// A file the store does not list fails the seed after a cleanup.
    pub hidden_files_fail: bool,
}

/// FNV-1a of a rendered plan, in hex.
pub fn fingerprint(rendered: &str) -> String {
    let hash = rendered
        .bytes()
        .fold(0xCBF2_9CE4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
        });
    format!("{hash:016x}")
}

/// The fingerprint of the plan a seed generates, for a replay command.
pub fn plan_fingerprint(scenario: &dyn Scenario, seed: u64, steps: usize) -> String {
    let plan = Plan::generate(
        scenario,
        seed,
        steps,
        !calibrating(),
        Selection::from_env().focus(),
        None,
    );
    fingerprint(&plan.render(scenario, seed, steps))
}

pub struct Failure {
    class: &'static str,
    step: usize,
    detail: String,
}

pub fn fail(class: &'static str, step: usize, detail: impl fmt::Display) -> Failure {
    Failure {
        class,
        step,
        detail: one_line(&detail.to_string()),
    }
}

pub fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    let message = panic
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_string());
    one_line(&message)
}
