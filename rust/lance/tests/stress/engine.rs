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
use crate::knobs::{Class, ENV_SET, Env, Knobs, STEPS};
use crate::scenario::{self, Scenario, SelfTest};
use crate::select::{ENV_ONLY, Selection};

use data::{Scale, rows};
use ops::Model;
use run::Run;

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
    let dir = PathBuf::from(env::<String>(ENV_DIR));
    let quiet = Duration::from_secs(env(ENV_QUIET_SECS));
    let generation: u64 = env(ENV_GENERATION);
    let journal = Arc::new(Journal::open(&dir.join(STEPS_LOG), started));
    let ChildPlan {
        seed,
        self_test,
        knobs,
        plan,
        rendered,
    } = ChildPlan::read(scenario, &Env::of_process());
    let setting = ChildSetting {
        scenario,
        quiet,
        generation,
        self_test,
        started,
        knobs,
    };
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

/// What a child plans, from the variables its parent gives it. The parent works
/// out the fingerprint a replay pins the same way, from the same variables.
pub struct ChildPlan {
    pub seed: u64,
    pub self_test: Option<&'static SelfTest>,
    pub knobs: Knobs,
    pub plan: Plan,
    pub rendered: String,
}

impl ChildPlan {
    pub fn read(scenario: &'static dyn Scenario, vars: &Env) -> Self {
        let number = |name: &str| -> u64 {
            let value = vars
                .get(name)
                .unwrap_or_else(|| panic!("{name} is not set"));
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} has a bad value {value:?}"))
        };
        let seed = number(ENV_SEED);
        let steps = number(ENV_STEPS) as usize;
        let self_test = vars.get(ENV_SELF_TEST).map(|name| {
            scenario
                .self_tests()
                .iter()
                .copied()
                .find(|test| test.name == name)
                .unwrap_or_else(|| panic!("{} has no self-test {name:?}", scenario.name()))
        });
        // The parent gives a child all of its knobs in one record, and nothing
        // else the child inherits sets one.
        let record = Env::from([(ENV_SET, vars.get(ENV_SET).unwrap_or_default())]);
        let knobs = Knobs::resolve(scenario, &record).unwrap_or_else(|error| panic!("{error}"));
        // A self-test arms no incident, so that only the injected defect can fail
        // it, and plans as if nothing were selected. A calibration plans no limit,
        // and reports what each operation uses.
        let focus = match self_test {
            Some(_) => BTreeSet::new(),
            None => Selection::read(vars).focus().clone(),
        };
        let calibrate = vars.get(ENV_CALIBRATE) == Some("1");
        let plan = Plan::generate(
            scenario,
            seed,
            steps,
            self_test.is_none() && !calibrate,
            &focus,
            self_test,
            &knobs,
        );
        let rendered = plan.render(scenario, seed, steps);
        Self {
            seed,
            self_test,
            knobs,
            plan,
            rendered,
        }
    }
}

/// What a child process knows about itself beyond its plan.
pub struct ChildSetting {
    scenario: &'static dyn Scenario,
    quiet: Duration,
    /// Which process of the seed this is, the first one being 1.
    generation: u64,
    self_test: Option<&'static SelfTest>,
    started: Instant,
    knobs: Knobs,
}

fn run_on_runtime(
    setting: ChildSetting,
    plan: Plan,
    dir: &Path,
    journal: Arc<Journal>,
) -> Result<(), Failure> {
    let wanted = setting.scenario.runtime(&setting.knobs);
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(wanted.worker_threads).enable_all();
    if let Some(keep_alive) = wanted.blocking_keep_alive {
        builder.thread_keep_alive(keep_alive);
    }
    let runtime = builder.build().expect("build the runtime");
    let Plan { header, body, .. } = plan;
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
    /// Its knobs of the class plan that are off their default, but those the
    /// header and the body show anyway.
    pub knobs: Vec<String>,
    pub body: Box<dyn Body>,
}

impl Plan {
    /// The focus of the header holds the operations of the list that the
    /// scenario can put its trouble on.
    pub fn generate(
        scenario: &dyn Scenario,
        seed: u64,
        steps: usize,
        with_incidents: bool,
        focus: &BTreeSet<&'static str>,
        self_test: Option<&'static SelfTest>,
        knobs: &Knobs,
    ) -> Self {
        let (mut rng, header) = Self::draw_header(scenario, seed, focus);
        let input = PlanInput {
            seed,
            steps,
            header: &header,
            with_incidents,
            self_test,
            knobs,
        };
        let body = scenario.plan(&mut rng, &input);
        let mut shown = vec![STEPS.name];
        shown.extend(body.shown_knobs());
        Self {
            header,
            knobs: knobs.off_default(Class::Plan, &shown),
            body,
        }
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

    /// The header, the knobs off their default, if any, and the body. The line of
    /// the knobs follows the header, whose line ends with the focus.
    pub fn render(&self, scenario: &dyn Scenario, seed: u64, steps: usize) -> String {
        let mut text = self.header.render(scenario, seed, steps);
        if !self.knobs.is_empty() {
            text.push_str(&format!("knobs {}\n", self.knobs.join(" ")));
        }
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
    pub knobs: &'a Knobs,
}

/// The part of a seed's plan that its scenario draws, and how it runs.
#[async_trait(?Send)]
pub trait Body {
    /// The lines of the plan below its header.
    fn render(&self) -> String;

    /// How many steps, rounds or samples it takes. The final checks are the one
    /// after.
    fn steps(&self) -> usize;

    /// The knobs its lines show, which the plan leaves out of its line of knobs.
    fn shown_knobs(&self) -> &'static [&'static str] {
        &[]
    }

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

/// What the plans of seeds 1 to 16 (1 to 8 of a `soak` of the default length) hash
/// to, by scenario and variant: the fingerprint of the lines `<seed> <fingerprint
/// of its plan>`.
#[cfg(test)]
const GOLDEN_PLANS: &[(&str, &str, &str)] = &[
    ("storage_faults", "default", "866bf5643b8fcda6"),
    ("storage_faults", "steps3", "32498b91787b528b"),
    ("storage_faults", "steps12", "4ece882cea5c2535"),
    ("storage_faults", "calibrate", "b0bb5b593b7d002f"),
    ("storage_faults", "op-optimize", "fa62236aa2863ee1"),
    ("storage_faults", "op-compact-cleanup", "c96454b2450b73e3"),
    ("cancellation", "default", "2c7f59eac06be1d3"),
    ("cancellation", "steps3", "b4bb84e7de3cd6d1"),
    ("cancellation", "steps12", "de8801a6abe89ba4"), // spellchecker:disable-line
    ("cancellation", "calibrate", "540d3c9c6151582c"),
    ("cancellation", "op-optimize", "ecfbef03cd60c0c8"),
    ("cancellation", "op-compact-cleanup", "053c359ab5e18cd2"),
    ("process_kill", "default", "00ac2c91d96143c8"),
    ("process_kill", "steps3", "4c2eb039395e165d"),
    ("process_kill", "steps12", "d772e48c1da35347"),
    ("process_kill", "calibrate", "e9c9e43c17305df8"),
    ("process_kill", "op-optimize", "440bac698fdc6741"),
    ("process_kill", "op-compact-cleanup", "48b93def6d8a4e90"),
    ("process_kill", "op-merge-scan", "c245888569c47a30"),
    ("memory_limits", "default", "951663b75ceb2c42"),
    ("memory_limits", "steps3", "dfff4be6b341b7df"),
    ("memory_limits", "steps12", "cf3f3b3a26c7929d"),
    ("memory_limits", "calibrate", "13e8417ac5863ff6"),
    ("memory_limits", "op-optimize", "1c89a49ed76e1f07"),
    ("memory_limits", "op-compact-cleanup", "9b6780e0373246a0"),
    ("memory_limits", "op-merge-scan", "03d1cf3501b91524"),
    ("heap_growth", "default", "5a16fabe3fb72e94"),
    ("concurrency", "default", "31a7f354be727653"),
    ("concurrency", "steps3", "8330af1f9050158b"),
    ("concurrency", "steps12", "1776f11ea954cd4b"),
    ("concurrency", "calibrate", "50c8d514da4c156b"),
    ("concurrency", "op-optimize", "485fdfb737a3f048"),
    ("concurrency", "op-compact-cleanup", "4557343a16ff6855"),
    ("concurrency", "op-merge-scan", "052bd882b573f0d9"),
    ("slow_storage", "default", "4a3be7cb320ea930"),
    ("slow_storage", "steps3", "124b4d804c3c54f2"),
    ("slow_storage", "steps12", "f4eaf2011f4213c0"),
    ("slow_storage", "calibrate", "8b7802cce0383213"),
    ("slow_storage", "op-optimize", "da249fec7682d914"),
    ("slow_storage", "op-compact-cleanup", "2316decd61b1762b"),
    ("soak", "default", "75041026bec910fc"),
    ("soak", "secs24", "cc88d5830e3e975b"),
];

/// The variables a parent gives the child of `seed`, as far as its plan goes:
/// the record of `knobs`, the steps the settings hand it, and the modes.
#[cfg(test)]
pub fn child_vars_of(
    scenario: &dyn Scenario,
    seed: u64,
    knobs: &Knobs,
    modes: &[(&'static str, &str)],
) -> Env {
    let steps = scenario.settings(None, knobs).steps;
    [
        (ENV_SEED, seed.to_string()),
        (ENV_STEPS, steps.to_string()),
        (ENV_SET, knobs.record()),
    ]
    .into_iter()
    .chain(modes.iter().map(|(name, value)| (*name, value.to_string())))
    .collect()
}

/// The plans a run draws by default stay what they were, and so do those of the
/// knobs the variables of the harness set before there were knobs: a change that
/// moves a draw, or a default, fails here rather than in a replay. The plans come
/// the way a child draws them, from the variables its parent gives it.
#[cfg(test)]
#[rstest::rstest]
#[case::default("default", "", &[])]
#[case::steps3("steps3", "steps=3", &[])]
#[case::steps12("steps12", "steps=12", &[])]
#[case::calibrate("calibrate", "", &[(ENV_CALIBRATE, "1")])]
#[case::op_optimize("op-optimize", "", &[(ENV_ONLY, "op:optimize_indices")])]
#[case::op_compact_cleanup("op-compact-cleanup", "", &[(ENV_ONLY, "op:compact,op:cleanup")])]
#[case::op_merge_scan("op-merge-scan", "", &[(ENV_ONLY, "op:merge_insert,op:scan")])]
#[case::secs24("secs24", "soak.secs=24s", &[])]
fn registry_default_plans_keep_their_fingerprints(
    #[case] variant: &str,
    #[case] set: &str,
    #[case] modes: &[(&'static str, &str)],
) {
    use std::fmt::Write as _;

    let golden: Vec<_> = GOLDEN_PLANS
        .iter()
        .filter(|(_, of, _)| *of == variant)
        .collect();
    assert!(!golden.is_empty(), "no golden plans of {variant}");
    for &&(name, _, expected) in &golden {
        let scenario = scenario::by_name(name).expect("a scenario of the registry");
        let knobs = Knobs::resolve(scenario, &Env::from([(ENV_SET, set)])).unwrap();
        let seeds = match name {
            "soak" if variant == "default" => 1..=8,
            _ => 1..=16,
        };
        let mut prints = String::new();
        for seed in seeds {
            let vars = child_vars_of(scenario, seed, &knobs, modes);
            let print = fingerprint(&ChildPlan::read(scenario, &vars).rendered);
            writeln!(prints, "{seed} {print}").unwrap();
        }
        assert_eq!(
            fingerprint(&prints),
            expected,
            "the plans of {name} ({variant}) changed; by seed:\n{prints}"
        );
    }
}

/// Each knob of the class plan, at either end of its bounds: a plan still draws,
/// draws the same again, and differs from the default plan of one of two seeds,
/// unless the knob decides only how a plan runs.
#[cfg(test)]
#[rstest::rstest]
#[case::storage_faults("storage_faults")]
#[case::cancellation("cancellation")]
#[case::process_kill("process_kill")]
#[case::memory_limits("memory_limits")]
#[case::heap_growth("heap_growth")]
#[case::concurrency("concurrency")]
#[case::slow_storage("slow_storage")]
#[case::soak("soak")]
fn registry_plan_knobs_reach_their_plans(#[case] name: &str) {
    use crate::knobs::{Fallback, Kind, Value};

    // How long a hold waits is up to the run, not the plan: the line of knobs
    // carries it into the fingerprint.
    const RUN_ONLY: [&str; 1] = ["concurrency.gate_limit"];
    // A knob that a plan of the default length cannot show, and what makes one
    // long enough: 24 samples are the fewest, which a soak of 3 min takes anyway.
    const LONGER: [(&str, &str); 1] = [("soak.interval", "soak.secs=1h")];
    let scenario = scenario::by_name(name).expect("a scenario of the registry");
    let none = BTreeSet::new();
    let body = |knobs: &Knobs, seed: u64| {
        let steps = scenario.settings(None, knobs).steps as usize;
        Plan::generate(scenario, seed, steps, true, &none, None, knobs)
            .body
            .render()
    };
    let resolve = |sets: &[&str]| {
        let set = sets.join(",");
        Knobs::resolve(scenario, &Env::from([(ENV_SET, set.as_str())]))
            .unwrap_or_else(|error| panic!("{set}: {error}"))
    };
    for knob in scenario.knobs() {
        if knob.class != Class::Plan {
            continue;
        }
        let ends = match knob.kind {
            Kind::Count { least, most } => [Value::Count(least), Value::Count(most)],
            Kind::Range { least, most } => [Value::Range(least, least), Value::Range(most, most)],
            Kind::Duration { least, most } => [Value::Duration(least), Value::Duration(most)],
            Kind::Bytes { least, most } => [Value::Bytes(least), Value::Bytes(most)],
            Kind::Share { least, most } => [Value::Share(least), Value::Share(most)],
        };
        let key = format!("{name}.{}", knob.name);
        let longer: Vec<&str> = LONGER
            .iter()
            .filter(|(knob, _)| *knob == key)
            .map(|(_, set)| *set)
            .collect();
        let base = resolve(&longer);
        let default_bodies = [body(&base, 1), body(&base, 2)];
        let mut moved = false;
        for end in ends {
            if knob.default == Fallback::Fixed(end) {
                continue;
            }
            let set = format!("{key}={end}");
            let knobs = resolve(&[longer.as_slice(), &[set.as_str()]].concat());
            for (seed, default_body) in (1..).zip(&default_bodies) {
                let drawn = body(&knobs, seed);
                assert_eq!(
                    drawn,
                    body(&knobs, seed),
                    "{set} seed {seed} draws two plans"
                );
                moved |= drawn != *default_body;
            }
        }
        assert_eq!(
            moved,
            !RUN_ONLY.contains(&key.as_str()),
            "{key} at either end of its bounds must change a plan, unless only a run reads it"
        );
    }
}

/// Each knob of the class judge, at the end of its bounds away from its default,
/// leaves the plan as it was, its line of knobs included: what passes is no part
/// of what runs.
#[cfg(test)]
#[rstest::rstest]
#[case::storage_faults("storage_faults")]
#[case::cancellation("cancellation")]
#[case::process_kill("process_kill")]
#[case::memory_limits("memory_limits")]
#[case::heap_growth("heap_growth")]
#[case::concurrency("concurrency")]
#[case::slow_storage("slow_storage")]
#[case::soak("soak")]
fn registry_judge_knobs_leave_their_plans(#[case] name: &str) {
    use crate::knobs::{Fallback, Kind, Value};

    let scenario = scenario::by_name(name).expect("a scenario of the registry");
    let plan = |knobs: &Knobs, seed: u64| {
        ChildPlan::read(scenario, &child_vars_of(scenario, seed, knobs, &[])).rendered
    };
    let defaults = Knobs::defaults(scenario);
    let default_plans = [plan(&defaults, 1), plan(&defaults, 2)];
    for knob in scenario.knobs() {
        if knob.class != Class::Judge {
            continue;
        }
        let (least, most) = match knob.kind {
            Kind::Count { least, most } => (Value::Count(least), Value::Count(most)),
            Kind::Bytes { least, most } => (Value::Bytes(least), Value::Bytes(most)),
            Kind::Duration { least, most } => (Value::Duration(least), Value::Duration(most)),
            other => panic!("{name}.{} judges by a {other:?}", knob.name),
        };
        let end = match knob.default == Fallback::Fixed(most) {
            true => least,
            false => most,
        };
        let set = format!("{name}.{}={end}", knob.name);
        let knobs = Knobs::resolve(scenario, &Env::from([(ENV_SET, set.as_str())]))
            .unwrap_or_else(|error| panic!("{set}: {error}"));
        for (seed, default_plan) in (1..).zip(&default_plans) {
            assert_eq!(plan(&knobs, seed), *default_plan, "{set} seed {seed}");
        }
    }
}

/// A knob of the plan off its default shows in its own line after the header,
/// which keeps the focus at its end, and so changes the fingerprint; the steps
/// show in the header alone.
#[test]
fn registry_plan_knobs_go_into_the_fingerprint() {
    let scenario = scenario::by_name("concurrency").expect("a scenario of the registry");
    let plan_of = |set: &str| {
        let knobs = Knobs::resolve(scenario, &Env::from([(ENV_SET, set)])).unwrap();
        let vars = child_vars_of(scenario, 1, &knobs, &[(ENV_ONLY, "op:cleanup")]);
        ChildPlan::read(scenario, &vars).rendered
    };
    let default = plan_of("");
    let gated = plan_of("concurrency.gate_limit=20s,steps=8");
    let lines: Vec<&str> = gated.lines().take(2).collect();
    assert!(lines[0].ends_with(" focus=cleanup"), "{gated}");
    assert_eq!(lines[1], "knobs concurrency.gate_limit=20s");
    // Only the line of knobs differs, and so does the fingerprint.
    assert_eq!(
        gated.replacen("knobs concurrency.gate_limit=20s\n", "", 1),
        default
    );
    assert_ne!(fingerprint(&gated), fingerprint(&default));
    let shorter = plan_of("steps=3");
    assert!(!shorter.contains("\nknobs "), "{shorter}");
}

/// A self-test of `soak` runs its own short length, whatever the knobs say.
#[test]
fn registry_soak_self_test_plans_its_own_length() {
    let soak = scenario::by_name("soak").expect("a scenario of the registry");
    let test = soak.self_tests()[0];
    let knobs = Knobs::resolve(soak, &Env::from([(ENV_SET, "soak.secs=1h")])).unwrap();
    let plan = Plan::generate(soak, 1, 0, false, &BTreeSet::new(), Some(test), &knobs);
    assert!(
        plan.body.render().starts_with("soak secs=24 "),
        "{}",
        plan.body.render()
    );
}
