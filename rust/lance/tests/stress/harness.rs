// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The parent side: runs each seed in a child process, watches it for
//! progress, and turns what the child left behind into a verdict.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::panic::AssertUnwindSafe;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use crate::engine::{self, ChildPlan};
use crate::incidents::CRASH_FIRED;
use crate::knobs::{
    Class, DEADLINE, ENV_SET, Env, Knobs, REPEAT, SEED_BASE, SEEDS, STALL, STEPS, UNPINNED, Value,
};
use crate::limits::{ENV_CGROUP, RunScope, SeedCgroup};
use crate::scenario::{self, Scenario, SelfTest};
use crate::select::{ENV_ONLY, Selection};

pub const ENV_CHILD: &str = "LANCE_STRESS_CHILD";
pub const ENV_SEED: &str = "LANCE_STRESS_SEED";
pub const ENV_DIR: &str = "LANCE_STRESS_DIR";
pub const ENV_GENERATION: &str = "LANCE_STRESS_GENERATION";
pub const ENV_PARENT_PID: &str = "LANCE_STRESS_PARENT_PID";
pub const ENV_STEPS: &str = "LANCE_STRESS_STEPS";
pub const ENV_QUIET_SECS: &str = "LANCE_STRESS_QUIET_SECS";
pub const ENV_SELF_TEST: &str = "LANCE_STRESS_SELF_TEST";
pub const ENV_PLAN: &str = "LANCE_STRESS_PLAN";
pub const ENV_STANDALONE: &str = "LANCE_STRESS_STANDALONE";
pub const ENV_CALIBRATE: &str = "LANCE_STRESS_CALIBRATE";
/// Workers run one operation at a time, in the order of their ranks: a failure of
/// `concurrency` that goes away with this needs a race.
pub const ENV_SERIAL: &str = "LANCE_STRESS_SERIAL";

/// How long a step may wait for the runtime to go quiet, but in a self-test.
pub const QUIET_SECS: u64 = 20;

pub const PLAN_LOG: &str = "plan.log";
pub const STEPS_LOG: &str = "steps.log";
pub const CALLS_LOG: &str = "calls.log";

const POLL: Duration = Duration::from_millis(200);
const GDB_TIMEOUT: Duration = Duration::from_secs(60);
const REAP_TIMEOUT: Duration = Duration::from_secs(30);
const STDERR_LIMIT: usize = 8 << 20;
/// Well short of the 20 seconds `systemd-oomd` waits, so that nothing else is at
/// risk from a thrashing child.
const THRASH_PATIENCE: Duration = Duration::from_secs(10);
/// A silent child that used this share of the stall on the CPU is busy, not
/// hung. A loop that never ends meets the deadline.
const BUSY_SHARE: f64 = 0.05;
/// Seeds, not plans: one seed that plans a limit or a hold point many times, or
/// runs many times, may never reach it by chance, as all its tries plan the same.
const DEAD_LIMIT_SEEDS: usize = 3;
const DEAD_HOLD_SEEDS: usize = 3;
/// Seeds, not rounds: 16 seeds force each pair twice, and one seed never judges
/// a pair, whether it is repeated or a focus forces the pair in several rounds.
const DEAD_PAIR_SEEDS: usize = 2;

struct RunConfig {
    scenario: &'static dyn Scenario,
    /// This invocation's own directory: a replay or a concurrent run never
    /// touches another run's artifacts.
    run_dir: PathBuf,
    /// The size of a plan, as its header and the child take it.
    steps: u64,
    /// The knobs of the run, its stall and deadline derived.
    knobs: Knobs,
    stall: Duration,
    deadline: Duration,
    quiet_secs: u64,
    keep: Keep,
    self_test: Option<&'static SelfTest>,
    /// Where the children of a run under memory limits live.
    scope: Option<RunScope>,
    /// What the children get, the variables of the checks of the scenario included.
    child_env: Vec<(&'static str, Option<String>)>,
    selection: Selection,
    calibrate: bool,
    serial: bool,
    /// The seed that `LANCE_STRESS_SEED` runs alone, and the plan a replay of it
    /// expects.
    replay: Option<(u64, Option<String>)>,
    /// The variables of Lance the run inherited, which its replay repeats.
    lance_env: Vec<(String, String)>,
}

/// What stays of a seed that passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keep {
    Nothing,
    /// The logs, without the dataset and the temporary files.
    Logs,
    Everything,
}

impl RunConfig {
    /// A run of `scenario`, or of one of its self-tests, with the knobs `env` sets.
    fn new(
        scenario: &'static dyn Scenario,
        self_test: Option<&'static SelfTest>,
        env: &Env,
        selection: Selection,
        knobs: Knobs,
    ) -> Self {
        // Self-tests run at the defaults, on plans of 3 steps: their timing is part
        // of what they check. A deadline, which a slow host may need, is all they
        // take from the knobs.
        let (mut knobs, quiet_secs) = match self_test {
            Some(_) => {
                let mut defaults = Knobs::defaults(scenario);
                if defaults.declares(STEPS.name) {
                    STEPS.set(&mut defaults, 3);
                }
                if let Some(deadline) = DEADLINE.explicit(&knobs) {
                    DEADLINE.set(&mut defaults, deadline);
                }
                (defaults, 2)
            }
            None => (knobs, QUIET_SECS),
        };
        let settings = scenario.settings(self_test, &knobs);
        knobs.derive(STALL, settings.stall);
        knobs.derive(DEADLINE, settings.deadline);
        let stall = match self_test {
            Some(_) => Duration::from_secs(15),
            None => STALL.get(&knobs),
        };
        assert!(
            stall >= Duration::from_secs(quiet_secs + 10),
            "stall={stall:?} must exceed the {quiet_secs}s a step may wait for quiet by 10s"
        );
        // `Knobs::resolve` held a deadline set by hand to the least one.
        let deadline = DEADLINE.get(&knobs);
        let name = match self_test {
            Some(test) => format!("self-test-{}", test.name),
            None => scenario.name().to_string(),
        };
        let run_dir = artifacts_root(env).join(format!(
            "{name}-{}-{}",
            chrono::Utc::now().format("%Y%m%d-%H%M%S"),
            std::process::id()
        ));
        let replay = env.get(ENV_SEED).map(|seed| {
            let seed = seed
                .parse()
                .unwrap_or_else(|_| panic!("{ENV_SEED} must be an unsigned integer, got {seed:?}"));
            (seed, env.get(ENV_PLAN).map(str::to_string))
        });
        let keep = if self_test.is_some() || replay.is_some() {
            Keep::Everything
        } else {
            match env.get("LANCE_STRESS_KEEP") {
                Some("1") => Keep::Everything,
                Some("logs") => Keep::Logs,
                _ => Keep::Nothing,
            }
        };
        let scope = settings.cgroup.then(|| {
            RunScope::create(&name).unwrap_or_else(|error| {
                panic!(
                    "{} runs in cgroups, which it cannot create here: {error}",
                    scenario.name()
                )
            })
        });
        let mut child_env = settings.child_env;
        for check in scenario.checks() {
            child_env.extend(check.child_env(self_test));
        }
        Self {
            scenario,
            run_dir,
            steps: settings.steps,
            knobs,
            stall,
            deadline,
            quiet_secs,
            keep,
            self_test,
            scope,
            child_env,
            selection,
            calibrate: env.get(ENV_CALIBRATE) == Some("1"),
            serial: env.get(ENV_SERIAL) == Some("1"),
            replay,
            lance_env: env
                .lance_tuning()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        }
    }

    /// The knobs a child gets: all of them, or none for a self-test, which runs at
    /// the defaults.
    fn child_record(&self) -> String {
        match self.self_test {
            Some(_) => String::new(),
            None => self.knobs.record(),
        }
    }
}

/// The artifacts root, absolute and without `.` or `..`. Lance spells local
/// paths that way, and the incidents tell a dataset's directories apart by
/// that spelling.
fn artifacts_root(env: &Env) -> PathBuf {
    let root = env
        .get("LANCE_STRESS_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("stress"));
    let absolute = std::path::absolute(&root).expect("resolve the artifacts root");
    let mut normal = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normal.pop();
            }
            other => normal.push(other),
        }
    }
    normal
}

/// The run that `env` asks of `scenario`, or of one of its self-tests, or none
/// when the selection leaves it out.
fn prepare(
    scenario: &'static dyn Scenario,
    self_test: Option<&'static SelfTest>,
    env: &Env,
) -> Option<RunConfig> {
    let selection = Selection::read(env);
    // Even when the selection leaves the scenario out: a key no scenario has must
    // fail every test.
    let knobs = Knobs::resolve(scenario, env).unwrap_or_else(|error| panic!("{error}"));
    let selected = match self_test {
        Some(_) => selection.runs_self_test(scenario),
        None => selection.runs(scenario),
    };
    if !selected {
        // libtest has no verdict for a test that did not run: it passes. A
        // replay names its scenario, so there it would pass for a seed it never
        // ran.
        let replay =
            self_test.is_none() && env.get(ENV_SEED).is_some() && env.get(ENV_PLAN).is_some();
        assert!(
            !replay,
            "{ENV_PLAN} replays a seed of {}, which {ENV_ONLY} leaves out",
            scenario.name()
        );
        return None;
    }
    knobs
        .check_least_deadline(scenario, self_test)
        .unwrap_or_else(|error| panic!("{error}"));
    Some(RunConfig::new(scenario, self_test, env, selection, knobs))
}

/// Runs every requested seed of a scenario and fails if any seed failed.
pub fn run(scenario: &'static dyn Scenario) {
    let Some(config) = prepare(scenario, None, &Env::of_process()) else {
        return;
    };
    // With fewer steps, a focus would fall on nothing.
    let focus = config.selection.focus();
    let focused: Vec<&str> = scenario
        .focusable_ops()
        .into_iter()
        .filter(|op| focus.contains(op))
        .collect();
    let least = scenario.least_focus_steps(&focused);
    assert!(
        focus.is_empty() || config.steps >= least,
        "the operations of {ENV_ONLY} need {} of at least {least} in {}, not {}",
        STEPS.name,
        scenario.name(),
        config.steps
    );
    fs::create_dir_all(&config.run_dir).expect("create the run directory");
    record_environment(&config.run_dir);
    let seeds: Vec<u64> = match config.replay {
        Some((seed, _)) => vec![seed],
        None => {
            let base = SEED_BASE.get(&config.knobs);
            (base..base + SEEDS.get(&config.knobs)).collect()
        }
    };
    let mut summary = Summary::new(&config, &seeds);
    for &seed in &seeds {
        for attempt in 1..=REPEAT.get(&config.knobs) {
            summary.add(run_seed(&config, seed, attempt));
            // Written after every seed, so an interrupted run keeps its verdicts.
            fs::write(config.run_dir.join("summary.txt"), summary.render())
                .expect("write the summary");
        }
    }
    assert!(summary.passed(), "{}", summary.render());
}

/// Runs one seed with a defect injected and checks that it is reported.
pub fn check_self_test(scenario: &'static dyn Scenario, test: &'static SelfTest) {
    assert!(
        scenario::by_name(scenario.name()).is_some(),
        "{} has self-tests but is missing from `scenarios!` in scenario.rs",
        scenario.name()
    );
    let Some(config) = prepare(scenario, Some(test), &Env::of_process()) else {
        return;
    };
    fs::create_dir_all(&config.run_dir).expect("create the run directory");
    let report = run_seed(&config, 1, 1);
    assert!(
        match report.class.as_deref() {
            Some(class) => test.expected.contains(&class),
            None => test.expected.is_empty(),
        },
        "the {} self-test was reported as {:?}: {} (artifacts in {})",
        test.name,
        report.class,
        report.detail,
        report.dir.display()
    );
    if let Some(reason) = test.reason {
        assert!(
            report.detail.contains(reason),
            "the {} self-test must fail with {reason:?}, but: {} (artifacts in {})",
            test.name,
            report.detail,
            report.dir.display()
        );
    }
    (test.verify)(&report);
    if test.expected == ["hang"] {
        let threads = fs::read_to_string(report.dir.join("threads.txt")).unwrap_or_default();
        assert!(
            !threads.is_empty(),
            "a hang must leave the thread states in {}",
            report.dir.display()
        );
        let refused = fs::read_to_string(report.dir.join("stacks.err"))
            .is_ok_and(|err| err.contains("Operation not permitted"));
        // Some hosts forbid ptrace altogether; the thread states still stand.
        if gdb_available() && !refused {
            let stacks = fs::read_to_string(report.dir.join("stacks.txt")).unwrap_or_default();
            assert!(
                stacks.contains("Thread "),
                "gdb did not attach to the hung child, see {}",
                report.dir.display()
            );
        }
    }
    fs::remove_dir_all(&config.run_dir).ok();
}

/// What the child inherits beyond what the harness sets, and the build.
fn record_environment(run_dir: &Path) {
    let mut text = String::new();
    writeln!(
        text,
        "exe={}\ndebug_assertions={}\navailable_parallelism={}",
        fs::read_link("/proc/self/exe")
            .map_or_else(|_| "?".into(), |exe| exe.display().to_string()),
        cfg!(debug_assertions),
        std::thread::available_parallelism().map_or(0, |n| n.get())
    )
    .unwrap();
    for (name, value) in recorded_vars(std::env::vars_os()) {
        writeln!(text, "{name}={value}").unwrap();
    }
    fs::write(run_dir.join("env.txt"), text).expect("write env.txt");
}

/// The variables of `vars` worth a record, sorted, a value that is not UTF-8 as
/// near as it reads.
fn recorded_vars(vars: impl IntoIterator<Item = (OsString, OsString)>) -> Vec<(String, String)> {
    let mut recorded: Vec<(String, String)> = vars
        .into_iter()
        .filter_map(|(name, value)| Some((name.into_string().ok()?, value)))
        .filter(|(name, _)| {
            ["LANCE_", "RUST_", "TOKIO_", "RAYON_"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
                && !["KEY", "SECRET", "TOKEN", "PASSWORD"]
                    .iter()
                    .any(|word| name.contains(word))
        })
        .map(|(name, value)| (name, value.to_string_lossy().into_owned()))
        .collect();
    recorded.sort();
    recorded
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./=:,".contains(c))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

/// The command that reruns `seed` with the same settings: the fingerprint of the
/// plan, every knob of the plan and the verdicts, the modes, and every Lance tuning
/// variable the run inherited. It removes from the shell the sources of what the
/// record leaves out, and a selection the run did not have, which would change
/// the replay without failing it.
fn replay_command(config: &RunConfig, seed: u64) -> String {
    let only = config
        .selection
        .given()
        .filter(|_| config.self_test.is_none());
    let mut vars: Vec<String> = UNPINNED
        .iter()
        .chain(only.is_none().then_some(&ENV_ONLY))
        .map(|name| format!("-u {name}"))
        .collect();
    vars.extend(
        config
            .lance_env
            .iter()
            .map(|(name, value)| format!("{name}={}", shell_quote(value))),
    );
    // The path of the test, which `scenarios!` and `self_tests!` make.
    if let Some(test) = config.self_test {
        // A self-test takes a deadline from the knobs and nothing else.
        let deadline = DEADLINE.explicit(&config.knobs).map(|deadline| {
            format!(
                "{}.{}={}",
                config.scenario.name(),
                DEADLINE.name,
                Value::Duration(deadline)
            )
        });
        vars.push(format!("{ENV_SET}={}", deadline.unwrap_or_default()));
        return format!(
            "env {} cargo test -p lance --test stress -- --exact scenario::{}::self_test::{} --nocapture",
            vars.join(" "),
            config.scenario.name(),
            test.name
        );
    }
    vars.push(format!("{ENV_SEED}={seed}"));
    // The plan its child draws, from the variables it would get.
    let child: Env = child_vars(config, seed, "", 1)
        .into_iter()
        .filter_map(|(name, value)| Some((name, value?)))
        .collect();
    let plan = ChildPlan::read(config.scenario, &child);
    vars.push(format!(
        "{ENV_PLAN}={}",
        engine::fingerprint(&plan.rendered)
    ));
    vars.push(format!("{ENV_SET}={}", shell_quote(&config.knobs.record())));
    // A mode that is off is pinned too, so that one left in the shell cannot
    // change what the replay runs.
    for (name, on) in [
        (ENV_CALIBRATE, config.calibrate),
        (ENV_SERIAL, config.serial),
    ] {
        vars.push(format!("{name}={}", u8::from(on)));
    }
    if let Some(only) = only {
        vars.push(format!("{ENV_ONLY}={}", shell_quote(only)));
    }
    format!(
        "env {} cargo test -p lance --test stress -- --exact scenario::{} --nocapture",
        vars.join(" "),
        config.scenario.name()
    )
}

/// The command that runs `seed` in this process, for gdb or rr, with the
/// variables `start_child` gives a child: it removes the ones a child does not
/// get, so that none of the shell's comes in.
fn standalone_command(config: &RunConfig, seed: u64) -> String {
    let (mut removed, mut set) = (String::new(), String::new());
    let vars = child_vars(config, seed, "<dir>", 1);
    for (name, value) in &vars {
        match value {
            Some(value) => write!(set, "{name}={} ", shell_quote(value)),
            None => write!(removed, "-u {name} "),
        }
        .unwrap();
    }
    // What a child inherits from the parent, which a fresh shell would not hold.
    for (name, value) in &config.lance_env {
        if !vars.iter().any(|(child, _)| child == name) {
            write!(set, "{name}={} ", shell_quote(value)).unwrap();
        }
    }
    format!(
        "# Or in one process, for gdb or rr, into a fresh directory, where no cgroup limit applies. After a death, go on with {ENV_GENERATION}=2 in the same directory:\n# env {removed}{set}{ENV_STANDALONE}=1 {} --exact child --nocapture\n",
        fs::read_link("/proc/self/exe").map_or_else(
            |_| "<test binary>".into(),
            |exe| shell_quote(&exe.display().to_string())
        )
    )
}

/// The variables the child of `seed` gets, each set or removed: `start_child`
/// gives them to each process of the seed, and the standalone command spells
/// them out. A self-test runs at the defaults and in no mode, and plans as if
/// nothing were selected, so that its defect lands on the steps it expects.
fn child_vars(
    config: &RunConfig,
    seed: u64,
    dir: &str,
    generation: u64,
) -> Vec<(&'static str, Option<String>)> {
    let test = config.self_test;
    let mode = |on: bool| (test.is_none() && on).then(|| "1".to_string());
    let tmp = format!("{dir}/tmp");
    let mut vars = vec![
        (ENV_CHILD, Some(config.scenario.name().to_string())),
        (ENV_SEED, Some(seed.to_string())),
        (ENV_DIR, Some(dir.to_string())),
        (ENV_GENERATION, Some(generation.to_string())),
        (ENV_STEPS, Some(config.steps.to_string())),
        (ENV_QUIET_SECS, Some(config.quiet_secs.to_string())),
        (ENV_SET, Some(config.child_record())),
        (ENV_SELF_TEST, test.map(|test| test.name.to_string())),
        (
            ENV_ONLY,
            config
                .selection
                .given()
                .filter(|_| test.is_none())
                .map(str::to_string),
        ),
        (ENV_CALIBRATE, mode(config.calibrate)),
        (ENV_SERIAL, mode(config.serial)),
        // A replay pins the plan it expects, so a changed generator cannot pass
        // for a failure that went away. A leftover value must not fail other runs.
        (
            ENV_PLAN,
            match (test, &config.replay) {
                (None, Some((_, Some(plan)))) => Some(plan.clone()),
                _ => None,
            },
        ),
        (ENV_CGROUP, None),
        // A temporary file left behind is a leak, so every seed counts them in a
        // directory of its own.
        ("TMPDIR", Some(tmp.clone())),
        ("TMP", Some(tmp.clone())),
        ("TEMP", Some(tmp)),
        ("LANCE_CPU_THREADS", Some("2".to_string())),
        ("RAYON_NUM_THREADS", Some("2".to_string())),
        ("RUST_BACKTRACE", Some("1".to_string())),
    ];
    vars.extend(config.child_env.iter().cloned());
    vars
}

pub struct SeedReport {
    pub seed: u64,
    pub attempt: u64,
    pub class: Option<String>,
    pub detail: String,
    pub dir: PathBuf,
    pub journal: String,
}

fn run_seed(config: &RunConfig, seed: u64, attempt: u64) -> SeedReport {
    let name = match REPEAT.get(&config.knobs) {
        1 => format!("seed-{seed}"),
        _ => format!("seed-{seed}-try{attempt}"),
    };
    let dir = config.run_dir.join(name);
    let report = |class: &str, detail: String| SeedReport {
        seed,
        attempt,
        class: Some(class.to_string()),
        detail,
        dir: dir.clone(),
        journal: String::new(),
    };
    let tmp = dir.join("tmp");
    if let Err(error) = fs::create_dir_all(&tmp) {
        return report(
            "harness",
            format!("cannot create {}: {error}", tmp.display()),
        );
    }
    let replay = replay_command(config, seed);
    let standalone = standalone_command(config, seed);
    if let Err(error) = fs::write(dir.join("replay.sh"), format!("{replay}\n{standalone}")) {
        return report("harness", format!("cannot write replay.sh: {error}"));
    }
    let cgroup = match &config.scope {
        Some(scope) => match scope.seed_cgroup(&format!("seed-{seed}-{attempt}")) {
            Ok(cgroup) => Some(cgroup),
            Err(error) => return report("harness", error),
        },
        None => None,
    };
    // A seed whose process dies goes on in a new process, up to one death per
    // step and a little more for a delayed death between steps.
    let max_generations = config.steps + 2;
    let mut generation = 1;
    let end = loop {
        let journal_start = fs::metadata(dir.join(STEPS_LOG)).map_or(0, |meta| meta.len());
        let oom_kills = cgroup.as_ref().map_or(0, SeedCgroup::oom_kills);
        if let Some(Err(error)) = cgroup.as_ref().map(SeedCgroup::lift_limit) {
            return report("harness", error);
        }
        let mut child = match start_child(config, seed, &dir, generation, cgroup.as_ref()) {
            Ok(child) => child,
            Err(error) => return report("harness", format!("cannot start the child: {error}")),
        };
        let end = watch(&mut child, &dir, config, generation, cgroup.as_ref());
        let Some(cause) = planned_death(&end, &dir, journal_start, oom_kills, cgroup.as_ref())
        else {
            break end;
        };
        if generation == max_generations {
            return report(
                "harness",
                format!("the seed still dies after {max_generations} processes: {cause}"),
            );
        }
        generation += 1;
        let logs: Vec<&str> = config
            .scenario
            .checks()
            .iter()
            .flat_map(|check| check.logs())
            .copied()
            .collect();
        if let Err(error) = note_the_death(&dir, generation, cause, &logs) {
            return report("harness", format!("cannot note a death: {error}"));
        }
    };
    drop(cgroup);
    // They outlive the seed's directory, which a pass removes, as
    // `<stem>-<seed's directory>.<extension>`.
    for artifact in config.scenario.artifacts() {
        let name = dir
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().to_string());
        let kept = match artifact.rsplit_once('.') {
            Some((stem, extension)) => format!("{stem}-{name}.{extension}"),
            None => format!("{artifact}-{name}"),
        };
        fs::copy(dir.join(artifact), config.run_dir.join(kept)).ok();
    }
    let mut report = judge(seed, attempt, dir, end);
    if report.class.is_some() {
        write!(report.detail, " | replay: {replay}").unwrap();
    } else {
        match config.keep {
            Keep::Everything => {}
            Keep::Nothing => {
                fs::remove_dir_all(&report.dir).ok();
            }
            Keep::Logs => {
                fs::remove_dir_all(report.dir.join("dataset")).ok();
                fs::remove_dir_all(report.dir.join("tmp")).ok();
            }
        }
    }
    report
}

fn start_child(
    config: &RunConfig,
    seed: u64,
    dir: &Path,
    generation: u64,
    cgroup: Option<&SeedCgroup>,
) -> Result<Child, String> {
    // On append: the logs hold all the processes of a seed, so the check for
    // panics below sees the ones of a process that died later too.
    let log = |name: &str| {
        File::options()
            .create(true)
            .append(true)
            .open(dir.join(name))
            .map_err(|error| format!("cannot open {name}: {error}"))
    };
    // /proc/self/exe runs this very binary even if a build has replaced the
    // file since the run started.
    let mut command = Command::new("/proc/self/exe");
    command
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .stdin(Stdio::null())
        .stdout(log("stdout.log")?)
        .stderr(log("stderr.log")?)
        // A process group of its own, so that a kill reaches all of it.
        .process_group(0);
    let dir = dir
        .to_str()
        .ok_or_else(|| format!("{} is not UTF-8", dir.display()))?;
    for (name, value) in child_vars(config, seed, dir, generation) {
        match value {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
    command
        .env(ENV_PARENT_PID, std::process::id().to_string())
        .env_remove(ENV_STANDALONE);
    if let Some(cgroup) = cgroup {
        command.env(ENV_CGROUP, cgroup.dir());
        cgroup.add_to(&mut command)?;
    }
    command.spawn().map_err(|error| error.to_string())
}

/// Why a child that was killed may simply go on in a new process: the OOM
/// killer of its cgroup, a death the scenario planned, or the parent's own
/// patience with a thrashing child. Any other end falls to `judge`.
fn planned_death(
    end: &ChildEnd,
    dir: &Path,
    journal_start: u64,
    oom_kills: u64,
    cgroup: Option<&SeedCgroup>,
) -> Option<&'static str> {
    let status = match end {
        ChildEnd::Exited(status) => status,
        ChildEnd::Thrashed => return Some("thrash"),
        ChildEnd::Hung { .. } => return None,
    };
    if status.signal() != Some(libc::SIGKILL) {
        return None;
    }
    // The limit of a step is still set when its process dies. A kill without one
    // comes from another OOM killer, and a check or a retry has no budget to
    // blow: that is no planned death.
    if cgroup.is_some_and(|cgroup| cgroup.is_limited() && cgroup.oom_kills() > oom_kills) {
        return Some("oom_kill");
    }
    let journal = fs::read(dir.join(STEPS_LOG)).ok()?;
    let this_process = journal.get(journal_start as usize..)?;
    String::from_utf8_lossy(this_process)
        .contains(CRASH_FIRED)
        .then_some(CRASH_FIRED)
}

/// A SIGKILL can tear a write at a page boundary, so whatever comes next to a
/// log of a dead process starts a line of its own: the logs of the harness, and
/// `logs` of the checks of the scenario.
fn note_the_death(dir: &Path, generation: u64, cause: &str, logs: &[&str]) -> std::io::Result<()> {
    for name in [STEPS_LOG, CALLS_LOG].iter().chain(logs) {
        let path = dir.join(name);
        if fs::read(&path).is_ok_and(|log| !log.is_empty() && !log.ends_with(b"\n")) {
            File::options().append(true).open(&path)?.write_all(b"\n")?;
        }
    }
    File::options()
        .append(true)
        .open(dir.join(STEPS_LOG))?
        .write_all(format!("generation={generation} observation=killed:{cause}\n").as_bytes())
}

enum ChildEnd {
    Exited(ExitStatus),
    Hung {
        reason: String,
        reaped: bool,
    },
    /// The parent killed a child that was thrashing under its limit.
    Thrashed,
}

/// The CPU time a process used so far, in seconds.
fn cpu_seconds(pid: u32) -> f64 {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    // The fields go on after the parenthesized name, which may contain spaces;
    // the first one there is number 3, and utime and stime are 14 and 15.
    let ticks: u64 = stat
        .rsplit_once(") ")
        .map(|(_, rest)| {
            rest.split(' ')
                .skip(11)
                .take(2)
                .filter_map(|ticks| ticks.parse::<u64>().ok())
                .sum()
        })
        .unwrap_or(0);
    // SAFETY: `sysconf` only reads a system setting.
    let per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    ticks as f64 / per_second.max(1) as f64
}

/// Waits for the child, which appends to its step log at every phase; a log
/// that stops growing means a hang.
fn watch(
    child: &mut Child,
    dir: &Path,
    config: &RunConfig,
    generation: u64,
    cgroup: Option<&SeedCgroup>,
) -> ChildEnd {
    let started = Instant::now();
    let mut log_len = 0;
    let mut last_progress = Instant::now();
    let mut cpu_at_progress = cpu_seconds(child.id());
    let mut timeline = cgroup.and_then(|_| {
        File::options()
            .create(true)
            .append(true)
            .open(dir.join("cgroup.log"))
            .ok()
    });
    let mut thrashing_since: Option<Instant> = None;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return ChildEnd::Exited(status),
            Ok(None) => {}
            Err(error) => {
                return ChildEnd::Hung {
                    reason: format!("cannot poll the child: {error}"),
                    reaped: kill_and_reap(child),
                };
            }
        }
        if let Some(cgroup) = cgroup {
            if let Some(timeline) = timeline.as_mut() {
                writeln!(timeline, "generation={generation} {}", cgroup.sample()).ok();
            }
            thrashing_since = cgroup
                .is_thrashing()
                .then(|| thrashing_since.unwrap_or_else(Instant::now));
            if thrashing_since.is_some_and(|since| since.elapsed() >= THRASH_PATIENCE) {
                fs::write(
                    dir.join(format!("thrash-{generation}-threads.txt")),
                    thread_states(child.id()),
                )
                .ok();
                fs::write(
                    dir.join(format!("thrash-{generation}-cgroup.txt")),
                    cgroup.dump(),
                )
                .ok();
                if gdb_available() {
                    gdb_stacks(child.id(), dir, &format!("thrash-{generation}-stacks"));
                }
                // A child that outlives the kill would run beside the next one.
                if !kill_and_reap(child) {
                    return ChildEnd::Hung {
                        reason: "thrashing".to_string(),
                        reaped: false,
                    };
                }
                return ChildEnd::Thrashed;
            }
        }
        let len = fs::metadata(dir.join(STEPS_LOG)).map_or(0, |meta| meta.len());
        if len != log_len {
            log_len = len;
            last_progress = Instant::now();
            cpu_at_progress = cpu_seconds(child.id());
        }
        // Under a limit a slow operation can stay silent while it works.
        if cgroup.is_some() && last_progress.elapsed() >= config.stall {
            let cpu = cpu_seconds(child.id());
            if cpu - cpu_at_progress >= config.stall.as_secs_f64() * BUSY_SHARE {
                last_progress = Instant::now();
                cpu_at_progress = cpu;
            }
        }
        let hang = if last_progress.elapsed() >= config.stall {
            Some(format!("no progress for {}s", config.stall.as_secs()))
        } else if started.elapsed() >= config.deadline {
            Some(format!(
                "still running after {}s",
                config.deadline.as_secs()
            ))
        } else {
            None
        };
        if let Some(reason) = hang {
            fs::write(dir.join("threads.txt"), thread_states(child.id())).ok();
            if let Some(cgroup) = cgroup {
                fs::write(dir.join("cgroup.txt"), cgroup.dump()).ok();
            }
            if gdb_available() {
                gdb_stacks(child.id(), dir, "stacks");
            }
            return ChildEnd::Hung {
                reason,
                reaped: kill_and_reap(child),
            };
        }
        std::thread::sleep(POLL);
    }
}

/// Kills the child's process group and waits for the child a bounded time: a
/// child stuck in an uninterruptible wait must not hang the parent too.
fn kill_and_reap(child: &mut Child) -> bool {
    // SAFETY: `kill` only sends a signal. The child leads its own process
    // group, so the negative pid names that group and nothing else.
    unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL) };
    child.kill().ok();
    let started = Instant::now();
    while started.elapsed() < REAP_TIMEOUT {
        if let Ok(Some(_)) = child.try_wait() {
            return true;
        }
        std::thread::sleep(POLL);
    }
    false
}

/// One line per thread: its id, state, kernel wait channel and name.
fn thread_states(pid: u32) -> String {
    let mut lines = String::new();
    let Ok(tasks) = fs::read_dir(format!("/proc/{pid}/task")) else {
        return lines;
    };
    let mut tids: Vec<String> = tasks
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    tids.sort();
    for tid in tids {
        let read = |file: &str| {
            fs::read_to_string(format!("/proc/{pid}/task/{tid}/{file}")).unwrap_or_default()
        };
        let stat = read("stat");
        // The state follows the parenthesized name, which may contain spaces.
        let state = stat
            .rsplit_once(") ")
            .and_then(|(_, rest)| rest.split(' ').next())
            .unwrap_or("?");
        writeln!(
            lines,
            "{tid} {state} {} {}",
            read("wchan").trim(),
            read("comm").trim()
        )
        .unwrap();
    }
    lines
}

fn gdb_available() -> bool {
    Command::new("gdb")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn gdb_stacks(pid: u32, dir: &Path, name: &str) {
    let (Ok(stdout), Ok(stderr)) = (
        File::create(dir.join(format!("{name}.txt"))),
        File::create(dir.join(format!("{name}.err"))),
    ) else {
        return;
    };
    let Ok(mut gdb) = Command::new("gdb")
        .args(["-p", &pid.to_string(), "-batch", "-nx"])
        .args(["-ex", "set pagination off", "-ex", "thread apply all bt"])
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
    else {
        return;
    };
    let started = Instant::now();
    while started.elapsed() < GDB_TIMEOUT {
        if let Ok(Some(_)) = gdb.try_wait() {
            return;
        }
        std::thread::sleep(POLL);
    }
    gdb.kill().ok();
    gdb.wait().ok();
}

fn read_lossy(path: &Path, limit: usize) -> String {
    let mut bytes = fs::read(path).unwrap_or_default();
    bytes.truncate(limit);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The verdict on one child. A panic comes first, since it is the likeliest
/// cause of whatever else went wrong, then the class the child wrote, then a
/// hang; the detail names everything that happened.
fn judge(seed: u64, attempt: u64, dir: PathBuf, end: ChildEnd) -> SeedReport {
    let journal = read_lossy(&dir.join(STEPS_LOG), usize::MAX);
    let stderr = read_lossy(&dir.join("stderr.log"), STDERR_LIMIT);
    let last_line = journal.lines().last().unwrap_or("").to_string();
    let result = journal
        .lines()
        .rev()
        .find(|line| field(line, "result").is_some())
        .map(str::to_string);
    let failed = result
        .as_deref()
        .filter(|line| field(line, "result") == Some("fail"));
    // The message follows the location on the next line.
    let panic = stderr
        .lines()
        .position(|line| line.contains("panicked at"))
        .map(|at| {
            stderr
                .lines()
                .skip(at)
                .take(2)
                .collect::<Vec<_>>()
                .join(" ")
        });
    let hung = match &end {
        ChildEnd::Hung { reason, reaped } => Some(format!(
            "hung: {reason}{}",
            if *reaped { "" } else { "; it survived SIGKILL" }
        )),
        ChildEnd::Exited(_) | ChildEnd::Thrashed => None,
    };
    let mut notes: Vec<String> = Vec::new();
    let class = if let Some(panic) = &panic {
        notes.push(panic.clone());
        notes.extend(failed.map(str::to_string));
        "panic".to_string()
    } else if let Some(line) = failed {
        notes.push(line.to_string());
        // A fail line torn before its step lost its class, or the end of it: the
        // verdict never ended.
        field(line, "class")
            .filter(|_| field(line, "step").is_some())
            .unwrap_or("crash")
            .to_string()
    } else if hung.is_some() {
        notes.push(format!("last: {last_line}"));
        "hang".to_string()
    } else {
        match (&end, result.as_deref()) {
            (ChildEnd::Exited(status), Some(line))
                if status.success() && field(line, "result") == Some("pass") =>
            {
                return SeedReport {
                    seed,
                    attempt,
                    class: None,
                    detail: String::new(),
                    dir,
                    journal,
                };
            }
            (ChildEnd::Exited(status), _) => {
                notes.push(format!("exited with {status}; last: {last_line}"));
            }
            (ChildEnd::Thrashed, _) => {
                notes.push(format!("killed for thrashing; last: {last_line}"));
            }
            (ChildEnd::Hung { .. }, _) => unreachable!("a hang is handled above"),
        }
        "crash".to_string()
    };
    notes.extend(hung);
    SeedReport {
        seed,
        attempt,
        class: Some(class),
        detail: notes.join("; "),
        dir,
        journal,
    }
}

/// The value of `key=` in a journal line. Only the part before ` detail=` is
/// searched: a detail is free text.
pub fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let head = line.split(" detail=").next().unwrap_or(line);
    head.split(' ')
        .find_map(|token| token.strip_prefix(key)?.strip_prefix('='))
}

#[derive(Default)]
struct Coverage {
    planned: usize,
    /// The seeds that planned it.
    seeds: BTreeSet<u64>,
    fired: usize,
    /// How many of those found their gate closed: a hold point only.
    waited: usize,
    /// `<outcome>+committed` or `<outcome>+uncommitted`, counted.
    outcomes: BTreeMap<String, usize>,
}

struct Summary {
    header: String,
    scenario: &'static dyn Scenario,
    runs: usize,
    outcomes: BTreeMap<String, usize>,
    observations: BTreeMap<String, usize>,
    /// The seeds that forced each pair.
    pair_seeds: BTreeMap<String, BTreeSet<u64>>,
    coverage: BTreeMap<String, Coverage>,
    holds: BTreeMap<String, Coverage>,
    failures: Vec<SeedReport>,
    /// Classes and observations the lists of `engine.rs` do not declare.
    undeclared: BTreeSet<String>,
}

impl Summary {
    fn new(config: &RunConfig, seeds: &[u64]) -> Self {
        let mut header = format!(
            "scenario={} run_dir={} steps={} repeat={} seeds={}..={}",
            config.scenario.name(),
            config.run_dir.display(),
            config.steps,
            REPEAT.get(&config.knobs),
            seeds.first().copied().unwrap_or_default(),
            seeds.last().copied().unwrap_or_default()
        );
        if let Some(only) = config.selection.given() {
            write!(header, " only={only}").unwrap();
        }
        // A knob off its default changes what a seed runs or what passes.
        for (title, class) in [("plan knobs", Class::Plan), ("judge knobs", Class::Judge)] {
            let knobs = config.knobs.off_default(class, &[]);
            if !knobs.is_empty() {
                write!(header, "\n{title}: {}", knobs.join(" ")).unwrap();
            }
        }
        Self {
            header,
            scenario: config.scenario,
            runs: 0,
            outcomes: BTreeMap::new(),
            observations: BTreeMap::new(),
            pair_seeds: BTreeMap::new(),
            coverage: BTreeMap::new(),
            holds: BTreeMap::new(),
            failures: Vec::new(),
            undeclared: BTreeSet::new(),
        }
    }

    /// Limits of Lance's own that a run planned often enough and that never bit:
    /// a limit that has stopped working would otherwise stay green. Per seed
    /// this could be chance, as a watcher can miss a short spill.
    fn dead_limits(&self) -> Vec<String> {
        let mut limits: BTreeMap<String, Coverage> = BTreeMap::new();
        for (kind, coverage) in &self.coverage {
            let mut parts = kind.split(':');
            if let (Some("memory"), Some(op), Some(limit @ ("pool" | "fts_memory"))) =
                (parts.next(), parts.next(), parts.next())
            {
                let entry = limits.entry(format!("{op}:{limit}")).or_default();
                entry.planned += coverage.planned;
                entry.seeds.extend(&coverage.seeds);
                entry.fired += coverage.fired;
            }
        }
        limits
            .into_iter()
            .filter(|(_, limit)| limit.seeds.len() >= DEAD_LIMIT_SEEDS && limit.fired == 0)
            .map(|(kind, limit)| {
                format!(
                    "{kind} planned={} seeds={} fired=0",
                    limit.planned,
                    limit.seeds.len()
                )
            })
            .collect()
    }

    /// Hold points a run planned often enough and that never fired: nothing
    /// waited there, so what the plan meant to race did not race. Per seed this
    /// could be chance, as an operation may find nothing to do.
    fn dead_holds(&self) -> Vec<String> {
        self.holds
            .iter()
            .filter(|(_, hold)| hold.seeds.len() >= DEAD_HOLD_SEEDS && hold.fired == 0)
            .map(|(kind, hold)| {
                format!(
                    "{kind} planned={} seeds={} fired=0",
                    hold.planned,
                    hold.seeds.len()
                )
            })
            .collect()
    }

    /// Whether the run passes: no seed failed, nothing that enough seeds planned
    /// stayed dead, and no name was undeclared.
    fn passed(&self) -> bool {
        self.failures.is_empty()
            && self.dead_limits().is_empty()
            && self.dead_holds().is_empty()
            && self.dead_pairs().is_empty()
            && self.undeclared.is_empty()
    }

    /// Pairs of operations a run forced often enough that never met: the first
    /// always committed before the second landed, so the rebase it is there for
    /// never ran. Per seed this could be chance, as an operation may find nothing
    /// to do.
    fn dead_pairs(&self) -> Vec<String> {
        let count = |prefix: &str, pair: &str| {
            self.observations
                .get(&format!("{prefix}:{pair}"))
                .copied()
                .unwrap_or_default()
        };
        self.pair_seeds
            .iter()
            .filter(|(pair, seeds)| seeds.len() >= DEAD_PAIR_SEEDS && count("pair_met", pair) == 0)
            .map(|(pair, seeds)| {
                format!(
                    "{pair} planned={} seeds={} met=0",
                    count("pair_planned", pair),
                    seeds.len()
                )
            })
            .collect()
    }

    fn add(&mut self, report: SeedReport) {
        self.runs += 1;
        let mut concluded_last = BTreeMap::new();
        for (index, line) in report.journal.lines().enumerate() {
            let head = line.split(" detail=").next().unwrap_or(line);
            if head.split(' ').any(|token| token == "verdict")
                && let Some(step) = field(line, "step")
            {
                concluded_last.insert(step, index);
            }
        }
        for (index, line) in report.journal.lines().enumerate() {
            if let Some(observation) = field(line, "observation") {
                if !engine::declared_observation(observation) {
                    self.undeclared.insert(format!("observation {observation}"));
                }
                *self
                    .observations
                    .entry(observation.to_string())
                    .or_default() += 1;
                if let Some(pair) = observation.strip_prefix("pair_planned:") {
                    self.pair_seeds
                        .entry(pair.to_string())
                        .or_default()
                        .insert(report.seed);
                }
            }
            if let Some(hold) = field(line, "hold") {
                let coverage = self.holds.entry(hold.to_string()).or_default();
                coverage.planned += 1;
                coverage.seeds.insert(report.seed);
                coverage.fired += usize::from(field(line, "fired").is_some_and(|f| f != "0"));
                coverage.waited += usize::from(field(line, "waited").is_some_and(|w| w != "0"));
            }
            let head = line.split(" detail=").next().unwrap_or(line);
            if !head.split(' ').any(|token| token == "verdict") {
                continue;
            }
            // A death before the end of a step gets it concluded a second time,
            // and the later verdict knows more.
            if field(line, "step")
                .and_then(|step| concluded_last.get(step))
                .is_some_and(|last| *last != index)
            {
                continue;
            }
            let outcome = format!(
                "{}+{}",
                field(line, "outcome").unwrap_or("?"),
                match field(line, "committed") {
                    Some("true") => "committed",
                    _ => "uncommitted",
                }
            );
            *self.outcomes.entry(outcome.clone()).or_default() += 1;
            if let Some(incident) = field(line, "incident").filter(|kind| *kind != "none") {
                let coverage = self.coverage.entry(incident.to_string()).or_default();
                coverage.planned += 1;
                coverage.seeds.insert(report.seed);
                coverage.fired += usize::from(field(line, "fired").is_some_and(|f| f != "0"));
                *coverage.outcomes.entry(outcome).or_default() += 1;
            }
        }
        if let Some(class) = &report.class {
            if !engine::CLASSES.contains(&class.as_str()) {
                self.undeclared.insert(format!("class {class}"));
            }
            self.failures.push(report);
        }
    }

    fn render(&self) -> String {
        let mut text = String::new();
        writeln!(
            text,
            "{}\nruns={} failed={}",
            self.header,
            self.runs,
            self.failures.len()
        )
        .unwrap();
        let sections = [
            ("outcomes", &self.outcomes),
            ("observations", &self.observations),
        ];
        for (title, counts) in sections {
            writeln!(text, "{title}:").unwrap();
            for (name, count) in counts {
                writeln!(text, "  {name}={count}").unwrap();
            }
        }
        writeln!(text, "incidents:").unwrap();
        for (kind, coverage) in &self.coverage {
            let outcomes: Vec<String> = coverage
                .outcomes
                .iter()
                .map(|(outcome, count)| format!("{outcome}={count}"))
                .collect();
            writeln!(
                text,
                "  {kind} planned={} fired={} {}",
                coverage.planned,
                coverage.fired,
                outcomes.join(" ")
            )
            .unwrap();
        }
        for limit in self.dead_limits() {
            writeln!(text, "dead limit: {limit}").unwrap();
        }
        writeln!(text, "holds:").unwrap();
        for (kind, hold) in &self.holds {
            writeln!(
                text,
                "  {kind} planned={} fired={} waited={}",
                hold.planned, hold.fired, hold.waited
            )
            .unwrap();
        }
        for hold in self.dead_holds() {
            writeln!(text, "dead hold: {hold}").unwrap();
        }
        for pair in self.dead_pairs() {
            writeln!(text, "dead pair: {pair}").unwrap();
        }
        // Something the guide cannot explain, which fails the run like a dead line.
        for name in &self.undeclared {
            writeln!(text, "undeclared: {name}").unwrap();
        }
        if !self.failures.is_empty() {
            writeln!(text, "failed:").unwrap();
            for report in &self.failures {
                writeln!(
                    text,
                    "  seed={} try={} class={} {}\n    artifacts: {}",
                    report.seed,
                    report.attempt,
                    report.class.as_deref().unwrap_or("?"),
                    report.detail,
                    report.dir.display()
                )
                .unwrap();
            }
            writeln!(
                text,
                "replay a seed with the command after `replay:`; scenario {}",
                self.scenario.name()
            )
            .unwrap();
        }
        text
    }
}

#[test]
fn selection_dead_pairs_count_seeds_not_rounds() {
    let report = |seed: u64| SeedReport {
        seed,
        attempt: 1,
        class: None,
        detail: String::new(),
        dir: PathBuf::new(),
        // A focus forces the same pair in several rounds of a seed.
        journal: "t=1 step=3 observation=pair_planned:append:cleanup\n\
                  t=2 step=5 observation=pair_planned:append:cleanup\n"
            .to_string(),
    };
    let mut summary = Summary {
        header: String::new(),
        scenario: scenario::concurrency::SCENARIO,
        runs: 0,
        outcomes: BTreeMap::new(),
        observations: BTreeMap::new(),
        pair_seeds: BTreeMap::new(),
        coverage: BTreeMap::new(),
        holds: BTreeMap::new(),
        failures: Vec::new(),
        undeclared: BTreeSet::new(),
    };
    summary.add(report(5));
    summary.add(report(5));
    assert!(summary.dead_pairs().is_empty(), "one seed, even repeated");
    assert!(summary.passed());
    // A name the lists of `engine.rs` do not declare fails the run, as a dead
    // line does.
    // Spelled in two parts, so that the scan of the sources does not take it.
    let lost = ["pair", "_lost:append:cleanup"].concat();
    summary.add(SeedReport {
        journal: format!(
            "t=1 step=2 observation=gate_timeout\n\
             t=2 step=2 observation=killed_at:append:fragment_begin\n\
             t=3 step=2 observation=fault_abs\n\
             t=4 step=2 observation={lost}\n"
        ),
        ..report(5)
    });
    assert!(!summary.passed());
    summary.add(SeedReport {
        class: Some("fail".to_string()),
        ..report(5)
    });
    assert_eq!(
        summary.undeclared,
        BTreeSet::from([
            "class fail".to_string(),
            "observation pair_lost:append:cleanup".to_string()
        ])
    );
    assert!(
        summary
            .render()
            .contains("undeclared: observation pair_lost:append:cleanup\n"),
        "{}",
        summary.render()
    );
    summary.add(report(6));
    assert_eq!(
        summary.dead_pairs(),
        ["append:cleanup planned=8 seeds=2 met=0"]
    );
}

/// A verdict that a death tore is a crash: a fail line without its class, or with
/// the class cut short before its step, names no class of its own.
#[test]
fn replay_judges_a_torn_verdict_as_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("stderr.log"), "").unwrap();
    for (result, class) in [
        ("result=fail", "crash"),
        ("result=fail class=lost_wr", "crash"),
        (
            "result=fail class=lost_write step=3 detail=gone",
            "lost_write",
        ),
    ] {
        fs::write(
            dir.path().join(STEPS_LOG),
            format!("t=1 step=1 verdict incident=none\nt=2 {result}"),
        )
        .unwrap();
        let end = ChildEnd::Exited(ExitStatus::from_raw(1 << 8));
        let report = judge(1, 1, dir.path().to_path_buf(), end);
        assert_eq!(report.class.as_deref(), Some(class), "{result}");
    }
}

#[test]
fn selection_dead_holds_and_limits_count_seeds_not_tries() {
    // One seed plans a hold point twice and a limit once, and neither fires.
    let report = |seed: u64, attempt: u64| {
        SeedReport {
        seed,
        attempt,
        class: None,
        detail: String::new(),
        dir: PathBuf::new(),
        journal: "t=1 step=2 hold=hold:append:hold:put:data:before worker=1 rank=1 until=0 fired=0 waited=0\n\
                  t=2 step=4 hold=hold:append:hold:put:data:before worker=2 rank=3 until=0 fired=0 waited=0\n\
                  t=3 step=5 verdict incident=memory:compact:pool:128 fired=0 outcome=ok committed=true\n"
            .to_string(),
    }
    };
    let mut summary = Summary {
        header: String::new(),
        scenario: scenario::concurrency::SCENARIO,
        runs: 0,
        outcomes: BTreeMap::new(),
        observations: BTreeMap::new(),
        pair_seeds: BTreeMap::new(),
        coverage: BTreeMap::new(),
        holds: BTreeMap::new(),
        failures: Vec::new(),
        undeclared: BTreeSet::new(),
    };
    for attempt in 1..=3 {
        summary.add(report(5, attempt));
    }
    summary.add(report(6, 1));
    assert!(
        summary.dead_holds().is_empty(),
        "two seeds, one tried 3 times"
    );
    assert!(
        summary.dead_limits().is_empty(),
        "two seeds, one tried 3 times"
    );
    summary.add(report(7, 1));
    assert_eq!(
        summary.dead_holds(),
        ["hold:append:hold:put:data:before planned=10 seeds=3 fired=0"]
    );
    assert_eq!(
        summary.dead_limits(),
        ["compact:pool planned=5 seeds=3 fired=0"]
    );
}

/// A run of `scenario` with the variables of `env` and no others.
#[cfg(test)]
fn config_of(
    scenario: &'static dyn Scenario,
    self_test: Option<&'static SelfTest>,
    env: &Env,
) -> RunConfig {
    prepare(scenario, self_test, env).expect("a run the selection keeps")
}

/// The variables a replay line leaves its command, as a shell and `env` read it:
/// a pair before `env` is the shell's, `-u` removes a variable, a pair sets one.
#[cfg(test)]
fn replay_vars(line: &str) -> Env {
    let (vars, _) = line.split_once(" cargo test ").expect("a replay line");
    let mut env = BTreeMap::new();
    let mut tokens = vars.split(' ');
    while let Some(token) = tokens.next() {
        match token {
            "env" => {}
            "-u" => {
                env.remove(tokens.next().expect("a variable after -u"));
            }
            pair => {
                let (name, value) = pair.split_once('=').expect("a variable");
                env.insert(name, value.to_string());
            }
        }
    }
    env.into_iter().collect()
}

#[test]
fn selection_leaves_no_knob_unchecked() {
    let faults = scenario::storage_faults::SCENARIO;
    let test = faults.self_tests()[0];
    // Even a test the selection leaves out fails on a key no scenario has.
    for self_test in [None, Some(test)] {
        let env = Env::from([(ENV_ONLY, "soak"), (ENV_SET, "stepz=3")]);
        let run = std::panic::catch_unwind(AssertUnwindSafe(|| {
            prepare(faults, self_test, &env).is_some()
        }));
        assert!(run.is_err(), "{self_test:?}");
        let env = Env::from([(ENV_ONLY, "soak"), (ENV_SET, "steps=3")]);
        assert!(prepare(faults, self_test, &env).is_none());
    }
}

/// A replay line, run in a shell, sets up the very run it came from: the same
/// knobs, modes and plan, and so the same line again.
#[rstest::rstest]
#[case::soak(
    "soak",
    &[("LANCE_STRESS_SOAK_SECS", "4321"), ("LANCE_STRESS_SEEDS", "3"), ("LANCE_STRESS_REPEAT", "2"), ("LANCE_IO_THREADS", "4")],
    "soak secs=4321 "
)]
#[case::focus(
    "storage_faults",
    &[(ENV_ONLY, "op:optimize_indices"), (ENV_CALIBRATE, "1")],
    " focus=optimize_indices\n"
)]
#[case::serial(
    "concurrency",
    &[(ENV_SET, "concurrency.gate_limit=20s,stall=200s"), (ENV_SERIAL, "1")],
    "\nknobs concurrency.gate_limit=20s\n"
)]
fn replay_line_runs_what_its_seed_ran(
    #[case] name: &str,
    #[case] vars: &[(&str, &str)],
    #[case] planned: &str,
) {
    let scenario = scenario::by_name(name).expect("a scenario of the registry");
    let env: Env = vars
        .iter()
        .map(|(name, value)| (*name, value.to_string()))
        .collect();
    let config = config_of(scenario, None, &env);
    let line = replay_command(&config, 7);
    for (name, value) in vars
        .iter()
        .filter(|(name, _)| name.starts_with("LANCE_") && !name.starts_with("LANCE_STRESS_"))
    {
        assert!(
            line.contains(&format!(" {name}={value} ")),
            "{name}: {line}"
        );
    }
    // A replay runs one seed, and pins no derived knob no source set.
    for absent in ["SOAK_SECS", "seeds=", "repeat=", "deadline="] {
        assert!(!line.contains(absent), "{absent}: {line}");
    }
    let replay = config_of(scenario, None, &replay_vars(&line));
    assert_eq!(replay.knobs.record(), config.knobs.record());
    assert_eq!(
        (replay.steps, replay.stall, replay.deadline),
        (config.steps, config.stall, config.deadline)
    );
    assert_eq!(
        (replay.calibrate, replay.serial, replay.selection.given()),
        (config.calibrate, config.serial, config.selection.given())
    );
    assert_eq!(replay_command(&replay, 7), line);
    // A mode that is off is pinned as well, against one left in the shell.
    for (name, on) in [
        (ENV_CALIBRATE, config.calibrate),
        (ENV_SERIAL, config.serial),
    ] {
        assert!(
            line.contains(&format!(" {name}={} ", u8::from(on))),
            "{name}: {line}"
        );
    }
    // The plan the line pins is the one its child draws, and that plan follows
    // the knobs and modes.
    let child: Env = child_vars(&replay, 7, "", 1)
        .into_iter()
        .filter_map(|(name, value)| Some((name, value?)))
        .collect();
    let rendered = ChildPlan::read(scenario, &child).rendered;
    let pinned = replay.replay.as_ref().and_then(|(_, plan)| plan.clone());
    assert_eq!(pinned, Some(engine::fingerprint(&rendered)));
    assert!(rendered.contains(planned), "{rendered}");
    // It is the plan the variables of the run draw, as a child reads them without
    // `child_vars`, and the child of the replay gets it to check against.
    let modes: Vec<(&'static str, &str)> = [ENV_ONLY, ENV_CALIBRATE]
        .into_iter()
        .filter_map(|name| Some((name, env.get(name)?)))
        .collect();
    let drawn = ChildPlan::read(
        scenario,
        &engine::child_vars_of(scenario, 7, &config.knobs, &modes),
    )
    .rendered;
    assert_eq!(pinned, Some(engine::fingerprint(&drawn)));
    assert!(
        child.get(ENV_PLAN) == pinned.as_deref(),
        "{ENV_PLAN}: {:?}",
        child.get(ENV_PLAN)
    );
    // In the shell of the replay, a short form of a knob it pins is refused.
    let short = match name {
        "soak" => "LANCE_STRESS_SOAK_SECS",
        _ => "LANCE_STRESS_STEPS",
    };
    let shell = replay_vars(&format!("{short}=60 {line}"));
    let error = Knobs::resolve(scenario, &shell).unwrap_err();
    assert!(error.contains(short), "{error}");
}

/// A deadline that suits the scenario that runs is taken, whatever another one
/// needs: a short form, which takes no prefix, sets the deadline of all of them.
#[test]
fn selection_takes_a_deadline_that_suits_what_runs() {
    let env = Env::from([("LANCE_STRESS_DEADLINE_SECS", "200")]);
    let config = config_of(scenario::storage_faults::SCENARIO, None, &env);
    assert_eq!(config.deadline, Duration::from_secs(200));
    let soak = scenario::soak::SCENARIO;
    assert!(prepare(soak, Some(soak.self_tests()[0]), &env).is_some());
    let refused =
        std::panic::catch_unwind(AssertUnwindSafe(|| prepare(soak, None, &env).is_some()))
            .expect_err("a soak of 3 min with a deadline of 200 s")
            .downcast::<String>()
            .expect("a message");
    assert!(
        refused.contains("is shorter than the 4m that soak needs"),
        "{refused}"
    );
}

/// What the shell of a replay holds besides the line does not reach the run: not
/// a stall, a deadline or a file of knobs, which the record leaves out, and not a
/// selection the run did not have.
#[test]
fn replay_line_leaves_the_shell_behind() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("knobs.txt");
    fs::write(&file, "storage_faults.stall=31s\n").unwrap();
    let shell = format!(
        "{}=31 {}=100 {}={} {ENV_ONLY}=op:append",
        crate::knobs::ENV_STALL_SECS,
        crate::knobs::ENV_DEADLINE_SECS,
        crate::knobs::ENV_SET_FILE,
        file.display()
    );
    let faults = scenario::storage_faults::SCENARIO;
    let config = config_of(faults, None, &Env::default());
    let line = replay_command(&config, 7);
    let replay = config_of(faults, None, &replay_vars(&format!("{shell} {line}")));
    assert_eq!(
        (replay.stall, replay.deadline, replay.selection.given()),
        (config.stall, config.deadline, None)
    );
    assert_eq!(replay_command(&replay, 7), line);
    // A self-test the shell's selection leaves out still runs from its line.
    let test = faults.self_tests()[0];
    let line = replay_command(&config_of(faults, Some(test), &Env::default()), 7);
    let env = replay_vars(&format!("{ENV_ONLY}=soak {shell} {line}"));
    assert!(prepare(faults, Some(test), &env).is_some(), "{line}");
    // It keeps a deadline the run set, and that alone.
    let env = Env::from([(ENV_SET, "deadline=2h,steps=12")]);
    let line = replay_command(&config_of(faults, Some(test), &env), 7);
    assert!(
        line.contains(&format!(
            " {ENV_SET}=storage_faults.deadline=2h cargo test "
        )),
        "{line}"
    );
    let replay = config_of(faults, Some(test), &replay_vars(&line));
    assert_eq!(replay.deadline, Duration::from_secs(7200));
}

/// `env.txt` takes a value that is not UTF-8 as near as it reads, and no secret.
#[test]
fn replay_records_the_environment_whatever_it_holds() {
    use std::os::unix::ffi::OsStringExt;
    let recorded = recorded_vars([
        (
            OsString::from("RUST_LOG"),
            OsString::from_vec(vec![b'i', 0xff]),
        ),
        (OsString::from("LANCE_IO_THREADS"), OsString::from("4")),
        (OsString::from("LANCE_S3_SECRET_KEY"), OsString::from("x")),
        (OsString::from("HOME"), OsString::from("/home")),
    ]);
    assert_eq!(
        recorded,
        [
            ("LANCE_IO_THREADS".to_string(), "4".to_string()),
            ("RUST_LOG".to_string(), "i\u{fffd}".to_string()),
        ]
    );
}

#[test]
fn replay_runs_a_self_test_standalone_as_its_child() {
    let faults = scenario::storage_faults::SCENARIO;
    let env = Env::from([
        (ENV_SET, "storage_faults.steps=5"),
        (ENV_ONLY, "op:optimize_indices"),
        (ENV_SERIAL, "1"),
        ("LANCE_IO_THREADS", "4"),
    ]);
    let config = config_of(faults, None, &env);
    let line = standalone_command(&config, 7);
    for expected in [
        " LANCE_IO_THREADS=4 ".to_string(),
        format!(" {ENV_SET}={} ", config.knobs.record()),
        format!(" {ENV_ONLY}=op:optimize_indices "),
        format!(" {ENV_SERIAL}=1 "),
        // What a child does not get, the shell may not give it either.
        format!(" -u {ENV_SELF_TEST} "),
        format!(" -u {ENV_CALIBRATE} "),
        format!(" -u {ENV_PLAN} "),
        " LANCE_CPU_THREADS=2 ".to_string(),
    ] {
        assert!(line.contains(&expected), "{expected}: {line}");
    }
    // A self-test runs at the defaults and in no mode, whatever the shell sets.
    let test = faults.self_tests()[0];
    let line = standalone_command(&config_of(faults, Some(test), &env), 7);
    for expected in [
        format!(" {ENV_SELF_TEST}={} ", test.name),
        format!(" {ENV_SET}= "),
        format!(" -u {ENV_ONLY} "),
        format!(" -u {ENV_SERIAL} "),
    ] {
        assert!(line.contains(&expected), "{expected}: {line}");
    }
}

#[test]
fn replay_of_a_self_test_takes_only_its_deadline_from_the_knobs() {
    let faults = scenario::storage_faults::SCENARIO;
    let test = faults.self_tests()[0];
    let env = Env::from([("LANCE_STRESS_SET", "deadline=2h,steps=12,stall=5m")]);
    let config = config_of(faults, Some(test), &env);
    assert_eq!(config.steps, 3);
    assert_eq!(config.stall, Duration::from_secs(15));
    assert_eq!(config.deadline, Duration::from_secs(7200));
    // Without one of its own, the deadline is that of a plan of 3 steps.
    let config = config_of(faults, Some(test), &Env::default());
    assert_eq!(config.deadline, Duration::from_secs(300));
}

/// A stall or a deadline no source set stays out of the replay line, which derives
/// it again: a knob edited in the line moves it as it would in a fresh run.
#[test]
fn replay_derives_what_no_source_set() {
    let slow = scenario::slow_storage::SCENARIO;
    let line = replay_command(&config_of(slow, None, &Env::default()), 7);
    let edited = line.replace("slow_storage.settle=2s", "slow_storage.settle=1m");
    assert_ne!(edited, line);
    let config = config_of(slow, None, &replay_vars(&edited));
    // 3 slowed calls of 3 s, a settle of 60 s and 20 s of quiet: 89 s of silence.
    assert_eq!(
        (config.stall, config.deadline),
        (Duration::from_secs(99), Duration::from_secs(120 + 8 * 89))
    );
}

#[test]
fn replay_keeps_a_pinned_stall_below_its_derived_default() {
    let concurrency = scenario::concurrency::SCENARIO;
    // A longer gate limit raises the stall it derives...
    let env = Env::from([(ENV_SET, "concurrency.gate_limit=30s")]);
    assert_eq!(
        config_of(concurrency, None, &env).stall,
        Duration::from_secs(450)
    );
    // ...but a stall a replay pins stays, held to the old rule alone.
    let env = Env::from([(ENV_SET, "concurrency.gate_limit=30s,stall=60s")]);
    let config = config_of(concurrency, None, &env);
    assert_eq!(config.stall, Duration::from_secs(60));
    let summary = Summary::new(&config, &[1]);
    assert!(
        summary
            .header
            .contains("\nplan knobs: concurrency.gate_limit=30s")
            && summary
                .header
                .contains("\njudge knobs: concurrency.stall=1m"),
        "{}",
        summary.header
    );
}
