// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The parent side: runs each seed in a child process, watches it for
//! progress, and turns what the child left behind into a verdict.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use crate::incidents::CRASH_FIRED;
use crate::limits::{ENV_CGROUP, RunScope, SeedCgroup};
use crate::scenario;
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
const ENV_STALL_SECS: &str = "LANCE_STRESS_STALL_SECS";
const ENV_DEADLINE_SECS: &str = "LANCE_STRESS_DEADLINE_SECS";
pub const ENV_CALIBRATE: &str = "LANCE_STRESS_CALIBRATE";
/// Workers run one operation at a time, in the order of their ranks: a failure of
/// `concurrency` that goes away with this needs a race.
pub const ENV_SERIAL: &str = "LANCE_STRESS_SERIAL";

pub const PLAN_LOG: &str = "plan.log";
pub const STEPS_LOG: &str = "steps.log";
pub const CALLS_LOG: &str = "calls.log";
pub const MEMORY_LOG: &str = "memory.log";
/// Reads have a log of their own: the watchdog counts the growth of the step log
/// as progress, and a busy reader must not hide a hung writer.
pub const READS_LOG: &str = "reads.log";

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
const DEAD_LIMIT_PLANNED: usize = 3;
const DEAD_HOLD_PLANNED: usize = 3;
/// Seeds, not rounds: 16 seeds force each pair twice, and one seed never judges
/// a pair, whether it is repeated or a focus forces the pair in several rounds.
const DEAD_PAIR_SEEDS: usize = 2;
/// How long a commit may keep retrying: a hold point can eat the default 30 s.
const COMMIT_RETRY_TIMEOUT_VARIABLE: &str = "LANCE_COMMIT_RETRY_TIMEOUT_SECS";
const COMMIT_RETRY_TIMEOUT_SECS: &str = "120";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScenarioKind {
    StorageFaults,
    Cancellation,
    ProcessKill,
    MemoryLimits,
    HeapGrowth,
    Concurrency,
}

impl ScenarioKind {
    pub const ALL: [Self; 6] = [
        Self::StorageFaults,
        Self::Cancellation,
        Self::ProcessKill,
        Self::MemoryLimits,
        Self::HeapGrowth,
        Self::Concurrency,
    ];

    /// Whether Lance reports its memory in this scenario, and the checks read it.
    pub fn observes_memory(self) -> bool {
        matches!(self, Self::ProcessKill | Self::MemoryLimits)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::StorageFaults => "storage_faults",
            Self::Cancellation => "cancellation",
            Self::ProcessKill => "process_kill",
            Self::MemoryLimits => "memory_limits",
            Self::HeapGrowth => "heap_growth",
            Self::Concurrency => "concurrency",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

/// A defect a child injects on purpose, to show that the parent reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfTest {
    Hang,
    Panic,
    TaskLeak,
    TmpLeak,
    ModelMismatch,
    ColumnMismatch,
    Orphan,
    /// A SIGKILL nobody planned: the seed must end there.
    ForeignKill,
    /// A file a manifest references goes away while the process is dead.
    KillThenMissingFile,
    /// A step goes over its limit: the seed goes on and ends well.
    OomRecovery,
    /// A step hangs under a limit.
    HangUnderLimit,
    /// An index build reports no memory events, though it must.
    MemoryEvents,
    /// Heap goes missing on every cycle of a repeated workload.
    HeapLeak,
    /// An update lands as two commits, each with half of its rows.
    TornSnapshot,
    /// A commit restores the version the round started from.
    LostWrite,
    /// A reader that should see the latest version reads the first of the round.
    StaleRead,
    /// A reader's pinned version turns into another between two reads.
    SnapshotChanged,
    /// A row is appended a second time.
    DuplicateKey,
    /// The history of a row's lineage is not what the rows say.
    Lineage,
    /// A hold point waits, with no limit, for what cannot happen, while a reader
    /// keeps reading: its reads must not count as progress.
    GateHang,
}

impl SelfTest {
    const ALL: [Self; 20] = [
        Self::Hang,
        Self::Panic,
        Self::TaskLeak,
        Self::TmpLeak,
        Self::ModelMismatch,
        Self::ColumnMismatch,
        Self::Orphan,
        Self::ForeignKill,
        Self::KillThenMissingFile,
        Self::OomRecovery,
        Self::HangUnderLimit,
        Self::MemoryEvents,
        Self::HeapLeak,
        Self::TornSnapshot,
        Self::LostWrite,
        Self::StaleRead,
        Self::SnapshotChanged,
        Self::DuplicateKey,
        Self::Lineage,
        Self::GateHang,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Hang => "hang",
            Self::Panic => "panic",
            Self::TaskLeak => "task_leak",
            Self::TmpLeak => "tmp_leak",
            Self::ModelMismatch => "model_mismatch",
            Self::ColumnMismatch => "column_mismatch",
            Self::Orphan => "orphan",
            Self::ForeignKill => "foreign_kill",
            Self::KillThenMissingFile => "kill_then_missing_file",
            Self::OomRecovery => "oom_recovery",
            Self::HangUnderLimit => "hang_under_limit",
            Self::MemoryEvents => "memory_events",
            Self::HeapLeak => "heap_leak",
            Self::TornSnapshot => "torn_snapshot",
            Self::LostWrite => "lost_write",
            Self::StaleRead => "stale_read",
            Self::SnapshotChanged => "snapshot_changed",
            Self::DuplicateKey => "duplicate_key",
            Self::Lineage => "lineage",
            Self::GateHang => "gate_hang",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|test| test.name() == name)
    }

    fn scenario(self) -> ScenarioKind {
        match self {
            Self::ForeignKill | Self::KillThenMissingFile | Self::MemoryEvents => {
                ScenarioKind::ProcessKill
            }
            Self::OomRecovery | Self::HangUnderLimit => ScenarioKind::MemoryLimits,
            Self::HeapLeak => ScenarioKind::HeapGrowth,
            Self::TornSnapshot
            | Self::LostWrite
            | Self::StaleRead
            | Self::SnapshotChanged
            | Self::DuplicateKey
            | Self::Lineage
            | Self::GateHang => ScenarioKind::Concurrency,
            _ => ScenarioKind::StorageFaults,
        }
    }

    /// The failure classes the parent may report for this defect; none means
    /// the seed must pass.
    fn expected_classes(self) -> &'static [&'static str] {
        match self {
            Self::Hang | Self::HangUnderLimit | Self::GateHang => &["hang"],
            Self::Panic => &["panic"],
            Self::TaskLeak => &["task_leak"],
            Self::TmpLeak => &["tmp_leak"],
            Self::ModelMismatch | Self::ColumnMismatch => &["content"],
            Self::Orphan => &["orphans"],
            Self::ForeignKill => &["crash"],
            // Which check meets the missing file first depends on the file.
            Self::KillThenMissingFile => &["validate", "missing_files", "unreadable"],
            Self::OomRecovery => &[],
            Self::MemoryEvents => &["memory_events"],
            Self::HeapLeak => &["heap_growth"],
            Self::TornSnapshot => &["torn_snapshot"],
            Self::LostWrite => &["lost_write"],
            Self::StaleRead => &["stale_read"],
            Self::SnapshotChanged => &["snapshot_changed"],
            Self::DuplicateKey => &["duplicate_key"],
            Self::Lineage => &["lineage"],
        }
    }
}

struct RunConfig {
    kind: ScenarioKind,
    /// This invocation's own directory: a replay or a concurrent run never
    /// touches another run's artifacts.
    run_dir: PathBuf,
    steps: u64,
    stall: Duration,
    deadline: Duration,
    quiet_secs: u64,
    keep: Keep,
    repeat: u64,
    self_test: Option<SelfTest>,
    /// Where the children of a run under memory limits live.
    scope: Option<RunScope>,
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
    fn new(kind: ScenarioKind, self_test: Option<SelfTest>) -> Self {
        // Self-tests ignore the tuning variables: their timing is part of what
        // they check.
        // A hold's wait goes into the first try of a commit, which sets the unit of
        // Lance's backoff: 10 s there can become a silent sleep of over a minute.
        let default_stall = match kind {
            ScenarioKind::Concurrency => 150,
            _ => 60,
        };
        let (steps, quiet_secs, stall_secs) = match self_test {
            Some(_) => (3, 2, 15),
            None => (
                env_u64(ENV_STEPS, 8),
                20,
                env_u64(ENV_STALL_SECS, default_stall),
            ),
        };
        // Repeated cycles are the point of this scenario: it has a size of its own.
        let steps = match kind {
            ScenarioKind::HeapGrowth => scenario::heap_growth_steps(),
            _ => steps,
        };
        let default_deadline = match kind {
            ScenarioKind::Concurrency => 300 + 120 * steps,
            _ => 120 + 60 * steps,
        };
        assert!(
            stall_secs >= quiet_secs + 10,
            "{ENV_STALL_SECS}={stall_secs} must exceed the {quiet_secs}s a step may wait for quiet by 10s"
        );
        let name = match self_test {
            Some(test) => format!("self-test-{}", test.name()),
            None => kind.name().to_string(),
        };
        let run_dir = artifacts_root().join(format!(
            "{name}-{}-{}",
            chrono::Utc::now().format("%Y%m%d-%H%M%S"),
            std::process::id()
        ));
        let replaying = std::env::var(ENV_SEED).is_ok();
        let keep = if self_test.is_some() || replaying {
            Keep::Everything
        } else {
            match std::env::var("LANCE_STRESS_KEEP").as_deref() {
                Ok("1") => Keep::Everything,
                Ok("logs") => Keep::Logs,
                _ => Keep::Nothing,
            }
        };
        let scope = (kind == ScenarioKind::MemoryLimits).then(|| {
            RunScope::create(&name).unwrap_or_else(|error| {
                panic!("the memory limit scenarios cannot run here: {error}")
            })
        });
        Self {
            kind,
            run_dir,
            steps,
            stall: Duration::from_secs(stall_secs),
            deadline: Duration::from_secs(env_u64(ENV_DEADLINE_SECS, default_deadline)),
            quiet_secs,
            keep,
            repeat: env_u64("LANCE_STRESS_REPEAT", 1).max(1),
            self_test,
            scope,
        }
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}")),
        Err(_) => default,
    }
}

/// The artifacts root, absolute and without `.` or `..`. Lance spells local
/// paths that way, and the incidents tell a dataset's directories apart by
/// that spelling.
fn artifacts_root() -> PathBuf {
    let root = std::env::var_os("LANCE_STRESS_ARTIFACTS")
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

/// Runs every requested seed of a scenario and fails if any seed failed.
pub fn run(kind: ScenarioKind) {
    // libtest has no verdict for a test that did not run: it passes. A replay
    // names its scenario, so there it would pass for a seed it never ran.
    let selection = Selection::from_env();
    if !selection.runs(kind) {
        let replay = std::env::var_os(ENV_SEED).is_some() && std::env::var_os(ENV_PLAN).is_some();
        assert!(
            !replay,
            "{ENV_PLAN} replays a seed of {}, which {ENV_ONLY} leaves out",
            kind.name()
        );
        return;
    }
    let config = RunConfig::new(kind, None);
    // A focus takes every other step, and in `concurrency` the forced pairs, from
    // the third round on: with fewer, it would fall on nothing.
    let least = match kind {
        ScenarioKind::Concurrency => 3,
        _ => 2,
    };
    assert!(
        selection.focus().is_empty() || config.steps >= least,
        "the operations of {ENV_ONLY} need {ENV_STEPS} of at least {least} in {}",
        kind.name()
    );
    fs::create_dir_all(&config.run_dir).expect("create the run directory");
    record_environment(&config.run_dir);
    let seeds: Vec<u64> = if std::env::var(ENV_SEED).is_ok() {
        vec![env_u64(ENV_SEED, 0)]
    } else {
        let base = env_u64("LANCE_STRESS_SEED_BASE", 1);
        (base..base + env_u64("LANCE_STRESS_SEEDS", 16)).collect()
    };
    let mut summary = Summary::new(&config, &seeds);
    for &seed in &seeds {
        for attempt in 1..=config.repeat {
            summary.add(run_seed(&config, seed, attempt));
            // Written after every seed, so an interrupted run keeps its verdicts.
            fs::write(config.run_dir.join("summary.txt"), summary.render())
                .expect("write the summary");
        }
    }
    assert!(
        summary.failures.is_empty()
            && summary.dead_limits().is_empty()
            && summary.dead_holds().is_empty()
            && summary.dead_pairs().is_empty(),
        "{}",
        summary.render()
    );
}

/// Runs one seed with a defect injected and checks that it is reported.
pub fn check_self_test(test: SelfTest) {
    if !Selection::from_env().runs_self_test(test.scenario()) {
        return;
    }
    let config = RunConfig::new(test.scenario(), Some(test));
    fs::create_dir_all(&config.run_dir).expect("create the run directory");
    let report = run_seed(&config, 1, 1);
    let expected = test.expected_classes();
    assert!(
        match report.class.as_deref() {
            Some(class) => expected.contains(&class),
            None => expected.is_empty(),
        },
        "the {} self-test was reported as {:?}: {} (artifacts in {})",
        test.name(),
        report.class,
        report.detail,
        report.dir.display()
    );
    let resumed = report.journal.contains("generation=2 ");
    match test {
        SelfTest::ForeignKill => assert!(
            !resumed,
            "a death nobody planned must end the seed, see {}",
            report.dir.display()
        ),
        SelfTest::KillThenMissingFile => assert!(
            resumed,
            "the death was planned, so a later process must look at the dataset, see {}",
            report.dir.display()
        ),
        SelfTest::OomRecovery => assert!(
            resumed
                && report.journal.contains("observation=killed:oom_kill")
                && report.journal.contains("outcome=killed"),
            "the step must have died of its limit and been concluded later, see {}",
            report.dir.display()
        ),
        SelfTest::HangUnderLimit => {
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
        _ => {}
    }
    if test == SelfTest::GateHang {
        // The stall, not the deadline, must have ended it, with the reader going
        // on long after the last line of the step log.
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
    if matches!(
        test,
        SelfTest::Hang | SelfTest::HangUnderLimit | SelfTest::GateHang
    ) {
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
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(name, _)| {
            ["LANCE_", "RUST_", "TOKIO_", "RAYON_"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
                && !["KEY", "SECRET", "TOKEN", "PASSWORD"]
                    .iter()
                    .any(|word| name.contains(word))
        })
        .collect();
    vars.sort();
    for (name, value) in vars {
        writeln!(text, "{name}={value}").unwrap();
    }
    fs::write(run_dir.join("env.txt"), text).expect("write env.txt");
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

/// The command that reruns `seed` with the same settings: the plan size, the
/// fingerprint of the plan, non-default limits, and every Lance tuning
/// variable the run inherited.
fn replay_command(config: &RunConfig, seed: u64) -> String {
    if let Some(test) = config.self_test {
        return format!(
            "cargo test -p lance --test stress -- --exact harness_reports_{} --nocapture",
            test.name()
        );
    }
    let mut vars: Vec<String> = std::env::vars()
        .filter(|(name, _)| name.starts_with("LANCE_") && !name.starts_with("LANCE_STRESS_"))
        .map(|(name, value)| format!("{name}={}", shell_quote(&value)))
        .collect();
    vars.sort();
    vars.push(format!("{ENV_SEED}={seed}"));
    vars.push(format!("{ENV_STEPS}={}", config.steps));
    vars.push(format!(
        "{ENV_PLAN}={}",
        scenario::plan_fingerprint(config.kind, seed, config.steps as usize)
    ));
    for name in [
        ENV_STALL_SECS,
        ENV_DEADLINE_SECS,
        ENV_CALIBRATE,
        ENV_SERIAL,
        ENV_ONLY,
    ] {
        if let Ok(value) = std::env::var(name) {
            vars.push(format!("{name}={}", shell_quote(&value)));
        }
    }
    format!(
        "{} cargo test -p lance --test stress -- --exact {} --nocapture",
        vars.join(" "),
        config.kind.name()
    )
}

struct SeedReport {
    seed: u64,
    attempt: u64,
    class: Option<String>,
    detail: String,
    dir: PathBuf,
    journal: String,
}

fn run_seed(config: &RunConfig, seed: u64, attempt: u64) -> SeedReport {
    let name = match config.repeat {
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
    let mut standalone_vars = String::new();
    if config.kind.observes_memory() {
        standalone_vars.push_str("LANCE_MEMORY_OBSERVE=1 ");
    }
    if std::env::var(ENV_CALIBRATE).is_ok_and(|value| value == "1") {
        write!(standalone_vars, "{ENV_CALIBRATE}=1 ").unwrap();
    }
    if config.kind == ScenarioKind::Concurrency {
        write!(
            standalone_vars,
            "{COMMIT_RETRY_TIMEOUT_VARIABLE}={COMMIT_RETRY_TIMEOUT_SECS} "
        )
        .unwrap();
    }
    if std::env::var(ENV_SERIAL).is_ok_and(|value| value == "1") {
        write!(standalone_vars, "{ENV_SERIAL}=1 ").unwrap();
    }
    if let Ok(only) = std::env::var(ENV_ONLY) {
        write!(standalone_vars, "{ENV_ONLY}={} ", shell_quote(&only)).unwrap();
    }
    let standalone = format!(
        "# Or in one process, for gdb or rr, into a fresh directory, where no cgroup limit applies. After a death, go on with {ENV_GENERATION}=2 in the same directory:\n# {standalone_vars}{ENV_STANDALONE}=1 {ENV_CHILD}={} {ENV_SEED}={seed} {ENV_GENERATION}=1 {ENV_STEPS}={} {ENV_QUIET_SECS}={} {ENV_DIR}=<dir> TMPDIR=<dir>/tmp {} --exact child --nocapture\n",
        config.kind.name(),
        config.steps,
        config.quiet_secs,
        fs::read_link("/proc/self/exe")
            .map_or_else(|_| "<test binary>".into(), |exe| exe.display().to_string())
    );
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
        if let Err(error) = note_the_death(&dir, generation, cause) {
            return report("harness", format!("cannot note a death: {error}"));
        }
    };
    drop(cgroup);
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
    let tmp = dir.join("tmp");
    // /proc/self/exe runs this very binary even if a build has replaced the
    // file since the run started.
    let mut command = Command::new("/proc/self/exe");
    command
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(ENV_CHILD, config.kind.name())
        .env(ENV_SEED, seed.to_string())
        .env(ENV_DIR, dir)
        .env(ENV_GENERATION, generation.to_string())
        .env(ENV_PARENT_PID, std::process::id().to_string())
        .env(ENV_STEPS, config.steps.to_string())
        .env(ENV_QUIET_SECS, config.quiet_secs.to_string())
        .env_remove(ENV_STANDALONE)
        .env_remove(ENV_PLAN)
        // A temporary file left behind is a leak, so every seed counts them in
        // a directory of its own.
        .env("TMPDIR", &tmp)
        .env("TMP", &tmp)
        .env("TEMP", &tmp)
        .env("LANCE_CPU_THREADS", "2")
        .env("RAYON_NUM_THREADS", "2")
        .env("RUST_BACKTRACE", "1")
        .stdin(Stdio::null())
        .stdout(log("stdout.log")?)
        .stderr(log("stderr.log")?)
        // A process group of its own, so that a kill reaches all of it.
        .process_group(0);
    // A self-test plans as if nothing were selected: its defect lands on the
    // steps it expects.
    match config.self_test {
        Some(test) => command.env(ENV_SELF_TEST, test.name()).env_remove(ENV_ONLY),
        None => command.env_remove(ENV_SELF_TEST),
    };
    // A replay pins the plan it expects, so a changed generator cannot pass
    // for a failure that went away. A leftover value must not fail other runs.
    if config.self_test.is_none()
        && std::env::var(ENV_SEED).is_ok()
        && let Ok(plan) = std::env::var(ENV_PLAN)
    {
        command.env(ENV_PLAN, plan);
    }
    // Read once, when a process first needs it, so it is set from the start.
    if config.kind.observes_memory() && config.self_test != Some(SelfTest::MemoryEvents) {
        command.env("LANCE_MEMORY_OBSERVE", "1");
    } else {
        command.env_remove("LANCE_MEMORY_OBSERVE");
    }
    // The steps of this scenario set the pool, and the switch would turn it off.
    if config.kind == ScenarioKind::MemoryLimits {
        command
            .env_remove("LANCE_MEM_POOL_SIZE")
            .env_remove("LANCE_BYPASS_SPILLING");
    }
    if config.kind == ScenarioKind::Concurrency {
        command.env(COMMIT_RETRY_TIMEOUT_VARIABLE, COMMIT_RETRY_TIMEOUT_SECS);
    }
    match cgroup {
        Some(cgroup) => {
            command.env(ENV_CGROUP, cgroup.dir());
            cgroup.add_to(&mut command)?;
        }
        None => {
            command.env_remove(ENV_CGROUP);
        }
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
/// log of a dead process starts a line of its own.
fn note_the_death(dir: &Path, generation: u64, cause: &str) -> std::io::Result<()> {
    for name in [STEPS_LOG, CALLS_LOG, MEMORY_LOG] {
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
        // A fail line without a class still fails.
        field(line, "class").unwrap_or("fail").to_string()
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
    fired: usize,
    /// How many of those found their gate closed: a hold point only.
    waited: usize,
    /// `<outcome>+committed` or `<outcome>+uncommitted`, counted.
    outcomes: BTreeMap<String, usize>,
}

struct Summary {
    header: String,
    kind: ScenarioKind,
    runs: usize,
    outcomes: BTreeMap<String, usize>,
    observations: BTreeMap<String, usize>,
    /// The seeds that forced each pair.
    pair_seeds: BTreeMap<String, BTreeSet<u64>>,
    coverage: BTreeMap<String, Coverage>,
    holds: BTreeMap<String, Coverage>,
    failures: Vec<SeedReport>,
}

impl Summary {
    fn new(config: &RunConfig, seeds: &[u64]) -> Self {
        let mut header = format!(
            "scenario={} run_dir={} steps={} repeat={} seeds={}..={}",
            config.kind.name(),
            config.run_dir.display(),
            config.steps,
            config.repeat,
            seeds.first().copied().unwrap_or_default(),
            seeds.last().copied().unwrap_or_default()
        );
        if let Ok(only) = std::env::var(ENV_ONLY) {
            write!(header, " only={only}").unwrap();
        }
        Self {
            header,
            kind: config.kind,
            runs: 0,
            outcomes: BTreeMap::new(),
            observations: BTreeMap::new(),
            pair_seeds: BTreeMap::new(),
            coverage: BTreeMap::new(),
            holds: BTreeMap::new(),
            failures: Vec::new(),
        }
    }

    /// Limits of Lance's own that a run planned often enough and that never bit:
    /// a limit that has stopped working would otherwise stay green. Per seed
    /// this could be chance, as a watcher can miss a short spill.
    fn dead_limits(&self) -> Vec<String> {
        let mut limits: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for (kind, coverage) in &self.coverage {
            let mut parts = kind.split(':');
            if let (Some("memory"), Some(op), Some(limit @ ("pool" | "fts_memory"))) =
                (parts.next(), parts.next(), parts.next())
            {
                let entry = limits.entry(format!("{op}:{limit}")).or_default();
                entry.0 += coverage.planned;
                entry.1 += coverage.fired;
            }
        }
        limits
            .into_iter()
            .filter(|(_, (planned, fired))| *planned >= DEAD_LIMIT_PLANNED && *fired == 0)
            .map(|(limit, (planned, _))| format!("{limit} planned={planned} fired=0"))
            .collect()
    }

    /// Hold points a run planned often enough and that never fired: nothing
    /// waited there, so what the plan meant to race did not race. Per seed this
    /// could be chance, as an operation may find nothing to do.
    fn dead_holds(&self) -> Vec<String> {
        self.holds
            .iter()
            .filter(|(_, hold)| hold.planned >= DEAD_HOLD_PLANNED && hold.fired == 0)
            .map(|(kind, hold)| format!("{kind} planned={} fired=0", hold.planned))
            .collect()
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
                coverage.fired += usize::from(field(line, "fired").is_some_and(|f| f != "0"));
                *coverage.outcomes.entry(outcome).or_default() += 1;
            }
        }
        if report.class.is_some() {
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
                self.kind.name()
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
        kind: ScenarioKind::Concurrency,
        runs: 0,
        outcomes: BTreeMap::new(),
        observations: BTreeMap::new(),
        pair_seeds: BTreeMap::new(),
        coverage: BTreeMap::new(),
        holds: BTreeMap::new(),
        failures: Vec::new(),
    };
    summary.add(report(5));
    summary.add(report(5));
    assert!(summary.dead_pairs().is_empty(), "one seed, even repeated");
    summary.add(report(6));
    assert_eq!(
        summary.dead_pairs(),
        ["append:cleanup planned=6 seeds=2 met=0"]
    );
}
