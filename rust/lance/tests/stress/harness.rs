// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The parent side: runs each seed in a child process, watches it for
//! progress, and turns what the child left behind into a verdict.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use crate::scenario;

pub const ENV_CHILD: &str = "LANCE_STRESS_CHILD";
pub const ENV_SEED: &str = "LANCE_STRESS_SEED";
pub const ENV_DIR: &str = "LANCE_STRESS_DIR";
pub const ENV_PARENT_PID: &str = "LANCE_STRESS_PARENT_PID";
pub const ENV_STEPS: &str = "LANCE_STRESS_STEPS";
pub const ENV_QUIET_SECS: &str = "LANCE_STRESS_QUIET_SECS";
pub const ENV_SELF_TEST: &str = "LANCE_STRESS_SELF_TEST";
pub const ENV_PLAN: &str = "LANCE_STRESS_PLAN";
pub const ENV_STANDALONE: &str = "LANCE_STRESS_STANDALONE";
const ENV_STALL_SECS: &str = "LANCE_STRESS_STALL_SECS";
const ENV_DEADLINE_SECS: &str = "LANCE_STRESS_DEADLINE_SECS";

pub const PLAN_LOG: &str = "plan.log";
pub const STEPS_LOG: &str = "steps.log";
pub const CALLS_LOG: &str = "calls.log";

const POLL: Duration = Duration::from_millis(200);
const GDB_TIMEOUT: Duration = Duration::from_secs(60);
const REAP_TIMEOUT: Duration = Duration::from_secs(30);
const STDERR_LIMIT: usize = 8 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenarioKind {
    StorageFaults,
    Cancellation,
}

impl ScenarioKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::StorageFaults => "storage_faults",
            Self::Cancellation => "cancellation",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [Self::StorageFaults, Self::Cancellation]
            .into_iter()
            .find(|kind| kind.name() == name)
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
}

impl SelfTest {
    const ALL: [Self; 7] = [
        Self::Hang,
        Self::Panic,
        Self::TaskLeak,
        Self::TmpLeak,
        Self::ModelMismatch,
        Self::ColumnMismatch,
        Self::Orphan,
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
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|test| test.name() == name)
    }

    /// The failure class the parent must report for this defect.
    fn expected_class(self) -> &'static str {
        match self {
            Self::Hang => "hang",
            Self::Panic => "panic",
            Self::TaskLeak => "task_leak",
            Self::TmpLeak => "tmp_leak",
            Self::ModelMismatch | Self::ColumnMismatch => "content",
            Self::Orphan => "orphans",
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
    keep: bool,
    repeat: u64,
    self_test: Option<SelfTest>,
}

impl RunConfig {
    fn new(kind: ScenarioKind, self_test: Option<SelfTest>) -> Self {
        // Self-tests ignore the tuning variables: their timing is part of what
        // they check.
        let (steps, quiet_secs, stall_secs) = match self_test {
            Some(_) => (3, 2, 15),
            None => (env_u64(ENV_STEPS, 8), 20, env_u64(ENV_STALL_SECS, 60)),
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
        Self {
            kind,
            run_dir,
            steps,
            stall: Duration::from_secs(stall_secs),
            deadline: Duration::from_secs(env_u64(ENV_DEADLINE_SECS, 120 + 60 * steps)),
            quiet_secs,
            keep: self_test.is_some()
                || replaying
                || std::env::var("LANCE_STRESS_KEEP").is_ok_and(|keep| keep == "1"),
            repeat: env_u64("LANCE_STRESS_REPEAT", 1).max(1),
            self_test,
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
    let config = RunConfig::new(kind, None);
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
    assert!(summary.failures.is_empty(), "{}", summary.render());
}

/// Runs one seed with a defect injected and checks that it is reported.
pub fn check_self_test(test: SelfTest) {
    let config = RunConfig::new(ScenarioKind::StorageFaults, Some(test));
    fs::create_dir_all(&config.run_dir).expect("create the run directory");
    let report = run_seed(&config, 1, 1);
    assert_eq!(
        report.class.as_deref(),
        Some(test.expected_class()),
        "the {} self-test was reported as {:?}: {} (artifacts in {})",
        test.name(),
        report.class,
        report.detail,
        report.dir.display()
    );
    if test == SelfTest::Hang {
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
    for name in [ENV_STALL_SECS, ENV_DEADLINE_SECS] {
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
    let standalone = format!(
        "# Or in one process, for gdb or rr, into a fresh directory:\n# {ENV_STANDALONE}=1 {ENV_CHILD}={} {ENV_SEED}={seed} {ENV_STEPS}={} {ENV_QUIET_SECS}={} {ENV_DIR}=<dir> TMPDIR=<dir>/tmp {} --exact child --nocapture\n",
        config.kind.name(),
        config.steps,
        config.quiet_secs,
        fs::read_link("/proc/self/exe")
            .map_or_else(|_| "<test binary>".into(), |exe| exe.display().to_string())
    );
    if let Err(error) = fs::write(dir.join("replay.sh"), format!("{replay}\n{standalone}")) {
        return report("harness", format!("cannot write replay.sh: {error}"));
    }
    let (stdout, stderr) = match (
        File::create(dir.join("stdout.log")),
        File::create(dir.join("stderr.log")),
    ) {
        (Ok(stdout), Ok(stderr)) => (stdout, stderr),
        (Err(error), _) | (_, Err(error)) => {
            return report(
                "harness",
                format!("cannot create the child's logs: {error}"),
            );
        }
    };

    // /proc/self/exe runs this very binary even if a build has replaced the
    // file since the run started.
    let mut command = Command::new("/proc/self/exe");
    command
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(ENV_CHILD, config.kind.name())
        .env(ENV_SEED, seed.to_string())
        .env(ENV_DIR, &dir)
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
        .stdout(stdout)
        .stderr(stderr)
        // A process group of its own, so that a kill reaches all of it.
        .process_group(0);
    match config.self_test {
        Some(test) => command.env(ENV_SELF_TEST, test.name()),
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
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return report("harness", format!("cannot start the child: {error}")),
    };
    let end = watch(&mut child, &dir, config);
    let mut report = judge(seed, attempt, dir, end);
    if report.class.is_none() && !config.keep {
        fs::remove_dir_all(&report.dir).ok();
    }
    if report.class.is_some() {
        write!(report.detail, " | replay: {replay}").unwrap();
    }
    report
}

enum ChildEnd {
    Exited(ExitStatus),
    Hung { reason: String, reaped: bool },
}

/// Waits for the child, which appends to its step log at every phase; a log
/// that stops growing means a hang.
fn watch(child: &mut Child, dir: &Path, config: &RunConfig) -> ChildEnd {
    let started = Instant::now();
    let mut log_len = 0;
    let mut last_progress = Instant::now();
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
        let len = fs::metadata(dir.join(STEPS_LOG)).map_or(0, |meta| meta.len());
        if len != log_len {
            log_len = len;
            last_progress = Instant::now();
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
            if gdb_available() {
                gdb_stacks(child.id(), dir);
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

fn gdb_stacks(pid: u32, dir: &Path) {
    let (Ok(stdout), Ok(stderr)) = (
        File::create(dir.join("stacks.txt")),
        File::create(dir.join("stacks.err")),
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
        ChildEnd::Exited(_) => None,
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
    /// `<outcome>+committed` or `<outcome>+uncommitted`, counted.
    outcomes: BTreeMap<String, usize>,
}

struct Summary {
    header: String,
    kind: ScenarioKind,
    runs: usize,
    outcomes: BTreeMap<String, usize>,
    observations: BTreeMap<String, usize>,
    coverage: BTreeMap<String, Coverage>,
    failures: Vec<SeedReport>,
}

impl Summary {
    fn new(config: &RunConfig, seeds: &[u64]) -> Self {
        let header = format!(
            "scenario={} run_dir={} steps={} repeat={} seeds={}..={}",
            config.kind.name(),
            config.run_dir.display(),
            config.steps,
            config.repeat,
            seeds.first().copied().unwrap_or_default(),
            seeds.last().copied().unwrap_or_default()
        );
        Self {
            header,
            kind: config.kind,
            runs: 0,
            outcomes: BTreeMap::new(),
            observations: BTreeMap::new(),
            coverage: BTreeMap::new(),
            failures: Vec::new(),
        }
    }

    fn add(&mut self, report: SeedReport) {
        self.runs += 1;
        for line in report.journal.lines() {
            if let Some(observation) = field(line, "observation") {
                *self
                    .observations
                    .entry(observation.to_string())
                    .or_default() += 1;
            }
            let head = line.split(" detail=").next().unwrap_or(line);
            if !head.split(' ').any(|token| token == "verdict") {
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
