// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory limits: the cgroups the children of a run live in, and the limit a
//! child puts on one step.
//!
//! A run gets one delegated systemd scope, and each seed a cgroup of its own
//! below it. An empty cgroup stays, so the parent can read what the kernel
//! counted after a child died. A scope exists only while a process is in it; a
//! `sleep` in a leaf of its own holds it, out of the OOM killer's reach, since
//! that killer only picks among the processes of the seed's cgroup.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io::Write as _;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lance_core::utils::tracing::TRACE_MEMORY;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

use crate::incidents::IncidentState;

pub const ENV_CGROUP: &str = "LANCE_STRESS_CGROUP";

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const SCOPE_TIMEOUT: Duration = Duration::from_secs(10);
const REMOVE_TIMEOUT: Duration = Duration::from_secs(5);
/// `systemd-oomd` acts at 50% for 20 seconds, on the whole user service. The
/// parent ends a child well before that. A child with next to no pressure of
/// its own is not the cause, and may just be hung.
const THRASH_PRESSURE: f64 = 40.0;
const OWN_PRESSURE_FLOOR: f64 = 5.0;

/// The cgroup of a process, by pid or `self`.
fn cgroup_of(process: &str) -> Result<PathBuf, String> {
    let text = fs::read_to_string(format!("/proc/{process}/cgroup"))
        .map_err(|error| format!("cannot read the cgroup of {process}: {error}"))?;
    let path = text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| format!("{process} is not in a cgroup v2 hierarchy: {text:?}"))?;
    Ok(Path::new(CGROUP_ROOT).join(path.trim_start_matches('/')))
}

fn write_file(path: &Path, value: &str) -> Result<(), String> {
    fs::write(path, value).map_err(|error| format!("cannot write {value:?} to {path:?}: {error}"))
}

/// A file of `key value` lines, such as `memory.events` or `memory.stat`.
fn read_flat(path: &Path) -> BTreeMap<String, u64> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            Some((key.to_string(), value.parse().ok()?))
        })
        .collect()
}

fn read_u64(path: &Path) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(0)
}

/// What a cgroup holds.
#[derive(Clone, Copy, Debug)]
pub struct Usage {
    pub current: u64,
    pub anon: u64,
    pub file: u64,
}

fn usage_of(dir: &Path) -> Usage {
    let stat = read_flat(&dir.join("memory.stat"));
    Usage {
        current: read_u64(&dir.join("memory.current")),
        anon: stat.get("anon").copied().unwrap_or(0),
        file: stat.get("file").copied().unwrap_or(0),
    }
}

/// `full avg10` of a `memory.pressure` file: the share of the last 10 seconds
/// in which every task of the cgroup waited for memory, in percent.
fn full_avg10(path: &Path) -> f64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("full avg10="))
                .and_then(|rest| rest.split(' ').next())
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(0.0)
}

/// The cgroup of the user's service manager above `dir`, if any.
fn user_service_of(dir: &Path) -> Option<PathBuf> {
    dir.ancestors()
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.as_bytes().starts_with(b"user@"))
        })
        .map(Path::to_path_buf)
}

/// The delegated scope of a run, and the process that keeps it alive.
pub struct RunScope {
    holder: Child,
    dir: PathBuf,
    /// The cgroup `systemd-oomd` watches on this machine, if the scope is
    /// below one.
    user_service: Option<PathBuf>,
}

impl RunScope {
    pub fn create(name: &str) -> Result<Self, String> {
        // A process joins a cgroup only with write access to the common ancestor
        // of where it is and where it goes, which root owns outside the service.
        let own = cgroup_of("self")?;
        if user_service_of(&own).is_none() {
            return Err(format!(
                "this process is in {}, outside the user's service manager: start it with `systemd-run --user --scope cargo test ...`",
                own.display()
            ));
        }
        // The pages of this binary are charged to the first cgroup that touches
        // them. Read here, they do not count against a child's limit, and the
        // child's own limit cannot evict its code and make it thrash.
        let mut binary = File::open("/proc/self/exe")
            .map_err(|error| format!("cannot open this binary: {error}"))?;
        std::io::copy(&mut binary, &mut std::io::sink())
            .map_err(|error| format!("cannot read this binary: {error}"))?;

        let unit = format!("lance-stress-{}-{name}", std::process::id());
        let mut command = Command::new("systemd-run");
        command
            .args(["--user", "--scope", "--quiet", "--collect"])
            .arg(format!("--unit={unit}"))
            // Without `continue`, systemd would stop the scope at the first kill.
            .args(["-p", "Delegate=yes", "-p", "OOMPolicy=continue"])
            .args(["sleep", "infinity"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        // SAFETY: `prctl` is async-signal-safe and only changes an attribute of
        // this process, which `exec` keeps. It takes the holder down with the
        // parent, whatever way the parent ends.
        unsafe {
            command.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong);
                Ok(())
            });
        }
        let mut holder = command.spawn().map_err(|error| {
            format!("the memory limit scenarios need `systemd-run --user`: {error}")
        })?;
        match Self::adopt(&mut holder, &unit) {
            Ok(dir) => {
                let user_service = user_service_of(&dir);
                Ok(Self {
                    holder,
                    dir,
                    user_service,
                })
            }
            Err(error) => {
                holder.kill().ok();
                holder.wait().ok();
                Err(error)
            }
        }
    }

    fn adopt(holder: &mut Child, unit: &str) -> Result<PathBuf, String> {
        let pid = holder.id().to_string();
        let started = Instant::now();
        let scope = format!("{unit}.scope");
        // `systemd-run` execs the command once the scope exists. Until then the
        // process is in its parent's cgroup, which can be a scope too, so only
        // the exact name of this unit counts.
        let dir = loop {
            if let Ok(Some(status)) = holder.try_wait() {
                let mut stderr = String::new();
                if let Some(pipe) = holder.stderr.as_mut() {
                    std::io::Read::read_to_string(pipe, &mut stderr).ok();
                }
                return Err(format!(
                    "systemd-run ended with {status}: {}",
                    stderr.trim()
                ));
            }
            let comm = fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
            let dir = cgroup_of(&pid)?;
            if comm.trim() == "sleep" && dir.file_name().is_some_and(|name| name == scope.as_str())
            {
                break dir;
            }
            if started.elapsed() >= SCOPE_TIMEOUT {
                return Err(format!(
                    "{pid} is still {comm:?} in {} after {SCOPE_TIMEOUT:?}",
                    dir.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let procs = fs::read_to_string(dir.join("cgroup.procs")).unwrap_or_default();
        if procs.trim() != pid {
            return Err(format!(
                "{} holds {procs:?}, not just the holder {pid}",
                dir.display()
            ));
        }
        // A cgroup that hands controllers to its children holds no process.
        let leaf = dir.join("holder");
        fs::create_dir(&leaf).map_err(|error| format!("cannot create {leaf:?}: {error}"))?;
        write_file(&leaf.join("cgroup.procs"), &pid)?;
        write_file(&dir.join("cgroup.subtree_control"), "+memory +pids")?;
        Ok(dir)
    }

    /// The cgroup of one try of a seed: without swap, or memory would move
    /// there instead of the limit killing anything.
    pub fn seed_cgroup(&self, name: &str) -> Result<SeedCgroup, String> {
        let dir = self.dir.join(name);
        fs::create_dir(&dir).map_err(|error| format!("cannot create {dir:?}: {error}"))?;
        write_file(&dir.join("memory.swap.max"), "0")?;
        Ok(SeedCgroup {
            dir,
            user_service: self.user_service.clone(),
        })
    }
}

impl Drop for RunScope {
    fn drop(&mut self) {
        fs::write(self.dir.join("cgroup.kill"), "1").ok();
        self.holder.kill().ok();
        self.holder.wait().ok();
    }
}

/// The parent's view of the cgroup of a seed. Its child processes come and go;
/// the cgroup and its counters stay.
pub struct SeedCgroup {
    dir: PathBuf,
    user_service: Option<PathBuf>,
}

impl SeedCgroup {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Makes a process join this cgroup right before it execs.
    pub fn add_to(&self, command: &mut Command) -> Result<(), String> {
        let procs = CString::new(self.dir.join("cgroup.procs").as_os_str().as_bytes())
            .map_err(|error| format!("a cgroup path with a NUL: {error}"))?;
        // SAFETY: `open`, `write` and `close` are async-signal-safe, and the
        // path was allocated before the fork.
        unsafe {
            command.pre_exec(move || {
                let fd = libc::open(procs.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC);
                if fd < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                let written = libc::write(fd, b"0".as_ptr().cast(), 1);
                let error = std::io::Error::last_os_error();
                libc::close(fd);
                if written == 1 { Ok(()) } else { Err(error) }
            });
        }
        Ok(())
    }

    pub fn oom_kills(&self) -> u64 {
        read_flat(&self.dir.join("memory.events"))
            .get("oom_kill")
            .copied()
            .unwrap_or(0)
    }

    /// A process that died under a limit leaves it set.
    pub fn lift_limit(&self) -> Result<(), String> {
        write_file(&self.dir.join("memory.max"), "max")
    }

    pub fn is_limited(&self) -> bool {
        fs::read_to_string(self.dir.join("memory.max")).is_ok_and(|text| text.trim() != "max")
    }

    /// Whether a limit makes the child thrash badly enough to put others at
    /// risk: `systemd-oomd` watches the ancestor, and a sibling can lift that
    /// one above this cgroup's own. Pressure falls slowly, so without a limit in
    /// force it says nothing about now.
    pub fn is_thrashing(&self) -> bool {
        let own = full_avg10(&self.dir.join("memory.pressure"));
        let service = self
            .user_service
            .as_ref()
            .map_or(0.0, |dir| full_avg10(&dir.join("memory.pressure")));
        self.is_limited() && own >= OWN_PRESSURE_FLOOR && own.max(service) >= THRASH_PRESSURE
    }

    /// One line of the timeline the parent keeps of a child.
    pub fn sample(&self) -> String {
        let usage = usage_of(&self.dir);
        let events = read_flat(&self.dir.join("memory.events"));
        let event = |name: &str| events.get(name).copied().unwrap_or(0);
        format!(
            "unix_ms={} current={} anon={} file={} max_events={} oom={} oom_kill={} full_avg10={:.2}",
            chrono::Utc::now().timestamp_millis(),
            usage.current,
            usage.anon,
            usage.file,
            event("max"),
            event("oom"),
            event("oom_kill"),
            full_avg10(&self.dir.join("memory.pressure"))
        )
    }

    /// Everything the kernel says about the memory of this cgroup, for a hang.
    pub fn dump(&self) -> String {
        let mut text = String::new();
        for name in [
            "memory.current",
            "memory.max",
            "memory.peak",
            "memory.swap.max",
            "memory.events",
            "memory.pressure",
            "memory.stat",
            "cgroup.procs",
        ] {
            let value = fs::read_to_string(self.dir.join(name)).unwrap_or_default();
            writeln!(text, "== {name}\n{value}").unwrap();
        }
        text
    }
}

impl Drop for SeedCgroup {
    fn drop(&mut self) {
        fs::write(self.dir.join("cgroup.kill"), "1").ok();
        let started = Instant::now();
        while fs::remove_dir(&self.dir).is_err() && started.elapsed() < REMOVE_TIMEOUT {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// A child's view of its own cgroup.
pub struct OwnCgroup {
    dir: PathBuf,
}

impl OwnCgroup {
    /// The cgroup the parent named, once it is certain to be this process's
    /// own: a limit must never land on any other.
    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(named) = std::env::var_os(ENV_CGROUP).map(PathBuf::from) else {
            return Ok(None);
        };
        let own = cgroup_of("self")?;
        if own != named {
            return Err(format!(
                "this process is in {}, not in {}",
                own.display(),
                named.display()
            ));
        }
        Ok(Some(Self { dir: named }))
    }

    /// Gives back what the process and the cgroup can do without: otherwise
    /// the real room of a step would be its budget plus the page cache plus
    /// the heap glibc kept after the checks of the last step.
    pub fn shrink(&self, dataset: &Path) {
        // SAFETY: `malloc_trim` only returns free heap to the system.
        unsafe { libc::malloc_trim(0) };
        // Reclaim skips dirty pages, so they have to reach the disk first.
        if let Ok(directory) = File::open(dataset) {
            // SAFETY: `syncfs` takes a descriptor this function holds open.
            unsafe { libc::syncfs(directory.as_raw_fd()) };
        }
        // Asking for all of it fails with EAGAIN, having reclaimed what it can.
        let current = read_u64(&self.dir.join("memory.current"));
        fs::write(self.dir.join("memory.reclaim"), current.to_string()).ok();
    }

    pub fn usage(&self) -> Usage {
        usage_of(&self.dir)
    }

    /// How many times the cgroup hit its limit so far.
    pub fn limit_hits(&self) -> u64 {
        read_flat(&self.dir.join("memory.events"))
            .get("max")
            .copied()
            .unwrap_or(0)
    }

    /// Sets the limit and returns what the kernel made of it: it rounds down
    /// to a page.
    pub fn set_limit(&self, bytes: u64) -> Result<u64, String> {
        let path = self.dir.join("memory.max");
        write_file(&path, &bytes.to_string())?;
        Ok(read_u64(&path))
    }

    pub fn lift_limit(&self) -> Result<(), String> {
        write_file(&self.dir.join("memory.max"), "max")
    }
}

/// One line of `memory.log`: the fields of an event as `name=value`.
struct EventLine(String);

impl Visit for EventLine {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        write!(self.0, " {}={value:?}", field.name()).unwrap();
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        write!(self.0, " {}={value}", field.name()).unwrap();
    }
}

/// Lance's memory events, with what this process was doing. Each goes out in a
/// single write and stays in the page cache, so a SIGKILL cannot take it back:
/// the last line of a dead process says where it stood.
struct MemoryLog {
    file: Mutex<File>,
    started: Instant,
    generation: u64,
    state: Arc<IncidentState>,
}

impl<S: Subscriber> Layer<S> for MemoryLog {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        let mut line = EventLine(format!(
            "t={} generation={} phase={}",
            self.started.elapsed().as_millis(),
            self.generation,
            self.state.phase()
        ));
        event.record(&mut line);
        // A SIGKILL can tear a write at a page boundary; `end=1` tells a whole line.
        line.0.push_str(" end=1\n");
        self.file
            .lock()
            .unwrap()
            .write_all(line.0.as_bytes())
            .expect("write the memory log");
    }
}

pub fn log_memory_events(
    path: &Path,
    started: Instant,
    generation: u64,
    state: Arc<IncidentState>,
) -> Result<(), String> {
    let file = File::options()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("cannot open {path:?}: {error}"))?;
    let layer = MemoryLog {
        file: Mutex::new(file),
        started,
        generation,
        state,
    }
    .with_filter(filter_fn(|metadata| metadata.target() == TRACE_MEMORY));
    tracing::subscriber::set_global_default(Registry::default().with(layer))
        .map_err(|error| format!("cannot install the memory log: {error}"))
}
