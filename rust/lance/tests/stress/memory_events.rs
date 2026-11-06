// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance's memory events: a log of them in each process of a seed, and a check
//! of each index build against what happened, in the scenarios that add it.
//! Taking it out of the harness takes this module, its line in `main.rs`, and
//! what names it in the scenarios: their `use` of it, its entries in `checks()`
//! and the line of its self-test.

use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use lance_core::utils::tracing::TRACE_MEMORY;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

use crate::engine::ops::{IndexKind, Op, Outcome};
use crate::engine::run::Run;
use crate::engine::steps::{Step, StepCheck};
use crate::engine::{Failure, fail};
use crate::harness::{CALLS_LOG, STEPS_LOG, field};
use crate::incidents::IncidentState;
use crate::scenario::{SelfTest, nothing_more};

const MEMORY_LOG: &str = "memory.log";
/// Read once, when a process first needs it, so a child gets it from the start.
const OBSERVE_VARIABLE: &str = "LANCE_MEMORY_OBSERVE";

/// An index build reports no memory events, though it must.
pub const SELF_TEST: SelfTest = SelfTest {
    name: "memory_events",
    expected: &["memory_events"],
    reason: None,
    verify: nothing_more,
};

pub const CHECK: &dyn StepCheck = &MemoryEvents;

struct MemoryEvents;

impl StepCheck for MemoryEvents {
    fn child_env(&self, self_test: Option<&SelfTest>) -> Vec<(&'static str, Option<String>)> {
        // Its own self-test turns the events off, whatever the environment says.
        let observe = (self_test != Some(&SELF_TEST)).then(|| "1".to_string());
        vec![(OBSERVE_VARIABLE, observe)]
    }

    fn prepare(&self, steps: &mut [Step], self_test: Option<&SelfTest>) {
        // The ids of later steps stay free, so this cannot clash with them.
        if self_test == Some(&SELF_TEST) {
            steps[0].op = Op::CreateIndex(IndexKind::BTree);
        }
    }

    fn logs(&self) -> &'static [&'static str] {
        &[MEMORY_LOG]
    }

    fn start(
        &self,
        dir: &Path,
        started: Instant,
        generation: u64,
        state: Arc<IncidentState>,
    ) -> Result<(), String> {
        log_memory_events(&dir.join(MEMORY_LOG), started, generation, state)
    }

    fn after_step(
        &self,
        run: &Run,
        n: usize,
        step: &Step,
        outcome: &Outcome,
        committed: bool,
    ) -> Result<(), Failure> {
        check_memory_events(run, n, step, outcome, committed)
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

fn log_memory_events(
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

/// What Lance logged about the memory of an index build must agree with
/// what happened: the build, its stages, the limit it ran under, and that
/// the end of a build is only ever logged after its commit.
fn check_memory_events(
    run: &Run,
    n: usize,
    step: &Step,
    outcome: &Outcome,
    committed: bool,
) -> Result<(), Failure> {
    if !matches!(step.op, Op::CreateIndex(_)) {
        return Ok(());
    }
    let wrong = |detail: String| Err(fail("memory_events", n, detail));
    let in_this_step = |line: &&str| field(line, "phase") == Some(n.to_string().as_str());
    let log = fs::read_to_string(run.seed_dir().join(MEMORY_LOG)).unwrap_or_default();
    // The last line of a dead process can be torn.
    let events: Vec<&str> = log
        .lines()
        .filter(|line| line.ends_with(" end=1"))
        .filter(in_this_step)
        .collect();
    if let Some(limit) = limit_of_step(run.seed_dir(), n) {
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
    let calls = fs::read_to_string(run.seed_dir().join(CALLS_LOG)).unwrap_or_default();
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
fn limit_of_step(seed_dir: &Path, n: usize) -> Option<u64> {
    let journal = fs::read_to_string(seed_dir.join(STEPS_LOG)).ok()?;
    journal
        .lines()
        .filter(|line| field(line, "step") == Some(n.to_string().as_str()))
        .find(|line| line.split(' ').any(|token| token == "armed"))
        .and_then(|line| field(line, "limit")?.parse().ok())
}
