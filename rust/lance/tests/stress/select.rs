// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_STRESS_ONLY`: which scenarios a run takes, by the trouble they make or
//! by name, and which operations meet that trouble.

use std::collections::BTreeSet;
use std::env::VarError;

use crate::harness::ScenarioKind;
use crate::scenario::focusable_ops;

pub const ENV_ONLY: &str = "LANCE_STRESS_ONLY";

const OP_PREFIX: &str = "op:";

/// Kinds of trouble and the scenarios that make them. A scenario may make more
/// than one.
const KINDS: [(&str, &[ScenarioKind]); 6] = [
    ("faults", &[ScenarioKind::StorageFaults]),
    ("cancel", &[ScenarioKind::Cancellation]),
    ("kill", &[ScenarioKind::ProcessKill]),
    (
        "memory",
        &[ScenarioKind::MemoryLimits, ScenarioKind::HeapGrowth],
    ),
    ("concurrency", &[ScenarioKind::Concurrency]),
    ("latency", &[ScenarioKind::SlowStorage]),
];

/// What a run takes. The default takes every scenario as it is.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    /// The scenarios that the kinds and names of the list pick; none picks all.
    picked: BTreeSet<ScenarioKind>,
    /// The operations that the trouble falls on; none leaves it everywhere.
    focus: BTreeSet<&'static str>,
}

impl Selection {
    /// The selection of this process. A value that names something unknown, or
    /// that leaves no scenario to run, fails right here: a typo must not pass for
    /// a run that checked nothing.
    pub fn from_env() -> Self {
        match std::env::var(ENV_ONLY) {
            Ok(value) => {
                Self::parse(&value).unwrap_or_else(|error| panic!("{ENV_ONLY}={value:?} {error}"))
            }
            Err(VarError::NotPresent) => Self::default(),
            Err(error) => panic!("{ENV_ONLY}: {error}"),
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        let mut selection = Self::default();
        let mut named = BTreeSet::new();
        for name in value.split(',').map(str::trim) {
            if let Some(op) = name.strip_prefix(OP_PREFIX).map(str::trim) {
                let known = known_ops();
                let op = known.iter().find(|known| **known == op).ok_or_else(|| {
                    format!("names no known operation in {name:?}{}", known_names())
                })?;
                selection.focus.insert(op);
            } else if let Some((_, kinds)) = KINDS.iter().find(|(kind, _)| *kind == name) {
                selection.picked.extend(kinds.iter().copied());
            } else if let Some(kind) = ScenarioKind::from_name(name) {
                selection.picked.insert(kind);
                named.insert(kind);
            } else {
                return Err(format!(
                    "names neither a kind of trouble nor a scenario in {name:?}{}",
                    known_names()
                ));
            }
        }
        // A kind of trouble may hold scenarios without the operations; a scenario
        // asked for by name must have one.
        if let Some(kind) = named.iter().find(|kind| !selection.runs(**kind)) {
            let ops = focusable_ops(*kind);
            let runs = match ops.is_empty() {
                true => "a fixed cycle".to_string(),
                false => ops.join(", "),
            };
            return Err(format!(
                "asks for {}, which has none of its operations: it runs {runs}",
                kind.name()
            ));
        }
        if !ScenarioKind::ALL.iter().any(|kind| selection.runs(*kind)) {
            return Err(format!(
                "leaves no scenario to run: none of the scenarios it picks has an operation it names{}",
                known_names()
            ));
        }
        Ok(selection)
    }

    /// Whether a scenario runs: the list picks it, and it has an operation the
    /// list names, if the list names any.
    pub fn runs(&self, kind: ScenarioKind) -> bool {
        self.runs_self_test(kind)
            && (self.focus.is_empty()
                || focusable_ops(kind).iter().any(|op| self.focus.contains(op)))
    }

    /// Whether the self-tests of a scenario run. They inject a defect into a plan
    /// of their own, so only the scenarios a list picks count, not its operations.
    pub fn runs_self_test(&self, kind: ScenarioKind) -> bool {
        self.picked.is_empty() || self.picked.contains(&kind)
    }

    /// The operations a plan puts its trouble on; empty for all of them.
    pub fn focus(&self) -> &BTreeSet<&'static str> {
        &self.focus
    }
}

/// Every operation some scenario can put its trouble on.
fn known_ops() -> BTreeSet<&'static str> {
    ScenarioKind::ALL
        .iter()
        .flat_map(|kind| focusable_ops(*kind))
        .collect()
}

fn known_names() -> String {
    let kinds: Vec<&str> = KINDS.iter().map(|(kind, _)| *kind).collect();
    let scenarios: Vec<&str> = ScenarioKind::ALL.iter().map(|kind| kind.name()).collect();
    let ops: Vec<String> = known_ops()
        .into_iter()
        .map(|op| format!("{OP_PREFIX}{op}"))
        .collect();
    format!(
        "; a comma-separated list of kinds of trouble ({}), scenarios ({}) and operations ({})",
        kinds.join(", "),
        scenarios.join(", "),
        ops.join(", ")
    )
}

#[test]
fn selection_reads_kinds_scenarios_and_operations() {
    let selection = Selection::parse("memory, process_kill ,op: compact").unwrap();
    assert!(selection.runs(ScenarioKind::MemoryLimits));
    assert!(selection.runs(ScenarioKind::ProcessKill));
    // Its cycle is fixed, so there is no compaction to put trouble on.
    assert!(!selection.runs(ScenarioKind::HeapGrowth));
    assert!(selection.runs_self_test(ScenarioKind::HeapGrowth));
    assert!(!selection.runs(ScenarioKind::StorageFaults));
    assert!(!selection.runs_self_test(ScenarioKind::StorageFaults));
    assert_eq!(
        selection.focus().iter().copied().collect::<Vec<_>>(),
        ["compact"]
    );

    // Operations alone pick every scenario that has one of them.
    let selection = Selection::parse("op:compact").unwrap();
    let running: Vec<&str> = ScenarioKind::ALL
        .iter()
        .filter(|kind| selection.runs(**kind))
        .map(|kind| kind.name())
        .collect();
    assert_eq!(running, ["process_kill", "memory_limits", "concurrency"]);
    assert!(
        ScenarioKind::ALL
            .iter()
            .all(|kind| selection.runs_self_test(*kind))
    );

    // These two run in every scenario but the one with a fixed cycle.
    for value in [
        "op:cleanup",
        "op:optimize_indices",
        "storage_faults,op:cleanup",
    ] {
        let selection = Selection::parse(value).unwrap();
        let running: Vec<&str> = ScenarioKind::ALL
            .iter()
            .filter(|kind| selection.runs(**kind))
            .map(|kind| kind.name())
            .collect();
        let expected: &[&str] = match value {
            "storage_faults,op:cleanup" => &["storage_faults"],
            _ => &[
                "storage_faults",
                "cancellation",
                "process_kill",
                "memory_limits",
                "concurrency",
                "slow_storage",
            ],
        };
        assert_eq!(running, expected, "{value}");
    }

    let everything = Selection::default();
    assert!(ScenarioKind::ALL.iter().all(|kind| everything.runs(*kind)));
    assert!(everything.focus().is_empty());
}

#[test]
fn selection_refuses_what_it_does_not_know_and_what_picks_nothing() {
    for value in [
        "",
        "faults,",
        "fault",
        "op:",
        "op:scans",
        "op:reserve",
        "heap_growth,op:compact",
        "faults,op:scan",
        "storage_faults,process_kill,op:compact",
    ] {
        assert!(
            Selection::parse(value).is_err(),
            "{value:?} must be refused"
        );
    }
    let error = Selection::parse("fault").unwrap_err();
    for name in ["faults", "storage_faults", "op:compact", "op:cleanup"] {
        assert!(error.contains(name), "{name} is missing from: {error}");
    }
}
