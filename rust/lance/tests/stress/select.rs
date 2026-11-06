// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_STRESS_ONLY`: which scenarios a run takes, by the trouble they make or
//! by name, and which operations meet that trouble.

use std::collections::BTreeSet;
use std::env::VarError;

use crate::scenario::{REGISTRY, Scenario};

pub const ENV_ONLY: &str = "LANCE_STRESS_ONLY";

const OP_PREFIX: &str = "op:";

/// Kinds of trouble and the scenarios that make them, by their places in the
/// registry, in the order the registry first names each kind. A scenario may make
/// more than one.
fn kinds() -> Vec<(&'static str, Vec<usize>)> {
    let mut kinds: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for (place, scenario) in REGISTRY.iter().enumerate() {
        for kind in scenario.trouble_kinds() {
            match kinds.iter_mut().find(|(known, _)| known == kind) {
                Some((_, places)) => places.push(place),
                None => kinds.push((kind, vec![place])),
            }
        }
    }
    kinds
}

fn place_named(name: &str) -> Option<usize> {
    REGISTRY.iter().position(|scenario| scenario.name() == name)
}

/// What a run takes. The default takes every scenario as it is.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    /// The scenarios that the kinds and names of the list pick, by their places
    /// in the registry, which orders them; none picks all.
    picked: BTreeSet<usize>,
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

    fn parse(value: &str) -> Result<Self, String> {
        let mut selection = Self::default();
        let mut named = BTreeSet::new();
        for name in value.split(',').map(str::trim) {
            if let Some(op) = name.strip_prefix(OP_PREFIX).map(str::trim) {
                let known = known_ops();
                let op = known.iter().find(|known| **known == op).ok_or_else(|| {
                    format!("names no known operation in {name:?}{}", known_names())
                })?;
                selection.focus.insert(op);
            } else if let Some((_, places)) = kinds().into_iter().find(|(kind, _)| *kind == name) {
                selection.picked.extend(places);
                // `soak` and `concurrency` name a scenario too, which must not drop out.
                named.extend(place_named(name));
            } else if let Some(place) = place_named(name) {
                selection.picked.insert(place);
                named.insert(place);
            } else {
                return Err(format!(
                    "names neither a kind of trouble nor a scenario in {name:?}{}",
                    known_names()
                ));
            }
        }
        // A kind of trouble may hold scenarios without the operations; a scenario
        // asked for by name must have one.
        if let Some(scenario) = named
            .iter()
            .map(|place| REGISTRY[*place])
            .find(|scenario| !selection.runs(*scenario))
        {
            let ops = scenario.focusable_ops();
            let runs = match ops.is_empty() {
                true => "a cycle of its own".to_string(),
                false => ops.join(", "),
            };
            return Err(format!(
                "asks for {}, which has none of its operations: it runs {runs}",
                scenario.name()
            ));
        }
        if !REGISTRY.iter().any(|scenario| selection.runs(*scenario)) {
            return Err(format!(
                "leaves no scenario to run: none of the scenarios it picks has an operation it names{}",
                known_names()
            ));
        }
        Ok(selection)
    }

    /// Whether a scenario runs: the list picks it, and it has an operation the
    /// list names, if the list names any.
    pub fn runs(&self, scenario: &dyn Scenario) -> bool {
        self.runs_self_test(scenario)
            && (self.focus.is_empty()
                || scenario
                    .focusable_ops()
                    .iter()
                    .any(|op| self.focus.contains(op)))
    }

    /// Whether the self-tests of a scenario run. They inject a defect into a plan
    /// of their own, so only the scenarios a list picks count, not its operations.
    pub fn runs_self_test(&self, scenario: &dyn Scenario) -> bool {
        self.picked.is_empty()
            || place_named(scenario.name()).is_some_and(|place| self.picked.contains(&place))
    }

    /// The operations a plan puts its trouble on; empty for all of them.
    pub fn focus(&self) -> &BTreeSet<&'static str> {
        &self.focus
    }
}

/// Every operation some scenario can put its trouble on.
fn known_ops() -> BTreeSet<&'static str> {
    REGISTRY
        .iter()
        .flat_map(|scenario| scenario.focusable_ops())
        .collect()
}

fn known_names() -> String {
    let kinds: Vec<&str> = kinds().into_iter().map(|(kind, _)| kind).collect();
    let scenarios: Vec<&str> = REGISTRY.iter().map(|scenario| scenario.name()).collect();
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

#[cfg(test)]
fn named(name: &str) -> &'static dyn Scenario {
    crate::scenario::by_name(name).expect("a scenario of the registry")
}

#[test]
fn selection_reads_kinds_scenarios_and_operations() {
    // A kind of trouble picks every scenario that makes it, in the order of the
    // registry, and is listed once.
    let kinds = kinds();
    for kind in REGISTRY
        .iter()
        .flat_map(|scenario| scenario.trouble_kinds())
    {
        let places: Vec<usize> = (0..REGISTRY.len())
            .filter(|place| REGISTRY[*place].trouble_kinds().contains(kind))
            .collect();
        assert!(kinds.contains(&(*kind, places)), "{kind}: {kinds:?}");
    }
    let listed: BTreeSet<&str> = kinds.iter().map(|(kind, _)| *kind).collect();
    assert_eq!(listed.len(), kinds.len(), "{kinds:?}");

    let selection = Selection::parse("memory, process_kill ,op: compact").unwrap();
    assert!(selection.runs(named("memory_limits")));
    assert!(selection.runs(named("process_kill")));
    // Their cycles are their own, so there is no compaction to put trouble on.
    assert!(!selection.runs(named("heap_growth")));
    assert!(selection.runs_self_test(named("heap_growth")));
    assert!(!selection.runs(named("soak")));
    assert!(selection.runs_self_test(named("soak")));
    assert!(!selection.runs(named("storage_faults")));
    assert!(!selection.runs_self_test(named("storage_faults")));
    assert_eq!(
        selection.focus().iter().copied().collect::<Vec<_>>(),
        ["compact"]
    );

    // Operations alone pick every scenario that has one of them.
    let selection = Selection::parse("op:compact").unwrap();
    for scenario in REGISTRY {
        let has = scenario.focusable_ops().contains(&"compact");
        assert_eq!(selection.runs(*scenario), has, "{}", scenario.name());
        assert!(selection.runs_self_test(*scenario), "{}", scenario.name());
    }
    for name in ["process_kill", "memory_limits", "concurrency"] {
        assert!(selection.runs(named(name)), "{name}");
    }

    // These two run in every scenario but the ones with a cycle of their own.
    for value in ["op:cleanup", "op:optimize_indices"] {
        let selection = Selection::parse(value).unwrap();
        for name in [
            "storage_faults",
            "cancellation",
            "process_kill",
            "memory_limits",
            "concurrency",
            "slow_storage",
        ] {
            assert!(selection.runs(named(name)), "{value} {name}");
        }
        for name in ["heap_growth", "soak"] {
            assert!(!selection.runs(named(name)), "{value} {name}");
        }
    }
    // A name picks that scenario alone.
    let selection = Selection::parse("storage_faults,op:cleanup").unwrap();
    let running: Vec<&str> = REGISTRY
        .iter()
        .filter(|scenario| selection.runs(**scenario))
        .map(|scenario| scenario.name())
        .collect();
    assert_eq!(running, ["storage_faults"]);

    let everything = Selection::default();
    assert!(REGISTRY.iter().all(|scenario| everything.runs(*scenario)));
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
        "soak,op:compact",
        "soak,process_kill,op:compact",
        "faults,op:scan",
        "storage_faults,process_kill,op:compact",
    ] {
        assert!(
            Selection::parse(value).is_err(),
            "{value:?} must be refused"
        );
    }
    // A kind that is also the name of a scenario counts as its name.
    let error = Selection::parse("soak,process_kill,op:compact").unwrap_err();
    assert!(
        error.contains("asks for soak") && error.contains("a cycle of its own"),
        "{error}"
    );
    let error = Selection::parse("fault").unwrap_err();
    for name in [
        "faults",
        "storage_faults",
        "soak",
        "op:compact",
        "op:cleanup",
    ] {
        assert!(error.contains(name), "{name} is missing from: {error}");
    }
}
