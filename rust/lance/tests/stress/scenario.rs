// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The scenarios: what each one decides, and the registry the tests of a run
//! come from.
//!
//! A scenario takes two lines here, its module below and its name in
//! `scenarios!`: rustfmt does not look into a module that a macro declares. The
//! registry makes a test of each, `scenario::<name>`, which runs its seeds, and
//! `self_tests!` in its module a test of each of its self-tests,
//! `scenario::<name>::self_test::<test>`.

use std::collections::BTreeSet;
use std::time::Duration;

use crate::engine::data::Scale;
use crate::engine::steps::StepCheck;
use crate::engine::{Body, PlanInput, PlanRng, StoreChoice};
use crate::harness::SeedReport;
use crate::knobs::{Knob, Knobs};

/// Declares the self-tests of a scenario: `SELF_TESTS` for its `Scenario`, and a
/// test of each, named as the self-test is.
macro_rules! self_tests {
    ($($test:ident: $spec:path),* $(,)?) => {
        const SELF_TESTS: &[&$crate::scenario::SelfTest] = &[$(&$spec),*];

        mod self_test {
            $(
                #[test]
                fn $test() {
                    // By name among the scenario's own, as the child finds it.
                    let test = super::SCENARIO
                        .self_tests()
                        .iter()
                        .copied()
                        .find(|test| test.name == stringify!($test))
                        .expect("the scenario lists a self-test named as its test");
                    $crate::harness::check_self_test(super::SCENARIO, test);
                }
            )*
        }
    };
}

/// Lists the scenarios: `REGISTRY`, in the order `LANCE_STRESS_ONLY` and its
/// messages take them, and a test of each that runs its seeds.
macro_rules! scenarios {
    ($($name:ident),* $(,)?) => {
        pub const REGISTRY: &[&dyn Scenario] = &[$($name::SCENARIO),*];

        /// The module of each scenario, in the order of the registry.
        const MODULES: &[&str] = &[$(stringify!($name)),*];

        $(
            #[test]
            fn $name() {
                $crate::harness::run($name::SCENARIO);
            }
        )*
    };
}

pub mod cancellation;
pub mod concurrency;
pub mod heap_growth;
pub mod memory_limits;
pub mod process_kill;
pub mod slow_storage;
pub mod soak;
pub mod storage_faults;

scenarios! {
    storage_faults,
    cancellation,
    process_kill,
    memory_limits,
    heap_growth,
    concurrency,
    slow_storage,
    soak,
}

/// What a scenario decides, for the parent that runs its seeds and for the child
/// that runs one.
pub trait Scenario: Sync {
    /// The name of its module, its test and its seeds' directories.
    fn name(&self) -> &'static str;

    /// The kinds of trouble of `LANCE_STRESS_ONLY` that pick it.
    fn trouble_kinds(&self) -> &'static [&'static str];

    /// The operations it can put its trouble on, as `LANCE_STRESS_ONLY` names
    /// them: what its plans draw, and in `concurrency` what its forced pairs run.
    fn focusable_ops(&self) -> Vec<&'static str>;

    /// The fewest steps a plan needs for these operations of a focus to have any:
    /// a focus takes every other step, and in `concurrency` the forced pairs from
    /// the third round on.
    fn least_focus_steps(&self, focused: &[&str]) -> u64;

    /// Its knobs, each with the default the code held before it was one.
    fn knobs(&self) -> Vec<Knob>;

    /// How its seeds run with these knobs.
    fn settings(&self, self_test: Option<&SelfTest>, knobs: &Knobs) -> Settings;

    /// Its self-tests, as `self_tests!` lists them.
    fn self_tests(&self) -> &'static [&'static SelfTest];

    /// The checks it adds to the steps of the step engine.
    fn checks(&self) -> &'static [&'static dyn StepCheck];

    /// Files of a seed's directory that outlive it, beside `summary.txt`.
    fn artifacts(&self) -> &'static [&'static str] {
        &[]
    }

    fn runtime(&self, knobs: &Knobs) -> Runtime;

    /// The store of a seed: the first draw of its plan.
    fn store(&self, rng: &mut PlanRng) -> StoreChoice;

    fn scale(&self) -> Scale;

    /// The rest of a seed's plan, after its header.
    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body>;
}

/// How the parent runs the seeds of a scenario.
pub struct Settings {
    /// The size of a plan, as its header and the child take it.
    pub steps: u64,
    /// The stall and the deadline the plans of these knobs need, unless the knobs
    /// set their own.
    pub stall: Duration,
    pub deadline: Duration,
    /// The least deadline the scenario can run under, and why.
    pub least_deadline: Option<(Duration, String)>,
    /// The variables a child gets, or must not inherit when `None`.
    pub child_env: Vec<(&'static str, Option<String>)>,
    /// Every seed runs in a cgroup of its own.
    pub cgroup: bool,
}

/// The runtime a child runs its seed on.
pub struct Runtime {
    pub worker_threads: usize,
    /// How long an idle thread of the blocking pool stays; tokio's own if none.
    pub blocking_keep_alive: Option<Duration>,
    /// A failed seed leaves its tasks running as the process exits.
    pub leaves_tasks_on_failure: bool,
}

/// A defect a child injects on purpose, to show that the parent reports it. It
/// runs a plan of 3 steps, without incidents or a focus.
#[derive(Debug)]
pub struct SelfTest {
    /// The name of its test, of `LANCE_STRESS_SELF_TEST` and of its directory,
    /// unique in the registry.
    pub name: &'static str,
    /// The failure classes the parent may report for this defect; none means
    /// the seed must pass.
    pub expected: &'static [&'static str],
    /// What the detail of the failure says, where several checks share its class.
    pub reason: Option<&'static str>,
    /// What else the artifacts of the seed must show.
    pub verify: fn(&SeedReport),
}

impl PartialEq for SelfTest {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// The `verify` of a self-test whose class says it all.
pub fn nothing_more(_: &SeedReport) {}

pub fn by_name(name: &str) -> Option<&'static dyn Scenario> {
    REGISTRY
        .iter()
        .copied()
        .find(|scenario| scenario.name() == name)
}

/// The names a section of a guide puts first in the rows of its table, in backticks,
/// each cut where its first `<` starts the values it carries.
#[cfg(test)]
fn table_names(doc: &str, heading: &str) -> BTreeSet<String> {
    let (_, section) = doc
        .split_once(&format!("\n## {heading}\n"))
        .unwrap_or_else(|| panic!("no section {heading:?}"));
    let section = section.split("\n## ").next().unwrap_or(section);
    section
        .lines()
        .filter_map(|line| line.strip_prefix("| "))
        .filter_map(|line| line.split(" | ").next())
        .flat_map(|cell| cell.split('`').skip(1).step_by(2))
        .map(|name| name.split('<').next().unwrap_or(name).to_string())
        .collect()
}

/// The arguments of a call whose text starts after its opening parenthesis, up to
/// the parenthesis that closes it.
#[cfg(test)]
fn arguments(text: &str) -> &str {
    let (mut depth, mut quoted) = (1, false);
    for (index, c) in text.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => {
                depth -= 1;
                if depth == 0 {
                    return &text[..index];
                }
            }
            _ => {}
        }
    }
    text
}

/// The guide names everything a run can report, and nothing it cannot: every
/// scenario and kind of trouble in `README.md`, every class in the tables of both
/// files, every observation in `AGENTS.md`. A class a call of `fail` spells out is
/// declared, even on a path no run takes.
#[test]
fn registry_docs_name_every_scenario_class_and_observation() {
    const README: &str = include_str!("README.md");
    const AGENTS: &str = include_str!("AGENTS.md");
    let set = |names: &[&str]| -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    };
    let scenarios: Vec<&str> = REGISTRY.iter().map(|scenario| scenario.name()).collect();
    assert_eq!(table_names(README, "Scenarios"), set(&scenarios));
    let kinds: Vec<&str> = REGISTRY
        .iter()
        .flat_map(|scenario| scenario.trouble_kinds().iter().copied())
        .collect();
    let line = README
        .lines()
        .find(|line| line.contains("takes kinds ("))
        .expect("the kinds of LANCE_STRESS_ONLY");
    let (_, listed) = line.split_once("takes kinds (").unwrap_or_default();
    let listed = listed.split(')').next().unwrap_or_default();
    let listed: Vec<&str> = listed.split('`').skip(1).step_by(2).collect();
    assert_eq!(set(&listed), set(&kinds));
    let classes = set(crate::engine::CLASSES);
    assert_eq!(table_names(README, "Failure classes"), classes, "README.md");
    assert_eq!(table_names(AGENTS, "Where to look"), classes, "AGENTS.md");
    assert_eq!(
        table_names(AGENTS, "Observations"),
        set(crate::engine::OBSERVATIONS),
        "AGENTS.md"
    );
    // What the code spells out is checked where it is written, not only when a run
    // meets it: a class in a call of `fail` or `report`, an observation in a call of
    // `observe`, as what a step absorbs, or in a journal line.
    let (mut classes_spelled, mut observations_spelled) = (BTreeSet::new(), BTreeSet::new());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stress");
    let mut dirs = vec![root];
    while let Some(dir) = dirs.pop() {
        for entry in
            std::fs::read_dir(&dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if !name.ends_with(".rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let file = path.display().to_string();
            // Spelled in parts, so that this test does not take its own patterns. A
            // class is the first argument; an observation any literal of its call.
            for (call, class) in [
                (["fail", "("].concat(), true),
                (["report", "("].concat(), true),
                (["observe", "("].concat(), false),
                (["observe_hidden_files", "("].concat(), false),
            ] {
                for (at, _) in source.match_indices(call.as_str()) {
                    // Only the engine's `fail` and the parent's `report`, which no
                    // method of another type shares a name with.
                    let before = source[..at].chars().next_back();
                    let method = before == Some('.') && class;
                    if before.is_some_and(|c| c.is_alphanumeric() || c == '_')
                        || method
                        || source[..at].ends_with("fn ")
                    {
                        continue;
                    }
                    let statement = arguments(&source[at + call.len()..]);
                    let literals: Vec<&str> = match class {
                        true => statement
                            .trim_start()
                            .strip_prefix('"')
                            .and_then(|literal| literal.split('"').next())
                            .into_iter()
                            .collect(),
                        false => statement.split('"').skip(1).step_by(2).collect(),
                    };
                    for literal in literals {
                        let value = literal.split(['{', ' ']).next().unwrap_or_default();
                        let spelled = match class {
                            true => &mut classes_spelled,
                            false => &mut observations_spelled,
                        };
                        if !value.is_empty() {
                            spelled.insert((value.to_string(), file.clone()));
                        }
                    }
                }
            }
            // What a step of the engine observes when it absorbed its trouble.
            let absorbed = ["absorbed", ": \""].concat();
            for (at, _) in source.match_indices(absorbed.as_str()) {
                let value = source[at + absorbed.len()..]
                    .split('"')
                    .next()
                    .unwrap_or_default();
                observations_spelled.insert((value.to_string(), file.clone()));
            }
            for (at, _) in source.match_indices("observation=") {
                let value = &source[at + "observation=".len()..];
                let value = value
                    .split(['{', ' ', '"', '`', '<', ',', ')', '\\'])
                    .next()
                    .unwrap_or_default();
                if !value.is_empty() {
                    observations_spelled.insert((value.to_string(), file.clone()));
                }
            }
        }
    }
    for (class, file) in &classes_spelled {
        assert!(
            crate::engine::CLASSES.contains(&class.as_str()),
            "{file}: the class {class:?} is not in CLASSES of engine.rs"
        );
    }
    for (observation, file) in &observations_spelled {
        assert!(
            crate::engine::declared_observation(observation),
            "{file}: the observation {observation:?} is not in OBSERVATIONS of engine.rs"
        );
    }
    // Nothing is declared that the code no longer spells out, but the verdicts the
    // parent gives a child it watched.
    let spelled: BTreeSet<&str> = classes_spelled
        .iter()
        .map(|(class, _)| class.as_str())
        .collect();
    let unspelled: BTreeSet<&str> = crate::engine::CLASSES
        .iter()
        .copied()
        .filter(|class| !spelled.contains(class))
        .collect();
    assert_eq!(unspelled, BTreeSet::from(["crash", "hang"]));
    for declared in crate::engine::OBSERVATIONS {
        assert!(
            observations_spelled
                .iter()
                .any(|(spelled, _)| declared == spelled || declared.starts_with(spelled.as_str())),
            "{declared:?} of OBSERVATIONS is spelled nowhere in the sources"
        );
    }
}

/// `scenario_template.rs` names every method of the traits a scenario implements,
/// those with a default too, which the compiler would not ask a copy for.
#[test]
fn registry_template_names_every_method_of_a_scenario() {
    const TEMPLATE: &str = include_str!("scenario_template.rs");
    let traits = [
        (include_str!("scenario.rs"), "pub trait Scenario"),
        (include_str!("engine/steps.rs"), "pub trait StepPlanner"),
        (include_str!("engine.rs"), "pub trait Body"),
    ];
    for (source, head) in traits {
        let (_, rest) = source
            .split_once(head)
            .unwrap_or_else(|| panic!("no {head}"));
        let body = rest.split("\n}\n").next().unwrap_or(rest);
        let methods: Vec<&str> = body
            .lines()
            .map(str::trim_start)
            .filter_map(|line| {
                line.strip_prefix("fn ")
                    .or_else(|| line.strip_prefix("async fn "))
            })
            .filter_map(|line| line.split(['(', '<']).next())
            .collect();
        assert!(methods.len() >= 5, "{head}: {methods:?}");
        let name = head.trim_start_matches("pub trait ");
        let (_, implemented) = TEMPLATE
            .split_once(&format!("impl {name} for "))
            .unwrap_or_else(|| panic!("scenario_template.rs implements no {name}"));
        let implemented = implemented.split("\n}\n").next().unwrap_or(implemented);
        for method in methods {
            assert!(
                implemented.contains(&format!("fn {method}("))
                    || implemented.contains(&format!("fn {method}<")),
                "scenario_template.rs has no {method} of {name}"
            );
        }
    }
}

#[test]
fn registry_names_scenarios_after_their_modules_and_self_tests_once() {
    let names: Vec<&str> = REGISTRY.iter().map(|scenario| scenario.name()).collect();
    assert_eq!(names, MODULES);
    // A file the registry leaves out would never run.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stress/scenario");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .map(|path| path.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    files.sort();
    let mut modules = MODULES.to_vec();
    modules.sort_unstable();
    assert_eq!(files, modules, "the files of {}", dir.display());
    // A self-test's directory and variable take its name alone.
    let tests: Vec<&str> = REGISTRY
        .iter()
        .flat_map(|scenario| scenario.self_tests())
        .map(|test| test.name)
        .collect();
    let unique: BTreeSet<&str> = tests.iter().copied().collect();
    assert_eq!(unique.len(), tests.len(), "{tests:?}");
}
