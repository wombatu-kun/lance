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

    /// How its seeds run, for a plan of `steps` steps.
    fn settings(&self, self_test: Option<&SelfTest>, steps: u64) -> Settings;

    /// Its self-tests, as `self_tests!` lists them.
    fn self_tests(&self) -> &'static [&'static SelfTest];

    /// The checks it adds to the steps of the step engine.
    fn checks(&self) -> &'static [&'static dyn StepCheck];

    /// Files of a seed's directory that outlive it, beside `summary.txt`.
    fn artifacts(&self) -> &'static [&'static str] {
        &[]
    }

    fn runtime(&self) -> Runtime;

    /// The store of a seed: the first draw of its plan.
    fn store(&self, rng: &mut PlanRng) -> StoreChoice;

    fn scale(&self) -> Scale;

    /// The rest of a seed's plan, after its header.
    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body>;
}

/// How the parent runs the seeds of a scenario.
pub struct Settings {
    /// How many seeds a run takes, unless `LANCE_STRESS_SEEDS` says otherwise.
    pub seeds: u64,
    /// The size of a plan, as its header and the child take it.
    pub steps: u64,
    pub stall_secs: u64,
    pub deadline_secs: u64,
    /// The least deadline the scenario can run under, and why.
    pub least_deadline: Option<(u64, String)>,
    /// The variables a child gets, or must not inherit when `None`.
    pub child_env: Vec<(&'static str, Option<String>)>,
    /// The variables a replay pins besides the fingerprint, as the plan holds them.
    pub replay_env: Vec<(&'static str, String)>,
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
