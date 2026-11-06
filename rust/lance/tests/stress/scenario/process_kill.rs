// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `process_kill`: the points of `cancellation`, where the process sends itself a
//! SIGKILL instead, on a dataset on disk; a later process goes on with the seed.

use crate::engine::data::{SMALL, Scale};
use crate::engine::ops::{IndexKind, Op};
use crate::engine::steps::{
    self, FOREIGN_KILL, KILL_THEN_MISSING_FILE, OpMix, Other, StepCheck, StepPlanner, StepRules,
    cancel_point,
};
use crate::engine::{Body, PlanInput, PlanRng, StoreChoice};
use crate::incidents::Incident;
use crate::memory_events;
use crate::scenario::{Runtime, Scenario, SelfTest, Settings};

pub const SCENARIO: &dyn Scenario = &ProcessKill;

self_tests! {
    foreign_kill: FOREIGN_KILL,
    kill_then_missing_file: KILL_THEN_MISSING_FILE,
    memory_events: memory_events::SELF_TEST,
}

struct ProcessKill;

/// In the order a focus picks from.
const OPS: [&str; 7] = [
    "append",
    "delete",
    "create_index",
    "optimize_indices",
    "merge_insert",
    "compact",
    "cleanup",
];

const INDICES: [IndexKind; 4] = [
    IndexKind::BTree,
    IndexKind::BTreeId,
    IndexKind::IvfFlat,
    IndexKind::Inverted,
];

const MIX: OpMix = OpMix {
    deletes: 35..55,
    cleanups: 20..25,
    optimizations: 25..30,
    others: &[
        (0..6, Other::Compact),
        (6..10, Other::IdIndex),
        (10..20, Other::MergeInsert),
    ],
};

const RULES: StepRules = StepRules {
    dies: true,
    absorbed: "kill_missed",
    incidents_explain_errors: false,
    stale_errors_are_conflicts: false,
    keeps_session: false,
    checks_kept_versions: true,
};

impl Scenario for ProcessKill {
    fn name(&self) -> &'static str {
        "process_kill"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["kill"]
    }

    fn focusable_ops(&self) -> Vec<&'static str> {
        OPS.to_vec()
    }

    fn least_focus_steps(&self, focused: &[&str]) -> u64 {
        steps::least_focus_steps(focused)
    }

    fn settings(&self, _self_test: Option<&SelfTest>, steps: u64) -> Settings {
        steps::settings(steps)
    }

    fn self_tests(&self) -> &'static [&'static SelfTest] {
        SELF_TESTS
    }

    fn checks(&self) -> &'static [&'static dyn StepCheck] {
        &[memory_events::CHECK]
    }

    fn runtime(&self) -> Runtime {
        steps::RUNTIME
    }

    fn store(&self, rng: &mut PlanRng) -> StoreChoice {
        match rng.chance(0.5) {
            true => StoreChoice::Local,
            false => StoreChoice::FileObjectStore,
        }
    }

    fn scale(&self) -> Scale {
        SMALL
    }

    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body> {
        Box::new(steps::plan(self, rng, input))
    }
}

impl StepPlanner for ProcessKill {
    fn rules(&self) -> &'static StepRules {
        &RULES
    }

    fn op_mix(&self) -> &'static OpMix {
        &MIX
    }

    fn index_kinds(&self) -> &'static [IndexKind] {
        &INDICES
    }

    /// A stale handle is covered by the scenarios of its own.
    fn always_resync(&self) -> bool {
        true
    }

    fn in_flight_us(&self, rng: &mut PlanRng) -> u64 {
        rng.range(0..=300)
    }

    fn incident(
        &self,
        rng: &mut PlanRng,
        store: StoreChoice,
        scale: &Scale,
        op: &Op,
    ) -> Option<Incident> {
        cancel_point(rng, store, scale, op)
    }
}

#[test]
fn selection_draws_every_focusable_operation() {
    steps::check_draws(&ProcessKill);
}

#[test]
fn selection_focus_takes_every_other_step() {
    steps::check_focus(&ProcessKill, 16);
}
