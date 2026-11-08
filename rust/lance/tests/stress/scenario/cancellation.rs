// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `cancellation`: the operations of `storage_faults`, dropped at a named point,
//! on a local or a `shared-memory://` dataset.

use crate::engine::data::{SMALL, Scale};
use crate::engine::ops::{IndexKind, Op};
use crate::engine::steps::{self, OpMix, StepCheck, StepPlanner, StepRules, TROUBLE, cancel_point};
use crate::engine::{Body, PlanInput, PlanRng, StoreChoice};
use crate::incidents::Incident;
use crate::knobs::{Knob, Knobs, STEPS};
use crate::scenario::{Runtime, Scenario, SelfTest, Settings};

pub const SCENARIO: &dyn Scenario = &Cancellation;

struct Cancellation;

/// In the order a focus picks from.
const OPS: [&str; 5] = [
    "append",
    "delete",
    "create_index",
    "optimize_indices",
    "cleanup",
];

const INDICES: [IndexKind; 3] = [IndexKind::BTree, IndexKind::IvfFlat, IndexKind::Inverted];

const MIX: OpMix = OpMix {
    deletes: 35..55,
    cleanups: 20..26,
    optimizations: 26..32,
    others: &[],
};

const RULES: StepRules = StepRules {
    dies: false,
    absorbed: "cancel_missed",
    incidents_explain_errors: true,
    stale_errors_are_conflicts: false,
    keeps_session: true,
    checks_kept_versions: true,
};

impl Scenario for Cancellation {
    fn name(&self) -> &'static str {
        "cancellation"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["cancel"]
    }

    fn focusable_ops(&self) -> Vec<&'static str> {
        OPS.to_vec()
    }

    fn least_focus_steps(&self, focused: &[&str]) -> u64 {
        steps::least_focus_steps(focused)
    }

    fn knobs(&self) -> Vec<Knob> {
        steps::knobs(0.85)
    }

    fn settings(&self, _self_test: Option<&SelfTest>, knobs: &Knobs) -> Settings {
        steps::settings(STEPS.get(knobs))
    }

    fn self_tests(&self) -> &'static [&'static SelfTest] {
        &[]
    }

    fn checks(&self) -> &'static [&'static dyn StepCheck] {
        &[]
    }

    fn runtime(&self, _knobs: &Knobs) -> Runtime {
        steps::RUNTIME
    }

    fn store(&self, rng: &mut PlanRng) -> StoreChoice {
        match rng.chance(0.5) {
            true => StoreChoice::Local,
            false => StoreChoice::SharedMemory,
        }
    }

    fn scale(&self) -> Scale {
        SMALL
    }

    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body> {
        Box::new(steps::plan(self, rng, input))
    }
}

impl StepPlanner for Cancellation {
    fn rules(&self) -> &'static StepRules {
        &RULES
    }

    fn op_mix(&self) -> &'static OpMix {
        &MIX
    }

    fn index_kinds(&self) -> &'static [IndexKind] {
        &INDICES
    }

    fn incident(
        &self,
        rng: &mut PlanRng,
        store: StoreChoice,
        scale: &Scale,
        op: &Op,
        knobs: &Knobs,
    ) -> Option<Incident> {
        cancel_point(rng, store, scale, op, TROUBLE.get(knobs))
    }
}

#[test]
fn selection_draws_every_focusable_operation() {
    steps::check_draws(&Cancellation);
}

#[test]
fn selection_focus_takes_every_other_step() {
    steps::check_focus(&Cancellation, 16);
}
