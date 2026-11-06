// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `storage_faults`: object store errors in the calls of an operation, before a
//! request or after it, on a `shared-memory://` dataset.

use crate::engine::data::{SMALL, Scale};
use crate::engine::ops::{IndexKind, Op};
use crate::engine::run::ORPHAN;
use crate::engine::steps::{
    self, COLUMN_MISMATCH, HANG, MODEL_MISMATCH, OpMix, PANIC, StepCheck, StepPlanner, StepRules,
    TASK_LEAK, TMP_LEAK, storage_calls,
};
use crate::engine::{Body, PlanInput, PlanRng, StoreChoice};
use crate::incidents::{Area, Incident, Method};
use crate::scenario::{Runtime, Scenario, SelfTest, Settings};

pub const SCENARIO: &dyn Scenario = &StorageFaults;

self_tests! {
    hang: HANG,
    panic: PANIC,
    task_leak: TASK_LEAK,
    tmp_leak: TMP_LEAK,
    model_mismatch: MODEL_MISMATCH,
    column_mismatch: COLUMN_MISMATCH,
    orphan: ORPHAN,
}

struct StorageFaults;

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
    absorbed: "fault_absorbed",
    incidents_explain_errors: true,
    stale_errors_are_conflicts: false,
    keeps_session: true,
    checks_kept_versions: true,
};

impl Scenario for StorageFaults {
    fn name(&self) -> &'static str {
        "storage_faults"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["faults"]
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
        &[]
    }

    fn runtime(&self) -> Runtime {
        steps::RUNTIME
    }

    fn store(&self, _rng: &mut PlanRng) -> StoreChoice {
        StoreChoice::SharedMemory
    }

    fn scale(&self) -> Scale {
        SMALL
    }

    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body> {
        Box::new(steps::plan(self, rng, input))
    }
}

impl StepPlanner for StorageFaults {
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
        _store: StoreChoice,
        scale: &Scale,
        op: &Op,
    ) -> Option<Incident> {
        if !rng.chance(0.75) {
            return None;
        }
        let targets = storage_calls(op, scale);
        let (method, area, max_nth) = targets[rng.range(0..=targets.len() as u64 - 1) as usize];
        let nth = rng.range(1..=max_nth);
        let after_call = method == Method::Put && rng.chance(0.5);
        let sticky = rng.chance(0.3);
        // Only a failed manifest write goes on to verify its outcome.
        let blind = method == Method::Put && area == Area::Manifests && rng.chance(0.5);
        Some(Incident::Fail {
            method,
            area,
            nth,
            after_call,
            sticky,
            blind,
        })
    }
}

#[test]
fn selection_draws_every_focusable_operation() {
    steps::check_draws(&StorageFaults);
}

#[test]
fn selection_focus_takes_every_other_step() {
    steps::check_focus(&StorageFaults, 16);
}
