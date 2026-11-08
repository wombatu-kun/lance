// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `memory_limits`: a limit on each step, of the kernel or of Lance's own, on the
//! plain local path, in a cgroup of its own for each seed.

use crate::engine::data::{LARGE, Scale};
use crate::engine::ops::{IndexKind, Limit, Model, Op, POOL_VARIABLE};
use crate::engine::steps::{
    self, HANG_UNDER_LIMIT, OOM_RECOVERY, OpMix, Other, StepCheck, StepPlanner, StepRules, TROUBLE,
};
use crate::engine::{Body, PlanInput, PlanRng, StoreChoice};
use crate::incidents::Incident;
use crate::knobs::{Class, Key, Knob, Knobs, STEPS};
use crate::memory_events;
use crate::scenario::{Runtime, Scenario, SelfTest, Settings};

pub const SCENARIO: &dyn Scenario = &MemoryLimits;

self_tests! {
    oom_recovery: OOM_RECOVERY,
    hang_under_limit: HANG_UNDER_LIMIT,
}

struct MemoryLimits;

/// In the order a focus picks from.
const OPS: [&str; 8] = [
    "append",
    "delete",
    "create_index",
    "optimize_indices",
    "merge_insert",
    "compact",
    "scan",
    "cleanup",
];

const INDICES: [IndexKind; 4] = [
    IndexKind::BTree,
    IndexKind::BTreeId,
    IndexKind::IvfFlat,
    IndexKind::Inverted,
];

/// The cleanups and optimizations are taken from the scans.
const MIX: OpMix = OpMix {
    deletes: 35..45,
    cleanups: 25..28,
    optimizations: 28..32,
    others: &[
        (0..6, Other::Compact),
        (6..10, Other::IdIndex),
        (10..20, Other::MergeInsert),
        (25..35, Other::Scan),
    ],
};

const RULES: StepRules = StepRules {
    dies: false,
    absorbed: "limit_absorbed",
    incidents_explain_errors: true,
    stale_errors_are_conflicts: false,
    keeps_session: false,
    checks_kept_versions: true,
};

impl Scenario for MemoryLimits {
    fn name(&self) -> &'static str {
        "memory_limits"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["memory"]
    }

    fn focusable_ops(&self) -> Vec<&'static str> {
        OPS.to_vec()
    }

    fn least_focus_steps(&self, focused: &[&str]) -> u64 {
        steps::least_focus_steps(focused)
    }

    fn knobs(&self) -> Vec<Knob> {
        let mut knobs = steps::knobs(0.85);
        knobs.push(BUDGET_SCALE.share(
            Class::Plan,
            1.0,
            0.25..=4.0,
            "a factor on the budgets of the cgroup's limits",
        ));
        knobs
    }

    fn settings(&self, _self_test: Option<&SelfTest>, knobs: &Knobs) -> Settings {
        Settings {
            // The steps of this scenario set the pool, and the switch would turn
            // it off.
            child_env: vec![(POOL_VARIABLE, None), ("LANCE_BYPASS_SPILLING", None)],
            cgroup: true,
            ..steps::settings(STEPS.get(knobs))
        }
    }

    fn self_tests(&self) -> &'static [&'static SelfTest] {
        SELF_TESTS
    }

    fn checks(&self) -> &'static [&'static dyn StepCheck] {
        &[memory_events::CHECK]
    }

    fn runtime(&self, _knobs: &Knobs) -> Runtime {
        steps::RUNTIME
    }

    fn store(&self, _rng: &mut PlanRng) -> StoreChoice {
        StoreChoice::Local
    }

    fn scale(&self) -> Scale {
        LARGE
    }

    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body> {
        Box::new(steps::plan(self, rng, input))
    }
}

impl StepPlanner for MemoryLimits {
    fn rules(&self) -> &'static StepRules {
        &RULES
    }

    fn op_mix(&self) -> &'static OpMix {
        &MIX
    }

    fn index_kinds(&self) -> &'static [IndexKind] {
        &INDICES
    }

    /// A merge insert honors the pool only with an index on its key.
    fn opening_op(&self, rng: &mut PlanRng, index: usize) -> Option<Op> {
        (index == 0 && rng.chance(0.5)).then_some(Op::CreateIndex(IndexKind::BTreeId))
    }

    /// A stale handle is covered by the scenarios of its own.
    fn always_resync(&self) -> bool {
        true
    }

    fn limit(&self, rng: &mut PlanRng, op: &Op, model: &Model, knobs: &Knobs) -> Option<Limit> {
        rng.chance(TROUBLE.get(knobs))
            .then(|| random_limit(rng, op, model, BUDGET_SCALE.get(knobs)))
    }

    fn incident(
        &self,
        _rng: &mut PlanRng,
        _store: StoreChoice,
        _scale: &Scale,
        _op: &Op,
        _knobs: &Knobs,
    ) -> Option<Incident> {
        None
    }
}

/// A factor on the budgets of the cgroup's limits, which keeps their ratios.
const BUDGET_SCALE: Key<f64> = Key::new("budget_scale");

/// The budgets a seed picks from for an operation: an eighth, a quarter, a half,
/// one and two times the heap a calibration run saw it take on the data of this
/// scenario (`LANCE_STRESS_CALIBRATE=1`). The table is fixed, so that a seed
/// decides its budgets whatever the machine does.
fn cgroup_budgets_kib(op: &Op) -> &'static [u64] {
    match op {
        Op::Compact => &[3072, 6144, 12288, 24576, 49152],
        Op::CreateIndex(IndexKind::IvfFlat) => &[1792, 3584, 7168, 14336, 28672],
        Op::OptimizeIndices => &[768, 1536, 3072, 6144, 12288],
        // Its heap comes back from the allocator, and the cgroup sees little
        // more than the manifests it reads: one cleanup in 60 reached even these.
        Op::Cleanup { .. } => &[16, 32, 64, 128, 256],
        Op::MergeInsert { .. } => &[800, 1600, 3200, 6400, 12800],
        // Most of what these hold is page cache, which the limit evicts, or a
        // few rows to sort.
        Op::Delete { .. } | Op::Scan | Op::CreateIndex(IndexKind::BTree | IndexKind::BTreeId) => {
            &[80, 160, 320, 640, 1280, 2560, 5120]
        }
        _ => &[640, 1280, 2560, 5120, 10240],
    }
}

fn pick(rng: &mut PlanRng, values: &[u64]) -> u64 {
    values[rng.range(0..=values.len() as u64 - 1) as usize]
}

/// The limit on a step. Half of the steps a limit of Lance's own applies to get
/// that one instead of the kernel's. Pool sizes follow what is sorted, not the
/// heap of the operation: a BTree sorts 12 bytes per row, around 0.5 MB here.
/// The budgets of the kernel's are `budget_scale` times those of the table.
fn random_limit(rng: &mut PlanRng, op: &Op, model: &Model, budget_scale: f64) -> Limit {
    let own = rng.chance(0.5);
    match op {
        Op::CreateIndex(IndexKind::BTree | IndexKind::BTreeId) if own => Limit::Pool {
            kib: pick(rng, &[32, 64, 128, 256, 512, 1024]),
        },
        // Without an index on the key, the join does not use the pool at all, and
        // the kernel's limit gets plenty of merges without one.
        Op::MergeInsert { .. }
            if model.indices.contains(&IndexKind::BTreeId) && (own || rng.chance(0.6)) =>
        {
            Limit::Pool {
                kib: pick(rng, &[512, 1024, 2048, 4096, 8192, 16384]),
            }
        }
        Op::CreateIndex(IndexKind::Inverted) if own => Limit::FtsMemory {
            mib: pick(rng, &[1, 2, 4]),
        },
        Op::Scan if own => Limit::IoBuffer {
            kib: pick(rng, &[4, 64, 1024, 16384]),
        },
        _ => Limit::Cgroup {
            budget_kib: (pick(rng, cgroup_budgets_kib(op)) as f64 * budget_scale).round() as u64,
        },
    }
}

#[test]
fn selection_draws_every_focusable_operation() {
    steps::check_draws(&MemoryLimits);
}

#[test]
fn selection_focus_takes_every_other_step() {
    // Its plans hold tens of thousands of rows.
    steps::check_focus(&MemoryLimits, 8);
}
