// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `slow_storage`: the operations of `storage_faults` on a slow store, where now
//! and then a call never answers, on a `shared-memory://` or a
//! `file-object-store://` dataset.

use std::collections::BTreeSet;
use std::ops::RangeInclusive;
use std::time::Duration;

use crate::engine::data::{SMALL, Scale};
use crate::engine::ops::{IndexKind, Op};
use crate::engine::steps::{
    self, HUNG_CALL, HUNG_SETTLE, LATE_CALL, OpMix, SETTLE, StepCheck, StepPlanner, StepRules,
    TROUBLE, WRITE_AFTER_RELEASE, pick_from, storage_calls,
};
use crate::engine::{Body, PlanInput, PlanRng, StoreChoice};
use crate::harness::QUIET_SECS;
use crate::incidents::{Area, DEFAULT_LATE_AFTER, Delay, HangPoint, Incident, LATE_AFTER, Method};
use crate::knobs::{Class, Key, Knob, Knobs, STEPS};
use crate::scenario::{Runtime, Scenario, SelfTest, Settings};

pub const SCENARIO: &dyn Scenario = &SlowStorage;

self_tests! {
    hung_call: HUNG_CALL,
    late_call: LATE_CALL,
    write_after_release: WRITE_AFTER_RELEASE,
}

struct SlowStorage;

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
    absorbed: "delay_absorbed",
    incidents_explain_errors: false,
    stale_errors_are_conflicts: true,
    keeps_session: true,
    checks_kept_versions: true,
};

/// How long `slow_storage` slows a call, `delay_scale` times: from a busy store to
/// a struggling one. Up to 5 times, since a slowed read of a data file that a step
/// left behind must end within the 20 s a step waits for quiet.
const DELAYS_MS: [u64; 6] = [10, 30, 100, 300, 1000, 3000];
const DELAY_SCALE: Key<f64> = Key::new("delay_scale");
const SLOWED_CALLS: Key<RangeInclusive<u64>> = Key::new("slowed_calls");
const HANG: Key<f64> = Key::new("hang");

/// The calls `slow_storage` slows: those of `storage_faults`, as `store` makes
/// them. A store that lists in order finds the latest version by listing; another
/// one through the hint and the manifests it names, and a commit rewrites the
/// hint there (`uses_version_hint`). A cleanup lists on both.
fn slow_calls(op: &Op, scale: &Scale, store: StoreChoice) -> Vec<(Method, Area, u64)> {
    let mut calls = storage_calls(op, scale);
    if store == StoreChoice::FileObjectStore && !matches!(op, Op::Cleanup { .. }) {
        let commits = calls
            .iter()
            .any(|&(method, area, _)| method == Method::Put && area == Area::Manifests);
        calls.retain(|&(method, area, _)| (method, area) != (Method::List, Area::Manifests));
        calls.push((Method::Get, Area::VersionHint, 1));
        calls.push((Method::Head, Area::Manifests, 2));
        if commits {
            calls.push((Method::Put, Area::VersionHint, 1));
        }
    }
    calls
}

impl Scenario for SlowStorage {
    fn name(&self) -> &'static str {
        "slow_storage"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["latency"]
    }

    fn focusable_ops(&self) -> Vec<&'static str> {
        OPS.to_vec()
    }

    fn least_focus_steps(&self, focused: &[&str]) -> u64 {
        steps::least_focus_steps(focused)
    }

    fn knobs(&self) -> Vec<Knob> {
        let mut knobs = steps::knobs(0.8);
        knobs.extend([
            // No smaller: the two longest delays must outlast the longest
            // `late_after`, or a call a task left behind could never end late.
            DELAY_SCALE.share(
                Class::Plan,
                1.0,
                0.4..=5.0,
                "a factor on the delays of slowed calls",
            ),
            SLOWED_CALLS.range(
                Class::Plan,
                1..=3,
                1..=8,
                "how many calls of a step it slows",
            ),
            HANG.share(
                Class::Plan,
                0.35,
                0.0..=1.0,
                "the share of slowed steps with a call that never answers",
            ),
            // Each at least its default, which is the margin for the last poll of
            // a task the operation aborted.
            LATE_AFTER.duration(
                Class::Judge,
                DEFAULT_LATE_AFTER,
                DEFAULT_LATE_AFTER..=Duration::from_millis(300),
                "how long after its operation a slow call may still end",
            ),
            SETTLE.duration(
                Class::Judge,
                HUNG_SETTLE,
                HUNG_SETTLE..=Duration::from_secs(60),
                "how long a step waits for the calls its operation left behind",
            ),
        ]);
        knobs
    }

    /// A step can stay silent while each call it slows waits in turn, the longest
    /// delay every time, then while it waits for what its operation left behind,
    /// and then for quiet.
    fn settings(&self, _self_test: Option<&SelfTest>, knobs: &Knobs) -> Settings {
        let longest = (DELAYS_MS[DELAYS_MS.len() - 1] as f64 * DELAY_SCALE.get(knobs)).round();
        let delays = longest as u64 * *SLOWED_CALLS.get(knobs).end();
        let silence =
            Duration::from_millis(delays) + SETTLE.get(knobs) + Duration::from_secs(QUIET_SECS);
        let count = STEPS.get(knobs);
        let per_step = Duration::from_secs(60).max(silence);
        Settings {
            stall: Duration::from_secs(60).max(silence + Duration::from_secs(10)),
            deadline: Duration::from_secs(120) + per_step * count as u32,
            ..steps::settings(count)
        }
    }

    fn self_tests(&self) -> &'static [&'static SelfTest] {
        SELF_TESTS
    }

    fn checks(&self) -> &'static [&'static dyn StepCheck] {
        &[]
    }

    fn runtime(&self, _knobs: &Knobs) -> Runtime {
        steps::RUNTIME
    }

    fn store(&self, rng: &mut PlanRng) -> StoreChoice {
        match rng.chance(0.5) {
            true => StoreChoice::SharedMemory,
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

impl StepPlanner for SlowStorage {
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
        if !rng.chance(TROUBLE.get(knobs)) {
            return None;
        }
        let targets = slow_calls(op, scale, store);
        let delay_scale = DELAY_SCALE.get(knobs);
        let mut delays: Vec<Delay> = Vec::new();
        for _ in 0..rng.range(SLOWED_CALLS.get(knobs)) {
            let (method, area, max_nth) = *pick_from(rng, &targets);
            let nth = rng.range(1..=max_nth);
            let millis = (*pick_from(rng, &DELAYS_MS) as f64 * delay_scale).round() as u64;
            // The store wrapper slows a listing or a deletion only before
            // its request.
            let after = !matches!(method, Method::List | Method::Delete) && rng.chance(0.5);
            if !delays
                .iter()
                .any(|delay| (delay.method, delay.area, delay.nth) == (method, area, nth))
            {
                delays.push(Delay {
                    method,
                    area,
                    nth,
                    millis,
                    after,
                });
            }
        }
        let hang = rng.chance(HANG.get(knobs)).then(|| {
            let (method, area, max_nth) = *pick_from(rng, &targets);
            HangPoint {
                method,
                area,
                nth: rng.range(1..=max_nth),
                // Only a write can land and then never answer.
                after: method == Method::Put && rng.chance(0.5),
                grace_ms: rng.range(20..=500),
            }
        });
        Some(Incident::Slow { delays, hang })
    }
}

#[test]
fn selection_draws_every_focusable_operation() {
    steps::check_draws(&SlowStorage);
}

#[test]
fn selection_focus_takes_every_other_step() {
    steps::check_focus(&SlowStorage, 16);
}

/// A slow step targets calls its operation makes, and slows or hangs a listing or a
/// deletion only before its request, the only way the store wrapper can.
#[test]
fn selection_plans_what_the_store_wrapper_can_do() {
    let everything = BTreeSet::new();
    let (mut delays, mut hangs) = (0, 0);
    for seed in 1..=48 {
        let (header, plan) = steps::plan_of(&SlowStorage, seed, &everything);
        for step in &plan.steps {
            let Some(Incident::Slow { delays: slow, hang }) = &step.incident else {
                continue;
            };
            let calls = slow_calls(&step.op, &header.scale, header.store);
            let made = |method: Method, area: Area, nth: u64| {
                calls
                    .iter()
                    .any(|&(m, a, highest)| m == method && a == area && nth <= highest)
            };
            assert!((1..=3).contains(&slow.len()), "seed {seed}: {slow:?}");
            for delay in slow {
                assert!(
                    made(delay.method, delay.area, delay.nth),
                    "seed {seed}: {delay:?}"
                );
                assert!(
                    !delay.after || !matches!(delay.method, Method::List | Method::Delete),
                    "seed {seed}: {delay:?}"
                );
            }
            if let Some(hang) = hang {
                assert!(
                    made(hang.method, hang.area, hang.nth),
                    "seed {seed}: {hang:?}"
                );
                assert!(
                    !hang.after || hang.method == Method::Put,
                    "seed {seed}: {hang:?}"
                );
                hangs += 1;
            }
            delays += slow.len();
        }
    }
    assert!(delays > 0 && hangs > 0, "{delays} delays and {hangs} hangs");
}
