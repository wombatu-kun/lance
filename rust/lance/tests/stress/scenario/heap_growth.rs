// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `heap_growth`: one cycle of work over and over on a steady number of rows,
//! and the heap after each cycle, with every handle dropped.

use std::ops::Range;

use async_trait::async_trait;

use crate::engine::data::{MEDIUM, Scale};
use crate::engine::ops::{IndexKind, Model, Op};
use crate::engine::run::Run;
use crate::engine::steps::{self, Step, StepCheck, StepRules, render_steps};
use crate::engine::{Body, ContentRules, Failure, PlanInput, PlanRng, StoreChoice, fail};
use crate::leaks::{live_heap_bytes, quiesce};
use crate::scenario::{Runtime, Scenario, SelfTest, Settings, nothing_more};

pub const SCENARIO: &dyn Scenario = &HeapGrowth;

/// Heap goes missing on every cycle of a repeated workload.
const HEAP_LEAK: SelfTest = SelfTest {
    name: "heap_leak",
    expected: &["heap_growth"],
    reason: None,
    verify: nothing_more,
};

self_tests! {
    heap_leak: HEAP_LEAK,
}

struct HeapGrowth;

/// One cycle, repeated until a leak has had time to show. The
/// first cycles fill global caches and pools: the heap grows by about 7 KiB per
/// cycle, then stops by the eighth and stays within 1 KiB.
const CYCLE_OPS: usize = 9;
const CYCLES: usize = 16;
const EARLY_CYCLES: Range<usize> = 8..11;
const LATE_CYCLES: Range<usize> = 13..16;
/// The heap of a healthy run varies by about 1 KiB between these cycles.
const GROWTH_LIMIT: i64 = 16 << 10;

const RULES: StepRules = StepRules {
    dies: false,
    absorbed: "limit_absorbed",
    incidents_explain_errors: true,
    stale_errors_are_conflicts: false,
    keeps_session: false,
    checks_kept_versions: false,
};

impl Scenario for HeapGrowth {
    fn name(&self) -> &'static str {
        "heap_growth"
    }

    fn trouble_kinds(&self) -> &'static [&'static str] {
        &["memory"]
    }

    /// Its cycles are its own.
    fn focusable_ops(&self) -> Vec<&'static str> {
        Vec::new()
    }

    fn least_focus_steps(&self, _focused: &[&str]) -> u64 {
        0
    }

    /// Repeated cycles are the point of this scenario: it has a size of its own.
    fn settings(&self, _self_test: Option<&SelfTest>, _steps: u64) -> Settings {
        steps::settings((CYCLES * CYCLE_OPS) as u64)
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
        StoreChoice::Local
    }

    fn scale(&self) -> Scale {
        MEDIUM
    }

    fn plan(&self, rng: &mut PlanRng, input: &PlanInput<'_>) -> Box<dyn Body> {
        let header = input.header;
        Box::new(Cycles {
            steps: plan_cycles(rng, &header.scale, header.initial_rows),
        })
    }
}

/// The same cycle over and over, on a steady number of rows: whatever grows is
/// not the data.
fn plan_cycles(rng: &mut PlanRng, scale: &Scale, initial_rows: i64) -> Vec<Step> {
    let added = rng.range(scale.added_rows.0..=scale.added_rows.1) as i64;
    let mut model = Model::new(initial_rows);
    let mut oldest = 0;
    let mut steps = Vec::with_capacity(CYCLES * CYCLE_OPS);
    for _ in 0..CYCLES {
        let before = model.live.len();
        let append = Op::Append {
            first_id: model.next_id,
            rows: added,
        };
        model.apply(&append);
        let merge = Op::MergeInsert {
            first_id: model.next_id - added / 2,
            rows: added,
        };
        model.apply(&merge);
        let grown = (model.live.len() - before) as i64;
        let delete = Op::Delete {
            lo: oldest,
            hi: oldest + grown,
        };
        model.apply(&delete);
        oldest += grown;
        for op in [
            // First, so that the merge takes its indexed path.
            Op::CreateIndex(IndexKind::BTreeId),
            append,
            merge,
            delete,
            Op::CreateIndex(IndexKind::BTree),
            Op::CreateIndex(IndexKind::IvfFlat),
            Op::CreateIndex(IndexKind::Inverted),
            Op::Compact,
            Op::Scan,
        ] {
            steps.push(Step {
                op,
                incident: None,
                resync: true,
                in_flight_us: 0,
                limit: None,
            });
        }
    }
    steps
}

/// The steps of a seed, cycle after cycle.
struct Cycles {
    steps: Vec<Step>,
}

#[async_trait(?Send)]
impl Body for Cycles {
    fn render(&self) -> String {
        render_steps(&self.steps)
    }

    fn steps(&self) -> usize {
        self.steps.len()
    }

    fn content(&self) -> ContentRules {
        ContentRules {
            versions_stay_listed: true,
            indices_cover_the_model: true,
            digests: false,
            hidden_files_fail: false,
        }
    }

    async fn run(&self, run: &mut Run) -> Result<(), Failure> {
        let mut heap_after_cycle = Vec::with_capacity(CYCLES);
        for (index, step) in self.steps.iter().enumerate() {
            Box::pin(run.run_step(index + 1, step, &RULES)).await?;
            if (index + 1) % CYCLE_OPS == 0 {
                heap_after_cycle.push(end_cycle(run, index + 1).await);
            }
        }
        check_growth(&heap_after_cycle, self.steps.len())
    }
}

/// The heap held once a cycle is over. The handle is new by now, so its session
/// holds nothing of the cycle.
async fn end_cycle(run: &Run, n: usize) -> i64 {
    if run.self_test == Some(&HEAP_LEAK) {
        Box::leak(vec![1u8; 8 << 10].into_boxed_slice());
    }
    quiesce(run.baseline_tasks, &run.tmp, run.quiet).await;
    let live = live_heap_bytes();
    run.journal
        .line(format_args!("step={n} heap_cycle live_kib={}", live >> 10));
    live
}

fn check_growth(heap_after_cycle: &[i64], n: usize) -> Result<(), Failure> {
    let median = |cycles: Range<usize>| {
        let mut values = heap_after_cycle[cycles].to_vec();
        values.sort_unstable();
        values[values.len() / 2]
    };
    let early = median(EARLY_CYCLES);
    let late = median(LATE_CYCLES);
    if late - early > GROWTH_LIMIT {
        return Err(fail(
            "heap_growth",
            n,
            format_args!(
                "the heap grew from {early} to {late} bytes over cycles of the same work: {heap_after_cycle:?}"
            ),
        ));
    }
    Ok(())
}
