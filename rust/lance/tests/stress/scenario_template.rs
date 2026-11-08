// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A scenario to start from: every method a scenario has, each with what it is
//! for. It builds with the harness, so it keeps up with the traits, and runs
//! nowhere, since `scenarios!` does not list it. `README.md` says how a copy of it
//! becomes a scenario.
//!
//! A scenario of the step engine draws a plan of steps, each one operation with at
//! most one incident or limit, and the engine runs, resumes and checks them: it
//! implements `StepPlanner` and returns `steps::plan` from `plan`, as
//! `scenario/cancellation.rs` does. One whose work does not fit steps, as `soak`
//! or `heap_growth`, draws a plan of its own and runs it through `Body`. Keep the
//! part you need.

use arrow_array::RecordBatchReader;
use async_trait::async_trait;

use crate::engine::data::Scale;
use crate::engine::ops::{IndexKind, Limit, Model, Op};
use crate::engine::run::Run;
use crate::engine::steps::{OpMix, StepCheck, StepPlanner, StepRules};
use crate::engine::{Body, ContentRules, Failure, Header, PlanInput, PlanRng, StoreChoice};
use crate::incidents::Incident;
use crate::knobs::{Knob, Knobs};
use crate::scenario::{Runtime, Scenario, SelfTest, Settings};

/// What `scenarios!` and `self_tests!` take the scenario from.
pub const SCENARIO: &dyn Scenario = &Template;

struct Template;

// Each self-test is a `SelfTest` const that plants one defect, and
// `self_tests! { <test>: <CONST>, ... }`, `<test>` being the `name` of its const,
// lists them as `SELF_TESTS` and makes a test of each,
// `scenario::<name>::self_test::<test>`. Here it stays a comment: its tests would
// run.

impl Scenario for Template {
    /// The name of its module, which is also that of its test, of the directories
    /// of its seeds and of `LANCE_STRESS_ONLY`.
    fn name(&self) -> &'static str {
        todo!()
    }

    /// The kinds of trouble that pick it in `LANCE_STRESS_ONLY`, such as `faults` or
    /// `memory`. A new kind goes into the list of `README.md` too.
    fn trouble_kinds(&self) -> &'static [&'static str] {
        todo!()
    }

    /// The operations its trouble can go on, as `op:` names them. In the step
    /// engine its `OPS`: every operation its plans draw.
    fn focusable_ops(&self) -> Vec<&'static str> {
        todo!()
    }

    /// The fewest steps a plan needs for a focus on these operations to have any.
    /// The step engine has `steps::least_focus_steps`.
    fn least_focus_steps(&self, _focused: &[&str]) -> u64 {
        todo!()
    }

    /// Its knobs, each with the number the code would otherwise hold as its
    /// default. They start from `knobs::harness(<seeds>)`, which the step engine's
    /// `steps::knobs(<share of steps in trouble>)` holds already.
    fn knobs(&self) -> Vec<Knob> {
        todo!()
    }

    /// How the parent runs its seeds with these knobs: the steps, the stall, the
    /// deadline, what a child gets in its environment, and whether a seed runs in a
    /// cgroup of its own. The step engine has `steps::settings`.
    fn settings(&self, _self_test: Option<&SelfTest>, _knobs: &Knobs) -> Settings {
        todo!()
    }

    /// Its self-tests: `SELF_TESTS` of `self_tests!`, or none.
    fn self_tests(&self) -> &'static [&'static SelfTest] {
        todo!()
    }

    /// The checks it adds after each step of the step engine, such as
    /// `memory_events::CHECK`. Usually none.
    fn checks(&self) -> &'static [&'static dyn StepCheck] {
        todo!()
    }

    /// Files of a seed's directory that the parent keeps beside `summary.txt`, as
    /// `soak` keeps `soak.csv`. None unless it says so: delete this to keep that.
    fn artifacts(&self) -> &'static [&'static str] {
        todo!()
    }

    /// The tokio runtime of a child: its worker threads, how long an idle thread
    /// of the blocking pool stays, and whether a failed seed may leave tasks behind
    /// as the process exits. The step engine has `steps::RUNTIME`.
    fn runtime(&self, _knobs: &Knobs) -> Runtime {
        todo!()
    }

    /// The store of a seed, the first draw of its plan: a local directory,
    /// `shared-memory://`, or a file object store.
    fn store(&self, _rng: &mut PlanRng) -> StoreChoice {
        todo!()
    }

    /// How much data a seed starts with and writes: `SMALL`, `MEDIUM` or `LARGE`
    /// of `engine/data.rs`.
    fn scale(&self) -> Scale {
        todo!()
    }

    /// The rest of a seed's plan, after its header: `Box::new(steps::plan(self,
    /// rng, input))` in the step engine, a `Body` of its own otherwise. It draws
    /// from `rng` alone, so that a seed plans the same every time.
    fn plan(&self, _rng: &mut PlanRng, _input: &PlanInput<'_>) -> Box<dyn Body> {
        todo!()
    }
}

/// For a scenario of the step engine.
impl StepPlanner for Template {
    /// What its trouble does and which checks its steps take: whether the process
    /// dies where another scenario cancels, what a step that absorbed its incident
    /// observes (a name of `OBSERVATIONS` in `engine.rs`), and whether a fired
    /// incident explains an error.
    fn rules(&self) -> &'static StepRules {
        todo!()
    }

    /// What a step that no focus takes runs, by a roll of 0 to 99: the ranges of
    /// deletes, cleanups and index optimizations first; a roll none takes builds an
    /// index from 55 on, and below runs the ranges of `others`, else an append.
    /// Every operation it can draw goes into `OPS`, and the other way round.
    fn op_mix(&self) -> &'static OpMix {
        todo!()
    }

    /// The indices a focus on `create_index` builds, and one on `optimize_indices`
    /// before there is an index.
    fn index_kinds(&self) -> &'static [IndexKind] {
        todo!()
    }

    /// The operation of a step that it decides before the mix, as `memory_limits`
    /// builds the index on the key first in half its seeds. None unless it says so:
    /// delete this to keep that.
    fn opening_op(&self, _rng: &mut PlanRng, _index: usize) -> Option<Op> {
        todo!()
    }

    /// Whether every step catches its handle up to the latest version first. No
    /// unless it says so: delete this to keep that.
    fn always_resync(&self) -> bool {
        todo!()
    }

    /// How long a request runs, in microseconds, before a planned death takes the
    /// process. 0 unless it says so: delete this to keep that.
    fn in_flight_us(&self, _rng: &mut PlanRng) -> u64 {
        todo!()
    }

    /// The limit of a step, given the model with its operation: a cgroup, which
    /// needs `cgroup: true` in `settings`, or one of Lance's own, its pool, the
    /// memory of an FTS build or the I/O buffer of a scan. None unless it says so:
    /// delete this to keep that.
    fn limit(&self, _rng: &mut PlanRng, _op: &Op, _model: &Model, _knobs: &Knobs) -> Option<Limit> {
        todo!()
    }

    /// The incident of a step: a fault of a store call, a point to cancel or kill
    /// at, a delay. It is drawn at every step, even where a focus, a self-test or a
    /// calibration drops it, so that dropping one moves no other draw.
    fn incident(
        &self,
        _rng: &mut PlanRng,
        _store: StoreChoice,
        _scale: &Scale,
        _op: &Op,
        _knobs: &Knobs,
    ) -> Option<Incident> {
        todo!()
    }
}

/// The plan of a scenario with a driver of its own.
struct TemplatePlan;

#[async_trait(?Send)]
impl Body for TemplatePlan {
    /// The lines of the plan below its header: they go into `plan.log`, and so into
    /// the fingerprint of the plan that a replay pins.
    fn render(&self) -> String {
        todo!()
    }

    /// How many steps, rounds or samples it takes. The final checks are the one
    /// after.
    fn steps(&self) -> usize {
        todo!()
    }

    /// The knobs its lines show already, which the line of knobs of the plan leaves
    /// out. None unless it says so: delete this to keep that.
    fn shown_knobs(&self) -> &'static [&'static str] {
        todo!()
    }

    /// What the check of the content, which every scenario takes, holds a seed to.
    fn content(&self) -> ContentRules {
        todo!()
    }

    /// Whether the dataset takes stable row ids. No unless it says so: delete this
    /// to keep that.
    fn stable_row_ids(&self) -> bool {
        todo!()
    }

    /// The rows the dataset starts with. The rows of `engine/data.rs` unless it
    /// says so: delete this to keep that.
    fn initial_source(&self, _header: &Header) -> Box<dyn RecordBatchReader + Send> {
        todo!()
    }

    /// The model of those rows, what the checks hold the dataset to. Delete this
    /// with `initial_source`.
    fn initial_model(&self, _header: &Header) -> Model {
        todo!()
    }

    /// Logs of its own in the directory of a seed, which the setup opens before it
    /// counts the descriptors of the baseline. None unless it says so: delete this
    /// to keep that.
    fn journals(&self) -> &'static [&'static str] {
        todo!()
    }

    /// Runs the plan on the dataset the setup wrote, checking as it goes, and
    /// returns the first `Failure`, with its class, or `Ok`.
    async fn run(&self, _run: &mut Run) -> Result<(), Failure> {
        todo!()
    }
}
