// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Seeded stress scenarios: storage faults and cancellation.
//!
//! Every seed runs in its own child process, a re-run of this test binary, so
//! a hang, an abort or a leak in one seed cannot hide in another, and the
//! parent can take the stacks of a hung child before killing it. A seed fixes
//! the whole plan of a run: the data, the operations, and where each of them
//! fails or is cancelled. It does not fix thread interleavings, which request
//! is the nth when requests race, or the unseeded sampling and k-means of an
//! IVF build, so a failure replays only with some probability.
//!
//! `storage_faults` injects object store errors into appends, deletes, index
//! builds and their commits, either before a request or after it (a lost
//! response), on a `shared-memory://` dataset that a separate session reads
//! back without faults. `cancellation` drops the same operations at a named
//! point: a storage call, an index build stage or a written fragment, on a
//! local or a `shared-memory://` dataset.
//!
//! After every step the dataset must hold exactly the expected rows in every
//! column, and the expected indices, whose answers must match the model. Files
//! that manifests reference must exist. An operation that returned an error
//! must not have committed unless it reported its status as unknown, one that
//! did not commit must succeed when retried, and no tokio task, temporary file
//! or descriptor may be left behind. The run ends with a cleanup that must
//! leave no unreferenced file.
//!
//! Known blind spots: work in the blocking pool and in the CPU and rayon pools
//! is visible only through what it leaves behind; a local store writes data,
//! index, transaction and deletion files around the object store, so faults
//! there reach only manifests and listings; a delete always runs on a handle
//! that has caught up, since its effect on a stale one depends on what it raced
//! with.
//!
//! Run with `cargo test -p lance --test stress -- --test-threads=2`. Each
//! invocation gets a directory of its own, with `summary.txt`, `env.txt` and a
//! directory per failing seed: the plan, the step log, the storage calls, the
//! child's output and, for a hang, its thread states and stacks. `replay.sh`
//! there holds the command that reruns the seed with the same settings.
//!
//! Environment:
//! - `LANCE_STRESS_SEED`: run only this seed, and keep its artifacts.
//! - `LANCE_STRESS_SEEDS` (16) and `LANCE_STRESS_SEED_BASE` (1): which seeds run.
//! - `LANCE_STRESS_REPEAT` (1): how many times each seed runs.
//! - `LANCE_STRESS_STEPS` (8): operations per seed.
//! - `LANCE_STRESS_ARTIFACTS`: where the artifacts go, `$CARGO_TARGET_TMPDIR/stress`
//!   by default.
//! - `LANCE_STRESS_KEEP=1`: keep the artifacts of passing seeds too.
//! - `LANCE_STRESS_STALL_SECS` (60): how long a child may go without progress.
//! - `LANCE_STRESS_DEADLINE_SECS` (120 + 60 per step): how long a child may run.
//! - `LANCE_STRESS_PLAN`: the plan a replay expects; a different one fails.
//! - `LANCE_STRESS_STANDALONE=1`: run a seed in this process, for gdb or rr.
#![cfg(target_os = "linux")]

mod harness;
mod incidents;
mod leaks;
mod scenario;

use harness::{ScenarioKind, SelfTest};

#[test]
fn storage_faults() {
    harness::run(ScenarioKind::StorageFaults);
}

#[test]
fn cancellation() {
    harness::run(ScenarioKind::Cancellation);
}

/// The entry point of a seed's child process. A normal test run skips it.
#[test]
fn child() {
    scenario::run_child_if_requested();
}

#[test]
fn harness_reports_hang() {
    harness::check_self_test(SelfTest::Hang);
}

#[test]
fn harness_reports_panic() {
    harness::check_self_test(SelfTest::Panic);
}

#[test]
fn harness_reports_task_leak() {
    harness::check_self_test(SelfTest::TaskLeak);
}

#[test]
fn harness_reports_tmp_leak() {
    harness::check_self_test(SelfTest::TmpLeak);
}

#[test]
fn harness_reports_model_mismatch() {
    harness::check_self_test(SelfTest::ModelMismatch);
}

#[test]
fn harness_reports_column_mismatch() {
    harness::check_self_test(SelfTest::ColumnMismatch);
}

#[test]
fn harness_reports_orphan() {
    harness::check_self_test(SelfTest::Orphan);
}
