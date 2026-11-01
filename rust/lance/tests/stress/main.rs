// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Seeded stress scenarios: storage faults, cancellation, the death of the
//! process, memory limits and heap growth.
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
//! `process_kill` plans the same points, but there the process sends itself a
//! SIGKILL. Its dataset is on disk, through the plain local path or through
//! `file-object-store://`, which lets every write meet a point. A seed goes on
//! in a new process, which first checks what the dead one left behind and then
//! concludes the step: the dataset must read as if the death was a clean end of
//! that step, and a cleanup at the end must remove everything the death left
//! that it can see. A write through a local object store goes by way of a file
//! `name#digits`, which its listings hide: those are only reported, in the
//! `hidden_after_kill` and `hidden_after_cleanup` observations.
//! Each death replays exactly, except for one in flight.
//!
//! `memory_limits` runs on the plain local path, in a cgroup of its own, below a
//! delegated systemd scope (`systemd-run --user`, no privileges needed; a host
//! without it fails the scenario). A step's limit is one of: `memory.max` at
//! what the cgroup holds plus a budget, which ends in the OOM killer's SIGKILL
//! and goes on as above; `LANCE_MEM_POOL_SIZE`, which ends in spilling or in a
//! clean error; the memory limit of an FTS build; the I/O buffer of a scan.
//! Checks and retries run without a limit, so a budget concerns one operation.
//! A merge insert that runs out of a pool because a `RepartitionExec` took a share
//! of it is a defect of Lance rather than of the budget: it does not fail the seed
//! and shows as the `known_defect:merge_insert_pool_share` observation.
//! A seed fixes the limits, not the moment of a kill, which is why the first
//! scenario is there too: kills land on allocation peaks and rarely in a commit.
//! Both turn on Lance's memory events and check them against what happened.
//!
//! `heap_growth` repeats one cycle of work on a steady number of rows and
//! compares the heap after the cycles, with every handle dropped. It counts what
//! Rust code holds rather than the resident set, which also holds what the
//! allocator keeps; what C libraries allocate for themselves does not show.
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
//! with; a merge insert writes what the rows already hold, so a lost update does
//! not show; a SIGKILL loses nothing the kernel holds, so a missing `fsync` does
//! not show either.
//!
//! Run with `cargo test -p lance --test stress -- --test-threads=2`. Each
//! invocation gets a directory of its own, with `summary.txt`, `env.txt` and a
//! directory per failing seed: the plan, the step log, the storage calls, the
//! child's output and, for a hang, its thread states and stacks. Under limits it
//! also holds `cgroup.log` with the memory of the cgroup every 200 ms and
//! `memory.log` with Lance's memory events: the last start without an end is
//! where a killed process stood. `replay.sh` there holds the command that reruns
//! the seed with the same settings.
//!
//! Environment:
//! - `LANCE_STRESS_SEED`: run only this seed, and keep its artifacts.
//! - `LANCE_STRESS_SEEDS` (16) and `LANCE_STRESS_SEED_BASE` (1): which seeds run.
//! - `LANCE_STRESS_REPEAT` (1): how many times each seed runs.
//! - `LANCE_STRESS_STEPS` (8): operations per seed.
//! - `LANCE_STRESS_ARTIFACTS`: where the artifacts go, `$CARGO_TARGET_TMPDIR/stress`
//!   by default.
//! - `LANCE_STRESS_KEEP=1`: keep the artifacts of passing seeds too; `logs` keeps
//!   them without the dataset and the temporary files.
//! - `LANCE_STRESS_CALIBRATE=1`: plan no incident and no limit, and report the
//!   heap each operation takes, to size the budgets in `scenario.rs`.
//! - `LANCE_STRESS_STALL_SECS` (60): how long a child may go without progress.
//! - `LANCE_STRESS_DEADLINE_SECS` (120 + 60 per step): how long a child may run.
//! - `LANCE_STRESS_PLAN`: the plan a replay expects; a different one fails.
//! - `LANCE_STRESS_STANDALONE=1`: run a seed in this process, for gdb or rr.
#![cfg(target_os = "linux")]

mod harness;
mod incidents;
mod leaks;
mod limits;
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

#[test]
fn process_kill() {
    harness::run(ScenarioKind::ProcessKill);
}

#[test]
fn memory_limits() {
    harness::run(ScenarioKind::MemoryLimits);
}

#[test]
fn heap_growth() {
    harness::run(ScenarioKind::HeapGrowth);
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

#[test]
fn harness_reports_foreign_kill() {
    harness::check_self_test(SelfTest::ForeignKill);
}

#[test]
fn harness_reports_kill_then_missing_file() {
    harness::check_self_test(SelfTest::KillThenMissingFile);
}

#[test]
fn harness_reports_oom_recovery() {
    harness::check_self_test(SelfTest::OomRecovery);
}

#[test]
fn harness_reports_hang_under_limit() {
    harness::check_self_test(SelfTest::HangUnderLimit);
}

#[test]
fn harness_reports_memory_events() {
    harness::check_self_test(SelfTest::MemoryEvents);
}

#[test]
fn harness_reports_heap_leak() {
    harness::check_self_test(SelfTest::HeapLeak);
}
