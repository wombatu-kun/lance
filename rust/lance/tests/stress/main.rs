// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Seeded stress scenarios: storage faults, cancellation, the death of the
//! process, memory limits, heap growth, concurrency, slow storage and a long run.
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
//! builds and optimizations, cleanups and the commits of all of them, either
//! before a request or after it (a lost response), on a `shared-memory://`
//! dataset that a separate session reads back without faults. `cancellation`
//! drops the same operations at a named point: a storage call, an index build
//! stage or a written fragment, on a local or a `shared-memory://` dataset.
//!
//! `slow_storage` slows the store under the same operations, on a
//! `shared-memory://` or a `file-object-store://` dataset. A step slows one to three
//! of the calls its operation makes by 10 ms to 3 s, before the request or once it
//! has completed, and now and then one call never answers: the step gives the
//! operation up a moment later, as an application whose deadline has passed, and
//! goes on as after a cancellation. A delay alone must neither fail an operation
//! nor change what it does. A slow or hung call that outlives its operation, in a
//! task the operation left behind, fails the seed as `hung_call`, whether still in
//! flight or done well after the operation was over: a write there could land
//! after the caller gave up. A read of a data or index file is the exception, as
//! Lance's I/O scheduler runs each one as a task of its own that a dropped scan
//! does not stop: a hung one shows as the `hung_read_left_behind` observation.
//! Lance retries such a read once it is let go, and a task that waited for it goes
//! on, so from then until the step is quiet no write may reach the store: one that
//! does comes from a task the operation left behind, and fails the seed as
//! `hung_call` too. The calls a step slows are those its operation makes on that
//! store: a `shared-memory://` dataset finds its latest version by listing, a
//! `file-object-store://` one through the version hint.
//!
//! `process_kill` plans the same points, but there the process sends itself a
//! SIGKILL. Its dataset is on disk, through the plain local path or through
//! `file-object-store://`, which lets every write meet a point. A seed goes on
//! in a new process, which first checks what the dead one left behind and then
//! concludes the step: the dataset must read as if the death was a clean end of
//! that step, and a cleanup at the end must remove everything the death left
//! that it can see. A write through a local object store goes by way of a file
//! `name#digits`, which its listings hide: those are only reported, in the
//! `hidden_after_kill` and `hidden_after_cleanup` observations, except in `soak`,
//! where nothing interrupts a write and one fails the seed as `orphans`.
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
//! `concurrency` runs rounds rather than single operations. In a round, 2 to 4
//! workers run their operations at once, each with a handle and a store wrapper
//! of its own, and by seed one `Session` for all or one each, while readers check
//! snapshots. A row belongs to the class `id % (workers + 2)`, and a worker only
//! changes its own class, so every snapshot must show a worker's rows as of some
//! prefix of its operations, while the rows lie mixed in the same fragments. All
//! workers upsert, update and delete the same keys of one shared class, and add 1
//! to the whole of the other. A plan can stop an operation at a named point until another one is
//! over, the points a cancellation uses, which is what makes commits overlap; the
//! `overlap:<X>:<Y>` observations count the pairs that did, and
//! `shared_fragment:<X>:<Y>` those that landed on a fragment in common. A point
//! that a run plans again and again and that never fires fails it. A fast operation
//! rarely sees a slow one land under it, so rounds 3, 5 and 7 of the default eight
//! each force one pair of `PAIRS`: the others stand on the version the round starts
//! from and commit over what the first worker landed, `pair_met:<X>:<Y>` when they
//! did, and a pair that a run forces twice and that never meets fails it. A cleanup
//! in a pair starts once one of the others is at its commit point, so that it runs
//! while files nobody has committed yet exist. Seeds take the pairs in turn. Round 1
//! leaves a hole in the history: the others prepare their write without a commit,
//! the first worker writes, commits a reservation that changes no rows and cleans
//! up every version but that one, and only then do the others commit. The `hole_*`
//! observations count what became of them, and a commit staged for a hole that the
//! round did not make fails it.
//! Lance's own conflict errors are legal only where
//! `rust/lance/src/io/commit/conflict_matrix/cases.rs` allows them, or as the refusal
//! of a commit over the hole, and the operation must then succeed once it is alone. What an
//! operation returns must be right too: the rows it counts, and a handle on a
//! version that holds its write. Once a round is over, every version it made is
//! opened and checked: nothing goes back, one commit does one thing, a row's lineage
//! follows `row_id_lineage.md`, indices answer as their version holds, a version
//! reads the same each time, and a cleanup takes what it should and nothing else. The
//! failure classes of its own are `torn_snapshot`, `lost_write`, `stale_read`,
//! `snapshot_changed`, `duplicate_key` and `lineage`.
//! It stays clear of one known defect and of one thing Lance allows. Outside
//! round 1 a cleanup keeps every version of its round and the one before, so that a
//! write held between its check for conflicts and its manifest never takes the
//! number of a version the cleanup removed
//! (<https://github.com/lance-format/lance/issues/3718>); round 1 commits its staged
//! writes only after its cleanup, and without holds. Workers only race for a
//! new key on a seed with a declared primary key: without one a duplicate is legal,
//! see #5633.
//!
//! After every step the dataset must hold exactly the expected rows in every
//! column, and the expected indices, whose answers must match the model. A
//! version stays listed, with every file it references, until a cleanup may
//! remove it, and the versions a cleanup keeps must read as they did, also
//! after an index build or optimization. A cleanup that returned Ok must have
//! removed every version below its boundary but those whose manifest it failed
//! to delete, and when no deletion failed and no incident fired, every file only
//! those versions referenced. One that may not delete unverified files, whether
//! it returned or was cancelled, must leave every file no version referenced,
//! which a write in progress may own. Only an injected error or a cgroup limit
//! may fail a deletion, and Lance must count each one. A cleanup deletes files
//! before manifests, so one that stopped half way can leave a version it expired
//! listed without some of its files, which shows as the `expired_version_broken`
//! observation. An index optimization that returned Ok must leave no fragment
//! out of an index, on a stale handle none that its version held. Lance skips an
//! index it fails to open and still returns Ok, so after an injected error
//! reading index files that shows as the `optimize_left_unindexed` observation.
//! An operation that returned an error must not have committed unless it
//! reported its status as unknown, one that did not commit must succeed when
//! retried, and no tokio task, temporary file or descriptor may be left behind.
//! The run ends with a cleanup that must leave no unreferenced file.
//!
//! `soak` runs a long-lived service for `LANCE_STRESS_SOAK_SECS`, on a local or a
//! `file-object-store://` dataset: 2 to 4 workers keep a handle each, on one
//! `Session` with caches of a few MiB, and go round appends, merge inserts,
//! updates, deletes, compaction, index optimizations, cleanups, reopening the
//! dataset, scans and index queries. A worker only changes the rows of its class
//! `id % workers`, and its scans and queries check them on its own handle,
//! whatever version that is on. The state of the dataset lasts as long as the run:
//! no operation deletes the older half of the rows a class starts with, and the
//! indices are built once, before the workers start, and from then on only
//! optimized, now and then merging every segment. A cleanup keeps every version
//! from the oldest one a handle is on, see #3718. The run takes a sample at least
//! every 7.5 s and at least 24, each once every worker has finished its operation,
//! maintenance the try it is on, and the runtime is quiet, into `soak.csv`: the
//! heap, what malloc has handed out, descriptors, threads, tokio tasks, the caches
//! of the session, the versions, files and bytes of the dataset, the fragments,
//! index segments and manifest size of its latest version, the partitions of its
//! vector index, and how often compaction, index optimization and the merging of
//! index segments were drawn and gave up. After every sample a cleanup that may
//! delete unverified files must leave no unreferenced file, and every fourth one
//! the dataset must read as the rows of every class. A cache may never hold more
//! than its capacity, from the middle third of the run to the last no series may
//! grow by more than its tolerance, and once the workers stop, merging the
//! segments of every index and then compacting must leave one fragment, which one
//! segment of each index covers; each fails the seed as `soak_growth`. The heap is the series that matters, and what malloc has
//! handed out adds what C code holds. The resident set goes into `soak.csv` as
//! well, but no verdict reads it: glibc keeps the top of every thread's arena,
//! which `malloc_trim` does not give back, so it wanders by tens of MiB with
//! nothing held. A worker's operations are a prefix of what its seed draws, but
//! how far it gets and what it races with depend on timing.
//!
//! Known blind spots: work in the blocking pool and in the CPU and rayon pools
//! is visible only through what it leaves behind; a local store writes data,
//! index, transaction and deletion files around the object store, so faults
//! there reach only manifests and listings; a delete always runs on a handle
//! that has caught up, since its effect on a stale one depends on what it raced
//! with; a merge insert writes what the rows already hold, so a lost update does
//! not show; a SIGKILL loses nothing the kernel holds, so a missing `fsync` does
//! not show either; `slow_storage` slows neither the parts of a multipart upload,
//! which bypass the store wrapper, nor files large enough to need one, and sees a
//! task an operation left behind only through the calls of it that the step slowed
//! or hung and that are still going while the step settles, a slowed read of a
//! data or index file aside, so a write such a task makes at full speed as the
//! operation ends goes unseen. `concurrency` runs in one process, with faults,
//! cancellation and limits off, and never overwrites, restores or changes the
//! schema; its order of events is only as exact as its hold points make it, so a
//! replay may not hit the same interleaving. `soak` checks no further of the
//! conflicts it meets than that a retry on the latest version lands, and does not
//! judge maintenance that keeps losing them: the workers that run it also make the
//! load it races with, so a compaction made 2 s slower loses four tries in five
//! and still lands half of its draws, and only the maintenance at the end of the
//! run, with nothing in flight, must land; its cleanups take turns, and its series
//! show a leak only once it outgrows the noise of a third of the run, and the
//! median of the last third misses growth that starts after the middle of that
//! third. It creates its indices before the first sample, so a leak in creating
//! one shows only in `heap_growth`, which creates all four again every cycle; of
//! creating an index, a run repeats only the training of centroids, when an
//! optimization splits a partition; and since a default optimization folds new
//! rows into the last segment of a scalar or full-text index, only the vector
//! index has segments to merge. A compaction or an index optimization that does
//! not merge only raises the level the series of the dataset settle at, since an
//! update moves rows out of their fragment and a commit drops a fragment with no
//! rows left and a segment that covers none unless it is the last of its index, so
//! only the maintenance at the end of the run shows it. A leak on opening a dataset grows
//! with the samples, whose checks open it without the session, and on the session
//! with the workers' reopenings. A cache is held to the sizes it reports for
//! its entries, so an entry that holds more than its `deep_size_of` shows only in
//! the heap, and only while the cache fills. A thread of the blocking pool lives no
//! longer than a sample interval, so what it keeps for itself goes at every sample.
//! Memory a library maps for itself rather than takes from malloc shows only in
//! the resident set.
//!
//! Run with `cargo test -p lance --test stress -- --test-threads=2`. Each
//! invocation gets a directory of its own, with `summary.txt`, `env.txt` and a
//! directory per failing seed: the plan, the step log, the storage calls, the
//! child's output and, for a hang, its thread states and stacks. Under limits it
//! also holds `cgroup.log` with the memory of the cgroup every 200 ms and
//! `memory.log` with Lance's memory events: the last start without an end is
//! where a killed process stood. `replay.sh` there holds the command that reruns
//! the seed with the same settings. A `soak` seed also leaves its samples beside
//! `summary.txt`, as `soak-seed-N.csv`.
//!
//! Environment:
//! - `LANCE_STRESS_SEED`: run only this seed, and keep its artifacts.
//! - `LANCE_STRESS_SEEDS` (16, for `soak` 1) and `LANCE_STRESS_SEED_BASE` (1): which
//!   seeds run.
//! - `LANCE_STRESS_REPEAT` (1): how many times each seed runs.
//! - `LANCE_STRESS_STEPS` (8): operations per seed; `soak` runs for a time instead.
//! - `LANCE_STRESS_ARTIFACTS`: where the artifacts go, `$CARGO_TARGET_TMPDIR/stress`
//!   by default.
//! - `LANCE_STRESS_KEEP=1`: keep the artifacts of passing seeds too; `logs` keeps
//!   them without the dataset and the temporary files.
//! - `LANCE_STRESS_CALIBRATE=1`: plan no incident and no limit, and report the
//!   heap each operation takes, to size the budgets in `scenario/memory_limits.rs`.
//! - `LANCE_STRESS_STALL_SECS` (60, for `concurrency` and `soak` 150): how long a
//!   child may go without progress.
//! - `LANCE_STRESS_DEADLINE_SECS` (120 + 60 per step, for `concurrency` 300 + 120 per
//!   step, for `soak` its length + 300): how long a child may run, for `soak` at
//!   least its length + 60.
//! - `LANCE_STRESS_SOAK_SECS` (180): how long a `soak` seed runs, at least 24; its
//!   plan and fingerprint hold it. A night takes hours.
//! - `LANCE_STRESS_PLAN`: the plan a replay expects; a different one fails.
//! - `LANCE_STRESS_STANDALONE=1`: run a seed in this process, for gdb or rr.
//! - `LANCE_STRESS_SERIAL=1`: run the operations of a round one at a time; a failure
//!   of `concurrency` that goes away with this needs a race.
//! - `LANCE_STRESS_ONLY`: what a run takes, a comma-separated list. A kind of
//!   trouble (`faults`, `cancel`, `kill`, `memory`, `concurrency`, `latency`,
//!   `soak`) or the name of a scenario picks scenarios, and the self-tests of
//!   those; `memory` takes `soak` too. `op:<operation>` makes
//!   every other step of a seed one of the operations it names that has work
//!   there and puts the incidents and limits on those steps alone, which takes at
//!   least 2 steps. An index optimization has work once an index misses fragments,
//!   so the steps between build an index and append when it has none, and never
//!   take that work away; it alone takes 4 steps. In `concurrency` it keeps the
//!   forced pairs that run one of them, which takes at least 3 rounds, and a round
//!   left without its pair, the first one included, is an ordinary one. A scenario
//!   a kind picks drops out if it has none of the operations, and one picked by
//!   name, `soak` and `concurrency` included, fails the run, as does a name it does
//!   not know or a list that leaves nothing to run.
#![cfg(target_os = "linux")]

mod engine;
mod harness;
mod incidents;
mod leaks;
mod limits;
mod memory_events;
mod scenario;
mod select;

/// The entry point of a seed's child process. A normal test run skips it.
#[test]
fn child() {
    engine::run_child_if_requested();
}
