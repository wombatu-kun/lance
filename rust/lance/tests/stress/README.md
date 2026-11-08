# Stress harness

The `stress` target runs Lance's write paths under planned trouble and checks every step against a model of what the dataset must hold. A seed fixes the plan: the operations, and where a store call fails, an operation is cancelled or killed, which limit applies and which operations race. Each seed runs in a child process, so a crash or a hang ends that seed only, and a failing seed replays from its seed.

What each scenario checks, what it cannot see and every environment variable are at the top of `main.rs`, its knobs in `knobs::list`. How to triage a failure, step by step, is in `AGENTS.md`. How to add a scenario or an operation is in the two sections before the runs on GitHub.

## Scenarios

| Scenario | Trouble | 16 seeds take |
| --- | --- | --- |
| `storage_faults` | Object store errors in writes, index builds, cleanups and commits, before a request or after it. | 30 s |
| `cancellation` | The same operations dropped at named points. | 30 s |
| `process_kill` | A SIGKILL at those points, then a new process that checks what the death left. | 50 s |
| `memory_limits` | A cgroup limit that ends in the OOM killer, or Lance's own limits. Needs cgroup v2 and a user service manager. | 3.5 min |
| `heap_growth` | A repeated cycle of work whose heap must come back. | 75 s a seed |
| `concurrency` | Writers, readers and cleanups at once, with forced commit races. | 2.5 min |
| `slow_storage` | Slowed and hung store calls. An operation given up must leave no call running. | 3 min |
| `soak` | A long-lived service whose memory, threads, tasks, descriptors and dataset must level off, and whose caches stay within their capacity. | `soak.secs` a seed, 180 s by default; one seed by default |

Times are rough: a 12-thread laptop, two scenarios at a time.

The seeds of a scenario are the test `scenario::<name>`. Its self-tests, `scenario::<name>::self_test::<test>`, each plant one defect and check that the harness reports it with the right class and files, or set up a hard case that must pass. Run all of them, `-- self_test::`, after any change to the harness; they take about 2 min.

## Running it

The target has `test = false`, so a plain `cargo test` skips it. It builds on Linux only.

```shell
# Every scenario on its default seeds, the self-tests and the unit tests.
cargo test --profile ci -p lance --test stress -- --test-threads=2
# Every self-test.
cargo test --profile ci -p lance --test stress -- self_test:: --test-threads=2
# One scenario, 200 seeds from 33.
LANCE_STRESS_SEEDS=200 LANCE_STRESS_SEED_BASE=33 cargo test --profile ci -p lance --test stress -- --exact scenario::cancellation
# One seed; its artifacts stay even if it passes.
LANCE_STRESS_SEED=7 cargo test --profile ci -p lance --test stress -- --exact scenario::concurrency
# Only some kinds of trouble, or trouble only on some operations.
LANCE_STRESS_ONLY=kill,latency cargo test --profile ci -p lance --test stress -- --test-threads=2
LANCE_STRESS_ONLY=op:optimize_indices cargo test --profile ci -p lance --test stress -- --test-threads=2
# memory_limits has to start inside the user's service manager, which a desktop terminal is; from ssh or CI, start it so.
systemd-run --user --scope --quiet -- cargo test --profile ci -p lance --test stress -- --exact scenario::memory_limits
# Knobs: longer plans and wider rounds, or every step in trouble.
LANCE_STRESS_SET=steps=12,concurrency.workers=4..=8 cargo test --profile ci -p lance --test stress -- --exact scenario::concurrency
LANCE_STRESS_SET=trouble=1 cargo test --profile ci -p lance --test stress -- --exact scenario::storage_faults
# The seed counts of the nightly run, from the root of the repository; a night's own seeds also take its LANCE_STRESS_SEED_BASE.
LANCE_STRESS_SET_FILE=$PWD/rust/lance/tests/stress/profiles/nightly.txt cargo test --profile ci -p lance --test stress -- --exact scenario::concurrency
# A three-hour soak.
LANCE_STRESS_SET=soak.secs=3h cargo test --profile ci -p lance --test stress -- --exact scenario::soak
# Every knob, with its default and bounds.
cargo test --profile ci -p lance --test stress -- --exact knobs::list --nocapture
```

- `LANCE_STRESS_ONLY` takes kinds (`faults`, `cancel`, `kill`, `memory`, `concurrency`, `latency`, `soak`), scenario names, and `op:<operation>`.
  - The operations are `append`, `delete`, `create_index`, `optimize_indices`, `merge_insert`, `compact`, `scan` and `cleanup`.
  - In `concurrency`, its own operations work too: `update`, `merge_insert_columns`, `contended_upsert`, `counter_update`.
  - The self-tests of the chosen kinds and scenarios run too; a list of `op:` alone keeps every self-test.
- Knobs change, without a rebuild, what the code holds as defaults.
  - `LANCE_STRESS_SET` takes `<knob>=<value>` pairs separated by commas. A key without a prefix sets that knob of every scenario that has it, `<scenario>.<knob>` that of one. A length takes a unit (`ms`, `s`, `m`, `h`, `d`), a size one too (`B`, `KiB`, `MiB`, `GiB`), a range is `a..=b`.
  - `LANCE_STRESS_SET_FILE` names a file of pairs, one per line with `#` comments, for a profile. What a variable sets wins over it. Empty, as an input of CI left blank gives it, it names no file.
  - The older variables still set their knobs: `LANCE_STRESS_SEEDS`, `_SEED_BASE`, `_REPEAT`, `_STEPS`, `_STALL_SECS`, `_DEADLINE_SECS`, `_SOAK_SECS`. A knob set both by one of them and in `LANCE_STRESS_SET` fails the run.
  - A key no scenario has, or a value of the wrong kind or out of bounds, fails every test of the run before a child starts.
  - More steps, cycles or a longer soak raise the deadline by themselves; a longer wait at a hold, or slower calls, raise the stall too. A value you set stays as set.
- `LANCE_STRESS_KEEP=logs` keeps the logs of passing seeds. The counts that show a run did what it planned come from them. `LANCE_STRESS_KEEP=1` keeps the datasets as well.
- On a laptop, run at most two scenarios at a time: more turns slowness into hangs.
- To keep running one build while the sources change, build with `--no-run` and hard-link the binary. Children re-execute `/proc/self/exe`.

## What a run leaves

Each scenario writes `<scenario>-<date>-<time>-<pid>/` under `LANCE_STRESS_ARTIFACTS`, by default `$CARGO_TARGET_TMPDIR/stress`. A seed's directory stays when the seed failed or `LANCE_STRESS_KEEP` is set.

| File | Holds |
| --- | --- |
| `summary.txt` | The knobs off their default, the counts and the failing seeds, rewritten after every seed. |
| `env.txt` | The environment of the run. |
| `seed-<n>/plan.log` | The plan: steps or rounds, operations, incidents, hold points, and a `knobs` line for the knobs of the plan off their default. |
| `seed-<n>/steps.log` | One line per event (`t=<ms> step=<n> ...`), ending in `result=pass` or `result=fail class=<class> step=<n> detail=...`. |
| `seed-<n>/calls.log` | Every store call: the phase that made it, the method, area and path, and what the harness did to it. |
| `seed-<n>/reads.log` | What the readers of `concurrency` read. |
| `seed-<n>/stdout.log`, `stderr.log` | The child's output. A panic lands in `stderr.log`. |
| `seed-<n>/replay.sh` | The command that replays the seed, and a commented one that runs it in one process for gdb or rr. |
| `seed-<n>/threads.txt`, `stacks.txt` | After a hang: the thread states and, when gdb can attach, the stacks. |
| `seed-<n>/cgroup.log`, `memory.log` | The cgroup's memory every 200 ms under a cgroup limit, and Lance's memory events in `memory_limits` and `process_kill`. The last start without an end is where a killed process stood. |
| `seed-<n>/soak.csv`, `soak-seed-<n>.csv` | `soak`: one row per sample, with the series it judges and the counts it only records. The parent copies it next to `summary.txt`. |

## Reading summary.txt

A run of `storage_faults`, shortened:

```
scenario=storage_faults run_dir=... steps=8 repeat=1 seeds=1..=16
runs=16 failed=0
outcomes:
  err+committed=6
  err+uncommitted=81
  ok+committed=41
observations:
  fault_absorbed=10
  status_unknown_landed=6
incidents:
  fail:put:manifests:response_lost:once planned=5 fired=5 ok+committed=5
holds:
```

- `plan knobs`, `judge knobs`: the knobs off their default, after the first line, when there are any. A judge knob off its default can let a defect pass, so a report names it.
- `runs`, `failed`: the tries run, a repeat of a seed counting as one more, and the tries that failed.
- `outcomes`: how steps ended (`ok`, `err`, `cancelled`, `killed`) and whether their commit landed.
- `observations`: counts of events, not verdicts. They show what a run exercised. A zero for what part of a plan exists to cause means that part tested nothing. `AGENTS.md` lists them.
- `incidents`: per kind, how often each was planned and fired, and how those steps ended. `holds`: how often each was planned, fired and waited. One planned and never fired tests nothing.
- `dead limit`, `dead hold`, `dead pair`: something that enough seeds planned never bit, fired or met. Seeds count, not tries: one seed plans the same in every try. They fail the run: the harness, or a change in Lance, stopped exercising it. They are not Lance defects in themselves.
- `undeclared`: a class or an observation that `CLASSES` or `OBSERVATIONS` of `engine.rs` does not declare, which this guide therefore cannot explain. It fails the run as a problem of the harness: declare the name and describe it.
- `failed`: per failing seed, the class, the failing journal line, the replay command and the directory.

## When a seed fails

1. Run the self-tests. If one fails, the harness is broken, not Lance.
2. Read the seed's entry under `failed`, `steps.log` back from `result=fail`, that step in `plan.log`, and what reached the store in `calls.log`.
3. Replay it: `sh <seed dir>/replay.sh` from the root of the repository, with `--profile ci` after `cargo test` if the run had it (`exe=` in `env.txt`). A replay on a changed harness fails as `plan_changed`.
   - The replay pins every knob but how many seeds run, in `LANCE_STRESS_SET`, and leaves out a stall or a deadline that no source set: it derives them again. It removes from the shell the variables that would change it unseen; a short form of a knob it pins, such as `LANCE_STRESS_STEPS`, fails it instead. Change a knob by editing its value there, and the stall and the deadline follow it as in a fresh run. An edited knob of the plan fails as `plan_changed` too: to run a `soak` seed longer, run it afresh with `LANCE_STRESS_SEED=<n>` and `soak.secs`, without `LANCE_STRESS_PLAN`.
   - Faults, cancellations and kills replay exactly, except a kill in flight.
   - For a race, repeat the replay with `LANCE_STRESS_REPEAT`.
   - For `concurrency`, also replay with `LANCE_STRESS_SERIAL=0` changed to `=1` in the line. A failure that goes away there needs a race.
4. Reduce it to a small test in `rust/lance/tests/` that shows the defect without the harness, next to a control case that behaves. That test goes into the issue and into the fix. A seed is no regression test, since the next change to the harness plans it differently.
5. Search the open issues and pull requests for the same mechanism before filing.

The classes `harness` and `plan_changed`, a `dead` or `undeclared` line, and a hang seen only on a loaded machine belong to the harness or the machine until shown otherwise.

## Failure classes

| Class | What failed |
| --- | --- |
| `content` | Rows, columns or index answers differ from the model. |
| `duplicate_key` | A version holds a key twice. |
| `lost_write` | A committed write is missing from a later version. |
| `torn_snapshot` | A version holds part of a commit, or a worker's rows as of no prefix of its operations. |
| `stale_read` | A reader saw a version older than one that finished before it opened. |
| `snapshot_changed` | A version read differently twice. |
| `lineage` | Row ids or row versions break `row_id_lineage.md`. |
| `lost_commit` | An operation returned Ok, but the latest version does not hold its commit, or holds less of it than it reported. |
| `error_after_commit` | An operation returned an error, although its commit landed. |
| `error_without_fault` | An operation failed with nothing in the plan to explain it. |
| `retry_failed` | The retry of a failed step failed, or so did an operation run alone after a legal conflict. |
| `version_jump` | The versions moved in a way the operations cannot explain, versions are gone that no cleanup could take, or the ways to find the latest version disagree. |
| `stale_handle` | An operation returned Ok, but its handle is not on the version it made. |
| `missing_files` | A manifest references files that are gone. |
| `orphans` | A cleanup left files that nothing references. |
| `unreadable`, `validate`, `cleanup` | Reading the dataset or `Dataset::validate` failed; a cleanup failed, took what it had to keep, or left what it had to take. |
| `index_not_used` | A query that has to use an index did not. |
| `memory_events` | An index build's memory events are missing, or disagree with its limit, its commit or its progress. |
| `heap_growth` | The heap grew over cycles of the same work. |
| `soak_growth` | In `soak`, a series grew from the middle third of the run to the last by more than its tolerance, a cache held more than its capacity, or the maintenance at the end left more than one fragment, or more than one segment of an index. |
| `task_leak`, `tmp_leak`, `fd_leak` | Tasks, temporary files or descriptors outlived the step. |
| `hung_call` | A slowed or hung store call outlived its operation, or a write reached the store after a hung read was let go. |
| `panic` | The child panicked. |
| `hang` | The child made no progress for the knob `stall`, or ran past `deadline`. |
| `crash` | The child ended without a whole verdict, or with one its exit contradicts, and not by a death the plan foresaw. |
| `setup` | The first process of a seed could not create its dataset. |
| `harness`, `plan_changed` | The harness failed, or a replay planned differently: another commit, or a knob of the plan edited. |

## Checking the harness itself

- **Self-tests**, after every change to the harness. They take a deadline from the knobs and nothing else, and run plans of 3 steps with a stall of 15 s.
- **Golden tests.** The unit tests of the target hold the plans that the defaults draw and the default of every knob: `cargo test --profile ci -p lance --test stress -- registry_ knobs:: selection_ replay_`. A change to either fails `registry_default_plans_keep_their_fingerprints` or `knobs::defaults_keep_their_record`.
- **Determinism.** The same seeds run twice with `LANCE_STRESS_KEEP=logs` must plan the same (`plan.log`) and get the same verdicts, except for how often a memory limit fired.
- **Mutants.** Plant a realistic defect in Lance, run the scenario meant to catch it, expect its class, and restore the source.
  - A mutant that no seed catches is a blind spot, or an edit on a line that never runs.
  - Repeat after changing a scenario.
  - The tooling is not in this repository.

## Adding a scenario

Start from `scenario_template.rs`: it has every method a scenario has, each with what it is for, and it builds with the harness, so it keeps up with the traits. A scenario of the step engine draws steps, each one operation with at most one incident or limit, and the engine runs, resumes and checks them; `scenario/cancellation.rs` is the smallest. One whose work does not fit steps, such as `soak`, draws and runs a plan of its own through `Body`.

1. Copy the template to `scenario/<name>.rs` and fill it in; `name()` returns `<name>`.
2. In `scenario.rs`, add `pub mod <name>;` and `<name>` in `scenarios!`. Without the first the build breaks, without the second `registry_names_scenarios_after_their_modules_and_self_tests_once` fails.
3. On the step engine, the names of `OPS` and the operations of the mix go together: `steps::check_draws` fails unless the plans draw every name of `OPS` and nothing else. Make it and `steps::check_focus` tests of the scenario, as `cancellation.rs` does.
4. A number a run may want to change is a knob of `knobs()`, with that number as its default.
5. Prove each check of its own with a self-test: a `SelfTest` that plants one defect and names the classes it may fail with, listed in `self_tests!` under its `name`. Plant the defect where the code compares `self_test` with its const: `Run::inject` or `die_if_a_self_test_says_so` in `engine/steps.rs`, or the scenario's own `Body::run`. A self-test nothing plants fails on a passing seed.
6. Golden tests:
   - `knobs::defaults_keep_their_record` fails until `GOLDEN_RECORDS` in `knobs.rs` has a line for the scenario; its message shows the line.
   - `GOLDEN_PLANS` in `engine.rs` holds the plans of the scenarios it lists: add rows for the new one, or nothing holds its plans.
   - Add its case to `registry_plan_knobs_reach_their_plans` and `registry_judge_knobs_leave_their_plans`, and its stall and deadline to `derived_defaults_hold_the_numbers_of_the_code_and_only_rise`.
7. Describe it: its row in the table of scenarios above, its paragraph and blind spots at the top of `main.rs`, and a new kind of trouble, class or observation in this file, in `AGENTS.md` and in `CLASSES` or `OBSERVATIONS` of `engine.rs`. `registry_docs_name_every_scenario_class_and_observation` fails until they agree.
8. Run the unit tests of the target (`-- registry_ knobs:: selection_ replay_`), its self-tests, its seeds twice with `LANCE_STRESS_KEEP=logs` to compare their plans, and a mutant of the Lance code it is there to check.

## Adding an operation

An operation of the step engine is a variant of `Op` in `engine/ops.rs`. `concurrency` and `soak` have operations of their own, `RoundOp` and `SoakOp`.

1. Add the variant; what it holds must be `Copy`. The compiler then asks for its arm in `focus_name`, `Display`, `Model::apply`, `op_future`, `storage_calls` and `cancel_point`.
2. Check the matches that fall back instead:
   - `Op::shape`, how many versions it may commit: `Exactly1` unless it says otherwise. A wrong shape fails a correct Lance as `lost_commit`, `version_jump` or `retry_failed`.
   - `Op::kind`; `pool_floor_kib` and the retry of a failed step in `engine/steps.rs`; the budgets of `memory_limits`.
   - `focused_op` in `engine/steps.rs` panics on a name it does not know.
   - `cancel_point` gives any operation but an append a point at its first read of data, and `slow_calls` of `slow_storage` takes any operation but a cleanup to commit through the version hint.
3. Give it its trouble: the store calls a fault or a delay can hit (`storage_calls`, and `slow_calls` of `slow_storage`) and the points a cancellation or a kill can take (`cancel_point`). An index build gets its stages from the progress hooks.
4. Add it to the plans of the scenarios that should run it: a variant of `Other` with its range in their `OpMix`, and its name in their `OPS`. Only their plans change. `registry_default_plans_keep_their_fingerprints` then names, per variant, the first scenario whose plans changed, with its new fingerprint: put it into `GOLDEN_PLANS` once the change is meant, and run it again until it passes.
5. In `concurrency`, an operation is a `RoundOp`: its weight in `OP_WEIGHTS`, its arm in `plan_op`, how many commits it makes, its hold points, and the conflicts it may lose in `is_legal_error`, which follows `rust/lance/src/io/commit/conflict_matrix/cases.rs`. In `soak` it is a `SoakOp` with a `Draw` and its share of `DRAWS`, which sum to 100; one that changes rows goes into `changes_rows`, or it gives up where it should fail. A new weight or share changes every plan of its scenario.
6. Run the unit tests of the target, the self-tests, the seeds of each scenario it joined, and a mutant of its Lance code.

## Automated runs on GitHub

This is a proposal. Nothing here runs in Lance's CI yet, and the example has not run on GitHub.

| When | What | Time after the build | Blocks merging |
| --- | --- | --- | --- |
| Every pull request | Build the target: add `--test stress` to the clippy job, as for `arrow_ordering`. | about a minute | yes |
| A pull request that changes `tests/stress/` | The unit tests and self-tests, then every scenario but `soak` on its default seeds. | about 20 min | yes |
| A pull request that changes commits, conflict resolution, cleanup or the write paths | `LANCE_STRESS_ONLY=concurrency,faults`, seeds 1-16. | about 3 min | not until its noise on the runners is measured |
| Every night | Every scenario but `soak`, on seeds no run has used, one scenario per job. | about an hour a job | no |
| Every week | `soak` for three hours, and the same seeds twice to compare plans. | 3 to 4 h | no |

- **Seeds.**
  - Pull requests use the fixed seeds 1-16, so a red check comes from the change.
  - Nights take new seeds, derived from the run number, and write the first one to the job summary.
  - In `concurrency`, 16 seeds in a row meet every forced pair and every kind of hole in the history; 64 meet every hole with every turn of the pairs. Fewer seeds, or fewer than 7 rounds, judge only some of the pairs, and the summary does not say which.
  - Keep no list of seeds that once failed: the next change to the harness plans them differently.
- **Runners.**
  - Use the `ci` profile: the budgets and watchdogs were sized on it.
  - On a slow runner, raise `<scenario>.stall` and `<scenario>.deadline` above their defaults at the top of `main.rs` before believing a `hang`: a key without a prefix sets them for every scenario, and lowers those whose default is higher. Install gdb, so that a hang leaves its stacks.
  - `memory_limits` and its self-tests need a user service manager, which a hosted runner lacks. The example starts one by enabling lingering, which is untried. Check that it worked in a step of its own, and when it did not, skip those tests and say so in the job summary.
  - After `memory_limits`, check that `systemd-oomd` killed nothing.
- **Reports.**
  - Run with `LANCE_STRESS_KEEP=logs`, put every `summary.txt` into the job summary, and upload the artifacts on failure for two weeks.
  - A failed night comments on the open issue of its scenario, or opens one, labelled `stress-triage` (the label has to exist).
  - Keep `bug:` out of the title. This repository routes `bug:` issues to automatic fixers, and a stress failure is not a Lance bug until it is reduced.
- **Schedules.**
  - They run on the default branch only, and GitHub turns them off after 60 days without activity in a public repository.
  - Failure mail goes only to whoever last changed the cron line, which is why the issues exist.
  - Guard the jobs with `if: github.repository == 'lance-format/lance'`.

### Example

The weekly soak is the same `stress` job with `--exact scenario::soak`, `LANCE_STRESS_SET=soak.secs=3h` and `timeout-minutes: 240`. A manual run takes knobs in the input `set`.

```yaml
name: Stress nightly

on:
  schedule:
    - cron: "17 3 * * *"
  workflow_dispatch:
    inputs:
      seed_base:
        description: "First seed; empty derives it from the run number"
        required: false
        default: ""
      set:
        description: "Knobs, as LANCE_STRESS_SET takes them"
        required: false
        default: ""

permissions:
  contents: read

concurrency:
  group: stress-nightly
  cancel-in-progress: false

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: "1"
  LANCE_STRESS_KEEP: logs

jobs:
  selftests:
    if: github.repository == 'lance-format/lance'
    runs-on: ubuntu-24.04
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - uses: actions-rust-lang/setup-rust-toolchain@a0b538fa0b742a6aa35d6e2c169b4bd06d225a98 # v1
      - uses: Swatinem/rust-cache@779680da715d629ac1d338a641029a2f4372abb5 # v2
        with:
          key: stress
      - uses: taiki-e/install-action@66068bfca13dcb2ea07c3f613ca2836a37c755d5 # protoc
        with:
          tool: protoc
      - name: Start a user service manager
        run: |
          sudo loginctl enable-linger "$USER"
          runtime="/run/user/$(id -u)"
          for _ in $(seq 30); do [ -S "$runtime/bus" ] && break; sleep 1; done
          echo "XDG_RUNTIME_DIR=$runtime" >> "$GITHUB_ENV"
      - name: Self-tests
        run: |
          if [ -S "$XDG_RUNTIME_DIR/bus" ]; then
            systemd-run --user --scope --quiet -- cargo test --profile ci --locked -p lance --test stress -- self_test:: --test-threads=2
          else
            echo "No user service manager: the self-tests of memory_limits did not run." >> "$GITHUB_STEP_SUMMARY"
            cargo test --profile ci --locked -p lance --test stress -- self_test:: --skip memory_limits::self_test --test-threads=2
          fi

  stress:
    needs: selftests
    name: ${{ matrix.scenario }}
    runs-on: ubuntu-24.04
    timeout-minutes: 120
    strategy:
      fail-fast: false
      matrix:
        scenario:
          - storage_faults
          - cancellation
          - process_kill
          - concurrency
          - slow_storage
          - memory_limits
          - heap_growth
    env:
      LANCE_STRESS_ARTIFACTS: ${{ github.workspace }}/stress-artifacts
      # How many seeds each scenario runs; what `set` gives wins over it.
      LANCE_STRESS_SET_FILE: ${{ github.workspace }}/rust/lance/tests/stress/profiles/nightly.txt
      LANCE_STRESS_SET: ${{ inputs.set }}
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - uses: actions-rust-lang/setup-rust-toolchain@a0b538fa0b742a6aa35d6e2c169b4bd06d225a98 # v1
      - uses: Swatinem/rust-cache@779680da715d629ac1d338a641029a2f4372abb5 # v2
        with:
          key: stress
      - uses: taiki-e/install-action@66068bfca13dcb2ea07c3f613ca2836a37c755d5 # protoc
        with:
          tool: protoc
      - name: Install gdb
        run: sudo apt-get install -y gdb
      - name: Pick the seeds
        env:
          SEED_BASE: ${{ inputs.seed_base }}
        run: |
          base="${SEED_BASE:-$((100000 + GITHUB_RUN_NUMBER * 1000))}"
          echo "LANCE_STRESS_SEED_BASE=$base" >> "$GITHUB_ENV"
          echo "Seeds of ${{ matrix.scenario }} from $base" >> "$GITHUB_STEP_SUMMARY"
      - name: Start a user service manager
        if: matrix.scenario == 'memory_limits'
        run: |
          sudo loginctl enable-linger "$USER"
          runtime="/run/user/$(id -u)"
          for _ in $(seq 30); do [ -S "$runtime/bus" ] && break; sleep 1; done
          echo "XDG_RUNTIME_DIR=$runtime" >> "$GITHUB_ENV"
      - name: Build
        run: cargo test --profile ci --locked -p lance --test stress --no-run
      - name: Run
        run: |
          wrap=()
          if [ "${{ matrix.scenario }}" = memory_limits ]; then
            wrap=(systemd-run --user --scope --quiet --)
          fi
          "${wrap[@]}" cargo test --profile ci --locked -p lance --test stress -- --exact "scenario::${{ matrix.scenario }}"
      - name: Summary
        if: always()
        run: |
          for summary in "$LANCE_STRESS_ARTIFACTS"/*/summary.txt; do
            [ -f "$summary" ] || continue
            { echo '```'; cat "$summary"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
          done
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4
        if: failure()
        with:
          name: stress-${{ matrix.scenario }}-${{ github.run_number }}
          path: stress-artifacts
          retention-days: 14

  triage:
    needs: [selftests, stress]
    if: failure() && github.event_name == 'schedule'
    runs-on: ubuntu-24.04
    permissions:
      actions: read
      issues: write
    steps:
      - name: Open or update an issue per failed job
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_REPO: ${{ github.repository }}
          RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
        run: |
          gh run view "$GITHUB_RUN_ID" --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name' |
            while read -r job; do
              title="stress: $job failed in the nightly run"
              number=$(gh issue list --state open --label stress-triage --search "in:title \"$title\"" --json number --jq '.[0].number // empty')
              if [ -n "$number" ]; then
                gh issue comment "$number" --body "Failed again: $RUN_URL"
              else
                gh issue create --title "$title" --label stress-triage --body "Run: $RUN_URL. Triage it as rust/lance/tests/stress/AGENTS.md describes before filing a bug."
              fi
            done
```
