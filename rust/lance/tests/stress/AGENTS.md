# Stress harness: agent guide

Use this guide to interpret a stress run, to triage a failing seed into a report, or to change the harness. `README.md` describes the artifacts, `summary.txt` and the failure classes. The top of `main.rs` says what each scenario checks and cannot see.

## Triage a failed run

Input: a run directory `<scenario>-<date>-<time>-<pid>/`, a CI artifact holding one (`gh run download <run id> -n stress-<scenario>-<run number>`), or the failure message of `cargo test`, which names each failing seed with its class, replay command and directory.

1. **Pin the run.** Note the commit, the settings from `env.txt`, and the `plan knobs` and `judge knobs` lines of `summary.txt`. Every later step uses that commit and those knobs.
2. **Rule out the harness.**
   - Run the self-tests on that commit: `cargo test --profile ci -p lance --test stress -- self_test:: --test-threads=2`. Under `systemd-run --user --scope` if the machine has no desktop session.
   - A failing self-test, the class `harness` or `plan_changed`, or a `dead` or `undeclared` line in `summary.txt` is a harness problem. Report it as one and stop.
3. **Rule out the machine.** These are the environment until a replay on an idle machine repeats them:
   - a `hang` whose stacks show nothing but waiting;
   - a `crash` with nothing in `stderr.log`;
   - an OOM kill that no planned limit explains (`journalctl -u systemd-oomd`, `dmesg`).
4. **Check what is known.** What the code excuses or avoids names its issue:
   - the cleanup boundary, and the order of round 1, that avoid #3718;
   - the duplicate key that Lance allows where none is declared (#5633);
   - the observations `hidden_after_kill`, `hidden_after_cleanup` (#10587) and `optimize_left_unindexed` (#10711).

   Then search lance-format/lance for the mechanism: `gh issue list --search '<words> in:title,body'` and `gh pr list --search ...`, open and recently closed.
5. **Read the evidence**, in this order:
   - the seed's `failed` entry in `summary.txt`;
   - `steps.log` back from `result=fail`;
   - that step in `plan.log`;
   - `calls.log` around the step;
   - the files the table below names for the class.

   Follow one write from its call to the version it made.
6. **Replay.**
   - Run `sh <seed dir>/replay.sh` from the root of the repository, with `--profile ci` after `cargo test` if the run had it (`exe=` in `env.txt`).
   - Repeat it with `LANCE_STRESS_REPEAT=10` and report how many tries failed.
   - For `concurrency`, also replay with `LANCE_STRESS_SERIAL=0` changed to `=1` in the line. A failure that goes away there needs a race.
   - A replay that fails as `plan_changed` ran on a different commit than the run, or with a knob of the plan edited in its `LANCE_STRESS_SET`.
7. **Reduce.** Write a test in `rust/lance/tests/` on the public API:
   - a few rows and one or two dataset handles;
   - the exact operations;
   - a store wrapper only where a fault is needed;
   - next to a control case that behaves.

   It must fail on the run's commit for the reason the seed failed. Without it the report says "not reduced", and its verdict cannot be "Lance defect".
8. **Write the report** below. Stop there: a person decides whether to file it.

## Where to look

| Class | Look at |
| --- | --- |
| `content`, `duplicate_key`, `torn_snapshot` | The detail. In `steps.log`, the operations that committed the version. The index that answered, if one did. |
| `lost_write`, `lost_commit`, `version_jump` | The versions in `steps.log`. The manifest puts and listings in `calls.log`. |
| `stale_read`, `snapshot_changed` | `reads.log`, and any cleanup in the round. Outside `concurrency`, the `digest_of=` lines of `steps.log`. |
| `lineage` | The rows in the detail, against `row_id_lineage.md`. |
| `error_after_commit` | The fault that fired: `fail_*` lines in `calls.log`. |
| `error_without_fault`, `retry_failed` | The error in the detail. In `concurrency`, the conflicts that `rust/lance/src/io/commit/conflict_matrix/cases.rs` allows. |
| `stale_handle` | The operation's lines in `steps.log`. |
| `missing_files` | The delete of the missing file in `calls.log`, and the cleanup or death behind it. |
| `orphans` | The files in the detail: their area, and their time against the oldest manifest kept. Hidden files are on the `hidden_after_cleanup` line. |
| `unreadable`, `validate`, `cleanup`, `setup` | The error in the detail, and `stderr.log`. |
| `index_not_used` | The plan in the detail. |
| `memory_events` | `memory.log`, against the limit of the step. |
| `heap_growth` | The heap per cycle in the detail. |
| `soak_growth` | The `soak series=` lines at the end of `steps.log`, and `soak.csv`. Does the series follow the operations count, or the versions and files of the dataset? Run the seed afresh twice as long: `LANCE_STRESS_SEED=<n>` with `soak.secs` in `LANCE_STRESS_SET` and no `LANCE_STRESS_PLAN`, since a replay pins the length. A leak keeps its rate per operation; warm-up levels off. The heap and `malloc_bytes` are judged; the resident set is only recorded, since it also holds what the allocator keeps. A cache over its capacity, or what the maintenance at the end left, the detail says. |
| `task_leak`, `tmp_leak`, `fd_leak` | The counts in the detail, and the last operation of the step. |
| `hung_call` | The slowed or hung call in `calls.log`: its phase, its method, and whether it writes. A write that lands after the caller gave up is the finding. |
| `panic`, `crash` | `stderr.log`. |
| `hang` | `threads.txt`, `stacks.txt`, and how loaded the machine was. |
| `harness`, `plan_changed` | Step 2: the harness, not Lance. For `plan_changed`, the commit of the replay and the knobs of the plan in its `LANCE_STRESS_SET`. |

## Observations

Counts in `summary.txt`, and `observation=` lines in `steps.log`. They are not failures. A count of zero for what a plan exists to cause means that part tested nothing.

| Observation | Means |
| --- | --- |
| `fault_absorbed`, `delay_absorbed`, `cancel_missed`, `kill_missed`, `limit_absorbed` | The trouble of a step fired, and the operation still ended Ok: Lance retried through a fault, a delay changed nothing, a cancellation or a kill came after the work, the work fit its limit. |
| `status_unknown_landed`, `status_unknown_lost` | After a lost response the status was unknown, and the commit landed or did not. The harness checks that it landed once at most. |
| `cancel_after_commit`, `killed_after_commit`, `late_commit` | The commit landed before the cancellation or the kill, or after the operation was given up. |
| `killed_at:<op>:<point>`, `killed:<cause>` | Where a death struck, and why. |
| `not_reached` | A planned incident or limit never fired in its step. A few are normal. A pool or FTS limit, a hold or a forced pair that enough seeds plan and that never bites makes a `dead` line. |
| `error_on_stale_handle` | An operation on a handle behind the latest version failed, which is allowed; in `slow_storage` only for a commit conflict. |
| `legal_error:<op>` | `concurrency`: an operation lost a conflict that `conflict_matrix/cases.rs` allows, or a commit over a hole was refused. |
| `internal_retry:<op>`, `lost_manifest_race:<op>` | Lance retried inside the operation, or lost the race to write a manifest and retried. |
| `overlap:<X>:<Y>`, `shared_fragment:<X>:<Y>` | `concurrency`: two commits overlapped, or also landed on a fragment in common. |
| `pair_planned:<X>:<Y>`, `pair_met:<X>:<Y>`, `pair_missed:<X>:<Y>` | A forced pair was planned, committed as planned, or committed before the operation it was to land over. |
| `hole_landed:<op>:<hidden>`, `hole_refused:<op>:<hidden>`, `hole_conflict:<op>:<hidden>`, `hole_masked:<op>:<hidden>`, `hole_error:<op>:<hidden>`, `hole_nothing:<op>:<hidden>` | What became of a commit staged over a hole in the history: it landed, Lance refused it, it lost a retryable conflict, a file gone first masked it, it failed otherwise, or there was nothing to stage. |
| `new_key_race`, `only_new_keys_race`, `only_new_keys_race_indexed` | Upserts of the same keys raced in a round of a seed that declares its key; `only_new_keys_race` when the keys were all new, `_indexed` when an index on the key was in place. |
| `old_snapshot_gone` | A cleanup in the round removed the version an old reader held. |
| `gate_timeout` | A planned wait gave up after `concurrency.gate_limit`, 10 s by default. Many mean a plan that cannot hold. |
| `spill:<op>`, `failed_deletes_under_limit` | Under a limit, an operation spilled, or the kernel refused an unlink with ENOMEM. |
| `orphans_after_kill`, `tmp_after_kill` | What a death left before the cleanup. |
| `hidden_after_kill`, `hidden_after_cleanup` | `name#digits` files of a killed local write, which listings hide (#10587). In `soak`, which kills nothing, one after a cleanup fails the seed as `orphans`. |
| `orphan_transactions`, `orphan_deletions` | Files newer than the oldest manifest kept, which cleanup does not list yet. |
| `index_dir_extra` | A file in an index directory that its index does not list. |
| `expired_version_broken` | A cleanup that stopped half way left an expired version without some of its files. Cleanup deletes files before manifests, by design. |
| `optimize_left_unindexed` | After an injected read error, an index optimization skipped an index and returned Ok (#10711). |
| `hung_read_left_behind` | A hung read of a data or index file outlived a dropped scan. Lance's I/O scheduler does this by design. |
| `soak_retry:<op>`, `soak_gave_up:<op>` | `soak`: an operation lost a conflict and retried, or one that changes no rows gave up after 4 tries. |
| `soak_unsettled` | `soak`: a sample waited its 5 s for quiet and measured what was still starting or ending. |

## Report

Write it in English, as Markdown:

```markdown
## <scenario> seed <n>: <class>, <what Lance did wrong, in Lance's terms>

- Run: <directory or CI run>, commit <sha>, `<replay command>`
- Knobs off their default: <the `plan knobs` and `judge knobs` lines of `summary.txt`, or none>
- Replay: fails <k> of <m> tries; serial: <result, or n/a>
- Expected: <what the model holds, and the contract behind it: a doc, an API, a test>
- Actual: <the detail or the log line, quoted>
- Sequence: <the steps, store calls and incidents that matter>
- Reduced test: <path, and how it fails on <sha>>, or "not reduced" and why
- Suspected cause: <file:line and the mechanism>, or "unknown"
- Prior reports: <issues and PRs searched, and what matched>
- Verdict: Lance defect | harness defect | environment | known (#n)
```

## Rules

- **Report, never act.** Do not file issues, comment, or open pull requests from a triage on your own: the report goes to a person.
- **Quote, do not paraphrase.** Values come from the files. Give paths and the few lines that matter, not log dumps.
- **Do not hide a failure.**
  - Never make a seed pass by weakening a check, raising a tolerance or adding an exclusion, unless the report shows the check is wrong. A judge knob off its default counts as raising a tolerance.
  - An exclusion for a Lance defect names its issue.
- **A seed number is not a regression test.** The next change to the harness plans it differently.

## Changing the harness

- **Know where things live.**
  - `engine/` runs a seed in its child process: the plan's header, the step engine (`engine/steps.rs`), the checks every scenario shares, and the data.
  - `scenario/<name>.rs` decides one scenario: its plan, its settings, its rules, its self-tests and the checks it adds. To add a scenario or an operation, follow the sections "Adding a scenario" and "Adding an operation" of `README.md`; `scenario_template.rs` is the scenario to copy.
  - A check that only some scenarios take, such as `memory_events.rs`, comes in through their `checks()`.
  - `harness.rs` is the parent side: it starts the children, watches them and judges what they leave.
- **Keep plans stable.** Plans must not change unless the change means to. Run the same seeds twice with `LANCE_STRESS_KEEP=logs`, and once on the previous build, and compare the `plan.log` files. The golden tests among the unit tests of the target (`-- registry_ knobs::`) hold the plans of the defaults and the default of every knob.
- **Make a number a knob, not an edit.** A number a run may want to change goes into the `knobs()` of its scenario, with the number the code held as its default, in the class `plan` when the plan or the run reads it, `judge` when a verdict does. The table of `knobs::list` shows it, the record of a replay pins it.
- **Prove every new check.** A new check needs a self-test that plants its defect, listed in `self_tests!` of its scenario. A new failure class goes into `CLASSES` of `engine.rs`, the table in `README.md` and the table above; a new observation into `OBSERVATIONS` and the table of observations.
- **Run the mutants** of the scenario you changed. Keep the header of `main.rs` true.
