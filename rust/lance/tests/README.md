Tests for memory and IO usage.

## Debugging memory usage

Once you've identified a test that is using too much memory, you can use
bytehound to find the source of the memory usage. (Note: we need to run
bytehound on the binary, not on cargo, so we have to extract the test binary path.)

The `RUST_ALLOC_TIMINGS` environment variable tells the tracking allocator
to logs the start and end of each allocation tracking session, which makes it
easier to correlate the bytehound output with the code.

```shell
TEST_BINARY=$(cargo test --test resource_tests --no-run 2>&1 | tail -n1 | sed -n 's/.*(\([^)]*\)).*/\1/p')
LD_PRELOAD=/usr/local/lib/libbytehound.so \
    RUST_ALLOC_TIMINGS=true \
    $TEST_BINARY resource_test::write::test_memory_usage_write \
bytehound server memory-profiling_*.dat
```

## Stress tests

The `stress` target runs seeded scenarios, one child process per seed: `storage_faults` injects object store errors into writes, index builds and commits, and `cancellation` drops those operations at named points. After every step it checks the data, the indices and the leftover resources against a model. The target has `test = false`, so a plain `cargo test` skips it; start it by name.

```shell
cargo test -p lance --test stress -- --test-threads=2
LANCE_STRESS_SEEDS=200 LANCE_STRESS_SEED_BASE=33 cargo test -p lance --test stress -- --exact cancellation
```

Artifacts go to `$CARGO_TARGET_TMPDIR/stress` unless `LANCE_STRESS_ARTIFACTS` says otherwise. The failure message lists every failing seed with its class, the command that replays it and its directory, which holds the plan, the step log, the storage calls, the child's output and, for a hang, its thread states and stacks. What a run checks, its blind spots and all the environment variables are described at the top of `stress/main.rs`.
