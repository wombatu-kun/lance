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

The `stress` target runs seeded scenarios, one child process per seed, and checks every step against a model. The scenarios cover storage faults, cancellation, process kills, memory limits, heap growth, concurrency, slow storage and a long soak. It has `test = false`, so a plain `cargo test` skips it:

```shell
cargo test --profile ci -p lance --test stress -- --test-threads=2
```

Where to read more:

- `stress/README.md`: how to run it, read what it reports, and add a scenario or an operation;
- `stress/AGENTS.md`: how to triage a failure and change the harness;
- the top of `stress/main.rs`: what each scenario checks.
