// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The graph against Lance's own `IVF_RQ`, at equal recall.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift VECTORS=0 DATASET_DIR=/tmp/vamana-bench \
//!   cargo run --profile release-no-lto --example ivf_rq_ab
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 200), `ROWS_PER_PARTITION` (default 8192), `NPROBES`
//! (default 7), `VAMANA_ROWS_PER_PARTITION`, `RQ_ROWS_PER_PARTITION`,
//! `VAMANA_NPROBES`, `RQ_NPROBES` (each defaults to the shared value above),
//! `DEGREE` (default 64), `CODE_BITS` (default 3), `CODE_KIND` (`rq` or `sq`,
//! default `rq`), `SQ_BITS` (default 8, read only when `CODE_KIND=sq`), `WIDTHS`
//! (default `10,20,30,40,60,80,120,160`, each a multiple of `k`),
//! `LIST_SCALES` (default `1`), `BUDGETS` and `QUEUES` (unset: the width sweep
//! above), `CONCURRENCY` (default 1), `CACHE_MB` (default 4096), `TARGET`
//! (default 95), `WARMUP` (default: every query), `RESIDENT_EDGES` (default
//! 0), `REFERENCE_POSITION` (`last` or `both`, default `last`), `ARMS`
//! (`scan`, `walk` or both, default both), `DATASET_DIR` (unset: temporary
//! directories thrown away at the end), `HNSW_EFS` (unset: no HNSW arm),
//! `HNSW_NPROBES` (default 1), `HNSW_URI` (default: the `-p1-hnswsq.lance`
//! directory beside the others) and `IVF_SQ` (default 0).
//!
//! `LANCE_RQ_PRUNE_STATS=1` is Lance's own knob, not this example's: `IVF_RQ`
//! tallies how many rows its two-stage estimator threw away on the binary code
//! alone and reports them through `log`. A binary with no logger installed
//! drops that silently, so this one installs a logger for exactly that target
//! when the knob is set - and only then, so that a pass being timed is never
//! made to write while it is timed. Pair it with
//! `LANCE_RQ_PRUNE_STATS_INTERVAL=1` to see every scan rather than every 1024th.
//!
//! `HNSW_EFS` adds Lance's other graph index, `IVF_HNSW_SQ`, as a third arm,
//! one width sweep per `ef`. It is measured through exactly the code that
//! measures `IVF_RQ`, phases included, because the two differ only in which
//! index the same scanner reaches; this example never builds it, so the
//! directory has to exist already (`examples/hnsw_index.rs` writes it).
//!
//! `IVF_SQ=1` adds a fourth arm: Lance's flat `IVF_SQ`, built here on demand in
//! the `IVF_RQ` arm's own IVF shape - the same partitions probed the same
//! number of times - so that the quantizer is the only thing between the two.
//! It carries Lance's shipped `SQBuildParams::default()`, which is the same
//! eight-bit scalar quantization the `IVF_HNSW_SQ` arm walks over and the same
//! width this crate's `CODE_KIND=sq` walk steers by. That makes it the flat
//! control for the question the graph arms leave open: whether a graph buys
//! anything once a scan is handed the same codes.
//!
//! Every earlier sweep in this crate compares the walk against a scan *this
//! crate* wrote. That scan is the walk's own parts with the graph switched off,
//! so it shares the crate's codes, its cache and its pooled re-score budget -
//! which makes it the right control for "does the graph pay" and the wrong one
//! for "is any of this new". Lance already ships `IVF_RQ`, and with
//! `refine_factor` set its query is the same three-step shape: rank a probed
//! partition's rows by their RaBitQ codes, merge the probes into one list, then
//! read the original vectors of that list and rank them exactly. This example
//! is that comparison. Both of this crate's arms are in it: the scan that won
//! the crate's own sweeps, and the walk the crate exists for.
//!
//! **What is held equal.** Both indexes are built over the same vectors in the
//! same row order at the same `CODE_BITS`, and are queried with the same `k`,
//! the same query set and the same ground truth. Both are given a warm cache,
//! and each point reopens its index so a cache starts empty and is filled by
//! the warmup rather than by the measured pass.
//!
//! **What is deliberately not equal: the shape.** Partition count and probe
//! budget are set per arm, `VAMANA_*` against `RQ_*`, defaulting to one shared
//! value so an invocation that names neither is the run this example started
//! as. They are separable because the two shapes worth comparing are not the
//! same shape: a graph is what answers inside one partition large enough that
//! there is nothing left to route, while `IVF_RQ` answers by probing a few
//! small ones - and at `d = 960` seven probes of 123 cannot reach 0.95 however
//! the rest is set, because the missing neighbours are in partitions the query
//! never opened. Forced onto one shape, one arm is always outside its working
//! point, and the comparison measures that instead of the index.
//!
//! **The one knob, swept.** Each arm carries a candidate list and re-scores it
//! exactly, so the sweep is over its width: `L` here, `k * refine_factor` there.
//! That is the comparison's real subject, because Lance spends one knob where
//! this crate spends two - `refine_factor` widens the candidate list *and* the
//! set of vectors read, while `SearchParams` sets `search_list_size` and
//! `rescore_budget` apart. `LIST_SCALES` asks what the second knob is worth by
//! tying it to the first: at a scale of `n` a probe keeps `n` times the budget
//! and re-scores the budget. A run at `refine_factor` unset is printed too: it
//! is the default, and it reads no original vectors at all.
//!
//! **And the second knob is a second axis, not a second setting of the first.**
//! `BUDGETS` with `QUEUES` sweeps the queue at a budget set outright, which is
//! the shape the knobs actually have: a walk needs a long queue to *reach* its
//! neighbours and a budget to *re-score* them, and at one partition of a
//! million rows those two came apart by a factor of 3.4 in bytes. It is off
//! unless both are set, because it is not the same sweep - the axis is the
//! queue rather than the width, and `WIDTHS` keeps driving the `IVF_RQ` arm, so
//! the reference at the recall target does not move. Note what cannot be asked:
//! Lance re-scores `k * refine_factor` with an integer factor, so a budget that
//! is not a multiple of `k` has no `IVF_RQ` counterpart at all, and the two arms
//! meet only at the recall target, never at a shared budget.
//!
//! **What is not equal, and cannot be made so.** The two builds run their own
//! k-means, so the partitions differ; RaBitQ's rotation is random per build, so
//! the codes differ. Both are the same algorithm drawn twice, and the sweep is
//! wide enough that a partitioning that routes slightly better shows up as a
//! shift along the recall axis rather than as a win. The asymmetry that does not
//! average out is where the exact re-score reads from: this crate keeps a copy
//! of the vectors inside the index partition, so a re-score is a few ranges of
//! one file, while Lance takes them from the dataset, where the same rows are
//! scattered across fragments. That is a real difference in shape and the
//! `requests` column is where it shows.
//!
//! **Time, and why it needs `CONCURRENCY`.** `us` is wall time per query and
//! `cpu` is the process's own core time over the same pass. With one query in
//! flight the first is a latency, and on a laptop it is not reproducible: the
//! same pass repeated varies by half, because a single thread runs at whatever
//! turbo state the chip is in. With enough queries in flight that throughput
//! has stopped growing, the chip sits at its all-core clock and the same pass
//! repeats to within two per cent - so the figure worth quoting is that one,
//! and `bytes`, `iops` and `requests` are unaffected either way. The two arms'
//! wall times are also not the same measurement: this crate's is a library
//! call, Lance's is a whole DataFusion plan, built and executed per query.
//! `IVF_RQ plan only` prints what building one costs with nothing executed, so
//! that part can be subtracted rather than argued about.
//!
//! **The split, and why the two arms get it differently.** Every arm here
//! reaches a candidate list by code and then corrects it by reading original
//! vectors, and the correction is very nearly the whole byte cost of a query -
//! so `search` and `rescore` are printed apart from each other. This crate's
//! arm reports them from the inside: each query attaches its own I/O sink to
//! the files it opens, one for each side of the barrier, and the pass asserts
//! that the two add up to what the scheduler counted. Lance's arm has no such
//! seam, so its split is a difference of two runs measured back to back: with
//! `refine_factor` unset the scan reads no original vector at all, which makes
//! that run exactly its search over codes, and everything the refined run
//! spends above it is the re-score. The recall of that unrefined run is
//! therefore the arm's recall *before* any re-score, printed as `coded`, and it
//! is the same number at every width - the top `k` of a coded list does not
//! depend on how much of the list is kept.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::StreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::scanner::{ExecutionStatsCallback, Scanner};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::IndexType;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::codes::CodeSpec;
use lance_vamana::query::{Neighbor, SearchParams, VamanaIndex, WalkMode};

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const ID_COLUMN: &str = "id";
const VECTOR_FIELD: &str = "vector";
const VAMANA_INDEX: &str = "vamana_idx";
const RQ_INDEX: &str = "rq_idx";
const SQ_INDEX: &str = "sq_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;
const K: usize = 10;

/// Prints Lance's RaBitQ prune tallies and nothing else. Deliberately not a
/// general logger: everything else Lance logs during a pass would land in the
/// same stdout the measurement's own table goes to.
struct PruneStatsLogger;

impl log::Log for PruneStatsLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.target() == "lance_index::vector::bq::prune_stats"
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("{}", record.args());
        }
    }

    fn flush(&self) {}
}

/// Whether Lance was asked for the tallies, spelled the way Lance spells it so
/// that one variable cannot switch on the counting and leave the printing off.
fn prune_stats_asked() -> bool {
    std::env::var("LANCE_RQ_PRUNE_STATS").is_ok_and(|value| {
        !matches!(
            value.to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        )
    })
}

fn parse_list(name: &str, raw: &str) -> Vec<usize> {
    raw.split(',')
        .map(|item| {
            item.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a comma-separated list of numbers"))
        })
        .collect()
}

fn env_list(name: &str, fallback: &str) -> Vec<usize> {
    parse_list(
        name,
        &std::env::var(name).unwrap_or_else(|_| fallback.to_string()),
    )
}

/// An unset or empty variable is an empty list rather than a default one: what
/// this selects is a different sweep, so it has to be possible not to ask for it.
fn env_list_opt(name: &str) -> Vec<usize> {
    match std::env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => parse_list(name, &raw),
        _ => Vec::new(),
    }
}

/// What one arm cost at one candidate width, per query.
#[derive(Clone, Copy, Default)]
struct Cost {
    recall: f64,
    bytes: f64,
    iops: f64,
    requests: f64,
    micros: f64,
    cpu_micros: f64,
    hit_ratio: f64,
    /// Loader runs over the whole pass rather than per query: how many times an
    /// arm read a partition instead of being handed one.
    ///
    /// The column `hit_ratio` cannot answer that. A caller that waits on a load
    /// another caller started is served without running the loader, and the
    /// backend counts it as a hit, so twelve queries stalled on one reload of a
    /// partition report eleven hits and one miss.
    loads: f64,
    /// What the cache held when the pass ended.
    ///
    /// A budget is a target and not a bound: an entry larger than the whole
    /// budget is admitted and reclaimed by later housekeeping, so a cache far
    /// too small to hold one partition still serves that partition for a while.
    held_bytes: f64,
    /// The recall of the answer this arm would have given had it stopped after
    /// its search over codes, before a single original vector was read.
    ///
    /// The question `recall` cannot be asked about: both arms reach a candidate
    /// list by code and then correct it by reading vectors, and correcting it is
    /// nearly the whole byte cost of the query, so what the correction *buys* is
    /// the difference between these two columns.
    coded_recall: f64,
    /// Wall time inside one query's own future, averaged over the pass.
    ///
    /// Not `micros`, and the difference is the whole reason it is here: that one
    /// is the pass divided by its queries, so at `n` in flight it is roughly the
    /// latency divided by `n`. The split below is measured per query and
    /// therefore adds up to *this* column, so both arms have to carry it or the
    /// two halves would be compared against different denominators.
    latency_micros: f64,
    /// The two halves of that latency, split at the moment the candidate list is
    /// settled and before anything is read to correct it.
    ///
    /// Their ratio is the number worth reading: the absolute level under load
    /// includes whatever the query spent waiting to be polled.
    search_micros: f64,
    rescore_micros: f64,
    /// Physical bytes, per query, split the same way. These do add up to
    /// `bytes`, and the vamana arm asserts that they do.
    search_bytes: f64,
    rescore_bytes: f64,
}

impl Cost {
    fn between(&self, other: &Self, fraction: f64) -> Self {
        let mix = |left: f64, right: f64| left + (right - left) * fraction;
        Self {
            recall: mix(self.recall, other.recall),
            bytes: mix(self.bytes, other.bytes),
            iops: mix(self.iops, other.iops),
            requests: mix(self.requests, other.requests),
            micros: mix(self.micros, other.micros),
            cpu_micros: mix(self.cpu_micros, other.cpu_micros),
            hit_ratio: mix(self.hit_ratio, other.hit_ratio),
            loads: mix(self.loads, other.loads),
            held_bytes: mix(self.held_bytes, other.held_bytes),
            coded_recall: mix(self.coded_recall, other.coded_recall),
            latency_micros: mix(self.latency_micros, other.latency_micros),
            search_micros: mix(self.search_micros, other.search_micros),
            rescore_micros: mix(self.rescore_micros, other.rescore_micros),
            search_bytes: mix(self.search_bytes, other.search_bytes),
            rescore_bytes: mix(self.rescore_bytes, other.rescore_bytes),
        }
    }
}

/// The process's own core time so far, summed over its threads.
///
/// The first field of a task's `schedstat` is the nanoseconds it has spent on a
/// cpu, and every thread of the process has one. The difference across a pass is
/// the work the machine did rather than the time the pass took, which is what
/// separates an arm that is waiting from an arm that is busy. A thread that
/// exits inside a pass takes its share with it; the pools here outlive one.
fn cpu_micros() -> f64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0.0;
    };
    let nanos: u64 = tasks
        .flatten()
        .filter_map(|task| std::fs::read_to_string(task.path().join("schedstat")).ok())
        .filter_map(|line| line.split_whitespace().next()?.parse::<u64>().ok())
        .sum();
    nanos as f64 / 1_000.0
}

/// The cost at exactly `target` recall, interpolated between the two widths
/// either side of it rather than read off the first width above it.
///
/// `false` says the narrowest width already cleared the target, so what comes
/// back is an upper bound and the true crossing is off the bottom of the grid.
fn at_recall(points: &[(usize, Cost)], target: f64) -> Option<(Cost, bool)> {
    let first = points.first()?;
    if first.1.recall >= target {
        return Some((first.1, false));
    }
    points
        .windows(2)
        .find_map(|pair| {
            let (below, above) = (&pair[0].1, &pair[1].1);
            (below.recall < target && above.recall >= target).then(|| {
                let span = above.recall - below.recall;
                let fraction = if span > 0.0 {
                    (target - below.recall) / span
                } else {
                    0.0
                };
                below.between(above, fraction)
            })
        })
        .map(|cost| (cost, true))
}

async fn write_dataset(uri: &str, vectors: &FixedSizeListArray) -> Dataset {
    let rows = vectors.len() as u64;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_FIELD, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(vectors.clone()),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams::default()),
    )
    .await
    .unwrap()
}

/// The base-vector position of every row, keyed by the address a search answers
/// in.
async fn positions_by_address(dataset: &Dataset) -> HashMap<u64, u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project(&[ID_COLUMN]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .zip(batch[ID_COLUMN].as_primitive::<UInt64Type>().values())
        .map(|(address, id)| (*address, *id))
        .collect()
}

/// Exact nearest `K` positions of one query, by brute force over every row.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef) -> Vec<u64> {
    let calculator = store.dist_calculator(query, 0.0);
    let mut scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id))
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(K, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(K);
    scored.into_iter().map(|(_, id)| id as u64).collect()
}

fn recall_of(found: &[u64], exact: &[u64]) -> f64 {
    found.iter().filter(|id| exact.contains(id)).count() as f64 / K as f64
}

/// One index and everything an arm is measured against, so an arm's own
/// arguments are only the knobs the sweep turns.
struct Fixture<'a> {
    queries: &'a [Vec<f32>],
    truth: &'a [Vec<u64>],
    /// Base-vector positions keyed by row address, of the dataset this fixture
    /// names - the two datasets are written alike but are not the same index.
    positions: &'a Arc<HashMap<u64, u64>>,
    nprobes: usize,
    cache_bytes: usize,
    warmup: usize,
    /// How many queries are in flight while the pass is timed. One is a latency
    /// measurement and reproduces badly; enough of them that throughput has
    /// stopped growing is a throughput one, and that is the reproducible state.
    concurrency: usize,
    /// Whether the walk holds `__neighbors` across queries instead of fetching
    /// a hop at a time. Reaches the `Flat` arm too and is ignored there, which
    /// is what makes the pass a comparison rather than two.
    resident_edges: bool,
    /// The search width of a Lance index that walks a graph, `None` for one
    /// that does not. It is the same knob this crate calls the queue, and
    /// leaving it unset on `IVF_HNSW_*` would measure Lance's default, not the
    /// point being compared.
    ef: Option<usize>,
}

/// This crate's own arms, both with one pooled budget of exact distances:
/// `Flat` throws the graph away, `Lazy` walks it.
async fn measure_vamana(
    dataset: &Dataset,
    fixture: &Fixture<'_>,
    list_size: usize,
    budget: usize,
    mode: WalkMode,
    beam_width: usize,
) -> Cost {
    let Fixture {
        queries,
        truth,
        positions,
        nprobes,
        cache_bytes,
        warmup,
        concurrency,
        resident_edges,
        // This arm carries its own queue as `list_size`; `ef` is the same knob
        // spelled the way Lance spells it, and only its arms read it.
        ef: _,
    } = *fixture;
    let params = SearchParams::new(K)
        .with_nprobes(nprobes)
        .with_search_list_size(list_size)
        .with_mode(mode)
        .with_beam_width(beam_width)
        .with_resident_edges(resident_edges)
        .with_report_coded(true)
        .with_rescore_budget(budget);
    let index = Arc::new(
        VamanaIndex::open(dataset, VAMANA_INDEX)
            .await
            .unwrap()
            .with_cache(LanceCache::with_capacity(cache_bytes)),
    );
    for query in queries.iter().take(warmup) {
        index.search(query, &params).await.unwrap();
    }

    let before = index.io_stats();
    let cache_before = index.cache_stats().await;
    let cpu_before = cpu_micros();
    let started = Instant::now();
    let totals = futures::stream::iter(queries.iter().zip(truth))
        .map(|(query, exact)| {
            let index = index.clone();
            let params = params.clone();
            let positions = Arc::clone(positions);
            let query = query.clone();
            let exact = exact.clone();
            // A task per query rather than a future per query: `buffered` alone
            // interleaves futures on the one task polling them, and both modes
            // here keep their arithmetic on that task, so bare futures would run
            // the queries one after another and report it as concurrency.
            tokio::spawn(async move {
                let call = Instant::now();
                let result = index.search(&query, &params).await.unwrap();
                let latency = call.elapsed().as_micros() as f64;
                let addresses = |neighbors: &[Neighbor]| {
                    neighbors
                        .iter()
                        .map(|neighbor| positions[&neighbor.row_addr])
                        .collect::<Vec<_>>()
                };
                // Checked rather than trusted: a coded answer that came back
                // empty would be reported as a recall of zero, which reads as a
                // finding rather than as a switch nobody turned on.
                assert_eq!(
                    result.coded_neighbors.len(),
                    K,
                    "the index answered {} coded neighbours rather than k = {K}",
                    result.coded_neighbors.len()
                );
                Reported {
                    recall: recall_of(&addresses(&result.neighbors), &exact),
                    coded_recall: recall_of(&addresses(&result.coded_neighbors), &exact),
                    latency_micros: latency,
                    search_micros: result.search.elapsed.as_micros() as f64,
                    rescore_micros: result.rescore.elapsed.as_micros() as f64,
                    search_bytes: result.search.bytes_read as f64,
                    rescore_bytes: result.rescore.bytes_read as f64,
                }
            })
        })
        .buffered(concurrency)
        .fold(Reported::default(), |totals, reported| async move {
            totals.plus(&reported.unwrap())
        })
        .await;
    let micros = started.elapsed().as_micros() as f64;
    let cpu = cpu_micros() - cpu_before;
    let after = index.io_stats();
    let (hit_ratio, loads, held_bytes) = match (cache_before, index.cache_stats().await) {
        (Some(before), Some(after)) => {
            let hits = after.hits - before.hits;
            let loads = after.misses - before.misses;
            let lookups = hits + loads;
            let ratio = if lookups == 0 {
                0.0
            } else {
                hits as f64 / lookups as f64
            };
            (ratio, loads as f64, after.size_bytes as f64)
        }
        _ => (0.0, 0.0, 0.0),
    };

    // Two independent counts of the same bytes: one off the index's scheduler
    // over the whole pass, one summed from a sink each query attached to the
    // files it opened. They have to agree exactly, and an arm whose split does
    // not add up has a read path the split does not see - which would be
    // invisible in every other column.
    let bytes = (after.bytes_read - before.bytes_read) as f64;
    let split = totals.search_bytes + totals.rescore_bytes;
    assert_eq!(
        bytes, split,
        "the pass read {bytes} bytes but its phases account for {split}: some read is outside \
         both sinks"
    );

    let queries = queries.len() as f64;
    Cost {
        recall: totals.recall / queries,
        bytes: bytes / queries,
        iops: (after.iops - before.iops) as f64 / queries,
        requests: (after.requests - before.requests) as f64 / queries,
        micros: micros / queries,
        cpu_micros: cpu / queries,
        hit_ratio,
        loads,
        held_bytes,
        coded_recall: totals.coded_recall / queries,
        latency_micros: totals.latency_micros / queries,
        search_micros: totals.search_micros / queries,
        rescore_micros: totals.rescore_micros / queries,
        search_bytes: totals.search_bytes / queries,
        rescore_bytes: totals.rescore_bytes / queries,
    }
}

/// What one query of this crate's arm reported, and what a pass sums them into.
///
/// A pass sums rather than averages because the average is one division at the
/// end, and because a per-query struct is what a `fold` over spawned tasks can
/// carry without a lock.
#[derive(Default)]
struct Reported {
    recall: f64,
    coded_recall: f64,
    latency_micros: f64,
    search_micros: f64,
    rescore_micros: f64,
    search_bytes: f64,
    rescore_bytes: f64,
}

impl Reported {
    fn plus(self, other: &Self) -> Self {
        Self {
            recall: self.recall + other.recall,
            coded_recall: self.coded_recall + other.coded_recall,
            latency_micros: self.latency_micros + other.latency_micros,
            search_micros: self.search_micros + other.search_micros,
            rescore_micros: self.rescore_micros + other.rescore_micros,
            search_bytes: self.search_bytes + other.search_bytes,
            rescore_bytes: self.rescore_bytes + other.rescore_bytes,
        }
    }
}

/// What one scan of Lance's plan reported, summed over the measured pass.
#[derive(Default)]
struct Counts {
    bytes: u64,
    iops: u64,
    requests: u64,
    hits: u64,
    misses: u64,
}

fn rq_scanner(
    dataset: &Dataset,
    query: &[f32],
    nprobes: usize,
    ef: Option<usize>,
    refine: Option<u32>,
    callback: Option<ExecutionStatsCallback>,
) -> Scanner {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.empty_project().unwrap();
    scanner.nearest(VECTOR_FIELD, &key, K).unwrap();
    scanner.nprobes(nprobes);
    if let Some(ef) = ef {
        scanner.ef(ef);
    }
    scanner.fast_search();
    if let Some(factor) = refine {
        scanner.refine(factor);
    }
    if let Some(callback) = callback {
        scanner.scan_stats_callback(callback);
    }
    scanner.with_row_id();
    scanner
}

async fn rq_neighbors(
    dataset: &Dataset,
    query: &[f32],
    nprobes: usize,
    ef: Option<usize>,
    refine: Option<u32>,
    callback: Option<ExecutionStatsCallback>,
) -> Vec<u64> {
    let batch = rq_scanner(dataset, query, nprobes, ef, refine, callback)
        .try_into_batch()
        .await
        .unwrap();
    batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec()
}

/// Lance's own arm: `IVF_RQ` with `refine_factor`, through the ordinary scan.
async fn measure_rq(uri: &str, fixture: &Fixture<'_>, refine: Option<u32>) -> Cost {
    let Fixture {
        queries,
        truth,
        positions,
        nprobes,
        ef,
        cache_bytes,
        warmup,
        concurrency,
        ..
    } = *fixture;
    let dataset = DatasetBuilder::from_uri(uri)
        .with_index_cache_size_bytes(cache_bytes)
        .load()
        .await
        .unwrap();
    for query in queries.iter().take(warmup) {
        rq_neighbors(&dataset, query, nprobes, ef, refine, None).await;
    }

    let counts = Arc::new(Mutex::new(Counts::default()));
    let sink = counts.clone();
    let callback: ExecutionStatsCallback = Arc::new(move |summary| {
        let mut counts = sink.lock().unwrap();
        counts.bytes += summary.bytes_read as u64;
        counts.iops += summary.iops as u64;
        counts.requests += summary.requests as u64;
        counts.hits += summary.index_cache_hits() as u64;
        counts.misses += summary.index_cache_misses() as u64;
    });

    let cpu_before = cpu_micros();
    let started = Instant::now();
    let recall = futures::stream::iter(queries.iter().zip(truth))
        .map(|(query, exact)| {
            let dataset = dataset.clone();
            let positions = Arc::clone(positions);
            let callback = callback.clone();
            let query = query.clone();
            let exact = exact.clone();
            tokio::spawn(async move {
                let call = Instant::now();
                let addresses =
                    rq_neighbors(&dataset, &query, nprobes, ef, refine, Some(callback)).await;
                let latency = call.elapsed().as_micros() as f64;
                let found = addresses
                    .iter()
                    .map(|address| positions[address])
                    .collect::<Vec<_>>();
                (recall_of(&found, &exact), latency)
            })
        })
        .buffered(concurrency)
        .fold((0.0f64, 0.0f64), |(recall, latency), joined| async move {
            let (hits, took) = joined.unwrap();
            (recall + hits, latency + took)
        })
        .await;
    let (recall, latency) = recall;
    let micros = started.elapsed().as_micros() as f64;
    let cpu = cpu_micros() - cpu_before;

    let counts = counts.lock().unwrap();
    let lookups = counts.hits + counts.misses;
    let queries = queries.len() as f64;
    Cost {
        recall: recall / queries,
        latency_micros: latency / queries,
        bytes: counts.bytes as f64 / queries,
        iops: counts.iops as f64 / queries,
        requests: counts.requests as f64 / queries,
        micros: micros / queries,
        cpu_micros: cpu / queries,
        hit_ratio: if lookups == 0 {
            0.0
        } else {
            counts.hits as f64 / lookups as f64
        },
        loads: counts.misses as f64,
        // Lance's index cache is the dataset's, reached through the plan's
        // summary rather than held here, and the summary carries no size.
        held_bytes: 0.0,
        // Filled in by `reference_sweep` from the run with `refine_factor`
        // unset: the split of this arm is a difference of two runs, not
        // something one run reports.
        coded_recall: 0.0,
        search_micros: 0.0,
        rescore_micros: 0.0,
        // Set above from the per-query clock, unlike the split.
        search_bytes: 0.0,
        rescore_bytes: 0.0,
    }
}

/// What building Lance's plan costs with nothing executed.
///
/// The `IVF_RQ` arm answers through the ordinary scanner, so its wall time is a
/// whole DataFusion plan built and executed per query, where this crate's is a
/// library call. This is the part that is not the index answering, measured the
/// same way and at the same concurrency so it can be subtracted.
async fn measure_rq_plan(uri: &str, fixture: &Fixture<'_>, refine: Option<u32>) -> (f64, f64) {
    let Fixture {
        queries,
        nprobes,
        ef,
        cache_bytes,
        warmup,
        concurrency,
        ..
    } = *fixture;
    let dataset = DatasetBuilder::from_uri(uri)
        .with_index_cache_size_bytes(cache_bytes)
        .load()
        .await
        .unwrap();
    for query in queries.iter().take(warmup) {
        rq_scanner(&dataset, query, nprobes, ef, refine, None)
            .create_plan()
            .await
            .unwrap();
    }

    let cpu_before = cpu_micros();
    let started = Instant::now();
    futures::stream::iter(queries.iter())
        .map(|query| {
            let dataset = dataset.clone();
            let query = query.clone();
            tokio::spawn(async move {
                rq_scanner(&dataset, &query, nprobes, ef, refine, None)
                    .create_plan()
                    .await
                    .unwrap();
            })
        })
        .buffered(concurrency)
        .fold((), |(), joined| async move { joined.unwrap() })
        .await;
    let queries = queries.len() as f64;
    (
        started.elapsed().as_micros() as f64 / queries,
        (cpu_micros() - cpu_before) / queries,
    )
}

/// The reference arm's width sweep, called once or twice depending on where in
/// the pass the reference is to be measured.
///
/// `label` is what the rows are called, and it decides more than a caption: the
/// summary takes `IVF_RQ refined` as the denominator of its `vs IVF_RQ` column,
/// so the sweep that keeps that name is the one every earlier log compares
/// against. A second sweep under another name is an extra row, not a new
/// baseline.
async fn reference_sweep(
    uri: &str,
    fixture: &Fixture<'_>,
    widths: &[usize],
    label: &str,
) -> Vec<(usize, Cost)> {
    // Lance spends one knob where this crate spends two, so its two phases are
    // two runs rather than two counters: with `refine_factor` unset the scan
    // reads no original vector at all, which makes that run exactly this arm's
    // search over codes. Its recall is the arm's recall before any re-score -
    // the top `k` of a coded list does not depend on how much of the list is
    // kept - and everything the refined run spends above it is the re-score.
    //
    // Measured here rather than at the end of the pass so that the two readings
    // are adjacent: a pass charges its later rows more than its earlier ones,
    // and a difference taken across thirty rows would carry that drift.
    let mut bare = measure_rq(uri, fixture, None).await;
    // For this row the two recalls are the same number by definition: it is the
    // arm that stops after its search over codes.
    bare.coded_recall = bare.recall;
    bare.search_micros = bare.latency_micros;
    assert!(
        bare.recall > 0.0,
        "the unrefined reference answered nothing, so the split would report a recall of zero \
         before the re-score as though that were a measurement"
    );
    report(&format!("{label} coded"), K, &bare);

    let mut points = Vec::with_capacity(widths.len());
    for width in widths {
        let mut cost = measure_rq(uri, fixture, Some((width / K) as u32)).await;
        cost.coded_recall = bare.recall;
        // Latency against latency, never pass time: at twelve queries in flight
        // the pass figure is about a twelfth of the latency, and a split taken
        // from one and compared against a split taken from the other would make
        // this arm look an order of magnitude quicker than it is.
        cost.search_micros = bare.latency_micros;
        cost.search_bytes = bare.bytes;
        // Deliberately unclamped. The bytes cannot come out negative and a
        // negative would be a bug worth seeing; the time can, and when it does
        // the honest reading is that the re-score is under this pass's own
        // noise floor rather than that it cost nothing.
        cost.rescore_micros = cost.latency_micros - bare.latency_micros;
        cost.rescore_bytes = cost.bytes - bare.bytes;
        report(label, *width, &cost);
        points.push((*width, cost));
    }
    points
}

/// One point of a vamana sweep: what the axis column reads, and the pair of
/// knobs it stands for. The two sweeps differ only in which of them moves.
#[derive(Clone, Copy)]
struct Point {
    axis: usize,
    list_size: usize,
    budget: usize,
}

/// One curve: the points along its axis, and what its label is called after the
/// arm name.
struct Curve {
    suffix: String,
    points: Vec<Point>,
}

fn report(label: &str, width: usize, cost: &Cost) {
    println!(
        "{label:<22} {width:>6} {:>8.4} {:>8.4} {:>12.0} {:>11.0} {:>11.0} {:>8.0} {:>9.1} \
         {:>10.0} {:>8.0} {:>10.0} {:>11.0} {:>9.0} {:>6.2} {:>6.0} {:>8}",
        cost.recall,
        cost.coded_recall,
        cost.bytes,
        cost.search_bytes,
        cost.rescore_bytes,
        cost.iops,
        cost.requests,
        cost.micros,
        cost.latency_micros,
        cost.search_micros,
        cost.rescore_micros,
        cost.cpu_micros,
        cost.hit_ratio,
        cost.loads,
        // Lance's arm reaches its cache through the plan's summary, which
        // carries no size, so there is nothing to print rather than a zero.
        if cost.held_bytes == 0.0 {
            "-".to_string()
        } else {
            format!("{:.0}", cost.held_bytes / (1 << 20) as f64)
        }
    );
}

#[tokio::main]
async fn main() {
    if prune_stats_asked() {
        // `set_logger` over `set_boxed_logger`: the boxed form is behind the
        // `log` crate's `std` feature, which nothing in this tree turns on.
        static LOGGER: PruneStatsLogger = PruneStatsLogger;
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    }
    let dir = std::env::var("SIFT_DIR").expect("set SIFT_DIR to the extracted dataset directory");
    let prefix = std::path::Path::new(&dir)
        .file_name()
        .and_then(|name| name.to_str())
        .expect("SIFT_DIR must end in the dataset name")
        .to_string();
    let (base, dim, total) = read_fvecs(&format!("{dir}/{prefix}_base.fvecs"));
    let (query_values, query_dim, total_queries) =
        read_fvecs(&format!("{dir}/{prefix}_query.fvecs"));
    assert_eq!(dim, query_dim);

    let requested = env_usize("VECTORS", 100_000);
    let rows = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let num_queries = env_usize("QUERIES", 200).min(total_queries);
    let rows_per_partition = env_usize("ROWS_PER_PARTITION", 8192);
    let nprobes = env_usize("NPROBES", 7);
    let vamana_rows_per_partition = env_usize("VAMANA_ROWS_PER_PARTITION", rows_per_partition);
    let rq_rows_per_partition = env_usize("RQ_ROWS_PER_PARTITION", rows_per_partition);
    let vamana_partitions = rows.div_ceil(vamana_rows_per_partition).max(1) as u32;
    let rq_partitions = rows.div_ceil(rq_rows_per_partition).max(1) as u32;
    let vamana_nprobes = env_usize("VAMANA_NPROBES", nprobes);
    let rq_nprobes = env_usize("RQ_NPROBES", nprobes);
    let degree = env_usize("DEGREE", 64) as u32;
    let code_bits = env_usize("CODE_BITS", 3) as u8;
    // `CODE_BITS` keeps driving the `IVF_RQ` reference arm whichever kind the
    // walk is given, so that switching this crate's codes does not quietly
    // change the arm every ratio is taken against.
    let vamana_codes = match std::env::var("CODE_KIND").as_deref().unwrap_or("rq") {
        "rq" => CodeSpec::Rabit {
            num_bits: code_bits,
        },
        "sq" => CodeSpec::Scalar {
            num_bits: env_usize("SQ_BITS", 8) as u16,
        },
        other => panic!("CODE_KIND is `rq` or `sq`, got `{other}`"),
    };
    let hnsw_efs = env_list_opt("HNSW_EFS");
    // Lance's flat IVF over eight-bit scalar codes. Deliberately without knobs
    // of its own: it borrows the `IVF_RQ` arm's partitions and probes so that a
    // difference between the two is the quantizer and nothing else, and it
    // takes Lance's shipped quantizer parameters rather than `SQ_BITS`, which
    // names this crate's codes. Lance has no other width to offer anyway -
    // `ScalarQuantizer::transform` scales to `u8` whatever it is told.
    let ivf_sq = env_usize("IVF_SQ", 0) != 0;
    let sq_params = SQBuildParams::default();
    let hnsw_nprobes = env_usize("HNSW_NPROBES", 1);
    let widths = env_list("WIDTHS", "10,20,30,40,60,80,120,160");
    assert!(
        widths.iter().all(|width| width % K == 0),
        "every width must be a multiple of k = {K}: Lance spends `k * refine_factor` where this \
         crate spends `L`, and a width it cannot express would compare two different lists"
    );
    let list_scales = env_list("LIST_SCALES", "1");
    let budgets = env_list_opt("BUDGETS");
    let queues = env_list_opt("QUEUES");
    assert_eq!(
        budgets.is_empty(),
        queues.is_empty(),
        "BUDGETS and QUEUES name one sweep between them: set both or neither"
    );
    assert!(
        budgets.iter().all(|budget| *budget >= K),
        "every budget must be at least k = {K}: a query that re-scores fewer vectors than it \
         returns could never return k neighbours, and the crate refuses it"
    );
    assert!(
        queues
            .iter()
            .all(|queue| budgets.iter().all(|budget| queue >= budget)),
        "every queue must be at least as long as every budget, or the budget cannot be spent and \
         the point measures a shorter list than its label claims"
    );
    let beam_width = env_usize("BEAM_WIDTH", 4);
    let cache_bytes = env_usize("CACHE_MB", 4096) << 20;
    let target = env_usize("TARGET", 95) as f64 / 100.0;
    let warmup = env_usize("WARMUP", num_queries).min(num_queries);
    let concurrency = env_usize("CONCURRENCY", 1).max(1);
    let resident_edges = env_usize("RESIDENT_EDGES", 0) != 0;
    // A pass charges its later rows more than its earlier ones - one and the
    // same reference point cost 1812 us after two vamana rows and 2028 after
    // thirty-two - so `both` reads the reference at each end and brackets the
    // crate's rows between two readings of it.
    let reference_position = std::env::var("REFERENCE_POSITION").unwrap_or_else(|_| "last".into());
    assert!(
        matches!(reference_position.as_str(), "last" | "both"),
        "REFERENCE_POSITION is `last` or `both`, not {reference_position:?}"
    );

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();
    let queries = (0..num_queries)
        .map(|i| query_values[i * dim..(i + 1) * dim].to_vec())
        .collect::<Vec<_>>();

    println!(
        "{prefix} {rows} x {dim}, R = {degree}, walk on {vamana_codes}, IVF_RQ on {code_bits} \
         bits, {num_queries} queries, \
         k = {K}, cache {} MB, {concurrency} in flight, walk edges {}, reference \
         measured {reference_position}",
        cache_bytes >> 20,
        if resident_edges {
            "resident"
        } else {
            "fetched"
        }
    );
    println!(
        "vamana: {vamana_partitions} partitions of about {vamana_rows_per_partition}, \
         {vamana_nprobes} probes | IVF_RQ: {rq_partitions} partitions of about \
         {rq_rows_per_partition}, {rq_nprobes} probes{}",
        match ivf_sq {
            true => format!(" | IVF_SQ: that same shape on {} bits", sq_params.num_bits),
            false => String::new(),
        }
    );

    let store = FlatFloatStorage::new(vectors.clone(), DISTANCE_TYPE);
    let started = Instant::now();
    let truth = queries
        .iter()
        .map(|query| {
            exact_top(
                &store,
                Arc::new(Float32Array::from(query.clone())) as ArrayRef,
            )
        })
        .collect::<Vec<_>>();
    println!(
        "brute force ground truth in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    drop(store);

    // Two datasets rather than two indexes on one, because the scanner picks a
    // vector index by field id alone and would not be given a choice.
    let scratch = std::env::var("DATASET_DIR").ok();
    let temp = scratch.is_none().then(|| tempfile::tempdir().unwrap());
    let home = match (&scratch, &temp) {
        (Some(dir), _) => dir.clone(),
        (None, Some(temp)) => temp.path().to_str().unwrap().to_string(),
        _ => unreachable!(),
    };
    // The suffix names the codes so that two kinds never collide in one
    // directory, and RaBitQ keeps the spelling it has always had so that an
    // index built by an earlier pass is still found rather than rebuilt.
    let code_suffix = match vamana_codes {
        CodeSpec::Rabit { num_bits } => format!("c{num_bits}"),
        CodeSpec::Scalar { num_bits } => format!("sq{num_bits}"),
    };
    let vamana_uri =
        format!("{home}/{prefix}-{rows}-p{vamana_partitions}-r{degree}-{code_suffix}.lance");
    let rq_uri = format!("{home}/{prefix}-{rows}-p{rq_partitions}-rq{code_bits}.lance");
    let sq_uri = format!(
        "{home}/{prefix}-{rows}-p{rq_partitions}-ivfsq{}.lance",
        sq_params.num_bits
    );
    let hnsw_uri = std::env::var("HNSW_URI")
        .unwrap_or_else(|_| format!("{home}/{prefix}-{rows}-p1-hnswsq.lance"));

    let vamana_dataset = if std::fs::metadata(&vamana_uri).is_ok() {
        let dataset = Dataset::open(&vamana_uri).await.unwrap();
        let index = VamanaIndex::open(&dataset, VAMANA_INDEX).await.unwrap();
        let metadata = index.metadata();
        assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
        assert_eq!(metadata.dimension as usize, dim);
        assert_eq!(metadata.max_degree, degree);
        assert_eq!(
            metadata.codes.as_ref().map(|codes| codes.spec()),
            Some(vamana_codes)
        );
        println!("reusing the vamana index at {vamana_uri}");
        dataset
    } else {
        let mut dataset = write_dataset(&vamana_uri, &vectors).await;
        let started = Instant::now();
        create_index(
            &mut dataset,
            VAMANA_INDEX,
            &IndexParams::new(VECTOR_FIELD, vamana_partitions)
                .with_distance_type(DISTANCE_TYPE)
                .with_codes(vamana_codes)
                .with_graph_params(BuildParams {
                    max_degree: degree,
                    ..Default::default()
                }),
        )
        .await
        .unwrap();
        println!(
            "vamana indexed in {:.1}s at {vamana_uri}",
            started.elapsed().as_secs_f64()
        );
        dataset
    };

    if std::fs::metadata(&rq_uri).is_ok() {
        let dataset = Dataset::open(&rq_uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
        println!("reusing the IVF_RQ index at {rq_uri}");
    } else {
        let mut dataset = write_dataset(&rq_uri, &vectors).await;
        let started = Instant::now();
        dataset
            .create_index(
                &[VECTOR_FIELD],
                IndexType::IvfRq,
                Some(RQ_INDEX.to_string()),
                &VectorIndexParams::with_ivf_rq_params(
                    DISTANCE_TYPE,
                    IvfBuildParams::new(rq_partitions as usize),
                    RQBuildParams {
                        num_bits: code_bits,
                        ..Default::default()
                    },
                ),
                false,
            )
            .await
            .unwrap();
        println!(
            "IVF_RQ indexed in {:.1}s at {rq_uri}",
            started.elapsed().as_secs_f64()
        );
    }

    if ivf_sq {
        if std::fs::metadata(&sq_uri).is_ok() {
            let dataset = Dataset::open(&sq_uri).await.unwrap();
            assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
            println!("reusing the IVF_SQ index at {sq_uri}");
        } else {
            let mut dataset = write_dataset(&sq_uri, &vectors).await;
            let started = Instant::now();
            dataset
                .create_index(
                    &[VECTOR_FIELD],
                    IndexType::IvfSq,
                    Some(SQ_INDEX.to_string()),
                    &VectorIndexParams::with_ivf_sq_params(
                        DISTANCE_TYPE,
                        IvfBuildParams::new(rq_partitions as usize),
                        sq_params.clone(),
                    ),
                    false,
                )
                .await
                .unwrap();
            println!(
                "IVF_SQ indexed in {:.1}s at {sq_uri}",
                started.elapsed().as_secs_f64()
            );
        }
    }

    let vamana_positions = Arc::new(positions_by_address(&vamana_dataset).await);
    let sq_positions = match ivf_sq {
        false => Arc::new(HashMap::new()),
        true => Arc::new(positions_by_address(&Dataset::open(&sq_uri).await.unwrap()).await),
    };
    let rq_positions = Arc::new(positions_by_address(&Dataset::open(&rq_uri).await.unwrap()).await);
    let hnsw_positions = match hnsw_efs.is_empty() {
        true => Arc::new(HashMap::new()),
        false => {
            assert!(
                std::fs::metadata(&hnsw_uri).is_ok(),
                "HNSW_EFS asked for the HNSW arm but {hnsw_uri} does not exist: build it with \
                 the hnsw_index example"
            );
            let dataset = Dataset::open(&hnsw_uri).await.unwrap();
            assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
            println!("reusing the HNSW index at {hnsw_uri}");
            Arc::new(positions_by_address(&dataset).await)
        }
    };
    let vamana_fixture = Fixture {
        queries: &queries,
        truth: &truth,
        positions: &vamana_positions,
        nprobes: vamana_nprobes,
        cache_bytes,
        warmup,
        concurrency,
        resident_edges,
        ef: None,
    };
    let rq_fixture = Fixture {
        positions: &rq_positions,
        nprobes: rq_nprobes,
        ..vamana_fixture
    };

    println!(
        "\n{:<22} {:>6} {:>8} {:>8} {:>12} {:>11} {:>11} {:>8} {:>9} {:>10} {:>8} {:>10} \
         {:>11} {:>9} {:>6} {:>6} {:>8}",
        "arm",
        if budgets.is_empty() { "width" } else { "queue" },
        "recall",
        "coded",
        "bytes",
        "search B",
        "rescore B",
        "iops",
        "requests",
        "us (warm)",
        "lat us",
        "search us",
        "rescore us",
        "cpu us",
        "hits",
        "loads",
        "held MB"
    );

    // Which of the two knobs moves along a curve is the whole difference
    // between the two sweeps, so it is decided once, here, rather than inside
    // the loop that measures.
    // Which of this crate's arms the pass measures. Both by default, because
    // the scan is what the walk has to beat and dropping it silently would make
    // every earlier log a different measurement; a campaign that has already
    // settled that question can ask for the walk alone.
    let arms = std::env::var("ARMS")
        .unwrap_or_else(|_| "scan,walk".to_string())
        .split(',')
        .map(|arm| match arm.trim() {
            "scan" => (WalkMode::Flat, "scan"),
            "walk" => (WalkMode::Lazy, "walk"),
            other => panic!("ARMS names `scan` and `walk`, not {other:?}"),
        })
        .collect::<Vec<_>>();
    assert!(!arms.is_empty(), "ARMS must name at least one arm");

    let curves: Vec<Curve> = if budgets.is_empty() {
        list_scales
            .iter()
            .map(|scale| Curve {
                suffix: match scale {
                    1 => String::new(),
                    _ => format!(" L={scale}x"),
                },
                points: widths
                    .iter()
                    .map(|width| Point {
                        axis: *width,
                        list_size: width * scale,
                        budget: *width,
                    })
                    .collect(),
            })
            .collect()
    } else {
        budgets
            .iter()
            .map(|budget| Curve {
                suffix: format!(" b={budget}"),
                points: queues
                    .iter()
                    .map(|queue| Point {
                        axis: *queue,
                        list_size: *queue,
                        budget: *budget,
                    })
                    .collect(),
            })
            .collect()
    };

    let mut sweeps: Vec<(String, Vec<(usize, Cost)>)> = Vec::with_capacity(curves.len() * 2 + 2);
    if reference_position == "both" {
        let points = reference_sweep(&rq_uri, &rq_fixture, &widths, "IVF_RQ early").await;
        sweeps.push(("IVF_RQ early".to_string(), points));
    }
    for curve in &curves {
        for (mode, name) in arms.iter().copied() {
            let label = format!("vamana {name}{}", curve.suffix);
            let mut measured = Vec::with_capacity(curve.points.len());
            for point in &curve.points {
                let cost = measure_vamana(
                    &vamana_dataset,
                    &vamana_fixture,
                    point.list_size,
                    point.budget,
                    mode,
                    beam_width,
                )
                .await;
                report(&label, point.axis, &cost);
                measured.push((point.axis, cost));
            }
            sweeps.push((label, measured));
        }
    }

    for ef in &hnsw_efs {
        let fixture = Fixture {
            positions: &hnsw_positions,
            nprobes: hnsw_nprobes,
            ef: Some(*ef),
            ..vamana_fixture
        };
        let label = format!("HNSW ef={ef}");
        let points = reference_sweep(&hnsw_uri, &fixture, &widths, &label).await;
        sweeps.push((label, points));
    }

    if ivf_sq {
        let fixture = Fixture {
            positions: &sq_positions,
            ..rq_fixture
        };
        let points = reference_sweep(&sq_uri, &fixture, &widths, "IVF_SQ").await;
        sweeps.push(("IVF_SQ".to_string(), points));
    }

    let rq_points = reference_sweep(&rq_uri, &rq_fixture, &widths, "IVF_RQ refined").await;
    sweeps.push(("IVF_RQ refined".to_string(), rq_points));

    let (plan_micros, plan_cpu) = measure_rq_plan(&rq_uri, &rq_fixture, Some(1)).await;
    println!(
        "{:<22} {:>6} {:>8} {:>8} {:>12} {:>11} {:>11} {:>8} {:>9} {:>10.0} {:>8} {:>10} \
         {:>11} {:>9.0}",
        "IVF_RQ plan only",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        plan_micros,
        "-",
        "-",
        "-",
        plan_cpu
    );

    println!("\nat recall {target:.2}, interpolated between the widths either side of it");
    println!(
        "{:<22} {:>8} {:>12} {:>11} {:>11} {:>8} {:>9} {:>10} {:>8} {:>10} {:>11} {:>9} \
         {:>10}",
        "arm",
        "coded",
        "bytes",
        "search B",
        "rescore B",
        "iops",
        "requests",
        "us (warm)",
        "lat us",
        "search us",
        "rescore us",
        "cpu us",
        "vs IVF_RQ"
    );
    let reference = sweeps
        .iter()
        .find(|(label, _)| label == "IVF_RQ refined")
        .and_then(|(_, points)| at_recall(points, target))
        .map(|(cost, _)| cost);
    for (label, points) in &sweeps {
        match at_recall(points, target) {
            None => println!(
                "{label:<22} never reaches {target:.2} on this grid (best {:.4})",
                points
                    .iter()
                    .map(|(_, cost)| cost.recall)
                    .fold(0.0, f64::max)
            ),
            Some((cost, bracketed)) => println!(
                "{label:<22} {:>8.4} {:>12.0} {:>11.0} {:>11.0} {:>8.0} {:>9.1} {:>10.0} \
                 {:>8.0} {:>10.0} {:>11.0} {:>9.0} {:>10}{}",
                cost.coded_recall,
                cost.bytes,
                cost.search_bytes,
                cost.rescore_bytes,
                cost.iops,
                cost.requests,
                cost.micros,
                cost.latency_micros,
                cost.search_micros,
                cost.rescore_micros,
                cost.cpu_micros,
                match reference {
                    Some(reference) => format!("{:.2}x", cost.bytes / reference.bytes),
                    None => "-".to_string(),
                },
                if bracketed {
                    ""
                } else {
                    "  (upper bound: the narrowest width already cleared it)"
                }
            ),
        }
    }
}
