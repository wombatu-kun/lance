// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a walk that reads only what it touches actually costs.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --profile release-with-debug --example lazy_walk
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 200), `ROWS_PER_PARTITION` (default 8192), `NPROBES`
//! (default 4), `DEGREE` (default 64), `CODE_BITS` (default 3), `BEAMS`
//! (default `20,24,28,32,40,56`), `WIDTHS` (default `1,2,4,8,16`),
//! `CACHE_WIDTHS` (default `4`), `CACHE_MB` (default 4096), `TARGET`
//! (default 95, the recall percentage the arms are compared at).
//!
//! Six arms through one index and one binary, which is the only comparison
//! worth making: the same graph, the same routing, the same codes, and a switch.
//!
//! - `exact` reads every partition it probes whole and measures against the
//!   vectors in it.
//! - `coded` reads them whole too and measures against the codes, re-scoring the
//!   candidate list exactly. It is the arm that isolates *steering* from
//!   *reading*: it walks exactly where `lazy` walks at a hop of one.
//! - `lazy` keeps only the row ids and the codes, and fetches the out-edges of
//!   the vertices it expands and the vectors of the candidates it ends up with.
//! - `cached` is `lazy` with the row ids and the codes kept across queries
//!   instead of read again by each of them, which is what a process serving
//!   queries would do and what leaves only the walk's own fetches.
//! - `flat` throws the graph away: it scores every vertex of the partition
//!   against its code and keeps the nearest `L`, so it reads what `lazy` reads
//!   minus the edges and measures thirty times as many distances. It is here
//!   because a walk's cost barely moves with the partition's size while a scan's
//!   is linear in it, so the two cross somewhere; the crossing sits well above
//!   the coarser granularity below, since a scanned vertex costs about two
//!   nanoseconds once RaBitQ's error bound is throwing out most of the extra-bit
//!   refinement.
//! - `pooled` is `cached` and `flat cached` with one change: the `L` exact
//!   distances are spent on the whole query rather than on each probe. Every
//!   other arm hands each partition the same `L` and re-scores all of them, so a
//!   query of `p` probes reads `p * L` vectors to answer for `k` - and a vector
//!   is the byte cost of these modes. What it trades is recall at a given `L`,
//!   which is why the comparison below is at equal recall and not equal beam. On
//!   SIFT1M it takes the scan from 52.2 kB a query to 11.3 at 8192 rows a
//!   partition and from 30.9 to 9.6 at 65536, and leaves nearly half the probes
//!   with nothing to fetch.
//!
//! `flat` clears a recall target at a beam the walks need a wider one for, so
//! the interpolation below often reports it at the narrowest beam on the grid.
//! `L` cannot go below `k`, so that point is not an artefact of the grid: it is
//! the cheapest the arm can be asked to be.
//!
//! **Every arm is warmed with the whole query set before it is measured**, so
//! `cached` is measured in the state a server reaches rather than in its first
//! second, and the arms that keep nothing are warmed identically so that nothing
//! but the cache differs. `WARMUP` overrides the count for a quicker sweep.
//!
//! **Compared at equal recall, not at equal beam.** A coded walk needs a wider
//! beam to reach a given recall than an exact one, so a table read across a row
//! flatters it; the crossing is interpolated between the two beams that bracket
//! the target rather than taken at the first beam above it, which on a flat
//! curve is a whole grid step out.
//!
//! **Bytes and iops are the measurement; warm microseconds are not.** The files
//! were written by this process moments earlier, so every read is served from
//! the page cache and dropping it needs root. The time is printed because a
//! scattered read decodes slower than a contiguous one and that is a real cost
//! this machine can see - but the latency of a store that is not the page cache
//! is what the iops column is for.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::query::{SearchParams, VamanaIndex, WalkMode};

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const ID_COLUMN: &str = "id";
const VECTOR_FIELD: &str = "vector";
const INDEX_NAME: &str = "vamana_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;
const K: usize = 10;

fn env_list(name: &str, fallback: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .map(|raw| {
            raw.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a comma-separated list of numbers"))
        })
        .collect()
}

/// What one arm cost at one beam, per query.
#[derive(Clone, Copy, Default)]
struct Cost {
    recall: f64,
    bytes: f64,
    iops: f64,
    requests: f64,
    micros: f64,
    comparisons: f64,
    /// Share of the measured pass's cache lookups that were served, or zero for
    /// an arm holding no cache. Low on an arm that has one says the budget is
    /// smaller than the working set, which is a different measurement from the
    /// one this sweep is taking.
    hit_ratio: f64,
    /// Bytes the cache held at the end of the pass - the resident cost of what
    /// the saving above was bought with, and the number a deployment sizes
    /// against. Per index rather than per query, so unlike every other field
    /// here it does not scale with the query count.
    held: f64,
}

impl Cost {
    /// The point `fraction` of the way from `self` to `other`.
    fn between(&self, other: &Self, fraction: f64) -> Self {
        let mix = |left: f64, right: f64| left + (right - left) * fraction;
        Self {
            recall: mix(self.recall, other.recall),
            bytes: mix(self.bytes, other.bytes),
            iops: mix(self.iops, other.iops),
            requests: mix(self.requests, other.requests),
            micros: mix(self.micros, other.micros),
            comparisons: mix(self.comparisons, other.comparisons),
            hit_ratio: mix(self.hit_ratio, other.hit_ratio),
            held: mix(self.held, other.held),
        }
    }
}

/// One arm of the sweep: what it is called, how it walks, and what it keeps.
struct Arm {
    label: String,
    mode: WalkMode,
    width: usize,
    /// Budget in bytes, or `None` for an arm that reads everything again.
    cache: Option<usize>,
    /// Whether `L` exact distances are the query's budget or each probe's.
    pooled: bool,
}

/// The cost at exactly `target` recall, and whether the grid actually bracketed
/// it.
///
/// Interpolated between the two beams either side of the target rather than read
/// off the first beam above it: on a flat recall curve those are a whole grid
/// step apart, which is a fifteen per cent error in the cost being compared.
///
/// `false` says the narrowest beam already cleared the target, so the true
/// crossing is off the bottom of the grid and what comes back is an upper bound.
/// `None` says nothing on the grid reached it at all. Both are facts about the
/// grid rather than about the arm, and papering over either would compare two
/// arms at recalls that differ.
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

async fn write_dataset(uri: &str, vectors: FixedSizeListArray) -> Dataset {
    let rows = vectors.len() as u64;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_FIELD, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(vectors),
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

/// The base-vector position of every row, keyed by the address the index answers
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

async fn measure(
    dataset: &Dataset,
    queries: &[Vec<f32>],
    truth: &[Vec<u64>],
    positions: &HashMap<u64, u64>,
    params: &SearchParams,
    arm: &Arm,
    warmup: usize,
) -> Cost {
    // A fresh index per point, so the byte count is the queries' and not the
    // queries' plus whatever opening the index read - and so that an arm that
    // caches starts from an empty one.
    let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
    let index = match arm.cache {
        Some(budget) => index.with_cache(LanceCache::with_capacity(budget)),
        None => index,
    };
    // Warm the page cache, the k-means centroids and, where there is one, the
    // cache: the first point of a sweep must not be charged for what every later
    // one gets free, and an arm that keeps things is only interesting once it
    // has them.
    for query in queries.iter().take(warmup) {
        index.search(query, params).await.unwrap();
    }

    let before = index.io_stats();
    let cache_before = index.cache_stats().await;
    let started = Instant::now();
    let mut recall = 0.0;
    let mut comparisons = 0u64;
    for (query, exact) in queries.iter().zip(truth) {
        let result = index.search(query, params).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| positions[&neighbor.row_addr])
            .collect::<Vec<_>>();
        recall += found.iter().filter(|id| exact.contains(id)).count() as f64 / K as f64;
        comparisons += result.comparisons;
    }
    let micros = started.elapsed().as_micros() as f64;
    let after = index.io_stats();
    let (hit_ratio, held) = match (cache_before, index.cache_stats().await) {
        (Some(before), Some(after)) => {
            let hits = after.hits - before.hits;
            let lookups = hits + (after.misses - before.misses);
            let ratio = if lookups == 0 {
                0.0
            } else {
                hits as f64 / lookups as f64
            };
            (ratio, after.size_bytes as f64)
        }
        _ => (0.0, 0.0),
    };

    let queries = queries.len() as f64;
    Cost {
        recall: recall / queries,
        bytes: (after.bytes_read - before.bytes_read) as f64 / queries,
        iops: (after.iops - before.iops) as f64 / queries,
        requests: (after.requests - before.requests) as f64 / queries,
        micros: micros / queries,
        comparisons: comparisons as f64 / queries,
        hit_ratio,
        held,
    }
}

fn report(label: &str, beam: usize, cost: &Cost) {
    println!(
        "{label:<12} {beam:>5} {:>8.4} {:>12.0} {:>8.0} {:>9.1} {:>10.0} {:>10.0} {:>7.2}",
        cost.recall,
        cost.bytes,
        cost.iops,
        cost.requests,
        cost.micros,
        cost.comparisons,
        cost.hit_ratio
    );
}

/// The point one named arm reaches `target` recall at.
fn arm_at(sweeps: &[(String, Vec<(usize, Cost)>)], label: &str, target: f64) -> Option<Cost> {
    sweeps
        .iter()
        .find(|(name, _)| name == label)
        .and_then(|(_, points)| at_recall(points, target))
        .map(|(cost, _)| cost)
}

#[tokio::main]
async fn main() {
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
    let partitions = rows.div_ceil(rows_per_partition).max(1) as u32;
    let nprobes = env_usize("NPROBES", 4);
    let degree = env_usize("DEGREE", 64) as u32;
    let code_bits = env_usize("CODE_BITS", 3) as u8;
    // A beam narrower than `k` is refused by the driver rather than answered
    // short, so it is dropped here with a word rather than taken as a panic ten
    // minutes into a build.
    let beams = env_list("BEAMS", "12,16,20,24,28,40")
        .into_iter()
        .filter(|beam| {
            let wide_enough = *beam >= K;
            if !wide_enough {
                println!("beam {beam} is narrower than k = {K}, skipping it");
            }
            wide_enough
        })
        .collect::<Vec<_>>();
    assert!(!beams.is_empty(), "BEAMS left nothing to sweep");
    let widths = env_list("WIDTHS", "1,2,4,8,16");
    let cache_widths = env_list("CACHE_WIDTHS", "4");
    let cache_bytes = env_usize("CACHE_MB", 4096) << 20;
    let target = env_usize("TARGET", 95) as f64 / 100.0;
    // The whole query set by default: an arm that keeps things is only worth
    // measuring once it has them, and every other arm is warmed the same way so
    // that the cache is the only thing that differs.
    let warmup = env_usize("WARMUP", num_queries).min(num_queries);

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();
    let queries = (0..num_queries)
        .map(|i| query_values[i * dim..(i + 1) * dim].to_vec())
        .collect::<Vec<_>>();

    println!(
        "SIFT {rows} x {dim}, {partitions} partitions of about {rows_per_partition}, R = {degree}, \
         {code_bits} code bits, {nprobes} probes, {num_queries} queries, k = {K}"
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

    let temp = tempfile::tempdir().unwrap();
    let uri = temp.path().to_str().unwrap();
    let mut dataset = write_dataset(uri, vectors).await;
    let started = Instant::now();
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_FIELD, partitions)
            .with_distance_type(DISTANCE_TYPE)
            .with_code_bits(code_bits)
            .with_graph_params(BuildParams {
                max_degree: degree,
                ..Default::default()
            }),
    )
    .await
    .unwrap();
    println!("indexed in {:.1}s", started.elapsed().as_secs_f64());
    let positions = positions_by_address(&dataset).await;

    let mut arms = vec![
        Arm {
            label: "exact".to_string(),
            mode: WalkMode::Exact,
            width: 1,
            cache: None,
            pooled: false,
        },
        Arm {
            label: "coded".to_string(),
            mode: WalkMode::Coded,
            width: 1,
            cache: None,
            pooled: false,
        },
    ];
    arms.extend(widths.iter().map(|width| Arm {
        label: format!("lazy W={width}"),
        mode: WalkMode::Lazy,
        width: *width,
        cache: None,
        pooled: false,
    }));
    arms.extend(cache_widths.iter().map(|width| Arm {
        label: format!("cached W={width}"),
        mode: WalkMode::Lazy,
        width: *width,
        cache: Some(cache_bytes),
        pooled: false,
    }));
    // Both, because the cache is what the comparison against `cached` has to be
    // made at - and the uncached one is what says how much of a scan's read is
    // the codes it would be holding anyway.
    arms.push(Arm {
        label: "flat".to_string(),
        mode: WalkMode::Flat,
        width: 1,
        cache: None,
        pooled: false,
    });
    arms.push(Arm {
        label: "flat cached".to_string(),
        mode: WalkMode::Flat,
        width: 1,
        cache: Some(cache_bytes),
        pooled: false,
    });
    // The same two arms the comparison is usually read across, with the exact
    // distances pooled over the query instead of dealt out per probe.
    arms.push(Arm {
        label: "flat pooled".to_string(),
        mode: WalkMode::Flat,
        width: 1,
        cache: Some(cache_bytes),
        pooled: true,
    });
    if let Some(width) = cache_widths.first() {
        arms.push(Arm {
            label: format!("pooled W={width}"),
            mode: WalkMode::Lazy,
            width: *width,
            cache: Some(cache_bytes),
            pooled: true,
        });
    }

    println!(
        "\n{:<12} {:>5} {:>8} {:>12} {:>8} {:>9} {:>10} {:>10} {:>7}",
        "arm", "beam", "recall", "bytes", "iops", "requests", "us (warm)", "distances", "hits"
    );
    let mut sweeps = Vec::with_capacity(arms.len());
    for arm in &arms {
        let mut points = Vec::with_capacity(beams.len());
        for beam in &beams {
            let mut params = SearchParams::new(K)
                .with_nprobes(nprobes)
                .with_search_list_size(*beam)
                .with_mode(arm.mode)
                .with_beam_width(arm.width);
            if arm.pooled {
                params = params.with_rescore_budget(*beam);
            }
            let cost = measure(&dataset, &queries, &truth, &positions, &params, arm, warmup).await;
            report(&arm.label, *beam, &cost);
            points.push((*beam, cost));
        }
        sweeps.push((arm.label.clone(), points));
    }

    println!("\nat recall {target:.2}, interpolated between the beams either side of it");
    println!(
        "{:<12} {:>12} {:>8} {:>9} {:>10} {:>10} {:>8}",
        "arm", "bytes", "iops", "requests", "us (warm)", "distances", "vs exact"
    );
    let reference = sweeps
        .first()
        .and_then(|(_, points)| at_recall(points, target))
        .map(|(cost, _)| cost);
    for (label, points) in &sweeps {
        match at_recall(points, target) {
            None => println!(
                "{label:<12} never reaches {target:.2} on this grid (best {:.4})",
                points
                    .iter()
                    .map(|(_, cost)| cost.recall)
                    .fold(0.0, f64::max)
            ),
            Some((cost, bracketed)) => println!(
                "{label:<12} {:>12.0} {:>8.0} {:>9.1} {:>10.0} {:>10.0} {:>8}{}",
                cost.bytes,
                cost.iops,
                cost.requests,
                cost.micros,
                cost.comparisons,
                reference
                    .map(|exact| format!("{:.3}x", cost.bytes / exact.bytes))
                    .unwrap_or_else(|| "-".to_string()),
                if bracketed {
                    ""
                } else {
                    "  (upper bound: the narrowest beam already cleared it)"
                },
            ),
        }
    }

    println!("\nwhat it means");
    // The widest arm that both sweeps have, so the lazy and cached arms being
    // compared differ in the cache and in nothing else.
    let width = cache_widths.first().copied().unwrap_or(4);
    let lazy = arm_at(&sweeps, &format!("lazy W={width}"), target);
    let cached = arm_at(&sweeps, &format!("cached W={width}"), target);
    let coded = arm_at(&sweeps, "coded", target);
    if let (Some(exact), Some(coded), Some(lazy)) = (reference, coded, lazy) {
        println!(
            "  a query reads {:.0} B lazily against {:.0} B whole ({:.2}x), and pays {:.0} iops \
             for it against {:.0}",
            lazy.bytes,
            exact.bytes,
            lazy.bytes / exact.bytes,
            lazy.iops,
            exact.iops
        );
        // Reading whole does not depend on the beam, so the difference between
        // the two whole-partition arms is exactly the code column of everything
        // this query probed - which is also what the lazy arm keeps reading and
        // what a cache across queries takes out. It is measured rather than
        // computed from the stride, because the stride is not what a Lance file
        // stores a column in.
        let codes = coded.bytes - exact.bytes;
        println!(
            "  of that, {:.0} B is the code column read whole, so a cache across queries should \
             leave about {:.0} B ({:.3}x of reading whole)",
            codes,
            lazy.bytes - codes,
            (lazy.bytes - codes) / exact.bytes,
        );
        match cached {
            None => println!(
                "  the cached arm never reached the target, so nothing to check it against"
            ),
            Some(cached) => {
                println!(
                    "  measured with a cache: {:.0} B ({:.5}x of reading whole, {:.0}x less than \
                     reading lazily), {:.0} iops against {:.0}, {:.0} us against {:.0}, {:.0}% of \
                     lookups served",
                    cached.bytes,
                    cached.bytes / exact.bytes,
                    lazy.bytes / cached.bytes,
                    cached.iops,
                    lazy.iops,
                    cached.micros,
                    lazy.micros,
                    cached.hit_ratio * 100.0,
                );
                println!(
                    "  the cache holds {:.1} MB to do it, which is what a deployment sizes against",
                    cached.held / (1 << 20) as f64,
                );
            }
        }
    }

    // The question the cache raised and could not answer: whether the graph is
    // earning the reads and the bytes it costs at this granularity. Both arms
    // hold the same codes, probe the same partitions and re-score the same way,
    // so what is left between them is the walk itself.
    if let (Some(cached), Some(flat)) = (cached, arm_at(&sweeps, "flat cached", target)) {
        println!(
            "  a scan of the same partitions with the same cache: {:.0} B against {:.0} ({:.2}x), \
             {:.1} requests against {:.1}, {:.0} us against {:.0}, {:.0} distances against {:.0}",
            flat.bytes,
            cached.bytes,
            flat.bytes / cached.bytes,
            flat.requests,
            cached.requests,
            flat.micros,
            cached.micros,
            flat.comparisons,
            cached.comparisons,
        );
        println!(
            "  so walking costs {:.2}x what scanning does here, and the graph it walks is {} \
             bytes a vertex of index that a scan would not have written",
            cached.micros / flat.micros,
            degree * 4,
        );
    }
}
