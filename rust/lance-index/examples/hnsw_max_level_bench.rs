// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Measures what the HNSW hierarchy buys, by sweeping `max_level` on a texmex
//! dataset (SIFT1M by default). Feeds the checklist of lancedb/lance#8036.
//!
//! `max_level = 1` clamps `random_level` to 0, so every node lands on level 0
//! and the graph is flat. The arms are exactly nested: the level RNG is seeded
//! and drawn sequentially over node ids, and `max_level` enters only through
//! the clamp, so `level_k(i) == min(level_7(i), k - 1)` for every node.
//!
//! PHASE=1 (default) collects deterministic metrics: recall, distance
//! comparisons, level occupancy, level-0 degree distribution, and the
//! `level_offsets` delta. Comparison counts require the counter from
//! lancedb/lance#8142.
//!
//! PHASE=2 collects wall-clock latency, holding all arms of a replicate in
//! memory at once and interleaving with rotated arm order so `max_level` is
//! not confounded with thermal drift.
//!
//! Run: SIFT_DIR=~/datasets/sift cargo run --release -p lance-index --example hnsw_max_level_bench

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::deepsize::DeepSizeOf;
use lance_index::metrics::LocalMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::hnsw::builder::{HNSW, HnswBuildParams, HnswQueryParams};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::distance::DistanceType;
use serde_json::json;

const K: usize = 10;
const ARMS: [u16; 4] = [1, 2, 3, 7];
const EFS: [usize; 6] = [10, 20, 40, 80, 160, 320];
const REPLICATES: usize = 3;
/// Rounds per (ef, replicate) in phase 2. Must equal the arm count so the
/// rotation puts every arm in every position exactly once; with fewer rounds,
/// position and arm stay partly confounded and whichever arm lands first
/// absorbs the cold-cache cost.
const TIMING_ROUNDS: usize = ARMS.len();
/// Untimed queries run against every arm before each timed group, so the
/// first timed block is not the one paying for cache warm-up.
const WARMUP_QUERIES: usize = 2000;

/// (flat values, dim, count)
fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert_eq!(bytes.len() % record, 0);
    let count = bytes.len() / record;
    let mut values = Vec::with_capacity(count * dim);
    for row in 0..count {
        let start = row * record + 4;
        for i in 0..dim {
            let offset = start + i * 4;
            values.push(f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }
    }
    (values, dim, count)
}

fn read_ivecs(path: &str) -> Vec<Vec<u32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert_eq!(bytes.len() % record, 0);
    (0..bytes.len() / record)
        .map(|row| {
            let start = row * record + 4;
            (0..dim)
                .map(|i| {
                    let offset = start + i * 4;
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
                })
                .collect()
        })
        .collect()
}

fn hits(got: &[u32], truth: &[u32]) -> u8 {
    truth.iter().filter(|id| got.contains(id)).count() as u8
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// Level-0 degree distribution, plus the per-decile means that test whether
/// nodes inserted early end up with worse connectivity.
struct DegreeStats {
    histogram: Vec<usize>,
    decile_means: Vec<f64>,
    mean: f64,
    saturated_fraction: f64,
}

/// Reconstructs level block boundaries from `to_batch()` without trusting
/// `level_offsets`: node 0 is forced into every level, so `__vector_id == 0`
/// marks each block start. Returns (boundaries, level-0 degree stats).
fn analyze_batch(hnsw: &HNSW, m: usize) -> (Vec<usize>, DegreeStats) {
    let batch = hnsw.to_batch().unwrap();
    let ids = batch
        .column_by_name("__vector_id")
        .expect("__vector_id column")
        .as_primitive::<arrow_array::types::UInt32Type>();
    let neighbors = batch
        .column_by_name("__neighbors")
        .expect("__neighbors column")
        .as_list::<i32>();

    let boundaries: Vec<usize> = (0..ids.len()).filter(|&i| ids.value(i) == 0).collect();
    let level0_end = boundaries.get(1).copied().unwrap_or(ids.len());

    let max_degree = m * 2;
    let mut histogram = vec![0usize; max_degree + 1];
    let mut degrees = Vec::with_capacity(level0_end);
    for i in 0..level0_end {
        let degree = neighbors.value_length(i) as usize;
        degrees.push(degree);
        if degree < histogram.len() {
            histogram[degree] += 1;
        } else {
            // a degree above m_max0 would be a pruning bug; keep it visible
            histogram.resize(degree + 1, 0);
            histogram[degree] += 1;
        }
    }

    // insertion order is node id order, so deciles over the id axis
    let mut decile_means = Vec::with_capacity(10);
    for d in 0..10 {
        let start = level0_end * d / 10;
        let end = level0_end * (d + 1) / 10;
        let sum: usize = degrees[start..end].iter().sum();
        decile_means.push(sum as f64 / (end - start).max(1) as f64);
    }

    let total: usize = degrees.iter().sum();
    let saturated = degrees.iter().filter(|&&d| d >= max_degree).count();
    let stats = DegreeStats {
        histogram,
        decile_means,
        mean: total as f64 / degrees.len().max(1) as f64,
        saturated_fraction: saturated as f64 / degrees.len().max(1) as f64,
    };
    (boundaries, stats)
}

fn query_params(ef: usize) -> HnswQueryParams {
    HnswQueryParams {
        ef,
        lower_bound: None,
        upper_bound: None,
        dist_q_c: 0.0,
        use_acorn: false,
    }
}

/// Runs every query at one `ef`, returning per-query (hits, comparisons).
fn measure_arm(
    hnsw: &HNSW,
    storage: &FlatFloatStorage,
    queries: &[Arc<dyn Array>],
    ground_truth: &[Vec<u32>],
    ef: usize,
) -> (Vec<u8>, Vec<u32>) {
    let params = query_params(ef);
    let mut all_hits = Vec::with_capacity(queries.len());
    let mut all_comparisons = Vec::with_capacity(queries.len());
    for (qid, query) in queries.iter().enumerate() {
        let metrics = LocalMetricsCollector::default();
        let batch = hnsw
            .search(
                query.clone(),
                K,
                params,
                storage,
                Arc::new(NoFilter),
                &metrics,
            )
            .unwrap();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        // FlatFloatStorage assigns identity row ids, so these are base offsets
        let got: Vec<u32> = row_ids.values().iter().map(|&id| id as u32).collect();
        all_hits.push(hits(&got, &ground_truth[qid][..K]));
        all_comparisons.push(metrics.comparisons.load(Ordering::Relaxed) as u32);
    }
    (all_hits, all_comparisons)
}

/// Guards the assumption the whole measurement rests on: with `NoFilter`,
/// `IvfSubIndex::search` must take the same traversal as `search_basic`. If it
/// diverged, phase 1's recall and phase 2's latency would describe different
/// algorithms, and the comparison counts would not describe either.
fn cross_check(hnsw: &HNSW, storage: &FlatFloatStorage, queries: &[Arc<dyn Array>], ef: usize) {
    let params = query_params(ef);
    for (qid, query) in queries.iter().enumerate().take(16) {
        let metrics = LocalMetricsCollector::default();
        let batch = hnsw
            .search(
                query.clone(),
                K,
                params,
                storage,
                Arc::new(NoFilter),
                &metrics,
            )
            .unwrap();
        let via_search: Vec<u32> = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .map(|&id| id as u32)
            .collect();
        let via_basic: Vec<u32> = hnsw
            .search_basic(query.clone(), K, &params, None, storage)
            .unwrap()
            .iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(
            via_search, via_basic,
            "query {qid}: NoFilter path diverged from search_basic"
        );
        let comparisons = metrics.comparisons.load(Ordering::Relaxed);
        assert!(
            comparisons > 0,
            "query {qid}: comparison counter stayed zero; is this the #8142 branch?"
        );
    }
    println!("  cross-check: search == search_basic on 16 queries, counter live");
}

fn build(storage: &FlatFloatStorage, arm: u16) -> (HNSW, f64) {
    let params = HnswBuildParams::default().max_level(arm);
    let start = Instant::now();
    let hnsw = HNSW::index_vectors(storage, params).unwrap();
    (hnsw, start.elapsed().as_secs_f64())
}

fn phase1(
    storage: &FlatFloatStorage,
    queries: &[Arc<dyn Array>],
    ground_truth: &[Vec<u32>],
    total: usize,
) -> serde_json::Value {
    let defaults = HnswBuildParams::default();
    let mut runs = Vec::new();

    for replicate in 0..REPLICATES {
        for &arm in ARMS.iter() {
            println!("phase1: replicate {replicate}, max_level {arm}: building...");
            let (hnsw, build_secs) = build(storage, arm);
            let graph_bytes = hnsw.deep_size_of();
            let occupancy: Vec<usize> = (0..arm)
                .map(|level| hnsw.num_nodes(level as usize))
                .collect();
            assert_eq!(
                occupancy[0], total,
                "level 0 must hold every node (got {}, expected {total})",
                occupancy[0]
            );
            println!(
                "  built in {build_secs:.1}s, {:.2} GB, occupancy {occupancy:?}",
                graph_bytes as f64 / 1e9
            );
            if replicate == 0 && arm == ARMS[0] {
                cross_check(&hnsw, storage, queries, EFS[EFS.len() / 2]);
            }

            // the graph batch is large, so only replicate 0 pays for it
            let mut structural = json!(null);
            if replicate == 0 {
                let level_offsets = hnsw.metadata().level_offsets.clone();
                let (boundaries, degrees) = analyze_batch(&hnsw, defaults.m);
                assert_eq!(
                    boundaries.len(),
                    arm as usize,
                    "expected one block start per configured level"
                );
                println!(
                    "  level_offsets {level_offsets:?} vs reconstructed {boundaries:?}, \
                     mean degree {:.2}, saturated {:.1}%",
                    degrees.mean,
                    degrees.saturated_fraction * 100.0
                );
                structural = json!({
                    "level_offsets": level_offsets,
                    "reconstructed_boundaries": boundaries,
                    "degree_histogram": degrees.histogram,
                    "degree_decile_means": degrees.decile_means,
                    "degree_mean": degrees.mean,
                    "degree_saturated_fraction": degrees.saturated_fraction,
                });
            }

            let mut ef_results = Vec::with_capacity(EFS.len());
            for &ef in EFS.iter() {
                let (hits, comparisons) = measure_arm(&hnsw, storage, queries, ground_truth, ef);
                let recall = hits.iter().map(|&h| h as f64).sum::<f64>() / (hits.len() * K) as f64;
                let mean_comparisons =
                    comparisons.iter().map(|&c| c as f64).sum::<f64>() / comparisons.len() as f64;
                println!("  ef {ef:>3}: recall@{K} {recall:.4}, comparisons {mean_comparisons:.0}");
                ef_results.push(json!({
                    "ef": ef,
                    "recall": recall,
                    "mean_comparisons": mean_comparisons,
                    "hits": hits,
                    "comparisons": comparisons,
                }));
            }

            runs.push(json!({
                "arm": arm,
                "replicate": replicate,
                "build_secs": build_secs,
                "graph_bytes": graph_bytes,
                "occupancy": occupancy,
                "structural": structural,
                "ef_results": ef_results,
            }));
        }
    }
    json!({ "runs": runs })
}

fn phase2(
    storage: &FlatFloatStorage,
    queries: &[Arc<dyn Array>],
    ground_truth: &[Vec<u32>],
) -> serde_json::Value {
    let mut runs = Vec::new();
    for replicate in 0..REPLICATES {
        println!(
            "phase2: replicate {replicate}: building all {} arms...",
            ARMS.len()
        );
        let graphs: Vec<(u16, HNSW)> = ARMS
            .iter()
            .map(|&arm| {
                let (hnsw, secs) = build(storage, arm);
                println!("  max_level {arm} built in {secs:.1}s");
                (arm, hnsw)
            })
            .collect();

        for &ef in EFS.iter() {
            let params = query_params(ef);
            for (_, hnsw) in graphs.iter() {
                for query in queries.iter().take(WARMUP_QUERIES) {
                    hnsw.search_basic(query.clone(), K, &params, None, storage)
                        .unwrap();
                }
            }
            for round in 0..TIMING_ROUNDS {
                let load_before = read_loadavg();
                // rotate arm order between rounds so a drifting clock cannot
                // systematically favour whichever arm always runs first
                for offset in 0..graphs.len() {
                    let (arm, hnsw) = &graphs[(offset + round) % graphs.len()];
                    let mut latencies = Vec::with_capacity(queries.len());
                    let mut hit_counts = Vec::with_capacity(queries.len());
                    for (qid, query) in queries.iter().enumerate() {
                        let start = Instant::now();
                        let nodes = hnsw
                            .search_basic(query.clone(), K, &params, None, storage)
                            .unwrap();
                        latencies.push(start.elapsed().as_secs_f64() * 1e6);
                        let got: Vec<u32> = nodes.iter().map(|n| n.id).collect();
                        hit_counts.push(hits(&got, &ground_truth[qid][..K]));
                    }
                    let mut sorted = latencies.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = percentile(&sorted, 0.50);
                    let p99 = percentile(&sorted, 0.99);
                    let recall = hit_counts.iter().map(|&h| h as f64).sum::<f64>()
                        / (hit_counts.len() * K) as f64;
                    let load_after = read_loadavg();
                    println!(
                        "  ef {ef:>3} round {round} max_level {arm}: p50 {p50:.0}us \
                         p99 {p99:.0}us recall {recall:.4} load {load_before:.2}->{load_after:.2}"
                    );
                    runs.push(json!({
                        "arm": arm,
                        "replicate": replicate,
                        "ef": ef,
                        "round": round,
                        "position": offset,
                        "p50_us": p50,
                        "p99_us": p99,
                        "recall": recall,
                        "latencies_us": latencies,
                        "loadavg_before": load_before,
                        "loadavg_after": load_after,
                    }));
                }
            }
        }
    }
    json!({ "runs": runs })
}

fn read_loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn main() {
    let sift_dir = std::env::var("SIFT_DIR").expect("set SIFT_DIR to the extracted dataset dir");
    let phase: usize = std::env::var("PHASE")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .expect("PHASE must be 1 or 2");
    let out_path =
        std::env::var("OUT").unwrap_or_else(|_| format!("hnsw_max_level_phase{phase}.json"));

    let prefix = std::path::Path::new(&sift_dir)
        .file_name()
        .and_then(|name| name.to_str())
        .expect("SIFT_DIR must end in the dataset name, e.g. sift or siftsmall")
        .to_string();
    println!("loading {prefix} from {sift_dir}...");
    let (base, dim, total) = read_fvecs(&format!("{sift_dir}/{prefix}_base.fvecs"));
    let (query_values, query_dim, num_queries) =
        read_fvecs(&format!("{sift_dir}/{prefix}_query.fvecs"));
    let ground_truth = read_ivecs(&format!("{sift_dir}/{prefix}_groundtruth.ivecs"));
    assert_eq!(dim, query_dim);
    assert_eq!(ground_truth.len(), num_queries);
    println!("base {total} x {dim}, {num_queries} queries, phase {phase}");

    let fsl =
        FixedSizeListArray::try_new_from_values(Float32Array::from(base), dim as i32).unwrap();
    let storage = FlatFloatStorage::new(fsl, DistanceType::L2);
    let queries: Vec<Arc<dyn Array>> = (0..num_queries)
        .map(|i| {
            Arc::new(Float32Array::from(
                query_values[i * dim..(i + 1) * dim].to_vec(),
            )) as Arc<dyn Array>
        })
        .collect();

    let defaults = HnswBuildParams::default();
    let wall = Instant::now();
    let results = match phase {
        1 => phase1(&storage, &queries, &ground_truth, total),
        2 => phase2(&storage, &queries, &ground_truth),
        other => panic!("PHASE must be 1 or 2, got {other}"),
    };

    let output = json!({
        "phase": phase,
        "dataset": prefix,
        "base_count": total,
        "dim": dim,
        "num_queries": num_queries,
        "k": K,
        "arms": ARMS,
        "efs": EFS,
        "replicates": REPLICATES,
        "build_params": {
            "m": defaults.m,
            "ef_construction": defaults.ef_construction,
            "prefetch_distance": defaults.prefetch_distance,
            "default_max_level": defaults.max_level,
        },
        "wall_secs": wall.elapsed().as_secs_f64(),
        "results": results,
    });
    std::fs::write(&out_path, serde_json::to_string(&output).unwrap()).unwrap();
    println!("\nwrote {out_path} in {:.1}s", wall.elapsed().as_secs_f64());
}
