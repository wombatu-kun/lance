// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Step 0 gate for lancedb/lance#5156: does slicing the serialized HNSW batch
//! by `level_offsets` actually lose nodes, or does the `__vector_id` keying in
//! `LevelLookup::Sparse` fully compensate?
//!
//! `level_offsets` is the running sum of `level_count`, which credits the
//! entry point only at level 0 while `to_batch` writes it at every level. This
//! compares, per level, the slice `load` takes against the block `to_batch`
//! actually emitted, then round-trips the graph and compares search results.
//!
//! Run: SIFT_DIR=~/datasets/sift cargo run --release -p lance-index --example hnsw_level_offsets_check

#![allow(clippy::print_stdout)]

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::metrics::LocalMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::hnsw::builder::{HNSW, HnswBuildParams, HnswQueryParams};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::distance::DistanceType;

const K: usize = 10;
const QUERIES: usize = 10000;
/// The hierarchy earns the most at low `ef`, so a divergence caused by broken
/// upper levels should surface here first.
const EFS: [usize; 5] = [10, 16, 32, 64, 128];

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

fn query_params(ef: usize) -> HnswQueryParams {
    HnswQueryParams {
        ef,
        lower_bound: None,
        upper_bound: None,
        dist_q_c: 0.0,
        use_acorn: false,
    }
}

/// Top-k ids and the distance comparisons the traversal used.
fn search(
    hnsw: &HNSW,
    storage: &FlatFloatStorage,
    query: Arc<dyn Array>,
    ef: usize,
) -> (Vec<u32>, usize) {
    let metrics = LocalMetricsCollector::default();
    let batch = hnsw
        .search(
            query,
            K,
            query_params(ef),
            storage,
            Arc::new(NoFilter),
            &metrics,
        )
        .unwrap();
    let ids = batch
        .column_by_name(ROW_ID)
        .expect("row id column")
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .map(|&id| id as u32)
        .collect();
    (ids, metrics.comparisons.load(Ordering::Relaxed))
}

fn main() {
    let sift_dir = std::env::var("SIFT_DIR").expect("set SIFT_DIR to the extracted dataset dir");
    let max_level: u16 = std::env::var("MAX_LEVEL")
        .unwrap_or_else(|_| "7".to_string())
        .parse()
        .unwrap();
    let prefix = std::path::Path::new(&sift_dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap()
        .to_string();

    let (base, dim, total) = read_fvecs(&format!("{sift_dir}/{prefix}_base.fvecs"));
    let (queries, _, num_queries) = read_fvecs(&format!("{sift_dir}/{prefix}_query.fvecs"));
    let ground_truth = read_ivecs(&format!("{sift_dir}/{prefix}_groundtruth.ivecs"));
    println!("{prefix}: {total} x {dim}, max_level {max_level}");

    let fsl =
        FixedSizeListArray::try_new_from_values(Float32Array::from(base), dim as i32).unwrap();
    let storage = FlatFloatStorage::new(fsl, DistanceType::L2);
    let hnsw =
        HNSW::index_vectors(&storage, HnswBuildParams::default().max_level(max_level)).unwrap();

    let batch = hnsw.to_batch().unwrap();
    let ids = batch
        .column_by_name("__vector_id")
        .unwrap()
        .as_primitive::<UInt32Type>();
    let level_offsets = hnsw.metadata().level_offsets;

    // node 0 is written into every level, so `__vector_id == 0` starts a block
    let block_starts: Vec<usize> = (0..ids.len()).filter(|&i| ids.value(i) == 0).collect();
    let mut block_ends: Vec<usize> = block_starts[1..].to_vec();
    block_ends.push(ids.len());

    println!(
        "\nto_batch rows {}, level_offsets last {}, surplus {}",
        ids.len(),
        level_offsets.last().unwrap(),
        ids.len() - level_offsets.last().unwrap()
    );
    println!(
        "level_offsets ({} entries): {level_offsets:?}",
        level_offsets.len()
    );
    println!(
        "true block starts ({} entries): {block_starts:?}",
        block_starts.len()
    );

    println!("\nlvl | slice          | true block     | lost | foreign | verdict");
    let mut any_loss = false;
    for level in 0..block_starts.len() {
        // exactly what `load` does: tuple_windows over level_offsets
        let (s, e) = (level_offsets[level], level_offsets[level + 1]);
        let (ts, te) = (block_starts[level], block_ends[level]);

        let sliced: HashSet<u32> = (s..e.min(ids.len())).map(|i| ids.value(i)).collect();
        let truth: HashSet<u32> = (ts..te).map(|i| ids.value(i)).collect();
        let lost = truth.difference(&sliced).count();
        let foreign = sliced.difference(&truth).count();
        any_loss |= lost > 0 || foreign > 0;
        let verdict = if lost == 0 && foreign == 0 {
            "exact".to_string()
        } else if sliced.is_disjoint(&truth) {
            "NO OVERLAP".to_string()
        } else {
            format!("{} of {} true nodes kept", truth.len() - lost, truth.len())
        };
        println!("{level:3} | [{s},{e}) | [{ts},{te}) | {lost:4} | {foreign:7} | {verdict}");
    }

    println!(
        "\nslicing verdict: {}",
        if any_loss {
            "LOSSY - the __vector_id keying does NOT fully compensate"
        } else {
            "clean - keying compensates, metadata hygiene only"
        }
    );

    // behavioral check: does it change what search returns?
    let loaded = HNSW::load(batch).unwrap();
    println!(
        "\nloaded max_level {} (fresh {})",
        loaded.max_level(),
        hnsw.max_level()
    );

    let n = QUERIES.min(num_queries);
    println!("\nfresh vs loaded over {n} queries:");
    println!(
        " ef | exact top-{K} | recall fresh | recall loaded | delta | comparisons fresh -> loaded"
    );
    for &ef in EFS.iter() {
        let (mut exact, mut fresh_hits, mut loaded_hits) = (0usize, 0usize, 0usize);
        let (mut fresh_cmp, mut loaded_cmp) = (0usize, 0usize);
        for i in 0..n {
            let q = || -> Arc<dyn Array> {
                Arc::new(Float32Array::from(queries[i * dim..(i + 1) * dim].to_vec()))
            };
            let (a, ca) = search(&hnsw, &storage, q(), ef);
            let (b, cb) = search(&loaded, &storage, q(), ef);
            exact += usize::from(a == b);
            let truth: HashSet<u32> = ground_truth[i][..K].iter().copied().collect();
            fresh_hits += a.iter().filter(|id| truth.contains(id)).count();
            loaded_hits += b.iter().filter(|id| truth.contains(id)).count();
            fresh_cmp += ca;
            loaded_cmp += cb;
        }
        let (rf, rl) = (
            fresh_hits as f64 / (n * K) as f64,
            loaded_hits as f64 / (n * K) as f64,
        );
        println!(
            "{ef:3} | {:>11.2}% | {rf:>12.5} | {rl:>13.5} | {:+.5} | {:.1} -> {:.1} ({:+.2}%)",
            exact as f64 * 100.0 / n as f64,
            rl - rf,
            fresh_cmp as f64 / n as f64,
            loaded_cmp as f64 / n as f64,
            (loaded_cmp as f64 / fresh_cmp as f64 - 1.0) * 100.0
        );
    }
}
