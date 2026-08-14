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
//! (default `20,24,28,32,40,56`), `WIDTHS` (default `1,2,4,8,16`), `TARGET`
//! (default 95, the recall percentage the arms are compared at).
//!
//! Three arms through one index and one binary, which is the only comparison
//! worth making: the same graph, the same routing, the same codes, and a switch.
//!
//! - `exact` reads every partition it probes whole and measures against the
//!   vectors in it.
//! - `coded` reads them whole too and measures against the codes, re-scoring the
//!   candidate list exactly. It is the arm that isolates *steering* from
//!   *reading*: it walks exactly where `lazy` walks at a hop of one.
//! - `lazy` keeps only the row ids and the codes, and fetches the out-edges of
//!   the vertices it expands and the vectors of the candidates it ends up with.
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
        }
    }
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
) -> Cost {
    // A fresh index per point, so the byte count is the queries' and not the
    // queries' plus whatever opening the index read.
    let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
    // Warm the page cache and the k-means centroids alike, so that the first
    // point of a sweep is not charged for what every later one gets free.
    for query in queries.iter().take(8) {
        index.search(query, params).await.unwrap();
    }

    let before = index.io_stats();
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

    let queries = queries.len() as f64;
    Cost {
        recall: recall / queries,
        bytes: (after.bytes_read - before.bytes_read) as f64 / queries,
        iops: (after.iops - before.iops) as f64 / queries,
        requests: (after.requests - before.requests) as f64 / queries,
        micros: micros / queries,
        comparisons: comparisons as f64 / queries,
    }
}

fn report(label: &str, beam: usize, cost: &Cost) {
    println!(
        "{label:<12} {beam:>5} {:>8.4} {:>12.0} {:>8.0} {:>9.1} {:>10.0} {:>10.0}",
        cost.recall, cost.bytes, cost.iops, cost.requests, cost.micros, cost.comparisons
    );
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
    let target = env_usize("TARGET", 95) as f64 / 100.0;

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

    let arms = std::iter::once(("exact".to_string(), WalkMode::Exact, 1))
        .chain(std::iter::once(("coded".to_string(), WalkMode::Coded, 1)))
        .chain(
            widths
                .iter()
                .map(|width| (format!("lazy W={width}"), WalkMode::Lazy, *width)),
        )
        .collect::<Vec<_>>();

    println!(
        "\n{:<12} {:>5} {:>8} {:>12} {:>8} {:>9} {:>10} {:>10}",
        "arm", "beam", "recall", "bytes", "iops", "requests", "us (warm)", "distances"
    );
    let mut sweeps = Vec::with_capacity(arms.len());
    for (label, mode, width) in &arms {
        let mut points = Vec::with_capacity(beams.len());
        for beam in &beams {
            let params = SearchParams::new(K)
                .with_nprobes(nprobes)
                .with_search_list_size(*beam)
                .with_mode(*mode)
                .with_beam_width(*width);
            let cost = measure(&dataset, &queries, &truth, &positions, &params).await;
            report(label, *beam, &cost);
            points.push((*beam, cost));
        }
        sweeps.push((label.clone(), points));
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
    let lazy = sweeps
        .iter()
        .find(|(label, _)| label.starts_with("lazy"))
        .and_then(|(_, points)| at_recall(points, target))
        .map(|(cost, _)| cost);
    let coded = sweeps
        .iter()
        .find(|(label, _)| label == "coded")
        .and_then(|(_, points)| at_recall(points, target))
        .map(|(cost, _)| cost);
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
        // what a cache across queries would take out. It is measured rather than
        // computed from the stride, because the stride is not what a Lance file
        // stores a column in.
        let codes = coded.bytes - exact.bytes;
        println!(
            "  of that, {:.0} B is the code column read whole, so a cache across queries would \
             leave about {:.0} B ({:.3}x of reading whole)",
            codes,
            lazy.bytes - codes,
            (lazy.bytes - codes) / exact.bytes,
        );
    }
}
