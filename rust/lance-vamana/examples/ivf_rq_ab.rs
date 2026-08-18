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
//! (default 7), `DEGREE` (default 64), `CODE_BITS` (default 3), `WIDTHS`
//! (default `10,20,30,40,60,80,120,160`, each a multiple of `k`), `CACHE_MB`
//! (default 4096), `TARGET` (default 95), `WARMUP` (default: every query),
//! `DATASET_DIR` (unset: temporary directories thrown away at the end).
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
//! same row order, into the same number of partitions, at the same `CODE_BITS`,
//! and are queried with the same `k`, the same `NPROBES`, the same query set and
//! the same ground truth. Both are given a warm cache, and each point reopens
//! its index so a cache starts empty and is filled by the warmup rather than by
//! the measured pass.
//!
//! **The one knob, swept.** Each arm carries a candidate list and re-scores it
//! exactly, so the sweep is over its width: `L` here, `k * refine_factor` there.
//! That is the comparison's real subject, because Lance spends one knob where
//! this crate spends two - `refine_factor` widens the candidate list *and* the
//! set of vectors read, while `SearchParams` sets `search_list_size` and
//! `rescore_budget` apart. `LIST_SCALES` is what asks whether the second knob
//! is worth anything: at a scale of `n` each probe keeps `n` times the budget
//! and the same budget is re-scored, which costs the same bytes and can only
//! pay if a probe's own truncation was throwing away a candidate the pooled
//! list wanted. A run at `refine_factor` unset is printed too: it is the
//! default, and it reads no original vectors at all.
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
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::scanner::ExecutionStatsCallback;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::IndexType;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::ivf::IvfBuildParams;
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
const VAMANA_INDEX: &str = "vamana_idx";
const RQ_INDEX: &str = "rq_idx";
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

/// What one arm cost at one candidate width, per query.
#[derive(Clone, Copy, Default)]
struct Cost {
    recall: f64,
    bytes: f64,
    iops: f64,
    requests: f64,
    micros: f64,
    hit_ratio: f64,
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
            hit_ratio: mix(self.hit_ratio, other.hit_ratio),
        }
    }
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
    positions: &'a HashMap<u64, u64>,
    nprobes: usize,
    cache_bytes: usize,
    warmup: usize,
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
    } = *fixture;
    let params = SearchParams::new(K)
        .with_nprobes(nprobes)
        .with_search_list_size(list_size)
        .with_mode(mode)
        .with_beam_width(beam_width)
        .with_rescore_budget(budget);
    let index = VamanaIndex::open(dataset, VAMANA_INDEX)
        .await
        .unwrap()
        .with_cache(LanceCache::with_capacity(cache_bytes));
    for query in queries.iter().take(warmup) {
        index.search(query, &params).await.unwrap();
    }

    let before = index.io_stats();
    let cache_before = index.cache_stats().await;
    let started = Instant::now();
    let mut recall = 0.0;
    for (query, exact) in queries.iter().zip(truth) {
        let result = index.search(query, &params).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| positions[&neighbor.row_addr])
            .collect::<Vec<_>>();
        recall += recall_of(&found, exact);
    }
    let micros = started.elapsed().as_micros() as f64;
    let after = index.io_stats();
    let hit_ratio = match (cache_before, index.cache_stats().await) {
        (Some(before), Some(after)) => {
            let hits = after.hits - before.hits;
            let lookups = hits + (after.misses - before.misses);
            if lookups == 0 {
                0.0
            } else {
                hits as f64 / lookups as f64
            }
        }
        _ => 0.0,
    };

    let queries = queries.len() as f64;
    Cost {
        recall: recall / queries,
        bytes: (after.bytes_read - before.bytes_read) as f64 / queries,
        iops: (after.iops - before.iops) as f64 / queries,
        requests: (after.requests - before.requests) as f64 / queries,
        micros: micros / queries,
        hit_ratio,
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

async fn rq_neighbors(
    dataset: &Dataset,
    query: &[f32],
    nprobes: usize,
    refine: Option<u32>,
    callback: Option<ExecutionStatsCallback>,
) -> Vec<u64> {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.empty_project().unwrap();
    scanner.nearest(VECTOR_FIELD, &key, K).unwrap();
    scanner.nprobes(nprobes);
    scanner.fast_search();
    if let Some(factor) = refine {
        scanner.refine(factor);
    }
    if let Some(callback) = callback {
        scanner.scan_stats_callback(callback);
    }
    scanner.with_row_id();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec()
}

/// Lance's own arm: `IVF_RQ` with `refine_factor`, through the ordinary scan.
async fn measure_rq(uri: &str, fixture: &Fixture<'_>, refine: Option<u32>) -> Cost {
    let Fixture {
        queries,
        truth,
        positions,
        nprobes,
        cache_bytes,
        warmup,
    } = *fixture;
    let dataset = DatasetBuilder::from_uri(uri)
        .with_index_cache_size_bytes(cache_bytes)
        .load()
        .await
        .unwrap();
    for query in queries.iter().take(warmup) {
        rq_neighbors(&dataset, query, nprobes, refine, None).await;
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

    let started = Instant::now();
    let mut recall = 0.0;
    for (query, exact) in queries.iter().zip(truth) {
        let addresses =
            rq_neighbors(&dataset, query, nprobes, refine, Some(callback.clone())).await;
        let found = addresses
            .iter()
            .map(|address| positions[address])
            .collect::<Vec<_>>();
        recall += recall_of(&found, exact);
    }
    let micros = started.elapsed().as_micros() as f64;

    let counts = counts.lock().unwrap();
    let lookups = counts.hits + counts.misses;
    let queries = queries.len() as f64;
    Cost {
        recall: recall / queries,
        bytes: counts.bytes as f64 / queries,
        iops: counts.iops as f64 / queries,
        requests: counts.requests as f64 / queries,
        micros: micros / queries,
        hit_ratio: if lookups == 0 {
            0.0
        } else {
            counts.hits as f64 / lookups as f64
        },
    }
}

fn report(label: &str, width: usize, cost: &Cost) {
    println!(
        "{label:<16} {width:>6} {:>8.4} {:>12.0} {:>8.0} {:>9.1} {:>10.0} {:>7.2}",
        cost.recall, cost.bytes, cost.iops, cost.requests, cost.micros, cost.hit_ratio
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
    let nprobes = env_usize("NPROBES", 7);
    let degree = env_usize("DEGREE", 64) as u32;
    let code_bits = env_usize("CODE_BITS", 3) as u8;
    let widths = env_list("WIDTHS", "10,20,30,40,60,80,120,160");
    assert!(
        widths.iter().all(|width| width % K == 0),
        "every width must be a multiple of k = {K}: Lance spends `k * refine_factor` where this \
         crate spends `L`, and a width it cannot express would compare two different lists"
    );
    let list_scales = env_list("LIST_SCALES", "1");
    let beam_width = env_usize("BEAM_WIDTH", 4);
    let cache_bytes = env_usize("CACHE_MB", 4096) << 20;
    let target = env_usize("TARGET", 95) as f64 / 100.0;
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
        "{prefix} {rows} x {dim}, {partitions} partitions of about {rows_per_partition}, \
         R = {degree}, {code_bits} code bits, {nprobes} probes, {num_queries} queries, k = {K}, \
         cache {} MB",
        cache_bytes >> 20
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
    let vamana_uri = format!("{home}/{prefix}-{rows}-p{partitions}-r{degree}-c{code_bits}.lance");
    let rq_uri = format!("{home}/{prefix}-{rows}-p{partitions}-rq{code_bits}.lance");

    let vamana_dataset = if std::fs::metadata(&vamana_uri).is_ok() {
        let dataset = Dataset::open(&vamana_uri).await.unwrap();
        let index = VamanaIndex::open(&dataset, VAMANA_INDEX).await.unwrap();
        let metadata = index.metadata();
        assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
        assert_eq!(metadata.dimension as usize, dim);
        assert_eq!(metadata.max_degree, degree);
        assert_eq!(
            metadata.codes.as_ref().map(|codes| codes.num_bits),
            Some(code_bits)
        );
        println!("reusing the vamana index at {vamana_uri}");
        dataset
    } else {
        let mut dataset = write_dataset(&vamana_uri, &vectors).await;
        let started = Instant::now();
        create_index(
            &mut dataset,
            VAMANA_INDEX,
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
                    IvfBuildParams::new(partitions as usize),
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

    let vamana_positions = positions_by_address(&vamana_dataset).await;
    let rq_positions = positions_by_address(&Dataset::open(&rq_uri).await.unwrap()).await;
    let vamana_fixture = Fixture {
        queries: &queries,
        truth: &truth,
        positions: &vamana_positions,
        nprobes,
        cache_bytes,
        warmup,
    };
    let rq_fixture = Fixture {
        positions: &rq_positions,
        ..vamana_fixture
    };

    println!(
        "\n{:<16} {:>6} {:>8} {:>12} {:>8} {:>9} {:>10} {:>7}",
        "arm", "width", "recall", "bytes", "iops", "requests", "us (warm)", "hits"
    );

    let mut sweeps: Vec<(String, Vec<(usize, Cost)>)> = Vec::with_capacity(list_scales.len() + 1);
    for scale in &list_scales {
        for (mode, name) in [(WalkMode::Flat, "scan"), (WalkMode::Lazy, "walk")] {
            let label = match scale {
                1 => format!("vamana {name}"),
                _ => format!("vamana {name} L={scale}x"),
            };
            let mut points = Vec::with_capacity(widths.len());
            for width in &widths {
                let cost = measure_vamana(
                    &vamana_dataset,
                    &vamana_fixture,
                    width * scale,
                    *width,
                    mode,
                    beam_width,
                )
                .await;
                report(&label, *width, &cost);
                points.push((*width, cost));
            }
            sweeps.push((label, points));
        }
    }

    let mut rq_points = Vec::with_capacity(widths.len());
    for width in &widths {
        let cost = measure_rq(&rq_uri, &rq_fixture, Some((width / K) as u32)).await;
        report("IVF_RQ refined", *width, &cost);
        rq_points.push((*width, cost));
    }
    sweeps.push(("IVF_RQ refined".to_string(), rq_points));

    let bare = measure_rq(&rq_uri, &rq_fixture, None).await;
    report("IVF_RQ default", K, &bare);

    println!("\nat recall {target:.2}, interpolated between the widths either side of it");
    println!(
        "{:<16} {:>12} {:>8} {:>9} {:>10} {:>10}",
        "arm", "bytes", "iops", "requests", "us (warm)", "vs IVF_RQ"
    );
    let reference = sweeps
        .iter()
        .find(|(label, _)| label == "IVF_RQ refined")
        .and_then(|(_, points)| at_recall(points, target))
        .map(|(cost, _)| cost);
    for (label, points) in &sweeps {
        match at_recall(points, target) {
            None => println!(
                "{label:<16} never reaches {target:.2} on this grid (best {:.4})",
                points
                    .iter()
                    .map(|(_, cost)| cost.recall)
                    .fold(0.0, f64::max)
            ),
            Some((cost, bracketed)) => println!(
                "{label:<16} {:>12.0} {:>8.0} {:>9.1} {:>10.0} {:>9}{}",
                cost.bytes,
                cost.iops,
                cost.requests,
                cost.micros,
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
