// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vamana against Lance's own HNSW on SIFT, at matched graph memory.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --release --example sift_recall
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 200), `DEGREE` (Vamana `R`, default 32), `SEARCH_LIST`
//! (Vamana `L`, default 100), `SEED` (default 42), `HNSW_EDGES` (default
//! `DEGREE / 2`) and `HNSW_EF_CONSTRUCTION` (default 150).
//!
//! `HNSW_EDGES` is what sets the memory the two indexes are compared at, and its
//! default is not that value: `m` has to be raised until the printed ratio of
//! HNSW edges to Vamana slots is near 1, and what that takes changes whenever
//! upstream changes how many edges an `m` buys.
//!
//! Both indexes are *queried* through the same counting wrapper, and an
//! assertion below pins that the wrapper and this crate's own counter agree, so
//! the per-query numbers - the ones the comparison rests on - are comparable by
//! construction rather than by trusting two sets of instrumentation to mean the
//! same thing.
//!
//! The build numbers are not: Vamana's come from its own `Comparisons` and
//! HNSW's from the wrapper, because Lance's builder takes a metrics argument and
//! ignores it. They count the same event and are printed side by side, but
//! nothing here proves they count it the same way.
//!
//! The comparison to read is **distances per query at equal recall**, never at
//! equal beam width: the two beams are not the same parameter and matching them
//! measures nothing.

use std::any::Any;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::SchemaRef;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::Result;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::hnsw::builder::{HnswBuildParams, HnswQueryParams};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::{BuildParams, build_partition};
use lance_vamana::search::{Comparisons, SearchScratch, greedy_search};

const K: usize = 10;
const DISTANCE_TYPE: DistanceType = DistanceType::L2;

/// A [`VectorStore`] that counts every distance its calculators compute.
///
/// Lance's HNSW builder takes a metrics argument and ignores it, so the only
/// way to hold both indexes to the same measure is to count underneath them.
#[derive(Debug, Clone)]
struct Counting<S: VectorStore> {
    inner: S,
    distances: Arc<AtomicU64>,
}

impl<S: VectorStore> Counting<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            distances: Arc::new(AtomicU64::new(0)),
        }
    }

    fn take(&self) -> u64 {
        self.distances.swap(0, Ordering::Relaxed)
    }
}

struct CountingCalculator<'a, S: VectorStore + 'a> {
    inner: S::DistanceCalculator<'a>,
    distances: Arc<AtomicU64>,
}

impl<S: VectorStore> DistCalculator for CountingCalculator<'_, S> {
    fn distance(&self, id: u32) -> f32 {
        self.distances.fetch_add(1, Ordering::Relaxed);
        self.inner.distance(id)
    }

    fn distance_all(&self, k_hint: usize) -> Vec<f32> {
        let all = self.inner.distance_all(k_hint);
        self.distances
            .fetch_add(all.len() as u64, Ordering::Relaxed);
        all
    }

    fn prefetch(&self, id: u32) {
        self.inner.prefetch(id)
    }
}

impl<S: VectorStore + 'static> VectorStore for Counting<S> {
    type DistanceCalculator<'a> = CountingCalculator<'a, S>;

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        self.inner.schema()
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send> {
        self.inner.to_batches()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn distance_type(&self) -> DistanceType {
        self.inner.distance_type()
    }

    fn row_id(&self, id: u32) -> u64 {
        self.inner.row_id(id)
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.inner.row_ids()
    }

    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self> {
        Ok(Self {
            inner: self.inner.append_batch(batch, vector_column)?,
            distances: self.distances.clone(),
        })
    }

    fn dist_calculator(&self, query: ArrayRef, dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        CountingCalculator {
            inner: self.inner.dist_calculator(query, dist_q_c),
            distances: self.distances.clone(),
        }
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        CountingCalculator {
            inner: self.inner.dist_calculator_from_id(id),
            distances: self.distances.clone(),
        }
    }
}

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
                    u32::from_le_bytes(bytes[start + i * 4..start + i * 4 + 4].try_into().unwrap())
                })
                .collect()
        })
        .collect()
}

fn storage(values: Vec<f32>, dim: usize) -> FlatFloatStorage {
    FlatFloatStorage::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32).unwrap(),
        DISTANCE_TYPE,
    )
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|raw| {
            raw.parse()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(fallback)
}

/// Exact top-`K` by brute force, computed rather than trusted.
///
/// The shipped ground truth answers the full base set; any prefix of it needs
/// its own. Computing both and comparing when the whole set is used is also the
/// only check that this harness measures distance the same way the dataset's
/// authors did.
fn ground_truth(store: &FlatFloatStorage, queries: &[ArrayRef]) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let calculator = store.dist_calculator(query.clone(), 0.0);
            let mut all = calculator
                .distance_all(K)
                .into_iter()
                .enumerate()
                .map(|(id, distance)| (distance, id as u32))
                .collect::<Vec<_>>();
            all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            all.iter().take(K).map(|(_, id)| *id).collect()
        })
        .collect()
}

fn recall(found: &[u32], exact: &[u32]) -> f64 {
    let exact = exact.iter().collect::<HashSet<_>>();
    found.iter().filter(|id| exact.contains(id)).count() as f64 / K as f64
}

/// Distances per query needed to reach `target` recall, read off the measured
/// curve by linear interpolation between the two beams that bracket it.
///
/// Equal recall is the only honest place to compare: `L` and `ef` are different
/// parameters, so a table read straight down at equal beam compares two points
/// that are not the same operating point. Interpolation on a convex curve
/// overestimates, and it overestimates more where the bracketing points are
/// further apart, which is why the beam grid is dense rather than doubling.
fn cost_at_recall(curve: &[(f64, f64)], target: f64) -> Option<f64> {
    let above = curve.iter().position(|(recall, _)| *recall >= target)?;
    if above == 0 {
        return None;
    }
    let (low_recall, low_cost) = curve[above - 1];
    let (high_recall, high_cost) = curve[above];
    let span = high_recall - low_recall;
    if span <= 0.0 {
        return Some(low_cost);
    }
    Some(low_cost + (target - low_recall) / span * (high_cost - low_cost))
}

/// Edges across every level of a built HNSW.
///
/// Counted rather than taken from `deep_size_of`, which on a freshly built
/// index also weighs the builder's own scaffolding - a second, ranked copy of
/// every neighbour list that never reaches disk. That reads as roughly four
/// times the graph and would make any memory-matched comparison a fiction.
fn hnsw_edges(hnsw: &HNSW) -> usize {
    let batch = hnsw.to_batch().unwrap();
    let neighbors = batch
        .column_by_name("__neighbors")
        .expect("HNSW batch has a neighbours column")
        .as_list::<i32>();
    neighbors.values().len()
}

fn main() {
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
    let vectors = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let num_queries = env_usize("QUERIES", 200).min(total_queries);
    let degree = env_usize("DEGREE", 32);

    let indexed = storage(base[..vectors * dim].to_vec(), dim);
    let queries = (0..num_queries)
        .map(|i| {
            Arc::new(Float32Array::from(
                query_values[i * dim..(i + 1) * dim].to_vec(),
            )) as ArrayRef
        })
        .collect::<Vec<_>>();

    println!("SIFT {vectors} x {dim}, {num_queries} queries, k = {K}");

    let started = Instant::now();
    let exact = ground_truth(&indexed, &queries);
    println!(
        "brute force ground truth in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    if vectors == total {
        let official = read_ivecs(&format!("{dir}/{prefix}_groundtruth.ivecs"));
        let agreement = exact
            .iter()
            .zip(&official)
            .map(|(ours, theirs)| recall(ours, &theirs[..K]))
            .sum::<f64>()
            / exact.len() as f64;
        println!("agreement with the shipped ground truth: {agreement:.4}");
        assert!(
            agreement > 0.999,
            "our distances disagree with the dataset's"
        );
    }

    let counting = Counting::new(indexed.clone());

    // Vamana. `R` fixes the memory: every vertex owns exactly `R` slots on disk.
    let params = BuildParams {
        max_degree: degree as u32,
        search_list_size: env_usize("SEARCH_LIST", 100),
        alpha: 1.2,
        // Exposed so that a build number that has moved can be told apart from
        // a build number that was drawn differently: the insertion order of both
        // passes comes off this seed, and its spread is the noise floor any
        // before-and-after comparison of a build has to clear.
        seed: env_usize("SEED", 42) as u64,
    };
    let building = Comparisons::default();
    let started = Instant::now();
    let built = build_partition(&indexed, &params, &building).unwrap();
    let vamana_build = started.elapsed().as_secs_f64();
    // Slots, not edges: the fixed stride is the point, so a partly filled
    // neighbour list still costs its full width on disk. Counting it any other
    // way would flatter us against an index that pays only for what it uses.
    let vamana_slots = vectors * degree;
    let used = (0..vectors as u32)
        .map(|vertex| built.graph.neighbors(vertex).unwrap().len())
        .sum::<usize>();
    println!(
        "\nvamana   R={degree} L={} alpha={}: built in {vamana_build:.1}s, \
         {:.0}M distances, {vamana_slots} slots ({:.1} MiB, {:.0}% filled)",
        params.search_list_size,
        params.alpha,
        building.get() as f64 / 1e6,
        (vamana_slots * 4) as f64 / (1024.0 * 1024.0),
        100.0 * used as f64 / vamana_slots as f64,
    );

    let hnsw_params = HnswBuildParams::default()
        .num_edges(env_usize("HNSW_EDGES", degree / 2))
        .ef_construction(env_usize("HNSW_EF_CONSTRUCTION", 150));
    counting.take();
    let started = Instant::now();
    let hnsw = HNSW::index_vectors(&counting, hnsw_params.clone()).unwrap();
    let hnsw_build = started.elapsed().as_secs_f64();
    let hnsw_build_distances = counting.take();
    let edges = hnsw_edges(&hnsw);
    println!(
        "hnsw     m={} ef_construction={}: built in {hnsw_build:.1}s, \
         {:.0}M distances, {edges} edges ({:.1} MiB), {:.2}x vamana's memory",
        hnsw_params.m,
        hnsw_params.ef_construction,
        hnsw_build_distances as f64 / 1e6,
        (edges * 4) as f64 / (1024.0 * 1024.0),
        edges as f64 / vamana_slots as f64,
    );

    println!(
        "\n{:<8} {:>6} {:>9} {:>14}",
        "index", "beam", "recall@10", "distances/query"
    );
    let beams = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480];
    let mut vamana_curve = Vec::new();
    let mut hnsw_curve = Vec::new();
    let mut scratch = SearchScratch::new(vectors);
    for beam in beams {
        let searching = Comparisons::default();
        counting.take();
        let mut total_recall = 0.0;
        for (query, exact) in queries.iter().zip(&exact) {
            // Through the wrapper, so both indexes are measured by one
            // instrument rather than by two that are believed to agree.
            let calculator = counting.dist_calculator(query.clone(), 0.0);
            let result = greedy_search(
                &built.graph,
                &calculator,
                built.medoid,
                beam,
                &mut scratch,
                &searching,
            )
            .unwrap();
            let found = result
                .candidates
                .iter()
                .take(K)
                .map(|node| node.id)
                .collect::<Vec<_>>();
            total_recall += recall(&found, exact);
        }
        let measured = counting.take();
        assert_eq!(
            measured,
            searching.get(),
            "the crate's own comparison counter disagrees with the store's"
        );
        let point = (
            total_recall / num_queries as f64,
            measured as f64 / num_queries as f64,
        );
        println!(
            "{:<8} {beam:>6} {:>9.4} {:>14.1}",
            "vamana", point.0, point.1
        );
        vamana_curve.push(point);
    }

    for beam in beams {
        counting.take();
        let mut total_recall = 0.0;
        for (query, exact) in queries.iter().zip(&exact) {
            let found = hnsw
                .search_basic(
                    query.clone(),
                    K,
                    &HnswQueryParams {
                        ef: beam,
                        lower_bound: None,
                        upper_bound: None,
                        dist_q_c: 0.0,
                        use_acorn: false,
                    },
                    None,
                    &counting,
                )
                .unwrap()
                .iter()
                .take(K)
                .map(|node| node.id)
                .collect::<Vec<_>>();
            total_recall += recall(&found, exact);
        }
        let point = (
            total_recall / num_queries as f64,
            counting.take() as f64 / num_queries as f64,
        );
        println!("{:<8} {beam:>6} {:>9.4} {:>14.1}", "hnsw", point.0, point.1);
        hnsw_curve.push(point);
    }

    // Recall is a mean over `queries * K` binary outcomes, so it carries a
    // standard error, and on the steep part of the curve a small error in
    // recall becomes a large one in cost. Printed alongside the comparison
    // because a 5% difference in cost means nothing next to a 7% error bar.
    println!(
        "\n{:>9} {:>10} {:>10} {:>12} {:>9}",
        "recall@10", "vamana", "hnsw", "vamana cost", "+/- cost"
    );
    let samples = (num_queries * K) as f64;
    for target in [0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995] {
        match (
            cost_at_recall(&vamana_curve, target),
            cost_at_recall(&hnsw_curve, target),
        ) {
            (Some(ours), Some(theirs)) => {
                let recall_error = (target * (1.0 - target) / samples).sqrt();
                let slope = slope_at(&vamana_curve, target).unwrap_or(0.0);
                println!(
                    "{target:>9.3} {ours:>10.0} {theirs:>10.0} {:>11.0}% {:>8.0}%",
                    100.0 * ours / theirs,
                    100.0 * recall_error * slope / ours
                );
            }
            _ => println!(
                "{target:>9.3} {:>10} {:>10} {:>12} {:>9}",
                "-", "-", "out of range", "-"
            ),
        }
    }
}

/// How fast cost grows with recall around `target`, from the bracketing points.
fn slope_at(curve: &[(f64, f64)], target: f64) -> Option<f64> {
    let above = curve.iter().position(|(recall, _)| *recall >= target)?;
    if above == 0 {
        return None;
    }
    let (low_recall, low_cost) = curve[above - 1];
    let (high_recall, high_cost) = curve[above];
    let span = high_recall - low_recall;
    (span > 0.0).then(|| (high_cost - low_cost) / span)
}
