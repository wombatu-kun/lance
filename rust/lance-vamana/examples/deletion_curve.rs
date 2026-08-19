// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What deleting rows under a built index costs it.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --release --example deletion_curve
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 2000), `PARTITIONS` (default 100), `NPROBES` (default 10),
//! `SEARCH_LIST` (query beam, default 100), `ROWS_PER_FRAGMENT` (default 25000).
//!
//! Deletion here is Lance's own: the rows go into deletion vectors. The curve is
//! walked twice over two identically built indices, and the pair is the point:
//!
//! - **buried** - nothing is ever consolidated. A deleted row's vertex stays in
//!   the graph, is walked through, and is dropped from the answer at the end, so
//!   this measures how fast a graph rots when its dead are never buried.
//! - **consolidated** - [`consolidate_index`] runs after every deletion step,
//!   which is the operating mode the connectivity measurement in
//!   `src/consolidate.rs` argues for: often and in small steps, because the
//!   one-hop repair cannot hold a graph together when it is asked to remove most
//!   of it at once.
//!
//! Row for row, the two tables are the before and after of consolidating at that
//! fraction. What to read them for is **bytes**: a tombstone costs the walk
//! nothing, so the whole return on consolidation is in what a query has to read
//! and what the index takes on disk.
//!
//! The rows deleted are chosen by `id % 10`, which spreads them evenly over
//! fragments and partitions. Deleting a contiguous range instead would empty
//! whole fragments, and an emptied fragment leaves the manifest - which narrows
//! the index's coverage and would measure that instead.
//!
//! Ground truth is computed once, to depth [`TRUTH_DEPTH`], and the live top-`K`
//! of each stage is the prefix of it that survives. That is exact rather than
//! approximate as long as every query keeps `K` live answers inside that depth,
//! which is asserted rather than assumed.

use std::collections::{HashMap, HashSet};
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
use lance::index::DatasetIndexExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::consolidator::consolidate_index;
use lance_vamana::query::{SearchParams, VamanaIndex};

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const K: usize = 10;
/// How deep the one exact answer is computed.
///
/// The live top-`K` at a stage is the first `K` surviving entries of it, so the
/// depth has to outlast the deletions: at the last stage a tenth of the rows are
/// left, and a tenth of 500 is still five times `K`.
const TRUTH_DEPTH: usize = 500;
const VECTOR_COLUMN: &str = "vector";
const ID_COLUMN: &str = "id";
const INDEX_NAME: &str = "vamana_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;
/// Deleted fractions, in tenths, cumulative: each stage deletes a superset of
/// the last, so no stage rebuilds anything the previous one already removed.
///
/// Past 50% because that is where the answer is. A tombstoned vertex costs the
/// walk nothing extra - it is visited either way - so what breaks first is not
/// cost but the beam: `k` live answers have to survive inside it, and a beam of
/// 100 asked for 10 has slack until nine rows in ten are gone.
const STAGES: [u64; 6] = [0, 1, 3, 5, 7, 9];

/// Exact nearest `depth` ids of one query, by brute force over every row.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef, depth: usize) -> Vec<u64> {
    let calculator = store.dist_calculator(query, 0.0);
    let mut scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id))
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(depth, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(depth);
    scored.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
    scored.into_iter().map(|(_, id)| id as u64).collect()
}

fn vectors_of(values: Vec<f32>, dim: usize) -> FixedSizeListArray {
    FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32).unwrap()
}

async fn write_dataset(
    uri: &str,
    vectors: FixedSizeListArray,
    rows_per_fragment: usize,
) -> Dataset {
    let rows = vectors.len();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_COLUMN, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            max_rows_per_group: rows_per_fragment.min(8192),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// The `id` of every row, keyed by the address the index answers in.
///
/// Read out of the dataset rather than derived from the write order: the index
/// returns row addresses, and the mapping from address to id is the dataset's to
/// state. Deletions never move a row, so this is built once and stays valid.
async fn ids_by_address(dataset: &Dataset) -> HashMap<u64, u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project(&[ID_COLUMN]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let ids = batch[ID_COLUMN].as_primitive::<UInt64Type>();
    let addresses = batch[ROW_ID].as_primitive::<UInt64Type>();
    addresses
        .values()
        .iter()
        .zip(ids.values())
        .map(|(address, id)| (*address, *id))
        .collect()
}

struct Measured {
    recall: f64,
    /// Neighbours actually returned, averaged. Separates "found the wrong rows"
    /// from "could not fill `k`": once the live rows inside the beam fall below
    /// `k`, the index answers short and no amount of graph quality helps.
    answered: f64,
    comparisons: f64,
    /// Bytes the index read, per query, over the measured pass only.
    ///
    /// Taken off the index's own scheduler and differenced across the pass, so
    /// the warm-up before it is excluded. This is the number consolidation is
    /// judged by - a tombstone costs no distances, only bytes.
    bytes: f64,
    median_micros: u128,
    p95_micros: u128,
}

/// What the committed index takes on disk, as Lance recorded it at commit.
async fn index_bytes(dataset: &Dataset) -> u64 {
    dataset
        .load_indices_by_name(INDEX_NAME)
        .await
        .unwrap()
        .iter()
        .flat_map(|index| index.files.iter().flatten())
        .map(|file| file.size_bytes)
        .sum()
}

async fn measure(
    index: &VamanaIndex,
    queries: &[ArrayRef],
    truth: &[Vec<u64>],
    ids: &HashMap<u64, u64>,
    deleted_below: u64,
    params: &SearchParams,
) -> Measured {
    let live = |id: u64| id % 10 >= deleted_below;
    let mut hits = 0usize;
    let mut answered = 0usize;
    let mut comparisons = 0u64;
    let mut latencies = Vec::with_capacity(queries.len());
    let bytes_before = index.io_stats().bytes_read;

    for (query, exact) in queries.iter().zip(truth) {
        let expected = exact
            .iter()
            .copied()
            .filter(|id| live(*id))
            .take(K)
            .collect::<HashSet<_>>();
        assert_eq!(
            expected.len(),
            K,
            "ground truth of depth {TRUTH_DEPTH} ran out of live answers at {}% deleted",
            deleted_below * 10
        );

        let started = Instant::now();
        let result = index
            .search(
                query
                    .as_primitive::<arrow_array::types::Float32Type>()
                    .values(),
                params,
            )
            .await
            .unwrap();
        latencies.push(started.elapsed().as_micros());

        comparisons += result.comparisons;
        answered += result.neighbors.len();
        for neighbor in &result.neighbors {
            let id = *ids
                .get(&neighbor.row_addr)
                .expect("the index answered with an address the dataset does not have");
            // The measurement's own correctness check: a deleted row coming back
            // would inflate nothing and deflate nothing - it would simply mean
            // the delete list is not being applied, and the whole curve below
            // would be measuring an index that never lost anything.
            assert!(live(id), "row {id} was deleted and still came back");
            hits += usize::from(expected.contains(&id));
        }
    }

    latencies.sort_unstable();
    Measured {
        recall: hits as f64 / (queries.len() * K) as f64,
        answered: answered as f64 / queries.len() as f64,
        comparisons: comparisons as f64 / queries.len() as f64,
        bytes: (index.io_stats().bytes_read - bytes_before) as f64 / queries.len() as f64,
        median_micros: latencies[latencies.len() / 2],
        p95_micros: latencies[latencies.len() * 95 / 100],
    }
}

#[tokio::main]
async fn main() {
    let dir =
        std::env::var("SIFT_DIR").expect("set SIFT_DIR to the directory holding sift_*.fvecs");
    let (base, dim, total) = read_fvecs(&format!("{dir}/sift_base.fvecs"));
    let (queries, query_dim, total_queries) = read_fvecs(&format!("{dir}/sift_query.fvecs"));
    assert_eq!(dim, query_dim);

    let requested = env_usize("VECTORS", 100_000);
    let rows = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let num_queries = env_usize("QUERIES", 2000).min(total_queries);
    let partitions = env_usize("PARTITIONS", 100) as u32;
    let nprobes = env_usize("NPROBES", 10);
    let search_list_size = env_usize("SEARCH_LIST", 100);
    let rows_per_fragment = env_usize("ROWS_PER_FRAGMENT", 25_000);
    println!(
        "SIFT {rows} x {dim}, {num_queries} queries, k = {K}, {partitions} partitions, \
         nprobes {nprobes}, beam {search_list_size}"
    );

    let vectors = vectors_of(base[..rows * dim].to_vec(), dim);
    let query_vectors = (0..num_queries)
        .map(|q| Arc::new(Float32Array::from(queries[q * dim..(q + 1) * dim].to_vec())) as ArrayRef)
        .collect::<Vec<_>>();

    let started = Instant::now();
    let store = FlatFloatStorage::new(vectors.clone(), DISTANCE_TYPE);
    let truth = query_vectors
        .iter()
        .map(|query| exact_top(&store, query.clone(), TRUTH_DEPTH))
        .collect::<Vec<_>>();
    println!(
        "exact top-{TRUTH_DEPTH} by brute force in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let bench = Bench {
        rows,
        rows_per_fragment,
        vectors,
        index: IndexParams::new(VECTOR_COLUMN, partitions).with_distance_type(DISTANCE_TYPE),
        search: SearchParams::new(K)
            .with_nprobes(nprobes)
            .with_search_list_size(search_list_size),
        queries: query_vectors,
        truth,
    };

    // A fresh dataset and a fresh build for each curve. Sharing one would make
    // the second curve's "before" an index the first curve had already
    // consolidated, and the two columns would stop being comparable.
    for (label, consolidating) in [("buried", false), ("consolidated", true)] {
        let temp = tempfile::tempdir().unwrap();
        bench
            .curve(temp.path().to_str().unwrap(), label, consolidating)
            .await;
    }
}

/// Everything both curves are run with, so that the only difference between
/// them is whether consolidation runs.
struct Bench {
    rows: usize,
    rows_per_fragment: usize,
    vectors: FixedSizeListArray,
    index: IndexParams,
    search: SearchParams,
    queries: Vec<ArrayRef>,
    truth: Vec<Vec<u64>>,
}

impl Bench {
    async fn curve(&self, uri: &str, label: &str, consolidating: bool) {
        let mut dataset = write_dataset(uri, self.vectors.clone(), self.rows_per_fragment).await;
        let ids = ids_by_address(&dataset).await;
        assert_eq!(ids.len(), self.rows);

        let started = Instant::now();
        let stats = create_index(&mut dataset, INDEX_NAME, &self.index)
            .await
            .unwrap();
        println!(
            "\n=== {label} ===\nindex built in {:.1}s over {} vectors in {} partitions, \
             {}M distances",
            started.elapsed().as_secs_f64(),
            stats.vectors,
            stats.partitions,
            stats.comparisons / 1_000_000
        );
        println!(
            "\n{:>8} {:>8} {:>10} {:>9} {:>10} {:>16} {:>12} {:>10} {:>9} {:>9}",
            "deleted",
            "live",
            "recall@10",
            "answered",
            "dist/query",
            "dist/live vector",
            "bytes/query",
            "index MiB",
            "p50 (us)",
            "p95 (us)"
        );

        for deleted_below in STAGES {
            if deleted_below > 0 {
                dataset
                    .delete(&format!("{ID_COLUMN} % 10 < {deleted_below}"))
                    .await
                    .unwrap();
                if consolidating {
                    self.consolidate(&mut dataset).await;
                }
            }
            let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
            let live = ids.values().filter(|id| *id % 10 >= deleted_below).count();
            assert_eq!(
                live,
                dataset.count_rows(None).await.unwrap(),
                "the dataset and the deletion predicate disagree about what is live"
            );
            // Warm-up, uncounted: the driver reads a partition per query with no
            // cache of its own, so the first pass over a stage is measuring the
            // page cache filling rather than the index.
            let warm = self.queries.len().min(100);
            self.measure(&index, &ids, deleted_below, warm).await;
            let measured = self
                .measure(&index, &ids, deleted_below, self.queries.len())
                .await;
            println!(
                "{:>7}% {:>8} {:>10.4} {:>9.2} {:>10.1} {:>15.4}% {:>12.0} {:>10.1} {:>9} {:>9}",
                deleted_below * 10,
                live,
                measured.recall,
                measured.answered,
                measured.comparisons,
                100.0 * measured.comparisons / live as f64,
                measured.bytes,
                index_bytes(&dataset).await as f64 / (1024.0 * 1024.0),
                measured.median_micros,
                measured.p95_micros
            );
        }
    }

    async fn consolidate(&self, dataset: &mut Dataset) {
        let started = Instant::now();
        let stats = consolidate_index(dataset, INDEX_NAME).await.unwrap();
        println!(
            "  consolidated in {:>5.1}s: {} repaired, {} rebuilt, {} copied, {} dropped, \
             {} vertices removed, {}M distances",
            started.elapsed().as_secs_f64(),
            stats.partitions_consolidated,
            stats.partitions_rebuilt,
            stats.partitions_copied,
            stats.partitions_dropped,
            stats.vertices_removed,
            stats.comparisons / 1_000_000
        );
    }

    async fn measure(
        &self,
        index: &VamanaIndex,
        ids: &HashMap<u64, u64>,
        deleted_below: u64,
        queries: usize,
    ) -> Measured {
        measure(
            index,
            &self.queries[..queries],
            &self.truth[..queries],
            ids,
            deleted_below,
            &self.search,
        )
        .await
    }
}
