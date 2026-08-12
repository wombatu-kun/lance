// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a delta segment costs a query.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --release --example insertion_curve
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 2000), `PARTITIONS` (default 100), `NPROBES` (default 10),
//! `SEARCH_LIST` (query beam, default 100), `ROWS_PER_FRAGMENT` (default 5000).
//!
//! Every arm ends with **the same dataset**: the same rows, in the same
//! fragments, at the same addresses. What differs is the history that produced
//! the index over it. The first arm builds once over everything; the rest build
//! over the first half and then append the second half in batches, indexing each
//! batch with [`insert_as_segment`]. So the table below is one question - what
//! does it cost to have grown an index rather than to have built it - and the
//! answer is read down the columns.
//!
//! `nprobes` is spent **per segment**, so the arithmetic to have in mind is that
//! eight segments read eight times the partitions of one. Whether that shows up
//! as eight times the bytes is exactly what is being measured: a delta's
//! partitions hold a fraction of the rows, and a partition is read whole.
//!
//! Two differences between the arms are inherent rather than controlled away,
//! and both are properties of the thing being measured. The one-segment arm
//! trains its router on every row, while a grown index trains on the first half
//! and every delta inherits it. And a delta's graph is built over its own rows
//! only, so a new row has no edge to an old one - within a delta the walk is
//! nearly exhaustive, which is a recall advantage at these sizes and a cost
//! disadvantage at any scale.
//!
//! # And what it costs to undo
//!
//! Every grown arm then folds its deltas back into the base with [`merge_index`]
//! and is measured again, on the same rows at the same addresses, as the row
//! marked `N -> 1`. That gives the number an index nobody rebuilds actually
//! needs: not whether folding is cheaper than a rebuild, but **after how many
//! queries a fold has paid for itself** - the seconds it took, divided by the
//! latency it takes off every query from then on. Below that many queries a
//! delta is the cheaper way to hold the rows, above it the fold is, and the
//! crossover is a property of the shape of the index rather than a threshold
//! anyone gets to pick.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::inserter::insert_as_segment;
use lance_vamana::merger::merge_index;
use lance_vamana::query::{SearchParams, VamanaIndex};

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const K: usize = 10;
const VECTOR_COLUMN: &str = "vector";
const ID_COLUMN: &str = "id";
const INDEX_NAME: &str = "vamana_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;
/// How many segments the index ends up with. One is the built-in baseline: the
/// same rows, indexed in one pass, by the same binary.
const SEGMENT_COUNTS: [usize; 4] = [1, 2, 4, 8];

/// Exact nearest `K` ids of one query, by brute force over every row.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef) -> Vec<u64> {
    let calculator = store.dist_calculator(query, 0.0);
    let mut scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id))
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(K, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(K);
    scored.into_iter().map(|(_, id)| id as u64).collect()
}

async fn write_rows(
    uri: &str,
    first_id: u64,
    vectors: FixedSizeListArray,
    rows_per_fragment: usize,
    mode: WriteMode,
) -> Dataset {
    let rows = vectors.len() as u64;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_COLUMN, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(first_id..first_id + rows)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            mode,
            max_rows_per_file: rows_per_fragment,
            max_rows_per_group: rows_per_fragment.min(8192),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// The `id` of every row, keyed by the address the index answers in.
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

/// Files of the committed index and what they take, as Lance recorded them.
///
/// The file count is a cost of its own and not a proxy for the bytes: Lance
/// carries one manifest entry per file of a committed index into every manifest
/// written afterwards, so a delta of many tiny partitions is paid for by every
/// later append, delete and open of the dataset.
async fn index_files(dataset: &Dataset) -> (usize, u64) {
    let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    let files = indices
        .iter()
        .flat_map(|index| index.files.iter().flatten())
        .collect::<Vec<_>>();
    (files.len(), files.iter().map(|file| file.size_bytes).sum())
}

struct Measured {
    recall: f64,
    comparisons: f64,
    partitions: f64,
    bytes: f64,
    iops: f64,
    /// The mean rather than the median, because this is the one latency that
    /// multiplies: what a fold saves over a run of queries is the mean saving
    /// times the count, and the median would understate a tail the deltas own.
    mean_micros: f64,
    median_micros: u128,
    p95_micros: u128,
}

async fn measure(
    index: &VamanaIndex,
    queries: &[ArrayRef],
    truth: &[Vec<u64>],
    ids: &HashMap<u64, u64>,
    params: &SearchParams,
) -> Measured {
    let mut hits = 0usize;
    let mut comparisons = 0u64;
    let mut partitions = 0usize;
    let mut latencies = Vec::with_capacity(queries.len());
    let before = index.io_stats();

    for (query, exact) in queries.iter().zip(truth) {
        let expected = exact.iter().copied().collect::<HashSet<_>>();
        let started = Instant::now();
        let result = index
            .search(query.as_primitive::<Float32Type>().values(), params)
            .await
            .unwrap();
        latencies.push(started.elapsed().as_micros());

        comparisons += result.comparisons;
        partitions += result.partitions_read;
        assert_eq!(
            result.neighbors.len(),
            K,
            "nothing is deleted here, so a short answer is a bug rather than a result"
        );
        for neighbor in &result.neighbors {
            let id = *ids
                .get(&neighbor.row_addr)
                .expect("the index answered with an address the dataset does not have");
            hits += usize::from(expected.contains(&id));
        }
    }

    latencies.sort_unstable();
    let after = index.io_stats();
    let per_query = queries.len() as f64;
    Measured {
        recall: hits as f64 / (queries.len() * K) as f64,
        comparisons: comparisons as f64 / per_query,
        partitions: partitions as f64 / per_query,
        bytes: (after.bytes_read - before.bytes_read) as f64 / per_query,
        iops: (after.iops - before.iops) as f64 / per_query,
        mean_micros: latencies.iter().sum::<u128>() as f64 / per_query,
        median_micros: latencies[latencies.len() / 2],
        p95_micros: latencies[latencies.len() * 95 / 100],
    }
}

/// `total` split into `groups` as evenly as it goes, remainder to the front.
fn split(total: usize, groups: usize) -> Vec<usize> {
    assert!(
        groups <= total,
        "cannot make {groups} deltas out of {total} fragments; lower ROWS_PER_FRAGMENT"
    );
    (0..groups)
        .map(|group| total / groups + usize::from(group < total % groups))
        .collect()
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
    let rows_per_fragment = env_usize("ROWS_PER_FRAGMENT", 5_000);
    assert_eq!(
        rows % (2 * rows_per_fragment),
        0,
        "the halves have to be whole fragments so that every arm ends with the same layout"
    );
    println!(
        "SIFT {rows} x {dim}, {num_queries} queries, k = {K}, {partitions} partitions, \
         nprobes {nprobes}, beam {search_list_size}, {} rows per fragment",
        rows_per_fragment
    );

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();
    let query_vectors = (0..num_queries)
        .map(|q| Arc::new(Float32Array::from(queries[q * dim..(q + 1) * dim].to_vec())) as ArrayRef)
        .collect::<Vec<_>>();

    let started = Instant::now();
    let store = FlatFloatStorage::new(vectors.clone(), DISTANCE_TYPE);
    let truth = query_vectors
        .iter()
        .map(|query| exact_top(&store, query.clone()))
        .collect::<Vec<_>>();
    println!(
        "exact top-{K} by brute force in {:.1}s",
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

    println!(
        "\n{:>8} {:>6} {:>10} {:>14} {:>10} {:>9} {:>10} {:>12} {:>9} {:>9} {:>9}",
        "segments",
        "files",
        "index MiB",
        "maintenance s",
        "recall@10",
        "parts/qry",
        "dist/query",
        "bytes/query",
        "iops/qry",
        "p50 (us)",
        "p95 (us)"
    );
    for segments in SEGMENT_COUNTS {
        let temp = tempfile::tempdir().unwrap();
        bench.arm(temp.path().to_str().unwrap(), segments).await;
    }
}

/// Everything every arm is run with, so that the only difference between them is
/// how the index came to exist.
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
    async fn arm(&self, uri: &str, segments: usize) {
        let half = self.rows / 2;
        let started = Instant::now();
        let mut dataset = write_rows(
            uri,
            0,
            self.vectors.slice(0, half),
            self.rows_per_fragment,
            WriteMode::Create,
        )
        .await;

        if segments == 1 {
            dataset = write_rows(
                uri,
                half as u64,
                self.vectors.slice(half, self.rows - half),
                self.rows_per_fragment,
                WriteMode::Append,
            )
            .await;
            create_index(&mut dataset, INDEX_NAME, &self.index)
                .await
                .unwrap();
        } else {
            create_index(&mut dataset, INDEX_NAME, &self.index)
                .await
                .unwrap();
            let mut next = half;
            for group in split((self.rows - half) / self.rows_per_fragment, segments - 1) {
                let batch = group * self.rows_per_fragment;
                dataset = write_rows(
                    uri,
                    next as u64,
                    self.vectors.slice(next, batch),
                    self.rows_per_fragment,
                    WriteMode::Append,
                )
                .await;
                insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
                next += batch;
            }
            assert_eq!(next, self.rows);
        }
        let maintenance = started.elapsed().as_secs_f64();

        let ids = ids_by_address(&dataset).await;
        assert_eq!(ids.len(), self.rows);
        let grown = self
            .row(&dataset, &segments.to_string(), segments, maintenance, &ids)
            .await;
        if segments == 1 {
            return;
        }

        // The same rows at the same addresses, with the deltas folded back into
        // the base: what the fold costs, and what every query stops paying.
        let started = Instant::now();
        let merged = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
        let fold = started.elapsed().as_secs_f64();
        assert_eq!(
            merged.vectors_inserted, 0,
            "the arm left an unindexed fragment behind, so this folds more than the deltas"
        );
        let folded = self
            .row(&dataset, &format!("{segments} -> 1"), 1, fold, &ids)
            .await;

        let saved = grown.mean_micros - folded.mean_micros;
        let payback = if saved > 0.0 {
            format!("paid back after {:.0} queries", fold * 1_000_000.0 / saved)
        } else {
            "never paid back at this size".to_string()
        };
        println!(
            "         folded {} vertices in {fold:.1}s, saving {saved:.1} us a query: {payback}",
            merged.vertices_folded
        );
    }

    /// Measure the index as the dataset now has it, and print its row.
    async fn row(
        &self,
        dataset: &Dataset,
        label: &str,
        segments: usize,
        maintenance: f64,
        ids: &HashMap<u64, u64>,
    ) -> Measured {
        let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
        assert_eq!(
            index.num_segments(),
            segments,
            "this arm was supposed to leave {segments} segments"
        );

        // Warm-up, uncounted: the driver reads a partition per query with no
        // cache of its own, so the first pass is measuring the page cache
        // filling rather than the index.
        let warm = self.queries.len().min(100);
        measure(
            &index,
            &self.queries[..warm],
            &self.truth[..warm],
            ids,
            &self.search,
        )
        .await;
        let measured = measure(&index, &self.queries, &self.truth, ids, &self.search).await;
        let (files, bytes) = index_files(dataset).await;
        println!(
            "{label:>8} {files:>6} {:>10.1} {maintenance:>14.1} {:>10.4} {:>9.1} {:>10.1} \
             {:>12.0} {:>9.1} {:>9} {:>9}",
            bytes as f64 / (1024.0 * 1024.0),
            measured.recall,
            measured.partitions,
            measured.comparisons,
            measured.bytes,
            measured.iops,
            measured.median_micros,
            measured.p95_micros
        );
        measured
    }
}
