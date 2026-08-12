// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a stream of deletions and insertions does to an index nobody rebuilds.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --release --example churn_cycle
//! ```
//!
//! Environment: `SIFT_DIR` (required), `MODE` (`two-pass` or `merge`, default
//! `two-pass`), `VECTORS` (rows the dataset holds, default 100000), `QUERIES`
//! (default 1000), `PARTITIONS` (default 100), `NPROBES` (default 10),
//! `SEARCH_LIST` (query beam, default 100), `ROWS_PER_FRAGMENT` (default 25000).
//!
//! The experiment the FreshVamana paper closes on, translated: a round deletes a
//! residue class of the ids, appends exactly as many fresh rows as it removed,
//! and brings the index back up to date. Five rounds over five residue classes,
//! so by the end **every row the index was built over is gone**, the population
//! is back where it started, and the index has never been rebuilt.
//!
//! The share removed grows round by round: a round takes its residue class of
//! every id ever issued, so it takes a fifth of the original rows and a fifth of
//! each replacement batch too. The last round therefore empties the original
//! fragments outright, which is exactly the state an in-place insert refuses.
//!
//! # The two ways to bring the index back up to date
//!
//! `MODE=two-pass` is the pair of calls this crate had first: [`consolidate_index`]
//! takes the deleted rows out, then [`insert_in_place`] links the replacements
//! in. The order is forced rather than preferred - a segment built over a
//! fragment the dataset no longer has cannot be grown in place, because its new
//! vertices would land under a coverage that no longer names the old ones - and
//! by the last round that refusal is live. Every partition that both lost and
//! gained a row crosses the disk twice, and every partition that did neither is
//! copied twice.
//!
//! `MODE=merge` is [`merge_index`], which does the same work in one pass and has
//! no order to get wrong. Everything else about the run is identical, so the two
//! arms are an A/B of one binary: what a round costs, and what the index it
//! leaves behind answers.
//!
//! The paper's claim is that recall holds to within about a percentage point
//! across such a cycle. What this measures is that, plus what the cycle costs in
//! bytes, files and time - and, at the end, the same index rebuilt from scratch
//! over the same rows by the same binary, which is the ceiling everything above
//! it is read against.
//!
//! Deletion is by `id % 5`, which spreads it evenly over fragments and
//! partitions rather than emptying anything. That matters twice: an emptied
//! fragment leaves the dataset and would be measuring coverage instead of
//! recall, and evenly spread deletion is the shape the one-hop repair in
//! `src/consolidate.rs` finds hardest.

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
use lance_vamana::consolidator::consolidate_index;
use lance_vamana::inserter::insert_in_place;
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
/// Rounds, and the residue class each one deletes. Five of five, so the cycle
/// removes every row the index started with.
const ROUNDS: u64 = 5;

/// What a round runs to bring the index back up to date.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// [`consolidate_index`] and then [`insert_in_place`], in that order.
    TwoPass,
    /// [`merge_index`], one pass over the partitions, no order to get wrong.
    Merge,
}

impl Mode {
    fn from_env() -> Self {
        match std::env::var("MODE").as_deref().unwrap_or("two-pass") {
            "two-pass" => Self::TwoPass,
            "merge" => Self::Merge,
            other => panic!("MODE is 'two-pass' or 'merge', not {other:?}"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::TwoPass => "two-pass",
            Self::Merge => "merge",
        }
    }
}

fn vectors_of(pool: &[f32], dim: usize, from: usize, count: usize) -> FixedSizeListArray {
    FixedSizeListArray::try_new_from_values(
        Float32Array::from(pool[from * dim..(from + count) * dim].to_vec()),
        dim as i32,
    )
    .unwrap()
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

/// The `id` of every live row, keyed by the address the index answers in.
///
/// Rebuilt every round rather than kept: an append gives out new addresses, and
/// this is the only thing that turns an answer back into a vector of the pool.
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

async fn index_files(dataset: &Dataset) -> (usize, u64) {
    let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    let files = indices
        .iter()
        .flat_map(|index| index.files.iter().flatten())
        .collect::<Vec<_>>();
    (files.len(), files.iter().map(|file| file.size_bytes).sum())
}

/// Exact nearest `K` of one query among `live`, by brute force.
fn exact_top(store: &FlatFloatStorage, query: &ArrayRef, live: &[u64]) -> Vec<u64> {
    let calculator = store.dist_calculator(query.clone(), 0.0);
    let mut scored = live
        .iter()
        .map(|id| (calculator.distance(*id as u32), *id))
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(K, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(K);
    scored.into_iter().map(|(_, id)| id).collect()
}

struct Measured {
    recall: f64,
    comparisons: f64,
    bytes: f64,
    iops: f64,
    median_micros: u128,
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
        bytes: (after.bytes_read - before.bytes_read) as f64 / per_query,
        iops: (after.iops - before.iops) as f64 / per_query,
        median_micros: latencies[latencies.len() / 2],
    }
}

#[tokio::main]
async fn main() {
    let dir =
        std::env::var("SIFT_DIR").expect("set SIFT_DIR to the directory holding sift_*.fvecs");
    let (pool, dim, total) = read_fvecs(&format!("{dir}/sift_base.fvecs"));
    let (query_pool, query_dim, total_queries) = read_fvecs(&format!("{dir}/sift_query.fvecs"));
    assert_eq!(dim, query_dim);

    let rows = env_usize("VECTORS", 100_000);
    // Enough spare vectors to replace everything the cycle deletes. A round
    // takes a residue class of *every* id issued so far, so the share it removes
    // grows: a fifth of the original rows in the first round, and more in each
    // one after it as the replacements themselves come up for deletion.
    let pool_rows = (rows * 3).min(total);
    let num_queries = env_usize("QUERIES", 1000).min(total_queries);
    let partitions = env_usize("PARTITIONS", 100) as u32;
    let nprobes = env_usize("NPROBES", 10);
    let search_list_size = env_usize("SEARCH_LIST", 100);
    let rows_per_fragment = env_usize("ROWS_PER_FRAGMENT", 25_000);
    let mode = Mode::from_env();
    println!(
        "SIFT {rows} x {dim}, {num_queries} queries, k = {K}, {partitions} partitions, \
         nprobes {nprobes}, beam {search_list_size}; {ROUNDS} rounds in {} mode, each deleting \
         id % {ROUNDS} and replacing exactly what it removed",
        mode.label()
    );

    let index_params =
        IndexParams::new(VECTOR_COLUMN, partitions).with_distance_type(DISTANCE_TYPE);
    let search = SearchParams::new(K)
        .with_nprobes(nprobes)
        .with_search_list_size(search_list_size);
    let queries = (0..num_queries)
        .map(|q| {
            Arc::new(Float32Array::from(
                query_pool[q * dim..(q + 1) * dim].to_vec(),
            )) as ArrayRef
        })
        .collect::<Vec<_>>();
    // Over the whole pool, so that a vector keeps one id whether it is in the
    // dataset yet, in it, or gone from it.
    let store = FlatFloatStorage::new(vectors_of(&pool, dim, 0, pool_rows), DISTANCE_TYPE);

    let temp = tempfile::tempdir().unwrap();
    let uri = temp.path().to_str().unwrap();
    let mut dataset = write_rows(
        uri,
        0,
        vectors_of(&pool, dim, 0, rows),
        rows_per_fragment,
        WriteMode::Create,
    )
    .await;
    let started = Instant::now();
    let built = create_index(&mut dataset, INDEX_NAME, &index_params)
        .await
        .unwrap();
    println!(
        "index built in {:.1}s over {} vectors, {}M distances",
        started.elapsed().as_secs_f64(),
        built.vectors,
        built.comparisons / 1_000_000
    );

    let mut live = (0..rows as u64).collect::<HashSet<_>>();
    let mut next_id = rows as u64;
    println!(
        "\n{:>7} {:>8} {:>6} {:>10} {:>9} {:>10} {:>10} {:>12} {:>9} {:>9}",
        "round",
        "live",
        "files",
        "index MiB",
        "maint s",
        "recall@10",
        "dist/query",
        "bytes/query",
        "iops/qry",
        "p50 (us)"
    );
    for round in 0..ROUNDS {
        dataset
            .delete(&format!("{ID_COLUMN} % {ROUNDS} == {round}"))
            .await
            .unwrap();
        live.retain(|id| id % ROUNDS != round);
        let churned = rows - live.len();
        assert!(
            next_id as usize + churned <= pool_rows,
            "the pool holds {pool_rows} vectors and the cycle wants {}",
            next_id as usize + churned
        );
        let replacements = vectors_of(&pool, dim, next_id as usize, churned);

        // Only the maintenance calls are timed; the append that feeds them is
        // the same work in both arms and belongs to neither.
        let (maintenance_secs, detail) = match mode {
            Mode::TwoPass => {
                // Consolidated *before* the insert, and by round 4 that order is
                // not a preference: the round has emptied the original fragments
                // and an in-place insert refuses to grow a segment whose
                // coverage names a fragment the dataset no longer has.
                let started = Instant::now();
                let consolidated = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
                let consolidate_secs = started.elapsed().as_secs_f64();

                dataset = write_rows(
                    uri,
                    next_id,
                    replacements,
                    rows_per_fragment,
                    WriteMode::Append,
                )
                .await;

                let started = Instant::now();
                let inserted = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
                let insert_secs = started.elapsed().as_secs_f64();
                assert_eq!(
                    inserted.vectors, churned,
                    "the round did not index the rows it appended: {inserted:?}"
                );
                (
                    consolidate_secs + insert_secs,
                    format!(
                        "consolidate {consolidate_secs:.1}s repaired {} rebuilt {} dropped {} \
                         removed {} | insert {insert_secs:.1}s grown {} created {} copied {}",
                        consolidated.partitions_consolidated,
                        consolidated.partitions_rebuilt,
                        consolidated.partitions_dropped,
                        consolidated.vertices_removed,
                        inserted.partitions_grown,
                        inserted.partitions_created,
                        inserted.partitions_copied
                    ),
                )
            }
            Mode::Merge => {
                dataset = write_rows(
                    uri,
                    next_id,
                    replacements,
                    rows_per_fragment,
                    WriteMode::Append,
                )
                .await;

                let started = Instant::now();
                let merged = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
                let merge_secs = started.elapsed().as_secs_f64();
                assert_eq!(
                    merged.vectors_inserted, churned,
                    "the round did not index the rows it appended: {merged:?}"
                );
                (
                    merge_secs,
                    format!(
                        "written {} rebuilt {} copied {} dropped {} removed {}",
                        merged.partitions_written,
                        merged.partitions_rebuilt,
                        merged.partitions_copied,
                        merged.partitions_dropped,
                        merged.vertices_removed
                    ),
                )
            }
        };
        live.extend(next_id..next_id + churned as u64);
        next_id += churned as u64;
        assert_eq!(
            live.len(),
            rows,
            "the population did not come back to {rows}"
        );

        let report = report(&dataset, &store, &queries, &live, &search).await;
        println!(
            "{round:>7} {:>8} {:>6} {:>10.1} {maintenance_secs:>9.1} {:>10.4} {:>10.1} {:>12.0} \
             {:>9.1} {:>9}",
            live.len(),
            report.1,
            report.2,
            report.0.recall,
            report.0.comparisons,
            report.0.bytes,
            report.0.iops,
            report.0.median_micros
        );
        println!("         {detail}");
    }

    // The ceiling: the same rows, indexed in one pass, by this same binary.
    let started = Instant::now();
    create_index(&mut dataset, INDEX_NAME, &index_params)
        .await
        .unwrap();
    let rebuild_secs = started.elapsed().as_secs_f64();
    let report = report(&dataset, &store, &queries, &live, &search).await;
    println!(
        "{:>7} {:>8} {:>6} {:>10.1} {rebuild_secs:>9.1} {:>10.4} {:>10.1} {:>12.0} {:>9.1} {:>9}",
        "rebuilt",
        live.len(),
        report.1,
        report.2,
        report.0.recall,
        report.0.comparisons,
        report.0.bytes,
        report.0.iops,
        report.0.median_micros
    );
}

/// Measure the index as it now stands, and say what it takes on disk.
async fn report(
    dataset: &Dataset,
    store: &FlatFloatStorage,
    queries: &[ArrayRef],
    live: &HashSet<u64>,
    search: &SearchParams,
) -> (Measured, usize, f64) {
    assert_eq!(
        live.len(),
        dataset.count_rows(None).await.unwrap(),
        "the dataset and the harness disagree about what is live"
    );
    let mut live_ids = live.iter().copied().collect::<Vec<_>>();
    live_ids.sort_unstable();
    let truth = queries
        .iter()
        .map(|query| exact_top(store, query, &live_ids))
        .collect::<Vec<_>>();

    let ids = ids_by_address(dataset).await;
    let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
    // Warm-up, uncounted: nothing is cached between queries, so the first pass
    // measures the page cache filling rather than the index.
    let warm = queries.len().min(100);
    measure(&index, &queries[..warm], &truth[..warm], &ids, search).await;
    let measured = measure(&index, queries, &truth, &ids, search).await;
    let (files, bytes) = index_files(dataset).await;
    (measured, files, bytes as f64 / (1024.0 * 1024.0))
}
