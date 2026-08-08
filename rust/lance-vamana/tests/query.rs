// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Does a query through our own driver find what Lance's brute force finds?
//!
//! The reference is Lance's own exhaustive k-NN with `use_index(false)`, not a
//! second implementation of ours - checking a graph against a scan written by
//! the same hand proves only that the hand is consistent.
//!
//! Every recall figure here comes with what it cost. A walk that reaches every
//! vertex in a partition has perfect recall and has answered nothing, so recall
//! on its own cannot tell a working index from a scan in a costume.

use std::collections::HashSet;

use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::query::{SearchParams, VamanaIndex};

mod common;
use common::{DatasetFixture, VECTOR_COLUMN, random_vectors, recall};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 4;
const K: usize = 10;
const BEAM: usize = 30;
const QUERIES: usize = 40;

/// Partitions of ~2048 vertices, because a graph only stops being a scan once a
/// partition is much larger than what one walk can reach. A walk reaches roughly
/// `expansions * max_degree` vertices, so at R=16 and L=30 that is around 500 -
/// and a 512-vertex partition would be exhausted by a single query. Measured, not
/// assumed: the smaller fixture this file started with scored recall 1.0 while
/// touching 77% of the dataset.
fn measurement_fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 4,
        rows_per_fragment: 2048,
        ..Default::default()
    }
}

/// Enough rows and fragments to be a real dataset, small enough to build in a
/// second. Used by everything that is not measuring.
fn small_fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 2,
        rows_per_fragment: 512,
        ..Default::default()
    }
}

/// A narrower graph than the default, so the tests build in seconds. The working
/// point measured on SIFT is R=64; nothing here is a quality statement.
fn params() -> IndexParams {
    IndexParams::new(VECTOR_COLUMN, PARTITIONS).with_graph_params(BuildParams {
        max_degree: 16,
        search_list_size: 64,
        ..Default::default()
    })
}

async fn indexed_dataset(uri: &str, fixture: &DatasetFixture) -> Dataset {
    let mut dataset = fixture.write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    dataset
}

/// The distance from `query` to its true nearest neighbour, from Lance.
async fn brute_force_best_distance(dataset: &Dataset, query: &[f32]) -> f32 {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.nearest(VECTOR_COLUMN, &key, 1).unwrap();
    scanner.use_index(false);
    let batch = scanner.try_into_batch().await.unwrap();
    batch["_distance"].as_primitive::<Float32Type>().value(0)
}

/// Lance's own exhaustive k-NN over the same column.
async fn brute_force(dataset: &Dataset, query: &[f32], k: usize) -> Vec<u64> {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.nearest(VECTOR_COLUMN, &key, k).unwrap();
    scanner.use_index(false);
    scanner.with_row_id();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

struct Measured {
    recall: f64,
    comparisons: f64,
    partitions: f64,
}

/// The exhaustive answer for every query, computed once: a Lance scan per query
/// per configuration would dominate the runtime and measure nothing new.
async fn ground_truth(dataset: &Dataset, queries: &[Vec<f32>]) -> Vec<Vec<u64>> {
    let mut truth = Vec::with_capacity(queries.len());
    for query in queries {
        truth.push(brute_force(dataset, query, K).await);
    }
    truth
}

async fn measure(
    index: &VamanaIndex,
    queries: &[Vec<f32>],
    truth: &[Vec<u64>],
    search: &SearchParams,
) -> Measured {
    let mut total_recall = 0.0;
    let mut total_comparisons = 0u64;
    let mut total_partitions = 0usize;
    for (query, exact) in queries.iter().zip(truth) {
        let result = index.search(query, search).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_id)
            .collect::<Vec<_>>();
        assert_eq!(found.len(), search.k, "a query returned the wrong count");
        total_recall += recall(&found, exact);
        total_comparisons += result.comparisons;
        total_partitions += result.partitions_read;
    }
    Measured {
        recall: total_recall / queries.len() as f64,
        comparisons: total_comparisons as f64 / queries.len() as f64,
        partitions: total_partitions as f64 / queries.len() as f64,
    }
}

#[tokio::test]
async fn top_k_matches_lance_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = measurement_fixture();
    let dataset = indexed_dataset(uri, &fixture).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let measured = measure(&index, &queries, &truth, &search).await;
    let rows = fixture.indexed_rows() as f64;
    println!(
        "nprobes={} L={} -> recall@{K}={:.4}, {:.0} comparisons ({:.1}% of {rows} rows), \
         {:.1} partitions",
        search.nprobes,
        search.search_list_size,
        measured.recall,
        measured.comparisons,
        100.0 * measured.comparisons / rows,
        measured.partitions
    );

    assert!(
        measured.recall >= 0.95,
        "recall@{K} was {:.4}",
        measured.recall
    );
    assert!(
        measured.comparisons < rows / 4.0,
        "a graph that touches a quarter of the dataset is a scan in a costume: {:.0} of {rows}",
        measured.comparisons
    );
}

/// Routing is a trade, and both halves of it have to be visible. A driver that
/// quietly opened every partition would still pass a recall bar.
#[tokio::test]
async fn a_narrow_probe_costs_recall_and_buys_work() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &measurement_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let base = SearchParams::new(K).with_search_list_size(BEAM);
    let narrow = measure(&index, &queries, &truth, &base.clone().with_nprobes(1)).await;
    let wide = measure(
        &index,
        &queries,
        &truth,
        &base.with_nprobes(PARTITIONS as usize),
    )
    .await;
    println!(
        "nprobes=1  -> recall={:.4}, {:.0} comparisons, {:.1} partitions\n\
         nprobes={PARTITIONS} -> recall={:.4}, {:.0} comparisons, {:.1} partitions",
        narrow.recall,
        narrow.comparisons,
        narrow.partitions,
        wide.recall,
        wide.comparisons,
        wide.partitions
    );

    assert!(narrow.partitions < wide.partitions);
    assert!(
        narrow.recall < wide.recall,
        "one probe scored as well as every probe ({:.4} against {:.4}); \
         either routing is not happening or the fixture is too easy to route",
        narrow.recall,
        wide.recall
    );
    assert!(
        narrow.comparisons < wide.comparisons / 2.0,
        "a narrow probe must actually save work: {:.0} against {:.0}",
        narrow.comparisons,
        wide.comparisons
    );
}

/// The row ids we return must fetch the vectors we claimed distances for. This
/// is what a mixed-up local id looks like from the outside, and it survives both
/// the commit and a recall bar.
#[tokio::test]
async fn every_answer_resolves_to_the_row_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    for query in random_vectors(8, 99) {
        let result = index.search(&query, &search).await.unwrap();
        let row_ids = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_id)
            .collect::<Vec<_>>();
        assert_eq!(
            row_ids.iter().collect::<HashSet<_>>().len(),
            row_ids.len(),
            "the same row was returned twice, so partitions overlap or the merge is wrong"
        );

        let taken = dataset
            .take_rows(
                &row_ids,
                ProjectionRequest::from_columns(
                    [VECTOR_COLUMN, lance_core::ROW_ID],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        let fetched = taken[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let vectors = taken[VECTOR_COLUMN].as_fixed_size_list();
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<Float32Type>().values();

        for neighbor in &result.neighbors {
            // Joined on `_rowid`: `take_rows` neither preserves nor reports the
            // positions it dropped.
            let row = fetched
                .iter()
                .position(|id| *id == neighbor.row_id)
                .expect("a returned row id is not in the dataset");
            let stored = &values[row * dim..(row + 1) * dim];
            let distance = stored
                .iter()
                .zip(&query)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            assert!(
                (distance - neighbor.distance).abs() < 1e-4,
                "row {} was reported at distance {} but is at {distance}",
                neighbor.row_id,
                neighbor.distance
            );
        }
    }
}

/// A compaction that could not open the index leaves it pointing at fragments
/// that no longer exist. Returning nothing would look like a real answer.
#[tokio::test]
async fn an_index_stranded_by_compaction_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;
    assert!(VamanaIndex::open(&dataset, INDEX_NAME).await.is_ok());

    dataset.delete("true").await.unwrap();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("a stranded index must not answer queries");
    assert!(error.to_string().contains("stranded"), "{error}");
}

#[tokio::test]
async fn an_absent_index_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = small_fixture().write(uri).await;

    let error = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap_err();
    assert!(error.to_string().contains("no index named"), "{error}");
}

#[tokio::test]
async fn a_query_of_the_wrong_width_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let error = index
        .search(&[0.0; 3], &SearchParams::new(K))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("3 dimensions"), "{error}");

    let error = index
        .search(
            &random_vectors(1, 1)[0],
            &SearchParams::new(K).with_search_list_size(K - 1),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("smaller than k"), "{error}");
}

/// Probing past the end of the routing table asks for every partition, not for
/// an error and not for a panic.
#[tokio::test]
async fn nprobes_beyond_the_partition_count_is_harmless() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let query = &random_vectors(1, 7)[0];
    let all = index
        .search(
            query,
            &SearchParams::new(K)
                .with_nprobes(PARTITIONS as usize)
                .with_search_list_size(BEAM),
        )
        .await
        .unwrap();
    let beyond = index
        .search(
            query,
            &SearchParams::new(K)
                .with_nprobes(PARTITIONS as usize * 10)
                .with_search_list_size(BEAM),
        )
        .await
        .unwrap();
    assert_eq!(all.neighbors, beyond.neighbors);
    assert_eq!(all.partitions_read, beyond.partitions_read);
}

/// An empty partition has no row in the segment table and no file of its own,
/// but routing can still name it. Stepping over it is the normal case.
///
/// Forced rather than hoped for: 256 rows drawn from 8 distinct vectors cannot
/// fill 64 centroids, and the test says so if the fixture stops producing any.
#[tokio::test]
async fn a_probed_partition_that_holds_nothing_is_skipped() {
    const MANY: u32 = 64;

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture {
        fragments: 2,
        rows_per_fragment: 128,
        distinct_vectors: Some(8),
        ..Default::default()
    }
    .write(uri)
    .await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, MANY).with_graph_params(BuildParams {
            max_degree: 16,
            search_list_size: 64,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let result = index
        .search(
            &random_vectors(1, 5)[0],
            &SearchParams::new(K)
                .with_nprobes(MANY as usize)
                .with_search_list_size(BEAM),
        )
        .await
        .unwrap();

    assert!(
        result.partitions_read < MANY as usize,
        "every partition holds rows, so this test proves nothing about skipping"
    );
    assert_eq!(result.neighbors.len(), K);

    // The populated partition ids are sparse here, so a lookup that used a
    // partition's *position* in the table instead of its id would open somebody
    // else's file and never notice. With one probe the answer has to be exact:
    // the routed partition holds the query's nearest vector, and no other does.
    let single = SearchParams::new(1)
        .with_nprobes(1)
        .with_search_list_size(BEAM);
    for query in random_vectors(8, 31) {
        let found = index.search(&query, &single).await.unwrap();
        let exact = brute_force_best_distance(&dataset, &query).await;
        assert!(
            (found.neighbors[0].distance - exact).abs() < 1e-4,
            "one probe returned {} where the nearest vector is at {exact}",
            found.neighbors[0].distance
        );
    }
}
