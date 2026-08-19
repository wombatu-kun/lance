// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An index that carries codes, and the walk that steers by them.
//!
//! One index answers both ways, which is the only comparison worth making: the
//! same graph, the same routing, the same beam, and a switch. Anything measured
//! across two builds would be measuring two k-means runs as much as two walks.
//!
//! What these tests are really guarding is a *silent* failure. RaBitQ's
//! raw-query estimator wants the raw query beside `|q - c|^2`, because the
//! centroid is already folded into each vertex's factors; hand it the residual
//! instead, or read a factor from the wrong offset, and the distances come back
//! wrong rather than approximate. Nothing errors. The only thing that notices is
//! recall, so recall is what is asserted.

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::consolidator::consolidate_index;
use lance_vamana::inserter::{insert_as_segment, insert_in_place};
use lance_vamana::merger::merge_index;
use lance_vamana::query::{SearchParams, VamanaIndex, WalkMode};

mod common;
use common::{
    DatasetFixture, VECTOR_COLUMN, brute_force, random_vectors, read_committed_segments, recall,
};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 4;
const K: usize = 10;
const BEAM: usize = 30;
const QUERIES: usize = 40;

/// The measured working point. Below it a walk needs a wider beam to reach the
/// same recall, which is the whole finding the default rests on.
const CODE_BITS: u8 = 3;

/// Partitions large enough that a walk cannot exhaust one, which is the only
/// regime in which steering matters at all: on a partition a single query
/// reaches the whole of, every arm scores the same and the comparison is empty.
fn fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 4,
        rows_per_fragment: 2048,
        ..Default::default()
    }
}

fn params() -> IndexParams {
    IndexParams::new(VECTOR_COLUMN, PARTITIONS)
        .with_graph_params(BuildParams {
            max_degree: 16,
            search_list_size: 64,
            ..Default::default()
        })
        .with_code_bits(CODE_BITS)
}

async fn coded_dataset(uri: &str) -> Dataset {
    let mut dataset = fixture().write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    dataset
}

fn search(mode: WalkMode) -> SearchParams {
    SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM)
        .with_mode(mode)
}

struct Measured {
    recall: f64,
    comparisons: f64,
}

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
    mode: WalkMode,
) -> Measured {
    let params = search(mode);
    let mut total_recall = 0.0;
    let mut total_comparisons = 0u64;
    for (query, exact) in queries.iter().zip(truth) {
        let result = index.search(query, &params).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(found.len(), K, "a query returned the wrong count");
        total_recall += recall(&found, exact);
        total_comparisons += result.comparisons;
    }
    Measured {
        recall: total_recall / queries.len() as f64,
        comparisons: total_comparisons as f64 / queries.len() as f64,
    }
}

/// The test the whole module exists for.
///
/// A coded walk that has been fed the wrong query, or reads a factor from the
/// wrong offset, does not fail: it wanders, and comes back with an answer that
/// is merely poor. The bar is therefore relative - within a tenth of what the
/// same index answers exactly - rather than an absolute recall number that a
/// broken arm could still clear on an easy fixture.
#[tokio::test]
async fn a_coded_walk_lands_where_the_exact_one_does() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let exact = measure(&index, &queries, &truth, WalkMode::Exact).await;
    let coded = measure(&index, &queries, &truth, WalkMode::Coded).await;
    println!(
        "exact: recall@{K}={:.4}, {:.0} comparisons; coded({CODE_BITS} bits): recall@{K}={:.4}, \
         {:.0} comparisons",
        exact.recall, exact.comparisons, coded.recall, coded.comparisons
    );

    assert!(
        exact.recall > 0.5,
        "the exact arm scored {:.4}, so the fixture is not measuring a working index",
        exact.recall
    );
    assert!(
        coded.recall > exact.recall - 0.1,
        "the coded arm scored {:.4} against the exact arm's {:.4}",
        coded.recall,
        exact.recall
    );
    // The re-scoring is counted, so a coded walk is never the cheaper of the two
    // in distances. What it saves is bytes, and only once a walk stops reading
    // the partition whole.
    assert!(
        coded.comparisons > exact.comparisons,
        "a coded walk re-scores its candidates, so it computes more distances, not fewer"
    );
}

/// The re-scoring covers the whole candidate list, so the answer's distances are
/// exact whichever way the walk was steered. A caller comparing a returned
/// distance against a threshold depends on it.
#[tokio::test]
async fn a_coded_answer_carries_exact_distances() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let query = &random_vectors(1, 909)[0];
    let coded = index.search(query, &search(WalkMode::Coded)).await.unwrap();
    let exact = index.search(query, &search(WalkMode::Exact)).await.unwrap();

    let by_row = exact
        .neighbors
        .iter()
        .map(|neighbor| (neighbor.row_addr, neighbor.distance))
        .collect::<std::collections::HashMap<_, _>>();
    let mut shared = 0;
    for neighbor in &coded.neighbors {
        if let Some(distance) = by_row.get(&neighbor.row_addr) {
            assert_eq!(
                neighbor.distance, *distance,
                "row {} came back at a different distance from the two arms",
                neighbor.row_addr
            );
            shared += 1;
        }
    }
    assert!(
        shared > 0,
        "the two arms shared no rows at all, so nothing was compared"
    );
    assert!(
        coded
            .neighbors
            .windows(2)
            .all(|pair| pair[0].distance <= pair[1].distance),
        "a coded answer came back out of order, so it was not re-sorted after re-scoring"
    );
}

/// An exact walk over a coded index must not pay for the codes. The projection
/// is the only thing standing between it and a thirteen per cent tax on every
/// partition it reads.
#[tokio::test]
async fn an_exact_walk_does_not_read_the_code_column() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let query = &random_vectors(1, 55)[0];

    let mut bytes = Vec::new();
    for mode in [WalkMode::Exact, WalkMode::Coded] {
        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        let opened = index.io_stats().bytes_read;
        index.search(query, &search(mode)).await.unwrap();
        bytes.push(index.io_stats().bytes_read - opened);
    }
    println!(
        "one query read {} bytes exactly and {} bytes by codes",
        bytes[0], bytes[1]
    );
    assert!(
        bytes[0] < bytes[1],
        "an exact walk read {} bytes and a coded one {}, so the code column was fetched either \
         way",
        bytes[0],
        bytes[1]
    );
}

/// Asked for something the index cannot do, and told so, rather than quietly
/// given the other walk.
#[tokio::test]
async fn an_index_without_codes_refuses_the_coded_walk() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = fixture().write(uri).await;
    let mut plain = params();
    plain.code_bits = None;
    create_index(&mut dataset, INDEX_NAME, &plain)
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let error = index
        .search(&random_vectors(1, 1)[0], &search(WalkMode::Coded))
        .await
        .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains("without codes"), "{error}");
}

/// A dimension RaBitQ cannot pack is refused at build time, not at query time.
///
/// The alternative - building an index that silently has no codes - would be
/// found out by a query that ran slower than it was meant to, which is the worst
/// place to find it out.
#[tokio::test]
async fn a_dimension_rabit_cannot_pack_refuses_a_coded_build() {
    const ODD_DIM: i32 = 12;
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();

    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        VECTOR_COLUMN,
        DataType::FixedSizeList(item, ODD_DIM),
        true,
    )]));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (0..64)
            .map(|row| {
                Some(
                    (0..ODD_DIM)
                        .map(|d| Some((row * ODD_DIM + d) as f32))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        ODD_DIM,
    );
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams::default()),
    )
    .await
    .unwrap();

    let error = create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, 2).with_code_bits(CODE_BITS),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains("multiple of 8"), "{error}");
}

/// Every pass that writes a segment inherits the rotation, and a partition it
/// rewrites is re-coded from the vectors it wrote.
///
/// Both halves matter and they fail differently. A pass that minted its own
/// rotation would leave a *copied* partition decoded under the wrong one -
/// meaningless distances, silently. A pass that carried codes across a vertex
/// move would leave them describing the vector that used to be at that local id.
/// The delete is what makes both possible: it moves vertices and it leaves other
/// partitions untouched to be copied.
#[tokio::test]
async fn the_maintenance_passes_keep_the_codes_in_step() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = coded_dataset(uri).await;
    let minted = read_committed_segments(&dataset, INDEX_NAME).await[0]
        .manifest
        .metadata()
        .codes
        .clone();
    assert!(minted.is_some(), "the fixture built no codes");

    // All four passes that write a segment, in an order that makes each of them
    // do the thing it can get wrong: a delta built beside the base, a second
    // append linked into the graphs in place, a delete that moves vertices,
    // consolidation to rewrite what it emptied, and a merge to fold the lot.
    fixture().append(uri).await;
    dataset = Dataset::open(uri).await.unwrap();
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    fixture().append(uri).await;
    dataset = Dataset::open(uri).await.unwrap();
    insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    dataset
        .delete("vec IS NOT NULL AND _rowid % 7 = 0")
        .await
        .unwrap();
    consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    merge_index(&mut dataset, INDEX_NAME).await.unwrap();

    for segment in read_committed_segments(&dataset, INDEX_NAME).await {
        assert_eq!(
            segment.manifest.metadata().codes,
            minted,
            "segment {} was written under a rotation of its own",
            segment.uuid
        );
    }

    // The codes are only in step with the vectors if the two walks still agree,
    // and after this much churn they can only agree by being rebuilt correctly.
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let exact = measure(&index, &queries, &truth, WalkMode::Exact).await;
    let coded = measure(&index, &queries, &truth, WalkMode::Coded).await;
    println!(
        "after churn - exact: recall@{K}={:.4}; coded: recall@{K}={:.4}",
        exact.recall, coded.recall
    );
    assert!(
        exact.recall > 0.5,
        "the exact arm scored {:.4} after the churn",
        exact.recall
    );
    assert!(
        coded.recall > exact.recall - 0.1,
        "the coded arm scored {:.4} against the exact arm's {:.4} after the churn, so a rewritten \
         partition's codes no longer describe its vectors",
        coded.recall,
        exact.recall
    );
}

/// Cosine takes a detour, and the codes have to take it too.
///
/// The builder normalises what it stores and the router works in L2 over those
/// unit vectors, because it panics on cosine. RaBitQ's own factors are defined
/// for L2 and dot only, so the codes are built and read in that same L2 - which
/// over unit vectors orders exactly as cosine does, and the answer's distances
/// come from the exact re-scoring, so a caller sees cosine throughout.
///
/// None of that is visible from outside, and the mapping has three places to be
/// got wrong: the encode, the query key and the storage. The reference here is
/// the *exact* arm of the same index rather than Lance, because what is in doubt
/// is the codes and not cosine - the metric itself is pinned elsewhere.
#[tokio::test]
async fn a_cosine_index_walks_by_codes_too() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = fixture().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &params().with_distance_type(DistanceType::Cosine),
    )
    .await
    .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.metadata().distance_type, DistanceType::Cosine);

    // The lazy arm rides along, because cosine's detour has a fourth place to go
    // wrong there: the walk is handed the normalised query for the codes and the
    // raw one for the re-scoring, and the two are the same array under every
    // other metric - so a mix-up is invisible everywhere but here.
    let mut overlap = [0.0, 0.0];
    let queries = random_vectors(QUERIES, 4242);
    for query in &queries {
        let exact = index.search(query, &search(WalkMode::Exact)).await.unwrap();
        let rows = exact
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        for (slot, mode) in [WalkMode::Coded, WalkMode::Lazy].into_iter().enumerate() {
            let result = index.search(query, &search(mode)).await.unwrap();
            let found = result
                .neighbors
                .iter()
                .map(|neighbor| neighbor.row_addr)
                .collect::<Vec<_>>();
            overlap[slot] += recall(&found, &rows);
        }
    }
    for (label, total) in ["coded", "lazy"].into_iter().zip(overlap) {
        let overlap = total / queries.len() as f64;
        println!("cosine: the {label} arm recovered {overlap:.4} of the exact arm's answer");
        assert!(
            overlap > 0.9,
            "the {label} arm recovered {overlap:.4} of what the exact arm found under cosine"
        );
    }
}
