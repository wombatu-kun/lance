// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A walk that reads only what it touches.
//!
//! One index, three modes, and the interesting comparisons are between them
//! rather than against a number. Two of them are the reference in different
//! ways: [`WalkMode::Coded`] is the same steering with the reading left alone,
//! so a lazy walk that has fetched the wrong bytes shows up as a *different
//! traversal*; [`WalkMode::Exact`] is what the answer is supposed to be, so a
//! lazy walk that steers badly shows up as recall.
//!
//! The pin that carries most of the weight is the first one. At a hop of one
//! vertex the lazy walk is the coded walk - same list, same order, same
//! candidates - so its answer has to be equal to the last bit, and almost every
//! way of getting the lazy read wrong breaks that equality: a neighbour list
//! sliced at the wrong offset, a re-scored distance taken for the wrong row, a
//! candidate list re-scored only down to `k`.

use lance::Dataset;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::query::{SearchParams, VamanaIndex, WalkMode};

mod common;
use common::{DatasetFixture, VECTOR_COLUMN, brute_force, random_vectors, recall};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 4;
const K: usize = 10;
const BEAM: usize = 30;
const QUERIES: usize = 40;
const CODE_BITS: u8 = 3;

/// Partitions a single walk cannot exhaust, because a lazy read of a partition
/// a walk reaches every vertex of has read the partition.
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

/// What a run of queries cost, and how much of the truth it recovered.
struct Measured {
    recall: f64,
    comparisons: f64,
    bytes: f64,
    requests: f64,
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
    params: &SearchParams,
) -> Measured {
    let before = index.io_stats();
    let mut total_recall = 0.0;
    let mut total_comparisons = 0u64;
    for (query, exact) in queries.iter().zip(truth) {
        let result = index.search(query, params).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(found.len(), K, "a query returned the wrong count");
        total_recall += recall(&found, exact);
        total_comparisons += result.comparisons;
    }
    let after = index.io_stats();
    let queries = queries.len() as f64;
    Measured {
        recall: total_recall / queries,
        comparisons: total_comparisons as f64 / queries,
        bytes: (after.bytes_read - before.bytes_read) as f64 / queries,
        requests: (after.requests - before.requests) as f64 / queries,
    }
}

/// The test the whole module rests on.
///
/// A hop of one vertex expands the nearest unexpanded candidate and no other,
/// which is what the whole-partition walk does, so the two walks are the same
/// walk over the same graph with the same distances. Every byte the lazy one
/// fetches is therefore checkable against an answer that was never fetched
/// lazily at all - and the ways of getting a lazy read wrong (a neighbour list
/// read at the wrong offset, a distance credited to the wrong row, a candidate
/// list re-scored only as far as `k`) all show up here as an inequality rather
/// than as a slightly worse recall that could be blamed on the codes.
#[tokio::test]
async fn a_hop_of_one_vertex_is_the_coded_walk_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let narrow = search(WalkMode::Lazy).with_beam_width(1);
    for query in random_vectors(8, 4242) {
        let coded = index
            .search(&query, &search(WalkMode::Coded))
            .await
            .unwrap();
        let lazy = index.search(&query, &narrow).await.unwrap();
        assert_eq!(
            lazy.neighbors, coded.neighbors,
            "a hop of one vertex answered differently from the walk it is supposed to be"
        );
        assert_eq!(
            lazy.comparisons, coded.comparisons,
            "the two walks measured a different number of distances, so they did not walk the \
             same graph"
        );
        assert_eq!(lazy.partitions_read, coded.partitions_read);
    }
}

/// What the mode is for: the same answer off a fraction of the bytes.
///
/// The saving is bounded from below by what stays resident, which at this
/// fixture's `d = 16` is unusually dear - RaBitQ pads its extended code out to
/// sixty-four dimensions, so a code is 38 bytes against a vertex's 136 rather
/// than the 68 against 776 it is at `d = 128`. A fixture that flattered the mode
/// would be the wrong one to guard it with.
#[tokio::test]
async fn a_lazy_walk_reads_a_fraction_of_the_partition() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;

    let mut measured = Vec::new();
    for mode in [WalkMode::Exact, WalkMode::Coded, WalkMode::Lazy] {
        // A fresh index per arm, so that the byte count is a query's and not a
        // query's plus whatever opening the index read.
        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        measured.push(measure(&index, &queries, &truth, &search(mode)).await);
    }
    let (exact, coded, lazy) = (&measured[0], &measured[1], &measured[2]);
    for (label, arm) in [("exact", exact), ("coded", coded), ("lazy", lazy)] {
        println!(
            "{label:<6} recall@{K}={:.4}  {:>8.0} B  {:>6.1} requests  {:>7.0} comparisons",
            arm.recall, arm.bytes, arm.requests, arm.comparisons
        );
    }

    assert!(
        exact.recall > 0.5,
        "the exact arm scored {:.4}, so the fixture is not measuring a working index",
        exact.recall
    );
    assert!(
        lazy.recall > exact.recall - 0.1,
        "the lazy arm scored {:.4} against the exact arm's {:.4}",
        lazy.recall,
        exact.recall
    );
    assert!(
        lazy.bytes < exact.bytes / 2.0,
        "a lazy query read {:.0} bytes against {:.0} read whole, which is not a lazy read",
        lazy.bytes,
        exact.bytes
    );
    // Against the coded arm rather than only against the exact one: the two
    // steer identically, so what is left between them is the reading.
    assert!(
        lazy.bytes < coded.bytes / 2.0,
        "a lazy query read {:.0} bytes against the coded walk's {:.0}",
        lazy.bytes,
        coded.bytes
    );
    // The price of the mode, and it is paid in round trips rather than bytes.
    assert!(
        lazy.requests > exact.requests,
        "a lazy query made {:.1} requests against {:.1} for reading whole, so nothing was \
         fetched a piece at a time",
        lazy.requests,
        exact.requests
    );
}

/// The width is the trade the mode exists to make, so it has to be visible.
///
/// A wider hop fetches the edges of several vertices in one request, which
/// divides the chain of dependent round trips - and expands vertices the
/// strictly greedy order would have reached later or not at all, which costs
/// distances. Both halves are asserted, because a `beam_width` that was quietly
/// ignored would leave recall and comparisons looking perfectly healthy.
#[tokio::test]
async fn a_wider_hop_trades_distances_for_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let queries = random_vectors(QUERIES, 909);
    let truth = ground_truth(&dataset, &queries).await;

    let mut measured = Vec::new();
    for width in [1usize, 8] {
        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        let params = search(WalkMode::Lazy).with_beam_width(width);
        let arm = measure(&index, &queries, &truth, &params).await;
        println!(
            "W={width}: recall@{K}={:.4}, {:.1} requests, {:.0} comparisons",
            arm.recall, arm.requests, arm.comparisons
        );
        measured.push(arm);
    }
    let (narrow, wide) = (&measured[0], &measured[1]);

    assert!(
        wide.requests < narrow.requests,
        "a hop of eight made {:.1} requests against {:.1} for a hop of one, so the width bought \
         no batching",
        wide.requests,
        narrow.requests
    );
    assert!(
        wide.comparisons >= narrow.comparisons,
        "a hop of eight computed {:.0} distances against {:.0} for a hop of one, which would mean \
         a wider hop expands fewer vertices",
        wide.comparisons,
        narrow.comparisons
    );
    assert!(
        wide.recall > narrow.recall - 0.05,
        "a hop of eight scored {:.4} against a hop of one's {:.4}",
        wide.recall,
        narrow.recall
    );
}

/// Deleted rows are walked and not answered, the same as every other mode - and
/// the lazy walk has its own reason to get this wrong, because the row ids it
/// filters by are the one column it reads whole.
#[tokio::test]
async fn a_lazy_walk_answers_only_live_rows() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = coded_dataset(uri).await;
    dataset
        .delete("vec IS NOT NULL AND _rowid % 3 = 0")
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let live = common::live_row_ids(&dataset).await;
    for query in random_vectors(8, 77) {
        let result = index.search(&query, &search(WalkMode::Lazy)).await.unwrap();
        assert_eq!(
            result.neighbors.len(),
            K,
            "fewer than k live rows came back"
        );
        for neighbor in &result.neighbors {
            assert!(
                live.contains(&neighbor.row_addr),
                "row {} was deleted and came back anyway",
                neighbor.row_addr
            );
        }
    }
}

/// The two ways a lazy walk can be asked for something it cannot do.
#[tokio::test]
async fn a_lazy_walk_refuses_what_it_cannot_do() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let query = &random_vectors(1, 1)[0];

    let error = index
        .search(query, &search(WalkMode::Lazy).with_beam_width(0))
        .await
        .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains("beam_width"), "{error}");

    let plain = tempfile::tempdir().unwrap();
    let plain_uri = plain.path().to_str().unwrap();
    let mut uncoded = fixture().write(plain_uri).await;
    let mut without = params();
    without.code_bits = None;
    create_index(&mut uncoded, INDEX_NAME, &without)
        .await
        .unwrap();
    let uncoded = VamanaIndex::open(&uncoded, INDEX_NAME).await.unwrap();

    let error = uncoded
        .search(query, &search(WalkMode::Lazy))
        .await
        .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains("without codes"), "{error}");
}

/// A beam wider than the partition, which is the case the mode is *not* for.
///
/// The walk then reaches every vertex, so it fetches every neighbour list and
/// every vector one scattered row at a time - the worst thing a lazy read can
/// do. It still has to answer correctly, and it still has to answer the same
/// thing, which is what this pins; that it is also slower is the point of the
/// mode having a switch.
#[tokio::test]
async fn a_walk_that_reaches_everything_still_answers() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture {
        fragments: 1,
        rows_per_fragment: 64,
        ..Default::default()
    }
    .write(uri)
    .await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, 1)
            .with_graph_params(BuildParams {
                max_degree: 8,
                search_list_size: 16,
                ..Default::default()
            })
            .with_code_bits(CODE_BITS),
    )
    .await
    .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let params = SearchParams::new(K)
        .with_search_list_size(64)
        .with_nprobes(1);
    let query = &random_vectors(1, 3)[0];
    let mut answers = Vec::new();
    for mode in [WalkMode::Coded, WalkMode::Lazy] {
        let result = index
            .search(query, &params.clone().with_mode(mode))
            .await
            .unwrap();
        answers.push(result.neighbors);
    }
    assert_eq!(
        answers[0], answers[1],
        "a walk that reached the whole partition answered differently when it read it lazily"
    );
    assert_eq!(answers[1].len(), K);
}

/// Several segments, which is the ordinary state of an index that has been
/// appended to, and the one place a lazy walk holds per-partition state that
/// could leak between partitions.
#[tokio::test]
async fn a_lazy_walk_answers_across_segments() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    coded_dataset(uri).await;
    fixture().append(uri).await;
    let mut dataset = Dataset::open(uri).await.unwrap();
    lance_vamana::insert_as_segment(&mut dataset, INDEX_NAME)
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert!(
        index.num_segments() > 1,
        "the append wrote no second segment"
    );

    let queries = random_vectors(QUERIES, 5150);
    let truth = ground_truth(&dataset, &queries).await;
    let exact = measure(&index, &queries, &truth, &search(WalkMode::Exact)).await;
    let lazy = measure(&index, &queries, &truth, &search(WalkMode::Lazy)).await;
    println!(
        "two segments - exact: recall@{K}={:.4}; lazy: recall@{K}={:.4}",
        exact.recall, lazy.recall
    );
    assert!(
        exact.recall > 0.5,
        "the exact arm scored {:.4}",
        exact.recall
    );
    assert!(
        lazy.recall > exact.recall - 0.1,
        "the lazy arm scored {:.4} against the exact arm's {:.4} over two segments",
        lazy.recall,
        exact.recall
    );
}

/// A query the driver refuses before any partition is opened must be refused
/// the same way whichever mode it names.
#[tokio::test]
async fn a_lazy_query_is_validated_like_any_other() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let mut query = random_vectors(1, 8)[0].clone();
    query[3] = f32::NAN;
    let error = index
        .search(&query, &search(WalkMode::Lazy))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("NaN"), "{error}");

    let error = index
        .search(&query[..2], &search(WalkMode::Lazy))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("dimensions"), "{error}");
}
