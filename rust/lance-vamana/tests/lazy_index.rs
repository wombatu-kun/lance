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
//!
//! The second half of the module is the cache, which is the same question asked
//! across queries rather than within one: a walk that keeps a partition's codes
//! must answer exactly what a walk that re-read them answers, whatever the
//! budget does with them in between.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::future::join_all;
use lance::Dataset;
use lance_core::cache::LanceCache;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::codes::{CodeParams, CodeSpec};
use lance_vamana::entry_points::{EntryPointParams, EntryPoints, PartitionEntryPoints};
use lance_vamana::format::VectorSource;
use lance_vamana::inserter::insert_as_segment;
use lance_vamana::io::{read_segment, scan_scheduler};
use lance_vamana::query::{
    Neighbor, QueryResult, SearchParams, VamanaIndex, WalkMode, WalkStart, committed_segments,
};

mod common;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{FixedSizeListArray, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use common::{
    DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, WIDE_DIM, brute_force, random_vectors,
    random_vectors_of, recall, wide_fixture,
};
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::{ColumnAlteration, NewColumnTransform, WriteParams};
use lance_file::version::LanceFileVersion;
use lance_linalg::distance::DistanceType;

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 4;
const K: usize = 10;
const BEAM: usize = 30;
const QUERIES: usize = 40;
const CODE_BITS: u8 = 3;
const MAX_DEGREE: u32 = 16;

/// Partitions a single walk cannot exhaust, because a lazy read of a partition
/// a walk reaches every vertex of has read the partition.
fn fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 4,
        rows_per_fragment: 2048,
        ..Default::default()
    }
}

/// The two kinds a segment can carry. Scalar codes are here for one reason: the
/// resident half of a lazy walk holds whichever store the segment's codes need,
/// and that is a different type for each kind.
const RABIT: CodeSpec = CodeSpec::Rabit {
    num_bits: CODE_BITS,
};
const SCALAR: CodeSpec = CodeSpec::Scalar { num_bits: 8 };

fn params(codes: CodeSpec) -> IndexParams {
    params_on(VECTOR_COLUMN, codes)
}

fn params_on(column: &str, codes: CodeSpec) -> IndexParams {
    IndexParams::new(column, PARTITIONS)
        .with_graph_params(BuildParams {
            max_degree: MAX_DEGREE,
            search_list_size: 64,
            ..Default::default()
        })
        .with_codes(codes)
}

async fn coded_dataset(uri: &str, codes: CodeSpec) -> Dataset {
    let mut dataset = fixture().write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params(codes))
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
async fn a_hop_of_one_vertex_is_the_coded_walk_exactly(codes: CodeSpec) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, codes).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let narrow = search(WalkMode::Lazy).with_beam_width(1);
    for query in random_vectors(8, 4242) {
        let coded = index
            .search(&query, &search(WalkMode::Coded))
            .await
            .unwrap();
        // Both look-aheads, because the query's whole-partition walk,
        // `greedy_search`, is the only code here that never learned about them:
        // it offers its neighbours one at a time where the lazy hop now collects
        // them first, so this is also what pins that restructuring.
        for ahead in [0, 2] {
            let lazy = index
                .search(&query, &narrow.clone().with_prefetch_ahead(ahead))
                .await
                .unwrap();
            assert_eq!(
                lazy.neighbors, coded.neighbors,
                "a hop of one vertex at a look-ahead of {ahead} answered differently from the \
                 walk it is supposed to be"
            );
            assert_eq!(
                lazy.comparisons, coded.comparisons,
                "the two walks measured a different number of distances, so they did not walk \
                 the same graph"
            );
            assert_eq!(lazy.partitions_read, coded.partitions_read);
        }
    }
}

#[tokio::test]
async fn a_rabit_hop_of_one_vertex_is_the_coded_walk_exactly() {
    a_hop_of_one_vertex_is_the_coded_walk_exactly(RABIT).await;
}

/// The same equality over scalar codes.
///
/// Worth its own case rather than trusting the RaBitQ one: the resident half of
/// a lazy walk holds a different store for each kind, and the query it is handed
/// carries a term one kind wants and the other ignores. Both walks reading the
/// same codes by two routes is exactly what this equality pins.
#[tokio::test]
async fn a_scalar_hop_of_one_vertex_is_the_coded_walk_exactly() {
    a_hop_of_one_vertex_is_the_coded_walk_exactly(SCALAR).await;
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
    let dataset = coded_dataset(uri, RABIT).await;
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
    let dataset = coded_dataset(uri, RABIT).await;
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

/// The degenerate margin: at zero, keeping no more than `k`, a walk expands its
/// nearest `k` and stops - which is the walk whose list is `k` long, down to
/// the last distance.
///
/// An equality rather than a recall bar, because the margin rewrites when every
/// lazy walk stops, and a walk that stopped one vertex early or late would
/// still score well. The cap is ten times `k` so that a list ignoring the
/// margin would walk much further, and that is checked as well: without it
/// the equality would also hold for a walk that never read the margin. Edges
/// are held resident, which changes no hop and spares the wide walk a read a
/// vertex.
async fn a_margin_of_zero_is_the_walk_at_k(codes: CodeSpec) {
    const NARROW: usize = 20;
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, codes).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let at_k = SearchParams::new(NARROW)
        .with_nprobes(PARTITIONS as usize)
        .with_mode(WalkMode::Lazy)
        .with_resident_edges(true)
        .with_search_list_size(NARROW)
        .with_rescore_budget(NARROW);
    let capped = at_k.clone().with_search_list_size(10 * NARROW);
    let mut further = 0;
    for width in [1, 4] {
        for query in random_vectors(8, 4343) {
            let fixed = index
                .search(&query, &at_k.clone().with_beam_width(width))
                .await
                .unwrap();
            let ruled = index
                .search(
                    &query,
                    &capped.clone().with_beam_width(width).with_stop_margin(0.0),
                )
                .await
                .unwrap();
            assert_eq!(
                ruled.neighbors, fixed.neighbors,
                "at a width of {width}, a margin of zero answered differently from the walk at \
                 L = k"
            );
            assert_eq!(
                ruled.comparisons, fixed.comparisons,
                "at a width of {width}, a margin of zero measured a different number of distances \
                 from the walk at L = k"
            );
            let unruled = index
                .search(&query, &capped.clone().with_beam_width(width))
                .await
                .unwrap();
            further += usize::from(unruled.comparisons > fixed.comparisons);
        }
    }
    assert!(
        further > 0,
        "the cap never let a walk without the margin go further than L = k, so nothing showed that \
         the margin was what stopped it"
    );
}

#[tokio::test]
async fn a_rabit_margin_of_zero_is_the_walk_at_k() {
    a_margin_of_zero_is_the_walk_at_k(RABIT).await;
}

#[tokio::test]
async fn a_scalar_margin_of_zero_is_the_walk_at_k() {
    a_margin_of_zero_is_the_walk_at_k(SCALAR).await;
}

/// A wider margin never measures fewer distances, query by query, and measures
/// more for some; and at zero it measures exactly what the walk at `L = k`
/// does, though it keeps twice that for the budget.
///
/// One vertex a hop and a cap no list reaches, which is when a wider margin
/// walks on from where a narrower one stopped (`search::tests`): so every
/// probe's count can only grow, and a query's with it. The budget holds the
/// re-score at twenty candidates whatever the walks keep. The equality at zero
/// is what pins the bar to the `k`-th candidate rather than the budget-th: at
/// zero the walk expands its `k` nearest and stops. Scalar codes, because their
/// distances are squared lengths and never below zero; a RaBitQ estimate can
/// be, and below zero a wider margin lowers the bar.
#[tokio::test]
async fn a_wider_margin_never_walks_less() {
    // The whole dataset, so that no partition's list can reach it.
    const CAP: usize = 8192;
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, SCALAR).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let base = search(WalkMode::Lazy)
        .with_beam_width(1)
        .with_resident_edges(true)
        .with_rescore_budget(20);
    let capped = base.clone().with_search_list_size(CAP);
    let mut grew = 0;
    for query in random_vectors(8, 5151) {
        let mut counts = Vec::new();
        for margin in [0.0f32, 0.05, 0.15, 0.4] {
            let result = index
                .search(&query, &capped.clone().with_stop_margin(margin))
                .await
                .unwrap();
            counts.push(result.comparisons);
        }
        assert!(
            counts.windows(2).all(|pair| pair[0] <= pair[1]),
            "margins 0, 0.05, 0.15 and 0.4 measured {counts:?} distances, and a wider one fewer"
        );
        let at_k = index
            .search(&query, &base.clone().with_search_list_size(K))
            .await
            .unwrap();
        assert_eq!(
            counts[0], at_k.comparisons,
            "a margin of zero measured a different number of distances from the walk at L = k"
        );
        grew += usize::from(counts[3] > counts[0]);
    }
    assert!(
        grew > 0,
        "no query measured more at a margin of 0.4 than at 0, so the margin never reached the walk"
    );
}

/// Deleted rows are walked and not answered, the same as every other mode - and
/// the lazy walk has its own reason to get this wrong, because the row ids it
/// filters by are the one column it reads whole.
#[tokio::test]
async fn a_lazy_walk_answers_only_live_rows() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = coded_dataset(uri, RABIT).await;
    dataset
        .delete("vec IS NOT NULL AND _rowid % 3 = 0")
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let live = common::live_row_ids(&dataset).await;
    // The answer before the re-score is held to the same rule, and has to be:
    // it is built from the candidate list, which is where dead vertices are
    // still walked - they carry the edges that keep the graph connected - so
    // nothing upstream of the two answers has dropped them yet.
    let params = search(WalkMode::Lazy).with_report_coded(true);
    for query in random_vectors(8, 77) {
        let result = index.search(&query, &params).await.unwrap();
        for (what, neighbors) in [
            ("the answer", &result.neighbors),
            ("the coded answer", &result.coded_neighbors),
        ] {
            assert_eq!(
                neighbors.len(),
                K,
                "{what}: fewer than k live rows came back"
            );
            for neighbor in neighbors {
                assert!(
                    live.contains(&neighbor.row_addr),
                    "{what}: row {} was deleted and came back anyway",
                    neighbor.row_addr
                );
            }
        }
    }

    // A stop margin with no budget re-scores whatever the list kept, dead
    // vertices included, so a list that kept only its nearest `k` would answer
    // with the live ones among them and come back short. One probe, so that no
    // other partition's candidates can make up the difference.
    let ruled = search(WalkMode::Lazy).with_nprobes(1).with_stop_margin(0.0);
    for query in random_vectors(8, 78) {
        let result = index.search(&query, &ruled).await.unwrap();
        assert_eq!(
            result.neighbors.len(),
            K,
            "a walk stopped by a margin, with no budget, came back short of k live rows"
        );
        for neighbor in &result.neighbors {
            assert!(
                live.contains(&neighbor.row_addr),
                "row {} was deleted and a walk stopped by a margin answered with it",
                neighbor.row_addr
            );
        }
    }
}

/// The ways a lazy walk can be asked for something it cannot do.
#[tokio::test]
async fn a_lazy_walk_refuses_what_it_cannot_do() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
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
    let mut without = params(RABIT);
    without.codes = None;
    create_index(&mut uncoded, INDEX_NAME, &without)
        .await
        .unwrap();
    let uncoded = VamanaIndex::open(&uncoded, INDEX_NAME).await.unwrap();

    // Both modes that steer by codes, because the refusal is the mode's and not
    // the walk's: a scan has no beam to fall back on either.
    for mode in [WalkMode::Lazy, WalkMode::Flat] {
        let error = uncoded.search(query, &search(mode)).await.unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("without codes"), "{error}");
    }

    // A budget is refused rather than ignored by the modes that cannot spend
    // it, for the same reason: a caller who set one is asking about cost.
    for mode in [WalkMode::Exact, WalkMode::Coded] {
        let error = index
            .search(query, &search(mode).with_rescore_budget(BEAM))
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("rescore_budget"), "{error}");
    }

    // And a budget too small to hold the answer, which would otherwise return
    // fewer rows than were asked for and say nothing about it.
    let error = index
        .search(query, &search(WalkMode::Flat).with_rescore_budget(K - 1))
        .await
        .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains("smaller than k"), "{error}");

    // A margin that cannot say how far a walk goes: below zero, or no number.
    // The last is finite, but its (1 + margin)^2 is not.
    for margin in [-0.1, f32::NAN, f32::INFINITY, 1e20] {
        let error = index
            .search(query, &search(WalkMode::Lazy).with_stop_margin(margin))
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("finite fraction"), "{error}");
    }
    // Refused rather than ignored by the walks that do not stop by it, and by
    // the scan, which has no walk to stop.
    for mode in [WalkMode::Exact, WalkMode::Coded, WalkMode::Flat] {
        let error = index
            .search(query, &search(mode).with_stop_margin(0.1))
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("does not stop by it"), "{error}");
    }
    // And a cap that would cut candidates the re-score is owed.
    let error = index
        .search(
            query,
            &search(WalkMode::Lazy)
                .with_rescore_budget(BEAM + 1)
                .with_stop_margin(0.1),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(
        error.to_string().contains("caps each walk's list below"),
        "{error}"
    );
    // A cap exactly as long as the budget holds everything the margin keeps.
    index
        .search(
            query,
            &search(WalkMode::Lazy)
                .with_rescore_budget(BEAM)
                .with_stop_margin(0.1),
        )
        .await
        .unwrap();
    // Where to re-score from, asked of the modes that have no re-score read.
    for mode in [WalkMode::Exact, WalkMode::Coded] {
        let error = index
            .search(query, &search(mode).with_rescore_from_dataset(true))
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains("no re-score read to redirect"),
            "{error}"
        );
    }
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
            .with_codes(CodeSpec::Rabit {
                num_bits: CODE_BITS,
            }),
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

/// The pin the flat arm rests on, and one no graph walk can offer.
///
/// A scan told to keep every vertex of every partition it probes has measured an
/// exact distance against every indexed row, so its answer is the brute-force
/// answer - not close to it, equal to it. Each way of getting a scan wrong lands
/// here as recall below one rather than as a number to argue about: codes read
/// for the wrong partition, a rank mapped to the wrong local id, a candidate
/// re-scored at the wrong position in the batch that came back.
///
/// The distance count is exact for the same reason. A scan is oblivious - it
/// measures every vertex whatever the query is - so the only number it can
/// produce is one per centroid ranked, one per vertex scored and one per
/// candidate re-scored.
#[tokio::test]
async fn a_flat_scan_that_keeps_everything_is_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let rows = fixture().indexed_rows();
    let everything = search(WalkMode::Flat).with_search_list_size(rows);
    for query in random_vectors(8, 1234) {
        let truth = brute_force(&dataset, &query, K).await;
        let result = index.search(&query, &everything).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(
            recall(&found, &truth),
            1.0,
            "a scan that kept every vertex still missed a true neighbour"
        );
        assert_eq!(
            result.comparisons,
            (PARTITIONS as usize + 2 * rows) as u64,
            "a scan measured a different number of distances from one per centroid, one per \
             vertex and one per candidate"
        );
    }

    // The selecting half, which the case above never reaches: a list wider than
    // the partition keeps everything without choosing. Exactly `L` survive each
    // probe, which is what a list truncated to `k` or ordered the wrong way
    // round fails - and the recall floor is what a reversed comparator fails,
    // since it would keep the farthest `L` instead. Both rest on the fixture's
    // partitions being far wider than the beam, which is what it is for.
    let narrow = search(WalkMode::Flat);
    for query in random_vectors(8, 1234) {
        let truth = brute_force(&dataset, &query, K).await;
        let result = index.search(&query, &narrow).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert!(
            recall(&found, &truth) >= 0.9,
            "a scan keeping the nearest {BEAM} of every partition scored {:.4}",
            recall(&found, &truth)
        );
        assert_eq!(
            result.comparisons,
            (PARTITIONS as usize + rows + result.partitions_read * BEAM) as u64,
            "a scan kept a different number of candidates from {BEAM} a probe"
        );
    }
}

/// The same pin over two segments, which is the ordinary state of an index that
/// has been appended to.
///
/// Worth its own case rather than a wider `nprobes` on the one above, because
/// what it can catch is different: a scan turns a rank into a local id and a
/// local id into a row id, and every segment of an index has a partition 0 and a
/// vertex 0. Codes taken from one segment beside row ids from another produce
/// plausible answers, and only an exact one shows it.
#[tokio::test]
async fn a_flat_scan_over_two_segments_is_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    coded_dataset(uri, RABIT).await;
    fixture().append(uri).await;
    let mut dataset = Dataset::open(uri).await.unwrap();
    lance_vamana::insert_as_segment(&mut dataset, INDEX_NAME)
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let segments = index.num_segments();
    assert!(segments > 1, "the append wrote no second segment");

    // Every row of both segments, since `nprobes` is per segment and the list is
    // wider than any partition either of them holds.
    let rows = segments * fixture().indexed_rows();
    let everything = search(WalkMode::Flat).with_search_list_size(rows);
    for query in random_vectors(8, 606) {
        let truth = brute_force(&dataset, &query, K).await;
        let result = index.search(&query, &everything).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(
            recall(&found, &truth),
            1.0,
            "a scan of every vertex of two segments missed a true neighbour"
        );
        assert_eq!(
            result.comparisons,
            (segments * PARTITIONS as usize + 2 * rows) as u64,
            "the centroids of both segments and every vertex of both, once each"
        );
    }
}

/// What the mode is for: the same partitions, without their graph.
///
/// Neither arm caches, so both pay for the codes of every partition they probe
/// on every query and what separates them is only what each *chooses* to fetch:
/// the vectors of the candidate list for both, plus the out-edges of every
/// vertex the walk expanded. A scan never opens `__neighbors`, so it has to read
/// strictly less.
///
/// The distances go the other way, by a factor the count alone overstates: a
/// scan's are measured in one batched call over the quantiser's block layout at
/// 16.8 ns each, a walk's one at a time at 40.0, so the column is a ratio of
/// work rather than of time (`examples/expansion_gate.rs`).
#[tokio::test]
async fn a_flat_scan_reads_no_edges() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 8888);
    let truth = ground_truth(&dataset, &queries).await;

    let mut measured = Vec::new();
    for mode in [WalkMode::Lazy, WalkMode::Flat] {
        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        measured.push(measure(&index, &queries, &truth, &search(mode)).await);
    }
    let (lazy, flat) = (&measured[0], &measured[1]);
    for (label, arm) in [("lazy", lazy), ("flat", flat)] {
        println!(
            "{label:<6} recall@{K}={:.4}  {:>8.0} B  {:>6.1} requests  {:>7.0} comparisons",
            arm.recall, arm.bytes, arm.requests, arm.comparisons
        );
    }

    assert!(
        flat.bytes < lazy.bytes,
        "a scan read {:.0} bytes against the walk's {:.0}, so it fetched something the walk did \
         and it should have fetched no edges at all",
        flat.bytes,
        lazy.bytes
    );
    assert!(
        flat.comparisons > lazy.comparisons,
        "a scan measured {:.0} distances against the walk's {:.0}, so it did not score the whole \
         partition",
        flat.comparisons,
        lazy.comparisons
    );
    // Not an equality and not a strict improvement either. A scan keeps the `L`
    // nearest of the partition by coded distance where a walk keeps the `L` it
    // found, so the walk's list can hold a true neighbour the codes ranked
    // outside the scan's - rarely, and never often enough to make the scan the
    // worse arm.
    assert!(
        flat.recall > lazy.recall - 0.01,
        "a scan that considered every vertex scored {:.4} against a walk's {:.4}",
        flat.recall,
        lazy.recall
    );
}

/// The degenerate budget: one wide enough for every candidate changes nothing.
///
/// The two-pass shape is a rewrite of the path every lazy query takes, so the
/// case where the budget decides nothing has to come back exactly - the same
/// rows in the same order, the same distance count, the same partitions read. A
/// recall bar would pass through a re-scoring that quietly dropped half of every
/// list, and both modes go through the same rewrite, so both are checked.
#[tokio::test]
async fn a_budget_wider_than_the_candidates_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let rows = fixture().indexed_rows();

    for mode in [WalkMode::Lazy, WalkMode::Flat] {
        for query in random_vectors(8, 77) {
            let unbudgeted = index.search(&query, &search(mode)).await.unwrap();
            // Every candidate every probe kept, and then two numbers past it.
            // The first is the boundary the allocation short-circuits on.
            let spent = unbudgeted.partitions_read * BEAM;
            for budget in [spent, spent + 1, rows] {
                let budgeted = index
                    .search(&query, &search(mode).with_rescore_budget(budget))
                    .await
                    .unwrap();
                let what = format!("{mode:?}, budget {budget}");
                assert_eq!(budgeted.neighbors, unbudgeted.neighbors, "{what}");
                assert_eq!(budgeted.comparisons, unbudgeted.comparisons, "{what}");
                assert_eq!(
                    budgeted.partitions_read, unbudgeted.partitions_read,
                    "{what}"
                );
            }
        }
    }
}

/// What the budget is for: the same recall off a fraction of the strides.
///
/// A scan reads nothing but its candidates, and its distance count says how many
/// of them there were - one per centroid ranked, one per vertex scored, one per
/// candidate re-scored. With no budget that last term is `L` a probe whatever
/// the probes turned out to be worth; with one it is the budget itself, and the
/// equality below is what a budget spent per partition rather than per query
/// fails.
///
/// The claim it exists to pin is the second half: a budget of `L` for the whole
/// query clears the recall bar that `L` *per probe* was set for, having read a
/// fraction of the rows and skipped whole probes on the way. The partitions are
/// still all read - a budget decides what is fetched to correct a candidate, not
/// what is probed - so `partitions_read` may not move.
#[tokio::test]
async fn a_budget_spends_the_strides_where_they_are_worth_most() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let rows = fixture().indexed_rows();

    for query in random_vectors(8, 4242) {
        let plain = index.search(&query, &search(WalkMode::Flat)).await.unwrap();
        let spent = plain.partitions_read * BEAM;
        assert_eq!(
            plain.comparisons,
            (PARTITIONS as usize + rows + spent) as u64,
            "an unbudgeted scan re-scored something other than {BEAM} a probe"
        );
        for budget in [spent - 1, spent / 2, BEAM, K] {
            let result = index
                .search(&query, &search(WalkMode::Flat).with_rescore_budget(budget))
                .await
                .unwrap();
            assert_eq!(
                result.comparisons,
                (PARTITIONS as usize + rows + budget) as u64,
                "a budget of {budget} re-scored a different number of candidates"
            );
            assert_eq!(result.partitions_read, plain.partitions_read);
        }
    }

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let unbudgeted = measure(&index, &queries, &truth, &search(WalkMode::Flat)).await;
    let budgeted = measure(
        &index,
        &queries,
        &truth,
        &search(WalkMode::Flat).with_rescore_budget(BEAM),
    )
    .await;
    println!(
        "unbudgeted recall@{K}={:.4}  {:>8.0} B  {:>6.1} requests\n\
         budgeted   recall@{K}={:.4}  {:>8.0} B  {:>6.1} requests",
        unbudgeted.recall,
        unbudgeted.bytes,
        unbudgeted.requests,
        budgeted.recall,
        budgeted.bytes,
        budgeted.requests
    );
    assert!(
        budgeted.requests < unbudgeted.requests,
        "a budget of {BEAM} for the whole query made {:.1} requests against the {:.1} of {BEAM} a \
         probe, so no probe was left with nothing to fetch",
        budgeted.requests,
        unbudgeted.requests
    );
    assert!(
        budgeted.bytes < unbudgeted.bytes,
        "a budget of {BEAM} read {:.0} bytes against {:.0}",
        budgeted.bytes,
        unbudgeted.bytes
    );
    assert!(
        budgeted.recall >= 0.9,
        "a budget of {BEAM} for the whole query scored {:.4}, where {BEAM} a probe scored {:.4}",
        budgeted.recall,
        unbudgeted.recall
    );
}

/// Several segments, which is the ordinary state of an index that has been
/// appended to, and the one place a lazy walk holds per-partition state that
/// could leak between partitions.
#[tokio::test]
async fn a_lazy_walk_answers_across_segments() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    coded_dataset(uri, RABIT).await;
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

/// A budget wide enough to hold this fixture many times over, for the arms
/// where what is being asked is what a cache does when it is not evicting.
const BUDGET: usize = 64 << 20;

/// Every answer of a run of queries, and what the run cost.
///
/// The answers as well as the cost because the two questions a cache raises are
/// asked of the same run: whether it changed what came back, and whether it
/// changed what was read to produce it.
async fn replay(
    index: &VamanaIndex,
    queries: &[Vec<f32>],
    params: &SearchParams,
) -> (Vec<QueryResult>, Cost) {
    let before = index.io_stats();
    let mut answers = Vec::with_capacity(queries.len());
    for query in queries {
        answers.push(index.search(query, params).await.unwrap());
    }
    let after = index.io_stats();
    let queries = queries.len() as f64;
    (
        answers,
        Cost {
            bytes: (after.bytes_read - before.bytes_read) as f64 / queries,
            requests: (after.requests - before.requests) as f64 / queries,
        },
    )
}

struct Cost {
    bytes: f64,
    requests: f64,
}

/// Two runs that are supposed to be the same run.
///
/// Down to the comparison count, not just the rows: a cache that served the
/// wrong partition's codes would steer a walk somewhere else and still return
/// `k` plausible rows, because the answer is re-scored exactly whatever the walk
/// looked at on the way.
fn assert_same(left: &[QueryResult], right: &[QueryResult], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: different runs");
    for (query, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left.neighbors, right.neighbors,
            "{what}: query {query} answered differently"
        );
        assert_eq!(
            left.comparisons, right.comparisons,
            "{what}: query {query} walked a different graph"
        );
        assert_eq!(
            left.partitions_read, right.partitions_read,
            "{what}: query {query} read a different number of partitions"
        );
    }
}

async fn cached(dataset: &Dataset, budget: usize) -> VamanaIndex {
    VamanaIndex::open(dataset, INDEX_NAME)
        .await
        .unwrap()
        .with_cache(LanceCache::with_capacity(budget))
}

/// A cache changes what a query reads and must change nothing else.
///
/// All three modes, because they take the cache in different amounts: the
/// whole-partition walks keep the file they have opened before and the layout
/// that came with it, while a lazy walk also keeps what it steers by. Both runs
/// of the cached arm are compared, so the answer is pinned against the read that
/// populated the cache as well as against the index that has none.
///
/// The guard is on bytes and not on cache hits. An index that holds a partition
/// file holds its footer inside it, so a second query that probes the same
/// partition never looks the layout up again - a whole-partition walk can
/// therefore be perfectly warm with a hit count of zero. What warmth actually
/// claims is that the second pass re-read less, and that is what is asserted.
#[tokio::test]
async fn a_cache_does_not_change_an_answer() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 606);

    for mode in [WalkMode::Exact, WalkMode::Coded, WalkMode::Lazy] {
        let params = search(mode);
        let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        let (uncached, _) = replay(&plain, &queries, &params).await;

        let index = cached(&dataset, BUDGET).await;
        let (cold, cold_cost) = replay(&index, &queries, &params).await;
        let (warm, warm_cost) = replay(&index, &queries, &params).await;

        assert_same(&uncached, &cold, &format!("{mode:?}, first pass"));
        assert_same(&uncached, &warm, &format!("{mode:?}, second pass"));
        assert!(
            warm_cost.bytes < cold_cost.bytes,
            "{mode:?}: the warm pass re-read as much as the cold one, {:.0} bytes a query \
             against {:.0}, so the equality above proves nothing",
            warm_cost.bytes,
            cold_cost.bytes
        );
        assert!(
            index.cache_stats().await.unwrap().num_entries > 0,
            "{mode:?}: the cache holds nothing, so nothing was shared through it"
        );
    }
}

/// What the cache is for: a query that has probed a partition before pays for
/// the rows its walk touches and nothing else.
///
/// Against an index with no cache and not against the run that filled the cache,
/// because a pass of forty queries is one cold query and thirty-nine warm ones -
/// the average over it is already most of the way to the answer, and comparing
/// two such passes measures nothing.
///
/// The saving is stated in bytes of code column rather than as a ratio, which is
/// what makes it a fact about the mode instead of a fact about this fixture:
/// what a cache removes is exactly the part of the read that is proportional to
/// the partition. The floor is not zero and is not meant to be. A lazy walk
/// still fetches the out-edges of every vertex it expands and the vectors of the
/// candidates it ends with, and which rows those are is a property of the query.
#[tokio::test]
async fn a_cache_removes_the_read_the_walk_does_not_choose() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 4242);
    let params = search(WalkMode::Lazy);

    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let (_, uncached) = replay(&plain, &queries, &params).await;

    let index = cached(&dataset, BUDGET).await;
    let (_, cold) = replay(&index, &queries, &params).await;
    let (_, warm) = replay(&index, &queries, &params).await;
    let stats = index.cache_stats().await.unwrap();
    println!(
        "no cache: {:.0} B, {:.1} requests; cached: {:.0} B, {:.1} requests; {} entries, {} B held",
        uncached.bytes,
        uncached.requests,
        warm.bytes,
        warm.requests,
        stats.num_entries,
        stats.size_bytes
    );

    assert!(
        uncached.bytes - warm.bytes > code_column_bytes() as f64,
        "a cached query read {:.0} bytes against {:.0} with no cache, a saving of {:.0} against a \
         code column of {}, so the codes were read again",
        warm.bytes,
        uncached.bytes,
        uncached.bytes - warm.bytes,
        code_column_bytes()
    );
    // The round trips as well as the bytes: the codes are one read of a
    // partition and its footer is another, and a cache that saved the first
    // without the second would leave the walk waiting for a layout it already
    // knows.
    assert!(
        warm.requests < uncached.requests,
        "a cached query made {:.1} requests against {:.1} with no cache",
        warm.requests,
        uncached.requests
    );
    assert!(
        warm.bytes > 0.0,
        "a warm query read nothing at all, so the walk is not fetching its own edges"
    );
    assert!(
        cold.bytes > warm.bytes,
        "the pass that filled the cache read no more than the pass that used it"
    );
}

/// What a query's codes weigh on disk: every partition of the one segment is
/// probed, so the rows behind them are the rows of the dataset.
/// What `__neighbors` weighs across the whole index: a fixed stride of
/// `max_degree` slots a vertex, so it does not depend on how the rows fell into
/// partitions.
fn edge_column_bytes() -> usize {
    fixture().fragments * fixture().rows_per_fragment * MAX_DEGREE as usize * size_of::<u32>()
}

fn code_column_bytes() -> usize {
    let dimension = VECTOR_DIM as u32;
    let stride = CodeParams::rabit(CODE_BITS, dimension)
        .unwrap()
        .stride(dimension)
        .unwrap() as usize;
    fixture().fragments * fixture().rows_per_fragment * stride
}

/// The two halves of what names a partition, and the fixture that makes both of
/// them load-bearing.
///
/// An index that has been appended to holds several segments, each with its own
/// partition 0. Once more partitions are probed than a segment has, some
/// partition id is shared by two of them - so a cache keyed by the id alone
/// would answer one segment's walk with the other's row ids, and one keyed by
/// the segment alone would answer every partition with the first one's. The
/// entry count is what pins it: one entry per partition and one per file, and
/// either mistake collapses pairs of them into one.
#[tokio::test]
async fn two_segments_do_not_share_a_cache_entry() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    coded_dataset(uri, RABIT).await;
    fixture().append(uri).await;
    let mut dataset = Dataset::open(uri).await.unwrap();
    lance_vamana::insert_as_segment(&mut dataset, INDEX_NAME)
        .await
        .unwrap();

    let queries = random_vectors(QUERIES, 5150);
    let params = search(WalkMode::Lazy);
    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert!(
        plain.num_segments() > 1,
        "the append wrote no second segment"
    );
    let (uncached, _) = replay(&plain, &queries, &params).await;

    let index = cached(&dataset, BUDGET).await;
    let (cold, _) = replay(&index, &queries, &params).await;
    let (warm, _) = replay(&index, &queries, &params).await;
    assert_same(&uncached, &cold, "two segments, first pass");
    assert_same(&uncached, &warm, "two segments, second pass");

    let probed = uncached[0].partitions_read;
    assert!(
        probed > PARTITIONS as usize,
        "{probed} partitions were probed across {} segments, so no partition id is shared by two \
         of them and this fixture cannot tell the keys apart",
        plain.num_segments()
    );
    assert_eq!(
        index.cache_stats().await.unwrap().num_entries,
        2 * probed,
        "{probed} probed partitions should hold one entry of codes and one of layout each"
    );
}

/// A budget that cannot hold one partition, which is the deployment the lazy
/// read exists for taken to its limit.
///
/// Everything is evicted before the next query, so every query re-reads every
/// partition - and has to answer exactly what it would have answered with room
/// to spare. This is the path where a cached `Arc` is dropped between the read
/// and the next use of it, which is the one way a caching bug can look like a
/// memory bug rather than a wrong answer.
#[tokio::test]
async fn a_budget_too_small_for_a_partition_still_answers() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(16, 31337);
    let params = search(WalkMode::Lazy);

    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let (uncached, _) = replay(&plain, &queries, &params).await;

    // Eviction is not immediate. Moka admits an entry whatever it weighs and
    // reclaims it when it next runs its housekeeping, and a quick enough run of
    // queries outpaces that entirely: left to itself, two passes here once made
    // 132 lookups and missed 15 of them. So the housekeeping is run after every
    // query - `size` runs it before counting - which is what makes the budget
    // bind between queries rather than whenever the cache gets round to it.
    let budget = LanceCache::with_capacity(1024);
    let index = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .with_cache(budget.clone());
    let mut passes = Vec::new();
    for _ in 0..2 {
        let mut answers = Vec::with_capacity(queries.len());
        for query in &queries {
            answers.push(index.search(query, &params).await.unwrap());
            budget.size().await;
        }
        passes.push(answers);
    }
    assert_same(&uncached, &passes[0], "a budget of 1 KiB, first pass");
    assert_same(&uncached, &passes[1], "a budget of 1 KiB, second pass");

    // Hits rather than misses, because misses count the first look at each
    // file's layout too, and four of those would cover for four partitions
    // served from what the budget could not hold.
    let stats = index.cache_stats().await.unwrap();
    assert_eq!(
        stats.hits,
        0,
        "a budget of 1 KiB served {} of {} lookups from what it could not hold",
        stats.hits,
        stats.hits + stats.misses
    );
    let rereads = (passes.len() * queries.len() * PARTITIONS as usize) as u64;
    assert!(
        stats.misses >= rereads,
        "{} lookups over two passes of {} queries probing {PARTITIONS} partitions are fewer \
         than one a partition a query",
        stats.misses,
        queries.len()
    );
}

/// An index holds nothing it was not given a budget for.
///
/// The default matters more than it looks: a cache that arrived switched on
/// would grow a server's resident set by the size of every partition it ever
/// probed, and it would do it without appearing in any allocation the caller
/// makes. It is also what every measurement of the mode is taken against, so
/// "no cache" has to mean no hits at all rather than few - which is why the
/// index holds no cache rather than an empty one.
#[tokio::test]
async fn an_index_holds_nothing_unless_it_is_given_a_cache() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(16, 2718);
    let params = search(WalkMode::Lazy);

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let (_, first) = replay(&index, &queries, &params).await;
    let (_, second) = replay(&index, &queries, &params).await;

    assert!(
        index.cache_stats().await.is_none(),
        "an index nobody gave a cache reports one"
    );
    assert_eq!(
        second.bytes, first.bytes,
        "a second pass read a different number of bytes, so something was kept between them"
    );
}

/// One read of a partition however many queries want it at once.
///
/// A server answers queries concurrently, and the moment an index is opened is
/// exactly the moment they all miss. Without single-flight loading the first
/// wave would read every partition once per query in flight - the largest read
/// the mode makes, multiplied by the concurrency - so the exact count is pinned
/// rather than a ratio.
#[tokio::test(flavor = "multi_thread")]
async fn a_partition_is_read_once_however_many_queries_want_it() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(8, 1234);
    let params = search(WalkMode::Lazy);

    let index = Arc::new(cached(&dataset, BUDGET).await);
    let together = join_all(
        queries
            .iter()
            .map(|query| index.search(query, &params))
            .collect::<Vec<_>>(),
    )
    .await
    .into_iter()
    .map(Result::unwrap)
    .collect::<Vec<_>>();

    let stats = index.cache_stats().await.unwrap();
    assert_eq!(
        stats.misses as usize, stats.num_entries,
        "{} lookups missed for {} entries, so a partition was read more than once",
        stats.misses, stats.num_entries
    );

    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let (alone, _) = replay(&plain, &queries, &params).await;
    assert_same(&alone, &together, "eight queries at once");
}

/// The budget is spent on what is held, which is not what was read.
///
/// A partition's codes are one contiguous stride a vertex on disk and are read
/// back into the seven columns Lance's estimator wants, so the resident form is
/// larger than the bytes it came from - and a budget in on-disk bytes would hold
/// a fraction of what it was asked to. The accounting is `DeepSizeOf`'s rather
/// than ours, so this pins that it is being asked at all.
#[tokio::test]
async fn the_budget_counts_the_resident_form_and_not_the_read_one() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let index = cached(&dataset, BUDGET).await;
    let params = search(WalkMode::Lazy);
    let result = index
        .search(&random_vectors(1, 88)[0], &params)
        .await
        .unwrap();

    let on_disk = code_column_bytes();
    let held = index.cache_stats().await.unwrap().size_bytes;
    println!(
        "{} partitions: {on_disk} B of codes read, {held} B held",
        result.partitions_read
    );

    assert!(
        held > on_disk,
        "the cache reports {held} bytes for codes that are {on_disk} bytes on disk, so the budget \
         is being spent in the wrong units"
    );
    assert!(
        held < 4 * on_disk,
        "the cache reports {held} bytes for {on_disk} bytes of codes, which is more than the \
         resident form should cost"
    );
}

/// A query the driver refuses before any partition is opened must be refused
/// the same way whichever mode it names.
#[tokio::test]
async fn a_lazy_query_is_validated_like_any_other() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
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

/// Asking for a code before the distance that reads it is a hint, and a hint
/// cannot change an answer.
///
/// Every depth against zero: one, the default two, a depth past any hop this
/// fixture produces, and `usize::MAX`. Both ways of reaching the edges, because
/// a hop collects its ids from a different place in each.
///
/// Scalar codes, and that is the whole reason this case exists rather than the
/// RaBitQ one: `ScalarQuantizationStorage` is the only code store this crate can
/// hold that implements `prefetch` at all, so on RaBitQ every depth would be a call
/// that returns and the case would be asserting that nothing changes nothing.
///
/// What it cannot pin, unlike `resident_edges_do_not_change_an_answer` below, is
/// that the depth arrived: a look-ahead leaves no trace in any counter, so a
/// walk that ignored the knob would pass this too. The knob reaching the hop is
/// read off `query.rs` and off `offer_all`'s own cases, not off this.
#[tokio::test]
async fn a_look_ahead_does_not_change_an_answer() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, SCALAR).await;
    let queries = random_vectors(QUERIES, 2129);

    for resident in [false, true] {
        let index = cached(&dataset, BUDGET).await;
        let none = search(WalkMode::Lazy)
            .with_resident_edges(resident)
            .with_prefetch_ahead(0);
        let (without, _) = replay(&index, &queries, &none).await;
        for ahead in [1, 2, 64, usize::MAX] {
            let (with, _) =
                replay(&index, &queries, &none.clone().with_prefetch_ahead(ahead)).await;
            assert_same(
                &without,
                &with,
                &format!("a look-ahead of {ahead}, resident edges {resident}"),
            );
        }
    }
}

/// Holding `__neighbors` across queries changes what a walk reads and must
/// change nothing else: the same hops in the same order over the same graph, and
/// so the same answer to the last bit.
///
/// The requests are pinned beside the equality because the equality alone would
/// hold just as well if the flag had done nothing at all.
#[tokio::test]
async fn resident_edges_do_not_change_an_answer() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 1717);
    let fetching = search(WalkMode::Lazy);
    let holding = search(WalkMode::Lazy).with_resident_edges(true);

    let index = cached(&dataset, BUDGET).await;
    replay(&index, &queries, &fetching).await;
    let (fetched, fetched_cost) = replay(&index, &queries, &fetching).await;

    let index = cached(&dataset, BUDGET).await;
    replay(&index, &queries, &holding).await;
    let (held, held_cost) = replay(&index, &queries, &holding).await;

    assert_same(&fetched, &held, "resident edges");
    assert!(
        held_cost.requests < fetched_cost.requests,
        "a walk holding the edges made {:.1} requests against {:.1} fetching them, so the column \
         was fetched either way and the equality above pins nothing",
        held_cost.requests,
        fetched_cost.requests
    );
}

/// The two flavours of a resident partition share a segment and a partition id,
/// so the key has to carry which of them it holds.
///
/// Without that, the entry a fetching walk left behind answers one that wanted
/// the edges held, and that walk silently goes back to fetching a hop at a time
/// - with a cache hit recorded, the right answer returned and nothing at all to
/// see in the run.
#[tokio::test]
async fn a_partition_held_without_its_edges_does_not_answer_a_walk_that_wants_them() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 606);
    let fetching = search(WalkMode::Lazy);
    let holding = search(WalkMode::Lazy).with_resident_edges(true);

    let index = cached(&dataset, BUDGET).await;
    replay(&index, &queries, &fetching).await;
    let (_, fetched_cost) = replay(&index, &queries, &fetching).await;
    replay(&index, &queries, &holding).await;
    let (_, held_cost) = replay(&index, &queries, &holding).await;

    assert!(
        held_cost.requests < fetched_cost.requests,
        "after a pass that fetched its edges, a walk asking to hold them still made {:.1} \
         requests against {:.1}, so it was handed the partition without them",
        held_cost.requests,
        fetched_cost.requests
    );
}

/// What holding them costs, in the units the budget is spent in.
///
/// Exactly the column rather than about it: `__neighbors` is a fixed stride of
/// `max_degree` slots a vertex, so the resident form has no per-vertex overhead
/// to hide behind, and a difference that is not the column means something else
/// was held or something was evicted.
#[tokio::test]
async fn holding_the_edges_costs_exactly_the_edge_column() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(2, 88);

    let fetching = cached(&dataset, BUDGET).await;
    replay(&fetching, &queries, &search(WalkMode::Lazy)).await;
    let holding = cached(&dataset, BUDGET).await;
    replay(
        &holding,
        &queries,
        &search(WalkMode::Lazy).with_resident_edges(true),
    )
    .await;

    let without = fetching.cache_stats().await.unwrap();
    let with = holding.cache_stats().await.unwrap();
    println!(
        "{} entries holding {} B, against {} entries holding {} B",
        with.num_entries, with.size_bytes, without.num_entries, without.size_bytes
    );

    assert_eq!(
        with.num_entries, without.num_entries,
        "the two arms hold a different number of entries, so their sizes are not comparable"
    );
    assert_eq!(
        with.size_bytes - without.size_bytes,
        edge_column_bytes(),
        "holding the edges cost {} bytes where the column is {}",
        with.size_bytes - without.size_bytes,
        edge_column_bytes()
    );
}

/// Sum the two halves of a run of queries, so that a phase split can be held
/// against what the scheduler counted for the same run.
fn phases(answers: &[QueryResult]) -> (u64, u64, u64, u64) {
    answers
        .iter()
        .fold((0, 0, 0, 0), |(sb, sr, rb, rr), answer| {
            (
                sb + answer.search.bytes_read,
                sr + answer.search.requests,
                rb + answer.rescore.bytes_read,
                rr + answer.rescore.requests,
            )
        })
}

/// Whether an `RWF_NOWAIT` read of a byte just written where the tests keep
/// their datasets is served.
///
/// Asked of a file written there for the purpose and never through the index,
/// so that an index which stopped reading in place cannot talk the test into
/// skipping the assertion that would catch it. The byte was just written, so
/// the page cache holds it: anything but a served read means a filesystem,
/// kernel or sandbox that refuses the flag.
#[cfg(target_os = "linux")]
fn serves_without_waiting() -> bool {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("probe");
    std::fs::write(&path, b"x").unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mut byte = [0u8; 1];
    matches!(
        rustix::io::preadv2(
            &file,
            &mut [std::io::IoSliceMut::new(&mut byte)],
            0,
            rustix::io::ReadWriteFlags::NOWAIT,
        ),
        Ok(1)
    )
}

/// Every byte a query reads belongs to exactly one of its two phases.
///
/// The two counts come from different places and must agree exactly: one is the
/// index's own scheduler over the whole run, the other is a sink each query
/// attached to the files it opened. A read the split does not see - a footer, a
/// projection built off the wrong handle, a path that opens a file of its own -
/// is invisible in every other column of every measurement this crate takes, and
/// shows up here as an inequality.
///
/// Both with a cache and without, because the two read different things: an
/// uncached query re-reads the codes of every partition it probes, a cached one
/// re-reads none of them, and only the second is the state a measurement is
/// taken in.
#[tokio::test]
async fn the_phases_of_a_query_add_up_to_what_it_read() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 4242);
    let params = search(WalkMode::Lazy)
        .with_rescore_budget(K)
        .with_report_coded(true);

    let bare = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let (answers, cost) = replay(&bare, &queries, &params).await;
    let (search_bytes, _, rescore_bytes, _) = phases(&answers);
    assert_eq!(
        search_bytes + rescore_bytes,
        (cost.bytes * queries.len() as f64) as u64,
        "an uncached run read {} bytes but accounts for {} + {}",
        cost.bytes * queries.len() as f64,
        search_bytes,
        rescore_bytes
    );

    let warm = cached(&dataset, BUDGET).await;
    replay(&warm, &queries, &params).await;
    let reads_before = warm.rescore_reads();
    let (answers, cost) = replay(&warm, &queries, &params).await;
    let reads_after = warm.rescore_reads();
    let (search_bytes, _, rescore_bytes, _) = phases(&answers);
    assert_eq!(
        search_bytes + rescore_bytes,
        (cost.bytes * queries.len() as f64) as u64,
        "a cached run read {} bytes but accounts for {} + {}",
        cost.bytes * queries.len() as f64,
        search_bytes,
        rescore_bytes
    );
    // The direction, not just the sum: a split that credited everything to one
    // phase would satisfy the equality above and say nothing.
    assert!(
        rescore_bytes > 0,
        "a cached run re-scored {} candidates a query and read nothing to do it",
        K
    );

    // Every re-score read of a cached index on local storage is one it made off
    // its own descriptor, on the thread that asked or on the blocking pool, so
    // the split of them has to add up to what the queries were charged.
    #[cfg(unix)]
    {
        let reads = reads_after.since(&reads_before);
        let charged = answers
            .iter()
            .map(|answer| answer.rescore.iops)
            .sum::<u64>();
        assert_eq!(
            reads.in_place + reads.handed_off,
            charged,
            "the re-scores split their reads as {reads:?}, but the queries were charged {charged}"
        );
        // And on a warm index, where the filesystem can be asked, none of them
        // left the thread that asked.
        #[cfg(target_os = "linux")]
        if serves_without_waiting() {
            assert_eq!(
                (reads.handed_off, reads.trips),
                (0, 0),
                "a warm index handed re-score reads to the blocking pool: {reads:?} of {charged}"
            );
        }
    }

    // The same question of the clocks, and the only form of it that is not a
    // flaky one: the two phases are disjoint stretches of one call, so together
    // they cannot outlast the call. A second phase timed from the start of the
    // first would double-count and break this by a factor of nearly two, while
    // no amount of scheduler noise can.
    let started = std::time::Instant::now();
    let answer = warm.search(&queries[0], &params).await.unwrap();
    let whole = started.elapsed();
    assert!(
        answer.search.elapsed + answer.rescore.elapsed <= whole,
        "the phases of one query took {:?} and {:?}, which is more than the {:?} the call took",
        answer.search.elapsed,
        answer.rescore.elapsed,
        whole
    );
}

/// With the codes and the edges resident, a walk reads nothing at all until it
/// re-scores - and what it then reads is set by the budget rather than by the
/// queue.
///
/// This is the claim every byte figure this crate quotes rests on, and it is the
/// one the split exists to make checkable. The arm with the edges on disk is
/// measured beside it because the equality of the two answers is what says the
/// difference between them is a read and not a different walk.
#[tokio::test]
async fn a_walk_that_holds_its_edges_reads_nothing_until_it_re_scores() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 909);
    let fetching = search(WalkMode::Lazy).with_rescore_budget(K);
    let holding = fetching.clone().with_resident_edges(true);

    let index = cached(&dataset, BUDGET).await;
    replay(&index, &queries, &holding).await;
    let (held, _) = replay(&index, &queries, &holding).await;
    let (search_bytes, search_requests, rescore_bytes, _) = phases(&held);
    assert_eq!(
        (search_bytes, search_requests),
        (0, 0),
        "a warm walk holding its edges still read {search_bytes} bytes in \
         {search_requests} requests before re-scoring"
    );
    assert!(rescore_bytes > 0, "the run re-scored nothing");

    let index = cached(&dataset, BUDGET).await;
    replay(&index, &queries, &fetching).await;
    let (fetched, _) = replay(&index, &queries, &fetching).await;
    let (fetched_search, fetched_requests, fetched_rescore, _) = phases(&fetched);
    assert!(
        fetched_search > 0 && fetched_requests > 0,
        "a warm walk fetching its edges read {fetched_search} bytes in {fetched_requests} \
         requests, so the two arms are not different reads at all"
    );

    assert_same(&held, &fetched, "resident edges");
    assert_eq!(
        rescore_bytes, fetched_rescore,
        "the same candidates cost {rescore_bytes} bytes to correct one way and \
         {fetched_rescore} the other, so the split is charging the edges to the re-score"
    );
}

/// The answer before the vectors were read is a different answer, and asking for
/// it changes nothing about the one the query returns.
///
/// Recall cannot make this case: a coded ordering and an exact one over the same
/// candidates differ by a permutation that recall is nearly blind to, and a
/// `coded_neighbors` that quietly held the *rescored* answer would score exactly
/// the same. So the claim is set inequality on a seeded fixture, pinned beside
/// the two ways it can come back empty.
#[tokio::test]
async fn the_coded_answer_is_the_answer_before_the_vectors_were_read() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, RABIT).await;
    let queries = random_vectors(QUERIES, 31337);
    let asked = search(WalkMode::Lazy)
        .with_rescore_budget(K * 4)
        .with_report_coded(true);
    let index = cached(&dataset, BUDGET).await;
    replay(&index, &queries, &asked).await;
    let (answers, _) = replay(&index, &queries, &asked).await;

    let mut differed = 0;
    for answer in &answers {
        assert_eq!(
            answer.coded_neighbors.len(),
            K,
            "a coded answer came back {} long",
            answer.coded_neighbors.len()
        );
        let mut rows = answer
            .coded_neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        rows.sort_unstable();
        let before = rows.len();
        rows.dedup();
        assert_eq!(before, rows.len(), "a coded answer returned a row twice");

        let mut exact = answer
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        exact.sort_unstable();
        if rows != exact {
            differed += 1;
        }
    }
    assert!(
        differed > 0,
        "not one of {QUERIES} queries changed its answer when its candidates were measured \
         exactly, so the coded answer is the exact one under another name"
    );

    // Not asked for, and asked for by a mode that has no second half.
    let (unasked, _) = replay(&index, &queries, &search(WalkMode::Lazy)).await;
    assert!(
        unasked
            .iter()
            .all(|answer| answer.coded_neighbors.is_empty()),
        "a query that did not ask for a coded answer was given one"
    );
    let (whole, _) = replay(
        &index,
        &queries,
        &search(WalkMode::Coded).with_report_coded(true),
    )
    .await;
    assert!(
        whole.iter().all(|answer| answer.coded_neighbors.is_empty()),
        "a mode that holds every vector reported an answer from before a re-score it never makes"
    );
    assert!(
        whole
            .iter()
            .all(|answer| answer.rescore == Default::default()),
        "a mode that never re-scores reported a re-score cost"
    );
}

/// Queries each re-score test asks both ways. Few, because what is compared is
/// exact and each query is two walks.
const BOTH_WAYS: usize = 12;

/// An index with eight-bit scalar codes, the kind the stand measures.
async fn scalar_index(
    uri: &str,
    fixture: &DatasetFixture,
    distance_type: DistanceType,
    vector_source: VectorSource,
) -> Dataset {
    let mut dataset = fixture.write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &params(SCALAR)
            .with_distance_type(distance_type)
            .with_vector_source(vector_source),
    )
    .await
    .unwrap();
    dataset
}

/// Every query answered twice by `index`, re-scoring from its partitions and
/// from the dataset, and the two answers held to be one: the same rows at the
/// same distances, to the last bit.
async fn assert_either_copy_answers_the_same(
    index: &VamanaIndex,
    params: &SearchParams,
    dimension: i32,
    what: &str,
) {
    let from_dataset = params.clone().with_rescore_from_dataset(true);
    for (n, query) in random_vectors_of(BOTH_WAYS, dimension, 4242)
        .iter()
        .enumerate()
    {
        let partition = index.search(query, params).await.unwrap();
        let dataset = index.search(query, &from_dataset).await.unwrap();
        assert_eq!(
            partition.neighbors.len(),
            K,
            "{what}: query {n} came back short"
        );
        assert_eq!(
            partition.neighbors, dataset.neighbors,
            "{what}: query {n} answered differently from the dataset's copy"
        );
    }
}

/// The dataset's copy of a vector is the partition's copy: both modes that
/// re-score answer the same from either, to the last bit, whether the index
/// opens its files per read or holds them - and every row is read by offset.
async fn a_re_score_from_the_dataset_is_the_re_score_from_the_partition(
    distance_type: DistanceType,
    storage_version: Option<LanceFileVersion>,
) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture {
        storage_version,
        ..wide_fixture()
    };
    let dataset = scalar_index(uri, &fixture, distance_type, VectorSource::Index).await;
    for held in [false, true] {
        let mut index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        if held {
            index = index.with_cache(LanceCache::with_capacity(BUDGET));
        }
        for mode in [WalkMode::Lazy, WalkMode::Flat] {
            let what =
                format!("{distance_type:?}, {storage_version:?}, files held {held}, {mode:?}");
            assert_either_copy_answers_the_same(&index, &search(mode), WIDE_DIM, &what).await;
        }
        assert_eq!(
            index.rescore_reads().through_lance,
            0,
            "{distance_type:?}, {storage_version:?}, files held {held}: a data file of \
             {WIDE_DIM}-wide vectors was not read by offset"
        );
    }
}

#[tokio::test]
async fn an_l2_re_score_from_the_dataset_is_the_re_score_from_the_partition() {
    a_re_score_from_the_dataset_is_the_re_score_from_the_partition(DistanceType::L2, None).await;
}

/// The partition holds unit vectors and the dataset does not, so what is read
/// has to be normalised exactly as the build normalised it.
#[tokio::test]
async fn a_cosine_re_score_from_the_dataset_is_the_re_score_from_the_partition() {
    a_re_score_from_the_dataset_is_the_re_score_from_the_partition(DistanceType::Cosine, None)
        .await;
}

/// Every file version the offset read accepts lays the column out the way it
/// reads it, not only the default one.
#[tokio::test]
async fn a_re_score_from_a_2_1_data_file_is_the_re_score_from_the_partition() {
    a_re_score_from_the_dataset_is_the_re_score_from_the_partition(
        DistanceType::L2,
        Some(LanceFileVersion::V2_1),
    )
    .await;
}

#[tokio::test]
async fn a_re_score_from_a_2_3_data_file_is_the_re_score_from_the_partition() {
    a_re_score_from_the_dataset_is_the_re_score_from_the_partition(
        DistanceType::L2,
        Some(LanceFileVersion::V2_3),
    )
    .await;
}

/// An index that keeps no vectors re-scores from the dataset what its twin with
/// vectors re-scores from its own copy: the same rows at the same distances, to
/// the last bit, in both modes that re-score - and whatever the query's switch
/// says, since there is no copy to switch to.
async fn an_index_without_vectors_answers_what_its_twin_with_vectors_answers(
    distance_type: DistanceType,
) {
    let dir = tempfile::tempdir().unwrap();
    let mut twins = Vec::new();
    for vector_source in [VectorSource::Index, VectorSource::Dataset] {
        let uri = dir.path().join(vector_source.to_string());
        let dataset = scalar_index(
            uri.to_str().unwrap(),
            &wide_fixture(),
            distance_type,
            vector_source,
        )
        .await;
        twins.push(VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap());
    }
    let [with, without] = twins.try_into().unwrap();
    assert_eq!(without.metadata().vector_source, VectorSource::Dataset);
    for mode in [WalkMode::Lazy, WalkMode::Flat] {
        for switch in [false, true] {
            let params = search(mode).with_rescore_from_dataset(switch);
            for (n, query) in random_vectors_of(BOTH_WAYS, WIDE_DIM, 4242)
                .iter()
                .enumerate()
            {
                let expected = with.search(query, &search(mode)).await.unwrap();
                let answered = without.search(query, &params).await.unwrap();
                assert_eq!(expected.neighbors.len(), K);
                assert_eq!(
                    answered.neighbors, expected.neighbors,
                    "{distance_type:?}, {mode:?}, switch {switch}: query {n}"
                );
            }
        }
    }
    assert_eq!(without.rescore_reads().through_lance, 0);
}

#[tokio::test]
async fn an_l2_index_without_vectors_answers_what_its_twin_with_vectors_answers() {
    an_index_without_vectors_answers_what_its_twin_with_vectors_answers(DistanceType::L2).await;
}

#[tokio::test]
async fn a_cosine_index_without_vectors_answers_what_its_twin_with_vectors_answers() {
    an_index_without_vectors_answers_what_its_twin_with_vectors_answers(DistanceType::Cosine).await;
}

/// The two modes that read partitions whole read every vector along with them,
/// and a partition of this index holds none.
#[tokio::test]
async fn an_index_without_vectors_refuses_the_modes_that_read_them_all() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = scalar_index(
        uri,
        &wide_fixture(),
        DistanceType::L2,
        VectorSource::Dataset,
    )
    .await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let query = random_vectors_of(1, WIDE_DIM, 7).remove(0);
    for mode in [WalkMode::Exact, WalkMode::Coded] {
        let error = index.search(&query, &search(mode)).await.unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{mode:?}: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("leaves its vectors to the dataset"),
            "{mode:?}: {error}"
        );
    }
    // A query that names no mode walks lazily, which this index can answer.
    let answered = index.search(&query, &SearchParams::new(K)).await.unwrap();
    assert_eq!(answered.neighbors.len(), K);
}

/// A segment added beside an index without vectors keeps none either, though
/// the parameters it is built from default to keeping them: the index opens
/// over both, which it would refuse over segments keeping their vectors in two
/// places, and finds an appended row where it is.
#[tokio::test]
async fn a_segment_added_to_an_index_without_vectors_keeps_none_either() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let base = wide_fixture();
    scalar_index(uri, &base, DistanceType::L2, VectorSource::Dataset).await;
    let appended = DatasetFixture {
        seed: 99,
        ..wide_fixture()
    };
    let mut dataset = appended.append(uri).await;
    let inserted = insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(inserted.fragments_indexed, appended.fragments);

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.num_segments(), 2);
    assert_eq!(index.metadata().vector_source, VectorSource::Dataset);
    // The appended fixture draws its rows the way `random_vectors_of` draws
    // queries, so the first of them is its first row.
    let first = random_vectors_of(1, WIDE_DIM, appended.seed).remove(0);
    let answer = index.search(&first, &search(WalkMode::Lazy)).await.unwrap();
    assert_eq!(
        (answer.neighbors[0].row_addr, answer.neighbors[0].distance),
        ((base.fragments as u64) << 32, 0.0)
    );
}

/// Vectors narrower than 256 bytes are laid out as mini-blocks, which no offset
/// reaches, so the rows are re-scored through Lance - at least one for every
/// answer - and still answer what the partition's copy answers. That no offset
/// reaches them is found out once, and no file is opened again to find it out.
#[tokio::test]
async fn narrow_vectors_are_re_scored_through_lance() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let narrow = DatasetFixture {
        dimension: VECTOR_DIM,
        ..wide_fixture()
    };
    let dataset = scalar_index(uri, &narrow, DistanceType::L2, VectorSource::Index).await;
    for held in [false, true] {
        let mut index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        if held {
            index = index.with_cache(LanceCache::with_capacity(BUDGET));
        }
        assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), VECTOR_DIM, "narrow")
            .await;
        let through_lance = index.rescore_reads().through_lance;
        assert!(
            through_lance >= (BOTH_WAYS * K) as u64,
            "held {held}: only {through_lance} rows went through Lance for {BOTH_WAYS} queries \
             of k = {K}"
        );
        // Every file was found out by the queries above, so none is opened
        // again: a re-score whose every row goes through Lance's take reads
        // nothing itself, whether or not the index keeps a cache.
        let query = random_vectors_of(1, VECTOR_DIM, 7).remove(0);
        let answered = index
            .search(
                &query,
                &search(WalkMode::Lazy).with_rescore_from_dataset(true),
            )
            .await
            .unwrap();
        assert_eq!(
            answered.rescore.iops, 0,
            "held {held}: a data file no offset reaches was opened again: {:?}",
            answered.rescore
        );
    }
}

/// A data file Lance 2.0 wrote has another grammar, so it is read through
/// Lance whatever its width.
#[tokio::test]
async fn a_lance_2_0_data_file_is_re_scored_through_lance() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let old = DatasetFixture {
        storage_version: Some(LanceFileVersion::V2_0),
        ..wide_fixture()
    };
    let dataset = scalar_index(uri, &old, DistanceType::L2, VectorSource::Index).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "Lance 2.0")
        .await;
    assert!(
        index.rescore_reads().through_lance > 0,
        "a Lance 2.0 data file was read by offset"
    );
    // The manifest alone sends a Lance 2.0 file to Lance, so none is opened: a
    // re-score whose every row goes through Lance's take reads nothing itself.
    let query = random_vectors_of(1, WIDE_DIM, 7).remove(0);
    let answered = index
        .search(
            &query,
            &search(WalkMode::Lazy).with_rescore_from_dataset(true),
        )
        .await
        .unwrap();
    assert_eq!(
        answered.rescore.iops, 0,
        "a Lance 2.0 data file was opened: {:?}",
        answered.rescore
    );
}

/// A column holding nulls carries validity beside its values, which moves them
/// off the offsets a full-zip page of bare values would put them at.
#[tokio::test]
async fn a_column_with_nulls_is_re_scored_through_lance() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let sparse = DatasetFixture {
        null_every: Some(7),
        ..wide_fixture()
    };
    let dataset = scalar_index(uri, &sparse, DistanceType::L2, VectorSource::Index).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "nulls").await;
    assert!(
        index.rescore_reads().through_lance > 0,
        "a column with nulls was read by offset"
    );
}

/// A shallow clone keeps its data files under the dataset it was cloned from,
/// a base path this crate does not resolve, so their rows are re-scored through
/// Lance - and the answer is the partition's.
#[tokio::test]
async fn a_shallow_clone_is_re_scored_through_lance() {
    let dir = tempfile::tempdir().unwrap();
    let source_uri = dir.path().join("source");
    let mut source = wide_fixture().write(source_uri.to_str().unwrap()).await;
    let version = source.version().version;
    let clone_uri = dir.path().join("clone");
    let mut dataset = source
        .shallow_clone(clone_uri.to_str().unwrap(), version, None)
        .await
        .unwrap();
    assert!(
        dataset.get_fragments().iter().all(|fragment| {
            fragment
                .metadata()
                .files
                .iter()
                .all(|file| file.base_id.is_some())
        }),
        "the clone wrote data files of its own, so no base path is under test"
    );
    create_index(&mut dataset, INDEX_NAME, &params(SCALAR))
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "shallow clone")
        .await;
    assert!(
        index.rescore_reads().through_lance > 0,
        "a data file under another base path was read by offset"
    );
}

/// A dead row is dropped before anything is read for it, which is the only way
/// past a candidate whose whole fragment has gone, since nothing is left to read
/// - and the answer is the one the partition's copy gives, which drops the same
/// rows afterwards.
#[tokio::test]
async fn a_re_score_from_the_dataset_answers_only_live_rows() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset =
        scalar_index(uri, &wide_fixture(), DistanceType::L2, VectorSource::Index).await;
    dataset
        .delete("vec IS NOT NULL AND _rowid % 3 = 0")
        .await
        .unwrap();
    // Every row of fragment 0, so the fragment itself goes.
    dataset.delete("_rowid < 200").await.unwrap();
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.id() != 0),
        "fragment 0 survived losing every row"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "deleted").await;
    let live = common::live_row_ids(&dataset).await;
    let from_dataset = search(WalkMode::Lazy).with_rescore_from_dataset(true);
    for query in random_vectors_of(BOTH_WAYS, WIDE_DIM, 4243) {
        for neighbor in index.search(&query, &from_dataset).await.unwrap().neighbors {
            assert!(
                live.contains(&neighbor.row_addr),
                "row {} was deleted and came back anyway",
                neighbor.row_addr
            );
        }
    }
}

/// A deferred compaction moves rows into new fragments and the index follows
/// the record of where they went: a re-score from the dataset reads each moved
/// row where it landed, by offset, since the new files are full-zip too.
#[tokio::test]
async fn a_re_score_from_the_dataset_reads_a_moved_row_where_it_landed() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset =
        scalar_index(uri, &wide_fixture(), DistanceType::L2, VectorSource::Index).await;
    let metrics = common::compact_indexed(&mut dataset).await;
    assert!(metrics.fragments_removed > 0, "nothing moved: {metrics:?}");

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "moved").await;
    assert_eq!(index.rescore_reads().through_lance, 0);
}

/// A segment indexed over appended rows re-scores from their fragments as the
/// base segment does from its own.
#[tokio::test]
async fn a_re_score_from_the_dataset_answers_across_segments() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    scalar_index(uri, &wide_fixture(), DistanceType::L2, VectorSource::Index).await;
    wide_fixture().append(uri).await;
    let mut dataset = Dataset::open(uri).await.unwrap();
    lance_vamana::insert_as_segment(&mut dataset, INDEX_NAME)
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert!(
        index.num_segments() > 1,
        "the append wrote no second segment"
    );
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "two segments")
        .await;
}

/// A dataset whose vectors are its second column: `id`, then `vec`.
async fn write_behind_an_id(uri: &str) -> Dataset {
    let fixture = wide_fixture();
    let rows = fixture.rows();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        wide_field(),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(wide_vectors(rows, 99)),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: fixture.rows_per_fragment,
            max_rows_per_group: fixture.rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn wide_field() -> Field {
    Field::new(
        VECTOR_COLUMN,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            WIDE_DIM,
        ),
        true,
    )
}

fn wide_vectors(rows: usize, seed: u64) -> FixedSizeListArray {
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        random_vectors_of(rows, WIDE_DIM, seed)
            .into_iter()
            .map(|vector| Some(vector.into_iter().map(Some).collect::<Vec<_>>())),
        WIDE_DIM,
    )
}

/// The vector column is found by field id through the manifest, not by name or
/// by position: behind another column of its own file, and again after a
/// rename.
#[tokio::test]
async fn a_re_score_from_the_dataset_finds_its_column_by_field_id() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_behind_an_id(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params(SCALAR))
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "behind an id")
        .await;
    assert_eq!(index.rescore_reads().through_lance, 0);

    dataset
        .alter_columns(&[
            ColumnAlteration::new(VECTOR_COLUMN.to_string()).rename("renamed".to_string())
        ])
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "renamed").await;
    assert_eq!(index.rescore_reads().through_lance, 0);
}

/// Two vector columns of one shape in one data file, an index over each, the
/// two sharing one cache: each re-scores from its own column, although what is
/// known about the file is known first for the other one. The neighbour column
/// has the very shape a re-score reads, so a read that landed on it would pass
/// every check on its pages and answer from the wrong vectors.
#[tokio::test]
async fn two_indices_over_two_columns_of_one_file_each_re_score_from_their_own() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = wide_fixture();
    let rows = fixture.rows();
    let other_column = "vec2";
    let other_index = "vamana_idx2";
    let schema = Arc::new(ArrowSchema::new(vec![
        wide_field(),
        Field::new(other_column, wide_field().data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(wide_vectors(rows, 97)),
            Arc::new(wide_vectors(rows, 96)),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: fixture.rows_per_fragment,
            max_rows_per_group: fixture.rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    create_index(&mut dataset, INDEX_NAME, &params(SCALAR))
        .await
        .unwrap();
    create_index(&mut dataset, other_index, &params_on(other_column, SCALAR))
        .await
        .unwrap();

    let cache = LanceCache::with_capacity(BUDGET);
    for name in [INDEX_NAME, other_index] {
        let index = VamanaIndex::open(&dataset, name)
            .await
            .unwrap()
            .with_cache(cache.clone());
        assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, name).await;
        assert_eq!(index.rescore_reads().through_lance, 0, "{name}");
    }
}

/// A fragment's vectors in its second data file, where adding a column puts
/// them: the file holding the field is the one read.
#[tokio::test]
async fn a_re_score_from_the_dataset_finds_the_data_file_holding_its_column() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = wide_fixture();
    let rows = fixture.rows();
    let ids = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        ids.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], ids),
        uri,
        Some(WriteParams {
            max_rows_per_file: fixture.rows_per_fragment,
            max_rows_per_group: fixture.rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let vectors = Arc::new(ArrowSchema::new(vec![wide_field()]));
    let batch =
        RecordBatch::try_new(vectors.clone(), vec![Arc::new(wide_vectors(rows, 98))]).unwrap();
    dataset
        .add_columns(
            NewColumnTransform::Reader(Box::new(RecordBatchIterator::new(
                vec![Ok(batch)],
                vectors,
            ))),
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata().files.len() == 2),
        "adding the column did not give every fragment a second data file"
    );

    create_index(&mut dataset, INDEX_NAME, &params(SCALAR))
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_either_copy_answers_the_same(&index, &search(WalkMode::Lazy), WIDE_DIM, "second file")
        .await;
    assert_eq!(index.rescore_reads().through_lance, 0);
}

/// An index is pinned to the version it was opened on, and so is the copy it
/// re-scores from. Once a compaction and a cleanup have removed that version's
/// data files, a re-score from the dataset fails and names the file it lost,
/// rather than answering from rows another version put at those addresses.
#[tokio::test]
async fn a_re_score_from_a_cleaned_up_version_names_the_file_it_lost() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset =
        scalar_index(uri, &wide_fixture(), DistanceType::L2, VectorSource::Index).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let lost = dataset.get_fragments()[0].metadata().files[0].path.clone();

    common::compact_indexed(&mut dataset).await;
    let stale = (1..dataset.version().version).collect::<Vec<_>>();
    dataset
        .cleanup_with_policy(
            CleanupPolicyBuilder::default()
                .versions(stale)
                .unwrap()
                .delete_unverified(true)
                .build(),
        )
        .await
        .unwrap();

    let from_dataset = search(WalkMode::Lazy).with_rescore_from_dataset(true);
    let mut refused = 0;
    for query in random_vectors_of(BOTH_WAYS, WIDE_DIM, 4244) {
        if let Err(error) = index.search(&query, &from_dataset).await {
            assert!(error.to_string().contains(&lost), "{error}");
            refused += 1;
        }
    }
    assert_eq!(
        refused, BOTH_WAYS,
        "a re-score read rows of a version that is gone"
    );
}

/// How many entry points a partition of these fixtures trains: few enough to
/// train in a test, more than one so that there is a choice to make.
const ENTRIES: usize = 8;

/// Every query's answer, its coded answer and its count, rows paired with the
/// bits of their distances so that two runs are held equal to the last bit.
async fn walked(
    index: &VamanaIndex,
    params: &SearchParams,
    queries: &[Vec<f32>],
) -> Vec<(Vec<(u64, u32)>, Vec<(u64, u32)>, u64)> {
    let bits = |neighbors: &[Neighbor]| {
        neighbors
            .iter()
            .map(|neighbor| (neighbor.row_addr, neighbor.distance.to_bits()))
            .collect::<Vec<_>>()
    };
    let mut answers = Vec::with_capacity(queries.len());
    for query in queries {
        let result = index.search(query, params).await.unwrap();
        answers.push((
            bits(&result.neighbors),
            bits(&result.coded_neighbors),
            result.comparisons,
        ));
    }
    answers
}

/// The walk with a list and the walk with a stop margin, each answering its
/// coded neighbours too.
fn both_walks() -> [SearchParams; 2] {
    let walk = search(WalkMode::Lazy).with_report_coded(true);
    let margin = walk
        .clone()
        .with_rescore_budget(2 * K)
        .with_stop_margin(0.05);
    [walk, margin]
}

/// An index given entry points it is not asked to use walks from the medoid
/// exactly as an index never given them.
///
/// What lets a timed round hand the same entry points to both of its arms, so
/// that the start is the only thing between them.
#[tokio::test]
async fn entry_points_a_walk_does_not_ask_for_change_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = coded_dataset(dir.path().to_str().unwrap(), SCALAR).await;
    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let trained = plain
        .train_entry_points(&EntryPointParams::new(ENTRIES))
        .await
        .unwrap();
    let given = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .with_entry_points(Arc::new(trained))
        .unwrap();
    let queries = random_vectors(QUERIES, 4242);
    for params in both_walks() {
        assert_eq!(
            walked(&given, &params, &queries).await,
            walked(&plain, &params, &queries).await,
            "margin {:?}",
            params.stop_margin
        );
    }
}

/// What the start is for: the same neighbours from a start nearer the query.
///
/// Recall only. That the start is the chosen entry point, and what choosing
/// costs, are [`a_walk_charges_the_entry_points_it_did_not_choose_and_leaves_them_unmarked`]'s.
async fn a_walk_from_the_nearest_entry_point_finds_the_neighbours(codes: CodeSpec) {
    let dir = tempfile::tempdir().unwrap();
    let dataset = coded_dataset(dir.path().to_str().unwrap(), codes).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let trained = index
        .train_entry_points(&EntryPointParams::new(ENTRIES))
        .await
        .unwrap();
    let index = index.with_entry_points(Arc::new(trained)).unwrap();
    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;

    let medoid = measure(&index, &queries, &truth, &search(WalkMode::Lazy)).await;
    let entry = measure(
        &index,
        &queries,
        &truth,
        &search(WalkMode::Lazy).with_start(WalkStart::NearestEntry),
    )
    .await;
    assert!(
        entry.recall >= 0.9 && entry.recall > medoid.recall - 0.02,
        "from the entry points recall {} against {} from the medoid",
        entry.recall,
        medoid.recall
    );
}

/// One entry point a partition, its medoid, for every partition of every
/// committed segment: the entry points under which the walk from the nearest
/// one is the walk from the medoid.
async fn medoids(dataset: &Dataset) -> Vec<PartitionEntryPoints> {
    let store = dataset.object_store(None).await.unwrap();
    let scheduler = scan_scheduler(&store);
    let mut partitions = Vec::new();
    for index in committed_segments(dataset, INDEX_NAME).await.unwrap() {
        let dir = dataset.indices_dir().join(index.uuid.to_string());
        let manifest = read_segment(&scheduler, &dir, None).await.unwrap();
        partitions.extend(
            manifest
                .partitions()
                .iter()
                .map(|entry| PartitionEntryPoints {
                    segment: index.uuid,
                    partition_id: entry.partition_id,
                    num_rows: entry.num_rows,
                    entries: vec![entry.medoid],
                }),
        );
    }
    partitions
}

/// The entry points a walk measured and did not choose are charged, one
/// distance each, and left unmarked, so that the walk can still reach them -
/// and the one it did choose is where it starts.
///
/// One partition, and the query is its medoid's own vector, so of the medoid
/// and two vertices the medoid's walk ended near, the medoid is the nearest by
/// code and the walk is the medoid's walk with two more distances. Marking the
/// two it passed over would keep them out of the list they ended in; charging
/// one entry point, or every one the parameters allow, would miss the count.
/// A single entry point that is not the medoid, at the cost of the medoid's
/// one distance, has to walk differently, or the start never reached the walk.
#[tokio::test]
async fn a_walk_charges_the_entry_points_it_did_not_choose_and_leaves_them_unmarked() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = wide_fixture().write(dir.path().to_str().unwrap()).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, 1)
            .with_graph_params(BuildParams {
                max_degree: MAX_DEGREE,
                search_list_size: 64,
                ..Default::default()
            })
            .with_codes(SCALAR),
    )
    .await
    .unwrap();
    let [only] = medoids(&dataset).await.try_into().unwrap();
    let medoid = only.entries[0];
    let segments = common::read_committed_batches(&dataset, INDEX_NAME).await;
    let batch = &segments[0].1[0];
    let rows = batch["__row_id"]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    let vectors = batch["__vector"].as_fixed_size_list();
    let query = vectors
        .value(medoid as usize)
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let local = |row_addr: u64| rows.iter().position(|&row| row == row_addr).unwrap() as u32;
    let given = |entries: Vec<u32>| {
        let entry_points = EntryPoints::from_partitions(
            EntryPointParams::new(ENTRIES),
            vec![PartitionEntryPoints {
                entries,
                ..only.clone()
            }],
        )
        .unwrap();
        let dataset = &dataset;
        async move {
            VamanaIndex::open(dataset, INDEX_NAME)
                .await
                .unwrap()
                .with_entry_points(Arc::new(entry_points))
                .unwrap()
        }
    };
    let bits = |neighbors: &[Neighbor]| {
        neighbors
            .iter()
            .map(|neighbor| (neighbor.row_addr, neighbor.distance.to_bits()))
            .collect::<Vec<_>>()
    };

    for params in both_walks() {
        let params = params.with_nprobes(1);
        let what = format!("margin {:?}", params.stop_margin);
        let from_medoid = index.search(&query, &params).await.unwrap();
        let mut near = from_medoid
            .coded_neighbors
            .iter()
            .map(|neighbor| local(neighbor.row_addr))
            .filter(|&id| id != medoid)
            .take(2)
            .collect::<Vec<_>>();
        assert_eq!(
            near.len(),
            2,
            "{what}: the medoid's walk ended near fewer than two others"
        );
        let entry = params.clone().with_start(WalkStart::NearestEntry);

        let mut three = vec![medoid, near[0], near[1]];
        three.sort_unstable();
        let from_three = given(three).await.search(&query, &entry).await.unwrap();
        assert_eq!(
            bits(&from_three.neighbors),
            bits(&from_medoid.neighbors),
            "{what}"
        );
        assert_eq!(
            bits(&from_three.coded_neighbors),
            bits(&from_medoid.coded_neighbors),
            "{what}"
        );
        assert_eq!(
            from_three.comparisons,
            from_medoid.comparisons + 2,
            "{what}"
        );

        let elsewhere = given(vec![near.pop().unwrap()]).await;
        let queries = random_vectors_of(BOTH_WAYS, WIDE_DIM, 4242);
        assert_ne!(
            walked(&elsewhere, &entry, &queries).await,
            walked(&index, &params, &queries).await,
            "{what}: a walk from one entry point that is not the medoid walked the medoid's walk"
        );
    }
}

#[tokio::test]
async fn a_rabit_walk_from_the_nearest_entry_point_finds_the_neighbours() {
    a_walk_from_the_nearest_entry_point_finds_the_neighbours(RABIT).await;
}

#[tokio::test]
async fn a_scalar_walk_from_the_nearest_entry_point_finds_the_neighbours() {
    a_walk_from_the_nearest_entry_point_finds_the_neighbours(SCALAR).await;
}

/// Training twice trains the same entry points, in the shape a walk relies on:
/// ascending, distinct, inside the partition, no more than asked for - and
/// [`EntryPoints::from_partitions`] takes back exactly what it trained.
#[tokio::test]
async fn entry_points_train_the_same_every_time() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = coded_dataset(dir.path().to_str().unwrap(), SCALAR).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let params = EntryPointParams::new(ENTRIES);
    let first = index.train_entry_points(&params).await.unwrap();
    assert_eq!(first, index.train_entry_points(&params).await.unwrap());
    assert_eq!(first.params(), &params);
    assert_eq!(first.partitions().len(), PARTITIONS as usize);
    for partition in first.partitions() {
        let entries = &partition.entries;
        assert!(
            !entries.is_empty() && entries.len() <= ENTRIES,
            "{partition:?}"
        );
        assert!(
            entries.windows(2).all(|pair| pair[0] < pair[1]),
            "{partition:?}"
        );
        assert!(
            entries.iter().all(|&entry| entry < partition.num_rows),
            "{partition:?}"
        );
    }
    let rebuilt =
        EntryPoints::from_partitions(first.params().clone(), first.partitions().to_vec()).unwrap();
    assert_eq!(rebuilt, first);
}

/// A partition with no more live vertices than entry points asked for trains
/// none, and its walks are the medoid's.
#[tokio::test]
async fn a_partition_no_larger_than_its_entry_points_starts_at_its_medoid() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = coded_dataset(dir.path().to_str().unwrap(), SCALAR).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let rows = fixture().fragments * fixture().rows_per_fragment;
    let trained = index
        .train_entry_points(&EntryPointParams::new(rows))
        .await
        .unwrap();
    assert!(
        trained
            .partitions()
            .iter()
            .all(|partition| partition.entries.is_empty()),
        "{trained:?}"
    );
    let index = index.with_entry_points(Arc::new(trained)).unwrap();
    let queries = random_vectors(QUERIES, 4242);
    for params in both_walks() {
        assert_eq!(
            walked(
                &index,
                &params.clone().with_start(WalkStart::NearestEntry),
                &queries
            )
            .await,
            walked(&index, &params, &queries).await,
            "margin {:?}",
            params.stop_margin
        );
    }
}

/// The bound is inclusive: a partition of exactly as many live vertices as
/// entry points asked for trains none, and one a vertex larger trains them.
#[tokio::test]
async fn a_partition_of_exactly_k_vertices_trains_no_entry_points() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = scalar_index(
        dir.path().to_str().unwrap(),
        &wide_fixture(),
        DistanceType::L2,
        VectorSource::Index,
    )
    .await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let shape = index
        .train_entry_points(&EntryPointParams::new(ENTRIES))
        .await
        .unwrap();
    let smallest = shape
        .partitions()
        .iter()
        .map(|partition| partition.num_rows)
        .min()
        .unwrap();
    assert!(
        shape
            .partitions()
            .iter()
            .any(|partition| partition.num_rows > smallest),
        "every partition is as small as the smallest, so the bound has nothing to split"
    );
    let trained = index
        .train_entry_points(&EntryPointParams::new(smallest as usize))
        .await
        .unwrap();
    for partition in trained.partitions() {
        assert_eq!(
            partition.entries.is_empty(),
            partition.num_rows <= smallest,
            "{partition:?}"
        );
    }
}

/// Each partition's row addresses in local-id order, by partition id, off the
/// files of the index's only segment.
async fn row_addresses(dataset: &Dataset) -> HashMap<u32, Vec<u64>> {
    let segments = common::read_committed_batches(dataset, INDEX_NAME).await;
    assert_eq!(segments.len(), 1, "the fixture has one segment");
    let (manifest, batches) = &segments[0];
    manifest
        .partitions()
        .iter()
        .zip(batches)
        .map(|(entry, batch)| {
            let rows = batch["__row_id"].as_primitive::<UInt64Type>().values();
            (entry.partition_id, rows.to_vec())
        })
        .collect()
}

/// A deleted row is never an entry point: training is over the live vertices
/// only, and a partition left with none starts at its medoid.
///
/// The rows deleted are exactly the entry points the first training chose, so
/// a training that read the dead vertices too would choose some of them again.
#[tokio::test]
async fn a_deleted_row_is_never_an_entry_point() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = coded_dataset(dir.path().to_str().unwrap(), SCALAR).await;
    let params = EntryPointParams::new(ENTRIES);
    let before = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .train_entry_points(&params)
        .await
        .unwrap();
    let addresses = row_addresses(&dataset).await;
    let emptied = before.partitions()[0].partition_id;
    let mut deleted = HashSet::new();
    for partition in before.partitions() {
        let rows = &addresses[&partition.partition_id];
        if partition.partition_id == emptied {
            deleted.extend(rows.iter().copied());
        } else {
            deleted.extend(partition.entries.iter().map(|&entry| rows[entry as usize]));
        }
    }
    let listed = deleted
        .iter()
        .map(|row| row.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    dataset
        .delete(&format!("_rowid IN ({listed})"))
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let after = index.train_entry_points(&params).await.unwrap();
    for partition in after.partitions() {
        let rows = &addresses[&partition.partition_id];
        assert_eq!(
            partition.entries.is_empty(),
            partition.partition_id == emptied,
            "{partition:?}"
        );
        for &entry in &partition.entries {
            assert!(
                !deleted.contains(&rows[entry as usize]),
                "partition {} chose deleted vertex {entry}",
                partition.partition_id
            );
        }
    }
    let index = index.with_entry_points(Arc::new(after)).unwrap();
    for query in random_vectors(QUERIES, 4243) {
        let result = index
            .search(
                &query,
                &search(WalkMode::Lazy).with_start(WalkStart::NearestEntry),
            )
            .await
            .unwrap();
        assert_eq!(result.neighbors.len(), K);
        assert!(
            result
                .neighbors
                .iter()
                .all(|neighbor| !deleted.contains(&neighbor.row_addr)),
            "a deleted row came back"
        );
    }
}

/// An index that keeps no vectors trains the entry points its twin with
/// vectors trains: the dataset's vectors are the partitions' copy to the last
/// bit, normalised alike under cosine. Its walks start at them too.
async fn an_index_without_vectors_trains_the_entry_points_its_twin_trains(
    distance_type: DistanceType,
) {
    let dir = tempfile::tempdir().unwrap();
    let params = EntryPointParams::new(ENTRIES);
    let mut datasets = Vec::new();
    for vector_source in [VectorSource::Index, VectorSource::Dataset] {
        let uri = dir.path().join(vector_source.to_string());
        datasets.push(
            scalar_index(
                uri.to_str().unwrap(),
                &wide_fixture(),
                distance_type,
                vector_source,
            )
            .await,
        );
    }
    let mut trained = Vec::new();
    let mut indexes = Vec::new();
    for dataset in &datasets {
        let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
        trained.push(index.train_entry_points(&params).await.unwrap());
        indexes.push(index);
    }
    // The twins' segments are two directories under two uuids.
    let entries = |entry_points: &EntryPoints| {
        entry_points
            .partitions()
            .iter()
            .map(|partition| {
                (
                    partition.partition_id,
                    partition.num_rows,
                    partition.entries.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        entries(&trained[0]),
        entries(&trained[1]),
        "{distance_type:?}"
    );
    assert!(
        trained[1]
            .partitions()
            .iter()
            .any(|partition| !partition.entries.is_empty()),
        "every partition was too small to train, so the twins agree on nothing"
    );

    // With rows deleted the twin without vectors reads only the live ones out
    // of the dataset, and the two still agree - on entry points none of which
    // is a deleted row.
    let mut after = Vec::new();
    for dataset in &mut datasets {
        dataset.delete("_rowid % 5 = 0").await.unwrap();
        let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
        after.push(index.train_entry_points(&params).await.unwrap());
    }
    assert_eq!(
        entries(&after[0]),
        entries(&after[1]),
        "{distance_type:?}, deleted"
    );
    let addresses = row_addresses(&datasets[1]).await;
    for partition in after[1].partitions() {
        for &entry in &partition.entries {
            assert_ne!(
                addresses[&partition.partition_id][entry as usize] % 5,
                0,
                "{distance_type:?}: partition {} chose a deleted vertex",
                partition.partition_id
            );
        }
    }

    let without = indexes
        .pop()
        .unwrap()
        .with_entry_points(Arc::new(trained.pop().unwrap()))
        .unwrap();
    assert_eq!(without.metadata().vector_source, VectorSource::Dataset);
    for query in random_vectors_of(BOTH_WAYS, WIDE_DIM, 4242) {
        let result = without
            .search(
                &query,
                &search(WalkMode::Lazy).with_start(WalkStart::NearestEntry),
            )
            .await
            .unwrap();
        assert_eq!(result.neighbors.len(), K, "{distance_type:?}");
    }
}

#[tokio::test]
async fn an_l2_index_without_vectors_trains_the_entry_points_its_twin_trains() {
    an_index_without_vectors_trains_the_entry_points_its_twin_trains(DistanceType::L2).await;
}

#[tokio::test]
async fn a_cosine_index_without_vectors_trains_the_entry_points_its_twin_trains() {
    an_index_without_vectors_trains_the_entry_points_its_twin_trains(DistanceType::Cosine).await;
}

/// A column holding nulls trains and walks from its entry points: a null row is
/// no vertex, so training reads nothing for it, and the walk answers `k` live
/// rows.
#[tokio::test]
async fn a_column_with_nulls_trains_entry_points_over_its_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let sparse = DatasetFixture {
        null_every: Some(7),
        ..wide_fixture()
    };
    let dataset = scalar_index(
        dir.path().to_str().unwrap(),
        &sparse,
        DistanceType::L2,
        VectorSource::Index,
    )
    .await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let trained = index
        .train_entry_points(&EntryPointParams::new(ENTRIES))
        .await
        .unwrap();
    let live = common::live_row_ids(&dataset).await;
    assert!(
        trained
            .partitions()
            .iter()
            .any(|partition| !partition.entries.is_empty()),
        "every partition was too small to train, so nulls were never in the way"
    );
    let index = index.with_entry_points(Arc::new(trained)).unwrap();
    for query in random_vectors_of(BOTH_WAYS, WIDE_DIM, 4242) {
        let result = index
            .search(
                &query,
                &search(WalkMode::Lazy).with_start(WalkStart::NearestEntry),
            )
            .await
            .unwrap();
        assert_eq!(result.neighbors.len(), K);
        assert!(
            result
                .neighbors
                .iter()
                .all(|neighbor| live.contains(&neighbor.row_addr))
        );
    }
}

/// Every segment of an index trains its own partitions' entry points, and the
/// entry points of the index before a segment was added are refused once it
/// has been: they name too few partitions.
#[tokio::test]
async fn every_segment_trains_entry_points_of_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri, SCALAR).await;
    let params = EntryPointParams::new(ENTRIES);
    let before = Arc::new(
        VamanaIndex::open(&dataset, INDEX_NAME)
            .await
            .unwrap()
            .train_entry_points(&params)
            .await
            .unwrap(),
    );
    // Other rows than the first segment's, or the second is a copy of it and a
    // lookup under the wrong segment would find the same entry points there.
    DatasetFixture {
        seed: 12,
        ..fixture()
    }
    .append(uri)
    .await;
    let mut dataset = Dataset::open(uri).await.unwrap();
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert!(
        index.num_segments() > 1,
        "the append wrote no second segment"
    );
    let refused = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .with_entry_points(before)
        .unwrap_err();
    assert!(
        refused.to_string().contains("trained for another index"),
        "{refused}"
    );

    let trained = index.train_entry_points(&params).await.unwrap();
    let segments = trained
        .partitions()
        .iter()
        .map(|partition| partition.segment)
        .collect::<HashSet<_>>();
    assert_eq!(segments.len(), index.num_segments());

    // Each segment's medoids as its only entry points walk the medoid's walks,
    // which a lookup under the other segment's uuid would not: the two
    // segments' partitions have other medoids.
    let medoids = medoids(&dataset).await;
    for one in &medoids {
        assert!(
            medoids
                .iter()
                .filter(|other| other.partition_id == one.partition_id)
                .all(|other| other.segment == one.segment || other.entries != one.entries),
            "partition {} has the same medoid in two segments: {medoids:?}",
            one.partition_id
        );
    }
    let as_medoids = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .with_entry_points(Arc::new(
            EntryPoints::from_partitions(EntryPointParams::new(1), medoids).unwrap(),
        ))
        .unwrap();
    let queries = random_vectors(QUERIES, 4244);
    for params in both_walks() {
        assert_eq!(
            walked(
                &as_medoids,
                &params.clone().with_start(WalkStart::NearestEntry),
                &queries
            )
            .await,
            walked(&as_medoids, &params, &queries).await,
            "margin {:?}",
            params.stop_margin
        );
    }

    let index = index.with_entry_points(Arc::new(trained)).unwrap();
    for query in random_vectors(QUERIES, 4242) {
        let result = index
            .search(
                &query,
                &search(WalkMode::Lazy).with_start(WalkStart::NearestEntry),
            )
            .await
            .unwrap();
        assert_eq!(result.neighbors.len(), K);
    }
}

/// Entry points in a shape no training produces are refused before an index
/// ever sees them, and so are parameters no training can run with.
#[test]
fn entry_points_out_of_shape_are_refused() {
    let segment = uuid::Uuid::new_v4();
    let partition = |partition_id: u32, entries: Vec<u32>| PartitionEntryPoints {
        segment,
        partition_id,
        num_rows: 100,
        entries,
    };
    let refused = |params: EntryPointParams, partitions: Vec<PartitionEntryPoints>| {
        let error = EntryPoints::from_partitions(params, partitions).unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{error}"
        );
        error.to_string()
    };
    let four = || EntryPointParams::new(4);
    for (partitions, expected) in [
        (vec![partition(0, vec![5, 3])], "ascending and distinct"),
        (vec![partition(0, vec![3, 3])], "ascending and distinct"),
        (vec![partition(0, vec![3, 100])], "outside partition 0"),
        (
            vec![partition(0, vec![1, 2, 3, 4, 5])],
            "more than num_entries 4",
        ),
        (
            vec![partition(0, vec![1]), partition(0, vec![2])],
            "listed twice",
        ),
    ] {
        let message = refused(four(), partitions);
        assert!(message.contains(expected), "{message}");
    }
    assert!(
        EntryPoints::from_partitions(
            four(),
            vec![partition(0, vec![]), partition(1, vec![0, 99])]
        )
        .is_ok()
    );

    for (params, expected) in [
        (EntryPointParams::new(0), "num_entries must be at least 1"),
        (
            EntryPointParams::new(4).with_sample_size(3),
            "smaller than num_entries 4",
        ),
        (
            EntryPointParams::new(4).with_sample_size(4 * 512 + 1),
            "over 512 vectors per entry point",
        ),
    ] {
        let message = refused(params, Vec::new());
        assert!(message.contains(expected), "{message}");
    }
    // Both ends of the sample's range are inside it.
    for sample_size in [4, 4 * 512] {
        assert!(
            EntryPoints::from_partitions(
                EntryPointParams::new(4).with_sample_size(sample_size),
                Vec::new()
            )
            .is_ok(),
            "sample_size {sample_size}"
        );
    }
}

/// A walk asked to start at entry points is refused by an index given none,
/// and by every mode but the lazy walk, rather than quietly started at the
/// medoid.
#[tokio::test]
async fn a_start_at_entry_points_is_refused_where_there_is_none_to_take() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = coded_dataset(dir.path().to_str().unwrap(), SCALAR).await;
    let query = random_vectors(1, 4242).remove(0);
    let entry = |mode: WalkMode| search(mode).with_start(WalkStart::NearestEntry);

    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let error = plain
        .search(&query, &entry(WalkMode::Lazy))
        .await
        .unwrap_err();
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(
        error.to_string().contains("given no entry points"),
        "{error}"
    );
    let error = plain
        .train_entry_points(&EntryPointParams::new(0))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("num_entries must be at least 1"),
        "{error}"
    );

    let trained = plain
        .train_entry_points(&EntryPointParams::new(ENTRIES))
        .await
        .unwrap();

    // Entry points that name the index's partitions at another size, or a
    // partition it does not have, are another index's.
    let refused = |partitions: Vec<PartitionEntryPoints>| {
        let entry_points =
            Arc::new(EntryPoints::from_partitions(trained.params().clone(), partitions).unwrap());
        let dataset = &dataset;
        async move {
            VamanaIndex::open(dataset, INDEX_NAME)
                .await
                .unwrap()
                .with_entry_points(entry_points)
                .unwrap_err()
        }
    };
    let mut grown = trained.partitions().to_vec();
    grown[0].num_rows += 1;
    let error = refused(grown).await;
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("which holds"), "{error}");
    let mut extra = trained.partitions().to_vec();
    extra.push(PartitionEntryPoints {
        segment: uuid::Uuid::new_v4(),
        ..extra[0].clone()
    });
    let error = refused(extra).await;
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(
        error.to_string().contains("this one has no partition"),
        "{error}"
    );

    let given = plain.with_entry_points(Arc::new(trained)).unwrap();
    for mode in [WalkMode::Exact, WalkMode::Coded, WalkMode::Flat] {
        let error = given.search(&query, &entry(mode)).await.unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("only a WalkMode::Lazy walk does"),
            "{mode:?}: {error}"
        );
    }
    assert!(given.search(&query, &entry(WalkMode::Lazy)).await.is_ok());
}
