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

use std::sync::Arc;

use futures::future::join_all;
use lance::Dataset;
use lance_core::cache::LanceCache;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::codes::CodeParams;
use lance_vamana::query::{QueryResult, SearchParams, VamanaIndex, WalkMode};

mod common;
use common::{DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, brute_force, random_vectors, recall};

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

/// The ways a lazy walk can be asked for something it cannot do.
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
    let dataset = coded_dataset(uri).await;
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
    coded_dataset(uri).await;
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
    let dataset = coded_dataset(uri).await;
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
    let dataset = coded_dataset(uri).await;
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
    let dataset = coded_dataset(uri).await;
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
/// All three modes, because two of them take the cache in different amounts:
/// the whole-partition walks keep only the layout of a file they have opened
/// before, while a lazy walk also keeps what it steers by. Both runs of the
/// cached arm are compared, so the answer is pinned against the read that
/// populated the cache as well as against the index that has none.
#[tokio::test]
async fn a_cache_does_not_change_an_answer() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let queries = random_vectors(QUERIES, 606);

    for mode in [WalkMode::Exact, WalkMode::Coded, WalkMode::Lazy] {
        let params = search(mode);
        let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        let (uncached, _) = replay(&plain, &queries, &params).await;

        let index = cached(&dataset, BUDGET).await;
        let (cold, _) = replay(&index, &queries, &params).await;
        let (warm, _) = replay(&index, &queries, &params).await;

        assert_same(&uncached, &cold, &format!("{mode:?}, first pass"));
        assert_same(&uncached, &warm, &format!("{mode:?}, second pass"));
        assert!(
            index.cache_stats().await.unwrap().hits > 0,
            "{mode:?}: the cache served nothing, so the equality above proves nothing"
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
    let dataset = coded_dataset(uri).await;
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
fn code_column_bytes() -> usize {
    let dimension = VECTOR_DIM as u32;
    let stride = CodeParams::mint(CODE_BITS, dimension)
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
    coded_dataset(uri).await;
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
/// Everything is evicted as fast as it is inserted, so every query re-reads
/// every partition - and has to answer exactly what it would have answered with
/// room to spare. This is the path where a cached `Arc` is dropped between the
/// read and the next use of it, which is the one way a caching bug can look like
/// a memory bug rather than a wrong answer.
#[tokio::test]
async fn a_budget_too_small_for_a_partition_still_answers() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = coded_dataset(uri).await;
    let queries = random_vectors(16, 31337);
    let params = search(WalkMode::Lazy);

    let plain = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let (uncached, _) = replay(&plain, &queries, &params).await;

    let index = cached(&dataset, 1024).await;
    let (first, _) = replay(&index, &queries, &params).await;
    let (second, _) = replay(&index, &queries, &params).await;
    assert_same(&uncached, &first, "a budget of 1 KiB, first pass");
    assert_same(&uncached, &second, "a budget of 1 KiB, second pass");

    // Misses rather than bytes: eviction is not immediate. Moka admits an entry
    // whatever it weighs and reclaims it when it next runs its housekeeping, so
    // a partition read a moment ago can still be served from a budget it does
    // not fit in - which makes the budget a bound on what is held for long
    // rather than a bound at any instant.
    let stats = index.cache_stats().await.unwrap();
    assert!(
        stats.misses > queries.len() as u64,
        "{} lookups missed over two passes of {} queries, so a budget of 1 KiB held everything it \
         was given",
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
    let dataset = coded_dataset(uri).await;
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
    let dataset = coded_dataset(uri).await;
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
    let dataset = coded_dataset(uri).await;
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
