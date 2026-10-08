// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What one merge does to an index that is behind its dataset in every way at
//! once.
//!
//! The other two drivers each answer half the question and have to be run in the
//! right order; this answers all of it in one call, and the tests are largely
//! about the states that used to need two. That a delta folds back into the base
//! rather than being searched beside it forever, that a compaction is repaired
//! where an in-place insert refuses outright, and that after any of it the index
//! holds exactly the dataset's live rows - once each.
//!
//! The fixture appends under a **different seed** than it was built with, so the
//! new rows are new points rather than second copies of the old ones.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_file::version::LanceFileVersion;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, build_index_segment, create_index,
};
use lance_vamana::consolidator::consolidate_index;
use lance_vamana::format::{FORMAT_VERSION, IndexMetadata};
use lance_vamana::inserter::insert_as_segment;
use lance_vamana::io::SegmentWriter;
use lance_vamana::merger::{MergeStats, merge_index};
use lance_vamana::query::{SearchParams, VamanaIndex, WalkMode, committed_segments};
use roaring::RoaringBitmap;
use uuid::Uuid;

mod common;
use common::{
    DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, assert_twins_hold_the_same, brute_force,
    committed_manifests, compact_indexed, live_row_ids, maintained_entry_point_params,
    random_vectors, read_committed_segments, recall, retrained_entry_points, sample_partition,
    stored_entry_points, twin_params, twins, wide_fixture,
};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 8;
const K: usize = 10;
const QUERIES: usize = 32;
const ROWS: usize = 3 * 512;

/// Graph parameters with **no** value in common with [`BuildParams::default`],
/// so that a segment which ignored the base and reached for the defaults is
/// visible.
fn base_graph() -> BuildParams {
    BuildParams {
        max_degree: 12,
        search_list_size: 40,
        alpha: 1.4,
        seed: 7,
    }
}

/// Without entry points: the tests of entry points build them on purpose, and
/// training them in every pass would cost a debug build several times its graph.
async fn indexed_dataset(uri: &str) -> Dataset {
    indexed_dataset_with(
        uri,
        IndexParams::new(VECTOR_COLUMN, PARTITIONS).without_entry_points(),
    )
    .await
}

/// [`indexed_dataset`] built from `params`, with the fixture's own graph.
async fn indexed_dataset_with(uri: &str, params: IndexParams) -> Dataset {
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &params.with_graph_params(base_graph()),
    )
    .await
    .unwrap();
    dataset
}

/// Three more fragments of 512 rows, drawn from a seed the index has never seen.
async fn with_new_rows(uri: &str, seed: u64) -> Dataset {
    DatasetFixture {
        seed,
        ..Default::default()
    }
    .append(uri)
    .await
}

/// Append the given vectors as new rows, so that a batch can be aimed at a
/// chosen region of the space instead of drawn at random.
async fn append_vectors(uri: &str, vectors: &[Vec<f32>]) -> Dataset {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        VECTOR_COLUMN,
        DataType::FixedSizeList(item, VECTOR_DIM),
        true,
    )]));
    let array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors
            .iter()
            .map(|vector| Some(vector.iter().map(|value| Some(*value)).collect::<Vec<_>>()))
            .collect::<Vec<_>>(),
        VECTOR_DIM,
    );
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn search() -> SearchParams {
    SearchParams::new(K)
        .with_mode(WalkMode::Exact)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(64)
}

async fn committed_uuids(dataset: &Dataset) -> Vec<Uuid> {
    committed_segments(dataset, INDEX_NAME)
        .await
        .unwrap()
        .iter()
        .map(|index| index.uuid)
        .collect()
}

/// Mean recall@10 against Lance's own exhaustive search over the whole dataset,
/// which is what makes an unindexed row count against the index rather than
/// being invisible to the comparison.
async fn measured_recall(dataset: &Dataset) -> f64 {
    let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
    let queries = random_vectors(QUERIES, 4242);
    let mut total = 0.0;
    for query in &queries {
        let truth = brute_force(dataset, query, K).await;
        let answer = index.search(query, &search()).await.unwrap();
        let found = answer
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(found.len(), K, "the index returned a short answer");
        total += recall(&found, &truth);
    }
    total / queries.len() as f64
}

/// Every row id the whole index physically stores, and how many vertex slots
/// they occupy.
///
/// The set and the count are both needed: a row folded in without being taken
/// out of where it came from leaves the set unchanged and the count too high.
async fn stored_row_ids(dataset: &Dataset) -> (HashSet<u64>, usize) {
    let segments = read_committed_segments(dataset, INDEX_NAME).await;
    let slots = segments
        .iter()
        .flat_map(|segment| segment.partitions.values())
        .map(|partition| partition.len())
        .sum();
    let rows = segments
        .iter()
        .flat_map(|segment| segment.partitions.values())
        .flat_map(|partition| partition.graph().row_ids().iter().copied())
        .collect();
    (rows, slots)
}

/// A delta stops being a second index to probe and becomes part of the first.
#[tokio::test]
async fn a_delta_is_folded_and_the_index_keeps_one_segment() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    let before = committed_uuids(&dataset).await;
    assert_eq!(before.len(), 2, "the fixture is supposed to have a delta");

    // Per partition, the smaller side is lifted into the larger graph rather
    // than the delta always into the base: linking the few into the many is both
    // cheaper and better, and a delta can hold more of a partition than the base
    // does. Which is why this is the sum over partitions and not simply the size
    // of the delta.
    let before_merge = read_committed_segments(&dataset, INDEX_NAME).await;
    let lifted = (0..PARTITIONS)
        .map(|partition_id| {
            let sizes = before_merge
                .iter()
                .filter_map(|segment| segment.partitions.get(&partition_id))
                .map(|partition| partition.len())
                .collect::<Vec<_>>();
            sizes.iter().sum::<usize>() - sizes.iter().max().copied().unwrap_or_default()
        })
        .sum::<usize>();
    assert!(lifted > 0, "the fixture folds nothing");

    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.segments_folded, 2, "{stats:?}");
    assert_eq!(
        stats.vertices_folded, lifted,
        "the smaller side of every partition was supposed to be lifted into the larger: {stats:?}"
    );
    assert_eq!(stats.fragments_indexed, 0, "{stats:?}");
    assert_eq!(stats.vertices_removed, 0, "{stats:?}");
    assert_eq!(
        stats.partitions_rebuilt, 0,
        "nothing was deleted, so nothing could tear: {stats:?}"
    );
    assert!(stats.partitions_written > 0, "{stats:?}");

    let after = committed_uuids(&dataset).await;
    assert_eq!(after.len(), 1, "the index did not come out as one segment");
    assert!(
        !before.contains(&after[0]),
        "the merge committed one of the segments it was folding"
    );
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index.covered_fragments(),
        &(0..6).collect::<RoaringBitmap>(),
        "the merged segment does not cover exactly the dataset"
    );
    assert!(measured_recall(&dataset).await >= 0.95);
}

/// Nothing is lost and nothing is duplicated by the fold.
///
/// Three segments rather than two, so that a partition is assembled from more
/// than one source and the pass has to merge three sorted partition lists rather
/// than pair one against another.
#[tokio::test]
async fn folding_holds_every_row_the_segments_held_exactly_once() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    let mut dataset = with_new_rows(uri, 1234).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(committed_uuids(&dataset).await.len(), 3);

    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.segments_folded, 3, "{stats:?}");
    assert_eq!(committed_uuids(&dataset).await.len(), 1);

    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset).await;
    assert_eq!(
        slots,
        live.len(),
        "the index stores {slots} vertices for {} rows",
        live.len()
    );
    assert_eq!(
        rows,
        live.iter().copied().collect::<HashSet<_>>(),
        "the index does not hold exactly the dataset's rows"
    );
}

/// The headline: deleted rows out, appended rows in, one call, no order to get
/// wrong.
#[tokio::test]
async fn merging_takes_out_the_deleted_and_puts_in_the_new_in_one_call() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    dataset.delete("_rowid % 5 == 0").await.unwrap();
    let deleted = ROWS - live_row_ids(&dataset).await.len();
    assert!(deleted > 0, "the fixture deleted nothing");

    let mut dataset = with_new_rows(uri, 99).await;
    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.segments_folded, 1, "{stats:?}");
    assert_eq!(stats.fragments_indexed, 3, "{stats:?}");
    assert_eq!(stats.vectors_inserted, ROWS, "{stats:?}");
    assert_eq!(stats.vertices_removed, deleted, "{stats:?}");
    assert_eq!(stats.vertices_folded, 0, "there was no delta: {stats:?}");

    // Both halves happened, and the tombstones are gone rather than carried.
    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset)
        .await
        .into_iter()
        .collect::<HashSet<_>>();
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
    assert_eq!(rows, live);

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    for query in random_vectors(8, 31) {
        for neighbor in index.search(&query, &search()).await.unwrap().neighbors {
            assert!(
                live.contains(&neighbor.row_addr),
                "a deleted row came back after the merge"
            );
        }
    }
    assert!(measured_recall(&dataset).await >= 0.95);

    // The new segment has to declare the fragments it just indexed, not only the
    // ones it inherited. It answers for their rows either way - the vertices are
    // stored - so the miss would not show up here until the next round of
    // maintenance indexed them a second time.
    let again = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        again,
        MergeStats::default(),
        "the merged segment left work behind, so a maintenance loop would repeat it"
    );
}

/// Merging an index that leaves its vectors to the dataset is merging its twin
/// that keeps them: a delta folded, deleted rows taken out and new rows put in,
/// in one call, into the same graphs and codes.
#[tokio::test]
async fn an_index_without_vectors_merges_as_its_twin_does() {
    let dir = tempfile::tempdir().unwrap();
    let mut twins = twins(
        dir.path(),
        &wide_fixture(),
        INDEX_NAME,
        &twin_params(PARTITIONS),
    )
    .await;
    let mut stats = Vec::new();
    for (uri, dataset) in &mut twins {
        *dataset = DatasetFixture {
            seed: 98,
            ..wide_fixture()
        }
        .append(uri)
        .await;
        insert_as_segment(dataset, INDEX_NAME).await.unwrap();
        dataset.delete("_rowid % 5 == 0").await.unwrap();
        *dataset = DatasetFixture {
            seed: 99,
            ..wide_fixture()
        }
        .append(uri)
        .await;
        stats.push(merge_index(dataset, INDEX_NAME).await.unwrap());
    }
    assert_eq!(stats[0], stats[1]);
    assert!(
        stats[0].segments_folded == 2
            && stats[0].vertices_folded > 0
            && stats[0].vertices_removed > 0
            && stats[0].vectors_inserted > 0,
        "{:?}",
        stats[0]
    );
    assert_twins_hold_the_same(&twins[0].1, &twins[1].1, INDEX_NAME).await;
}

/// Over Lance 2.0 data files, which no offset reaches, the live rows a merge
/// measures are read through Lance's take, and the merge folds as its twin's
/// does.
#[tokio::test]
async fn an_index_without_vectors_merges_over_lance_2_0_files_as_its_twin_does() {
    let dir = tempfile::tempdir().unwrap();
    let written_as_2_0 = |seed| DatasetFixture {
        seed,
        storage_version: Some(LanceFileVersion::V2_0),
        ..wide_fixture()
    };
    let mut twins = twins(
        dir.path(),
        &written_as_2_0(wide_fixture().seed),
        INDEX_NAME,
        &twin_params(PARTITIONS),
    )
    .await;
    let mut stats = Vec::new();
    for (uri, dataset) in &mut twins {
        dataset.delete("_rowid % 5 == 0").await.unwrap();
        *dataset = written_as_2_0(99).append(uri).await;
        stats.push(merge_index(dataset, INDEX_NAME).await.unwrap());
    }
    assert_eq!(stats[0], stats[1]);
    assert!(
        stats[0].vertices_removed > 0 && stats[0].vectors_inserted > 0,
        "{:?}",
        stats[0]
    );
    assert_twins_hold_the_same(&twins[0].1, &twins[1].1, INDEX_NAME).await;
}

/// After a deferred compaction the deleted rows' vertices keep the addresses
/// they were stored under, in fragments the compaction took away. A merge of an
/// index that leaves its vectors to the dataset reads only the live vertices,
/// where their rows moved to, and folds as its twin does.
#[tokio::test]
async fn an_index_without_vectors_merges_after_a_deferred_compaction_as_its_twin_does() {
    let dir = tempfile::tempdir().unwrap();
    let mut twins = twins(
        dir.path(),
        &wide_fixture(),
        INDEX_NAME,
        &twin_params(PARTITIONS),
    )
    .await;
    let mut stats = Vec::new();
    for (uri, dataset) in &mut twins {
        dataset.delete("_rowid % 11 == 0").await.unwrap();
        let metrics = compact_indexed(dataset).await;
        assert!(metrics.fragments_removed > 0, "{metrics:?}");
        *dataset = DatasetFixture {
            seed: 99,
            ..wide_fixture()
        }
        .append(uri)
        .await;
        stats.push(merge_index(dataset, INDEX_NAME).await.unwrap());
    }
    assert_eq!(stats[0], stats[1]);
    assert!(
        stats[0].vertices_removed > 0 && stats[0].vectors_inserted > 0,
        "{:?}",
        stats[0]
    );
    assert_twins_hold_the_same(&twins[0].1, &twins[1].1, INDEX_NAME).await;
}

/// A deferred compaction moves every row the index stores, and the index follows
/// the move rather than losing the rows. The merge writes every partition out
/// again at the new addresses, and nothing is taken out of a graph or linked back
/// into one to get there.
#[tokio::test]
async fn merging_readdresses_what_a_deferred_compaction_moved() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let metrics = compact_indexed(&mut dataset).await;
    assert!(metrics.fragments_removed > 0, "{metrics:?}");

    let mut dataset = Dataset::open(uri).await.unwrap();
    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        (
            stats.vertices_removed,
            stats.vectors_inserted,
            stats.vertices_folded,
            stats.partitions_written,
            stats.partitions_copied,
        ),
        (0, 0, 0, 0, 0),
        "{stats:?}"
    );
    assert!(stats.partitions_readdressed > 0, "{stats:?}");

    assert_eq!(committed_uuids(&dataset).await.len(), 1);
    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset)
        .await
        .into_iter()
        .collect::<HashSet<_>>();
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
    assert_eq!(rows, live);
    assert!(measured_recall(&dataset).await >= 0.95);
    assert_eq!(
        merge_index(&mut dataset, INDEX_NAME).await.unwrap(),
        MergeStats::default(),
        "the merged segment still depends on the record of the move"
    );
}

/// Nothing to do means no commit at all, not an empty one: this is meant to be
/// safe to call on a schedule.
/// The plainest thing a merge is asked to do: rows were appended, nothing was
/// deleted, and there is one segment. Every branch that decides "nothing to do"
/// has to let this through.
#[tokio::test]
async fn merging_indexes_new_rows_when_nothing_was_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 17).await;

    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.fragments_indexed, 3, "{stats:?}");
    assert_eq!(stats.vectors_inserted, ROWS, "{stats:?}");
    assert_eq!(stats.vertices_removed, 0, "nothing was deleted: {stats:?}");
    assert!(stats.partitions_written > 0, "{stats:?}");

    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset)
        .await
        .into_iter()
        .collect::<HashSet<_>>();
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
    assert_eq!(rows, live, "the appended rows are not all in the index");
    assert!(measured_recall(&dataset).await >= 0.95);
}

#[tokio::test]
async fn merging_when_there_is_nothing_to_do_commits_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let before = committed_uuids(&dataset).await;

    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats, Default::default(), "{stats:?}");
    assert_eq!(committed_uuids(&dataset).await, before);
}

/// A partition the base no longer has and a delta does crosses over as the bytes
/// it already is - no decode, no arithmetic at all.
///
/// Reached by emptying one partition of the base and appending the very vectors
/// that were deleted, so they route back to the same centroid by construction
/// rather than by luck. Every other partition is untouched, which makes this the
/// case where a whole merge costs zero distance computations.
#[tokio::test]
async fn a_partition_only_a_delta_holds_crosses_over_undecoded() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    let (emptied, partition) = segments[0]
        .partitions
        .iter()
        .min_by_key(|(_, partition)| partition.len())
        .unwrap();
    let doomed = partition.graph().row_ids().to_vec();
    let vectors = (0..partition.len() as u32)
        .map(|local| partition.vector(local).unwrap().to_vec())
        .collect::<Vec<_>>();
    let emptied = *emptied;

    dataset
        .delete(&format!(
            "_rowid IN ({})",
            doomed
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .await
        .unwrap();
    let mut dataset = append_vectors(uri, &vectors).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        stats.partitions_copied, PARTITIONS as usize,
        "every partition was either untouched or held only by the delta: {stats:?}"
    );
    assert_eq!(stats.partitions_written, 0, "{stats:?}");
    assert_eq!(
        stats.comparisons, 0,
        "a merge of nothing but copies spent distance computations: {stats:?}"
    );
    assert_eq!(stats.vertices_removed, doomed.len(), "{stats:?}");
    assert_eq!(stats.vertices_folded, 0, "{stats:?}");

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert_eq!(segments.len(), 1);
    assert_eq!(
        segments[0].partitions[&emptied].len(),
        vectors.len(),
        "partition {emptied} did not come across from the delta"
    );
    assert!(measured_recall(&dataset).await >= 0.95);
}

/// The invariant the fixed-width layout is, checked against the files rather
/// than against what was in memory when they were written.
#[tokio::test]
async fn every_partition_read_back_respects_the_degree() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    dataset.delete("_rowid % 7 == 0").await.unwrap();
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    let mut dataset = with_new_rows(uri, 1234).await;
    merge_index(&mut dataset, INDEX_NAME).await.unwrap();

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert_eq!(segments.len(), 1);
    for (partition_id, partition) in &segments[0].partitions {
        let graph = partition.graph();
        assert_eq!(graph.max_degree(), base_graph().max_degree);
        for vertex in 0..graph.len() as u32 {
            let neighbors = graph.neighbors(vertex).unwrap();
            assert!(
                neighbors.len() <= base_graph().max_degree as usize,
                "partition {partition_id} vertex {vertex} has degree {}",
                neighbors.len()
            );
            assert!(
                neighbors.iter().all(|id| (*id as usize) < graph.len()),
                "partition {partition_id} vertex {vertex} points outside the partition"
            );
        }
    }
}

/// Folding by partition number is only sound while the segments share one
/// numbering, and nothing about a segment records whose centroids it was written
/// under. So the centroids themselves are compared, before anything is read.
#[tokio::test]
async fn merging_segments_that_disagree_is_refused() {
    // Without codes on both sides of both halves: two segments built apart mint
    // codes of their own, and opening the index would refuse them before the
    // merge got to say what it refuses them for.
    let without_codes = || IndexParams::new(VECTOR_COLUMN, PARTITIONS).without_codes();
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset_with(uri, without_codes()).await;

    // Same width, its own router: partition n of this segment is a different
    // region of the space than partition n of the base.
    let mut dataset = with_new_rows(uri, 99).await;
    let (stranger, _) = build_index_segment(
        &dataset,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS - 3)
            .with_graph_params(base_graph())
            .without_codes(),
        &[3, 4, 5],
    )
    .await
    .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![stranger])
        .await
        .unwrap();
    let error = merge_index(&mut dataset, INDEX_NAME)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("centroids of its own"),
        "the refusal does not say what is wrong: {error}"
    );

    // And a segment of another width, which opening an index permits and folding
    // does not.
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset_with(uri, without_codes()).await;
    let mut dataset = with_new_rows(uri, 99).await;
    let (odd, _) = build_index_segment(
        &dataset,
        &without_codes().with_graph_params(BuildParams {
            max_degree: 20,
            ..base_graph()
        }),
        &[3, 4, 5],
    )
    .await
    .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![odd])
        .await
        .unwrap();
    let error = merge_index(&mut dataset, INDEX_NAME)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("degree 20"),
        "the refusal does not say what is wrong: {error}"
    );
}

/// A merged index answers as well as one built over the same rows from scratch.
///
/// The number that says folding is not a second-class way to arrive at an index.
/// Both arms end with the same dataset and the same parameters, and differ only
/// in whether the graph was grown or built.
#[tokio::test]
async fn a_merged_index_answers_as_well_as_a_rebuild() {
    let grown_dir = tempfile::tempdir().unwrap();
    let grown_uri = grown_dir.path().to_str().unwrap();
    let mut grown = indexed_dataset(grown_uri).await;
    grown.delete("_rowid % 5 == 0").await.unwrap();
    let mut grown = with_new_rows(grown_uri, 99).await;
    insert_as_segment(&mut grown, INDEX_NAME).await.unwrap();
    let mut grown = with_new_rows(grown_uri, 1234).await;
    merge_index(&mut grown, INDEX_NAME).await.unwrap();
    assert_eq!(committed_uuids(&grown).await.len(), 1);
    let grown_recall = measured_recall(&grown).await;

    let built_dir = tempfile::tempdir().unwrap();
    let built_uri = built_dir.path().to_str().unwrap();
    let mut built = indexed_dataset(built_uri).await;
    built.delete("_rowid % 5 == 0").await.unwrap();
    with_new_rows(built_uri, 99).await;
    let mut built = with_new_rows(built_uri, 1234).await;
    built.drop_index(INDEX_NAME).await.unwrap();
    create_index(
        &mut built,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS).with_graph_params(base_graph()),
    )
    .await
    .unwrap();
    let built_recall = measured_recall(&built).await;

    assert!(
        grown_recall >= built_recall - 0.02,
        "a merged index answers at {grown_recall} where one built whole answers at {built_recall}"
    );
}

/// [`indexed_dataset`] with entry points ([`maintained_entry_point_params`]).
async fn indexed_with_entry_points(uri: &str) -> Dataset {
    indexed_dataset_with(
        uri,
        IndexParams::new(VECTOR_COLUMN, PARTITIONS)
            .with_entry_point_params(maintained_entry_point_params()),
    )
    .await
}

/// A merge trains the entry points of every partition it writes - a base's
/// with a delta's folded in and a deleted entry point taken out, and one no
/// segment holds that only new rows fill - so that afterwards every list is
/// what training at open trains. The delta is aimed at the folded partition
/// alone, so that no segment holds the emptied one when the merge runs.
#[tokio::test]
async fn a_merge_trains_the_entry_points_of_what_it_writes() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_with_entry_points(uri).await;
    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    // The smallest partition is emptied and consolidated away, and its rows
    // come back as new ones, which only a merge's new-partition path takes in.
    let (emptied, partition) = segments[0]
        .partitions
        .iter()
        .min_by_key(|(_, partition)| partition.len())
        .unwrap();
    let emptied = *emptied;
    let doomed = partition.graph().row_ids().to_vec();
    let vectors = (0..partition.len() as u32)
        .map(|local| partition.vector(local).unwrap().to_vec())
        .collect::<Vec<_>>();
    dataset
        .delete(&format!(
            "_rowid IN ({})",
            doomed
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .await
        .unwrap();
    let consolidated = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(consolidated.partitions_dropped, 1, "{consolidated:?}");

    let base = &read_committed_segments(&dataset, INDEX_NAME).await[0];
    let folded = base
        .manifest
        .partitions()
        .iter()
        .find(|entry| entry.partition_id != emptied)
        .unwrap();
    let aimed = {
        let partition = &base.partitions[&folded.partition_id];
        (0..partition.len() as u32)
            .map(|local| partition.vector(local).unwrap().to_vec())
            .collect::<Vec<_>>()
    };
    let row =
        base.partitions[&folded.partition_id].graph().row_ids()[folded.entry_points[0] as usize];
    let mut dataset = append_vectors(uri, &aimed).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    dataset.delete(&format!("_rowid = {row}")).await.unwrap();
    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert_eq!(
        segments[1].partitions.keys().copied().collect::<Vec<_>>(),
        vec![folded.partition_id],
        "the delta holds other partitions than the one it was aimed at"
    );
    assert!(
        segments
            .iter()
            .all(|segment| !segment.partitions.contains_key(&emptied)),
        "a segment holds partition {emptied}, so the merge folds it rather than writes it new"
    );
    let mut dataset = append_vectors(uri, &vectors).await;

    let stats = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert!(
        stats.partitions_written > 0 && stats.vertices_removed > 0,
        "{stats:?}"
    );
    let stored = stored_entry_points(&dataset, INDEX_NAME).await;
    let segment = committed_uuids(&dataset).await[0];
    assert!(
        !stored[&(segment, emptied)].is_empty(),
        "partition {emptied} came back without entry points"
    );
    assert_eq!(stored, retrained_entry_points(&dataset, INDEX_NAME).await);
}

/// A merge of segments whose entry points were trained under different
/// parameters is refused before anything is read: a partition it writes would
/// be retrained under the base's whatever its segment used, and one it copies
/// is a list the merged segment's metadata does not describe. Every build of
/// one index inherits its base's, so the other segment is written by hand.
#[tokio::test]
async fn a_merge_of_segments_trained_apart_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_with_entry_points(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    let (base_uuid, base) = committed_manifests(&dataset, INDEX_NAME).await.remove(0);
    let covered = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .filter(|id| !base.metadata().fragments.contains(id))
        .collect::<Vec<_>>();
    let other = maintained_entry_point_params().with_seed(8);
    let uuid = Uuid::new_v4();
    let mut writer = SegmentWriter::new(
        dataset.object_store(None).await.unwrap(),
        dataset.indices_dir().join(uuid.to_string()),
        IndexMetadata {
            fragments: covered.clone(),
            entry_point_params: Some(other.clone()),
            ..base.metadata().clone()
        },
        base.ivf().clone(),
    );
    writer
        .write_partition(
            0,
            0,
            Arc::from([]),
            &sample_partition(base.metadata().max_degree, 8, VECTOR_DIM as u32),
        )
        .await
        .unwrap();
    writer.finish().await.unwrap();
    dataset
        .commit_existing_index_segments(
            INDEX_NAME,
            VECTOR_COLUMN,
            vec![lance::index::IndexSegment::new(
                uuid,
                covered,
                [dataset.schema().field(VECTOR_COLUMN).unwrap().id],
                Arc::new(prost_types::Any {
                    type_url: INDEX_DETAILS_TYPE_URL.to_string(),
                    value: Vec::new(),
                }),
                FORMAT_VERSION as i32,
                dataset.manifest.version,
                vec![],
            )],
        )
        .await
        .unwrap();
    // Rows no segment covers, which the merge would read as arrivals before
    // a refusal that came after them.
    let indexed = dataset.get_fragments().len();
    let mut dataset = with_new_rows(uri, 1234).await;
    assert!(dataset.get_fragments().len() > indexed, "nothing arrived");
    let version = dataset.manifest.version;

    // The data files out of reach, so that a refusal after the arrivals were
    // read would fail on them instead.
    let data_files = dir.path().join("data");
    assert!(std::fs::read_dir(&data_files).unwrap().count() > 0);
    let away = dir.path().join("data.away");
    std::fs::rename(&data_files, &away).unwrap();
    let error = merge_index(&mut dataset, INDEX_NAME).await.unwrap_err();
    std::fs::rename(&away, &data_files).unwrap();
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "segment {uuid} trained its entry points under {other}"
        )) && message.contains(&format!("where segment {base_uuid} trained them under")),
        "{message}"
    );
    assert_eq!(
        dataset.manifest.version, version,
        "the refused merge committed"
    );
}
