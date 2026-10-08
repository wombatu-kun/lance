// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What indexing a dataset's new rows does to the index that holds the old ones.
//!
//! The build path already knows how to make a segment over a chosen set of
//! fragments; what can only be asked here is whether the result is *the index
//! the dataset has*. That the new segment stands beside the old rather than
//! replacing it, that it inherits the base's routing so the two are comparable,
//! that no row ends up stored twice, and that the appended rows are answered for
//! afterwards when they were not before.
//!
//! The fixture appends under a **different seed** than it was built with, so the
//! new rows are new points rather than second copies of the old ones. With
//! duplicates the recall numbers below would still move, but they would move
//! because ties broke differently.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::utils::address::RowAddress;
use lance_file::version::LanceFileVersion;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, build_index_segment, create_index, live_fragments};
use lance_vamana::consolidator::consolidate_index;
use lance_vamana::entry_points::EntryPointParams;
use lance_vamana::format::VectorSource;
use lance_vamana::inserter::{InsertStats, insert_as_segment, insert_in_place};
use lance_vamana::merger::merge_index;
use lance_vamana::query::{SearchParams, VamanaIndex, WalkMode, committed_segments};
use roaring::RoaringBitmap;
use uuid::Uuid;

mod common;
use common::{
    DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, WIDE_DIM, assert_twins_hold_the_same, brute_force,
    commit_overlay_of, live_row_ids, maintained_entry_point_params, random_vectors,
    random_vectors_of, read_committed_segments, recall, retrained_entry_points,
    stored_entry_points, twin_params, twins, wide_fixture,
};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 8;
const K: usize = 10;
const QUERIES: usize = 32;

/// Graph parameters with **no** value in common with [`BuildParams::default`].
///
/// A delta is supposed to be built to the base's shape, and against a base built
/// with the defaults that claim is untestable: a delta that ignored the base
/// entirely and reached for `BuildParams::default()` would agree with it on
/// every field.
fn base_graph() -> BuildParams {
    BuildParams {
        max_degree: 12,
        search_list_size: 40,
        alpha: 1.4,
        seed: 7,
    }
}

/// Three fragments of 512 rows, indexed over eight partitions, without entry
/// points: the tests of entry points build them on purpose, and training them
/// in every pass would cost a debug build several times its graph.
async fn indexed_dataset(uri: &str) -> Dataset {
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS)
            .with_graph_params(base_graph())
            .without_entry_points(),
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
/// Across every segment, because that is the only level at which "stored twice"
/// is a question: two segments each holding a row look perfectly ordinary from
/// inside either one.
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

/// The point of the whole thing: rows appended after the build are invisible,
/// and indexing them makes them findable without touching what was there.
///
/// Measured, the two numbers are 0.5 and 1.0 exactly, over a per-query spread of
/// 0.2 to 0.9 before. What that means is that this fixture cannot fail on graph
/// *quality*: eight partitions of some four hundred rows searched with a beam of
/// 64 is very nearly exhaustive, and any working graph answers perfectly. It is
/// a test of reach, not of quality; quality is measured where the graph is.
#[tokio::test]
async fn appended_rows_are_answered_for_after_they_are_indexed() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;

    // Half the dataset is outside the index, so half the true neighbours are
    // unreachable however good the graph is.
    let before = measured_recall(&dataset).await;
    assert!(
        (0.4..0.6).contains(&before),
        "half the rows are unindexed, so recall should be about a half, got {before}"
    );

    let stats = insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.fragments_indexed, 3, "{stats:?}");
    assert_eq!(stats.vectors, 3 * 512, "{stats:?}");
    assert!(
        stats.partitions_created > 0 && stats.partitions_created <= PARTITIONS as usize,
        "{stats:?}"
    );
    assert!(stats.comparisons > 0, "{stats:?}");

    let after = measured_recall(&dataset).await;
    assert!(
        after >= 0.95,
        "the appended rows are indexed now, so recall should be near one, got {after}"
    );
}

/// The base is not replaced, not rewritten and not reopened: a delta is an
/// addition, and the fragments it covers are exactly the ones nothing covered.
#[tokio::test]
async fn the_new_segment_stands_beside_the_one_that_was_there() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    let base = committed_uuids(&dataset).await;
    assert_eq!(base.len(), 1);

    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    let after = committed_uuids(&dataset).await;
    assert_eq!(after.len(), 2, "the delta did not join the base");
    assert!(
        after.contains(&base[0]),
        "the base segment was replaced instead of kept"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index.covered_fragments(),
        &(0..6).collect::<RoaringBitmap>(),
        "the index does not cover exactly the dataset"
    );

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    let delta = segments
        .iter()
        .find(|segment| segment.uuid != base[0])
        .unwrap();
    assert_eq!(
        delta.manifest.metadata().fragments,
        vec![3, 4, 5],
        "the delta recorded a coverage other than the fragments it read"
    );
}

/// No row is stored by two segments.
///
/// The set and the slot count are both needed: a row indexed into both segments
/// leaves the set unchanged and the count too high, and asserting on either
/// alone would miss it.
#[tokio::test]
async fn every_row_is_stored_by_exactly_one_segment() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

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

/// A delta routes by the base's centroids and is built to the base's shape.
///
/// The centroids are the load-bearing half: with a router of its own a delta
/// would still answer correctly, but partition 17 of the two segments would hold
/// unrelated regions of the space and folding one into the other would mean
/// re-routing every row.
#[tokio::test]
async fn a_delta_inherits_the_routing_and_the_shape_of_its_base() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    let base_uuid = committed_uuids(&dataset).await[0];
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    let base = segments
        .iter()
        .find(|segment| segment.uuid == base_uuid)
        .unwrap();
    let delta = segments
        .iter()
        .find(|segment| segment.uuid != base_uuid)
        .unwrap();

    assert_eq!(
        delta.manifest.ivf().centroids,
        base.manifest.ivf().centroids,
        "the delta trained a router of its own"
    );
    let (base_shape, delta_shape) = (base.manifest.metadata(), delta.manifest.metadata());
    assert_eq!(
        (
            delta_shape.max_degree,
            delta_shape.search_list_size,
            delta_shape.alpha,
            delta_shape.distance_type,
            delta_shape.dimension,
        ),
        (
            base_shape.max_degree,
            base_shape.search_list_size,
            base_shape.alpha,
            base_shape.distance_type,
            base_shape.dimension,
        ),
        "the delta was built to a shape of its own"
    );
    assert!(
        delta
            .partitions
            .keys()
            .all(|partition_id| *partition_id < PARTITIONS),
        "the delta wrote a partition outside the base's numbering"
    );
}

/// A second delta inherits from the base, not from the delta before it.
///
/// With one segment there is no choice to get wrong, so this is the only place
/// the rule is visible at all. Getting it wrong would not break anything today -
/// the first delta carries the base's centroids anyway - but it would as soon as
/// a delta is ever built any other way, and by then the segments that disagree
/// are already on disk.
#[tokio::test]
async fn a_second_delta_inherits_from_the_base_and_not_from_the_first() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;

    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    let mut dataset = with_new_rows(uri, 1234).await;
    let stats = insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.fragments_indexed, 3, "{stats:?}");

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert_eq!(segments.len(), 3);
    let base = &segments[0];
    assert_eq!(
        base.manifest.metadata().fragments,
        vec![0, 1, 2],
        "the widest segment is not the base"
    );
    for segment in &segments[1..] {
        assert_eq!(
            segment.manifest.ivf().centroids,
            base.manifest.ivf().centroids,
            "segment {} routes by centroids the base does not have",
            segment.uuid
        );
    }
    assert_eq!(measured_recall(&dataset).await, 1.0);
}

/// Which segment a delta copies from is decided by coverage, not by recency:
/// the widest, and on a tie the one the manifest lists first.
///
/// Invisible while every segment of an index came out of this crate's own
/// drivers, because they all already carry the base's numbers. It becomes
/// visible the moment a segment arrives any other way, and `VamanaIndex::open`
/// explicitly permits that for the degree and the pruning slack - so the
/// fixture here commits a same-width segment with a degree of its own and then
/// asks what the *next* delta was built to.
#[tokio::test]
async fn a_delta_takes_its_shape_from_the_base_and_not_from_the_newest_segment() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    // Without codes, base and odd segment alike: two segments built apart mint
    // codes of their own and would not open as one index, and what is asked
    // here is the degree.
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS)
            .with_graph_params(base_graph())
            .without_codes(),
    )
    .await
    .unwrap();

    let mut dataset = with_new_rows(uri, 99).await;
    let (odd, _) = build_index_segment(
        &dataset,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS)
            .with_graph_params(BuildParams {
                max_degree: 20,
                ..base_graph()
            })
            .without_codes(),
        &[3, 4, 5],
    )
    .await
    .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![odd])
        .await
        .unwrap();

    let mut dataset = with_new_rows(uri, 1234).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert_eq!(
        segments
            .iter()
            .map(|segment| segment.manifest.metadata().max_degree)
            .collect::<Vec<_>>(),
        vec![base_graph().max_degree, 20, base_graph().max_degree],
        "the manifest lists the base first, the odd segment second, and the new \
         delta - built to the base's degree, not the odd one's - third"
    );
}

/// Nothing new means no commit at all, not an empty one: this is meant to be
/// safe to call on a schedule.
#[tokio::test]
async fn indexing_when_nothing_is_new_commits_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let uuids = committed_uuids(&dataset).await;
    let version = dataset.manifest.version;

    let stats = insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    assert_eq!(stats, InsertStats::default(), "{stats:?}");
    assert_eq!(committed_uuids(&dataset).await, uuids);
    assert_eq!(
        dataset.manifest.version, version,
        "an empty insert still moved the dataset forward"
    );
}

/// The base segment after a deferred compaction rewrote part of what it covers,
/// three fragments appended after that, and a delta indexed over them.
///
/// This is where Lance and this crate used to part ways: Lance credits the base
/// with the fragment the rows moved into, so a delta over that fragment, or a
/// consolidation that left it out, was refused for orphaning fragments.
async fn partially_compacted_with_a_delta(uri: &str) -> Dataset {
    let mut dataset = indexed_dataset(uri).await;
    // One deleted row makes fragment 0, and only fragment 0, worth rewriting, and
    // it is the row the rewrite drops.
    dataset.delete("_rowid = 7").await.unwrap();
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            target_rows_per_fragment: 512,
            materialize_deletions_threshold: 0.001,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        (metrics.fragments_removed, metrics.fragments_added),
        (1, 1),
        "{metrics:?}"
    );

    let compacted = live_fragments(&dataset)
        .into_iter()
        .collect::<RoaringBitmap>();
    let mut dataset = with_new_rows(uri, 99).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.covered_fragments(), &compacted);

    let refusal = insert_in_place(&mut dataset, INDEX_NAME)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains("deferred compaction") && refusal.contains("consolidate the index first"),
        "{refusal}"
    );

    let inserted = insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(inserted.fragments_indexed, 3, "{inserted:?}");
    dataset
}

/// What the repair after such a compaction has to leave behind: every stored
/// address a live row, once Lance has forgotten the move.
async fn assert_only_live_rows_once_the_move_is_forgotten(mut dataset: Dataset, uri: &str) {
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let dataset = Dataset::open(uri).await.unwrap();
    assert!(
        dataset
            .frag_reuse_index()
            .await
            .unwrap()
            .is_none_or(|remap| remap.is_empty()),
        "Lance kept the record of the move, so nothing below depends on the rewrite"
    );
    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset)
        .await
        .into_iter()
        .collect::<HashSet<_>>();
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
    assert_eq!(rows, live);
    assert!(measured_recall(&dataset).await >= 0.95);
}

/// The index follows the move, so the delta goes through beside the base, and
/// consolidation writes the moved addresses into the base, after which Lance can
/// forget the move without the index noticing.
#[tokio::test]
async fn a_partial_deferred_compaction_leaves_every_repair_open() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = partially_compacted_with_a_delta(uri).await;

    let consolidated = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        (
            consolidated.segments_rewritten,
            consolidated.segments_untouched,
            consolidated.vertices_removed,
            consolidated.partitions_copied,
            consolidated.partitions_consolidated + consolidated.partitions_rebuilt,
        ),
        (1, 1, 1, 0, 1),
        "{consolidated:?}"
    );
    assert!(consolidated.partitions_readdressed > 0, "{consolidated:?}");
    assert_only_live_rows_once_the_move_is_forgotten(dataset, uri).await;
}

/// Folding the delta into the base reads both into one graph, so the base's
/// vertices have to be readdressed before they are merged with the delta's, not
/// only in a partition that is carried across whole.
#[tokio::test]
async fn merging_a_partially_compacted_index_readdresses_what_it_folds() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = partially_compacted_with_a_delta(uri).await;

    let merged = merge_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        (
            merged.segments_folded,
            merged.vertices_removed,
            merged.partitions_copied,
        ),
        (2, 1, 0),
        "{merged:?}"
    );
    assert!(merged.partitions_written > 0, "{merged:?}");
    assert_only_live_rows_once_the_move_is_forgotten(dataset, uri).await;
}

/// The base is rewritten, not joined: the index keeps one segment, under a new
/// uuid, covering everything.
#[tokio::test]
async fn inserting_in_place_replaces_the_segment_instead_of_adding_one() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    let before = committed_uuids(&dataset).await;

    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.fragments_indexed, 3, "{stats:?}");
    assert_eq!(stats.vectors, 3 * 512, "{stats:?}");
    assert!(stats.partitions_grown > 0, "{stats:?}");

    let after = committed_uuids(&dataset).await;
    assert_eq!(after.len(), 1, "an in-place insert added a segment");
    assert_ne!(after[0], before[0], "the base was left as it was");

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index.covered_fragments(),
        &(0..6).collect::<RoaringBitmap>()
    );
    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset).await;
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
    assert!(measured_recall(&dataset).await >= 0.95);
}

/// A batch smaller than the partition count leaves most partitions with nothing
/// to do, and those are copied rather than decoded and re-encoded. Both counters
/// have to be non-zero in one run, or the branch that fired is not the one under
/// test.
#[tokio::test]
async fn a_partition_that_drew_nothing_is_copied_not_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = DatasetFixture {
        fragments: 1,
        rows_per_fragment: 4,
        seed: 77,
        ..Default::default()
    }
    .append(uri)
    .await;

    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.vectors, 4, "{stats:?}");
    assert!(stats.partitions_grown > 0, "{stats:?}");
    assert!(stats.partitions_copied > 0, "{stats:?}");
    assert_eq!(
        stats.partitions_grown + stats.partitions_copied + stats.partitions_created,
        PARTITIONS as usize,
        "the counters do not add up to the partitions of the segment: {stats:?}"
    );
    let (rows, slots) = stored_row_ids(&dataset).await;
    assert_eq!((slots, rows.len()), (3 * 512 + 4, 3 * 512 + 4));
}

/// A partition consolidation dropped comes back when a row routes to its
/// centroid again.
///
/// The only way this crate can produce a hole in the partition numbering, and
/// therefore the only way to reach the branch that builds a partition from
/// nothing. The rows appended are the very vectors that were deleted, so they
/// route to the same centroid by construction rather than by luck.
#[tokio::test]
async fn a_partition_consolidation_dropped_is_created_again_by_an_insert() {
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
    let consolidated = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        consolidated.partitions_dropped, 1,
        "the partition was supposed to be emptied: {consolidated:?}"
    );
    assert!(
        !read_committed_segments(&dataset, INDEX_NAME).await[0]
            .partitions
            .contains_key(&emptied),
        "partition {emptied} is still in the segment"
    );

    let mut dataset = append_vectors(uri, &vectors).await;
    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert!(
        stats.partitions_created > 0,
        "the dropped partition was not created again: {stats:?}"
    );
    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert!(
        segments[0].partitions.contains_key(&emptied),
        "partition {emptied} did not come back"
    );
    assert!(measured_recall(&dataset).await >= 0.95);
}

/// The invariant the fixed-width layout is, checked against the files rather
/// than against what was in memory when they were written.
#[tokio::test]
async fn every_partition_read_back_respects_the_degree() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();

    for segment in read_committed_segments(&dataset, INDEX_NAME).await {
        for (partition_id, partition) in &segment.partitions {
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
}

/// A disjoint delta is not named in the commit and survives it.
#[tokio::test]
async fn a_delta_segment_is_left_alone_by_an_in_place_insert() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_dataset(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    let delta = committed_uuids(&dataset).await[1];

    let mut dataset = with_new_rows(uri, 1234).await;
    insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();

    let uuids = committed_uuids(&dataset).await;
    assert_eq!(uuids.len(), 2, "the delta was folded in or dropped");
    assert!(
        uuids.contains(&delta),
        "the delta did not survive the commit"
    );
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index.covered_fragments(),
        &(0..9).collect::<RoaringBitmap>()
    );
    assert!(measured_recall(&dataset).await >= 0.95);
}

/// Deletion and insertion compose: the tombstones the base carries are still
/// tombstones after it has been grown, and the new rows are answerable.
#[tokio::test]
async fn deleting_and_inserting_compose() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    dataset.delete("_rowid % 5 == 0").await.unwrap();

    let mut dataset = with_new_rows(uri, 99).await;
    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert!(stats.partitions_grown > 0, "{stats:?}");

    // The deleted rows are still stored - they are routers - and still absent
    // from every answer.
    let (rows, _) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset)
        .await
        .into_iter()
        .collect::<HashSet<_>>();
    assert!(
        rows.len() > live.len(),
        "the tombstones were quietly dropped"
    );
    assert!(live.is_subset(&rows), "a live row is not in the index");

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    for query in random_vectors(8, 31) {
        for neighbor in index.search(&query, &search()).await.unwrap().neighbors {
            assert!(
                live.contains(&neighbor.row_addr),
                "a deleted row came back after the insert"
            );
        }
    }
    assert!(measured_recall(&dataset).await >= 0.95);

    // And consolidation still clears them afterwards.
    let consolidated = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(consolidated.segments_rewritten, 1, "{consolidated:?}");
    let (rows, slots) = stored_row_ids(&dataset).await;
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
}

/// Inserting in place into an index that leaves its vectors to the dataset is
/// inserting into its twin that keeps them. The insertion walks through the
/// deleted vertices too, so their vectors are read back as well - by offset, the
/// bytes of a deleted row staying in its data file.
#[tokio::test]
async fn an_index_without_vectors_grows_in_place_as_its_twin_does() {
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
        dataset.delete("_rowid % 5 == 0").await.unwrap();
        *dataset = DatasetFixture {
            seed: 99,
            ..wide_fixture()
        }
        .append(uri)
        .await;
        stats.push(insert_in_place(dataset, INDEX_NAME).await.unwrap());
    }
    assert_eq!(stats[0], stats[1]);
    assert!(stats[0].partitions_grown > 0, "{:?}", stats[0]);
    assert_twins_hold_the_same(&twins[0].1, &twins[1].1, INDEX_NAME).await;
}

/// Where the dataset gives vectors only through Lance - a Lance 2.0 data file
/// here - the deleted rows an insertion walks through are read by one Lance
/// take from the fragment without its deletion file, and the index grows in
/// place as its twin keeping the vectors does.
#[tokio::test]
async fn an_index_without_vectors_grows_over_lance_2_0_files_as_its_twin_does() {
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
        stats.push(insert_in_place(dataset, INDEX_NAME).await.unwrap());
    }
    assert_eq!(stats[0], stats[1]);
    assert!(stats[0].partitions_grown > 0, "{:?}", stats[0]);
    assert_twins_hold_the_same(&twins[0].1, &twins[1].1, INDEX_NAME).await;
}

/// An overlay the build saw keeps a fragment's vectors in a file of its own,
/// so no offset reaches them, and the deleted rows an insertion walks through
/// there are read through Lance with the overlay applied - what the build read,
/// and what the twin keeping its vectors holds. Some of the rows deleted are
/// overlaid ones, whose base values the overlay replaced. The new values are
/// drawn as the others are, so that the overlaid rows lie where the insertion
/// walks: values far from every other row would be passed by whatever they
/// were.
#[tokio::test]
async fn an_index_without_vectors_grows_over_an_overlay_as_its_twin_does() {
    let dir = tempfile::tempdir().unwrap();
    let overlaid = (0..60).collect::<Vec<u32>>();
    let replacement = || {
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            random_vectors_of(overlaid.len(), WIDE_DIM, 77)
                .into_iter()
                .map(|vector| Some(vector.into_iter().map(Some).collect::<Vec<_>>())),
            WIDE_DIM,
        )
    };
    let mut twins = Vec::new();
    let mut stats = Vec::new();
    for vector_source in [VectorSource::Index, VectorSource::Dataset] {
        let uri = dir.path().join(vector_source.to_string());
        let uri = uri.to_str().unwrap();
        let dataset = wide_fixture().write(uri).await;
        let mut dataset = commit_overlay_of(dataset, 1, &overlaid, "seen", replacement()).await;
        create_index(
            &mut dataset,
            INDEX_NAME,
            &twin_params(PARTITIONS).with_vector_source(vector_source),
        )
        .await
        .unwrap();
        dataset.delete("_rowid % 5 == 0").await.unwrap();
        let mut dataset = DatasetFixture {
            seed: 99,
            ..wide_fixture()
        }
        .append(uri)
        .await;
        stats.push(insert_in_place(&mut dataset, INDEX_NAME).await.unwrap());
        twins.push(dataset);
    }
    assert_eq!(stats[0], stats[1]);
    assert!(stats[0].partitions_grown > 0, "{:?}", stats[0]);
    assert_twins_hold_the_same(&twins[0], &twins[1], INDEX_NAME).await;
}

/// Rewriting a segment whose fragments are gone would store their vertices under
/// a coverage that no longer names them, where nothing keeps them out of an
/// answer. Refused, with the remedy named.
#[tokio::test]
async fn inserting_in_place_refuses_a_segment_whose_fragments_are_gone() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    dataset.delete("_rowid < 512").await.unwrap();
    assert_eq!(live_fragments(&dataset), vec![1, 2]);

    let mut dataset = with_new_rows(uri, 99).await;
    let error = insert_in_place(&mut dataset, INDEX_NAME)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("the dataset no longer has")
            && error.contains("consolidate the index first"),
        "the refusal does not name the cause and the remedy: {error}"
    );
}

/// [`indexed_dataset`] with entry points ([`maintained_entry_point_params`]).
async fn indexed_with_entry_points(uri: &str) -> Dataset {
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS)
            .with_graph_params(base_graph())
            .with_entry_point_params(maintained_entry_point_params()),
    )
    .await
    .unwrap();
    dataset
}

/// One segment's stored or retrained entry points by partition id.
fn of_segment(
    entry_points: &BTreeMap<(Uuid, u32), Vec<u32>>,
    segment: Uuid,
) -> BTreeMap<u32, Vec<u32>> {
    entry_points
        .iter()
        .filter(|((uuid, _), _)| *uuid == segment)
        .map(|((_, partition_id), entries)| (*partition_id, entries.clone()))
        .collect()
}

/// The row an entry point of `partition_id` of the base stands for.
async fn entry_point_row(dataset: &Dataset, partition_id: u32) -> u64 {
    let segments = read_committed_segments(dataset, INDEX_NAME).await;
    let entry = segments[0].manifest.partition(partition_id).unwrap();
    segments[0].partitions[&partition_id].graph().row_ids()[entry.entry_points[0] as usize]
}

/// An insert in place trains the entry points of the partition it grows over
/// the vertices still live and the new ones, and carries the list of a
/// partition it copies as it is. Rows aimed at one partition grow it alone,
/// and an entry point is deleted from it and from a partition left to be
/// copied: the grown one's list is what training at open trains, the copied
/// one's still names its deleted vertex, which training at open would not.
#[tokio::test]
async fn an_insert_trains_what_it_grows_and_carries_what_it_copies() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_with_entry_points(uri).await;
    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    let mut ids = segments[0].partitions.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    let (grown, copied) = (ids[0], ids[1]);
    let aimed = {
        let partition = &segments[0].partitions[&grown];
        (0..partition.len() as u32)
            .map(|local| partition.vector(local).unwrap().to_vec())
            .collect::<Vec<_>>()
    };
    for partition_id in [grown, copied] {
        let row = entry_point_row(&dataset, partition_id).await;
        dataset.delete(&format!("_rowid = {row}")).await.unwrap();
    }
    let before = stored_entry_points(&dataset, INDEX_NAME).await;
    let base = segments[0].uuid;

    let mut dataset = append_vectors(uri, &aimed).await;
    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        (stats.partitions_grown, stats.partitions_created),
        (1, 0),
        "{stats:?}"
    );
    let segment = read_committed_segments(&dataset, INDEX_NAME).await[0].uuid;
    let stored = of_segment(&stored_entry_points(&dataset, INDEX_NAME).await, segment);
    let retrained = of_segment(&retrained_entry_points(&dataset, INDEX_NAME).await, segment);
    let before = of_segment(&before, base);
    assert_eq!(stored[&grown], retrained[&grown]);
    assert_eq!(stored[&copied], before[&copied]);
    assert_ne!(
        stored[&copied], retrained[&copied],
        "the copied partition's deleted entry point did not change what training at open trains, \
         so nothing here tells carrying from retraining"
    );
}

/// A partition consolidation dropped and an insert creates again trains its
/// entry points as a build would: every one of its vertices is new.
#[tokio::test]
async fn a_partition_an_insert_creates_trains_its_entry_points() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_with_entry_points(uri).await;
    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
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

    let mut dataset = append_vectors(uri, &vectors).await;
    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert!(stats.partitions_created > 0, "{stats:?}");
    let stored = stored_entry_points(&dataset, INDEX_NAME).await;
    let segment = read_committed_segments(&dataset, INDEX_NAME).await[0].uuid;
    assert!(!stored[&(segment, emptied)].is_empty());
    assert_eq!(stored, retrained_entry_points(&dataset, INDEX_NAME).await);
}

/// The new rows of a grown partition are live by position, not by the row
/// filter: once another segment has moved, the filter of the index the insert
/// opened admits only rows inside the coverage each segment had then, and the
/// new fragments are in none of it. Here the delta moves and the base does
/// not, so the insert goes ahead and the trap is armed; the base's lists still
/// come out as training at open trains them, new rows included.
#[tokio::test]
async fn an_insert_beside_a_moved_segment_counts_its_new_rows_live() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    indexed_with_entry_points(uri).await;
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    let delta = committed_uuids(&dataset).await[1];
    // One deleted row makes the delta's first fragment, and only that one,
    // worth rewriting.
    let row = RowAddress::new_from_parts(3, 7);
    dataset
        .delete(&format!("_rowid = {}", u64::from(row)))
        .await
        .unwrap();
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            target_rows_per_fragment: 512,
            materialize_deletions_threshold: 0.001,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        (metrics.fragments_removed, metrics.fragments_added),
        (1, 1),
        "{metrics:?}"
    );
    let moved = u64::from(RowAddress::new_from_parts(3, 0));
    assert_ne!(
        dataset
            .frag_reuse_index()
            .await
            .unwrap()
            .expect("the compaction left no record of the move")
            .remap_row_id(moved),
        Some(moved),
        "the delta did not move"
    );

    let mut dataset = with_new_rows(uri, 1234).await;
    let stats = insert_in_place(&mut dataset, INDEX_NAME).await.unwrap();
    assert!(stats.partitions_grown > 0, "{stats:?}");
    let uuids = committed_uuids(&dataset).await;
    assert_eq!(uuids.len(), 2);
    assert!(uuids.contains(&delta), "the moved delta did not survive");
    let base = *uuids.iter().find(|uuid| **uuid != delta).unwrap();
    let stored = of_segment(&stored_entry_points(&dataset, INDEX_NAME).await, base);
    assert!(
        stored.values().any(|entries| !entries.is_empty()),
        "the base stores no entry points: {stored:?}"
    );
    assert_eq!(
        stored,
        of_segment(&retrained_entry_points(&dataset, INDEX_NAME).await, base)
    );
}

/// A delta trains its entry points under the base's parameters, not under the
/// default its own build parameters would ask for.
#[tokio::test]
async fn a_delta_trains_its_entry_points_under_the_bases_parameters() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let params = EntryPointParams::new(4).with_seed(7);
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS)
            .with_graph_params(base_graph())
            .with_entry_point_params(params.clone()),
    )
    .await
    .unwrap();
    let mut dataset = with_new_rows(uri, 99).await;
    insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();

    let segments = read_committed_segments(&dataset, INDEX_NAME).await;
    assert_eq!(segments.len(), 2);
    for segment in &segments {
        assert_eq!(
            segment.manifest.metadata().entry_point_params,
            Some(params.clone())
        );
    }
    let delta = segments[1].uuid;
    assert_eq!(
        of_segment(&stored_entry_points(&dataset, INDEX_NAME).await, delta),
        of_segment(&retrained_entry_points(&dataset, INDEX_NAME).await, delta)
    );
}
