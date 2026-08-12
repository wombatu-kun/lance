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

use std::collections::HashSet;

use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::index::DatasetIndexExt;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, build_index_segment, create_index};
use lance_vamana::inserter::{InsertStats, insert_as_segment};
use lance_vamana::query::{SearchParams, VamanaIndex};
use roaring::RoaringBitmap;
use uuid::Uuid;

mod common;
use common::{
    DatasetFixture, VECTOR_COLUMN, brute_force, live_row_ids, random_vectors,
    read_committed_segments, recall,
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

/// Three fragments of 512 rows, indexed over eight partitions.
async fn indexed_dataset(uri: &str) -> Dataset {
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS).with_graph_params(base_graph()),
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

fn search() -> SearchParams {
    SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(64)
}

async fn committed_uuids(dataset: &Dataset) -> Vec<Uuid> {
    dataset
        .load_indices_by_name(INDEX_NAME)
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
        stats.partitions_written > 0 && stats.partitions_written <= PARTITIONS as usize,
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
    indexed_dataset(uri).await;

    let mut dataset = with_new_rows(uri, 99).await;
    let (odd, _) = build_index_segment(
        &dataset,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS).with_graph_params(BuildParams {
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

/// Compaction strands this index over fragments that no longer exist, and the
/// README's answer to that used to be a rebuild. It is not: the compacted rows
/// are new rows like any other, and indexing them puts the index back.
///
/// The stranded segment goes with the same commit, without being named in it -
/// Lance drops an existing segment whose live coverage is empty.
#[tokio::test]
async fn a_compacted_dataset_is_repaired_by_indexing_it_again() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let stranded = committed_uuids(&dataset).await[0];

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert!(
        metrics.fragments_removed > 0,
        "compaction rewrote nothing, so the rest of this proves nothing: {metrics:?}"
    );
    let mut dataset = Dataset::open(uri).await.unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert!(
        index.covered_fragments().is_empty(),
        "compaction was supposed to leave this index covering nothing"
    );
    let answer = index
        .search(&random_vectors(1, 7)[0], &search())
        .await
        .unwrap();
    assert!(
        answer.neighbors.is_empty(),
        "a stranded index answered with rows the dataset no longer has"
    );

    let stats = insert_as_segment(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.vectors, 3 * 512, "{stats:?}");

    let uuids = committed_uuids(&dataset).await;
    assert_eq!(
        uuids.len(),
        1,
        "the stranded segment outlived the commit that made it unnecessary"
    );
    assert_ne!(uuids[0], stranded);
    let (rows, slots) = stored_row_ids(&dataset).await;
    let live = live_row_ids(&dataset).await;
    assert_eq!((slots, rows.len()), (live.len(), live.len()));
    let after = measured_recall(&dataset).await;
    assert!(after >= 0.95, "recall after the repair is {after}");
}
