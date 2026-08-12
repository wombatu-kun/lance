// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What consolidating a committed index does to the dataset that holds it.
//!
//! The graph half of consolidation is characterised where it lives. What can
//! only be asked here is whether the result is *the index the dataset has*: that
//! the rewritten segment replaces the one it came from instead of joining it,
//! that a partition nothing was deleted from crosses over unchanged, that a
//! partition everything was deleted from leaves no trace, and that the answers
//! afterwards are as good and cost less to get.
//!
//! The fixture is three fragments of 512 rows over eight partitions on purpose:
//! several rows per partition and several partitions per fragment, so that a
//! counter that over-filtered or a table row that was written twice would show
//! up rather than being masked by a partition of one.

use std::collections::HashSet;

use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::utils::address::RowAddress;
use lance_vamana::builder::{IndexParams, build_index_segment, create_index};
use lance_vamana::consolidator::{ConsolidateStats, consolidate_index};
use lance_vamana::query::{SearchParams, VamanaIndex};
use roaring::RoaringBitmap;
use uuid::Uuid;

mod common;
use common::{
    DatasetFixture, VECTOR_COLUMN, brute_force, live_row_ids, random_vectors,
    read_committed_segment, recall,
};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 8;
const K: usize = 10;

async fn indexed_dataset(uri: &str) -> Dataset {
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS),
    )
    .await
    .unwrap();
    dataset
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

/// The uuid of the one segment committed under [`INDEX_NAME`].
///
/// Asserting there is exactly one is half of what the replacement tests check:
/// the manifest layer under `commit_existing_index_segments` dedups by uuid and
/// not by name, so a replacement that failed to remove the old row would leave
/// two segments here rather than an error.
async fn committed_uuid(dataset: &Dataset) -> Uuid {
    let uuids = committed_uuids(dataset).await;
    assert_eq!(uuids.len(), 1, "expected exactly one committed segment");
    uuids[0]
}

/// Commit a segment covering exactly `fragments`, beside whatever is already
/// under this index name.
async fn commit_segment_over(dataset: &mut Dataset, fragments: &[u32]) -> Uuid {
    let (segment, _) = build_index_segment(
        dataset,
        &IndexParams::new(VECTOR_COLUMN, PARTITIONS),
        fragments,
    )
    .await
    .unwrap();
    let uuid = segment.uuid();
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![segment])
        .await
        .unwrap();
    uuid
}

/// Every row id the index physically stores, and how many slots they occupy.
///
/// The two differ exactly when a row was written into two partitions, which no
/// assertion on the set alone could see.
async fn stored_row_ids(dataset: &Dataset) -> (HashSet<u64>, usize) {
    let (_, partitions) = read_committed_segment(dataset, INDEX_NAME).await;
    let slots = partitions.values().map(|partition| partition.len()).sum();
    let rows = partitions
        .values()
        .flat_map(|partition| partition.graph().row_ids().iter().copied())
        .collect();
    (rows, slots)
}

#[tokio::test]
async fn consolidating_replaces_the_segment_it_rewrote() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let before = committed_uuid(&dataset).await;

    dataset.delete("_rowid % 7 == 0").await.unwrap();
    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.segments_rewritten, 1, "{stats:?}");

    assert_ne!(
        committed_uuid(&dataset).await,
        before,
        "the rewritten segment did not replace the one it came from"
    );
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let found = index
        .search(&random_vectors(1, 77)[0], &search())
        .await
        .unwrap();
    assert_eq!(
        found.neighbors.len(),
        K,
        "the replacement stopped answering"
    );
}

/// Every live row is stored once and no deleted row is stored at all.
#[tokio::test]
async fn the_consolidated_index_holds_only_live_rows() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;

    let (before, before_slots) = stored_row_ids(&dataset).await;
    dataset.delete("_rowid % 7 == 0").await.unwrap();
    let live = live_row_ids(&dataset)
        .await
        .into_iter()
        .collect::<HashSet<_>>();
    assert!(
        live.len() < before.len(),
        "the delete was meant to remove indexed rows"
    );

    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        stats.vertices_removed,
        before.len() - live.len(),
        "{stats:?}"
    );

    let (after, after_slots) = stored_row_ids(&dataset).await;
    assert_eq!(after, live, "the index does not hold exactly the live rows");
    assert_eq!(after_slots, after.len(), "a row was stored twice");
    assert_eq!(before_slots, before.len());
}

/// A partition nothing was deleted from is copied, not re-encoded - and the
/// copy is the file it came from, vertex for vertex.
///
/// Three rows can fall into at most three of the eight partitions, so both
/// branches are reached whatever the router did with them. That matters:
/// asserting only that the counts add up would pass a driver that rewrote
/// everything.
#[tokio::test]
async fn a_partition_without_deletions_is_copied_not_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let (before, before_partitions) = read_committed_segment(&dataset, INDEX_NAME).await;

    dataset.delete("_rowid in (0, 1, 2)").await.unwrap();
    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();

    assert_eq!(stats.vertices_removed, 3, "{stats:?}");
    assert!(stats.partitions_copied > 0, "nothing was copied: {stats:?}");
    assert!(
        stats.partitions_consolidated > 0,
        "nothing was repaired: {stats:?}"
    );
    assert_eq!(
        stats.partitions_copied
            + stats.partitions_consolidated
            + stats.partitions_rebuilt
            + stats.partitions_dropped,
        before.partitions().len(),
        "the partitions of the old segment are not accounted for: {stats:?}"
    );

    let (after, after_partitions) = read_committed_segment(&dataset, INDEX_NAME).await;
    let untouched = after_partitions
        .iter()
        .filter(|(id, partition)| before_partitions[id].len() == partition.len())
        .count();
    assert_eq!(untouched, stats.partitions_copied);
    for (id, partition) in &after_partitions {
        if before_partitions[id].len() != partition.len() {
            continue;
        }
        assert_eq!(
            partition, &before_partitions[id],
            "copied partition {id} is not the partition it was copied from"
        );
        assert_eq!(
            after.partition(*id).unwrap().medoid,
            before.partition(*id).unwrap().medoid,
            "copied partition {id} lost its entry point"
        );
    }
}

/// A partition whose every row is deleted gets no file and no table row.
///
/// The rows to delete are read out of the committed segment rather than guessed
/// at, because which vertex the router put where is not something a fixture can
/// arrange from outside.
#[tokio::test]
async fn a_partition_whose_every_row_is_deleted_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let (before, partitions) = read_committed_segment(&dataset, INDEX_NAME).await;

    let (doomed, rows) = partitions
        .iter()
        .min_by_key(|(_, partition)| partition.len())
        .map(|(id, partition)| (*id, partition.graph().row_ids().to_vec()))
        .expect("the fixture indexes something");
    let addresses = rows
        .iter()
        .map(|row| row.to_string())
        .collect::<Vec<_>>()
        .join(",");
    dataset
        .delete(&format!("_rowid in ({addresses})"))
        .await
        .unwrap();

    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.partitions_dropped, 1, "{stats:?}");
    assert_eq!(stats.vertices_removed, rows.len(), "{stats:?}");

    let (after, _) = read_committed_segment(&dataset, INDEX_NAME).await;
    assert!(
        after.partition(doomed).is_none(),
        "partition {doomed} is still listed"
    );
    assert_eq!(after.partitions().len(), before.partitions().len() - 1);
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index
            .search(&random_vectors(1, 77)[0], &search())
            .await
            .unwrap()
            .neighbors
            .len(),
        K,
        "the segment stopped answering once a partition left it"
    );
}

/// Deleting a fragment's last row takes the fragment out of the manifest, and
/// the rewritten segment is committed over what is left.
///
/// The commit is the part worth pinning. `commit_existing_index_segments`
/// refuses an incoming set that would orphan a fragment of the segment it
/// replaces, and the coverage here is *narrower* than the old segment's - it
/// only gets through because what the old one is credited with is already
/// narrowed to the fragments the dataset still has.
#[tokio::test]
async fn consolidation_narrows_coverage_to_the_fragments_that_are_left() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;

    let first_row_of_fragment = u64::from(RowAddress::new_from_parts(1, 0));
    dataset
        .delete(&format!("_rowid >= {first_row_of_fragment}"))
        .await
        .unwrap();
    assert_eq!(
        dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id())
            .collect::<Vec<_>>(),
        vec![0],
        "the delete was meant to leave one fragment"
    );

    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.segments_rewritten, 1, "{stats:?}");

    let (after, _) = read_committed_segment(&dataset, INDEX_NAME).await;
    assert_eq!(
        after.metadata().fragments,
        vec![0],
        "the segment still records fragments the dataset does not have"
    );
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.covered_fragments(), &RoaringBitmap::from_iter([0u32]));

    let (stored, _) = stored_row_ids(&dataset).await;
    assert_eq!(
        stored,
        live_row_ids(&dataset)
            .await
            .into_iter()
            .collect::<HashSet<_>>()
    );
}

/// An index of several segments is the ordinary state of one that has been
/// appended to, and consolidation rewrites only the segments that need it.
///
/// The segment left alone is the assertion that matters. It is not named in the
/// commit at all, and it survives only because its fragment coverage is disjoint
/// from the incoming set - the same rule that makes the rewritten one *replace*
/// its old self rather than join it. Committing the whole index instead would
/// pass every other test in this file.
#[tokio::test]
async fn only_the_segment_holding_a_deleted_row_is_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture::default().write(uri).await;
    let base = commit_segment_over(&mut dataset, &[0]).await;
    let delta = commit_segment_over(&mut dataset, &[1, 2]).await;
    assert_eq!(committed_uuids(&dataset).await.len(), 2);

    // Inside the base segment's one fragment and nowhere else.
    let first_row_of_fragment = u64::from(RowAddress::new_from_parts(1, 0));
    dataset
        .delete(&format!(
            "_rowid < {first_row_of_fragment} AND _rowid % 5 == 0"
        ))
        .await
        .unwrap();

    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(stats.segments_rewritten, 1, "{stats:?}");
    assert_eq!(stats.segments_untouched, 1, "{stats:?}");

    let after = committed_uuids(&dataset).await;
    assert_eq!(
        after.len(),
        2,
        "an index of two segments came back as {after:?}"
    );
    assert!(
        after.contains(&delta),
        "the segment the commit never mentioned did not survive it"
    );
    assert!(
        !after.contains(&base),
        "the rewritten segment was kept beside its own replacement"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.num_segments(), 2);
    assert_eq!(
        index.covered_fragments(),
        &RoaringBitmap::from_iter([0u32, 1, 2])
    );
    assert_eq!(
        index
            .search(&random_vectors(1, 77)[0], &search())
            .await
            .unwrap()
            .neighbors
            .len(),
        K
    );
}

#[tokio::test]
async fn consolidating_an_index_without_deletions_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    let before = committed_uuid(&dataset).await;

    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        stats,
        ConsolidateStats {
            segments_untouched: 1,
            ..Default::default()
        }
    );
    assert_eq!(
        committed_uuid(&dataset).await,
        before,
        "an index with nothing deleted was rewritten anyway"
    );
}

/// The point of the exercise: after consolidation a query reads fewer bytes,
/// and the answers are not worse for it.
///
/// The beam is deliberately narrow - twelve, where a partition holds 192
/// vertices before and 48 after. That is what makes this test able to fail. A
/// generous beam sees most of the partition either way, so consolidation cannot
/// move recall and the assertion would be vacuous; a narrow one fills with
/// tombstones, which are ranked by distance along with everything else and only
/// dropped from the answer at the end. Measured here: 0.825 before and 1.0
/// after, which is also a limit on the earlier finding that deleting rows costs
/// almost no recall - that was measured at a beam of 100, and it stops being
/// true once the beam is narrow next to the reciprocal of the live fraction.
#[tokio::test]
async fn consolidation_reads_fewer_bytes_for_the_same_answers() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri).await;
    dataset.delete("_rowid % 4 != 0").await.unwrap();

    let queries = random_vectors(20, 77);
    let mut truth = Vec::with_capacity(queries.len());
    for query in &queries {
        truth.push(brute_force(&dataset, query, K).await);
    }
    let narrow = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(12);

    let before = measure(&dataset, &queries, &truth, &narrow).await;
    let stats = consolidate_index(&mut dataset, INDEX_NAME).await.unwrap();
    let after = measure(&dataset, &queries, &truth, &narrow).await;

    assert!(
        after.bytes_read * 2 < before.bytes_read,
        "three rows in four were deleted and the read only went from {} bytes to {}: {stats:?}",
        before.bytes_read,
        after.bytes_read
    );
    assert!(
        after.recall >= before.recall && after.recall >= 0.95,
        "recall went from {} to {}",
        before.recall,
        after.recall
    );
}

struct Measured {
    recall: f64,
    bytes_read: u64,
}

/// Run every query against a freshly opened index and report what it cost.
///
/// The bytes come off the index's own scheduler. A tracker wrapped around the
/// object store under-counts a local read, and the index reads through one
/// scheduler for its whole life, so opening it here is what makes the number a
/// per-run total rather than a per-process one.
async fn measure(
    dataset: &Dataset,
    queries: &[Vec<f32>],
    truth: &[Vec<u64>],
    params: &SearchParams,
) -> Measured {
    let index = VamanaIndex::open(dataset, INDEX_NAME).await.unwrap();
    let mut total = 0.0;
    for (query, exact) in queries.iter().zip(truth) {
        let found = index
            .search(query, params)
            .await
            .unwrap()
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        total += recall(&found, exact);
    }
    Measured {
        recall: total / queries.len() as f64,
        bytes_read: index.io_stats().bytes_read,
    }
}
