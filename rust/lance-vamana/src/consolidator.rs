// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Taking a dataset's deleted rows out of the index that still holds them.
//!
//! [`crate::consolidate`] is the graph half of this and knows nothing about
//! datasets. This is the other half: which rows are dead, which partitions that
//! makes worth rewriting, and how the result is committed.
//!
//! What consolidation buys is **bytes**. Measured on SIFT 100k against the same
//! index left alone, at 10/30/50/70/90% deleted: bytes read per query and the
//! index on disk both track the live fraction to within half a percentage point,
//! where the index left alone holds at 8.1 MB a query and 73.9 MiB on disk
//! whatever is deleted. Latency follows the bytes - 4.3ms down to 1.6ms at 90% -
//! because a partition is read whole.
//!
//! What it does not buy is recall, which is unchanged to four decimal places up
//! to 70% deleted. The exception is instructive: at 90% it *gains*, 0.9474 to
//! 0.9540, because a beam of 100 over a partition of 1000 vertices with 100 live
//! ones fills with tombstones - they are ranked by distance along with
//! everything else and only dropped at the end. So a tombstone is nearly free
//! only while the beam is wide next to the reciprocal of the live fraction.
//!
//! Distances per query do fall, 7492.9 to 1140.0 at 90%, but they lag the bytes
//! badly (84% of the original where bytes are at 70%): the walk is bounded by
//! the beam, so shrinking a partition saves no arithmetic until the partition
//! approaches the beam.
//!
//! The one way consolidation can make an index *worse* is by running late. The
//! one-hop repair guarantees that no edge dangles, not that the graph stays in
//! one piece, and on a 1000-vertex build at `R=16` removing 90% of the rows
//! evenly leaves 2 of 100 survivors reachable. Every partition is therefore
//! walked after it is repaired, and one that came apart is rebuilt outright
//! rather than written out in pieces.
//!
//! The unit of commit is a segment, which is aligned to fragments, while the
//! unit of rewrite is a partition, which is not - an IVF partition draws its
//! vectors from every fragment. So a segment with one dead row in it is
//! rewritten whole. The partitions with nothing dead in them are copied rather
//! than re-encoded, which is what keeps that from being as expensive as it
//! sounds when deletions are clustered in the space the index measures.

use std::sync::Arc;

use futures::stream::{self, StreamExt, TryStreamExt};
use lance::Dataset;
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::build::BuildParams;
use crate::builder::{INDEX_DETAILS_TYPE_URL, index_column};
use crate::format::{FORMAT_VERSION, IndexMetadata, ROW_ID_COLUMN};
use crate::io::{SegmentWriter, check_partition_shape, open_file, read_partition, read_row_ids};
use crate::merge::merge_partition;
use crate::query::{Segment, VamanaIndex};
use crate::search::Comparisons;

/// What consolidating an index did, and what it cost.
///
/// Every partition of every rewritten segment falls into exactly one of the
/// first four counters, so they add up to the partitions of those segments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConsolidateStats {
    /// Segments rewritten and committed in place of their old selves.
    pub segments_rewritten: usize,
    /// Segments left alone because nothing they hold is deleted.
    pub segments_untouched: usize,
    /// Segments abandoned because every fragment they were built over is gone.
    ///
    /// Nothing is written for one of these. Their manifest rows are removed by
    /// the commit this call makes for the other segments, and if there is no
    /// such commit they stay - consolidation repairs graphs, and an index that
    /// covers nothing needs rebuilding rather than repairing.
    pub segments_abandoned: usize,
    /// Partitions repaired by the one-hop inheritance and written out.
    pub partitions_consolidated: usize,
    /// Partitions the repair left in pieces, which were built again from
    /// scratch over their survivors.
    pub partitions_rebuilt: usize,
    /// Partitions with nothing deleted in them, copied across as they were.
    pub partitions_copied: usize,
    /// Partitions whose every row was deleted, given no file and no table row.
    pub partitions_dropped: usize,
    /// Vertices that are no longer stored.
    pub vertices_removed: usize,
    /// Distance computations, across the repairs and the rebuilds.
    pub comparisons: u64,
}

/// Rewrite every segment of `index_name` that holds a deleted row, and commit.
///
/// There is no threshold here, deliberately: asked to consolidate, this
/// consolidates. *When* to ask is the caller's, and the answer the measurements
/// give is "often". On SIFT 100k a round costs 0.3 to 3 seconds against 14 for a
/// build, five rounds of it never once needed a partition rebuilt, and the index
/// tracked the live fraction the whole way - 73.9 MiB down to 7.5 at 90%
/// deleted, where the same index left alone stays at 73.9.
///
/// It is the only maintenance call that never reads the dataset's vector column
/// and never routes: what it needs is the delete list and the graphs it already
/// holds. When something other than deletions is pending too - a delta segment,
/// or fragments no segment covers - [`crate::merger::merge_index`] does this and
/// those in one pass over the same partitions.
///
/// The delete list is a snapshot taken when the index is opened, so a row
/// deleted while this runs is simply left for the next call - it stays filtered
/// out of every answer in the meantime. A dataset compaction landing in the
/// middle is the same hazard a build has: `(CreateIndex, Rewrite)` do not
/// conflict, so the commit succeeds over fragments the dataset no longer has and
/// the coverage narrows. A concurrent commit under this same index *name* does
/// conflict, and retrying it re-runs this call from the beginning.
///
/// The old segment's files are not deleted. They stay referenced by the manifest
/// versions that came before this commit and go when `cleanup_old_versions`
/// takes those versions, so an index costs both copies until then. That matters
/// here more than it would elsewhere, because saving bytes is the entire point.
pub async fn consolidate_index(
    dataset: &mut Dataset,
    index_name: &str,
) -> Result<ConsolidateStats> {
    let index = VamanaIndex::open(dataset, index_name).await?;
    let mut stats = ConsolidateStats::default();
    if index.row_filter().is_empty() {
        stats.segments_untouched = index.num_segments();
        return Ok(stats);
    }

    let store = dataset.object_store(None).await?;
    let io_parallelism = store.io_parallelism();
    let dataset_version = dataset.manifest.version;
    let mut rewritten = Vec::new();

    for segment in index.segments() {
        if segment.coverage.is_empty() {
            log::warn!(
                "Vamana index '{index_name}' segment {} was built over {} fragments the dataset \
                 no longer has, all of them, so there is nothing left in it to repair; rebuild \
                 the index",
                segment.uuid,
                segment.manifest.metadata().fragments.len()
            );
            stats.segments_abandoned += 1;
            continue;
        }

        let dead = dead_by_partition(&index, segment, io_parallelism).await?;
        if dead.iter().all(RoaringBitmap::is_empty) {
            stats.segments_untouched += 1;
            continue;
        }

        let uuid = Uuid::new_v4();
        let dir = dataset.indices_dir().join(uuid.to_string());
        let metadata = IndexMetadata {
            fragments: segment.coverage.iter().collect(),
            ..segment.manifest.metadata().clone()
        };
        let mut writer = SegmentWriter::new(
            store.clone(),
            dir,
            metadata.clone(),
            segment.manifest.ivf().clone(),
        );
        rewrite_segment(&index, segment, &dead, &metadata, &mut writer, &mut stats).await?;
        writer.finish().await?;

        log::info!(
            "Vamana index '{index_name}' segment {} was rewritten as {uuid}",
            segment.uuid
        );
        rewritten.push(IndexSegment::new(
            uuid,
            segment.coverage.iter(),
            segment.fields.iter().copied(),
            Arc::new(prost_types::Any {
                type_url: INDEX_DETAILS_TYPE_URL.to_string(),
                value: Vec::new(),
            }),
            FORMAT_VERSION as i32,
            dataset_version,
        ));
        stats.segments_rewritten += 1;
    }

    if rewritten.is_empty() {
        // Every segment abandoned means the index answers for no row of this
        // dataset at all. Reported rather than passed over in silence, because
        // the caller asked for deleted rows to be taken out of an index and the
        // honest answer is that there is no index left to take them out of.
        if stats.segments_abandoned == index.num_segments() {
            return Err(Error::invalid_input(format!(
                "Vamana cannot consolidate index '{index_name}': every fragment its {} segments \
                 were built over is gone, so it answers for nothing; rebuild it instead",
                stats.segments_abandoned
            )));
        }
        return Ok(stats);
    }

    let column = index_column(dataset, index_name, &index.segments()[0].fields)?;
    dataset
        .commit_existing_index_segments(index_name, &column, rewritten)
        .await?;
    Ok(stats)
}

/// Which vertices of each partition of `segment` are no longer live, by local id.
///
/// One projected read of `__row_id` per partition, and they wait on each other
/// rather than in turn: a segment of thousands of partitions read one at a time
/// on a store with any latency is minutes before the first partition is
/// rewritten. Nothing but the row ids is fetched, so the whole pass costs eight
/// bytes a vertex against the 776 the crate's own working point stores.
pub(crate) async fn dead_by_partition(
    index: &VamanaIndex,
    segment: &Segment,
    io_parallelism: usize,
) -> Result<Vec<RoaringBitmap>> {
    stream::iter(segment.manifest.partitions().iter().map(|entry| async {
        let reader = open_file(
            index.scheduler(),
            &segment.dir.clone().join(entry.file.as_str()),
            Some(&[ROW_ID_COLUMN]),
            segment.file_sizes.get(&entry.file).copied(),
        )
        .await?;
        Ok::<_, Error>(
            read_row_ids(&reader, entry.num_rows)
                .await?
                .iter()
                .enumerate()
                .filter(|(_, row_addr)| index.row_filter().rejects(**row_addr))
                .map(|(local_id, _)| local_id as u32)
                .collect::<RoaringBitmap>(),
        )
    }))
    .buffered(io_parallelism)
    .try_collect()
    .await
}

/// Write every partition of `segment` into `writer`, repairing what needs it.
///
/// One partition at a time, and deliberately: a partition is read whole, so a
/// pipeline here would be a working set of however many partitions it kept in
/// flight. That is the same shape - and the same open question about overlapping
/// the reads with the arithmetic - as the build path's `write_partitions`.
async fn rewrite_segment(
    index: &VamanaIndex,
    segment: &Segment,
    dead: &[RoaringBitmap],
    metadata: &IndexMetadata,
    writer: &mut SegmentWriter,
    stats: &mut ConsolidateStats,
) -> Result<()> {
    for (entry, dead) in segment.manifest.partitions().iter().zip(dead) {
        stats.vertices_removed += dead.len() as usize;
        if dead.is_empty() {
            writer
                .copy_partition(&segment.dir, &segment.manifest, entry.partition_id)
                .await?;
            stats.partitions_copied += 1;
            continue;
        }
        if dead.len() == entry.num_rows as u64 {
            // No file and no table row. `consolidate_partition` refuses a
            // partition of nothing rather than returning one, so that dropping
            // it is a decision taken here and not a shape written to disk.
            stats.partitions_dropped += 1;
            continue;
        }

        let reader = open_file(
            index.scheduler(),
            &segment.dir.clone().join(entry.file.as_str()),
            None,
            segment.file_sizes.get(&entry.file).copied(),
        )
        .await?;
        let partition = read_partition(&reader, entry.num_rows).await?;
        check_partition_shape(&partition, entry, metadata.max_degree, metadata.dimension)?;

        // Minutes of arithmetic over a whole segment, and not one await in it,
        // so it runs on the CPU pool rather than on the runtime the scheduler
        // reads through. Nothing inside waits on anything, which is what
        // `spawn_cpu` requires.
        let dead = dead.clone();
        let metadata = metadata.clone();
        let entry_point = entry.medoid;
        let (repaired, comparisons) = spawn_cpu(move || {
            let comparisons = Comparisons::default();
            // No newcomers: consolidation is the half of a merge that only takes
            // rows out, and the other half is what `crate::merger` adds.
            let repaired = merge_partition(
                &partition,
                entry_point,
                &dead,
                None,
                metadata.distance_type,
                &BuildParams::maintenance(&metadata),
                &comparisons,
            )?;
            Ok::<_, Error>((repaired, comparisons.get()))
        })
        .await?;

        writer
            .write_partition(entry.partition_id, repaired.medoid, &repaired.partition)
            .await?;
        stats.comparisons = stats.comparisons.saturating_add(comparisons);
        if repaired.rebuilt {
            stats.partitions_rebuilt += 1;
        } else {
            stats.partitions_consolidated += 1;
        }
    }
    Ok(())
}
