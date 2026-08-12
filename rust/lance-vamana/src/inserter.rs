// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Putting a dataset's new rows into an index that was built before them.
//!
//! The other half of maintenance. [`crate::consolidator`] takes deleted rows out
//! of an index; this puts appended rows in, and without it the only answer to an
//! append is a full rebuild.
//!
//! Rows that arrived after the build live in fragments no segment covers, which
//! is exactly what makes them findable: `live_fragments - covered_fragments` is
//! the whole of the question, and it needs no bookkeeping of its own.
//!
//! # Where the new rows go
//!
//! A new segment of their own. The index grows a second segment beside the base,
//! the base is not touched, and a query then probes both - `nprobes` partitions
//! per segment, so two segments cost twice the reads of one. That is the price,
//! and it is the price the FreshDiskANN paper pays too: its RW-Temp index is a
//! second index searched alongside the long-term one.
//!
//! # What that price actually is
//!
//! Measured on SIFT 100k, over four indices covering **the same rows in the same
//! fragments** and differing only in whether they were built once or grown, at
//! one, two, four and eight segments:
//!
//! | segments | files | recall@10 | partitions/query | bytes/query | iops/query | p50 |
//! |---|---|---|---|---|---|---|
//! | 1 | 101 | 0.9777 | 10 | 8.11 MB | 50 | 4.4 ms |
//! | 2 | 202 | 0.9781 | 20 | 8.67 MB | 100 | 5.4 ms |
//! | 4 | 404 | 0.9782 | 40 | 8.75 MB | 200 | 7.8 ms |
//! | 8 | 807 | 0.9782 | 80 | 8.92 MB | 400 | 13.1 ms |
//!
//! **Eight times the partitions, ten percent the bytes.** A partition is read
//! whole and a delta's partitions are proportionally smaller, so the rows read
//! per query barely move; the ten percent is per-file overhead, and nearly all
//! of it arrives with the second segment. **Recall does not fall, it rises** -
//! a partition of seventy vertices under a beam of a hundred is searched
//! exhaustively, which is also why distances per query rise by half.
//!
//! What a delta really costs is **read operations, latency and files**: 400
//! reads against 50, three times the latency, and 807 manifest entries against
//! 101 - and Lance copies that list into every manifest the dataset writes
//! afterwards. Against that, growing the index took 9.0 seconds where building
//! it once took 14.3.
//!
//! So the case for putting new rows into the base's own graphs instead is not
//! bytes and not recall. It is that a query stays on one segment's worth of
//! reads.
//!
//! # Why the delta inherits the base's centroids
//!
//! It could train a router of its own, and the read path would not notice -
//! `route` ranks each segment's centroids separately. Three things say inherit.
//!
//! A router trained on a handful of new rows is a router trained on a handful of
//! rows: k-means over 500 vectors cannot produce the 4096 centroids the base
//! has, and `train_router` refuses outright below `rows < k`, so a delta would
//! need a partition count of its own and a rule for choosing it.
//!
//! Inheriting costs nothing. Training the router is the one part of a build that
//! reads the whole column twice over, and a delta skips it entirely.
//!
//! Most of all it keeps every segment of an index on **one partition
//! numbering**: partition 17 of the delta holds the rows nearest the same
//! centroid as partition 17 of the base. Folding a delta back into the base is
//! then a concatenation of like with like rather than a re-routing of every row.
//!
//! What it costs is files. A delta writes one file per partition that drew a
//! row, so a 500-row delta against a 4096-partition base writes up to 500 tiny
//! files. That is the cost the table above turns into a number, and it is the
//! reason the delta cannot be the only answer forever.
//!
//! # The rows of one fragment go into one segment
//!
//! Not a choice. `commit_existing_index_segments` refuses a set of segments
//! whose fragment coverage overlaps, so a batch split across two segments cannot
//! be committed at all. Coverage is per fragment and there is no finer grain to
//! divide it on.

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray};
use lance::Dataset;
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use lance_index::vector::ivf::storage::IvfModel;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::build::BuildParams;
use crate::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, assign, build_index_segment_with_router, build_one,
    gather, group_by_partition, index_column, read_vectors,
};
use crate::format::{FORMAT_VERSION, IndexMetadata};
use crate::insert::insert_into_partition;
use crate::io::{SegmentWriter, check_partition_shape, open_file, read_partition};
use crate::query::{Segment, VamanaIndex};
use crate::search::Comparisons;

/// What indexing a dataset's new rows did, and what it cost.
///
/// The three partition counters are exclusive. A delta segment only ever creates
/// partitions; an in-place insert produces all three, and they add up to the
/// partitions of the segment it rewrote plus the ones it had to add.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InsertStats {
    /// Fragments the index did not cover before and covers now.
    pub fragments_indexed: usize,
    /// Vectors indexed, which is the rows of those fragments minus the ones
    /// whose vector is null.
    pub vectors: usize,
    /// Partitions written from nothing: every partition of a delta segment, and
    /// the ones an in-place insert found a row for and no file behind.
    pub partitions_created: usize,
    /// Partitions that already existed, drew new rows and were rewritten with
    /// them linked in.
    pub partitions_grown: usize,
    /// Partitions that already existed, drew nothing, and crossed into the new
    /// segment as the bytes they were.
    pub partitions_copied: usize,
    /// Distance computations spent building and linking the graphs.
    pub comparisons: u64,
}

/// Index every row of `dataset` that `index_name` does not cover, as a segment
/// of its own.
///
/// Takes no parameters, and there are none to take: the column comes from the
/// field the index is recorded against, and the metric, the width, the degree,
/// the beam and the pruning slack all come from the base segment's own metadata.
/// A delta built with anything else would be a delta a query has to reconcile
/// with the base, and two of those numbers - the metric and the width - would
/// make the whole index unopenable rather than merely worse.
///
/// Nothing is committed when there is nothing new, so calling this on a schedule
/// is cheap: it costs one `open`, which is one small read per segment plus the
/// delete list. The delete list is the one part an insert has no use for, and it
/// is paid for anyway because `open` is where the refusals live - an index whose
/// data moved underneath it must not have a delta committed beside it, and that
/// is not a check worth having a second copy of.
///
/// What the delta costs afterwards is read operations: a query probes `nprobes`
/// partitions in it as in every other segment. [`crate::merger::merge_index`]
/// folds it back into the base, and on SIFT 100k that fold has paid for itself
/// after 533 queries against eight segments - so what a schedule of these is
/// paired with is a fold, not a rebuild.
///
/// A concurrent commit under the same index name is a retryable conflict, and
/// the retry re-runs this call from the beginning against the manifest that won.
/// Should another writer have indexed the same fragments in the meantime, its
/// segment is replaced by this one rather than joined - the incoming coverage
/// covers it whole. Should it have indexed a *superset*, the commit is refused
/// outright, because removing its segment would orphan the fragments this call
/// did not read.
///
/// A compaction is a special case worth naming, because it looks like a
/// disaster and is not. Compaction rewrites rows into brand new fragments and
/// strands the index over fragments the dataset no longer has;
/// [`VamanaIndex::open`] narrows the coverage to what survived, so the compacted
/// rows are simply new rows to this call, and the vertices left behind for them
/// are already rejected from every answer. Indexing then consolidating puts the
/// index back where it was without a rebuild.
///
/// Every new row must have a vector. A set of new fragments whose every vector
/// is null has nothing to index and is refused rather than covered, so a
/// pipeline that appends such a batch has to skip this call for it.
pub async fn insert_as_segment(dataset: &mut Dataset, index_name: &str) -> Result<InsertStats> {
    let index = VamanaIndex::open(dataset, index_name).await?;
    let new_fragments = index.unindexed_fragments(dataset);
    if new_fragments.is_empty() {
        return Ok(InsertStats::default());
    }

    let base = index.base_segment()?;
    let column = index_column(dataset, index_name, &base.fields)?;
    let params = inherited_params(&column, base.manifest.metadata(), base.manifest.ivf());
    let (segment, built) = build_index_segment_with_router(
        dataset,
        &params,
        &new_fragments,
        Some(base.manifest.ivf().clone()),
    )
    .await?;

    log::info!(
        "Vamana index '{index_name}' indexed {} new fragments into segment {}, beside the {} \
         segments already there",
        new_fragments.len(),
        segment.uuid(),
        index.num_segments()
    );
    // Disjoint from every segment there is, by construction: these are the
    // fragments nothing covers. So this commit removes nothing.
    dataset
        .commit_existing_index_segments(index_name, &column, vec![segment])
        .await?;

    Ok(InsertStats {
        fragments_indexed: new_fragments.len(),
        vectors: built.vectors,
        partitions_created: built.partitions,
        comparisons: built.comparisons,
        ..Default::default()
    })
}

/// Link every row of `dataset` that `index_name` does not cover into the graphs
/// of the segment that already holds their neighbours.
///
/// The canonical FreshVamana insert, applied a partition at a time because a
/// partition file is the unit of rewrite. New rows are routed by the target
/// segment's own centroids, and each partition that drew any of them is read,
/// grown and written out; the partitions that drew none are copied across
/// without being decoded, which is what keeps the cost proportional to the batch
/// rather than to the index.
///
/// The whole batch goes into **one** segment, and that is not a choice:
/// `commit_existing_index_segments` refuses a set of segments whose fragment
/// coverage overlaps, and coverage is per fragment, so a batch split across two
/// segments could not be committed at all. The target is the segment covering
/// the most fragments - the base rather than a delta.
///
/// What this buys over [`insert_as_segment`] is not bytes and not recall, both
/// of which a delta costs almost nothing. It is that a query keeps probing one
/// segment: measured on SIFT 100k, eight segments cost eight times the read
/// operations, three times the latency and eight times the files of one, at the
/// same recall.
///
/// # When it refuses
///
/// A target segment built over a fragment the dataset no longer has. Rewriting
/// one means writing its vertices into a segment whose coverage has narrowed to
/// what survived, and those vertices would then be stored under fragments the
/// new segment does not declare - so nothing would filter them out and a query
/// would answer with rows that are gone. [`crate::consolidator::consolidate_index`]
/// is what clears that state, by removing exactly those vertices.
///
/// Deleted rows are a different matter and are carried across untouched. Their
/// vertices stay in the graph as routers, the new segment still declares their
/// fragments, and the delete list still keeps them out of every answer.
///
/// # Where it sits in a maintenance pipeline
///
/// **Consolidate first, then insert.** Not a preference: a delete that empties a
/// fragment takes that fragment out of the dataset, and the refusal above then
/// stops the insert outright. Consolidation is what clears it. Running the two
/// the other way round works until the day a fragment empties, and then stops
/// working - measured, `examples/churn_cycle.rs` in the order insert-then-
/// consolidate dies at its fourth round.
///
/// [`crate::merger::merge_index`] is that round in one call and has no such
/// order. Over the cycle below it comes out with the same recall and the same
/// distances per query as this pair in every round, for 3% less time.
///
/// What that pipeline costs, and what it buys, measured over five rounds of
/// "delete a residue class, consolidate, append as many rows as were removed,
/// insert them" on SIFT 100k - a cycle that replaces every row the index was
/// built over, without ever rebuilding it:
///
/// | round | recall@10 | distances/query | files | iops/query | maintenance |
/// |---|---|---|---|---|---|
/// | 0 | 0.9778 | 8045 | 101 | 50 | 4.6 s |
/// | 4 | 0.9764 | 8463 | 101 | 50 | 8.6 s |
/// | rebuilt | 0.9812 | 7801 | 101 | 50 | 14.3 s |
///
/// Recall loses **0.14 of a percentage point** across the whole cycle and stops
/// falling after the first round, against the roughly one point the FreshVamana
/// paper allows itself. What churn really costs shows up beside it: the worn
/// graph spends 8.5% more distances than a rebuilt one for the same answer, and
/// a rebuild takes the recall back. Files, read operations and index size do not
/// move at all, which is the whole difference from [`insert_as_segment`].
pub async fn insert_in_place(dataset: &mut Dataset, index_name: &str) -> Result<InsertStats> {
    let index = VamanaIndex::open(dataset, index_name).await?;
    let new_fragments = index.unindexed_fragments(dataset);
    if new_fragments.is_empty() {
        return Ok(InsertStats::default());
    }

    let target = index.base_segment()?;
    let built_over = target
        .manifest
        .metadata()
        .fragments
        .iter()
        .copied()
        .collect::<RoaringBitmap>();
    if built_over != target.coverage {
        return Err(Error::invalid_input(format!(
            "Vamana cannot insert into index '{index_name}' in place: segment {} was built over \
             {} fragments the dataset no longer has, and rewriting it would store their vertices \
             under a coverage that no longer names them, where nothing would keep them out of an \
             answer; consolidate the index first",
            target.uuid,
            (&built_over - &target.coverage).len()
        )));
    }

    let column = index_column(dataset, index_name, &target.fields)?;
    let params = Arc::new(inherited_params(
        &column,
        target.manifest.metadata(),
        target.manifest.ivf(),
    ));
    let (row_ids, vectors) = read_vectors(dataset, &column, &new_fragments).await?;
    let row_ids = Arc::new(row_ids);

    // Normalising, routing and grouping are one uninterrupted pass over the
    // whole batch, which is the same reason the build path hands them to the
    // pool rather than running them on the runtime the scheduler reads through.
    let routing_model = target.manifest.ivf().clone();
    let (vectors, members) = {
        let params = params.clone();
        let row_ids = row_ids.clone();
        spawn_cpu(move || {
            let vectors = if params.distance_type == DistanceType::Cosine {
                normalize_fsl_owned(vectors)?
            } else {
                vectors
            };
            let assignment = assign(&routing_model, &vectors, &row_ids, params.distance_type)?;
            let members = group_by_partition(&assignment, params.num_partitions);
            Ok::<_, Error>((vectors, members))
        })
        .await?
    };

    let uuid = Uuid::new_v4();
    let mut coverage = target.coverage.clone();
    coverage.extend(new_fragments.iter().copied());
    let metadata = IndexMetadata {
        fragments: coverage.iter().collect(),
        ..target.manifest.metadata().clone()
    };
    let mut writer = SegmentWriter::new(
        dataset.object_store(None).await?,
        dataset.indices_dir().join(uuid.to_string()),
        metadata.clone(),
        target.manifest.ivf().clone(),
    );

    let mut stats = InsertStats {
        fragments_indexed: new_fragments.len(),
        vectors: vectors.len(),
        ..Default::default()
    };
    let growth = Growth {
        index: &index,
        target,
        members,
        row_ids,
        vectors: Arc::new(vectors),
        params,
        metadata,
    };
    grow_segment(&growth, &mut writer, &mut stats).await?;
    writer.finish().await?;

    let dataset_version = dataset.manifest.version;
    log::info!(
        "Vamana index '{index_name}' grew segment {} into {uuid}: {} partitions gained rows, {} \
         were created, {} were copied",
        target.uuid,
        stats.partitions_grown,
        stats.partitions_created,
        stats.partitions_copied
    );
    dataset
        .commit_existing_index_segments(
            index_name,
            &column,
            vec![IndexSegment::new(
                uuid,
                coverage.iter(),
                target.fields.iter().copied(),
                Arc::new(prost_types::Any {
                    type_url: INDEX_DETAILS_TYPE_URL.to_string(),
                    value: Vec::new(),
                }),
                FORMAT_VERSION as i32,
                dataset_version,
            )],
        )
        .await?;
    Ok(stats)
}

/// Everything the per-partition loop reads, gathered so that the loop's own
/// signature stays legible.
struct Growth<'a> {
    index: &'a VamanaIndex,
    target: &'a Segment,
    /// Which rows of [`Self::vectors`] each partition drew, by partition id.
    members: Vec<Vec<u32>>,
    row_ids: Arc<Vec<u64>>,
    vectors: Arc<FixedSizeListArray>,
    params: Arc<IndexParams>,
    /// What the segment being written declares, which is the target's own
    /// metadata with the new fragments folded into the coverage.
    metadata: IndexMetadata,
}

/// Write every partition of the grown segment, in ascending id order.
///
/// One pass over the whole partition space rather than over the two lists
/// separately, because the ids of a segment's partitions and the ids that drew a
/// new row are two sorted sets that have to be merged, and the writer accepts
/// them in ascending order only. Walking the space costs a lookup per centroid
/// against a partition read per occupied one.
///
/// A partition at a time, as consolidation does it and for the same reason: a
/// partition is read whole, so overlapping the reads would mean holding as many
/// of them in memory as are kept in flight.
async fn grow_segment(
    growth: &Growth<'_>,
    writer: &mut SegmentWriter,
    stats: &mut InsertStats,
) -> Result<()> {
    for (partition_id, members) in growth.members.iter().enumerate() {
        let partition_id = partition_id as u32;
        match (
            growth.target.manifest.partition(partition_id),
            members.is_empty(),
        ) {
            // A centroid nothing was ever assigned to. It has no file and no
            // table row, and it still has none.
            (None, true) => continue,
            (Some(_), true) => {
                writer
                    .copy_partition(&growth.target.dir, &growth.target.manifest, partition_id)
                    .await?;
                stats.partitions_copied += 1;
            }
            // A centroid the base drew nothing for and this batch did. Built
            // rather than inserted into: there is no graph to insert into, and
            // no entry point to search from.
            (None, false) => {
                let members = members.clone();
                let row_ids = growth.row_ids.clone();
                let vectors = growth.vectors.clone();
                let params = growth.params.clone();
                let built =
                    spawn_cpu(move || build_one(&members, row_ids.as_slice(), &vectors, &params))
                        .await?;
                writer
                    .write_partition(partition_id, built.medoid, &built.partition)
                    .await?;
                stats.comparisons = stats.comparisons.saturating_add(built.comparisons);
                stats.partitions_created += 1;
            }
            (Some(entry), false) => {
                let reader = open_file(
                    growth.index.scheduler(),
                    &growth.target.dir.clone().join(entry.file.as_str()),
                    None,
                    growth.target.file_sizes.get(&entry.file).copied(),
                )
                .await?;
                let partition = read_partition(&reader, entry.num_rows).await?;
                check_partition_shape(
                    &partition,
                    entry,
                    growth.metadata.max_degree,
                    growth.metadata.dimension,
                )?;

                let members = members.clone();
                let row_ids = growth.row_ids.clone();
                let vectors = growth.vectors.clone();
                let params = growth.params.clone();
                let entry_point = entry.medoid;
                let (inserted, comparisons) = spawn_cpu(move || {
                    let batch = gather(&vectors, &members)?;
                    let batch_row_ids = members
                        .iter()
                        .map(|row| row_ids[*row as usize])
                        .collect::<Vec<_>>();
                    let comparisons = Comparisons::default();
                    let inserted = insert_into_partition(
                        &partition,
                        &batch_row_ids,
                        &batch,
                        entry_point,
                        params.distance_type,
                        &params.graph,
                        &comparisons,
                    )?;
                    Ok::<_, Error>((inserted, comparisons.get()))
                })
                .await?;
                writer
                    .write_partition(partition_id, inserted.medoid, &inserted.partition)
                    .await?;
                stats.comparisons = stats.comparisons.saturating_add(comparisons);
                stats.partitions_grown += 1;
            }
        }
    }
    Ok(())
}

/// Build parameters that will produce a segment the base can stand beside.
///
/// The graph half of them is what every maintenance pass works at, seed
/// included: [`BuildParams::maintenance`]. What this adds is the routing, which
/// only a driver over a dataset has an opinion about.
pub(crate) fn inherited_params(
    column: &str,
    base: &IndexMetadata,
    router: &IvfModel,
) -> IndexParams {
    IndexParams::new(column, router.num_partitions() as u32)
        .with_distance_type(base.distance_type)
        .with_graph_params(BuildParams::maintenance(base))
}
