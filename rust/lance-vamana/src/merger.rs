// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bringing an index up to date with its dataset in one pass over it.
//!
//! [`crate::consolidator`] takes deleted rows out, [`crate::inserter`] puts new
//! rows in, and each of them reads and rewrites every partition it touches. A
//! round of maintenance runs both, so a partition that lost a row and gained one
//! crosses the disk twice, and a partition that did neither is copied twice.
//! This is the two as a single pass, plus the one thing neither of them can do:
//! folding a delta segment back into the base.
//!
//! Everything an index has pending is the same shape - rows the graphs should no
//! longer hold, and rows they should. So there is one call, and no order to get
//! wrong:
//!
//! - **deleted rows** are vertices to take out, wherever they sit;
//! - **a delta segment's rows** are vertices to move: out of the delta, into the
//!   graph of the base partition with the same number;
//! - **an unindexed fragment's rows** are vertices to add, routed by the base's
//!   own centroids.
//!
//! What comes out is one segment covering every fragment the dataset has. Every
//! old segment is covered by it whole, so the commit removes all of them - that
//! is `commit_existing_index_segments`' own rule about coverage, not a deletion
//! this asks for.
//!
//! # Why partition 17 folds into partition 17
//!
//! Because [`crate::inserter::insert_as_segment`] makes a delta inherit the
//! base's centroids, every segment of an index shares one partition numbering: a
//! row in partition 17 of the delta is a row nearest centroid 17, which is the
//! centroid partition 17 of the base was filled from. Folding is then a
//! concatenation of like with like, and the delta's partitions can be read one
//! at a time as the pass reaches them.
//!
//! That is a property of how the segments were written, not of the format, and
//! nothing in [`crate::query::VamanaIndex::open`] checks it - a query ranks each
//! segment's centroids separately and does not care. So it is checked here,
//! before anything is written: segments whose centroids differ are refused
//! rather than folded, because folding them by number would file rows under a
//! centroid they are not nearest to, where the query that should find them
//! probes elsewhere. Re-routing them instead would be correct and would mean
//! holding a whole delta in memory, which is the one thing this pass is built
//! not to do.
//!
//! # What a partition costs
//!
//! A partition that nothing happened to is copied undecoded, exactly as
//! consolidation copies it. A partition that gained or lost anything is read
//! once, from each segment that holds a piece of it, and written once. So the
//! working set is one partition per segment, and the arithmetic is proportional
//! to what changed rather than to what the index holds - except in a partition
//! the deletions tore apart, which is built again from scratch. See
//! [`crate::merge`] for when that happens and what is deliberately not checked.

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array};
use lance::Dataset;
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use lance_index::vector::ivf::storage::IvfModel;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::build::build_partition;
use crate::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, assign, gather, group_by_partition, index_column,
    read_vectors,
};
use crate::consolidator::dead_by_partition;
use crate::format::{FORMAT_VERSION, IndexMetadata};
use crate::insert::concat_vectors;
use crate::inserter::inherited_params;
use crate::io::{SegmentWriter, check_partition_shape, open_file, read_partition};
use crate::merge::{Newcomers, merge_partition};
use crate::partition::Partition;
use crate::query::{Segment, VamanaIndex};
use crate::search::{Comparisons, flat_storage};
use crate::segment::PartitionEntry;

/// What bringing an index up to date did, and what it cost.
///
/// `partitions_written`, `partitions_copied` and `partitions_dropped` are
/// exclusive and between them account for every partition any segment held plus
/// every one the new rows called for. `partitions_rebuilt` is not a fourth
/// class: it counts the written ones that had to be built from scratch because
/// the merged graph came apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergeStats {
    /// Segments the merge replaced, which is every segment the index had.
    pub segments_folded: usize,
    /// Fragments the index did not cover before and covers now.
    pub fragments_indexed: usize,
    /// Rows of those fragments that were indexed, which is all of them but the
    /// ones whose vector is null.
    pub vectors_inserted: usize,
    /// Vertices lifted out of one segment's graph and linked into another's.
    ///
    /// The price of folding, and the number that says whether it was worth it:
    /// each of these was searched for and pruned into place exactly as a new row
    /// would be, because a graph's edges are local ids and mean nothing outside
    /// the partition file they were written in.
    pub vertices_folded: usize,
    /// Vertices no longer stored, because their rows are deleted or their
    /// fragment is gone.
    pub vertices_removed: usize,
    /// Partitions encoded by this pass, whether merged into or built.
    pub partitions_written: usize,
    /// Of those, the ones whose merged graph came apart and was built again.
    pub partitions_rebuilt: usize,
    /// Partitions nothing happened to, copied across undecoded.
    pub partitions_copied: usize,
    /// Partitions with nothing left in them, given no file and no table row.
    pub partitions_dropped: usize,
    /// Distance computations, across the repairs, the links and the rebuilds.
    pub comparisons: u64,
}

/// Apply everything `index_name` has pending and leave it as one segment.
///
/// Deleted rows go out of the graphs, delta segments fold into the base, and
/// rows in fragments no segment covers are indexed - all in one pass, in which
/// a partition is read at most once and written at most once.
///
/// Takes no parameters and has no threshold, as its neighbours do not: asked to
/// merge, this merges. *When* to ask is the caller's, and it is the one question
/// here that is really about money - a merge costs what it costs once, and a
/// delta left in place costs `nprobes` extra partition reads on every query
/// until it is folded. Measured on SIFT 100k, that crossover is **533 queries**
/// for eight segments, 1345 for four and 6509 for two: the fold costs 3.5 to 4.5
/// seconds whatever the count, while what it saves grows with it. At those
/// numbers a threshold would only be a slower way of saying "fold".
///
/// Nothing is committed when there is nothing to do, and "nothing to do" is
/// decided on the vertices rather than on the dataset's delete list, which
/// outlives them: one segment and no unindexed fragments returns a zero
/// [`MergeStats`] having paid for one `open` and, if rows have ever been deleted
/// from these fragments, one pass over the deletion vectors.
///
/// # Against the two calls it replaces
///
/// [`crate::consolidator::consolidate_index`] is still the cheaper answer when
/// deletions are all there is: it never reads a vector, never routes and leaves
/// the segments as they are. [`crate::inserter::insert_as_segment`] is still the
/// cheaper answer when new rows are all there is, and it is the one that does
/// not touch the base at all.
///
/// What this does that neither can: it needs no ordering. A delete that empties
/// a fragment takes it out of the dataset, and
/// [`crate::inserter::insert_in_place`] refuses such a segment outright, because
/// rewriting it would store vertices under a coverage that no longer names them.
/// Here those vertices are dropped by the same pass that adds the new ones -
/// `RowFilter` rejects a missing fragment exactly as it rejects a deleted row.
///
/// What it does not buy is speed. Over the five-round churn cycle of
/// `examples/churn_cycle.rs` on SIFT 100k this leaves an index with the same
/// recall and the same distances per query as the pair, to the last digit, in
/// every round - it runs the same operations over the same data without putting
/// the partition on disk in between - and costs 35.9 seconds against 37.0. The
/// missing pass is one read and one write of the index per round, and that is 3%
/// of a round on local storage: maintenance here is bound by the arithmetic of
/// the graph. The saving is worth having, and it is not the reason to call this.
///
/// # When it refuses
///
/// Segments that disagree about the degree or about the centroids. Opening an
/// index permits both, because a query walks each segment separately; folding
/// permits neither, because it moves vertices from one graph into another. And
/// an index whose every fragment is gone with nothing new to index, which
/// answers for nothing and wants rebuilding rather than merging.
///
/// A concurrent commit under the same index name is a retryable conflict, and
/// the retry re-runs this call from the beginning against the manifest that won.
/// The old segments' files stay referenced by the manifest versions before this
/// commit and go when `cleanup_old_versions` takes those versions, so an index
/// costs both copies until then.
pub async fn merge_index(dataset: &mut Dataset, index_name: &str) -> Result<MergeStats> {
    let index = VamanaIndex::open(dataset, index_name).await?;
    let new_fragments = index.unindexed_fragments(dataset);
    let has_dead = !index.row_filter().is_empty();
    if new_fragments.is_empty() && !has_dead && index.num_segments() == 1 {
        return Ok(MergeStats::default());
    }

    let base = index.base_segment()?;
    if base.coverage.is_empty() && new_fragments.is_empty() {
        return Err(Error::invalid_input(format!(
            "Vamana cannot merge index '{index_name}': every fragment its {} segments were built \
             over is gone and the dataset has nothing unindexed to put in their place, so there \
             would be nothing left to write; rebuild it instead",
            index.num_segments()
        )));
    }
    let router = base.manifest.ivf();
    let max_degree = base.manifest.metadata().max_degree;
    for segment in index.segments() {
        // Opening an index permits its segments to differ on the degree, because
        // a query only walks each of them separately. Folding one graph into
        // another does not: the slot count is the layout, so a vertex arriving
        // from a segment of another width has to be re-linked at this one, and
        // then the fold is a rebuild wearing a fold's name.
        if segment.manifest.metadata().max_degree != max_degree {
            return Err(Error::invalid_input(format!(
                "Vamana cannot merge index '{index_name}': segment {} holds graphs of degree {} \
                 where segment {} holds graphs of degree {max_degree}, and one cannot be folded \
                 into the other without being built again; rebuild the index",
                segment.uuid,
                segment.manifest.metadata().max_degree,
                base.uuid
            )));
        }
        if segment.manifest.ivf().centroids != router.centroids {
            return Err(Error::invalid_input(format!(
                "Vamana cannot merge index '{index_name}': segment {} was routed by centroids of \
                 its own rather than by segment {}'s, so partition n of the one is not partition n \
                 of the other, and folding them by number would file rows under a centroid they \
                 are not nearest to; rebuild the index",
                segment.uuid, base.uuid
            )));
        }
    }

    let column = index_column(dataset, index_name, &base.fields)?;
    let params = Arc::new(inherited_params(&column, base.manifest.metadata(), router));
    let arrivals = arrivals(
        dataset,
        &column,
        &new_fragments,
        &params,
        router,
        base.manifest.metadata().dimension,
    )
    .await?;

    let store = dataset.object_store(None).await?;
    let io_parallelism = store.io_parallelism();
    // The base first, so that a partition it holds is the graph the others fold
    // into rather than the other way about.
    let mut segments = Vec::with_capacity(index.num_segments());
    for segment in std::iter::once(base).chain(index.segments().iter().filter(|other| {
        // `base_segment` returns a reference into the same vector, so identity
        // is the comparison, not equality of two large manifests.
        !std::ptr::eq(*other, base)
    })) {
        let dead = if has_dead {
            dead_by_partition(&index, segment, io_parallelism).await?
        } else {
            vec![RoaringBitmap::new(); segment.manifest.partitions().len()]
        };
        segments.push((segment, dead));
    }
    // The dataset's delete list outlives the vertices it condemned: a row taken
    // out by an earlier round is still in it, so `has_dead` stays true for the
    // life of the fragment while the graphs hold nothing of it. Deciding on the
    // vertices instead is what stops a maintenance loop rewriting the whole
    // index every run, and it is the guard `consolidate_index` makes per segment.
    if new_fragments.is_empty()
        && index.num_segments() == 1
        && segments
            .iter()
            .all(|(_, dead)| dead.iter().all(RoaringBitmap::is_empty))
    {
        return Ok(MergeStats::default());
    }

    let uuid = Uuid::new_v4();
    let mut coverage = index.covered_fragments().clone();
    coverage.extend(new_fragments.iter().copied());
    let metadata = IndexMetadata {
        fragments: coverage.iter().collect(),
        ..base.manifest.metadata().clone()
    };
    let mut writer = SegmentWriter::new(
        store.clone(),
        dataset.indices_dir().join(uuid.to_string()),
        metadata.clone(),
        router.clone(),
    );

    let mut stats = MergeStats {
        segments_folded: index.num_segments(),
        fragments_indexed: new_fragments.len(),
        ..Default::default()
    };
    let fold = Fold {
        index: &index,
        segments,
        arrivals,
        params,
        metadata,
    };
    for partition_id in 0..router.num_partitions() as u32 {
        fold_partition(&fold, partition_id, &mut writer, &mut stats).await?;
    }
    if stats.partitions_written + stats.partitions_copied == 0 {
        return Err(Error::invalid_input(format!(
            "Vamana cannot merge index '{index_name}': every one of its vertices is deleted and \
             there is nothing unindexed to replace them with, so there would be nothing left to \
             write; rebuild it instead"
        )));
    }
    writer.finish().await?;

    let dataset_version = dataset.manifest.version;
    log::info!(
        "Vamana index '{index_name}' folded {} segments into {uuid}: {} partitions written, {} \
         copied, {} dropped, {} vertices folded and {} removed",
        stats.segments_folded,
        stats.partitions_written,
        stats.partitions_copied,
        stats.partitions_dropped,
        stats.vertices_folded,
        stats.vertices_removed
    );
    dataset
        .commit_existing_index_segments(
            index_name,
            &column,
            vec![IndexSegment::new(
                uuid,
                coverage.iter(),
                base.fields.iter().copied(),
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

/// The dataset's new rows, read once and routed by the base's centroids.
struct Arrivals {
    row_ids: Vec<u64>,
    vectors: FixedSizeListArray,
    /// Which rows of [`Self::vectors`] each partition drew, by partition id.
    members: Vec<Vec<u32>>,
}

/// Read every row of `fragments` and decide which partition each belongs to.
///
/// Empty fragments are the common case - most merges are folds and clean-ups -
/// and they cost nothing here: no scan, no routing pass, and an empty vector
/// array that the partition loop never reaches, because a partition with no
/// members never asks for its rows.
async fn arrivals(
    dataset: &Dataset,
    column: &str,
    fragments: &[u32],
    params: &IndexParams,
    router: &IvfModel,
    dimension: u32,
) -> Result<Arrivals> {
    if fragments.is_empty() {
        return Ok(Arrivals {
            row_ids: Vec::new(),
            vectors: FixedSizeListArray::try_new_from_values(
                Float32Array::from(Vec::<f32>::new()),
                dimension as i32,
            )?,
            members: vec![Vec::new(); params.num_partitions as usize],
        });
    }

    let (row_ids, vectors) = read_vectors(dataset, column, fragments).await?;
    // Normalising, routing and grouping are one uninterrupted pass over the
    // whole batch, which is why they go to the pool rather than run on the
    // runtime the scheduler reads through.
    let router = router.clone();
    let distance_type = params.distance_type;
    let num_partitions = params.num_partitions;
    let (row_ids, vectors, members) = spawn_cpu(move || {
        let vectors = if distance_type == DistanceType::Cosine {
            normalize_fsl_owned(vectors)?
        } else {
            vectors
        };
        let assignment = assign(&router, &vectors, &row_ids, distance_type)?;
        let members = group_by_partition(&assignment, num_partitions);
        Ok::<_, Error>((row_ids, vectors, members))
    })
    .await?;

    Ok(Arrivals {
        row_ids,
        vectors,
        members,
    })
}

/// One segment's contribution to one partition.
struct Source<'a> {
    segment: &'a Segment,
    entry: &'a PartitionEntry,
    dead: &'a RoaringBitmap,
}

/// Everything the per-partition loop reads, gathered so that the loop's own
/// signature stays legible.
struct Fold<'a> {
    index: &'a VamanaIndex,
    /// Every segment of the index, base first, each with the dead vertices of
    /// each of its partitions in the order the segment lists them.
    segments: Vec<(&'a Segment, Vec<RoaringBitmap>)>,
    arrivals: Arrivals,
    params: Arc<IndexParams>,
    /// What the segment being written declares.
    metadata: IndexMetadata,
}

/// Write partition `partition_id` of the merged segment, from whatever the
/// index's segments hold of it and whatever the new rows added to it.
async fn fold_partition(
    fold: &Fold<'_>,
    partition_id: u32,
    writer: &mut SegmentWriter,
    stats: &mut MergeStats,
) -> Result<()> {
    let arrivals = &fold.arrivals.members[partition_id as usize];
    let mut sources = Vec::with_capacity(fold.segments.len());
    let mut had_a_file = false;
    for (segment, dead) in &fold.segments {
        let Ok(position) = segment
            .manifest
            .partitions()
            .binary_search_by_key(&partition_id, |entry| entry.partition_id)
        else {
            continue;
        };
        had_a_file = true;
        let entry = &segment.manifest.partitions()[position];
        let dead = &dead[position];
        // Dropped here rather than read: a segment whose every vertex of this
        // partition is deleted has nothing to carry over, and reading it would
        // be the whole file for nothing. It also keeps the rule below simple -
        // every source that survives this loop has at least one live row.
        if dead.len() == entry.num_rows as u64 {
            stats.vertices_removed += dead.len() as usize;
            continue;
        }
        sources.push(Source {
            segment,
            entry,
            dead,
        });
    }

    if sources.is_empty() && arrivals.is_empty() {
        if had_a_file {
            stats.partitions_dropped += 1;
        }
        return Ok(());
    }

    // Nothing happened to it: one segment holds it, nothing in it is deleted and
    // no new row landed in it. The bytes are already a partition of exactly the
    // shape this segment declares - every segment of the index agrees on the
    // degree, or the merge refused before reading anything - so they cross over
    // without being decoded.
    if arrivals.is_empty() && sources.len() == 1 && sources[0].dead.is_empty() {
        writer
            .copy_partition(
                &sources[0].segment.dir,
                &sources[0].segment.manifest,
                partition_id,
            )
            .await?;
        stats.partitions_copied += 1;
        return Ok(());
    }

    let mut read = Vec::with_capacity(sources.len());
    for source in &sources {
        let reader = open_file(
            fold.index.scheduler(),
            &source.segment.dir.clone().join(source.entry.file.as_str()),
            None,
            source.segment.file_sizes.get(&source.entry.file).copied(),
        )
        .await?;
        let partition = read_partition(&reader, source.entry.num_rows).await?;
        check_partition_shape(
            &partition,
            source.entry,
            fold.metadata.max_degree,
            fold.metadata.dimension,
        )?;
        read.push(partition);
    }

    // The graph the others fold into: the largest of them, because linking the
    // smaller side into the bigger graph is both cheaper and better than the
    // other way about. Usually that is the base segment's, which is why it is
    // first in the list and wins a tie.
    let mut into: Option<usize> = None;
    for (position, partition) in read.iter().enumerate() {
        if into.is_none_or(|best| partition.len() > read[best].len()) {
            into = Some(position);
        }
    }

    let mut newcomer_row_ids = Vec::new();
    let mut newcomer_vectors = Vec::new();
    for (position, partition) in read.iter().enumerate() {
        if into == Some(position) {
            continue;
        }
        let dead = sources[position].dead;
        let live = (0..partition.len() as u32)
            .filter(|id| !dead.contains(*id))
            .collect::<Vec<_>>();
        stats.vertices_removed += dead.len() as usize;
        stats.vertices_folded += live.len();
        newcomer_row_ids.extend(
            live.iter()
                .map(|id| partition.graph().row_ids()[*id as usize]),
        );
        newcomer_vectors.push(gather(partition.vectors(), &live)?);
    }
    if !arrivals.is_empty() {
        newcomer_row_ids.extend(
            arrivals
                .iter()
                .map(|row| fold.arrivals.row_ids[*row as usize]),
        );
        newcomer_vectors.push(gather(&fold.arrivals.vectors, arrivals)?);
        stats.vectors_inserted += arrivals.len();
    }

    let into = into.map(|position| {
        let dead = sources[position].dead.clone();
        stats.vertices_removed += dead.len() as usize;
        (
            read.swap_remove(position),
            sources[position].entry.medoid,
            dead,
        )
    });
    let distance_type = fold.metadata.distance_type;
    let params = fold.params.clone();
    let (partition, medoid, rebuilt, comparisons) = spawn_cpu(move || {
        let comparisons = Comparisons::default();
        let vectors = if newcomer_vectors.is_empty() {
            None
        } else {
            Some(concat_vectors(&newcomer_vectors)?)
        };
        let (partition, medoid, rebuilt) = match into {
            Some((base, entry_point, dead)) => {
                let newcomers = vectors.as_ref().map(|vectors| Newcomers {
                    row_ids: &newcomer_row_ids,
                    vectors,
                });
                let merged = merge_partition(
                    &base,
                    entry_point,
                    &dead,
                    newcomers,
                    distance_type,
                    &params.graph,
                    &comparisons,
                )?;
                (merged.partition, merged.medoid, merged.rebuilt)
            }
            // No segment held this partition, so there is no graph to insert
            // into and no entry point to search from: everything here is a
            // newcomer, and a graph over newcomers alone is a build.
            None => {
                let vectors = vectors.ok_or_else(|| {
                    Error::internal(
                        "a Vamana partition reached the merge with neither a graph nor rows"
                            .to_string(),
                    )
                })?;
                let store = flat_storage(&newcomer_row_ids, &vectors, distance_type)?;
                let built = build_partition(&store, &params.graph, &comparisons)?;
                (
                    Partition::try_new(built.graph, vectors)?,
                    built.medoid,
                    false,
                )
            }
        };
        Ok::<_, Error>((partition, medoid, rebuilt, comparisons.get()))
    })
    .await?;

    writer
        .write_partition(partition_id, medoid, &partition)
        .await?;
    stats.comparisons = stats.comparisons.saturating_add(comparisons);
    stats.partitions_written += 1;
    if rebuilt {
        stats.partitions_rebuilt += 1;
    }
    Ok(())
}
