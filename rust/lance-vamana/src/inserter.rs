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

use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::{Error, Result};
use lance_index::vector::ivf::storage::IvfModel;

use crate::build::{BuildParams, MAINTENANCE_SEED};
use crate::builder::{IndexParams, build_index_segment_with_router, index_column, live_fragments};
use crate::format::IndexMetadata;
use crate::query::{Segment, VamanaIndex};

/// What indexing a dataset's new rows did, and what it cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InsertStats {
    /// Fragments the index did not cover before and covers now.
    pub fragments_indexed: usize,
    /// Vectors indexed, which is the rows of those fragments minus the ones
    /// whose vector is null.
    pub vectors: usize,
    /// Partitions that drew at least one row and were therefore written.
    pub partitions_written: usize,
    /// Distance computations spent building the graphs.
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
    let new_fragments = unindexed_fragments(dataset, &index);
    if new_fragments.is_empty() {
        return Ok(InsertStats::default());
    }

    let base = base_segment(&index)?;
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
        partitions_written: built.partitions,
        comparisons: built.comparisons,
    })
}

/// Fragments the dataset has and the index does not answer for.
///
/// Against `covered_fragments` rather than against any segment's own record,
/// because the two differ exactly when a fragment has gone: a fragment a segment
/// was built over and the dataset has dropped is answered for by nobody, and if
/// a compaction rewrote its rows into a new fragment then that new fragment
/// belongs in this list.
fn unindexed_fragments(dataset: &Dataset, index: &VamanaIndex) -> Vec<u32> {
    live_fragments(dataset)
        .into_iter()
        .filter(|fragment| !index.covered_fragments().contains(*fragment))
        .collect()
}

/// The segment a further one should be modelled on: the one covering the most
/// fragments, and on a tie the one the manifest lists first.
///
/// The base rather than a delta, which is what "most fragments" means in
/// practice, so that deltas inherit the routing of the index's largest graph
/// instead of inheriting each other's. Deterministic on purpose: whose centroids
/// a segment was written under is not recoverable from the segment afterwards.
fn base_segment(index: &VamanaIndex) -> Result<&Segment> {
    index
        .segments()
        .iter()
        .reduce(|base, segment| {
            if segment.coverage.len() > base.coverage.len() {
                segment
            } else {
                base
            }
        })
        .ok_or_else(|| Error::internal("an opened Vamana index has no segments".to_string()))
}

/// Build parameters that will produce a segment the base can stand beside.
///
/// The seed is the one thing not taken from the base, because the base does not
/// record it: see [`MAINTENANCE_SEED`].
fn inherited_params(column: &str, base: &IndexMetadata, router: &IvfModel) -> IndexParams {
    IndexParams::new(column, router.num_partitions() as u32)
        .with_distance_type(base.distance_type)
        .with_graph_params(BuildParams {
            max_degree: base.max_degree,
            search_list_size: base.search_list_size,
            alpha: base.alpha,
            seed: MAINTENANCE_SEED,
        })
}
