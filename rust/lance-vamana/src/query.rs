// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Answering a nearest-neighbour query from a committed Vamana index.
//!
//! The driver is ours end to end: it finds the index's segments through the
//! dataset's public index metadata, routes the query with the IVF model each
//! segment carries, walks the graph of every probed partition and merges the
//! answers into dataset row ids. Lance's scanner never sees the query, which is
//! what makes this work without a patch to Lance - and also what it costs.
//!
//! What this driver does not do, and a caller has to know. The crate README
//! carries the same list for a reader who is not in the source; the two are
//! meant to say the same thing.
//!
//! - **The delete list is a snapshot taken at open.** Deleted rows are excluded
//!   from answers, but the list is read once, when the index is opened. A row
//!   deleted afterwards keeps coming back until the index is reopened, and
//!   nothing about the answer reveals it - which is why it is spelled out.
//! - **Fewer than `k` rows come back when a probed partition is mostly
//!   deleted.** Deleted vertices are still walked - they carry the edges that
//!   hold the graph together - but they are dropped from the answer, and a walk
//!   only ever produces `search_list_size` candidates to draw from.
//! - **Rows added after the build are invisible.** The index answers from the
//!   fragments it was built over; Lance's scanner would scan the remainder.
//! - **No predicate prefilter and no refine step.** Both live in the scanner.
//! - **Partitions are read whole, and nothing is cached between queries.** A
//!   query keeps a few reads going at once, so its working
//!   set is a few partitions rather than every partition it probes - but the
//!   lazy per-vertex traversal and the cache budget are both still ahead, and
//!   putting either in early would make the first honest measurement of this
//!   path harder to read.
//!
//! [`VamanaIndex::open`] refuses outright, rather than answering from what is
//! left, when the fragments have been compacted away, when the dataset has
//! edited the index's coverage underneath it, when the manifest records a format
//! version this build does not read, or when the segments disagree about the
//! vectors they hold. Each refusal names what to do about it, which is always to
//! rebuild.
//!
//! Committing an index also breaks Lance's own vector search on that column -
//! see the crate README, and the test that pins it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array};
use futures::stream::{self, StreamExt, TryStreamExt};
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::vector::storage::VectorStore;
use lance_io::scheduler::ScanScheduler;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use object_store::path::Path;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::builder::{routing_distance_type, supported_distance_type};
use crate::format::{FORMAT_VERSION, INDEX_FILE_NAME, IndexMetadata, RowIdMode};
use crate::io::{open_file, read_partition, read_segment, scan_scheduler};
use crate::partition::Partition;
use crate::search::{Comparisons, SearchScratch, flat_storage, greedy_search};
use crate::segment::{PartitionEntry, SegmentManifest};

/// One answer: a dataset row id and its distance from the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub row_id: u64,
    pub distance: f32,
}

/// How far a query is allowed to look.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// How many neighbours to return.
    pub k: usize,
    /// How many IVF partitions that hold vectors to open, per segment.
    ///
    /// Empty partitions do not count against it. Routing scores every centroid
    /// whether or not anything was assigned to it, so a budget spent on centroids
    /// rather than on data would let the nearest one silently return nothing.
    pub nprobes: usize,
    /// `L`: how wide a search list each graph walk keeps.
    pub search_list_size: usize,
}

impl SearchParams {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            nprobes: 1,
            search_list_size: k + k / 2,
        }
    }

    pub fn with_nprobes(mut self, nprobes: usize) -> Self {
        self.nprobes = nprobes;
        self
    }

    pub fn with_search_list_size(mut self, search_list_size: usize) -> Self {
        self.search_list_size = search_list_size;
        self
    }
}

/// What a query found, and what it cost to find it.
///
/// The cost travels with the answer rather than being logged, because recall
/// without a cost is not a number: a walk that reaches every vertex in the
/// partition scores perfectly and has answered nothing.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Nearest first.
    pub neighbors: Vec<Neighbor>,
    /// Every distance this query computed: one per centroid of every segment it
    /// routed through, plus one per vertex any graph walk considered.
    ///
    /// Routing is counted because it is paid unconditionally and does not scale
    /// with `nprobes` - a segment of 4096 centroids charges 4096 distances
    /// before a single vertex is read. Reporting only the walk would make a
    /// finely partitioned index look cheap at exactly the point it stops being.
    pub comparisons: u64,
    pub partitions_read: usize,
}

/// A committed Vamana index, opened for querying.
#[derive(Debug)]
pub struct VamanaIndex {
    scheduler: Arc<ScanScheduler>,
    metadata: IndexMetadata,
    segments: Vec<Segment>,
    /// Row addresses deleted as of [`VamanaIndex::open`].
    ///
    /// A snapshot, not a live view: the graph files hold vertices for rows that
    /// have since been deleted, and nothing rewrites them, so the only way to
    /// tell a live vertex from a dead one is to ask the dataset - once, here,
    /// rather than on every query.
    deleted: RoaringTreemap,
}

#[derive(Debug)]
struct Segment {
    dir: Path,
    manifest: SegmentManifest,
    /// Byte size of each file of this segment, as Lance recorded it at commit.
    ///
    /// Lance fills this by listing the directory, so it is a fact about the
    /// files rather than a second copy of one this crate wrote. Handing it to
    /// the reader is what turns opening a partition into one read rather than a
    /// size probe followed by a read.
    file_sizes: HashMap<String, u64>,
}

/// One partition a query has decided to read, and all of what reading it needs.
///
/// Owned rather than borrowed out of the segment. A `Probe<'a>` would make the
/// closure that turns probes into reads higher-ranked over `'a`, and the search
/// future built from it stops being `Send` - which takes `tokio::spawn` away
/// from every caller. The clones are one small string and two numbers per
/// partition actually read.
#[derive(Debug)]
struct Probe {
    path: Path,
    size_bytes: Option<u64>,
    entry: PartitionEntry,
    /// What the segment declares, to be checked against what the file holds.
    max_degree: u32,
    dimension: u32,
}

/// How many partition reads a query keeps in flight.
///
/// The bound is on memory: a partition is read whole, so this is the working set
/// in partitions however many a query probes. Four rather than one because a
/// walk cannot start until a read finishes and a store with any latency would
/// then sit idle through every walk; four rather than `nprobes` because that is
/// not a bound at all. What the number should be on a high-latency store is a
/// measurement nobody has taken, so it is deliberately on the small side.
const PARTITIONS_IN_FLIGHT: usize = 4;

impl VamanaIndex {
    /// Open every segment of `index_name`.
    pub async fn open(dataset: &Dataset, index_name: &str) -> Result<Self> {
        // `load_indices_by_name` and not `load_index_by_name`: the latter errors
        // out as soon as an index has more than one segment, which is the normal
        // state of anything that has ever been appended to.
        let indices = dataset.load_indices_by_name(index_name).await?;
        if indices.is_empty() {
            return Err(Error::index(format!(
                "dataset has no index named '{index_name}'"
            )));
        }

        let live = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<RoaringBitmap>();
        let scheduler = scan_scheduler(&dataset.object_store(None).await?);

        let mut segments = Vec::with_capacity(indices.len());
        for index in indices.iter() {
            // Checked here as well as in the segment's own metadata, because the
            // two are separate records in separate files and either can be the
            // one that is wrong. This one is what makes a refusal cost nothing:
            // a segment written by a future build is turned away before a single
            // one of its files is opened.
            if index.index_version != FORMAT_VERSION as i32 {
                return Err(Error::not_supported(format!(
                    "index '{index_name}' segment {} is at format version {}, and this build \
                     reads version {FORMAT_VERSION}",
                    index.uuid, index.index_version
                )));
            }

            // A compaction that cannot open an index does not remove it: the
            // manifest entry survives, still naming the fragments it was built
            // over. The rows of any fragment that has since been rewritten or
            // dropped are stored here under row addresses that will never
            // resolve again, and they would win places in the top-k and then be
            // silently discarded by the caller's `take_rows`.
            //
            // So the test is equality with the *declared* coverage, not merely
            // a non-empty intersection: a compaction usually retires only the
            // fragments below its size threshold, which leaves the intersection
            // non-empty and half the index dangling.
            //
            // The `None` arm is for manifests older than the field itself:
            // `IndexSegment` carries a plain bitmap, so nothing this crate can
            // commit reaches it and no test can produce one.
            let Some(declared) = index.fragment_bitmap.as_ref() else {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} records no fragment coverage",
                    index.uuid
                )));
            };
            let still_live = declared & &live;
            if still_live != *declared {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built over {} fragments the dataset no \
                     longer has, so it holds row addresses that cannot resolve; rebuild the index",
                    index.uuid,
                    declared.len() - still_live.len()
                )));
            }
            let dir = dataset.indices_dir().join(index.uuid.to_string());
            let file_sizes = index
                .files
                .iter()
                .flatten()
                .map(|file| (file.path.clone(), file.size_bytes))
                .collect::<HashMap<_, _>>();
            let manifest =
                read_segment(&scheduler, &dir, file_sizes.get(INDEX_FILE_NAME).copied()).await?;

            // The check above asks whether the dataset still has the fragments.
            // This one asks whether the dataset still credits the segment with
            // the fragments it was built from, which is a different question
            // with a different answer: Lance edits an index's coverage in place
            // and never touches the segment's own files. An in-place column
            // update removes the rewritten fragments from the bitmap while the
            // fragment ids and every row address survive, so the fragments are
            // all still live and the vectors stored here are all stale. A pure
            // row rewrite goes the other way and credits us with a fragment we
            // never read. Equality catches both; a subset test catches neither.
            let built_over = manifest
                .metadata()
                .fragments
                .iter()
                .copied()
                .collect::<RoaringBitmap>();
            if built_over != *declared {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built over {} fragments but the dataset \
                     now credits it with {}, so something rewrote data under it and the vectors it \
                     holds no longer match the rows at those addresses; rebuild the index",
                    index.uuid,
                    built_over.len(),
                    declared.len()
                )));
            }
            segments.push(Segment {
                dir,
                manifest,
                file_sizes,
            });
        }

        let metadata = segments[0].manifest.metadata().clone();
        for segment in &segments[1..] {
            let other = segment.manifest.metadata();
            // Degree and pruning slack may legitimately differ between a base
            // segment and one appended later; the identifier space, the metric
            // and the width may not, because a query mixes their answers.
            if (other.dimension, other.distance_type, other.row_id_mode)
                != (
                    metadata.dimension,
                    metadata.distance_type,
                    metadata.row_id_mode,
                )
            {
                return Err(Error::index(format!(
                    "index '{index_name}' has segments that disagree about the vectors they hold: \
                     {:?} against {:?}",
                    metadata, other
                )));
            }
        }

        if metadata.row_id_mode != RowIdMode::Address || dataset.manifest().uses_stable_row_ids() {
            return Err(Error::index(format!(
                "index '{index_name}' was built for {:?} row ids but the dataset uses {}",
                metadata.row_id_mode,
                if dataset.manifest().uses_stable_row_ids() {
                    "stable ones"
                } else {
                    "addresses"
                }
            )));
        }
        supported_distance_type(metadata.distance_type)?;

        let covered = segments
            .iter()
            .flat_map(|segment| segment.manifest.metadata().fragments.iter().copied())
            .collect::<RoaringBitmap>();
        let deleted = deleted_row_addresses(dataset, &covered).await?;

        Ok(Self {
            scheduler,
            metadata,
            segments,
            deleted,
        })
    }

    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Find the `k` nearest row ids to `query`.
    pub async fn search(&self, query: &[f32], params: &SearchParams) -> Result<QueryResult> {
        if params.k == 0 {
            return Err(Error::invalid_input(
                "k must be greater than zero".to_string(),
            ));
        }
        if params.nprobes == 0 {
            return Err(Error::invalid_input(
                "nprobes must be greater than zero".to_string(),
            ));
        }
        if params.search_list_size < params.k {
            return Err(Error::invalid_input(format!(
                "search_list_size {} is smaller than k {}, so a walk could never return k \
                 neighbours",
                params.search_list_size, params.k
            )));
        }
        if query.len() != self.metadata.dimension as usize {
            return Err(Error::invalid_input(format!(
                "query has {} dimensions but the index holds {}",
                query.len(),
                self.metadata.dimension
            )));
        }

        let query: ArrayRef = Arc::new(Float32Array::from(query.to_vec()));
        let routing_type = routing_distance_type(self.metadata.distance_type);
        // The router only knows L2 and dot - it panics on anything else - so a
        // cosine index routes a unit query by L2, over the unit vectors the
        // builder stored. The graph walk itself still uses the real metric.
        let routing_query = if self.metadata.distance_type == DistanceType::Cosine {
            normalize_arrow(query.as_ref())?.0
        } else {
            query.clone()
        };

        let (probes, mut comparisons) = self.route(&routing_query, routing_type, params)?;
        let mut found = Vec::with_capacity(probes.len() * params.k);
        let mut partitions_read = 0usize;
        let mut reads = std::pin::pin!(
            stream::iter(probes)
                .map(|probe| self.read_probe(probe))
                .buffered(PARTITIONS_IN_FLIGHT)
        );

        while let Some((partition, medoid)) = reads.try_next().await? {
            partitions_read += 1;
            // Created and dropped inside one iteration, never held across the
            // await above: `Comparisons` holds a `Cell`, and a reference to one
            // alive across an await point makes this whole future `!Send`, which
            // would take `tokio::spawn` away from every caller.
            let walked = Comparisons::default();
            let vectors = flat_storage(
                partition.graph().row_ids(),
                partition.vectors(),
                self.metadata.distance_type,
            )?;
            let calculator = vectors.dist_calculator(query.clone(), 0.0);
            let mut scratch = SearchScratch::new(partition.len());
            let walk = greedy_search(
                partition.graph(),
                &calculator,
                medoid,
                params.search_list_size,
                &mut scratch,
                &walked,
            )?;
            // Local ids are per partition, so they become row ids *before* the
            // merge: every partition has a vertex 0, and they are different rows.
            //
            // Deleted vertices are dropped here and not earlier. They are still
            // walked, because they carry the out-edges that keep the graph
            // connected - removing them from the traversal would strand whatever
            // they were the only route to. Filtering before `take` rather than
            // after is what makes `k` mean "k live rows" instead of "k rows, some
            // of which the caller will find missing".
            found.extend(
                walk.candidates
                    .iter()
                    .map(|node| Neighbor {
                        row_id: partition.graph().row_ids()[node.id as usize],
                        distance: node.dist.0,
                    })
                    .filter(|neighbor| !self.deleted.contains(neighbor.row_id))
                    .take(params.k),
            );
            comparisons = comparisons.saturating_add(walked.get());
        }

        Ok(QueryResult {
            neighbors: merge(found, params.k),
            comparisons,
            partitions_read,
        })
    }

    /// Decide which partitions to read, and say what deciding cost.
    ///
    /// No I/O: routing is pure arithmetic over the centroids each segment
    /// carries, and separating it from the reading is what lets the reads run
    /// against each other afterwards.
    fn route(
        &self,
        routing_query: &ArrayRef,
        routing_type: DistanceType,
        params: &SearchParams,
    ) -> Result<(Vec<Probe>, u64)> {
        let mut probes = Vec::new();
        let mut routing = 0u64;
        for segment in &self.segments {
            // Every centroid is ranked, not just `nprobes` of them, because a
            // centroid with nothing assigned to it is still a centroid: it can be
            // the nearest one, and a probe spent on it would read no vectors at
            // all. Ranking them all costs nothing extra - `find_partitions`
            // measures the query against every centroid either way.
            let (partitions, _) = segment.manifest.ivf().find_partitions(
                routing_query.as_ref(),
                segment.manifest.ivf().num_partitions(),
                routing_type,
            )?;
            routing = routing.saturating_add(segment.manifest.ivf().num_partitions() as u64);
            let mut probed = 0;
            for partition_id in partitions.values() {
                if probed == params.nprobes {
                    break;
                }
                // An empty partition has no row in the segment table and no file
                // of its own. Skipping it is the normal case rather than a sign
                // of a damaged segment.
                let Some(entry) = segment.manifest.partition(*partition_id) else {
                    continue;
                };
                probed += 1;
                let declared = segment.manifest.metadata();
                probes.push(Probe {
                    path: segment.dir.clone().join(entry.file.as_str()),
                    size_bytes: segment.file_sizes.get(&entry.file).copied(),
                    entry: entry.clone(),
                    max_degree: declared.max_degree,
                    dimension: declared.dimension,
                });
            }
        }
        Ok((probes, routing))
    }

    /// Read one probed partition whole.
    async fn read_probe(&self, probe: Probe) -> Result<(Partition, u32)> {
        let reader = open_file(&self.scheduler, &probe.path, None, probe.size_bytes).await?;
        let partition = read_partition(&reader, probe.entry.num_rows).await?;
        // The writer checks both against the segment on the way out; the reader
        // has to check them on the way back in. A partition whose width
        // disagrees with the manifest would be searched with a query of the
        // wrong length against `flat_storage`, which takes its dimension from
        // the array - silently wrong distances, not an error.
        if partition.graph().max_degree() != probe.max_degree
            || partition.dimension() != probe.dimension
        {
            return Err(Error::corrupt_file_named(
                probe.entry.file.as_str(),
                format!(
                    "Vamana partition {} holds degree {} and dimension {} but its segment \
                     declares degree {} and dimension {}",
                    probe.entry.partition_id,
                    partition.graph().max_degree(),
                    partition.dimension(),
                    probe.max_degree,
                    probe.dimension
                ),
            ));
        }
        Ok((partition, probe.entry.medoid))
    }
}

/// Every walk's candidates as one answer: nearest first, each row once, `k` long.
///
/// Nothing upstream of here guarantees a row appears once. That rests on Lance
/// refusing to commit segments whose fragment coverage overlaps, which is
/// somebody else's invariant, so the merge does not lean on it. Nor does the
/// dedup ride along with the ordering the caller sees: keyed on the row id in a
/// pass of its own, it collapses two copies of a row whatever their distances,
/// where a dedup run after a distance sort would only collapse the copies that
/// agree to the last bit - and the ones that disagree are exactly the ones worth
/// not returning twice.
fn merge(mut found: Vec<Neighbor>, k: usize) -> Vec<Neighbor> {
    found.sort_by(|left, right| {
        left.row_id
            .cmp(&right.row_id)
            .then(left.distance.total_cmp(&right.distance))
    });
    found.dedup_by_key(|neighbor| neighbor.row_id);
    found.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then(left.row_id.cmp(&right.row_id))
    });
    found.truncate(k);
    found
}

/// Row addresses deleted from the fragments an index covers.
///
/// Deletion vectors are per fragment and always in address space, which is why
/// the index refuses to open over a stable-row-id dataset: there the stored ids
/// are logical, and a list built here would filter live rows and keep dead ones.
///
/// Only the covered fragments are read. The rest cannot contribute a vertex, so
/// their deletions are somebody else's problem and their deletion files are a
/// per-fragment read this query would pay for nothing.
async fn deleted_row_addresses(
    dataset: &Dataset,
    covered: &RoaringBitmap,
) -> Result<RoaringTreemap> {
    let mut deleted = RoaringTreemap::new();
    for fragment in dataset.get_fragments() {
        let fragment_id = fragment.id() as u32;
        if !covered.contains(fragment_id) {
            continue;
        }
        let Some(deletion_vector) = fragment.get_deletion_vector().await? else {
            continue;
        };
        for row_offset in deletion_vector.iter() {
            deleted.insert(RowAddress::new_from_parts(fragment_id, row_offset).into());
        }
    }
    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neighbors(pairs: &[(u64, f32)]) -> Vec<Neighbor> {
        pairs
            .iter()
            .map(|(row_id, distance)| Neighbor {
                row_id: *row_id,
                distance: *distance,
            })
            .collect()
    }

    fn pairs(neighbors: &[Neighbor]) -> Vec<(u64, f32)> {
        neighbors
            .iter()
            .map(|neighbor| (neighbor.row_id, neighbor.distance))
            .collect()
    }

    /// Two copies of a row at different distances are far apart once sorted by
    /// distance, so a dedup that rode along with that ordering would keep both.
    #[test]
    fn the_merge_keeps_the_nearest_copy_of_a_repeated_row() {
        let merged = merge(neighbors(&[(7, 5.0), (3, 1.0), (7, 0.5), (9, 2.0)]), 10);
        assert_eq!(pairs(&merged), vec![(7, 0.5), (3, 1.0), (9, 2.0)]);
    }

    /// `k` counts distinct rows, so the truncation has to come after the dedup
    /// and not before it.
    #[test]
    fn the_merge_fills_k_with_distinct_rows() {
        let merged = merge(
            neighbors(&[(1, 0.1), (1, 0.2), (1, 0.3), (2, 0.4), (3, 0.5)]),
            3,
        );
        assert_eq!(pairs(&merged), vec![(1, 0.1), (2, 0.4), (3, 0.5)]);
    }
}
