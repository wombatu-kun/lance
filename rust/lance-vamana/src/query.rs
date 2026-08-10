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
//! edited the index's coverage underneath it, when an overlay has replaced the
//! indexed values under it, when the manifest records a format version this
//! build does not read, or when the segments disagree about the vectors they
//! hold. Each refusal names what to do about it, which is always to rebuild.
//!
//! Committing an index also breaks Lance's own vector search on that column -
//! see the crate README, and the test that pins it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array};
use futures::stream::{self, StreamExt, TryStreamExt};
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use lance_index::vector::storage::VectorStore;
use lance_io::scheduler::ScanScheduler;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use lance_table::format::overlay::DataOverlayFile;
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
            // Saturating because `k` is the caller's number and this is a
            // constructor, not a place to panic on arithmetic.
            search_list_size: k.saturating_add(k / 2),
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
    ///
    /// Shared rather than owned because each partition's walk runs on the CPU
    /// pool, which takes `'static` work, and the filter has to be applied inside
    /// the walk's own result - before `take(k)`, so that `k` means k live rows.
    deleted: Arc<RoaringTreemap>,
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

/// What one partition's walk produced, and what it cost.
struct Walked {
    neighbors: Vec<Neighbor>,
    comparisons: u64,
}

/// One partition a query has decided to read, and all of what reading it needs.
///
/// Owned rather than borrowed out of the segment, because the probes outlive the
/// borrow: they are collected by `route`, which returns them, and then consumed
/// by a stream that reads them concurrently. Borrowing would tie every read to
/// the segment vector for as long as the stream lives and leave the shape of
/// `route` fighting the borrow checker for nothing - the clones are one small
/// string and two numbers per partition actually read.
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
/// in partitions however many a query probes. It is also the *only* such bound -
/// the scheduler's byte budget does not apply to these reads, for the reason
/// [`crate::io::scan_scheduler`] spells out. Four rather than one because a walk
/// cannot start until a read finishes and a store with any latency would then
/// sit idle through every walk; four rather than `nprobes` because that is not a
/// bound at all. What the number should be on a high-latency store is a
/// measurement nobody has taken, so it is deliberately on the small side.
///
/// Dropping a search future abandons these reads but does not cancel them: the
/// io tasks already in the scheduler's queue still run to completion and their
/// bytes are read and thrown away. A caller that times a query out and retries
/// pays for both attempts.
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

        let fragments = dataset.get_fragments();
        let live = fragments
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<RoaringBitmap>();
        // Overlays are rare, so this is empty on the common path and the check
        // below costs nothing. Collected once rather than per segment: an index
        // of forty segments would otherwise walk every fragment forty times.
        let overlaid = fragments
            .iter()
            .filter(|fragment| !fragment.metadata().overlays.is_empty())
            .map(|fragment| (fragment.id() as u32, fragment.metadata()))
            .collect::<Vec<_>>();
        let scheduler = scan_scheduler(&dataset.object_store(None).await?);

        // Everything a segment can be refused for without reading it, first:
        // the round trips below are the expensive part of opening an index, and
        // a refusal should not pay for them.
        let mut planned = Vec::with_capacity(indices.len());
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
            // A base id says the segment's files live under some other dataset's
            // root, which a shallow clone stamps onto every index it inherits.
            // Resolving one needs `Dataset::indice_files_dir` and
            // `object_store_for_index`, both `pub(crate)`, so the directory
            // computed below would be the wrong one - while `files` would still
            // report the right sizes, making the mismatch look like corruption
            // rather than a path this build cannot follow.
            if index.base_id.is_some() {
                return Err(Error::not_supported(format!(
                    "index '{index_name}' segment {} was inherited from another dataset and its \
                     files live under a base path this crate cannot resolve; rebuild the index in \
                     this dataset",
                    index.uuid
                )));
            }
            let dir = dataset.indices_dir().join(index.uuid.to_string());
            let file_sizes = index
                .files
                .iter()
                .flatten()
                .map(|file| (file.path.clone(), file.size_bytes))
                .collect::<HashMap<_, _>>();
            planned.push((index, dir, file_sizes));
        }

        // One round trip per segment, and they wait on each other rather than in
        // turn: an index of forty segments is the ordinary state of anything
        // appended to, and on a store with 30ms of latency reading them one at a
        // time is more than a second before the first query can start.
        let store = dataset.object_store(None).await?;
        let manifests = stream::iter(planned.iter().map(|(_, dir, file_sizes)| {
            read_segment(&scheduler, dir, file_sizes.get(INDEX_FILE_NAME).copied())
        }))
        .buffered(store.io_parallelism())
        .try_collect::<Vec<_>>()
        .await?;

        let mut segments = Vec::with_capacity(planned.len());
        for ((index, dir, file_sizes), manifest) in planned.into_iter().zip(manifests) {
            let declared = index
                .fragment_bitmap
                .as_ref()
                .expect("checked above, before any file was read");

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
            // The two checks above ask what the *manifest* says about this
            // segment's coverage. An overlay changes neither: `Operation::
            // DataOverlay` rewrites fragment metadata and leaves every index
            // entry alone, so the fragment ids, the bitmap and this segment's
            // own record of what it read all still agree - while the values at
            // those addresses have been replaced. Ranking would run on the
            // pre-overlay vectors and `take_rows` would return the post-overlay
            // ones, with nothing in the answer to show for it.
            if let Some((fragment_id, _)) = overlaid.iter().find(|(fragment_id, fragment)| {
                declared.contains(*fragment_id)
                    && overlay_supersedes_segment(
                        &fragment.overlays,
                        &index.fields,
                        index.dataset_version,
                        dataset.schema(),
                    )
            }) {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built at dataset version {} and fragment \
                     {fragment_id} has since had its indexed values replaced by an overlay, so the \
                     vectors it ranks are not the ones the rows now hold; rebuild the index",
                    index.uuid, index.dataset_version
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
        let deleted =
            Arc::new(deleted_row_addresses(dataset, &covered, store.io_parallelism()).await?);

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
        // Nothing downstream would report this. Every distance against a
        // non-finite query is NaN, every ordering here goes through `total_cmp`,
        // and a negative NaN sorts *ahead* of negative infinity - so the walk
        // returns `k` arbitrary rows with a NaN distance and a caller comparing
        // that distance against a threshold accepts all of them.
        if let Some(position) = query.iter().position(|value| !value.is_finite()) {
            return Err(Error::invalid_input(format!(
                "query holds {} at position {position}, which no distance can be measured from",
                query[position]
            )));
        }
        // Cosine reaches the same place by a different road: the query is
        // normalised before it is routed, and dividing by a zero norm produces
        // the NaN directly. Underflow counts - a query of values around 1e-30 has
        // finite components and a norm of exactly zero in f32.
        if self.metadata.distance_type == DistanceType::Cosine {
            let norm_squared = query.iter().map(|value| value * value).sum::<f32>();
            if norm_squared == 0.0 {
                return Err(Error::invalid_input(
                    "query has zero length, which cosine distance is not defined for".to_string(),
                ));
            }
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
        // Sized from what the partitions can actually yield rather than from
        // `probes.len() * k`: `k` is the caller's, and the product overflows or
        // asks the allocator for a terabyte long before the walk would notice.
        let capacity = probes
            .iter()
            .map(|probe| (probe.entry.num_rows as usize).min(params.k))
            .sum::<usize>();
        let mut found = Vec::with_capacity(capacity);
        let mut partitions_read = 0usize;
        // Unordered, because the merge sorts everything anyway: ordering would
        // only make a finished partition wait for a slower one that was started
        // earlier, and `buffered` holds those finished results in memory while
        // they wait.
        let mut reads = std::pin::pin!(
            stream::iter(probes)
                .map(|probe| self.read_probe(probe))
                .buffer_unordered(PARTITIONS_IN_FLIGHT)
        );

        while let Some((partition, medoid)) = reads.try_next().await? {
            partitions_read += 1;
            let walked = self
                .walk_partition(partition, medoid, query.clone(), params)
                .await?;
            found.extend(walked.neighbors);
            comparisons = comparisons.saturating_add(walked.comparisons);
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
            // all - so a budget of `nprobes` centroids would silently return
            // fewer partitions than asked for.
            //
            // The distances are free, since `find_partitions` measures the query
            // against every centroid whichever bound it is given. The ordering is
            // not: asking for all of them turns a bounded selection into a full
            // `O(P log P)` sort plus a `take` that builds a `P`-element array
            // this driver discards. At the partition counts a graph index wants -
            // hundreds, not tens of thousands - that is far below the cost of one
            // partition read, and buying it back would mean tracking which
            // centroids are empty separately from the segment table.
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

    /// Walk one partition, on the CPU pool rather than on this runtime.
    ///
    /// A walk is milliseconds of uninterrupted arithmetic - at `L = 100`,
    /// `R = 64` and 768 dimensions it is on the order of ten million flops - with
    /// no await inside it to yield at. Left here it would run on the same
    /// runtime as the scheduler's io loop and every decode task, so on a
    /// single-threaded runtime, which is what an ordinary `#[tokio::test]`
    /// gives, the reads this method is supposed to overlap with would not
    /// advance at all and `PARTITIONS_IN_FLIGHT` would buy nothing.
    ///
    /// Everything the closure needs is moved into it because the pool takes
    /// `'static` work: the partition is owned already, the query is an `Arc`
    /// clone and the delete list is shared. Nothing in it waits on anything,
    /// which is what the pool requires.
    async fn walk_partition(
        &self,
        partition: Partition,
        medoid: u32,
        query: ArrayRef,
        params: &SearchParams,
    ) -> Result<Walked> {
        let distance_type = self.metadata.distance_type;
        let deleted = self.deleted.clone();
        let search_list_size = params.search_list_size;
        let k = params.k;
        spawn_cpu(move || {
            let walked = Comparisons::default();
            let vectors = flat_storage(
                partition.graph().row_ids(),
                partition.vectors(),
                distance_type,
            )?;
            let calculator = vectors.dist_calculator(query, 0.0);
            let mut scratch = SearchScratch::new(partition.len());
            let walk = greedy_search(
                partition.graph(),
                &calculator,
                medoid,
                search_list_size,
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
            let neighbors = walk
                .candidates
                .iter()
                .map(|node| Neighbor {
                    row_id: partition.graph().row_ids()[node.id as usize],
                    distance: node.dist.0,
                })
                .filter(|neighbor| !deleted.contains(neighbor.row_id))
                .take(k)
                .collect();
            Ok(Walked {
                neighbors,
                comparisons: walked.get(),
            })
        })
        .await
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

/// Whether an overlay has replaced indexed values under a segment built at
/// `dataset_version`.
///
/// Lance answers the same question for its own indices and answers it more
/// finely: `Scanner::overlay_stale_vector_rows` excludes the affected *rows* and
/// re-evaluates them on the flat path, so the index stays usable. That machinery
/// is `pub(crate)` and reaches into the scan plan, which this driver bypasses
/// entirely, so the question here is the coarse one - is any covered row stale -
/// and the answer is a refusal. The remedy is the same either way: rebuild.
///
/// Both halves of Lance's test are kept. The version gate: an overlay committed
/// at or before the segment's dataset version is already in the vectors it
/// holds. The field test in both directions: an overlay of a parent struct
/// replaces the leaf an index reads, and an overlay of a leaf replaces part of a
/// parent an index was built over.
fn overlay_supersedes_segment(
    overlays: &[DataOverlayFile],
    indexed_fields: &[i32],
    dataset_version: u64,
    schema: &Schema,
) -> bool {
    overlays
        .iter()
        .filter(|overlay| overlay.committed_version > dataset_version)
        .any(|overlay| {
            overlay.data_file.fields.iter().any(|overlaid| {
                indexed_fields.iter().any(|indexed| {
                    indexed == overlaid
                        || descends_from(schema, *overlaid, *indexed)
                        || descends_from(schema, *indexed, *overlaid)
                })
            })
        })
}

/// Whether `field` is `ancestor` itself or sits beneath it in `schema`.
fn descends_from(schema: &Schema, field: i32, ancestor: i32) -> bool {
    schema
        .field_ancestry_by_id(field)
        .is_some_and(|ancestry| ancestry.iter().any(|step| step.id == ancestor))
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
    io_parallelism: usize,
) -> Result<RoaringTreemap> {
    // One read per covered fragment, in flight against each other: five hundred
    // covered fragments read in turn is the difference between opening an index
    // in a second and opening it in fifteen.
    let vectors = stream::iter(
        dataset
            .get_fragments()
            .into_iter()
            .filter(|fragment| covered.contains(fragment.id() as u32))
            .map(|fragment| async move {
                let fragment_id = fragment.id() as u32;
                Ok::<_, Error>((fragment_id, fragment.get_deletion_vector().await?))
            }),
    )
    .buffered(io_parallelism)
    .try_collect::<Vec<_>>()
    .await?;

    let mut deleted = RoaringTreemap::new();
    for (fragment_id, deletion_vector) in vectors {
        let Some(deletion_vector) = deletion_vector else {
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

    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_file::version::ConcreteFileVersion;
    use lance_table::format::DataFile;
    use lance_table::format::overlay::OverlayCoverage;

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

    /// `k` is the caller's number and the constructor derives a beam from it, so
    /// the arithmetic has to hold at the top of the range rather than panic
    /// before the query is even described.
    #[test]
    fn an_enormous_k_does_not_overflow_the_beam() {
        let params = SearchParams::new(usize::MAX);
        assert_eq!(params.search_list_size, usize::MAX);
        assert!(params.search_list_size >= params.k);
    }

    /// A struct column with a vector leaf, so field ids exist on both sides of a
    /// parent/child relationship and the ancestry tests have something to walk.
    /// Ids are assigned depth first: `id` 0, `emb` 1, `emb.vec` 2, `emb.vec.item` 3.
    fn nested_schema() -> Schema {
        let vector = Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4),
            false,
        );
        let arrow = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("emb", DataType::Struct(vec![vector].into()), false),
        ]);
        Schema::try_from(&arrow).unwrap()
    }

    fn overlay(fields: Vec<i32>, committed_version: u64) -> DataOverlayFile {
        let mut data_file = DataFile::new_unstarted("overlay.lance", ConcreteFileVersion::V2_1);
        data_file.fields = fields.into();
        DataOverlayFile {
            data_file,
            coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter([0u32]))),
            committed_version,
        }
    }

    /// The version gate: an overlay committed at or before the segment's dataset
    /// version is already baked into the vectors the segment stores. Without the
    /// gate every index built over a previously overlaid column would refuse to
    /// open.
    #[test]
    fn an_overlay_the_build_already_saw_is_not_stale() {
        let schema = nested_schema();
        assert!(!overlay_supersedes_segment(
            &[overlay(vec![2], 7)],
            &[2],
            7,
            &schema
        ));
        assert!(overlay_supersedes_segment(
            &[overlay(vec![2], 8)],
            &[2],
            7,
            &schema
        ));
    }

    /// The field test, in both directions and with a negative arm: an overlay of
    /// the parent struct replaces the leaf this index reads, an overlay of the
    /// leaf replaces part of a parent it was built over, and an overlay of an
    /// unrelated column replaces nothing this index ranks by.
    #[test]
    fn only_an_overlay_of_the_indexed_field_is_stale() {
        let schema = nested_schema();
        let version = 1;
        for (overlaid, indexed, expected, what) in [
            (vec![2], 2, true, "the indexed leaf itself"),
            (vec![1], 2, true, "the parent of the indexed leaf"),
            (vec![2], 1, true, "a leaf under the indexed parent"),
            (vec![0], 2, false, "an unrelated column"),
            (vec![0, 1], 2, true, "an unrelated column and the parent"),
        ] {
            assert_eq!(
                overlay_supersedes_segment(
                    &[overlay(overlaid, version + 1)],
                    &[indexed],
                    version,
                    &schema
                ),
                expected,
                "{what}"
            );
        }
    }
}
