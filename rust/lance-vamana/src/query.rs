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
//! - **Rows added after the build are invisible** until they are indexed. The
//!   index answers from the fragments it was built over; Lance's scanner would
//!   scan the remainder. [`crate::inserter::insert_as_segment`] is the remedy.
//! - **A fragment the dataset has dropped is answered for by nobody.** A delete
//!   that empties a fragment, and a compaction that rewrites one, both take it
//!   out of the dataset, and the vertices stored for it are then unreachable
//!   rather than wrong. The index narrows itself to what is left and says so
//!   through [`VamanaIndex::covered_fragments`]. After a compaction the rows
//!   are still there, at new addresses in fragments this index does not cover -
//!   which is the same situation as rows appended after the build, and has the
//!   same remedy.
//! - **No predicate prefilter and no refine step.** Both live in the scanner.
//! - **Nothing is cached between queries unless the index is given a cache.**
//!   A query keeps a few reads going at once, so its working set is a few
//!   partitions rather than every partition it probes - and by default every
//!   query pays for its own partitions again. [`VamanaIndex::with_cache`] is
//!   what changes that, and what it keeps is the part of a partition that does
//!   not depend on the query: the layout of its file, and for a
//!   [`WalkMode::Lazy`] walk the codes and row ids it steers by, which are nine
//!   tenths of what such a query reads.
//! - **A partition is read whole unless the walk is told not to.**
//!   [`WalkMode::Lazy`] keeps the row ids and the codes and fetches the rest as
//!   it turns out to need it. Which of the two is right is a property of the
//!   deployment rather than of the index, and it was measured rather than
//!   assumed.
//!
//!   Reading only what a walk touches does not pay on its own
//!   (`examples/memory_gate.rs`): a walk expands a few dozen vertices in a
//!   partition and measures a distance against twenty-five to forty times as
//!   many, because each expanded vertex hands it `R` neighbours to score.
//!   Fetching exactly that set halves the pages moved at best and costs *more*
//!   CPU than reading the partition whole at fine granularity, because thousands
//!   of scattered reads decode slower than a few large ones. It pays with
//!   quantised codes standing in for those vectors, which leaves only the
//!   adjacency of the expanded vertices to fetch. And it pays only while the
//!   cache holds a fraction of the index - replaying real probe sequences
//!   through an LRU that holds all of it serves 25 to 250 queries per load, far
//!   past the crossover where reading whole is cheaper.
//!
//!   What "quantised codes" has to mean is measured too
//!   (`examples/coded_walk.rs`). Walked by RaBitQ distances, the same graph
//!   reaches the same recall for two to thirteen per cent more comparisons -
//!   but from three bits a dimension, 68 bytes a vertex at `d = 128`, not from
//!   one. A one-bit code needs a beam one and a half to three and a half times
//!   wider, which multiplies the very reads it was there to save, and it degrades
//!   as partitions coarsen: its error stays where it is while the number of
//!   neighbours that error can reorder grows with the partition. The answer also
//!   has to be re-scored from the whole candidate list rather than from its
//!   nearest `K`, because a coded walk's own ordering tops out around 0.95 recall
//!   at any code width.
//!
//!   What a walk must *not* do is read a vertex's vector as it expands it, the
//!   way DiskANN gets one free from the page that carries its edges. Correcting a
//!   distance seats that vertex at the back of the search list, the back of the
//!   list is the bar the next candidate has to beat to be admitted, and so the
//!   walk expands more - eight per cent more at three bits, three times more at
//!   one. At equal work a wider beam on plain codes reaches higher recall.
//!
//!   See [`crate::codes`] for the column, and `examples/lazy_walk.rs` for what
//!   the three modes cost against each other at equal recall.
//!
//! [`VamanaIndex::open`] refuses outright, rather than answering from what is
//! left, when the dataset has edited a segment's coverage while the fragments
//! themselves are still there, when it credits a segment with a fragment that
//! segment never read, when an overlay has replaced the indexed values under
//! one, when the manifest records a format version this build does not read,
//! when a segment was inherited from another dataset, or when the segments
//! disagree about the vectors they hold or about the codes they were built with.
//! Each refusal names what to do about it, which is always to rebuild.
//!
//! Committing an index also breaks Lance's own vector search on that column -
//! see the crate README, and the test that pins it.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array};
use futures::stream::{self, StreamExt, TryStreamExt};
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::cache::{CacheStats, LanceCache};
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_io::scheduler::{ScanScheduler, ScanStats};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use lance_table::format::overlay::DataOverlayFile;
use object_store::path::Path;
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

use crate::builder::{live_fragments, routing_distance_type, supported_distance_type};
use crate::cache;
use crate::codes::{self, CODE_COLUMN, centroid_distance};
use crate::format::{
    FORMAT_VERSION, INDEX_FILE_NAME, IndexMetadata, NEIGHBORS_COLUMN, ROW_ID_COLUMN, RowIdMode,
    VECTOR_COLUMN,
};
use crate::io::{
    PartitionFile, check_partition_shape, read_partition_batch, read_segment, scan_scheduler,
};
use crate::lazy::LazyWalk;
use crate::partition::Partition;
use crate::search::{Comparisons, SearchScratch, flat_storage, greedy_search};
use crate::segment::{PartitionEntry, SegmentManifest};

/// One answer: where the row is, and how far it was from the query.
///
/// A row *address* - fragment id in the high 32 bits, offset within it in the
/// low - because [`RowIdMode::Address`] is the only mode this crate builds and
/// the only one it opens. A stable row id is a different number for the same
/// row, and the two are one `u64` as far as a compiler is concerned, so this
/// name is the only thing standing between a caller and an API that wants the
/// other one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub row_addr: u64,
    pub distance: f32,
}

/// What a walk measures its distances against, and what it reads to do it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WalkMode {
    /// Read the partition whole and measure against the vectors it stores.
    #[default]
    Exact,
    /// Read the partition whole and measure against its codes, with the
    /// candidate list re-scored exactly before it is answered from.
    ///
    /// Only for an index built with [`crate::IndexParams::with_code_bits`], and
    /// refused rather than quietly downgraded for one that was not. On its own
    /// it costs a few per cent more comparisons and reads no fewer bytes: it is
    /// [`Self::Lazy`] with the reading left alone, which is the useful arm to
    /// hold a walk against when what is in question is the *steering*.
    Coded,
    /// Read the row ids and the codes, and nothing else until the walk asks for
    /// it: the out-edges of a vertex when it expands one, the vectors of the
    /// candidate list when there is one to re-score.
    ///
    /// What the codes were built for. On SIFT1M at 65536 rows a partition and
    /// equal recall it reads 18.2 MB a query against 198.6 MB
    /// (`examples/lazy_walk.rs`), and spends less CPU doing it - decoding two
    /// hundred megabytes costs more than fetching eighteen even when every byte
    /// is already in the page cache. What it pays is round trips: twenty
    /// requests become fifty-four at the default [`SearchParams::beam_width`].
    ///
    /// Half of that is here and the other half is [`VamanaIndex::with_cache`],
    /// because nine tenths of the 18.2 MB is the codes, which do not depend on
    /// the query and are re-read by every one of them. Given somewhere to keep
    /// them, the same query reads **71.9 kB** and takes 3.2 ms against 131.0 -
    /// the mode's real number, and the reason it is worth having whenever the
    /// index does not fit in the memory available to it.
    ///
    /// Requires codes, same as [`Self::Coded`].
    Lazy,
}

impl WalkMode {
    /// Whether this walk can only run on an index that carries codes.
    fn needs_codes(self) -> bool {
        matches!(self, Self::Coded | Self::Lazy)
    }
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
    /// What the walk measures its distances against.
    pub mode: WalkMode,
    /// `W`: how many vertices one hop of a [`WalkMode::Lazy`] walk expands, and
    /// therefore how many rows of `__neighbors` it asks for in one request.
    ///
    /// Ignored by the walks that read a partition whole, which have every edge
    /// already. For the lazy one it is the trade the mode exists to make: the
    /// chain of dependent round trips divides by it, while a wider hop expands
    /// vertices the strictly greedy order would have skipped. Four is the width
    /// the phase gate modelled and is deliberately on the low side - what it
    /// should be on a high-latency store is a measurement nobody has taken.
    pub beam_width: usize,
}

impl SearchParams {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            nprobes: 1,
            // Saturating because `k` is the caller's number and this is a
            // constructor, not a place to panic on arithmetic.
            search_list_size: k.saturating_add(k / 2),
            mode: WalkMode::default(),
            beam_width: 4,
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

    pub fn with_mode(mut self, mode: WalkMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
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
    /// What a query keeps of the partitions it probes, for the queries after it.
    ///
    /// `None` and not [`LanceCache::no_cache`], which would have let one code
    /// path serve both and does not mean what it says: a cache of capacity zero
    /// still admits an entry and reclaims it when it next runs its housekeeping,
    /// so a partition read a moment ago is served from a cache that is supposed
    /// to be holding nothing. An index nobody asked to cache has to read every
    /// time, not almost every time. See [`Self::with_cache`].
    cache: Option<LanceCache>,
    metadata: IndexMetadata,
    segments: Vec<Segment>,
    /// Fragments this index still answers for: what its segments were built
    /// over, minus what the dataset has since dropped.
    covered: RoaringBitmap,
    /// Which stored vertices must not reach an answer, as of
    /// [`VamanaIndex::open`].
    ///
    /// Shared rather than owned because each partition's walk runs on the CPU
    /// pool, which takes `'static` work, and the filter has to be applied inside
    /// the walk's own result - before `take(k)`, so that `k` means k live rows.
    rows: Arc<RowFilter>,
}

/// The stored vertices a walk must not return.
///
/// A snapshot, not a live view: the graph files hold vertices for rows that have
/// since gone away, and nothing rewrites them, so the only way to tell a live
/// vertex from a dead one is to ask the dataset - once, at open, rather than on
/// every query. A row deleted afterwards keeps coming back until the index is
/// reopened.
#[derive(Debug)]
pub(crate) struct RowFilter {
    /// Rows deleted from a fragment this index still covers.
    deleted: RoaringTreemap,
    /// Fragments the dataset no longer has. Every vertex stored for one of them
    /// is unreachable, and a whole dead fragment is a bitmap entry rather than
    /// 2^32 addresses in `deleted`: the `roaring` crate has no run containers,
    /// so a full fragment's worth of addresses would be half a gigabyte.
    missing_fragments: RoaringBitmap,
}

impl RowFilter {
    pub(crate) fn rejects(&self, row_addr: u64) -> bool {
        self.missing_fragments
            .contains(RowAddress::from(row_addr).fragment_id())
            || self.deleted.contains(row_addr)
    }

    /// Whether this filter rejects nothing, so that every stored vertex is live.
    pub(crate) fn is_empty(&self) -> bool {
        self.deleted.is_empty() && self.missing_fragments.is_empty()
    }
}

#[derive(Debug)]
pub(crate) struct Segment {
    pub(crate) uuid: Uuid,
    pub(crate) dir: Path,
    pub(crate) manifest: SegmentManifest,
    /// Byte size of each file of this segment, as Lance recorded it at commit.
    ///
    /// Lance fills this by listing the directory, so it is a fact about the
    /// files rather than a second copy of one this crate wrote. Handing it to
    /// the reader is what turns opening a partition into one read rather than a
    /// size probe followed by a read.
    pub(crate) file_sizes: HashMap<String, u64>,
    /// Schema field ids the dataset credits this segment's index row with.
    pub(crate) fields: Vec<i32>,
    /// What this segment was built over that the dataset still has.
    ///
    /// Narrower than the segment's own `fragments` exactly when a fragment has
    /// gone; every vertex stored for one of those is already rejected by
    /// [`RowFilter`], so this is the coverage a rewrite of this segment would be
    /// committed with.
    pub(crate) coverage: RoaringBitmap,
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
    /// Which segment this partition belongs to, which is half of what names it:
    /// every segment of an index has its own partition 0.
    segment: Uuid,
    entry: PartitionEntry,
    /// What the segment declares, to be checked against what the file holds.
    max_degree: u32,
    dimension: u32,
    /// `|q - c|^2` against this partition's centroid, for a walk that runs on
    /// codes; `None` for one that runs on the stored vectors.
    ///
    /// RaBitQ's raw-query estimator wants exactly this beside the *raw* query,
    /// because the centroid is already folded into each vertex's own factors.
    /// Handing it the residual instead produces distances that are wrong rather
    /// than approximate, which a recall number reports as bad codes.
    dist_q_c: Option<f32>,
}

/// One partition read off disk, and what a walk over it needs.
struct Probed {
    partition: Partition,
    medoid: u32,
    /// The code column and `|q - c|^2`, for a walk that runs on codes.
    ///
    /// The column rather than the batch it was read out of: the batch also holds
    /// the row ids, the edges and the vectors, all of which `partition` has
    /// already taken its own copy of, and it would stay alive for the length of
    /// the walk.
    coded: Option<(FixedSizeListArray, f32)>,
}

/// How many partitions a query holds at once.
///
/// The bound is on memory: this many partitions' worth of resident data however
/// many a query probes, which for the walks that read whole is this many whole
/// partitions and for [`WalkMode::Lazy`] is this many partitions' row ids and
/// codes - a tenth of that at `d = 128`. It bounds what a query holds *of its
/// own*; an index given a cache holds that cache's budget beside it, and holds
/// it whether or not a query is running. The scheduler's byte budget bounds
/// neither, for the reason [`crate::io::scan_scheduler`] spells out. Four rather
/// than one because a walk
/// cannot start until a read finishes and a store with any latency would then
/// sit idle through every walk; four rather than `nprobes` because that is not a
/// bound at all. What the number should be on a high-latency store is a
/// measurement nobody has taken, so it is deliberately on the small side.
///
/// Per search call, and there is nothing above it: a server answering `n`
/// queries at once holds up to `n` times this many partitions, so an index whose
/// partitions are large enough to matter has to be bounded by its caller.
///
/// Dropping a search future abandons these reads but does not cancel them: the
/// io tasks already in the scheduler's queue still run to completion and their
/// bytes are read and thrown away. A caller that times a query out and retries
/// pays for both attempts.
const PARTITIONS_IN_FLIGHT: usize = 4;

impl VamanaIndex {
    /// Open every segment of `index_name`.
    ///
    /// The index reads through one scheduler for its whole life, and a scheduler
    /// is an io loop spawned on whichever runtime this call is awaited in - for
    /// every store but `file+uring`, which is served without one. The index is
    /// therefore bound to that runtime: opened inside a `Runtime` that is later
    /// dropped, its reads are queued to a loop that no longer runs and nothing
    /// ever pops them, so a search hangs rather than failing.
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

            // The `None` arm is for manifests older than the field itself:
            // `IndexSegment` carries a plain bitmap, so nothing this crate can
            // commit reaches it and no test can produce one. What the coverage
            // has to agree with is checked below, once the segment's own record
            // of it has been read.
            let Some(declared) = index.fragment_bitmap.as_ref() else {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} records no fragment coverage",
                    index.uuid
                )));
            };
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
            planned.push((index, dir, file_sizes, declared));
        }

        // One round trip per segment, and they wait on each other rather than in
        // turn: an index of forty segments is the ordinary state of anything
        // appended to, and on a store with 30ms of latency reading them one at a
        // time is more than a second before the first query can start.
        let store = dataset.object_store(None).await?;
        let manifests = stream::iter(planned.iter().map(|(_, dir, file_sizes, _)| {
            read_segment(&scheduler, dir, file_sizes.get(INDEX_FILE_NAME).copied())
        }))
        .buffered(store.io_parallelism())
        .try_collect::<Vec<_>>()
        .await?;

        let mut segments = Vec::with_capacity(planned.len());
        let mut covered = RoaringBitmap::new();
        let mut missing_fragments = RoaringBitmap::new();
        for ((index, dir, file_sizes, declared), manifest) in planned.into_iter().zip(manifests) {
            // Three records of one thing, and every disagreement between them
            // means something different. `built_over` is what the segment wrote
            // about itself and never changes; `declared` is what the dataset
            // credits it with, which Lance edits in place and which never touches
            // the segment's own files; `live` is which fragments the dataset
            // still has at all.
            let built_over = manifest
                .metadata()
                .fragments
                .iter()
                .copied()
                .collect::<RoaringBitmap>();

            // Credited with a fragment it never read. Lance widens a bitmap in
            // `register_pure_rewrite_rows_update_frags_in_indices` and in the
            // pruning path of a deferred commit; the first is gated on stable row
            // ids, which the builder refuses outright, so today only the second
            // can produce it - but a bitmap naming a fragment this segment never
            // read is unanswerable either way, and which upstream path widened it
            // is not something a reader can tell.
            if !(declared - &built_over).is_empty() {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built over {} fragments but the dataset \
                     credits it with {}, so it is expected to answer for rows it never read; \
                     rebuild the index",
                    index.uuid,
                    built_over.len(),
                    declared.len()
                )));
            }
            // Built over a fragment that is still here, but no longer credited
            // with it. That is Lance saying the data under those addresses was
            // rewritten: an in-place column update, or the coverage pruning a
            // deferred commit runs. The fragment ids and every row address
            // survive it, so nothing downstream would notice - the vectors this
            // segment ranks by are simply not the ones the rows now hold.
            let rewritten = (&built_over - declared) & &live;
            if !rewritten.is_empty() {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built over {} fragments the dataset still \
                     has but no longer credits it with, so something rewrote data under it and the \
                     vectors it holds no longer match the rows at those addresses; rebuild the index",
                    index.uuid,
                    rewritten.len()
                )));
            }
            // Built over a fragment the dataset no longer has at all. Its rows
            // are unreachable rather than wrong: fragment ids are a monotonic
            // high water mark in the manifest (`Manifest::update_max_fragment_id`
            // keeps it across deletions, and `max_fragment_id` is documented as
            // not supporting reuse), so no address stored here can ever resolve
            // to some other dataset row. That makes narrowing the coverage the
            // honest answer rather than a refusal, and it is the same answer
            // Lance gives itself: `IndexMetadata::effective_fragment_bitmap` is
            // `declared & existing`, and the rewrite path of a stable-row-id
            // commit drops rewritten fragments from an address-domain index's
            // coverage and leaves the scanner to cover them.
            //
            // Which of the two got us here - a delete that emptied the fragment,
            // or a compaction that moved its rows elsewhere - is not something a
            // reader can tell, and it does not change what this index can do. It
            // changes what the *caller* should do, so the narrowing is logged and
            // `covered_fragments` reports the result.
            let gone = &built_over - &live;
            if !gone.is_empty() {
                log::warn!(
                    "Vamana index '{index_name}' segment {} was built over {} fragments the \
                     dataset no longer has; it will answer for the remaining {}, and the rows of \
                     the rest are the caller's to scan",
                    index.uuid,
                    gone.len(),
                    built_over.len() - gone.len()
                );
                missing_fragments |= gone;
            }
            let coverage = &built_over & &live;
            covered |= &coverage;

            // The checks above ask what the *manifest* says about this
            // segment's coverage. An overlay changes none of it: `Operation::
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
                uuid: index.uuid,
                dir,
                manifest,
                file_sizes,
                fields: index.fields.clone(),
                coverage,
            });
        }

        let metadata = segments[0].manifest.metadata().clone();
        for segment in &segments[1..] {
            let other = segment.manifest.metadata();
            // Degree and pruning slack may legitimately differ between a base
            // segment and one appended later; the identifier space, the metric,
            // the width and the codes may not, because a query mixes their
            // answers - and one segment coded where another is not would make
            // the walk mode mean two different things in one query.
            if (
                other.dimension,
                other.distance_type,
                other.row_id_mode,
                &other.codes,
            ) != (
                metadata.dimension,
                metadata.distance_type,
                metadata.row_id_mode,
                &metadata.codes,
            ) {
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

        let deleted = deleted_row_addresses(dataset, &covered, store.io_parallelism()).await?;

        Ok(Self {
            scheduler,
            cache: None,
            metadata,
            segments,
            covered,
            rows: Arc::new(RowFilter {
                deleted,
                missing_fragments,
            }),
        })
    }

    /// Keep what a query reads about a partition, for the queries after it.
    ///
    /// Without one every query re-reads the codes of every partition it probes,
    /// which for [`WalkMode::Lazy`] is nine tenths of what it reads at all: on
    /// SIFT1M at 65536 rows a partition it is 17.5 MB of the 18.2 MB
    /// (`examples/lazy_walk.rs`). What the walk fetches for itself - the edges
    /// of the vertices it expands, the vectors of the candidates it ends with -
    /// is the remainder, and is not cached, because which rows those are is a
    /// property of the query rather than of the partition.
    ///
    /// The cache arrives from the caller rather than being sized here, because
    /// its budget is a deployment's to spend: several indices can share one, and
    /// a [`lance_core::cache::CacheBackend`] can put it somewhere other than
    /// memory. What it costs is a property of the data - at three bits and
    /// `d = 128` a vertex is 68 bytes on disk and about 116 held, so a million
    /// rows is 110 MiB - and an entry too large for the budget is simply never
    /// kept, which costs a re-read rather than an error.
    ///
    /// Nothing here has to be invalidated. Every entry describes one file of one
    /// segment, and a segment is written once: deleting rows edits no index file
    /// at all, and adding rows or consolidating writes a *new* segment under a
    /// new uuid, so what an old entry describes is either still exactly true or
    /// no longer named by anything. Which of the two it is decides only when the
    /// budget reclaims it.
    pub fn with_cache(mut self, cache: LanceCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// What the cache has served and what it holds, or `None` for an index that
    /// was never given one.
    ///
    /// Counts both kinds of entry a query looks up - a partition's codes and a
    /// partition file's layout - so a hit ratio here is per lookup rather than
    /// per query.
    pub async fn cache_stats(&self) -> Option<CacheStats> {
        let cache = self.cache.as_ref()?;
        Some(cache.stats().await)
    }

    /// Open one probed partition's file, through the cache if there is one.
    async fn partition_file(&self, probe: &Probe) -> Result<PartitionFile> {
        match &self.cache {
            Some(cache) => {
                PartitionFile::open_cached(&self.scheduler, &probe.path, probe.size_bytes, cache)
                    .await
            }
            None => PartitionFile::open(&self.scheduler, &probe.path, probe.size_bytes).await,
        }
    }

    /// What this index answers for: every fragment its segments were built over
    /// that the dataset still has.
    ///
    /// The number a caller needs to scan the remainder. It is not the same as
    /// the coverage the segments were built with - a fragment the dataset has
    /// since dropped is answered for by nobody - and it is not
    /// `metadata().fragments` either, which is one segment's record.
    pub fn covered_fragments(&self) -> &RoaringBitmap {
        &self.covered
    }

    /// The first segment's metadata.
    ///
    /// Everything a query mixes - the width, the metric, the identifier space -
    /// is checked to agree across the segments on the way in, so reading it off
    /// the first one is reading it off all of them. Its `fragments` field is the
    /// exception: coverage is per segment and the segments of an index are
    /// disjoint, so that field is a *part* of what the index holds. Use
    /// [`Self::covered_fragments`] for the whole of it.
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Every byte this index has read since it was opened.
    ///
    /// Taken off the index's own scheduler rather than off a tracker wrapped
    /// around the store, which under-counts a local read.
    pub fn io_stats(&self) -> ScanStats {
        self.scheduler.stats()
    }

    /// What opening the index established about its segments, for the one
    /// caller in this crate that rewrites them.
    ///
    /// Consolidation needs exactly what a query needs and one thing more - the
    /// index row's field ids, to commit a replacement under - and it needs the
    /// same refusals to have run first. Reproducing [`Self::open`] instead would
    /// be a second copy of nine checks and a delete list.
    pub(crate) fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub(crate) fn row_filter(&self) -> &RowFilter {
        &self.rows
    }

    pub(crate) fn scheduler(&self) -> &Arc<ScanScheduler> {
        &self.scheduler
    }

    /// The segment another one should be modelled on: the one covering the most
    /// fragments, and on a tie the one the manifest lists first.
    ///
    /// The base rather than a delta, which is what "most fragments" means in
    /// practice, so that what maintenance writes inherits the routing of the
    /// index's largest graph instead of inheriting a delta's. Deterministic on
    /// purpose: whose centroids a segment was written under is not recoverable
    /// from the segment afterwards.
    pub(crate) fn base_segment(&self) -> Result<&Segment> {
        self.segments
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

    /// Fragments `dataset` has that this index does not answer for.
    ///
    /// Against [`Self::covered_fragments`] rather than against any segment's own
    /// record, because the two differ exactly when a fragment has gone: one a
    /// segment was built over and the dataset has since dropped is answered for
    /// by nobody, and if a compaction rewrote its rows into a new fragment then
    /// that new fragment belongs in this list.
    pub(crate) fn unindexed_fragments(&self, dataset: &Dataset) -> Vec<u32> {
        live_fragments(dataset)
            .into_iter()
            .filter(|fragment| !self.covered.contains(*fragment))
            .collect()
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
        if params.beam_width == 0 {
            return Err(Error::invalid_input(
                "beam_width must be greater than zero".to_string(),
            ));
        }
        // Refused rather than answered exactly. A caller asking for a coded
        // walk is asking about cost, and quietly giving them a walk that reads
        // every vector would be an answer to a different question.
        if params.mode.needs_codes() && self.metadata.codes.is_none() {
            return Err(Error::invalid_input(
                "this Vamana index was built without codes, so it cannot be walked by them; \
                 rebuild it with IndexParams::with_code_bits"
                    .to_string(),
            ));
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
        // normalised before it is routed, and a norm that is not a positive
        // finite number turns the whole vector into NaNs or zeroes there.
        //
        // Both ends of the range do it, and the guard above catches neither
        // because it looks at the components rather than at what they add up to.
        // Underflow: a query of values around 1e-30 has finite components and a
        // norm of exactly zero in f32, and dividing by it gives NaN. Overflow: a
        // query of 1e20 is finite componentwise while the sum of squares is
        // `+inf`, so `normalize_arrow` divides by infinity and hands routing a
        // vector of *zeroes* - under cosine every vertex is then at distance
        // exactly 1.0, and the answer is `k` arbitrary rows with a plausible
        // distance attached and no error anywhere.
        if self.metadata.distance_type == DistanceType::Cosine {
            let norm_squared = query.iter().map(|value| value * value).sum::<f32>();
            if norm_squared == 0.0 || !norm_squared.is_finite() {
                return Err(Error::invalid_input(format!(
                    "query has a squared length of {norm_squared}, which cosine distance is not \
                     defined for"
                )));
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
        // Grown as the answers arrive rather than sized up front. Everything
        // available before the first read is a claim: `k` is the caller's, and
        // the only bound on a partition's row count is the one its own segment
        // table states, which nothing has yet been asked to honour - the file it
        // describes is checked against it in `read_partition`, afterwards. A
        // `k` of `usize::MAX` against a table claiming `MAX_PARTITION_ROWS` is a
        // sixty-gigabyte allocation off a number read out of a file.
        let mut found = Vec::new();
        let mut partitions_read = 0usize;
        // Unordered, because the merge sorts everything anyway: ordering would
        // only make a finished partition wait for a slower one that was started
        // earlier, and `buffered` holds those finished results in memory while
        // they wait.
        //
        // Where the concurrency sits differs by mode, and it has to. A walk over
        // a partition held in memory never waits, so the reads run ahead of it
        // and the walks themselves are pulled one at a time; a lazy walk waits
        // once a hop, so the whole walk is what goes in flight and one
        // partition's next hop overlaps another's arithmetic.
        // [`PARTITIONS_IN_FLIGHT`] bounds both, and means the same thing in
        // both: how many partitions' worth of resident data a query holds.
        let mut walks = match params.mode {
            WalkMode::Lazy => stream::iter(probes)
                .map({
                    let query = query.clone();
                    let routing_query = routing_query.clone();
                    move |probe| {
                        self.walk_lazily(probe, query.clone(), routing_query.clone(), params)
                    }
                })
                .buffer_unordered(PARTITIONS_IN_FLIGHT)
                .boxed(),
            _ => stream::iter(probes)
                .map(|probe| self.read_probe(probe))
                .buffer_unordered(PARTITIONS_IN_FLIGHT)
                .and_then({
                    let query = query.clone();
                    let routing_query = routing_query.clone();
                    move |probed| {
                        self.walk_partition(probed, query.clone(), routing_query.clone(), params)
                    }
                })
                .boxed(),
        };

        while let Some(walked) = walks.try_next().await? {
            partitions_read += 1;
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
                // Computed rather than taken from the ranking above, which is a
                // routing distance whose scale is `find_partitions`' business.
                // This one is a term of RaBitQ's estimator, so what it has to be
                // is unambiguous, and `dimension` flops a probed partition is
                // nothing beside reading one.
                let dist_q_c = params
                    .mode
                    .needs_codes()
                    .then(|| {
                        centroid_distance(segment.manifest.ivf(), *partition_id, routing_query)
                    })
                    .transpose()?;
                probes.push(Probe {
                    path: segment.dir.clone().join(entry.file.as_str()),
                    size_bytes: segment.file_sizes.get(&entry.file).copied(),
                    segment: segment.uuid,
                    entry: entry.clone(),
                    max_degree: declared.max_degree,
                    dimension: declared.dimension,
                    dist_q_c,
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
    ///
    /// The pool takes work, not futures, so dropping a search abandons the
    /// walk's result and not the walk: it runs to the end on a pool thread. Same
    /// bargain as the reads [`PARTITIONS_IN_FLIGHT`] describes, and the reason a
    /// query that is timed out and retried goes on spending CPU on the attempt
    /// its caller has already given up on.
    async fn walk_partition(
        &self,
        probed: Probed,
        query: ArrayRef,
        routing_query: ArrayRef,
        params: &SearchParams,
    ) -> Result<Walked> {
        let distance_type = self.metadata.distance_type;
        let dimension = self.metadata.dimension;
        let code_params = self.metadata.codes.clone();
        let rows = self.rows.clone();
        let search_list_size = params.search_list_size;
        let k = params.k;
        spawn_cpu(move || {
            let Probed {
                partition,
                medoid,
                coded,
            } = probed;
            let walked = Comparisons::default();
            let vectors = flat_storage(
                partition.graph().row_ids(),
                partition.vectors(),
                distance_type,
            )?;
            let exact = vectors.dist_calculator(query, 0.0);
            let mut scratch = SearchScratch::new(partition.len());

            // Either way the list this comes back as is sorted by an *exact*
            // distance, which is what the merge and the checks below rest on.
            let candidates = match coded {
                None => {
                    let walk = greedy_search(
                        partition.graph(),
                        &exact,
                        medoid,
                        search_list_size,
                        &mut scratch,
                        &walked,
                    )?;
                    walk.candidates
                        .into_iter()
                        .map(|node| (node.id, node.dist.0))
                        .collect::<Vec<_>>()
                }
                Some((column, dist_q_c)) => {
                    let code_params = code_params.ok_or_else(|| {
                        Error::internal(
                            "a Vamana coded walk was scheduled for a segment without codes"
                                .to_string(),
                        )
                    })?;
                    let store = codes::storage(
                        &code_params,
                        distance_type,
                        dimension,
                        partition.graph().row_ids(),
                        &column,
                    )?;
                    let walk = greedy_search(
                        partition.graph(),
                        &store.dist_calculator(routing_query, dist_q_c),
                        medoid,
                        search_list_size,
                        &mut scratch,
                        &walked,
                    )?;
                    // The whole list, not its nearest `k`: a coded walk's own
                    // ordering tops out around 0.95 recall at any code width, so
                    // the rows that make up the difference are the ones its
                    // ordering put behind `k`. Measured in `examples/coded_walk.rs`.
                    walked.record(walk.candidates.len() as u64);
                    let mut rescored = walk
                        .candidates
                        .into_iter()
                        .map(|node| (node.id, exact.distance(node.id)))
                        .collect::<Vec<_>>();
                    rescored.sort_by(|left, right| left.1.total_cmp(&right.1));
                    rescored
                }
            };

            Ok(Walked {
                neighbors: answer(candidates, partition.graph().row_ids(), &rows, k)?,
                comparisons: walked.get(),
            })
        })
        .await
    }

    /// Walk one partition without reading it, on this runtime rather than on the
    /// CPU pool.
    ///
    /// The opposite bargain from [`Self::walk_partition`], and forced rather than
    /// chosen: the pool takes work that never waits, and this waits once a hop.
    /// What it hands the pool instead is nothing at all - a hop is `beam_width`
    /// times `max_degree` coded distances, tens of microseconds, below the size
    /// at which the pool's own overhead starts to pay.
    ///
    /// The read of the row ids and the codes is the one thing here that is
    /// proportional to the partition. It is also what makes the walk possible at
    /// all, and it is a tenth of what reading the partition whole would be at
    /// `d = 128`.
    async fn walk_lazily(
        &self,
        probe: Probe,
        query: ArrayRef,
        routing_query: ArrayRef,
        params: &SearchParams,
    ) -> Result<Walked> {
        let Some(dist_q_c) = probe.dist_q_c else {
            return Err(Error::internal(
                "a Vamana lazy walk was scheduled for a segment without codes".to_string(),
            ));
        };
        let file = self.partition_file(&probe).await?;
        let resident = cache::resident(
            self.cache.as_ref(),
            probe.segment,
            &probe.entry,
            &file,
            &self.metadata,
        )
        .await?;

        let (candidates, comparisons) = LazyWalk {
            file: &file,
            codes: &resident.codes,
            row_ids: &resident.row_ids,
            medoid: probe.entry.medoid,
            max_degree: probe.max_degree,
            dimension: probe.dimension,
            distance_type: self.metadata.distance_type,
            search_list_size: params.search_list_size,
            beam_width: params.beam_width,
        }
        .run(routing_query, dist_q_c, query)
        .await?;

        Ok(Walked {
            neighbors: answer(candidates, &resident.row_ids, &self.rows, params.k)?,
            comparisons,
        })
    }

    /// Read one probed partition whole.
    ///
    /// Projected on the columns the walk will use, so that an index carrying
    /// codes does not pay for them on a query that measures against the vectors:
    /// thirteen per cent of a partition at `d = 128`.
    async fn read_probe(&self, probe: Probe) -> Result<Probed> {
        let mut columns = vec![ROW_ID_COLUMN, NEIGHBORS_COLUMN, VECTOR_COLUMN];
        if probe.dist_q_c.is_some() {
            columns.push(CODE_COLUMN);
        }
        // Through the cache for the file's layout, same as the lazy walk, and
        // for the same reason: the footer is a round trip whatever is read
        // afterwards. What it does *not* take from the cache is the codes, which
        // arrive in this read along with everything else.
        let file = self.partition_file(&probe).await?;
        let reader = file.project(&columns).await?;
        let batch = read_partition_batch(&reader, probe.entry.num_rows).await?;
        let partition = Partition::try_from_batch(&batch)?;
        check_partition_shape(&partition, &probe.entry, probe.max_degree, probe.dimension)?;
        let coded = probe
            .dist_q_c
            .map(|dist_q_c| Ok::<_, Error>((codes::column(&batch)?, dist_q_c)))
            .transpose()?;
        Ok(Probed {
            partition,
            medoid: probe.entry.medoid,
            coded,
        })
    }
}

/// Turn one walk's candidate list into that partition's share of the answer.
///
/// Shared by every mode, and the reason the three of them return the same shape:
/// what separates them is how a candidate list is arrived at, and nothing after
/// that may differ. `candidates` is nearest first by an *exact* distance
/// whichever walk produced it, which is what the merge downstream rests on.
fn answer(
    candidates: Vec<(u32, f32)>,
    row_ids: &[u64],
    rows: &RowFilter,
    k: usize,
) -> Result<Vec<Neighbor>> {
    // A stored vector that is not finite makes every distance measured against
    // it NaN, and a NaN goes wherever `total_cmp` puts it: a negative one sorts
    // ahead of every real answer, survives the merge and comes back as the
    // nearest neighbour, with a caller comparing it against a threshold
    // accepting it. The vectors column is not swept for this on the way in -
    // that is `rows * dimension` per partition on the hot path of every query,
    // more work than the walk it would be protecting - so it is caught here
    // instead, over the `search_list_size` candidates the walk actually kept.
    if let Some((id, distance)) = candidates.iter().find(|(_, d)| !d.is_finite()) {
        return Err(Error::corrupt_file_named(
            "partition",
            format!(
                "Vamana row {} is at distance {distance} from a finite query, so the vector \
                 stored for it is not finite",
                row_ids[*id as usize],
            ),
        ));
    }
    // Local ids are per partition, so they become row ids *before* the merge:
    // every partition has a vertex 0, and they are different rows.
    //
    // Dead vertices are dropped here and not earlier. They are still walked,
    // because they carry the out-edges that keep the graph connected - removing
    // them from the traversal would strand whatever they were the only route to.
    // Filtering before `take` rather than after is what makes `k` mean "k live
    // rows" instead of "k rows, some of which the caller will find missing".
    Ok(candidates
        .into_iter()
        .map(|(id, distance)| Neighbor {
            row_addr: row_ids[id as usize],
            distance,
        })
        .filter(|neighbor| !rows.rejects(neighbor.row_addr))
        .take(k)
        .collect())
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
/// dedup ride along with the ordering the caller sees: keyed on the address in a
/// pass of its own, it collapses two copies of a row whatever their distances,
/// where a dedup run after a distance sort would only collapse the copies that
/// agree to the last bit - and the ones that disagree are exactly the ones worth
/// not returning twice.
fn merge(mut found: Vec<Neighbor>, k: usize) -> Vec<Neighbor> {
    found.sort_by(|left, right| {
        left.row_addr
            .cmp(&right.row_addr)
            .then(left.distance.total_cmp(&right.distance))
    });
    found.dedup_by_key(|neighbor| neighbor.row_addr);
    found.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then(left.row_addr.cmp(&right.row_addr))
    });
    found.truncate(k);
    found
}

/// Row addresses deleted from the fragments an index covers.
///
/// Public because the spike tests measure this exact question - whether a delete
/// list can be built from outside Lance, at a cost proportional to the deletions
/// rather than to the dataset - and a private copy of it in a test is a copy
/// that drifts.
///
/// Deletion vectors are per fragment and always in address space, which is why
/// the index refuses to open over a stable-row-id dataset: there the stored ids
/// are logical, and a list built here would filter live rows and keep dead ones.
///
/// Only the covered fragments are read. The rest cannot contribute a vertex, so
/// their deletions are somebody else's problem and their deletion files are a
/// per-fragment read this query would pay for nothing.
pub async fn deleted_row_addresses(
    dataset: &Dataset,
    covered: &RoaringBitmap,
    io_parallelism: usize,
) -> Result<RoaringTreemap> {
    // One read per covered fragment, in flight against each other: five hundred
    // covered fragments read in turn is the difference between opening an index
    // in a second and opening it in fifteen.
    //
    // Folded as they arrive rather than collected first. A deletion vector is a
    // bitmap over a whole fragment, so collecting them all would hold every
    // deletion of every covered fragment in two forms at once, where this holds
    // `io_parallelism` of them beside the treemap they are going into.
    let mut vectors = std::pin::pin!(
        stream::iter(
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
    );

    let mut deleted = RoaringTreemap::new();
    while let Some((fragment_id, deletion_vector)) = vectors.try_next().await? {
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
            .map(|(row_addr, distance)| Neighbor {
                row_addr: *row_addr,
                distance: *distance,
            })
            .collect()
    }

    fn pairs(neighbors: &[Neighbor]) -> Vec<(u64, f32)> {
        neighbors
            .iter()
            .map(|neighbor| (neighbor.row_addr, neighbor.distance))
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
