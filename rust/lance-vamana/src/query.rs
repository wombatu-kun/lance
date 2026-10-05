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
//!   only ever produces `search_list_size` candidates to draw from, of which a
//!   query with a [`SearchParams::rescore_budget`] re-scores that many.
//! - **Rows added after the build are invisible** until they are indexed. The
//!   index answers from the fragments it was built over; Lance's scanner would
//!   scan the remainder. [`crate::inserter::insert_as_segment`] is the remedy.
//! - **A fragment the dataset has dropped is answered for by nobody.** A delete
//!   that empties a fragment takes it out of the dataset, and the vertices
//!   stored for it are then unreachable rather than wrong. The index narrows
//!   itself to what is left and says so through
//!   [`VamanaIndex::covered_fragments`]. A compaction run with
//!   `defer_index_remap` is not that case: Lance records where it moved every
//!   row, and the index answers for the moved rows at their new addresses.
//! - **No predicate prefilter and no refine step.** Both live in the scanner.
//! - **Nothing is cached between queries unless the index is given a cache.**
//!   A query keeps a few reads going at once, so its working set is a few
//!   partitions rather than every partition it probes - and by default every
//!   query pays for its own partitions again. [`VamanaIndex::with_cache`] is
//!   what changes that, and what it keeps is the part of a partition that does
//!   not depend on the query: the layout of its file, and for a
//!   [`WalkMode::Lazy`] walk or a [`WalkMode::Flat`] scan the codes and row ids
//!   they measure by, which are nine tenths of what such a query reads - and,
//!   for a re-score from the dataset, where each data file keeps them. What an
//!   index keeps without one is scratch rather than data: the visited marks of
//!   its lazy walks, one byte a vertex of the largest partition walked, for
//!   as many walks as have run at once and up to one per core - and which of
//!   the dataset's data files no offset read can serve, learnt from a file's
//!   footer the first time it is opened, so that it is not opened again only to
//!   be sent to Lance's take. That is where a read goes, never what it returns:
//!   every byte of every answer is still read every time.
//! - **A re-score from the dataset reads the version the index was opened at.**
//!   An index that leaves its vectors to the dataset, or a query that asks with
//!   [`SearchParams::rescore_from_dataset`], reads them from that version's
//!   data files. Once a compaction and `cleanup_old_versions` have removed
//!   them, a re-score that has to open one fails, naming it, until the index is
//!   opened again - the rule Lance's own indices live by.
//! - **A partition is read whole unless the walk is told not to.**
//!   [`WalkMode::Lazy`] keeps the row ids and the codes and fetches the rest as
//!   it turns out to need it; [`WalkMode::Flat`] keeps the same and fetches even
//!   less, because it scores every vertex instead of following edges to a few of
//!   them. Which of the three is right is a property of the deployment rather
//!   than of the index, and it was measured rather than assumed - except for an
//!   index that leaves its vectors to the dataset, which only the last two can
//!   walk.
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
//! one - a visible overlay, or one a deferred compaction baked into the rows it
//! moved, which it reads the dataset version that compaction recorded to find -
//! when the manifest records a format version this build does not read,
//! when a segment was inherited from another dataset, when the segments
//! disagree about the vectors they hold or about the codes they were built with,
//! or when an index that leaves its vectors to the dataset is opened over a
//! column that cannot supply them. Each refusal names what to do about it, which
//! is always to rebuild.
//!
//! Lance's own paths do not see a committed index: its scanner answers the column
//! as if there were none, its listings leave it out, and its default compaction
//! holds back the fragments the index covers - see the crate README, and the
//! tests that pin it.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use futures::stream::{self, StreamExt, TryStreamExt};
use lance::Dataset;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::{CacheStats, LanceCache};
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_io::scheduler::{IoStats, ScanScheduler, ScanStats};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use lance_table::format::Fragment;
use lance_table::format::overlay::DataOverlayFile;
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::frag_reuse::CompactFragReuseIndex;
use object_store::path::Path;
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

use crate::builder::{live_fragments, routing_distance_type, supported_distance_type};
use crate::cache;
use crate::codes::{self, CODE_COLUMN};
use crate::dataset_vectors::{DatasetVectors, FileAccess};
use crate::entry_points::{self, EntryPointParams, EntryPoints, PartitionEntryPoints};
use crate::format::{
    FORMAT_VERSION, INDEX_FILE_NAME, IndexMetadata, NEIGHBORS_COLUMN, ROW_ID_COLUMN, RowIdMode,
    VECTOR_COLUMN, VectorSource,
};
use lance_io::object_store::ObjectStore;

use crate::io::{
    DirectReads, OPEN_FILES, OpenFiles, PartitionFile, check_partition_shape, open_file,
    read_partition, read_partition_batch, read_segment, scan_scheduler,
};
use crate::lazy::{self, Candidate, LazyProbe};
use crate::partition::{Partition, graph_from_batch, row_ids_from_batch, vectors_of};
use crate::search::{
    Comparisons, ScratchPool, SearchScratch, StopRule, flat_storage, greedy_search,
};
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
    ///
    /// Refused for an index that leaves its vectors to the dataset
    /// ([`VectorSource::Dataset`]), whose partitions store none.
    #[default]
    Exact,
    /// Read the partition whole and measure against its codes, with the
    /// candidate list re-scored exactly before it is answered from.
    ///
    /// Only for an index built with [`crate::IndexParams::with_codes`], and
    /// refused rather than quietly downgraded for one that was not - and for
    /// one that leaves its vectors to the dataset, whose partitions it would
    /// read whole for vectors they do not store. On its own
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
    /// them, the same query reads **72.1 kB** and takes 3.5 ms against 130.5 -
    /// the mode's real number, and the reason it is worth having whenever the
    /// index does not fit in the memory available to it. With
    /// [`SearchParams::rescore_budget`] set as well it is 43.6 kB, off the same
    /// cache.
    ///
    /// Requires codes, same as [`Self::Coded`].
    Lazy,
    /// Do not use the graph at all: score every vertex of the partition against
    /// its code, keep the nearest [`SearchParams::search_list_size`], and
    /// re-score those exactly.
    ///
    /// Reads what [`Self::Lazy`] reads minus the edges - the row ids and the
    /// codes, then the vectors of the candidate list - so `__neighbors` is never
    /// opened and, at this mode's granularity, need not have been written.
    ///
    /// A walk's cost hardly moves with the size of the partition, since its hops
    /// are set by the beam and the graph's diameter rather than by the vertex
    /// count, while a scan's is linear in it - so the two cross somewhere. On
    /// SIFT1M at equal recall (`examples/lazy_walk.rs`) the crossing is not
    /// between the two granularities this crate quotes: against a cached lazy
    /// walk a scan reads 52.2 kB a query against 126.6 at 8192 rows a partition
    /// and 30.9 against 72.1 at 65536, makes one request to a probe against the
    /// walk's eight, reaches higher recall at every beam, and takes 1.6 ms
    /// against 4.5 and 1.8 against 3.5. With
    /// [`SearchParams::rescore_budget`] set as well it reads **11.3 kB and
    /// 9.6**, and makes 3.7 requests and 2.1 - eleven and seven times less than
    /// the walk, off fewer round trips than there are probes.
    ///
    /// It wins the arithmetic too, and not by doing less of it: it measures ten
    /// times as many coded distances but pays about two nanoseconds for each
    /// where a walk pays sixteen, because a scan can hand the whole partition to
    /// [`lance_index::vector::storage::DistCalculator::accumulate_topk_with_scratch`]
    /// and let RaBitQ's error bound throw out most of the extra-bit refinement,
    /// while a walk has to ask one vertex at a time and cannot know in advance
    /// which ones. So the choice is round trips against arithmetic over one
    /// unchanged index file, but at this granularity the arithmetic is no longer
    /// what decides it: a deployment whose CPU is saturated still wants
    /// [`Self::Lazy`] once a partition is large enough, and everything else
    /// wants this.
    ///
    /// Requires codes, same as [`Self::Coded`].
    Flat,
}

impl WalkMode {
    /// Whether this mode can only run on an index that carries codes.
    fn needs_codes(self) -> bool {
        matches!(self, Self::Coded | Self::Lazy | Self::Flat)
    }
}

/// Where a [`WalkMode::Lazy`] walk starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WalkStart {
    /// The partition's medoid, the vertex nearest the mean of its vectors: one
    /// vertex for every query, which the walk then has to get from to wherever
    /// the query is.
    #[default]
    Medoid,
    /// The partition's entry point nearest the query by code, out of the
    /// [`EntryPoints`] given to the index with [`VamanaIndex::with_entry_points`].
    ///
    /// Choosing costs one coded distance per entry point of the partition,
    /// which [`QueryResult::comparisons`] counts. Only the start is marked
    /// visited, so an entry point it was chosen over is measured again if the
    /// walk reaches it. A partition trained no entry points - one with no more
    /// live vertices than [`crate::EntryPointParams::num_entries`] - starts at
    /// its medoid.
    ///
    /// Counted at equal recall on four million-vector datasets at one
    /// partition (`examples/entry_points_walk.rs`), 64 entry points measured 15,
    /// 6, 16 and 9 per cent fewer distances than the medoid at top-10 on SIFT,
    /// GloVe-200, Cohere and GIST, the choice included, and 2 to 8 per cent at
    /// top-100. Timed on the same indexes (`examples/ivf_rq_ab.rs`), each start
    /// at its own margin pair at the recall bar, the search phase took 0.91,
    /// 0.92, 0.89 and 0.91 of the medoid start's at top-10 with one query in
    /// flight, and the time a query at twelve in flight 1.00, 0.94, 0.90 and
    /// 0.92; at top-100 the search phase took 0.90, 0.97, 0.96 and 0.94. See
    /// [`crate::entry_points`].
    ///
    /// Refused for every mode but [`WalkMode::Lazy`], and for an index that was
    /// given no entry points.
    NearestEntry,
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
    /// `L`: how wide a search list each graph walk keeps, and how many
    /// candidates a [`WalkMode::Flat`] scan keeps out of the whole partition.
    ///
    /// With [`Self::stop_margin`] set it is a cap instead: the margin decides
    /// how far a walk goes, and this only how long its list may grow.
    pub search_list_size: usize,
    /// What the walk measures its distances against.
    pub mode: WalkMode,
    /// `W`: how many vertices one hop of a [`WalkMode::Lazy`] walk expands, and
    /// therefore how many rows of `__neighbors` it asks for in one request.
    ///
    /// Ignored by the walks that read a partition whole, which have every edge
    /// already, and by [`WalkMode::Flat`], which reads none. For the lazy one it
    /// is the trade the mode exists to make: the
    /// chain of dependent round trips divides by it, while a wider hop expands
    /// vertices the strictly greedy order would have skipped. Four is the width
    /// the phase gate modelled and is deliberately on the low side - what it
    /// should be on a high-latency store is a measurement nobody has taken.
    pub beam_width: usize,
    /// How many neighbours further on a hop of a [`WalkMode::Lazy`] walk asks
    /// the processor for a code while it measures the current one.
    ///
    /// A hop knows every id it will measure before it measures any of them, and
    /// a code sits at a place in the code array that only the graph knows, so
    /// the load is one no hardware prefetcher can start by itself. Telling it
    /// early is free where the code was resident anyway and is a memory latency
    /// hidden where it was not.
    ///
    /// **Only scalar codes are asked.** The ask is `DistCalculator::prefetch`,
    /// and of the code stores this crate can hold only
    /// `ScalarQuantizationStorage` implements it; `RabitDistCalculator`
    /// inherits the trait's empty default, so on
    /// [`crate::codes::CodeSpec::Rabit`] - the default kind - every depth is a
    /// call that returns. A RaBitQ code is not one run of bytes either: the
    /// binary code, the blocked extended code and two factor arrays are four
    /// places a vertex has to be fetched from, which is presumably why Lance
    /// never wrote one.
    ///
    /// Two, which is what Lance's own HNSW asks for at search time
    /// (`hnsw/builder.rs`, `search_basic` passes `Some(2)`). Zero asks for
    /// nothing, which is the control for the ask itself - though not for the
    /// walk as it was before this existed, which also offered its hop one
    /// neighbour at a time. The depth that pays is a measurement and not a
    /// constant: it has to cover the memory latency of one code without asking
    /// for more cache lines than the processor can have in flight, and an
    /// eight-bit code is two cache lines at `d = 128` against fifteen at
    /// `d = 960`.
    ///
    /// Ignored by the walks that hold a whole partition, which go through
    /// [`crate::search::greedy_search`] and ask for nothing: a look-ahead was
    /// measured in a build's walk ([`crate::build::BUILD_PREFETCH_AHEAD`]) and
    /// never in theirs. Ignored by [`WalkMode::Flat`] too, which sweeps the
    /// partition in order.
    pub prefetch_ahead: usize,
    /// Whether a [`WalkMode::Lazy`] probe holds the whole `__neighbors` column
    /// of a partition it opens, rather than fetching a hop's rows at a time.
    ///
    /// Off by default, because it is memory the alternative does not spend: 256
    /// bytes a vertex at `R = 64`, against the 68 a three-bit code occupies at
    /// `d = 128` and the 376 at `d = 960`. What it buys is every request a walk
    /// makes before its re-score, which after the codes are resident is every
    /// request a walk makes at all. It changes no answer - the same hops, the
    /// same candidate list - and it is ignored by [`WalkMode::Flat`], which
    /// opens the column never and would only be charged for it.
    pub resident_edges: bool,
    /// How many candidates a query measures exactly, counted across every
    /// partition it probes rather than within each of them.
    ///
    /// `None` measures all of them - one exact distance per candidate per probe,
    /// which was the only behaviour before this existed.
    ///
    /// The knob exists because that is where the bytes are. Every mode arrives
    /// at its candidates by code and then corrects them by reading a vector, and
    /// a vector is 512 bytes at `d = 128` against the 68 its code occupies. For
    /// [`WalkMode::Lazy`] and [`WalkMode::Flat`], where the vector is not
    /// resident, that correction is very nearly the whole byte cost of the
    /// query. Spreading it evenly over the probes spends it where the query
    /// looked rather than where the answer is: at seven probes and an `L` of
    /// sixteen, a `k = 10` query corrects a hundred and twelve rows, and the
    /// partitions nearest the query deserve most of them.
    ///
    /// On SIFT1M at equal recall (`examples/lazy_walk.rs`), setting it to `L`
    /// takes a cached [`WalkMode::Flat`] query from 52.2 kB to **11.3** at 8192
    /// rows a partition and from 30.9 to **9.6** at 65536, and a cached
    /// [`WalkMode::Lazy`] one from 126.6 to 67.4 and from 72.1 to 43.6. Nearly
    /// half the probes then fetch nothing at all - a scan makes 3.7 requests
    /// against 7.0 and 2.1 against 4.0 - which is where the round trips go, and
    /// it is the partition gate this crate measured and did not build, arrived
    /// at from the other end.
    ///
    /// What it costs is a wider `L` for the same recall, which is why the
    /// figures above are taken at equal recall and not equal beam: at a fixed
    /// narrow beam a pooled scan is well behind, the two curves meet by
    /// `L = 24`, and from there it reads a seventh as much for a thousandth of
    /// recall and tops out at the same ceiling.
    ///
    /// Refused for [`WalkMode::Exact`], which has no coded ordering to choose
    /// by, and for [`WalkMode::Coded`], which holds every vector already and
    /// would be spending recall on a saving it cannot collect.
    pub rescore_budget: Option<usize>,
    /// How far past its `k`-th nearest candidate a [`WalkMode::Lazy`] walk goes
    /// on expanding, as a fraction of that candidate's length: the `gamma` of
    /// adaptive beam search (Al-Jazzazi et al., NeurIPS 2025).
    ///
    /// `None` walks until its search list has nothing left to expand, which is
    /// how far [`Self::search_list_size`] reaches for every query alike - so a
    /// list long enough for the hardest queries a caller tuned it on is spent
    /// on every easy one too. `Some(gamma)` stops a walk once none of its
    /// unexpanded candidates is among its `k` nearest or nearer than `1 + gamma`
    /// times the length to the `k`-th. A query whose neighbours stand clear of
    /// everything else stops soon after finding them; one crowded by
    /// near-equals walks further. The distances a walk measures are squared
    /// lengths, so the bar is `(1 + gamma)^2` times the `k`-th's distance - or,
    /// over RaBitQ codes, an estimate of one, which can fall below zero: there
    /// the bar sits below the `k`-th, and the walk expands its nearest `k` and
    /// stops whatever the margin.
    ///
    /// Each walk keeps the candidates the re-score is owed whatever their
    /// distance - its nearest [`Self::rescore_budget`], or everything its list
    /// holds without a budget - and past those only what is nearer than the
    /// bar. The list length stops being the knob and becomes a cap, which must
    /// hold the budget, and past which a candidate is dropped however near the
    /// bar it is: a walk meant to follow the margin wants a cap it never
    /// reaches.
    ///
    /// At zero a walk expands its nearest `k` and stops, so with a budget of
    /// `k` it is exactly the walk at a `search_list_size` of `k`.
    ///
    /// Measured on four million-vector datasets at one partition, `R = 70`,
    /// eight-bit scalar codes, resident edges and a budget of 20
    /// (`examples/ivf_rq_ab.rs`), each at its recall bar against the list
    /// length that reaches the same bar: SIFT at 0.99 (a margin of 0.058
    /// against `L = 44`), GloVe-200 at 0.85 (0.042 against 162), Cohere at 0.98
    /// (0.034 against 47.5) and GIST at 0.95 (0.039 against 78). The margin
    /// measured 5, 22, 24 and 13 per cent fewer distances, and its search phase
    /// took 0.92, 0.72, 0.74 and 0.86 of the list's time with one query in
    /// flight and 0.91, 0.74, 0.73 and 0.86 with twelve. The price is the
    /// slowest queries: the 99th percentile measured 1.1 to 1.7 times the
    /// distances. The margin a bar needs depends on the data, as a list length
    /// does.
    ///
    /// Refused for every mode but [`WalkMode::Lazy`] rather than ignored, since
    /// no other mode stops by it; and refused when it is negative, not finite,
    /// or so wide that `(1 + gamma)^2` is not finite, or when
    /// `search_list_size` is shorter than the budget.
    pub stop_margin: Option<f32>,
    /// Whether the answer carries [`QueryResult::coded_neighbors`] beside the
    /// answer itself.
    ///
    /// Off by default because it is a second answer rather than a second
    /// number: the cost fields are two clock reads and a pair of counters, but
    /// this one ranks and dedups the candidate list again, and only a caller
    /// asking what the re-score bought has any use for the result.
    pub report_coded: bool,
    /// Whether the re-score reads the candidates' vectors from the dataset's
    /// own data files instead of the partition's copy.
    ///
    /// The answer is the same either way, to the last bit: the vectors are the
    /// same ones, normalised by the same function for cosine, and measured by
    /// the same arithmetic. What changes is which file is read - and, for a
    /// fragment whose vectors cannot be read by offset (a layout other than bare
    /// full-zip values, an overlay, a file under another base path), how: those
    /// rows go through Lance's take, which [`RescoreReads::through_lance`]
    /// counts and no count of bytes includes. A deleted candidate is dropped
    /// before the read rather than after the measure, so
    /// [`QueryResult::comparisons`] counts only the live ones.
    ///
    /// An index built with [`VectorSource::Dataset`] has no copy to read, and
    /// re-scores from the dataset whatever this says.
    ///
    /// Refused for [`WalkMode::Exact`] and [`WalkMode::Coded`], which have no
    /// re-score read to move: they read every vector along with the partition.
    /// Refused as well, before anything is read, when the dataset cannot supply
    /// the vectors the index was built over.
    pub rescore_from_dataset: bool,
    /// Where a [`WalkMode::Lazy`] walk starts: the medoid, or the entry point
    /// nearest the query ([`WalkStart`]).
    pub start: WalkStart,
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
            prefetch_ahead: 2,
            resident_edges: false,
            rescore_budget: None,
            stop_margin: None,
            report_coded: false,
            rescore_from_dataset: false,
            start: WalkStart::default(),
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

    pub fn with_prefetch_ahead(mut self, prefetch_ahead: usize) -> Self {
        self.prefetch_ahead = prefetch_ahead;
        self
    }

    pub fn with_resident_edges(mut self, resident_edges: bool) -> Self {
        self.resident_edges = resident_edges;
        self
    }

    pub fn with_rescore_budget(mut self, rescore_budget: usize) -> Self {
        self.rescore_budget = Some(rescore_budget);
        self
    }

    pub fn with_stop_margin(mut self, stop_margin: f32) -> Self {
        self.stop_margin = Some(stop_margin);
        self
    }

    /// The rule [`Self::stop_margin`] stands for, as a walk applies it: the
    /// bar hangs off the `k`-th candidate, and the walk keeps what the re-score
    /// is owed.
    pub(crate) fn stop_rule(&self) -> Option<StopRule> {
        self.stop_margin.map(|margin| StopRule {
            margin,
            rank: self.k,
            keep: self.rescore_budget.unwrap_or(self.search_list_size),
        })
    }

    pub fn with_report_coded(mut self, report_coded: bool) -> Self {
        self.report_coded = report_coded;
        self
    }

    pub fn with_rescore_from_dataset(mut self, rescore_from_dataset: bool) -> Self {
        self.rescore_from_dataset = rescore_from_dataset;
        self
    }

    pub fn with_start(mut self, start: WalkStart) -> Self {
        self.start = start;
        self
    }
}

/// What one half of a two-phase query cost.
///
/// A query of [`WalkMode::Lazy`] or [`WalkMode::Flat`] arrives at a candidate
/// list by reading codes and then pays for that list by reading vectors, and the
/// two halves are not the same kind of cost: the first is bounded by the
/// partition and served by a cache, the second is bounded by
/// [`SearchParams::rescore_budget`] and is nobody's to keep. One running total
/// over both cannot say which of them a deployment is actually paying, so the
/// answer carries them apart.
///
/// `elapsed` is wall time inside this query's own future, so under concurrency
/// it includes the time the future spent waiting to be polled: the two add up to
/// the query's latency, not to the pass's throughput.
///
/// The byte counters are physical, taken off the scheduler after coalescing,
/// and they are per query rather than per index - each query attaches its own
/// sink to the partition files it opens. What they cannot see is a read another
/// query started: two queries missing on the same partition at once share one
/// load, and it is charged to whichever of them ran the loader.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseCost {
    pub elapsed: Duration,
    pub bytes_read: u64,
    pub iops: u64,
    pub requests: u64,
}

impl PhaseCost {
    fn new(elapsed: Duration, stats: ScanStats) -> Self {
        Self {
            elapsed,
            bytes_read: stats.bytes_read,
            iops: stats.iops,
            requests: stats.requests,
        }
    }
}

/// Where an index's re-scores read their vectors: see
/// [`VamanaIndex::rescore_reads`].
///
/// Counted in reads as the scheduler would have made them, after coalescing, so
/// `in_place + handed_off` is the part of [`VamanaIndex::io_stats`] iops that
/// re-scores read off the index's own descriptors - all of the re-score's share
/// for an index with a cache, on local storage, on Unix, unless a partition's
/// vectors could not be addressed and were decoded instead. Cumulative since
/// the index was opened.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RescoreReads {
    /// Served by the page cache on the thread that asked for them.
    pub in_place: u64,
    /// Read on the blocking pool: the page cache did not hold all of the read,
    /// the partition's batch was over a megabyte, or the read could not be
    /// asked for without waiting - a platform other than Linux, a kernel older
    /// than 4.14, or a filesystem or sandbox that refuses `RWF_NOWAIT`.
    pub handed_off: u64,
    /// Trips to the blocking pool to read: one for each file's batch that
    /// handed off anything, however much, so up to one per probe a query
    /// re-scores - or, re-scoring from the dataset, per fragment its candidates
    /// sit in. The trip that opens a file's descriptor - on its first
    /// re-score, and again after the open-file cap evicts it - is not counted.
    pub trips: u64,
    /// Rows a re-score from the dataset read through Lance's take rather than
    /// by offset, because their fragment's vectors cannot be read that way -
    /// see [`SearchParams::rescore_from_dataset`]. Rows, not reads, and counted
    /// whether or not the index has a cache. Lance reads them through its own
    /// store, so their bytes are in no count at all: not the three above, not
    /// [`VamanaIndex::io_stats`], not [`QueryResult::rescore`].
    pub through_lance: u64,
}

impl RescoreReads {
    /// What was read between `earlier` and this snapshot, both taken from one
    /// index between passes.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            in_place: self.in_place.saturating_sub(earlier.in_place),
            handed_off: self.handed_off.saturating_sub(earlier.handed_off),
            trips: self.trips.saturating_sub(earlier.trips),
            through_lance: self.through_lance.saturating_sub(earlier.through_lance),
        }
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
    /// routed through, plus one per vertex any graph walk considered - every
    /// entry point a walk chose its start among included, and again any it
    /// then reached.
    ///
    /// Routing is counted because it is paid unconditionally and does not scale
    /// with `nprobes` - a segment of 4096 centroids charges 4096 distances
    /// before a single vertex is read. Reporting only the walk would make a
    /// finely partitioned index look cheap at exactly the point it stops being.
    pub comparisons: u64,
    pub partitions_read: usize,
    /// Reaching a candidate list: routing, the partition's layout, the codes
    /// and row ids a walk steers by, and the out-edges of every vertex it
    /// expanded. The whole query for the modes that read a partition whole.
    pub search: PhaseCost,
    /// Correcting that list: the vectors of the candidates a budget was spent
    /// on, and the exact distances measured against them. Zero for the modes
    /// that hold every vector already and never re-score, and short of the
    /// rows read through Lance's take ([`RescoreReads::through_lance`]). A
    /// re-score from the dataset by an index given no cache also counts here
    /// where each data file keeps the column - its tail and that column's
    /// metadata - once for every file the query reads by offset.
    pub rescore: PhaseCost,
    /// The `k` this query would have answered with had it stopped after
    /// [`Self::search`]: the candidate list ranked by its *coded* distances,
    /// which are estimates, so `distance` here is an estimate too.
    ///
    /// Empty unless [`SearchParams::report_coded`] asked for it, and empty for
    /// the modes that have no separate re-score to stop before.
    pub coded_neighbors: Vec<Neighbor>,
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
    /// time, not almost every time. What it does remember is which of the
    /// dataset's data files no offset read can serve, so as not to open one
    /// again only to be sent to Lance: where a read goes rather than anything
    /// a read returns, and the rows themselves are read every time. See
    /// [`Self::with_cache`].
    cache: Option<LanceCache>,
    /// Whether the index was opened for a pass that rewrites partitions, whose
    /// cache holds only what it learns of the dataset's data files and whose
    /// reads of them go through the scheduler. See
    /// [`Self::open_for_maintenance`].
    is_maintenance: bool,
    metadata: IndexMetadata,
    segments: Vec<Segment>,
    /// Fragments this index still answers for: what its segments were built
    /// over, followed through any deferred compaction, minus what the dataset
    /// has since dropped.
    covered: RoaringBitmap,
    /// Which stored vertices must not reach an answer, as of
    /// [`VamanaIndex::open`].
    ///
    /// Shared rather than owned because each partition's walk runs on the CPU
    /// pool, which takes `'static` work, and the filter has to be applied inside
    /// the walk's own result - before `take(k)`, so that `k` means k live rows.
    rows: Arc<RowFilter>,
    /// Visited marks for the lazy walks, kept between queries whether or not
    /// the index has a cache. The walks that read a partition whole allocate
    /// their own, beside the partition they hold.
    scratches: ScratchPool,
    /// Partition files this index has open, shared by the queries that probe
    /// them. Empty for an index with no cache, which reads rather than holds.
    files: OpenFiles,
    /// Kept to tell a local file from a remote one, which - for an index with a
    /// cache - decides whether a re-score reads through the scheduler or off the
    /// index's own descriptors.
    store: Arc<ObjectStore>,
    /// What this index has read without going through its scheduler, which is
    /// the only place those bytes are counted, and which of those reads went to
    /// the blocking pool.
    direct: Arc<DirectReads>,
    /// The dataset's own copy of the indexed vectors. The only copy an index of
    /// [`VectorSource::Dataset`] has, read by every re-score and by every pass
    /// that repairs a partition - one whose rows only moved is written from
    /// its own file; an index keeping its own reads it only for a query that
    /// asks with [`SearchParams::rescore_from_dataset`].
    dataset_vectors: DatasetVectors,
    /// Where a walk asked to start at entry points ([`WalkStart::NearestEntry`])
    /// finds them, checked against this index's partitions by
    /// [`Self::with_entry_points`]. `None` until it is called, and such a walk is
    /// refused.
    entry_points: Option<Arc<EntryPoints>>,
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
    /// `None` unless a deferred compaction moved rows some segment stores.
    moved: Option<MovedRows>,
}

/// Where deferred compactions moved the rows of an index, and which fragments
/// each segment answers for once they have.
#[derive(Debug)]
struct MovedRows {
    remap: Arc<CompactFragReuseIndex>,
    /// By segment, because a row can move into a fragment another segment
    /// covers - a delta indexed over it, say - and is still not this one's.
    coverage: HashMap<Uuid, RoaringBitmap>,
}

impl RowFilter {
    /// The address a vertex `segment` stores answers under now, or `None` when
    /// it must not reach an answer.
    pub(crate) fn admit(&self, segment: Uuid, stored: u64) -> Option<u64> {
        let current = match &self.moved {
            None => {
                if self
                    .missing_fragments
                    .contains(RowAddress::from(stored).fragment_id())
                {
                    return None;
                }
                stored
            }
            Some(moved) => {
                let current = moved.remap.remap_row_id(stored)?;
                // A group the segment covered only part of moves its rows into a
                // fragment Lance does not credit the segment with, and leaves
                // them to whoever scans the unindexed fragments.
                let fragment_id = RowAddress::from(current).fragment_id();
                if !moved
                    .coverage
                    .get(&segment)
                    .is_some_and(|coverage| coverage.contains(fragment_id))
                {
                    return None;
                }
                current
            }
        };
        (!self.deleted.contains(current)).then_some(current)
    }

    pub(crate) fn rejects(&self, segment: Uuid, stored: u64) -> bool {
        self.admit(segment, stored).is_none()
    }

    /// Whether every stored vertex is live and still at the address it was
    /// stored under.
    pub(crate) fn is_empty(&self) -> bool {
        self.deleted.is_empty() && self.missing_fragments.is_empty() && self.moved.is_none()
    }

    /// A partition of `segment` with each vertex at the address it answers under
    /// now, and a vertex that answers under none left where it was stored.
    pub(crate) fn readdress(&self, segment: Uuid, partition: Partition) -> Result<Partition> {
        let (mut graph, vectors) = partition.into_parts();
        for row_addr in graph.row_ids_mut() {
            if let Some(current) = self.admit(segment, *row_addr) {
                *row_addr = current;
            }
        }
        Partition::try_new(graph, vectors)
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
    /// What the dataset credits this segment with and still has, in the
    /// fragment ids its rows live under now.
    ///
    /// Narrower than the segment's own `fragments` exactly when a fragment has
    /// gone or is no longer credited to it; every vertex stored for one of those
    /// is already rejected by [`RowFilter`], so this is the coverage a rewrite of
    /// this segment would be committed with.
    pub(crate) coverage: RoaringBitmap,
    /// Whether a deferred compaction moved rows this segment stores, so that a
    /// rewrite has to write every vertex at [`RowFilter::readdress`]'s address
    /// rather than copy a partition as it is.
    pub(crate) moved: bool,
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
    segment: Uuid,
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

/// One probed partition after its walk or its scan, and before anything has
/// been read to correct what they measured.
///
/// The one thing a query holds per probe rather than per probe *in flight*, and
/// deliberately small enough for that: a file handle whose metadata is shared
/// with the cache, a width, and `L` candidates of sixteen bytes each. The codes
/// and the row ids the walk ran on are gone by the time one of these exists -
/// [`crate::lazy::Candidate`] carries the row address forward so that nothing
/// downstream needs them - so [`PARTITIONS_IN_FLIGHT`] still bounds what a query
/// holds of a partition.
struct Probing {
    segment: Uuid,
    file: PartitionFile,
    /// What the segment declares the vector width to be, carried so that
    /// re-scoring can check it against what comes back.
    dimension: u32,
    /// Ascending by local id, which is the order re-scoring reads them in and
    /// the order [`allocate`] leaves them in.
    candidates: Vec<Candidate>,
    comparisons: u64,
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

/// What an index opened for a pass that rewrites partitions may keep: the
/// layout of one column of each of the dataset's data files it reads, some
/// twenty-four bytes for every 8 MiB page of that column - about 11 KiB for a
/// file of a million 960-wide vectors, so the layouts of a few thousand such
/// files. Past that the cache drops entries, which are read again when a
/// partition asks, at the cost of a footer read and not of an answer.
const MAINTENANCE_CACHE_BYTES: usize = 64 << 20;

/// Every committed segment named `index_name`, in manifest order.
///
/// Read from the manifest's own index section rather than through
/// `DatasetIndexExt::load_indices_by_name`: since upstream #8529 that view leaves
/// out every index whose details type has no reader in the Lance build, and
/// [`crate::builder::INDEX_DETAILS_TYPE_URL`] never has one.
///
/// The bitmaps are as the manifest stores them, without the fragment-reuse remap
/// Lance applies when it lists indices. [`VamanaIndex::open`] applies that remap
/// itself, to these and to each segment's own record of what it read alike.
pub async fn committed_segments(
    dataset: &Dataset,
    index_name: &str,
) -> Result<Vec<lance_table::format::IndexMetadata>> {
    let store = dataset.object_store(None).await?;
    let indices =
        read_manifest_indexes(&store, dataset.manifest_location(), dataset.manifest()).await?;
    Ok(indices
        .into_iter()
        .filter(|index| index.name == index_name)
        .collect())
}

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
        let indices = committed_segments(dataset, index_name).await?;
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
        let store = dataset.object_store(None).await?;
        let scheduler = scan_scheduler(&store);
        let remap = dataset.frag_reuse_index().await?;

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
        let mut history = HashMap::new();
        for ((index, dir, file_sizes, declared), manifest) in planned.into_iter().zip(manifests) {
            // Three records of one thing, and every disagreement between them
            // means something different. `stored_over` is what the segment wrote
            // about itself and never changes; `declared` is what the dataset
            // credits it with, which Lance edits in place and which never touches
            // the segment's own files; `live` is which fragments the dataset
            // still has at all.
            let stored_over = manifest
                .metadata()
                .fragments
                .iter()
                .copied()
                .collect::<RoaringBitmap>();
            // A deferred compaction moves rows into new fragments and records the
            // move in the fragment-reuse index, which Lance applies to every
            // bitmap it lists and persists on its next commit. Both records are
            // put through it, so that they can be compared in the fragment ids
            // the rows live under now.
            let mut built_over = stored_over.clone();
            let mut credited = declared.clone();
            if let Some(remap) = &remap {
                remap.remap_fragment_bitmap(&mut built_over)?;
                remap.remap_fragment_bitmap(&mut credited)?;
            }

            // Credited with a fragment it never read. Lance widens a bitmap in
            // `register_pure_rewrite_rows_update_frags_in_indices` and in the
            // pruning path of a deferred commit; the first is gated on stable row
            // ids, which the builder refuses outright, so today only the second
            // can produce it - but a bitmap naming a fragment this segment never
            // read is unanswerable either way, and which upstream path widened it
            // is not something a reader can tell.
            if !(&credited - &built_over).is_empty() {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built over {} fragments but the dataset \
                     credits it with {}, so it is expected to answer for rows it never read; \
                     rebuild the index",
                    index.uuid,
                    built_over.len(),
                    credited.len()
                )));
            }
            // Built over a fragment that is still here, but no longer credited
            // with it. That is Lance saying the data under those addresses was
            // rewritten: an in-place column update, or the coverage pruning a
            // deferred commit runs. The fragment ids and every row address
            // survive it, so nothing downstream would notice - the vectors this
            // segment ranks by are simply not the ones the rows now hold.
            //
            // Asked of the records as stored. A fragment a compaction moved is
            // not still here, and one Lance stopped crediting before it recorded
            // the move - a build committed after a compaction that raced it - is
            // narrowed below rather than refused.
            let rewritten = (&stored_over - declared) & &live;
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
            // A compaction the fragment-reuse index recorded is not one of these:
            // its rows were followed above. What is left is a delete that emptied
            // the fragment, or a move that record no longer holds, and which of
            // the two it was does not change what this index can do. It changes
            // what the *caller* should do, so the narrowing is logged and
            // `covered_fragments` reports the result.
            let coverage = &credited & &live;
            let gone = &built_over - &coverage;
            if !gone.is_empty() {
                log::warn!(
                    "Vamana index '{index_name}' segment {} was built over {} fragments the \
                     dataset no longer has or no longer credits it with; it will answer for the \
                     remaining {}, and the rows of the rest are the caller's to scan",
                    index.uuid,
                    gone.len(),
                    coverage.len()
                );
                missing_fragments |= gone;
            }
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
                credited.contains(*fragment_id)
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
            let moved = built_over != stored_over;
            if moved && let Some(remap) = &remap {
                refuse_overlays_moved_with_rows(
                    dataset,
                    index_name,
                    index,
                    &stored_over,
                    remap,
                    &mut history,
                )
                .await?;
            }
            segments.push(Segment {
                uuid: index.uuid,
                dir,
                manifest,
                file_sizes,
                fields: index.fields.clone(),
                coverage,
                moved,
            });
        }

        let metadata = segments[0].manifest.metadata().clone();
        for segment in &segments[1..] {
            let other = segment.manifest.metadata();
            // Degree and pruning slack may legitimately differ between a base
            // segment and one appended later; the identifier space, the metric,
            // the width, the codes and where the vectors are may not, because a
            // query mixes their answers - and one segment coded where another is
            // not would make the walk mode mean two different things in one
            // query.
            if (
                other.dimension,
                other.distance_type,
                other.row_id_mode,
                &other.codes,
                other.vector_source,
            ) != (
                metadata.dimension,
                metadata.distance_type,
                metadata.row_id_mode,
                &metadata.codes,
                metadata.vector_source,
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

        let dataset_vectors = DatasetVectors::of(
            dataset,
            &segments[0].fields,
            metadata.dimension,
            metadata.distance_type,
            &covered,
        );
        // Refused here rather than by the first query, which is where an index
        // with a copy of its own finds out: this one has nothing else to read.
        // Ahead of the delete list, which costs a read a fragment, because the
        // manifest alone decides it.
        if metadata.vector_source == VectorSource::Dataset
            && let Some(reason) = dataset_vectors.unavailable()
        {
            return Err(Error::index(format!(
                "index '{index_name}' leaves its vectors to the dataset, which cannot supply \
                 them: {reason}; rebuild the index"
            )));
        }

        let deleted = deleted_row_addresses(dataset, &covered, store.io_parallelism()).await?;
        let moved = remap
            .filter(|_| segments.iter().any(|segment| segment.moved))
            .map(|remap| MovedRows {
                remap,
                coverage: segments
                    .iter()
                    .map(|segment| (segment.uuid, segment.coverage.clone()))
                    .collect(),
            });

        Ok(Self {
            scheduler,
            cache: None,
            is_maintenance: false,
            metadata,
            segments,
            covered,
            rows: Arc::new(RowFilter {
                deleted,
                missing_fragments,
                moved,
            }),
            scratches: ScratchPool::new(),
            files: OpenFiles::new(OPEN_FILES),
            store,
            direct: Arc::new(DirectReads::default()),
            dataset_vectors,
            entry_points: None,
        })
    }

    /// [`Self::open`], for a pass that rewrites partitions - consolidation, a
    /// merge, an insertion in place - which, for an index that leaves its
    /// vectors to the dataset, reads the vectors of a partition it repairs out
    /// of every fragment the partition's rows are in.
    ///
    /// Given a cache of its own for the length of the pass, so that where a
    /// data file keeps its vectors is read out of the file's footer once
    /// rather than once for every partition that reads from it, and so that
    /// the file stays open between them, for as many files as an index holds
    /// open ([`OPEN_FILES`]). Nothing else a pass reads goes through the cache.
    /// Its reads of those files go through the scheduler rather than off
    /// descriptors of its own: a partition read whole is thousands of rows
    /// scattered over each file, which the scheduler reads in parallel.
    pub(crate) async fn open_for_maintenance(dataset: &Dataset, index_name: &str) -> Result<Self> {
        let mut index = Self::open(dataset, index_name).await?;
        index.cache = Some(LanceCache::with_capacity(MAINTENANCE_CACHE_BYTES));
        index.is_maintenance = true;
        Ok(index)
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
    /// segment, or one column of a data file a re-score from the dataset read,
    /// and each is written once: deleting rows edits no index file at all, adding rows or
    /// consolidating writes a *new* segment under a new uuid, and Lance writes a
    /// changed column to a new data file rather than into an old one. So what an
    /// old entry describes is either still exactly true or no longer named by
    /// anything. Which of the two it is decides only when the budget reclaims it.
    pub fn with_cache(mut self, cache: LanceCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// What the cache has served and what it holds, or `None` for an index that
    /// was never given one.
    ///
    /// Counts every kind of entry a query looks up - a partition's codes, a
    /// partition file's layout, and the layout of the column a re-score from
    /// the dataset reads out of a data file - so a hit ratio here is per lookup
    /// rather than per query.
    pub async fn cache_stats(&self) -> Option<CacheStats> {
        let cache = self.cache.as_ref()?;
        Some(cache.stats().await)
    }

    /// Train entry points for every partition of this index, for
    /// [`Self::with_entry_points`] to hand to this index or to another opening
    /// of it.
    ///
    /// Reads each partition's live vectors once, one partition at a time - its
    /// own `__vector`, or the dataset's for an index that leaves its vectors
    /// there, which a cosine index reads normalised either way, so the two
    /// train to the same entry points. Those reads go through this index's
    /// scheduler, so they count in [`Self::io_stats`] - all but the dataset's
    /// rows that only Lance's take can reach, which count in
    /// [`Self::rescore_reads`] instead: train on an opening that is not being
    /// measured, or take differences. Nothing is left in this index's cache.
    ///
    /// How a partition's entry points are picked is [`crate::entry_points`]'s
    /// business, and how many and from what sample is `params`'s.
    pub async fn train_entry_points(&self, params: &EntryPointParams) -> Result<EntryPoints> {
        params.validate()?;
        // Where the layout of each data file a vector-less index reads is kept
        // for the length of the training, so that it is read once rather than
        // once for every partition with rows in the file - the cache a pass
        // that rewrites partitions is given, and dropped with it.
        let layouts = LanceCache::with_capacity(MAINTENANCE_CACHE_BYTES);
        let mut partitions = Vec::new();
        for segment in &self.segments {
            for entry in segment.manifest.partitions() {
                let (row_ids, vectors) = self.partition_vectors(segment, entry, &layouts).await?;
                let live = row_ids
                    .iter()
                    .enumerate()
                    .filter(|(_, stored)| !self.rows.rejects(segment.uuid, **stored))
                    .map(|(local_id, _)| local_id as u32)
                    .collect::<Vec<_>>();
                let entries = entry_points::train_partition(
                    vectors,
                    row_ids,
                    live,
                    self.metadata.distance_type,
                    params,
                )
                .await?;
                partitions.push(PartitionEntryPoints {
                    segment: segment.uuid,
                    partition_id: entry.partition_id,
                    num_rows: entry.num_rows,
                    entries,
                });
            }
        }
        EntryPoints::from_partitions(params.clone(), partitions)
    }

    /// Let walks asked to start at entry points ([`WalkStart::NearestEntry`])
    /// find them.
    ///
    /// Refused unless `entry_points` names exactly this index's partitions -
    /// every segment's, at the vertex count each holds - which is what tells
    /// one index's apart from another's, or from this one's before rows were
    /// added. Rows deleted since they were trained do not: a deleted vertex is
    /// still a vertex of the graph, and a walk may start at it.
    pub fn with_entry_points(mut self, entry_points: Arc<EntryPoints>) -> Result<Self> {
        let mut listed = HashSet::new();
        for segment in &self.segments {
            for entry in segment.manifest.partitions() {
                let Some(trained) = entry_points.of(segment.uuid, entry.partition_id) else {
                    return Err(Error::invalid_input(format!(
                        "the entry points were trained for another index: they have none for \
                         partition {} of segment {}",
                        entry.partition_id, segment.uuid
                    )));
                };
                if trained.num_rows != entry.num_rows {
                    return Err(Error::invalid_input(format!(
                        "the entry points were trained over {} vertices of partition {} of \
                         segment {}, which holds {}",
                        trained.num_rows, entry.partition_id, segment.uuid, entry.num_rows
                    )));
                }
                listed.insert((segment.uuid, entry.partition_id));
            }
        }
        if let Some(extra) = entry_points
            .partitions()
            .iter()
            .find(|trained| !listed.contains(&(trained.segment, trained.partition_id)))
        {
            return Err(Error::invalid_input(format!(
                "the entry points were trained for another index: this one has no partition {} \
                 of segment {}",
                extra.partition_id, extra.segment
            )));
        }
        self.entry_points = Some(entry_points);
        Ok(self)
    }

    /// Open one probed partition's file, or take the one this index already has
    /// open, with this query's sink bound onto it.
    ///
    /// Files are held only by an index that was given a cache. An index that was
    /// not is documented to read every time rather than almost every time, and a
    /// held file holds its footer with it, which is most of what "every time"
    /// was about.
    async fn partition_file(&self, probe: &Probe, stats: &IoStats) -> Result<PartitionFile> {
        let Some(cache) = &self.cache else {
            return PartitionFile::open(
                &self.scheduler,
                &probe.path,
                probe.size_bytes,
                Some(stats),
            )
            .await;
        };
        if let Some(open) = self.files.get(&probe.path, stats) {
            return Ok(open);
        }
        let opened = PartitionFile::open_cached(
            &self.scheduler,
            &probe.path,
            probe.size_bytes,
            cache,
            Some(stats),
        )
        .await?
        .with_local_reads(&self.store, &self.direct);
        Ok(self.files.put(&probe.path, opened, stats))
    }

    /// What this index answers for: every fragment its segments were built over,
    /// followed through any deferred compaction, that the dataset still has.
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
    /// around the store, which under-counts a local read - plus the re-scores
    /// read off the index's own descriptors, in place or on the blocking pool,
    /// which the scheduler never sees.
    pub fn io_stats(&self) -> ScanStats {
        let scheduled = self.scheduler.stats();
        let direct = self.direct.totals.snapshot();
        ScanStats {
            iops: scheduled.iops + direct.iops,
            requests: scheduled.requests + direct.requests,
            bytes_read: scheduled.bytes_read + direct.bytes_read,
        }
    }

    /// Which thread read the vectors this index's re-scores asked for, since it
    /// was opened.
    ///
    /// The reads [`Self::io_stats`] adds to its scheduler's, split by who made
    /// them. With the partition files in the page cache none should be handed
    /// off, so a count here then means a batch over a megabyte, a filesystem or
    /// sandbox that refuses to read without waiting, or a platform other than
    /// Linux. Zero throughout, [`RescoreReads::through_lance`] apart, for an
    /// index not given a cache ([`Self::with_cache`]), one whose store is not
    /// local or reads through io_uring, or one off Unix: each of those
    /// re-scores through its scheduler. Take it between passes, and difference
    /// two of them with
    /// [`RescoreReads::since`]: the counters are read one after the other, so
    /// while queries are in flight the split can be off by a batch.
    pub fn rescore_reads(&self) -> RescoreReads {
        RescoreReads {
            through_lance: self.dataset_vectors.lance_rows(),
            ..self.direct.split()
        }
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

    pub(crate) fn row_filter(&self) -> &Arc<RowFilter> {
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
    /// record, because the two differ when a fragment has gone and when a
    /// deferred compaction moved rows: a fragment the dataset has dropped is
    /// answered for by nobody, and one a compaction moved rows into is answered
    /// for by the segment they came from - unless that segment covered only part
    /// of what was compacted with them, in which case it belongs in this list.
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
        if let Some(budget) = params.rescore_budget {
            // Refused rather than ignored, for the reason the coded modes are:
            // a caller setting a budget is asking about cost, and a mode that
            // cannot spend it would answer a question they did not ask.
            if !matches!(params.mode, WalkMode::Lazy | WalkMode::Flat) {
                return Err(Error::invalid_input(format!(
                    "rescore_budget was set for {:?}, which cannot spend it: WalkMode::Exact has \
                     no coded ordering to choose a budget by, and WalkMode::Coded holds every \
                     vector already, so a budget would cost it recall and save it nothing",
                    params.mode
                )));
            }
            if budget < params.k {
                return Err(Error::invalid_input(format!(
                    "rescore_budget {budget} is smaller than k {}, so the query could never \
                     return k neighbours",
                    params.k
                )));
            }
        }
        // Refused rather than ignored, like a budget: the caller is asking where
        // a re-score reads, and these modes have no read of their own to move -
        // they take every vector along with the partition.
        if params.rescore_from_dataset && !matches!(params.mode, WalkMode::Lazy | WalkMode::Flat) {
            return Err(Error::invalid_input(format!(
                "rescore_from_dataset was set for {:?}, which has no re-score read to redirect: \
                 it reads every vector of the partitions it probes along with them",
                params.mode
            )));
        }
        // Refused before the walk rather than by the read after it, which a
        // query whose probes keep no live candidate never reaches: that one
        // would answer as if the switch were off.
        if params.rescore_from_dataset
            && let Some(reason) = self.dataset_vectors.unavailable()
        {
            return Err(Error::invalid_input(format!(
                "rescore_from_dataset was set, but the dataset cannot supply the vectors this \
                 index was built over: {reason}"
            )));
        }
        if let Some(stop) = params.stop_rule() {
            if params.mode != WalkMode::Lazy {
                return Err(Error::invalid_input(format!(
                    "stop_margin was set for {:?}, which does not stop by it: only a \
                     WalkMode::Lazy walk does",
                    params.mode
                )));
            }
            // A square that overflows would make the bar over a zero distance
            // not a number, and which way that sorts is the platform's choice.
            let widened = 1.0 + stop.margin;
            if !stop.margin.is_finite() || stop.margin < 0.0 || !(widened * widened).is_finite() {
                return Err(Error::invalid_input(format!(
                    "stop_margin {} is not a finite fraction of at least zero with a finite \
                     (1 + stop_margin)^2, so it cannot say how far past its k-th candidate a \
                     walk goes on",
                    stop.margin
                )));
            }
            if params.search_list_size < stop.keep {
                return Err(Error::invalid_input(format!(
                    "stop_margin keeps the rescore_budget of {} candidates in every walk, but \
                     search_list_size {} caps each walk's list below that",
                    stop.keep, params.search_list_size
                )));
            }
        }
        if params.start == WalkStart::NearestEntry {
            // Refused rather than ignored, like a stop margin: no other mode
            // walks from a start this could move.
            if params.mode != WalkMode::Lazy {
                return Err(Error::invalid_input(format!(
                    "start NearestEntry was set for {:?}, which does not start from entry \
                     points: only a WalkMode::Lazy walk does",
                    params.mode
                )));
            }
            if self.entry_points.is_none() {
                return Err(Error::invalid_input(
                    "start NearestEntry was set, but this index was given no entry points; \
                     train them with VamanaIndex::train_entry_points and hand them over with \
                     VamanaIndex::with_entry_points"
                        .to_string(),
                ));
            }
        }
        if self.metadata.vector_source == VectorSource::Dataset
            && matches!(params.mode, WalkMode::Exact | WalkMode::Coded)
        {
            return Err(Error::invalid_input(format!(
                "this Vamana index leaves its vectors to the dataset, so {:?} cannot search it: \
                 that mode reads partitions whole, vectors included, and they hold none; use \
                 WalkMode::Lazy or WalkMode::Flat",
                params.mode
            )));
        }
        // Refused rather than answered exactly. A caller asking for a coded
        // walk is asking about cost, and quietly giving them a walk that reads
        // every vector would be an answer to a different question.
        if params.mode.needs_codes() && self.metadata.codes.is_none() {
            return Err(Error::invalid_input(
                "this Vamana index was built without codes, so it cannot be walked by them; \
                 rebuild it with IndexParams::with_codes"
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

        // One sink a phase, per query rather than per index: the counters the
        // scheduler keeps are shared by every query in flight, so a split read
        // off them would be one query's phase minus another's.
        let search_stats = IoStats::new();
        let rescore_stats = IoStats::new();
        let started = Instant::now();

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
        // Deferred rather than defaulted: every arm below sets it, and a default
        // here would let an arm that forgot to report a zero-cost query.
        let search;
        let mut rescore = PhaseCost::default();
        let mut coded_neighbors = Vec::new();
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
        match params.mode {
            // Two passes with a barrier between them, because deciding which
            // candidates deserve an exact distance needs every probe's list at
            // once. The barrier costs no round trip: the reads of the second
            // pass are independent of each other and go out together, exactly as
            // they did when each probe issued its own, and what used to overlap
            // them was another probe's *arithmetic* rather than another read.
            WalkMode::Lazy | WalkMode::Flat => {
                let mut probings = Vec::new();
                let mut probed = stream::iter(probes)
                    .map({
                        let routing_query = routing_query.clone();
                        let search_stats = &search_stats;
                        move |probe| {
                            self.probe_lazily(probe, routing_query.clone(), params, search_stats)
                        }
                    })
                    .buffer_unordered(PARTITIONS_IN_FLIGHT)
                    .boxed();
                while let Some(probing) = probed.try_next().await? {
                    partitions_read += 1;
                    comparisons = comparisons.saturating_add(probing.comparisons);
                    probings.push(probing);
                }
                drop(probed);

                if let Some(budget) = params.rescore_budget {
                    allocate(&mut probings, budget);
                }
                // After `allocate` and not before it, so that what this reports
                // is the answer the query would have given rather than one taken
                // from candidates it went on to throw away. The two agree while
                // the budget is at least `k`, which is checked above; taking it
                // here means they agree by construction rather than by argument.
                if params.report_coded {
                    coded_neighbors = coded_answer(&probings, &self.rows, params.k);
                }

                search = PhaseCost::new(started.elapsed(), search_stats.snapshot());
                let rescore_started = Instant::now();
                let mut rescoring = stream::iter(probings)
                    .map({
                        let query = query.clone();
                        let rescore_stats = &rescore_stats;
                        move |probing| {
                            self.rescore_probing(probing, query.clone(), params, rescore_stats)
                        }
                    })
                    .buffer_unordered(PARTITIONS_IN_FLIGHT)
                    .boxed();
                while let Some(walked) = rescoring.try_next().await? {
                    found.extend(walked.neighbors);
                    comparisons = comparisons.saturating_add(walked.comparisons);
                }
                rescore = PhaseCost::new(rescore_started.elapsed(), rescore_stats.snapshot());
            }
            WalkMode::Exact | WalkMode::Coded => {
                let mut walks = stream::iter(probes)
                    .map(|probe| self.read_probe(probe, &search_stats))
                    .buffer_unordered(PARTITIONS_IN_FLIGHT)
                    .and_then({
                        let query = query.clone();
                        let routing_query = routing_query.clone();
                        move |probed| {
                            self.walk_partition(
                                probed,
                                query.clone(),
                                routing_query.clone(),
                                params,
                            )
                        }
                    })
                    .boxed();
                while let Some(walked) = walks.try_next().await? {
                    partitions_read += 1;
                    found.extend(walked.neighbors);
                    comparisons = comparisons.saturating_add(walked.comparisons);
                }
                // Nothing here re-scores, so the whole query is the one phase
                // and the other stays at zero rather than being left unset.
                search = PhaseCost::new(started.elapsed(), search_stats.snapshot());
            }
        }

        Ok(QueryResult {
            neighbors: merge(found, params.k),
            comparisons,
            partitions_read,
            search,
            rescore,
            coded_neighbors,
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
                // What this term is depends on the kind of code the segment
                // carries, so the segment's own parameters decide it; for RaBitQ
                // it is `|q - c|^2` and `dimension` flops a probed partition is
                // nothing beside reading one.
                let dist_q_c = params
                    .mode
                    .needs_codes()
                    .then(|| {
                        let codes = declared.codes.as_ref().ok_or_else(|| {
                            Error::invalid_input(
                                "a Vamana coded walk was scheduled for a segment without codes"
                                    .to_string(),
                            )
                        })?;
                        codes.query_offset(segment.manifest.ivf(), *partition_id, routing_query)
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
                segment,
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

            let row_ids = partition.graph().row_ids();
            let neighbors = candidates
                .into_iter()
                .map(|(id, distance)| Neighbor {
                    row_addr: row_ids[id as usize],
                    distance,
                })
                .collect::<Vec<_>>();
            Ok(Walked {
                neighbors: answer(neighbors, &rows, segment, k)?,
                comparisons: walked.get(),
            })
        })
        .await
    }

    /// Answer from one partition without reading it, on this runtime rather than
    /// on the CPU pool.
    ///
    /// The opposite bargain from [`Self::walk_partition`], and forced rather than
    /// chosen: the pool takes work that never waits, and this waits once a hop.
    /// What it hands the pool instead is nothing at all - a hop is `beam_width`
    /// times `max_degree` coded distances, tens of microseconds, below the size
    /// at which the pool's own overhead starts to pay. A [`WalkMode::Flat`] scan
    /// waits twice however large the partition is, and is milliseconds of
    /// arithmetic in between, so it is the one thing here the pool would suit -
    /// which is a measurement to take once the mode has earned it.
    ///
    /// The read of the row ids and the codes is the one thing here that is
    /// proportional to the partition. It is also what makes both modes possible
    /// at all, and it is a tenth of what reading the partition whole would be at
    /// `d = 128`.
    async fn probe_lazily(
        &self,
        probe: Probe,
        routing_query: ArrayRef,
        params: &SearchParams,
        stats: &IoStats,
    ) -> Result<Probing> {
        let Some(dist_q_c) = probe.dist_q_c else {
            return Err(Error::internal(
                "a Vamana lazy walk was scheduled for a segment without codes".to_string(),
            ));
        };
        let file = self.partition_file(&probe, stats).await?;
        // A scan opens `__neighbors` never, so holding it for one would charge
        // the arm without a graph for the graph.
        let hold_edges = params.resident_edges && !matches!(params.mode, WalkMode::Flat);
        let resident = cache::resident(
            self.cache.as_ref(),
            probe.segment,
            &probe.entry,
            &file,
            &self.metadata,
            hold_edges.then_some(probe.max_degree),
        )
        .await?;

        let entries = match params.start {
            WalkStart::Medoid => None,
            WalkStart::NearestEntry => {
                let trained = self
                    .entry_points
                    .as_ref()
                    .and_then(|entry_points| {
                        entry_points.of(probe.segment, probe.entry.partition_id)
                    })
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "partition {} of segment {} has no entry points, though the index \
                             accepted them",
                            probe.entry.partition_id, probe.segment
                        ))
                    })?;
                // Empty for a partition too small for entry points to pay,
                // which starts at its medoid.
                (!trained.entries.is_empty()).then_some(trained.entries.as_slice())
            }
        };
        let (candidates, comparisons) = {
            let probing = LazyProbe {
                file: &file,
                codes: &resident.codes,
                row_ids: &resident.row_ids,
                medoid: probe.entry.medoid,
                entries,
                max_degree: probe.max_degree,
                search_list_size: params.search_list_size,
                beam_width: params.beam_width,
                prefetch_ahead: params.prefetch_ahead,
                edges: resident.edges.as_deref(),
                stop: params.stop_rule(),
            };
            match params.mode {
                WalkMode::Flat => probing.scan(routing_query, dist_q_c),
                WalkMode::Exact | WalkMode::Coded | WalkMode::Lazy => {
                    let mut scratch = self.scratches.take();
                    probing.walk(routing_query, dist_q_c, &mut scratch).await?
                }
            }
        };

        Ok(Probing {
            segment: probe.segment,
            file,
            dimension: probe.dimension,
            candidates,
            comparisons,
        })
    }

    /// Measure the query exactly against the candidates one probe still has, and
    /// turn them into that partition's share of the answer.
    ///
    /// The second half of a lazy probe, split off from the first because
    /// [`allocate`] sits between them. A probe whose candidates were all taken
    /// by better ones elsewhere reads nothing at all - that is where the saving
    /// is, and it is a whole request rather than a fraction of one.
    async fn rescore_probing(
        &self,
        probing: Probing,
        query: ArrayRef,
        params: &SearchParams,
        stats: &IoStats,
    ) -> Result<Walked> {
        if probing.candidates.is_empty() {
            return Ok(Walked {
                neighbors: Vec::new(),
                comparisons: 0,
            });
        }
        let rescored = if params.rescore_from_dataset
            || self.metadata.vector_source == VectorSource::Dataset
        {
            // Admitted before the read rather than only after it, as `answer`
            // does: a dead row can sit in a fragment the dataset no longer has,
            // where there is nothing to read, and the answer drops it either way.
            // The survivors keep the address the segment stored, which `answer`
            // admits again and rewrites.
            let (candidates, rows): (Vec<Candidate>, Vec<u64>) = probing
                .candidates
                .iter()
                .filter_map(|candidate| {
                    self.rows
                        .admit(probing.segment, candidate.row_addr)
                        .map(|current| (*candidate, current))
                })
                .unzip();
            if candidates.is_empty() {
                return Ok(Walked {
                    neighbors: Vec::new(),
                    comparisons: 0,
                });
            }
            let values = self
                .dataset_vectors
                .fetch(&rows, stats, &self.file_access())
                .await?;
            lazy::measure(&candidates, &values, self.metadata.distance_type, query)?
        } else {
            lazy::rescore(
                &probing.file,
                probing.dimension,
                self.metadata.distance_type,
                &probing.candidates,
                query,
                stats,
            )
            .await?
        };
        let comparisons = rescored.len() as u64;
        Ok(Walked {
            neighbors: answer(rescored, &self.rows, probing.segment, params.k)?,
            comparisons,
        })
    }

    /// Read one probed partition whole.
    ///
    /// Projected on the columns the walk will use, so that an index carrying
    /// codes does not pay for them on a query that measures against the vectors:
    /// thirteen per cent of a partition at `d = 128`.
    async fn read_probe(&self, probe: Probe, stats: &IoStats) -> Result<Probed> {
        let mut columns = vec![ROW_ID_COLUMN, NEIGHBORS_COLUMN, VECTOR_COLUMN];
        if probe.dist_q_c.is_some() {
            columns.push(CODE_COLUMN);
        }
        // Through the cache for the file's layout, same as the lazy walk, and
        // for the same reason: the footer is a round trip whatever is read
        // afterwards. What it does *not* take from the cache is the codes, which
        // arrive in this read along with everything else.
        let file = self.partition_file(&probe, stats).await?;
        let reader = file.project(&columns).await?;
        let batch = read_partition_batch(&reader, probe.entry.num_rows).await?;
        let partition = Partition::try_from_batch(&batch)?;
        check_partition_shape(&partition, &probe.entry, probe.max_degree, probe.dimension)?;
        let coded = probe
            .dist_q_c
            .map(|dist_q_c| Ok::<_, Error>((codes::column(&batch)?, dist_q_c)))
            .transpose()?;
        Ok(Probed {
            segment: probe.segment,
            partition,
            medoid: probe.entry.medoid,
            coded,
        })
    }

    /// How this index opens a dataset's data file: on the terms it opens its
    /// own partition files.
    fn file_access(&self) -> FileAccess<'_> {
        FileAccess {
            scheduler: &self.scheduler,
            cache: self.cache.as_ref(),
            local_reads: !self.is_maintenance,
            store: &self.store,
            direct: &self.direct,
        }
    }

    /// One partition of `segment` read whole, for a pass that rewrites it.
    ///
    /// A segment that keeps its vectors reads them off its own file. One that
    /// leaves them to the dataset reads its graph off its file and the vectors
    /// of `vertices` out of the dataset: a live vertex's at the address its row
    /// answers under now, a dead one's at the address it was stored under. A
    /// dead vertex whose vector is not wanted gets zeros, which nothing reads -
    /// a pass that asks for [`Vertices::Live`] takes the dead out before it
    /// measures anything.
    pub(crate) async fn read_partition_whole(
        &self,
        segment: &Segment,
        entry: &PartitionEntry,
        vertices: Vertices,
    ) -> Result<Partition> {
        let path = segment.dir.clone().join(entry.file.as_str());
        let size_bytes = segment.file_sizes.get(&entry.file).copied();
        match segment.manifest.metadata().vector_source {
            VectorSource::Index => {
                let reader = open_file(&self.scheduler, &path, None, size_bytes).await?;
                read_partition(&reader, entry.num_rows).await
            }
            VectorSource::Dataset => {
                let columns = [ROW_ID_COLUMN, NEIGHBORS_COLUMN];
                let reader = open_file(&self.scheduler, &path, Some(&columns), size_bytes).await?;
                let graph =
                    graph_from_batch(&read_partition_batch(&reader, entry.num_rows).await?)?;
                let vectors = self
                    .vectors_from_dataset(segment, graph.row_ids(), vertices, &self.file_access())
                    .await?;
                Partition::try_new(graph, vectors)
            }
        }
    }

    /// One partition's row ids and vectors, in local-id order, for training its
    /// entry points.
    ///
    /// Of the partition file only `__row_id` and, where the segment keeps them,
    /// `__vector`: the codes and the edges are most of the rest of the file and
    /// training needs neither. A segment that leaves its vectors to the dataset
    /// has the live vertices' read out of it the way a pass that rewrites a
    /// partition reads them - through the scheduler, which reads thousands of
    /// scattered rows several at a time, with each data file's layout kept in
    /// `layouts` rather than in this index's cache - and gives a dead vertex
    /// zeros, which training never reads.
    async fn partition_vectors(
        &self,
        segment: &Segment,
        entry: &PartitionEntry,
        layouts: &LanceCache,
    ) -> Result<(Vec<u64>, FixedSizeListArray)> {
        let path = segment.dir.clone().join(entry.file.as_str());
        let size_bytes = segment.file_sizes.get(&entry.file).copied();
        let metadata = segment.manifest.metadata();
        match metadata.vector_source {
            VectorSource::Index => {
                let columns = [ROW_ID_COLUMN, VECTOR_COLUMN];
                let reader = open_file(&self.scheduler, &path, Some(&columns), size_bytes).await?;
                let batch = read_partition_batch(&reader, entry.num_rows).await?;
                Ok((
                    row_ids_from_batch(&batch)?,
                    vectors_of(&batch, metadata.dimension)?,
                ))
            }
            VectorSource::Dataset => {
                let columns = [ROW_ID_COLUMN];
                let reader = open_file(&self.scheduler, &path, Some(&columns), size_bytes).await?;
                let row_ids =
                    row_ids_from_batch(&read_partition_batch(&reader, entry.num_rows).await?)?;
                let access = FileAccess {
                    scheduler: &self.scheduler,
                    cache: Some(layouts),
                    local_reads: false,
                    store: &self.store,
                    direct: &self.direct,
                };
                let vectors = self
                    .vectors_from_dataset(segment, &row_ids, Vertices::Live, &access)
                    .await?;
                Ok((row_ids, vectors))
            }
        }
    }

    /// One partition of `segment` as its file holds it, and the address each of
    /// its vertices answers under now, for a pass that writes it out again with
    /// nothing changed but where its rows live
    /// ([`SegmentWriter::write_readdressed`](crate::io::SegmentWriter::write_readdressed)).
    ///
    /// Every column the file has is read as one batch, no graph built of it and
    /// no code taken again, the vectors included where the segment keeps them,
    /// since the new file keeps them too; a segment that leaves them to the
    /// dataset reads nothing of it.
    /// Only for a partition none of whose vertices is dead, so every one of
    /// them answers under some address: one that answers under none is an
    /// error rather than a vertex written back where it was stored.
    pub(crate) async fn read_readdressed(
        &self,
        segment: &Segment,
        entry: &PartitionEntry,
    ) -> Result<(RecordBatch, Vec<u64>)> {
        let path = segment.dir.clone().join(entry.file.as_str());
        let size_bytes = segment.file_sizes.get(&entry.file).copied();
        let reader = open_file(&self.scheduler, &path, None, size_bytes).await?;
        let stored = read_partition_batch(&reader, entry.num_rows).await?;
        let row_ids = row_ids_from_batch(&stored)?
            .into_iter()
            .map(|stored_at| {
                self.rows.admit(segment.uuid, stored_at).ok_or_else(|| {
                    Error::internal(format!(
                        "Vamana partition {} of segment {} was to be written out again with \
                         nothing deleted from it, and its vertex at {stored_at} answers under no \
                         address",
                        entry.partition_id, segment.uuid
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((stored, row_ids))
    }

    /// The vectors of one partition's vertices, in local-id order, out of the
    /// dataset, read on `access`'s terms.
    async fn vectors_from_dataset(
        &self,
        segment: &Segment,
        row_ids: &[u64],
        vertices: Vertices,
        access: &FileAccess<'_>,
    ) -> Result<FixedSizeListArray> {
        let (mut live, mut live_at) = (Vec::new(), Vec::new());
        let (mut dead, mut dead_at) = (Vec::new(), Vec::new());
        for (local_id, &stored) in row_ids.iter().enumerate() {
            match self.rows.admit(segment.uuid, stored) {
                Some(current) => {
                    live.push(local_id);
                    live_at.push(current);
                }
                None => {
                    dead.push(local_id);
                    dead_at.push(stored);
                }
            }
        }
        let stats = IoStats::new();
        let read = self.dataset_vectors.fetch(&live_at, &stats, access).await?;
        if dead.is_empty() {
            return Ok(read);
        }

        let width = self.metadata.dimension as usize;
        let mut values = vec![0.0f32; row_ids.len() * width];
        place(&mut values, &live, &read, width)?;
        if vertices == Vertices::All {
            let read = self
                .dataset_vectors
                .fetch_deleted(&dead_at, &stats, access)
                .await?;
            place(&mut values, &dead, &read, width)?;
        }
        Ok(FixedSizeListArray::try_new_from_values(
            Float32Array::from(values),
            width as i32,
        )?)
    }
}

/// Whose vectors a pass that rewrites a partition measures against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vertices {
    /// The live vertices: consolidation and merging take the dead ones out
    /// before they measure anything.
    Live,
    /// Every vertex: an insertion walks through the dead ones too.
    All,
}

/// Copy the `n`th vector of `read` into the slot of the `n`th of `local_ids`.
fn place(
    values: &mut [f32],
    local_ids: &[usize],
    read: &FixedSizeListArray,
    width: usize,
) -> Result<()> {
    let read = read.values().as_primitive::<Float32Type>().values();
    if read.len() != local_ids.len() * width {
        return Err(Error::internal(format!(
            "the dataset gave {} values for {} vectors of width {width}",
            read.len(),
            local_ids.len()
        )));
    }
    for (vector, &local_id) in read.chunks_exact(width).zip(local_ids) {
        values[local_id * width..(local_id + 1) * width].copy_from_slice(vector);
    }
    Ok(())
}

/// Turn one partition's re-scored candidates into its share of the answer.
///
/// Shared by every mode, and the reason they all return the same shape: what
/// separates them is how a candidate list is arrived at, and nothing after that
/// may differ. `candidates` is nearest first by an *exact* distance whichever
/// walk produced it, which is what the merge downstream rests on.
fn answer(
    candidates: Vec<Neighbor>,
    rows: &RowFilter,
    segment: Uuid,
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
    if let Some(candidate) = candidates.iter().find(|c| !c.distance.is_finite()) {
        return Err(Error::corrupt_file_named(
            "partition",
            format!(
                "Vamana row {} is at distance {} from a finite query, so the vector it was \
                 measured against - the partition's copy, or the dataset's - is not finite",
                candidate.row_addr, candidate.distance,
            ),
        ));
    }
    // Dead vertices are dropped here and not earlier. They are still walked,
    // because they carry the out-edges that keep the graph connected - removing
    // them from the traversal would strand whatever they were the only route to.
    // Filtering before `take` rather than after is what makes `k` mean "k live
    // rows" instead of "k rows, some of which the caller will find missing".
    Ok(candidates
        .into_iter()
        .filter_map(|neighbor| {
            rows.admit(segment, neighbor.row_addr)
                .map(|row_addr| Neighbor {
                    row_addr,
                    ..neighbor
                })
        })
        .take(k)
        .collect())
}

/// The `k` a query would have answered with had it stopped before the re-score.
///
/// The same answer arrived at the same way - live rows only, each row once,
/// nearest first - over the *coded* distances the walk steered by instead of
/// over exact ones. What it is for is the question a final recall number cannot
/// answer on its own: whether reading the vectors bought anything the codes had
/// not already found.
///
/// It borrows the probes rather than consuming them, because they are about to
/// be re-scored. This is a second answer taken from the same candidates, not a
/// diversion of them, and it costs a sort of at most
/// [`SearchParams::rescore_budget`] elements - which is why
/// [`SearchParams::report_coded`] gates it rather than the cost fields beside
/// it.
fn coded_answer(probings: &[Probing], rows: &RowFilter, k: usize) -> Vec<Neighbor> {
    let candidates = probings
        .iter()
        .flat_map(|probing| {
            probing
                .candidates
                .iter()
                .map(move |candidate| (probing.segment, candidate))
        })
        .filter_map(|(segment, candidate)| {
            rows.admit(segment, candidate.row_addr)
                .map(|row_addr| Neighbor {
                    row_addr,
                    distance: candidate.coded,
                })
        })
        .collect::<Vec<_>>();
    merge(candidates, k)
}

/// Refuse `index` when a deferred compaction that moved its rows had an overlay
/// to bake into them.
///
/// A compaction writes a fragment's overlays into the base data of the fragment
/// it writes, which then carries none, and on a deferred compaction Lance's own
/// pruning of stale coverage runs before the move reaches any bitmap and prunes
/// nothing. So the fragments the segment's rows moved out of are checked as the
/// compaction read them, at the dataset version it recorded - the same history
/// read `prune_stale_segment_coverage` makes when an index is committed.
///
/// `history` holds the fragments of each version read so far, since every
/// segment an index had before a compaction was moved by it.
async fn refuse_overlays_moved_with_rows(
    dataset: &Dataset,
    index_name: &str,
    index: &lance_table::format::IndexMetadata,
    stored_over: &RoaringBitmap,
    remap: &CompactFragReuseIndex,
    history: &mut HashMap<u64, HashMap<u32, Fragment>>,
) -> Result<()> {
    let mut moved_through = stored_over.clone();
    for version in &remap.details.versions {
        for group in &version.groups {
            let moved_out = group
                .old_frags
                .iter()
                .map(|fragment| fragment.id as u32)
                .filter(|fragment_id| moved_through.contains(*fragment_id))
                .collect::<Vec<_>>();
            if moved_out.is_empty() {
                continue;
            }
            let fragments = match history.entry(version.dataset_version) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let read = dataset
                        .checkout_version(version.dataset_version)
                        .await
                        .map_err(|error| {
                            Error::index(format!(
                                "index '{index_name}' segment {} stores rows a deferred compaction \
                                 moved, and whether that compaction baked an overlay into them can \
                                 only be read from dataset version {}, which cannot be opened: \
                                 {error}; rebuild the index",
                                index.uuid, version.dataset_version
                            ))
                        })?;
                    entry.insert(
                        read.get_fragments()
                            .iter()
                            .map(|fragment| (fragment.id() as u32, fragment.metadata().clone()))
                            .collect(),
                    )
                }
            };
            if let Some(fragment_id) = moved_out.into_iter().find(|fragment_id| {
                fragments.get(fragment_id).is_some_and(|fragment| {
                    overlay_supersedes_segment(
                        &fragment.overlays,
                        &index.fields,
                        index.dataset_version,
                        dataset.schema(),
                    )
                })
            }) {
                return Err(Error::index(format!(
                    "index '{index_name}' segment {} was built at dataset version {} and fragment \
                     {fragment_id} had its indexed values replaced by an overlay before a \
                     compaction at version {} moved its rows, so the vectors it ranks are not the \
                     ones the rows now hold; rebuild the index",
                    index.uuid, index.dataset_version, version.dataset_version
                )));
            }
            moved_through.extend(group.new_frags.iter().map(|fragment| fragment.id as u32));
        }
    }
    Ok(())
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
pub(crate) fn descends_from(schema: &Schema, field: i32, ancestor: i32) -> bool {
    schema
        .field_ancestry_by_id(field)
        .is_some_and(|ancestry| ancestry.iter().any(|step| step.id == ancestor))
}

/// Spend a query's exact distances on its `budget` most promising candidates,
/// wherever they were probed from.
///
/// A candidate costs a stride of `__vector` to correct, and correcting them is
/// what a lazy query's bytes are. Each probe arrives having chosen the best `L`
/// its own partition could offer, which is the wrong question: a query is one
/// answer, and the partition nearest it usually deserves several probes' worth
/// of the budget while the farthest deserves none.
///
/// Ties break on the row address rather than on which probe raised the
/// candidate. The probes complete in no fixed order - that is what
/// `buffer_unordered` is for - so a tie broken by arrival order would let the
/// same query on the same index return different rows from one run to the next.
///
/// What comes back is still ascending by local id within each probe, which is
/// the order the reader coalesces by and the order the candidates arrived in.
fn allocate(probings: &mut [Probing], budget: usize) {
    let total = probings
        .iter()
        .map(|probing| probing.candidates.len())
        .sum::<usize>();
    if total <= budget {
        return;
    }

    let mut ranked = probings
        .iter()
        .enumerate()
        .flat_map(|(probe, probing)| {
            probing
                .candidates
                .iter()
                .enumerate()
                .map(move |(position, candidate)| {
                    (candidate.coded, candidate.row_addr, probe, position)
                })
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
    ranked.truncate(budget);

    let mut kept = vec![Vec::new(); probings.len()];
    for (_, _, probe, position) in ranked {
        kept[probe].push(position);
    }
    for (probing, mut positions) in probings.iter_mut().zip(kept) {
        positions.sort_unstable();
        let candidates = positions
            .into_iter()
            .map(|position| probing.candidates[position])
            .collect::<Vec<_>>();
        probing.candidates = candidates;
    }
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

    use arrow_array::RecordBatchIterator;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance::dataset::WriteParams;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_file::version::ConcreteFileVersion;
    use lance_table::format::DataFile;
    use lance_table::format::overlay::OverlayCoverage;
    use lance_table::system_index::frag_reuse::{
        FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
    };

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::builder::{IndexParams, create_index};
    use crate::codes::CodeSpec;
    use crate::partition::PartitionGraph;

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

    /// The rule a margin stands for: the bar hangs off the `k`-th candidate,
    /// and a walk keeps what the re-score is owed - the budget's worth, or
    /// everything its list holds without one.
    ///
    /// Pinned here because nothing downstream can tell the numbers apart: a
    /// walk that measured the bar off the budget-th, or kept only `k`, would
    /// still answer, just from a different stopping point.
    #[test]
    fn a_margin_hangs_off_k_and_keeps_what_the_re_score_is_owed() {
        let params = SearchParams::new(10).with_search_list_size(300);
        assert_eq!(params.stop_rule(), None);
        assert_eq!(
            params.clone().with_stop_margin(0.25).stop_rule(),
            Some(StopRule {
                margin: 0.25,
                rank: 10,
                keep: 300
            })
        );
        assert_eq!(
            params
                .with_rescore_budget(20)
                .with_stop_margin(0.25)
                .stop_rule(),
            Some(StopRule {
                margin: 0.25,
                rank: 10,
                keep: 20
            })
        );
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

    fn address(fragment_id: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(fragment_id, offset))
    }

    /// One compaction that moved rows three ways: fragment 1, four rows with the
    /// second one deleted, into fragment 10; and fragments 2 and 3, two rows
    /// each, into fragment 11.
    fn one_compaction() -> Arc<CompactFragReuseIndex> {
        let changed = |addresses: &[u64]| {
            let changed = addresses.iter().copied().collect::<RoaringTreemap>();
            let mut bytes = Vec::with_capacity(changed.serialized_size());
            changed.serialize_into(&mut bytes).unwrap();
            bytes
        };
        let digest = |id, physical_rows| FragDigest {
            id,
            physical_rows,
            num_deleted_rows: 0,
        };
        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 1,
                groups: vec![
                    FragReuseGroup {
                        changed_row_addrs: changed(&[address(1, 0), address(1, 2), address(1, 3)]),
                        old_frags: vec![digest(1, 4)],
                        new_frags: vec![digest(10, 3)],
                    },
                    FragReuseGroup {
                        changed_row_addrs: changed(&[
                            address(2, 0),
                            address(2, 1),
                            address(3, 0),
                            address(3, 1),
                        ]),
                        old_frags: vec![digest(2, 2), digest(3, 2)],
                        new_frags: vec![digest(11, 4)],
                    },
                ],
            }],
        };
        Arc::new(CompactFragReuseIndex::try_new(Uuid::new_v4(), details).unwrap())
    }

    /// Of the second group the segment covered only fragment 2, so Lance credits
    /// it with neither fragment of that group afterwards, and the row that moved
    /// from 2 into 11 is left to whoever covers 11 - here a delta indexed over it,
    /// which stores that row under its own address.
    #[test]
    fn a_moved_row_answers_where_it_landed() {
        let remap = one_compaction();
        let mut base_coverage = RoaringBitmap::from_iter([1u32, 2, 4]);
        remap.remap_fragment_bitmap(&mut base_coverage).unwrap();
        assert_eq!(base_coverage, RoaringBitmap::from_iter([4u32, 10]));
        let (base, delta) = (Uuid::new_v4(), Uuid::new_v4());
        let rows = RowFilter {
            deleted: RoaringTreemap::from_iter([address(10, 0), address(4, 1)]),
            missing_fragments: RoaringBitmap::from_iter([1u32, 2]),
            moved: Some(MovedRows {
                remap,
                coverage: HashMap::from([
                    (base, base_coverage),
                    (delta, RoaringBitmap::from_iter([11u32])),
                ]),
            }),
        };
        for (segment, stored, expected, what) in [
            (
                base,
                address(1, 0),
                None,
                "moved onto an address deleted since",
            ),
            (base, address(1, 1), None, "dropped by the compaction"),
            (base, address(1, 2), Some(address(10, 1)), "moved"),
            (
                base,
                address(2, 1),
                None,
                "moved where another segment answers",
            ),
            (
                delta,
                address(11, 1),
                Some(address(11, 1)),
                "stored by that segment",
            ),
            (base, address(4, 0), Some(address(4, 0)), "never moved"),
            (base, address(4, 1), None, "deleted where it was"),
        ] {
            assert_eq!(rows.admit(segment, stored), expected, "{what}");
        }

        let partition = Partition::try_new(
            PartitionGraph::try_new(
                4,
                vec![address(1, 1), address(1, 2), address(4, 0)],
                vec![vec![1], vec![2], vec![0]],
            )
            .unwrap(),
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                2,
                Arc::new(Float32Array::from(vec![0.0; 6])),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            rows.readdress(base, partition).unwrap().graph().row_ids(),
            [address(1, 1), address(10, 1), address(4, 0)]
        );
    }

    /// A pass that rewrites partitions reads where a data file keeps its
    /// vectors once, however many partitions read from the file: an index
    /// opened for it has a cache of its own, which every data file's layout is
    /// loaded into once - and is not looked up again, since the file itself
    /// stays open for the partitions after.
    #[tokio::test]
    async fn a_pass_lays_out_each_data_file_once_for_all_its_partitions() {
        const FRAGMENTS: usize = 4;
        const ROWS: usize = 400;
        const WIDTH: i32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let mut rng = SmallRng::seed_from_u64(5);
        let values = (0..ROWS * WIDTH as usize)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_from_iter(vec![(
            "vec",
            Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), WIDTH).unwrap(),
            ) as ArrayRef,
        )])
        .unwrap();
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir.path().to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: ROWS / FRAGMENTS,
                max_rows_per_group: ROWS / FRAGMENTS,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), FRAGMENTS);
        create_index(
            &mut dataset,
            "vamana_idx",
            &IndexParams::new("vec", 2)
                .with_codes(CodeSpec::Scalar { num_bits: 8 })
                .with_vector_source(VectorSource::Dataset),
        )
        .await
        .unwrap();

        let index = VamanaIndex::open_for_maintenance(&dataset, "vamana_idx")
            .await
            .unwrap();
        let mut partitions = 0;
        for segment in index.segments() {
            for entry in segment.manifest.partitions() {
                let partition = index
                    .read_partition_whole(segment, entry, Vertices::Live)
                    .await
                    .unwrap();
                assert_eq!(partition.len(), entry.num_rows as usize);
                partitions += 1;
            }
        }
        assert!(
            partitions > 1,
            "one partition reads each file once whatever the pass does, so this tests nothing"
        );
        let stats = index
            .cache_stats()
            .await
            .expect("an index opened for a pass was given no cache");
        assert_eq!(
            (stats.hits, stats.misses, stats.num_entries),
            (0, FRAGMENTS as u64, FRAGMENTS),
            "{stats:?}"
        );
    }

    /// Rows and the bits of their distances, so that two answers are held equal
    /// to the last bit rather than to `f32`'s equality.
    fn bits(neighbors: &[Neighbor]) -> Vec<(u64, u32)> {
        neighbors
            .iter()
            .map(|neighbor| (neighbor.row_addr, neighbor.distance.to_bits()))
            .collect()
    }

    /// A walk whose only entry point is its partition's medoid is the walk from
    /// the medoid, to the last bit.
    ///
    /// One entry point costs the one distance the medoid does and can only be
    /// chosen, so the entry-point start has to come out as the medoid start
    /// does: a start offered at another distance, or charged otherwise, or the
    /// entry points of another partition - the two partitions' medoids differ -
    /// would each show up here as an inequality, where a recall bar would call
    /// it noise. What one entry point cannot show - the marks and the charge of
    /// the ones not chosen - is `tests/lazy_index.rs`'s to pin. Both kinds of
    /// code, because each walks another store, and with and without the stop
    /// margin, because the margin's list is the one that drops what it is
    /// offered.
    #[tokio::test]
    async fn a_walk_from_the_medoid_as_its_only_entry_point_is_the_medoid_walk() {
        const ROWS: usize = 1200;
        const WIDTH: i32 = 16;
        let mut rng = SmallRng::seed_from_u64(9);
        let values = (0..ROWS * WIDTH as usize)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let queries = (0..24)
            .map(|_| (0..WIDTH).map(|_| rng.random::<f32>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for codes in [
            CodeSpec::Scalar { num_bits: 8 },
            CodeSpec::Rabit { num_bits: 3 },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let batch = RecordBatch::try_from_iter(vec![(
                "vec",
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(values.clone()),
                        WIDTH,
                    )
                    .unwrap(),
                ) as ArrayRef,
            )])
            .unwrap();
            let schema = batch.schema();
            let mut dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                dir.path().to_str().unwrap(),
                None,
            )
            .await
            .unwrap();
            create_index(
                &mut dataset,
                "vamana_idx",
                &IndexParams::new("vec", 2).with_codes(codes),
            )
            .await
            .unwrap();

            let plain = VamanaIndex::open(&dataset, "vamana_idx").await.unwrap();
            let medoids = plain
                .segments()
                .iter()
                .flat_map(|segment| {
                    segment
                        .manifest
                        .partitions()
                        .iter()
                        .map(|entry| PartitionEntryPoints {
                            segment: segment.uuid,
                            partition_id: entry.partition_id,
                            num_rows: entry.num_rows,
                            entries: vec![entry.medoid],
                        })
                })
                .collect::<Vec<_>>();
            assert!(
                medoids.len() > 1
                    && medoids.iter().all(|one| medoids
                        .iter()
                        .filter(|other| other.entries == one.entries)
                        .count()
                        == 1),
                "a start taken from another partition shows only where the medoids differ: {medoids:?}"
            );
            let entry_points =
                EntryPoints::from_partitions(EntryPointParams::new(1), medoids).unwrap();
            let given = VamanaIndex::open(&dataset, "vamana_idx")
                .await
                .unwrap()
                .with_entry_points(Arc::new(entry_points))
                .unwrap();

            let walk = SearchParams::new(10)
                .with_nprobes(2)
                .with_search_list_size(30)
                .with_mode(WalkMode::Lazy)
                .with_report_coded(true);
            let margin = walk.clone().with_rescore_budget(20).with_stop_margin(0.05);
            for params in [walk, margin] {
                for (n, query) in queries.iter().enumerate() {
                    let medoid = plain.search(query, &params).await.unwrap();
                    let entry = given
                        .search(query, &params.clone().with_start(WalkStart::NearestEntry))
                        .await
                        .unwrap();
                    let what = format!("{codes:?}, margin {:?}, query {n}", params.stop_margin);
                    assert_eq!(bits(&entry.neighbors), bits(&medoid.neighbors), "{what}");
                    assert_eq!(
                        bits(&entry.coded_neighbors),
                        bits(&medoid.coded_neighbors),
                        "{what}"
                    );
                    assert_eq!(entry.comparisons, medoid.comparisons, "{what}");
                    assert_eq!(entry.partitions_read, medoid.partitions_read, "{what}");
                }
            }
        }
    }
}
