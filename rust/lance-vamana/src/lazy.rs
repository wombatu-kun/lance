// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Answering from a partition without reading it.
//!
//! The whole-partition walk reads `__row_id`, `__neighbors` and `__vector` for
//! every vertex and then measures against a few hundred of them. This one keeps
//! resident only what it measures *with* - the row ids and the codes, seventy
//! bytes a vertex at `d = 128` against seven hundred and seventy-six - and
//! fetches the rest as it turns out to need it: the out-edges of a vertex when it
//! expands it, and the vectors of the candidate list once there is one to
//! re-score.
//!
//! Two ways of arriving at that candidate list live here, and which of them is
//! right is a property of the deployment rather than of the index.
//! [`LazyProbe::walk`] steers by the graph; [`LazyProbe::scan`] ignores it and
//! scores every vertex of the partition instead - ten to fifty times the
//! distances, and strictly fewer bytes, because it opens no `__neighbors` at
//! all. Measured at equal recall (`examples/lazy_walk.rs`), the scan reads 0.43
//! of what the walk reads and makes a tenth of the round trips at both
//! granularities, reaches higher recall at every beam, and is 3.0x quicker at
//! 8192 rows a partition and 1.8x at 65536.
//!
//! It was not quicker at 65536 until the scan stopped asking for distances and
//! started asking for a top-`L`. A scanned vertex costs 16 ns when every one of
//! them is refined from its extra bits and about 2 when RaBitQ's error bound is
//! allowed to throw most of the refinements out, and the 14.8 ns between those
//! came off two granularities alike, which is what a per-vertex saving should
//! do. So the graph's own claim is narrower than it looked: it buys a partition
//! large enough for two nanoseconds a vertex to add up, which on SIFT1M is
//! several times the coarsest granularity here.
//!
//! Three measurements decide the shape, and none of them is obvious:
//!
//! - **Codes are not optional here.** A walk expands a few dozen vertices and
//!   measures a distance against twenty-five to forty times as many, because each
//!   expanded vertex hands it `R` neighbours to score. Fetching a vector for each
//!   of those halves the pages moved at best and costs *more* CPU than reading
//!   the partition whole, so the lazy read only pays with the vectors replaced by
//!   something resident (`examples/memory_gate.rs`).
//! - **A hop is batched, a vertex at a time is not.** One request for 256
//!   scattered vertices measured half the iops and half the bytes of 256
//!   requests, and the chain of dependent round trips divides by the width. So
//!   the walk takes the `beam_width` nearest unexpanded candidates at once rather
//!   than the single nearest (`examples/lazy_read_probe.rs`).
//! - **It must not read the vector of a vertex it expands**, tempting as that is
//!   with a request already in flight for that row. Correcting a distance seats
//!   that vertex at the back of the search list, the back of the list is the bar
//!   the next candidate has to clear, and so the walk expands more - eight per
//!   cent more at three bits, three times more at one (`examples/coded_walk.rs`).
//!
//! What is *not* here is the cache, and the difference between the two is what
//! this module is bounded by. Everything fetched here is chosen by the query -
//! which vertices this walk expanded, which candidates it ended with - and is
//! therefore nobody's to keep. What is worth keeping is what the walk needed
//! before it could start, and [`crate::cache`] keeps it: a query that has probed
//! a partition before fetches 71.9 kB on SIFT1M where one that has not fetches
//! 18.2 MB, and the 18.1 MB between them is the codes and the row ids, read once
//! rather than by every query.

use std::collections::BinaryHeap;

use arrow_array::ArrayRef;
use lance_core::{Error, Result};
use lance_index::vector::bq::storage::RabitQuantizationStorage;
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;

use crate::format::{NEIGHBORS_COLUMN, VECTOR_COLUMN};
use crate::io::{PartitionFile, read_scattered};
use crate::partition::{checked_neighbors, neighbor_slots, vectors_of};
use crate::search::{Comparisons, SearchList, SearchScratch, flat_storage};

/// One partition, and everything answering from it lazily needs but the query.
///
/// The codes and the row ids are already in hand - they are the one thing read
/// whole here, so reading them is the caller's business and reading them is also
/// what tells it how many vertices there are. Everything else is a number off
/// the segment, held here so that what comes back off disk can be checked
/// against what the segment claims: nothing on this path holds a whole
/// [`crate::Partition`], so nothing else is in a position to notice a partition
/// file whose stride disagrees with the segment listing it.
///
/// [`Self::scan`] leaves `medoid`, `max_degree` and `beam_width` unread: they
/// describe the graph, and a scan is the arm that does not have one.
pub(crate) struct LazyProbe<'a> {
    pub(crate) file: &'a PartitionFile,
    pub(crate) codes: &'a RabitQuantizationStorage,
    pub(crate) row_ids: &'a [u64],
    pub(crate) medoid: u32,
    pub(crate) max_degree: u32,
    pub(crate) dimension: u32,
    pub(crate) distance_type: DistanceType,
    pub(crate) search_list_size: usize,
    /// How many vertices one hop expands, and therefore how many rows of
    /// `__neighbors` one request asks for.
    pub(crate) beam_width: usize,
}

impl LazyProbe<'_> {
    /// Walk, then re-score the candidate list exactly.
    ///
    /// Returns the list as local ids and *exact* distances, nearest first, which
    /// is the same thing the whole-partition walks return and is what lets every
    /// mode share the step that turns a candidate list into an answer.
    ///
    /// `routing_query` and `dist_q_c` are what a coded distance is assembled
    /// from; `query` is the raw one the re-scoring measures against. The two are
    /// the same array for every metric but cosine, and passing the wrong one is
    /// silent, which is why they arrive named.
    ///
    /// Nothing here is handed to the CPU pool, unlike the walk over a partition
    /// held in memory. It cannot be: the pool takes work that never waits, and
    /// this waits once a hop. What it does instead is stay small between the
    /// waits - a hop is `beam_width * R` coded distances, tens of microseconds,
    /// which is under the size at which handing work to the pool starts to pay.
    pub(crate) async fn walk(
        &self,
        routing_query: ArrayRef,
        dist_q_c: f32,
        query: ArrayRef,
    ) -> Result<(Vec<(u32, f32)>, u64)> {
        let num_rows = self.row_ids.len();
        if self.medoid as usize >= num_rows {
            return Err(Error::corrupt_file_named(
                "partition",
                format!(
                    "Vamana entry point {} is outside a partition of {num_rows} vertices",
                    self.medoid
                ),
            ));
        }
        if self.beam_width == 0 {
            return Err(Error::invalid_input(
                "Vamana beam width must be greater than zero".to_string(),
            ));
        }

        let comparisons = Comparisons::default();
        let coded = self.codes.dist_calculator(routing_query, dist_q_c);
        let mut scratch = SearchScratch::new(num_rows);
        let mut list = SearchList::new(self.search_list_size, num_rows);
        scratch.begin();
        scratch.mark(self.medoid);
        comparisons.record(1);
        list.offer(self.medoid, coded.distance(self.medoid));

        let width = self.max_degree as usize;
        let edges = self.file.project(&[NEIGHBORS_COLUMN]).await?;
        let mut frontier = Vec::with_capacity(self.beam_width);
        loop {
            frontier.clear();
            while frontier.len() < self.beam_width {
                let Some(node) = list.next_unexpanded() else {
                    break;
                };
                frontier.push(node.id);
            }
            if frontier.is_empty() {
                break;
            }
            // By id, because that is what the reader coalesces by. It also makes
            // the walk deterministic: whose neighbours are offered first decides
            // which of two equally distant candidates survives a full list.
            frontier.sort_unstable();

            let batch = read_scattered(&edges, &frontier).await?;
            let slots = neighbor_slots(&batch, self.max_degree)?;
            for (position, vertex) in frontier.iter().enumerate() {
                let start = position * width;
                let out_edges = checked_neighbors(&slots[start..start + width], *vertex, num_rows)?;
                for neighbor in out_edges {
                    if !scratch.mark(*neighbor) {
                        continue;
                    }
                    comparisons.record(1);
                    list.offer(*neighbor, coded.distance(*neighbor));
                }
            }
        }

        let mut candidates = list
            .into_candidates()
            .into_iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok((Vec::new(), comparisons.get()));
        }
        candidates.sort_unstable();
        let rescored = self.rescore(&candidates, query).await?;
        comparisons.record(rescored.len() as u64);
        Ok((rescored, comparisons.get()))
    }

    /// Score every vertex of the partition, keep the nearest `L`, and re-score
    /// those exactly.
    ///
    /// The graph is never opened: this reads the codes the caller already holds
    /// and then the vectors of the candidate list, and `__neighbors` - 256 bytes
    /// a vertex at `R = 64`, more than a quarter of a partition file - does not
    /// enter into it. What it costs instead is arithmetic, `num_rows` coded
    /// distances against the few hundred a walk measures, which is the one
    /// column where this mode is the expensive one.
    ///
    /// That is why it asks for a top-`L` rather than for the distances. A
    /// multi-bit RaBitQ distance is two passes: a binary inner product every
    /// vertex pays, then an extra-bit refinement that costs several times as
    /// much. The binary pass carries an error bound, so a vertex whose bound is
    /// already worse than the `L`-th best found so far cannot enter the list and
    /// need not be refined, and [`DistCalculator::accumulate_topk_with_scratch`]
    /// classifies sixteen at a time against that bound and refines only the
    /// survivors. Both arms of that branch are the same arithmetic this crate's
    /// codes would get from [`DistCalculator::distance_all`], because the
    /// refinement has a packed bulk form and Lance only builds it for indices
    /// without error factors; our stride carries them. Below two bits there is
    /// no refinement and no bound, and the call degrades to exactly the scan it
    /// replaced.
    ///
    /// Measured, that is 14.8 ns off every vertex scanned - the same figure at
    /// 8192 rows a partition and at 65536, which is what a saving that is per
    /// vertex rather than per query has to look like. It is most of a coded
    /// distance, and it is the whole reason a scan of the coarser granularity
    /// beats a walk of it. Note what it does *not* change: the comparison count
    /// this returns is still one a vertex, because the binary pass is what every
    /// vertex pays and Lance hands back no count of what it refined. For this
    /// mode that column is an upper bound on the arithmetic, and the clock is
    /// where the bound shows.
    ///
    /// The distances it does compute are not the walk's distances at the walk's
    /// price. A batched pass over the quantiser's block layout costs 16.8 ns a
    /// vertex where [`DistCalculator::distance`] one at a time costs 40.0
    /// (`examples/expansion_gate.rs`), and a walk cannot have the cheaper one:
    /// it does not know which vertices it wants until it has scored the ones
    /// before them. So the two arms are not comparable by their distance counts,
    /// only by what they cost and what they read.
    pub(crate) async fn scan(
        &self,
        routing_query: ArrayRef,
        dist_q_c: f32,
        query: ArrayRef,
    ) -> Result<(Vec<(u32, f32)>, u64)> {
        let num_rows = self.row_ids.len();
        let comparisons = Comparisons::default();
        let coded = self.codes.dist_calculator(routing_query, dist_q_c);

        // Bounded by the partition as well as by `L`, like `SearchList::new`
        // and for the same reason: `L` comes from a caller who may have passed
        // `usize::MAX`. The quantiser's four scratch buffers are the caller's
        // to own, and there is nothing here to own them across probes - each
        // arrives with its own partition and its own borrow of the cache.
        let mut nearest: BinaryHeap<OrderedNode<u64>> =
            BinaryHeap::with_capacity(self.search_list_size.min(num_rows));
        let mut dists = Vec::new();
        let mut quantized_dists = Vec::new();
        let mut quantized_dists_table = Vec::new();
        let mut hacc_quantized_dists = Vec::new();
        coded.accumulate_topk_with_scratch(
            self.search_list_size,
            None,
            None,
            u64::from,
            &mut nearest,
            &mut dists,
            &mut quantized_dists,
            &mut quantized_dists_table,
            &mut hacc_quantized_dists,
        );
        debug_assert_eq!(
            dists.len(),
            num_rows,
            "the code storage and the row ids of a partition disagree about its length"
        );
        comparisons.record(dists.len() as u64);

        if nearest.is_empty() {
            return Ok((Vec::new(), comparisons.get()));
        }
        // Local ids, because that is what the closure above put in the heap.
        // Ascending, because that is what the reader coalesces by and a heap
        // yields its contents in no order at all.
        let mut candidates = nearest
            .into_iter()
            .map(|node| node.id as u32)
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        let rescored = self.rescore(&candidates, query).await?;
        comparisons.record(rescored.len() as u64);
        Ok((rescored, comparisons.get()))
    }

    /// Fetch the vectors of a candidate list and put it in exact order.
    ///
    /// Every candidate and not the nearest `k` of them: a coded ordering tops
    /// out around 0.95 recall at any code width, and the rows that make up the
    /// difference are the ones it put behind `k`. One request either way, so the
    /// whole list costs `L - k` strides more.
    ///
    /// `candidates` arrives ascending, which is the order the reader coalesces
    /// by, and comes back reordered by an exact distance.
    ///
    /// The distances it measures are counted by the caller rather than here. A
    /// `&Comparisons` held across the awaits below would make this future
    /// `!Send` - the counter is a `Cell` - and the two callers both have the
    /// length in hand anyway.
    async fn rescore(&self, candidates: &[u32], query: ArrayRef) -> Result<Vec<(u32, f32)>> {
        let vectors = self.file.project(&[VECTOR_COLUMN]).await?;
        let batch = read_scattered(&vectors, candidates).await?;
        let values = vectors_of(&batch, self.dimension)?;
        let row_ids = candidates
            .iter()
            .map(|id| self.row_ids[*id as usize])
            .collect::<Vec<_>>();
        let store = flat_storage(&row_ids, &values, self.distance_type)?;
        let exact = store.dist_calculator(query, 0.0);

        // Positions in what came back, not local ids: the batch holds only the
        // candidates, in the order they were asked for.
        let mut rescored = candidates
            .iter()
            .enumerate()
            .map(|(position, id)| (*id, exact.distance(position as u32)))
            .collect::<Vec<_>>();
        rescored.sort_by(|left, right| left.1.total_cmp(&right.1));
        Ok(rescored)
    }
}
