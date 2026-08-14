// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Walking a partition without reading it.
//!
//! The whole-partition walk reads `__row_id`, `__neighbors` and `__vector` for
//! every vertex and then measures against a few hundred of them. This one keeps
//! resident only what it measures *with* - the row ids and the codes, seventy
//! bytes a vertex at `d = 128` against seven hundred and seventy-six - and
//! fetches the rest as it turns out to need it: the out-edges of a vertex when it
//! expands it, and the vectors of the candidate list once there is one to
//! re-score.
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

use arrow_array::ArrayRef;
use lance_core::{Error, Result};
use lance_index::vector::bq::storage::RabitQuantizationStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;

use crate::format::{NEIGHBORS_COLUMN, VECTOR_COLUMN};
use crate::io::{PartitionFile, read_scattered};
use crate::partition::{checked_neighbors, neighbor_slots, vectors_of};
use crate::search::{Comparisons, SearchList, SearchScratch, flat_storage};

/// One partition, and everything a lazy walk over it needs but the query.
///
/// The codes and the row ids are already in hand - they are the one thing this
/// walk reads whole, so reading them is the caller's business and reading them
/// is also what tells it how many vertices there are. Everything else is a
/// number off the segment, held here so that what comes back off disk can be
/// checked against what the segment claims: a lazy walk never holds a whole
/// [`crate::Partition`], so nothing else is in a position to notice a partition
/// file whose stride disagrees with the segment listing it.
pub(crate) struct LazyWalk<'a> {
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

impl LazyWalk<'_> {
    /// Walk, then re-score the candidate list exactly.
    ///
    /// Returns the list as local ids and *exact* distances, nearest first, which
    /// is the same thing the whole-partition walks return and is what lets the
    /// three modes share the step that turns a candidate list into an answer.
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
    pub(crate) async fn run(
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
        // Every candidate and not the nearest `k` of them: a coded walk's own
        // ordering tops out around 0.95 recall at any code width, and the rows
        // that make up the difference are the ones it put behind `k`. One
        // request either way, so the whole list costs `L - k` strides more.
        candidates.sort_unstable();
        let vectors = self.file.project(&[VECTOR_COLUMN]).await?;
        let batch = read_scattered(&vectors, &candidates).await?;
        let values = vectors_of(&batch, self.dimension)?;
        let row_ids = candidates
            .iter()
            .map(|id| self.row_ids[*id as usize])
            .collect::<Vec<_>>();
        let store = flat_storage(&row_ids, &values, self.distance_type)?;
        let exact = store.dist_calculator(query, 0.0);
        comparisons.record(candidates.len() as u64);

        // Positions in what came back, not local ids: the batch holds only the
        // candidates, in the order they were asked for.
        let mut rescored = candidates
            .iter()
            .enumerate()
            .map(|(position, id)| (*id, exact.distance(position as u32)))
            .collect::<Vec<_>>();
        rescored.sort_by(|left, right| left.1.total_cmp(&right.1));
        Ok((rescored, comparisons.get()))
    }
}
