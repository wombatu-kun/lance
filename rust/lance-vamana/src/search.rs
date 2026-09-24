// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Greedy search over one partition's graph.
//!
//! This is Algorithm 1 of the DiskANN paper, and it is used twice: a build runs
//! it once per vertex to collect the visited set a prune works from, and a query
//! runs it to answer. Both want the same thing, so it lives on its own.

use std::cell::Cell;
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array};
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::flat::index::FlatMetadata;
use lance_index::vector::flat::storage::{FLAT_COLUMN, FlatFloatStorage};
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::quantizer::QuantizerStorage;
use lance_index::vector::storage::DistCalculator;
use lance_linalg::distance::DistanceType;

use crate::partition::PartitionGraph;

/// Wrap vectors and their row ids in the [`VectorStore`] the primitives here want.
///
/// [`FlatFloatStorage::try_from_batch`] and not [`FlatFloatStorage::new`]: the
/// latter synthesises row ids `0..n`, and a build reads them straight into the
/// graph. The graph would then name positions instead of rows, and every answer
/// would point at the wrong data - while committing, reopening and searching
/// perfectly happily.
///
/// [`VectorStore`]: lance_index::vector::storage::VectorStore
pub fn flat_storage(
    row_ids: &[u64],
    vectors: &FixedSizeListArray,
    distance_type: DistanceType,
) -> Result<FlatFloatStorage> {
    let batch = RecordBatch::try_from_iter_with_nullable(vec![
        (
            ROW_ID,
            Arc::new(UInt64Array::from(row_ids.to_vec())) as ArrayRef,
            false,
        ),
        (FLAT_COLUMN, Arc::new(vectors.clone()) as ArrayRef, false),
    ])?;
    FlatFloatStorage::try_from_batch(
        batch,
        &FlatMetadata {
            dim: vectors.value_length() as usize,
        },
        distance_type,
        None,
    )
}

/// Counts distance computations.
///
/// Recall is only half of a graph index's story; the other half is what it cost
/// to get there. Lance measures neither for HNSW - its builder takes a metrics
/// argument and ignores it - so this crate carries its own from the first
/// algorithm, before there is anything to be tempted to flatter.
#[derive(Debug, Default)]
pub struct Comparisons(Cell<u64>);

impl Comparisons {
    /// Saturating rather than checked: this is a measurement, and a counter that
    /// has run out is not a reason to fail the query it was measuring.
    #[inline]
    pub fn record(&self, count: u64) {
        self.0.set(self.0.get().saturating_add(count));
    }

    pub fn get(&self) -> u64 {
        self.0.get()
    }
}

/// Reusable scratch space for [`greedy_search`].
///
/// The visited marks are the only allocation that scales with the partition,
/// and a build runs one search per vertex, so they are stamped with a
/// generation counter and reused instead of reallocated per search.
///
/// A byte a vertex rather than four. A stamp only has to tell this search from
/// the ones before it, and every byte the marks do not take is cache the codes
/// a walk reads can have. The price is clearing the whole buffer once every 255
/// searches instead of once every four billion.
#[derive(Debug)]
pub struct SearchScratch {
    seen: Vec<u8>,
    generation: u8,
}

impl SearchScratch {
    pub fn new(num_vertices: usize) -> Self {
        Self {
            seen: vec![0; num_vertices],
            generation: 0,
        }
    }

    /// Start a search: every mark from the previous one stops counting.
    pub(crate) fn begin(&mut self) {
        self.generation = match self.generation.checked_add(1) {
            Some(next) => next,
            // 255 searches later the stamps stop being unique, so the marks
            // are cleared once and numbering restarts. All of them, not the
            // partition's share: a pooled buffer is as long as the largest
            // partition it has served, and the next one may read its tail.
            None => {
                self.seen.fill(0);
                1
            }
        };
    }

    /// Mark `id` as reached, returning whether this search had not reached it.
    pub(crate) fn mark(&mut self, id: u32) -> bool {
        let slot = &mut self.seen[id as usize];
        if *slot == self.generation {
            false
        } else {
            *slot = self.generation;
            true
        }
    }

    /// Make room for a partition of `num_vertices`.
    ///
    /// Too short, the buffer is replaced rather than grown: every mark in it is
    /// stale by the next [`Self::begin`] anyway, zero is no search's
    /// generation, and a new buffer is exactly as long as asked for, where a
    /// grown one would keep amortized headroom for as long as a pool keeps it.
    pub(crate) fn cover(&mut self, num_vertices: usize) {
        if self.seen.len() < num_vertices {
            self.seen = vec![0; num_vertices];
        }
    }
}

/// Visited marks a finished walk hands on to the next one.
///
/// A scratch holds a slot for every vertex of the partition it walks, a
/// megabyte at a million rows, and a new one per walk is that much memory
/// allocated and zeroed before the first hop, however short the walk. Handed
/// on, it costs a generation bump instead, and one hand-on in 255 a clear of the
/// whole buffer when the generation runs out.
///
/// It keeps as many scratches as walks have run at once, up to one per core,
/// each as long as the largest partition it has served, for the life of the
/// index and outside any cache budget. A walk over resident edges never waits
/// while it holds one, so no more are in use together than there are threads
/// polling queries, which on a default runtime is one per core. A walk that
/// fetches its edges holds its scratch across reads, and one beyond the cap is
/// allocated and dropped as before.
pub(crate) struct ScratchPool {
    /// A poisoned lock is taken as it is: a push or a pop cannot leave the list
    /// half-changed.
    idle: Mutex<Vec<SearchScratch>>,
    max_idle: usize,
}

impl ScratchPool {
    pub(crate) fn new() -> Self {
        Self {
            idle: Mutex::default(),
            // Unknown only where the platform cannot say; one idle scratch
            // still serves queries that arrive one at a time.
            max_idle: std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
        }
    }

    /// Lend out the scratch handed back last, or an empty one when none is
    /// idle; the walk sizes it with [`SearchScratch::cover`].
    pub(crate) fn take(&self) -> PooledScratch<'_> {
        let idle = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        PooledScratch {
            pool: self,
            scratch: idle.unwrap_or_else(|| SearchScratch::new(0)),
        }
    }

    /// Keep `scratch` for the next walk, unless `max_idle` are idle already.
    fn put(&self, scratch: SearchScratch) {
        let mut idle = self.idle.lock().unwrap_or_else(PoisonError::into_inner);
        if idle.len() < self.max_idle {
            idle.push(scratch);
        }
    }
}

/// A scratch on loan from a [`ScratchPool`], back in the pool when dropped:
/// after an error or a cancelled query as much as after an answer.
pub(crate) struct PooledScratch<'a> {
    pool: &'a ScratchPool,
    scratch: SearchScratch,
}

impl Deref for PooledScratch<'_> {
    type Target = SearchScratch;

    fn deref(&self) -> &SearchScratch {
        &self.scratch
    }
}

impl DerefMut for PooledScratch<'_> {
    fn deref_mut(&mut self) -> &mut SearchScratch {
        &mut self.scratch
    }
}

impl Drop for PooledScratch<'_> {
    fn drop(&mut self) {
        self.pool
            .put(std::mem::replace(&mut self.scratch, SearchScratch::new(0)));
    }
}

/// Counts rather than contents: a derived one would print every slot of every
/// idle scratch.
impl fmt::Debug for ScratchPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let idle = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
        f.debug_struct("ScratchPool")
            .field("idle", &idle)
            .field("max_idle", &self.max_idle)
            .finish()
    }
}

/// A vertex in the search list, and whether its out-edges have been followed.
///
/// Not `Candidate`: [`crate::lazy::Candidate`] owns that name, and means the
/// other end of the same walk - a vertex the list finished with, on its way to
/// an exact distance.
#[derive(Debug, Clone)]
struct Entry {
    node: OrderedNode,
    expanded: bool,
}

/// When a walk stops before its list runs out of vertices to expand: once none
/// of them is among its `rank` nearest candidates or nearer than `1 + margin`
/// times the length to the `rank`-th, which is adaptive beam search
/// (Al-Jazzazi et al., NeurIPS 2025) with `gamma = margin` and `k = rank`.
///
/// `keep` is the other half, and a walk that re-scores needs both: the nearest
/// `keep` candidates stay in the list whatever their distance, because they are
/// what the re-score is owed. It is at least `rank`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StopRule {
    pub(crate) margin: f32,
    pub(crate) rank: usize,
    pub(crate) keep: usize,
}

/// A [`StopRule`] as a list applies it.
#[derive(Debug, Clone, Copy)]
struct Margin {
    rank: usize,
    keep: usize,
    /// `(1 + margin)^2`: the rule is about lengths, and every distance a walk
    /// measures here is a squared one - or, over RaBitQ codes, an estimate of
    /// one, which can fall below zero. There the bar sits below the `rank`-th,
    /// nothing past it is expanded whatever the margin, and a wider margin
    /// lowers the bar rather than raising it.
    factor: f32,
    /// `factor` times the distance of the entry at `rank - 1`, set the moment
    /// the list is that long and read only once it is.
    ///
    /// It only ever falls, because the entry it is taken from can only be
    /// replaced by a nearer one, and that is what makes it safe to throw away
    /// what lies beyond it: a candidate the bar has passed over now stays
    /// passed over.
    bar: OrderedFloat,
}

/// `L`: the beam a walk keeps, nearest first.
///
/// Shared rather than written twice because there are two walks over it and they
/// have to be the same walk. [`greedy_search`] holds the whole partition and
/// takes one vertex at a time; the lazy walk in `crate::lazy` fetches the edges
/// of several at once, because a round trip it can batch is what it is paying
/// in. Everything else - what gets in, what gets pushed out, what order
/// the answer comes back in - has to be identical, and the way to know it is
/// identical is for there to be one copy of it.
#[derive(Debug)]
pub struct SearchList {
    list: Vec<Entry>,
    size: usize,
    /// Every entry before this one has been expanded.
    ///
    /// A walk expands nearest first, so the expanded entries gather at the
    /// front. An entry lands among them only by being nearer than one of them,
    /// and moves this back to where it landed.
    cursor: usize,
    /// The walk's [`StopRule`], if it has one; `None` expands until nothing
    /// in the list is left unexpanded.
    ///
    /// With one, every entry at or past `keep` is nearer than the bar. So an
    /// entry the rule would not expand can only sit between `rank` and `keep`,
    /// and the list is always the front of the plain list fed the same offers
    /// under the same cap, with nothing cut.
    margin: Option<Margin>,
}

impl SearchList {
    /// A list of at most `search_list_size` entries over `num_vertices`.
    ///
    /// Bounded by the graph as well as by `L`, because the list can never hold
    /// more than one entry per vertex and `L` comes from a caller who may have
    /// passed `usize::MAX` - which this would otherwise hand to the allocator.
    pub fn new(search_list_size: usize, num_vertices: usize) -> Self {
        Self {
            list: Vec::with_capacity(search_list_size.min(num_vertices).saturating_add(1)),
            size: search_list_size,
            cursor: 0,
            margin: None,
        }
    }

    /// A list that stops its walk by `stop`, capped at `search_list_size`.
    ///
    /// The cap still binds: an entry past it is dropped however near the bar
    /// it is, so a walk that means to follow the rule needs a cap it never
    /// reaches. The caller checks `1 <= rank <= keep <= search_list_size`.
    pub(crate) fn with_margin(
        search_list_size: usize,
        num_vertices: usize,
        stop: StopRule,
    ) -> Self {
        debug_assert!(
            1 <= stop.rank && stop.rank <= stop.keep && stop.keep <= search_list_size,
            "a stop rule at rank {} keeping {} does not fit a list of {search_list_size}",
            stop.rank,
            stop.keep
        );
        let widened = 1.0 + stop.margin;
        debug_assert!(
            (widened * widened).is_finite(),
            "a stop margin of {} has no finite square to measure a bar by",
            stop.margin
        );
        Self {
            margin: Some(Margin {
                rank: stop.rank,
                keep: stop.keep,
                factor: widened * widened,
                bar: OrderedFloat(f32::INFINITY),
            }),
            ..Self::new(search_list_size, num_vertices)
        }
    }

    /// Offer a vertex at `distance`, keeping it if it beats the back of the list.
    ///
    /// Nothing here checks whether `id` is already in the list: a caller must
    /// have marked it in its [`SearchScratch`] first, which is what makes a
    /// vertex measured once and offered once.
    ///
    /// A full list turns away a vertex no nearer than its back before searching
    /// for a place: the search would put it after every entry it is no nearer
    /// than, which is past the end, and most of what a walk offers once its
    /// list has filled is exactly that.
    ///
    /// A list with a [`StopRule`] also turns away a vertex that would land past
    /// its `keep` nearest and is not nearer than the bar, and cuts off every
    /// entry past `keep` that is not nearer than it either: the one an insert
    /// pushes past `keep`, and those a falling bar has passed over. The rule
    /// would never expand them, the re-score never sees them, and the bar only
    /// falls, so none of them can come back into play. Among equals at `keep`
    /// the first offered stays, as it does at the cap.
    pub fn offer(&mut self, id: u32, distance: f32) {
        let distance = OrderedFloat(distance);
        if self.list.len() >= self.size
            && self
                .list
                .last()
                .is_some_and(|back| back.node.dist <= distance)
        {
            return;
        }
        if let Some(margin) = &self.margin
            && self.list.len() >= margin.keep
            && margin.bar <= distance
            && self.list[margin.keep - 1].node.dist <= distance
        {
            return;
        }
        let at = self
            .list
            .partition_point(|entry| entry.node.dist <= distance);
        if at >= self.size {
            return;
        }
        self.list.insert(
            at,
            Entry {
                node: OrderedNode::new(id, distance),
                expanded: false,
            },
        );
        self.list.truncate(self.size);
        self.cursor = self.cursor.min(at);
        if let Some(margin) = &mut self.margin {
            if at < margin.rank && self.list.len() >= margin.rank {
                margin.bar = OrderedFloat(margin.factor * self.list[margin.rank - 1].node.dist.0);
            }
            // An insert before `keep` pushes the entry at `keep - 1` past it,
            // and a new bar may have passed over some of those already there.
            if at < margin.keep && self.list.len() > margin.keep {
                let nearer =
                    self.list[margin.keep..].partition_point(|entry| entry.node.dist < margin.bar);
                self.list.truncate(margin.keep + nearer);
            }
        }
        // Truncating to the cap cuts a list back only when it was full before
        // the insert, and cutting to the bar only past `keep`, beyond the
        // insert - so neither ever cuts below the cursor.
        debug_assert!(
            self.cursor <= self.list.len(),
            "a search list's cursor {} ran past its {} entries",
            self.cursor,
            self.list.len()
        );
    }

    /// Measure every id in `ids` and offer it, asking for the code of the one
    /// `ahead` further on before the current one is measured.
    ///
    /// Every id must be a vertex of `calculator`'s store, and must already be
    /// marked, exactly as [`Self::offer`] requires - this asks for a code at the
    /// id as well as measuring one, and both index the same array.
    ///
    /// The ask is [`DistCalculator::prefetch`], which is a hint and only a hint:
    /// it changes what the processor has already loaded by the time a distance
    /// reads it, and it cannot change what the distance is. An eight-bit code
    /// sits `id * d` bytes into an array holding every code of the partition,
    /// and the ids of one hop are as unrelated to one another as the graph made
    /// them, so no hardware prefetcher can guess the next one. What it can be
    /// told is the whole hop at once, because a hop knows every id it will
    /// measure before it measures any of them.
    ///
    /// `ahead` of zero asks for nothing at all. That is the hint's own control -
    /// not the walk as it was, which also offered its hop one neighbour at a
    /// time rather than collecting it first.
    ///
    /// An `ahead` at or past the length of a hop degenerates: the whole hop is
    /// asked for back to back with no distance in between, which is the opposite
    /// of hiding a load behind work and can evict the lines it just asked for.
    /// Nothing rejects it, because there is no value at which it stops being
    /// merely unwise.
    pub fn offer_all(&mut self, ids: &[u32], calculator: &impl DistCalculator, ahead: usize) {
        // The first `ahead` codes have no full distance to be loaded behind, so
        // they are asked for before any measuring starts rather than left out,
        // which is where Lance's own look-ahead leaves them. The very first is a
        // wash - it is read immediately after - and is asked for anyway to keep
        // the rule one ask per code.
        for id in ids.iter().take(ahead) {
            calculator.prefetch(*id);
        }
        for (position, id) in ids.iter().enumerate() {
            if ahead != 0 {
                // Saturating because `ahead` is the caller's number, and a sum
                // past the end of `usize` says the same thing as a sum past the
                // end of the slice.
                if let Some(later) = ids.get(position.saturating_add(ahead)) {
                    calculator.prefetch(*later);
                }
            }
            self.offer(*id, calculator.distance(*id));
        }
    }

    /// The nearest vertex whose out-edges have not been followed, marked as
    /// followed.
    ///
    /// Called `n` times in a row without an [`Self::offer`] between them, it
    /// yields the `n` nearest unexpanded vertices - which is exactly the
    /// frontier a lazy hop fetches in one request.
    ///
    /// The search starts at the cursor rather than at the front, past entries
    /// that are all expanded already.
    ///
    /// With a [`StopRule`] it is also `None` when that vertex is one the rule
    /// does not expand: past the `rank` nearest and not nearer than the bar.
    /// Every unexpanded vertex after it is no nearer, so that is where the walk
    /// stops - until an offer lands a nearer one.
    pub fn next_unexpanded(&mut self) -> Option<OrderedNode> {
        let position = self.cursor
            + self.list[self.cursor..]
                .iter()
                .position(|entry| !entry.expanded)?;
        if let Some(margin) = &self.margin
            && position >= margin.rank
            && margin.bar <= self.list[position].node.dist
        {
            // Everything before it is expanded, so the cursor may wait there.
            self.cursor = position;
            return None;
        }
        self.list[position].expanded = true;
        self.cursor = position + 1;
        Some(self.list[position].node.clone())
    }

    /// The list itself, nearest first.
    pub fn into_candidates(self) -> Vec<OrderedNode> {
        self.list.into_iter().map(|entry| entry.node).collect()
    }
}

#[derive(Debug)]
pub struct SearchResult {
    /// The search list, nearest first, at most `search_list_size` long: `L`.
    pub candidates: Vec<OrderedNode>,
    /// Vertices whose out-edges were followed, in the order they were followed:
    /// `V` in the paper, and the input a build's prune works from.
    ///
    /// This is not every vertex whose distance was computed. A vertex that
    /// entered the list and was pushed out before its turn is not in `V`, which
    /// is what the paper specifies and what keeps the prune's candidate set
    /// bounded by the search's work rather than by the partition's size.
    pub visited: Vec<OrderedNode>,
}

/// Walk the graph from `entry_point` towards `query`.
///
/// `search_list_size` is `L`: the beam kept while walking. Larger `L` costs
/// distance computations and buys recall, and at `L = 1` this degenerates into
/// plain hill climbing.
pub fn greedy_search(
    graph: &PartitionGraph,
    query: &impl DistCalculator,
    entry_point: u32,
    search_list_size: usize,
    scratch: &mut SearchScratch,
    comparisons: &Comparisons,
) -> Result<SearchResult> {
    if search_list_size == 0 {
        return Err(Error::invalid_input(
            "Vamana search list size must be greater than zero".to_string(),
        ));
    }
    if entry_point as usize >= graph.len() {
        return Err(Error::invalid_input(format!(
            "Vamana entry point {entry_point} is outside a partition of {} vertices",
            graph.len()
        )));
    }
    if scratch.seen.len() < graph.len() {
        return Err(Error::invalid_input(format!(
            "Vamana search scratch holds {} vertices but the partition has {}",
            scratch.seen.len(),
            graph.len()
        )));
    }

    scratch.begin();
    scratch.mark(entry_point);
    comparisons.record(1);
    let mut list = SearchList::new(search_list_size, graph.len());
    list.offer(entry_point, query.distance(entry_point));
    let mut visited = Vec::new();

    while let Some(nearest_unexpanded) = list.next_unexpanded() {
        visited.push(nearest_unexpanded.clone());

        for neighbor in graph.neighbors(nearest_unexpanded.id)? {
            if !scratch.mark(*neighbor) {
                continue;
            }
            comparisons.record(1);
            list.offer(*neighbor, query.distance(*neighbor));
        }
    }

    Ok(SearchResult {
        candidates: list.into_candidates(),
        visited,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;

    use arrow_array::Float32Array;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::storage::VectorStore;
    use lance_linalg::distance::DistanceType;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    use super::*;

    /// A calculator that writes down what it was asked for, in the order it was
    /// asked.
    ///
    /// A prefetch has no effect a test can see - that is what makes it a hint -
    /// so what is pinned here is the asking: every code asked for once, and
    /// asked for the agreed number of distances before the one that reads it.
    #[derive(Default)]
    struct Recorder {
        asked: RefCell<Vec<Ask>>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Ask {
        Prefetched(u32),
        Measured(u32),
    }

    impl DistCalculator for Recorder {
        fn distance(&self, id: u32) -> f32 {
            self.asked.borrow_mut().push(Ask::Measured(id));
            id as f32
        }

        fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
            unreachable!("a hop measures the ids it collected, never the partition")
        }

        fn prefetch(&self, id: u32) {
            self.asked.borrow_mut().push(Ask::Prefetched(id));
        }
    }

    fn only(asked: &[Ask], wanted: fn(&Ask) -> Option<u32>) -> Vec<u32> {
        asked.iter().filter_map(wanted).collect()
    }

    fn prefetched(ask: &Ask) -> Option<u32> {
        match ask {
            Ask::Prefetched(id) => Some(*id),
            Ask::Measured(_) => None,
        }
    }

    fn measured(ask: &Ask) -> Option<u32> {
        match ask {
            Ask::Measured(id) => Some(*id),
            Ask::Prefetched(_) => None,
        }
    }

    /// Every code asked for exactly once, and `ahead` distances before its own.
    ///
    /// The gap is the whole point: a code asked for one distance early hides one
    /// distance of memory latency. Nearer the start of the hop there is less to
    /// hide behind, which is why the first `ahead` of them are asked for
    /// together before any measuring - so the gap is `ahead` distances or every
    /// distance so far, whichever is fewer.
    #[test]
    fn a_look_ahead_asks_for_every_code_once_and_early() {
        let ids: Vec<u32> = (10..20).collect();
        for ahead in [1, 2, 3, 9] {
            let recorder = Recorder::default();
            let mut list = SearchList::new(ids.len(), 64);
            list.offer_all(&ids, &recorder, ahead);
            let asked = recorder.asked.into_inner();

            assert_eq!(
                only(&asked, prefetched),
                ids,
                "at a look-ahead of {ahead} the hop did not ask for each of its codes once"
            );
            assert_eq!(
                only(&asked, measured),
                ids,
                "at a look-ahead of {ahead} the hop measured something other than what it collected"
            );
            for (position, id) in ids.iter().enumerate() {
                let asked_at = asked
                    .iter()
                    .position(|ask| *ask == Ask::Prefetched(*id))
                    .unwrap();
                let measured_at = asked
                    .iter()
                    .position(|ask| *ask == Ask::Measured(*id))
                    .unwrap();
                let between = asked[asked_at..measured_at]
                    .iter()
                    .filter(|ask| matches!(ask, Ask::Measured(_)))
                    .count();
                assert_eq!(
                    between,
                    position.min(ahead),
                    "at a look-ahead of {ahead}, the code of {id} was asked for {between} \
                     distances before the one that read it"
                );
            }
        }
    }

    /// No look-ahead asks for nothing at all.
    ///
    /// This is the arm a measurement of the look-ahead compares against, so it
    /// has to ask for nothing rather than ask for the code it is about to read
    /// anyway - which on eight-bit codes at `d = 960` would be fifteen wasted
    /// instructions a neighbour, charged to the control.
    #[test]
    fn no_look_ahead_asks_for_nothing() {
        let ids: Vec<u32> = (10..20).collect();
        let recorder = Recorder::default();
        let mut list = SearchList::new(ids.len(), 64);
        list.offer_all(&ids, &recorder, 0);
        assert_eq!(
            recorder.asked.into_inner(),
            ids.iter().map(|id| Ask::Measured(*id)).collect::<Vec<_>>(),
        );
    }

    /// A look-ahead longer than the hop asks for the hop and stops there.
    ///
    /// `usize::MAX` included, because the depth is a caller's number and
    /// `position + ahead` is arithmetic it would otherwise get a say in.
    #[test]
    fn a_look_ahead_past_the_hop_is_the_hop() {
        let ids: Vec<u32> = (10..20).collect();
        for ahead in [ids.len(), ids.len() + 1, usize::MAX] {
            let recorder = Recorder::default();
            let mut list = SearchList::new(ids.len(), 64);
            list.offer_all(&ids, &recorder, ahead);
            let asked = recorder.asked.into_inner();
            assert_eq!(
                only(&asked, prefetched),
                ids,
                "at a look-ahead of {ahead} the hop did not ask for each of its codes once"
            );
            assert_eq!(
                only(&asked, measured),
                ids,
                "at a look-ahead of {ahead} the hop measured something other than what it collected"
            );
        }
    }

    /// A calculator that cannot tell its vertices apart.
    struct Level;

    impl DistCalculator for Level {
        fn distance(&self, _id: u32) -> f32 {
            1.0
        }

        fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
            unreachable!("a hop measures the ids it collected, never the partition")
        }
    }

    /// A full list keeps the first of equals, and a hop is offered in the order
    /// it was handed.
    ///
    /// The walk leans on both. Which of two equally distant vertices survives a
    /// full list is decided by which was offered first, so a hop that collected
    /// its ids in some other order would answer differently wherever codes tie -
    /// and coarse codes tie. Nothing else in the crate pins this: a fixture of
    /// random vectors never produces two equal distances, so a walk over one
    /// answers the same whatever order its hops are offered in.
    #[test]
    fn a_full_list_keeps_the_first_of_equals() {
        let ids = [5, 9, 2, 7];
        let kept = |list: SearchList| {
            list.into_candidates()
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>()
        };

        let mut apart = SearchList::new(2, 16);
        for id in &ids {
            apart.offer(*id, Level.distance(*id));
        }
        assert_eq!(
            kept(apart),
            vec![5, 9],
            "a full list did not keep the two equals it was offered first"
        );

        let mut together = SearchList::new(2, 16);
        together.offer_all(&ids, &Level, 2);
        assert_eq!(
            kept(together),
            vec![5, 9],
            "a hop was not offered in the order it was handed"
        );
    }

    /// Offering a run of ids is offering them one at a time.
    #[test]
    fn a_hop_offered_together_is_a_hop_offered_one_at_a_time() {
        let ids: Vec<u32> = vec![7, 3, 9, 1, 5, 2];
        let storage = line_storage(16);
        let calculator = storage.dist_calculator_from_id(4);
        for ahead in [0, 1, 4] {
            let mut together = SearchList::new(3, 16);
            together.offer_all(&ids, &calculator, ahead);
            let mut apart = SearchList::new(3, 16);
            for id in &ids {
                apart.offer(*id, calculator.distance(*id));
            }
            assert_eq!(
                together.into_candidates(),
                apart.into_candidates(),
                "a look-ahead of {ahead} changed the list the hop left behind"
            );
        }
    }

    /// Vertices on a line at 0, 1, 2, ... so every distance is hand-checkable.
    fn line_storage(num_vertices: usize) -> FlatFloatStorage {
        let values = Float32Array::from((0..num_vertices).map(|i| i as f32).collect::<Vec<_>>());
        FlatFloatStorage::new(
            arrow_array::FixedSizeListArray::try_new_from_values(values, 1).unwrap(),
            DistanceType::L2,
        )
    }

    /// A path 0 - 1 - 2 - ... - n-1, so reaching the far end takes n-1 hops.
    fn path_graph(num_vertices: usize) -> PartitionGraph {
        let adjacency = (0..num_vertices)
            .map(|i| match i {
                0 => vec![1],
                last if last == num_vertices - 1 => vec![(last - 1) as u32],
                middle => vec![(middle - 1) as u32, (middle + 1) as u32],
            })
            .collect();
        PartitionGraph::try_new(4, (0..num_vertices as u64).collect(), adjacency).unwrap()
    }

    fn search(
        graph: &PartitionGraph,
        storage: &FlatFloatStorage,
        query: u32,
        entry_point: u32,
        search_list_size: usize,
    ) -> (SearchResult, u64) {
        let calculator = storage.dist_calculator_from_id(query);
        let comparisons = Comparisons::default();
        let mut scratch = SearchScratch::new(graph.len());
        let result = greedy_search(
            graph,
            &calculator,
            entry_point,
            search_list_size,
            &mut scratch,
            &comparisons,
        )
        .unwrap();
        (result, comparisons.get())
    }

    #[test]
    fn a_walk_along_a_path_reaches_the_far_end() {
        let graph = path_graph(16);
        let (result, _) = search(&graph, &line_storage(16), 15, 0, 4);

        assert_eq!(result.candidates[0].id, 15);
        assert_eq!(
            result
                .visited
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>(),
            "a path graph leaves no choice about the order vertices are expanded in"
        );
    }

    /// Every vertex the walk passes is a distance computation, and the count is
    /// the number of *edges* followed plus the entry point - not the number of
    /// vertices - because a vertex reached twice is only measured once.
    #[test]
    fn comparisons_count_each_vertex_once() {
        let graph = path_graph(16);
        let (_, comparisons) = search(&graph, &line_storage(16), 15, 0, 4);
        assert_eq!(comparisons, 16);
    }

    #[test]
    fn the_search_list_comes_back_exactly_as_wide_as_it_may_be() {
        const VERTICES: usize = 64;
        let graph = path_graph(VERTICES);
        for search_list_size in [1, 2, 7, 64, 128] {
            let (result, _) = search(&graph, &line_storage(VERTICES), 63, 0, search_list_size);
            // Equality, not a ceiling. Every vertex of this path is reached, so
            // the list is full whenever `L` allows it, and an upper bound alone
            // would pass for a walk that kept one candidate at any `L`.
            assert_eq!(
                result.candidates.len(),
                search_list_size.min(VERTICES),
                "at L = {search_list_size}"
            );
            assert!(
                result.candidates.windows(2).all(|pair| pair[0] <= pair[1]),
                "the search list came back unsorted at L = {search_list_size}"
            );
        }
    }

    /// A graph with a trap, because a path graph cannot show what `L` is for:
    /// every vertex along a path is strictly closer than the last, so one slot
    /// walks it exactly like a hundred and the whole beam is inert.
    ///
    /// Vertices sit on a line at 0, 50, 40, 20, 99 and the query is vertex 4, at
    /// 99. Expanding the entry point offers vertex 1 (at 50) and vertex 2 (at
    /// 40). With `L = 1` only the nearer of them survives, its own neighbour is
    /// worse still, and the walk ends having never seen the answer. With `L = 2`
    /// vertex 2 stays in the list, and the answer hangs off it.
    fn trap_graph() -> (PartitionGraph, FlatFloatStorage) {
        let positions = [0.0f32, 50.0, 40.0, 20.0, 99.0];
        let storage = FlatFloatStorage::new(
            arrow_array::FixedSizeListArray::try_new_from_values(
                Float32Array::from(positions.to_vec()),
                1,
            )
            .unwrap(),
            DistanceType::L2,
        );
        let graph = PartitionGraph::try_new(
            2,
            (0..positions.len() as u64).collect(),
            vec![vec![1, 2], vec![3], vec![4], vec![], vec![]],
        )
        .unwrap();
        (graph, storage)
    }

    #[test]
    fn a_wider_search_list_escapes_a_local_minimum() {
        let (graph, storage) = trap_graph();

        let (narrow, _) = search(&graph, &storage, 4, 0, 1);
        assert_eq!(
            narrow.candidates[0].id, 1,
            "a one-slot list must fall into the trap, or the fixture is not a trap"
        );

        let (wide, _) = search(&graph, &storage, 4, 0, 2);
        assert_eq!(
            wide.candidates[0].id, 4,
            "a two-slot list must find the answer"
        );
    }

    /// A vertex in another component is unreachable however good its distance.
    #[test]
    fn a_walk_cannot_leave_its_component() {
        let graph = PartitionGraph::try_new(
            4,
            (0..6).collect(),
            vec![vec![1], vec![0], vec![3], vec![2], vec![5], vec![4]],
        )
        .unwrap();
        let (result, _) = search(&graph, &line_storage(6), 5, 0, 6);

        assert_eq!(
            result
                .visited
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        // Vertex 1 is the best the walk can do: it is nearer the query than the
        // entry point, and everything nearer still is in another component.
        assert_eq!(result.candidates[0].id, 1);
    }

    #[test]
    fn an_entry_point_outside_the_partition_is_rejected() {
        let graph = path_graph(4);
        let storage = line_storage(4);
        let calculator = storage.dist_calculator_from_id(0);
        let error = greedy_search(
            &graph,
            &calculator,
            4,
            4,
            &mut SearchScratch::new(4),
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("entry point 4"), "{error}");
    }

    #[test]
    fn a_zero_length_search_list_is_rejected() {
        let graph = path_graph(4);
        let storage = line_storage(4);
        let calculator = storage.dist_calculator_from_id(0);
        let error = greedy_search(
            &graph,
            &calculator,
            0,
            0,
            &mut SearchScratch::new(4),
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("search list size"), "{error}");
    }

    /// Scratch too small for the partition would index out of bounds rather
    /// than misbehave, so it is caught at the door.
    #[test]
    fn undersized_scratch_is_rejected() {
        let graph = path_graph(8);
        let storage = line_storage(8);
        let calculator = storage.dist_calculator_from_id(0);
        let error = greedy_search(
            &graph,
            &calculator,
            0,
            4,
            &mut SearchScratch::new(4),
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("scratch"), "{error}");
    }

    /// Scratch is reused across searches, so a stale mark from the previous
    /// search would silently cut the next one short.
    #[test]
    fn reused_scratch_does_not_leak_between_searches() {
        let graph = path_graph(16);
        let storage = line_storage(16);
        let mut scratch = SearchScratch::new(16);
        let comparisons = Comparisons::default();

        let mut lengths = Vec::new();
        for _ in 0..3 {
            let calculator = storage.dist_calculator_from_id(15);
            let result =
                greedy_search(&graph, &calculator, 0, 4, &mut scratch, &comparisons).unwrap();
            lengths.push(result.visited.len());
        }
        assert_eq!(lengths, vec![16, 16, 16]);
        assert_eq!(comparisons.get(), 48);
    }

    /// A pooled scratch arrives with the marks of whatever it served last, over
    /// a partition larger or smaller than this one, and has to walk exactly as a
    /// new one does. One scratch serves the whole run, so what is left idle at
    /// the end is that one, grown once to the longest path.
    #[test]
    fn a_pooled_scratch_walks_like_a_new_one() {
        let pool = ScratchPool::new();
        for num_vertices in [4, 16, 8, 16] {
            let graph = path_graph(num_vertices);
            let storage = line_storage(num_vertices);
            let calculator = storage.dist_calculator_from_id(num_vertices as u32 - 1);
            let walk = |scratch: &mut SearchScratch| {
                let result =
                    greedy_search(&graph, &calculator, 0, 4, scratch, &Comparisons::default())
                        .unwrap();
                result
                    .visited
                    .iter()
                    .map(|node| node.id)
                    .collect::<Vec<_>>()
            };

            let mut pooled = pool.take();
            pooled.cover(num_vertices);
            let from_pool = walk(&mut pooled);
            drop(pooled);
            assert_eq!(
                from_pool,
                walk(&mut SearchScratch::new(num_vertices)),
                "a pooled scratch walked a path of {num_vertices} differently"
            );
        }

        let idle = pool
            .idle
            .lock()
            .unwrap()
            .iter()
            .map(|scratch| scratch.seen.len())
            .collect::<Vec<_>>();
        assert_eq!(
            idle,
            vec![16],
            "the walks were not handed one scratch between them"
        );
    }

    /// A one-byte generation runs out every 255 walks, which at seven thousand
    /// walks a second - a thousand queries of seven probes - is every 36
    /// milliseconds. The numbering then restarts at 1, which some slot may
    /// still hold from 255 walks before.
    #[test]
    fn a_scratch_that_runs_out_of_generations_starts_clean() {
        let mut scratch = SearchScratch::new(4);
        scratch.begin();
        assert!(scratch.mark(2));
        scratch.generation = u8::MAX;
        scratch.begin();

        assert_eq!(scratch.generation, 1);
        assert!(
            scratch.mark(2),
            "a mark stamped at generation 1 survived the restart and read as this search's"
        );
    }

    /// The pool is worth its lock only if what it is handed is what it hands
    /// out next, and safe to keep only if it stops at its cap.
    #[test]
    fn a_pool_hands_back_what_it_was_given_up_to_its_cap() {
        let pool = ScratchPool {
            idle: Mutex::default(),
            max_idle: 2,
        };
        let lent = (0..3)
            .map(|_| {
                let mut scratch = pool.take();
                scratch.cover(16);
                scratch
            })
            .collect::<Vec<_>>();
        let buffers = lent
            .iter()
            .map(|scratch| scratch.seen.as_ptr())
            .collect::<Vec<_>>();
        drop(lent);

        assert_eq!(pool.idle.lock().unwrap().len(), 2);
        let (last, first, fresh) = (pool.take(), pool.take(), pool.take());
        assert_eq!(last.seen.as_ptr(), buffers[1]);
        assert_eq!(first.seen.as_ptr(), buffers[0]);
        assert!(
            fresh.seen.is_empty(),
            "an empty pool handed out a scratch someone had used"
        );
    }

    /// A full counter pins itself rather than panicking in a debug build and
    /// wrapping to nearly zero in a release one - the two ways a metric can
    /// take a query down with it, or lie about it.
    #[test]
    fn a_full_comparison_counter_stops_rather_than_wraps() {
        let comparisons = Comparisons::default();
        comparisons.record(u64::MAX - 1);
        comparisons.record(7);
        assert_eq!(comparisons.get(), u64::MAX);
    }

    /// [`SearchList`] as it was before it learned to turn a candidate away at
    /// the back and to resume at a cursor: a binary search on every offer and a
    /// scan from the front on every expansion.
    ///
    /// Kept here rather than trusted to the walks, because the walk over a
    /// partition in memory and the lazy walk both go through the one
    /// `SearchList`, so a change to it that alters an answer alters both
    /// answers alike, and the test comparing the two walks cannot see it.
    struct PlainList {
        list: Vec<(OrderedNode, bool)>,
        size: usize,
    }

    impl PlainList {
        fn new(size: usize) -> Self {
            Self {
                list: Vec::new(),
                size,
            }
        }

        fn offer(&mut self, id: u32, distance: f32) {
            let distance = OrderedFloat(distance);
            let at = self.list.partition_point(|(node, _)| node.dist <= distance);
            if at >= self.size {
                return;
            }
            self.list
                .insert(at, (OrderedNode::new(id, distance), false));
            self.list.truncate(self.size);
        }

        fn next_unexpanded(&mut self) -> Option<OrderedNode> {
            let position = self.list.iter().position(|(_, expanded)| !expanded)?;
            self.list[position].1 = true;
            Some(self.list[position].0.clone())
        }
    }

    /// A node down to the bits of its distance: `OrderedFloat` derives its
    /// equality, under which a NaN is not equal to itself.
    fn bits(node: OrderedNode) -> (u32, u32) {
        (node.id, node.dist.0.to_bits())
    }

    fn held(list: &SearchList) -> Vec<(u32, u32, bool)> {
        list.list
            .iter()
            .map(|entry| (entry.node.id, entry.node.dist.0.to_bits(), entry.expanded))
            .collect()
    }

    fn plainly_held(list: &PlainList) -> Vec<(u32, u32, bool)> {
        list.list
            .iter()
            .map(|(node, expanded)| (node.id, node.dist.0.to_bits(), *expanded))
            .collect()
    }

    /// Every offer and every expansion leaves a list exactly as it left the
    /// plain one, including where distances tie, run negative or are not
    /// numbers at all.
    ///
    /// Ties are the case that matters. A full list keeps the first of equals,
    /// so a list that turned a candidate away one comparison early or late
    /// would keep a different one - and coarse codes tie all the time.
    #[test]
    fn a_list_keeps_and_expands_exactly_what_a_plain_list_does() {
        const DISTANCES: [f32; 9] = [
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            f32::INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let mut rng = SmallRng::seed_from_u64(7);
        for size in [0, 1, 2, 3, 5, 64, usize::MAX] {
            for sequence in 0..150 {
                let mut list = SearchList::new(size, 256);
                let mut plain = PlainList::new(size);
                for step in 0..100 {
                    if rng.random_bool(0.7) {
                        let id = rng.random_range(0..256);
                        let distance = DISTANCES[rng.random_range(0..DISTANCES.len())];
                        list.offer(id, distance);
                        plain.offer(id, distance);
                    } else {
                        // Four in a row as well as one, because a lazy hop
                        // takes its whole frontier with no offer in between.
                        let expansions = if rng.random_bool(0.25) { 4 } else { 1 };
                        for _ in 0..expansions {
                            assert_eq!(
                                list.next_unexpanded().map(bits),
                                plain.next_unexpanded().map(bits),
                                "L = {size}, sequence {sequence}, step {step}: the lists expanded \
                                 different vertices"
                            );
                        }
                    }
                    assert_eq!(
                        held(&list),
                        plainly_held(&plain),
                        "L = {size}, sequence {sequence}, step {step}: the lists hold different \
                         entries"
                    );
                }
            }
        }
    }

    /// A calculator that reads every distance off a table.
    struct Table(Vec<f32>);

    impl DistCalculator for Table {
        fn distance(&self, id: u32) -> f32 {
            self.0[id as usize]
        }

        fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
            self.0.clone()
        }
    }

    /// A graph the write path accepts, chosen at random: out-edges distinct,
    /// none a self-edge, degrees anywhere from zero to `max_degree`.
    fn random_graph(num_vertices: usize, max_degree: usize, rng: &mut SmallRng) -> PartitionGraph {
        let adjacency = (0..num_vertices as u32)
            .map(|vertex| {
                let mut others = (0..num_vertices as u32)
                    .filter(|other| *other != vertex)
                    .collect::<Vec<_>>();
                others.shuffle(rng);
                others.truncate(rng.random_range(0..=max_degree));
                others
            })
            .collect();
        PartitionGraph::try_new(
            max_degree as u32,
            (0..num_vertices as u64).collect(),
            adjacency,
        )
        .unwrap()
    }

    /// What a walk leaves: its list, the vertices it expanded in order, and the
    /// distances it measured.
    type Walk = (Vec<(u32, u32)>, Vec<(u32, u32)>, u64);

    /// Algorithm 1 written again from nothing [`greedy_search`] uses: the plain
    /// list, and a set for the marks.
    fn reference_walk(
        graph: &PartitionGraph,
        table: &Table,
        entry_point: u32,
        search_list_size: usize,
    ) -> Walk {
        let mut reached = HashSet::from([entry_point]);
        let mut list = PlainList::new(search_list_size);
        list.offer(entry_point, table.distance(entry_point));
        let mut comparisons = 1;
        let mut visited = Vec::new();
        while let Some(node) = list.next_unexpanded() {
            visited.push(bits(node.clone()));
            for neighbor in graph.neighbors(node.id).unwrap() {
                if reached.insert(*neighbor) {
                    comparisons += 1;
                    list.offer(*neighbor, table.distance(*neighbor));
                }
            }
        }
        let candidates = list.list.into_iter().map(|(node, _)| bits(node)).collect();
        (candidates, visited, comparisons)
    }

    fn walk(
        graph: &PartitionGraph,
        table: &Table,
        entry_point: u32,
        search_list_size: usize,
        scratch: &mut SearchScratch,
    ) -> Walk {
        scratch.cover(graph.len());
        let comparisons = Comparisons::default();
        let result = greedy_search(
            graph,
            table,
            entry_point,
            search_list_size,
            scratch,
            &comparisons,
        )
        .unwrap();
        (
            result.candidates.into_iter().map(bits).collect(),
            result.visited.into_iter().map(bits).collect(),
            comparisons.get(),
        )
    }

    /// A walk is the reference walk, on distances drawn to tie and on one
    /// scratch handed from walk to walk the way a pool hands it on.
    ///
    /// This is what a build rests on as much as a query: the prune that shapes
    /// the graph works from the vertices a walk expanded, in the order it
    /// expanded them.
    ///
    /// The partitions come in a random order rather than smallest first. In
    /// that order the buffer grows to the largest early and every later walk of
    /// a smaller one leaves stamps of other walks in its tail, where smallest
    /// first would hand each size a fresh buffer; and there are enough walks
    /// for the generation to run out twice.
    #[test]
    fn a_walk_is_the_reference_walk_where_distances_tie() {
        const DISTANCES: [f32; 5] = [-1.0, 0.0, 1.0, 1.0, 2.0];
        let mut rng = SmallRng::seed_from_u64(11);
        let mut partitions = Vec::new();
        for num_vertices in [1, 2, 3, 17, 64] {
            for _ in 0..24 {
                let graph = random_graph(num_vertices, 4, &mut rng);
                let table = Table(
                    (0..num_vertices)
                        .map(|_| DISTANCES[rng.random_range(0..DISTANCES.len())])
                        .collect(),
                );
                let entry_point = rng.random_range(0..num_vertices as u32);
                partitions.push((graph, table, entry_point));
            }
        }
        let mut walks = (0..partitions.len())
            .flat_map(|partition| {
                [1, 2, 4, 16, usize::MAX].map(|search_list_size| (partition, search_list_size))
            })
            .collect::<Vec<_>>();
        walks.shuffle(&mut rng);
        assert!(
            walks.len() > 2 * usize::from(u8::MAX),
            "{} walks are too few for a one-byte generation to run out twice",
            walks.len()
        );

        let mut scratch = SearchScratch::new(0);
        for (partition, search_list_size) in walks {
            let (graph, table, entry_point) = &partitions[partition];
            assert_eq!(
                walk(graph, table, *entry_point, search_list_size, &mut scratch),
                reference_walk(graph, table, *entry_point, search_list_size),
                "a walk over {} vertices at L = {search_list_size} left a different list, \
                 expanded a different sequence or measured a different number of distances",
                graph.len()
            );
        }
    }

    /// A walk that lands on the wrap of the generation is the reference walk,
    /// with stamps from 255 walks before still in the tail of the buffer; and
    /// so is a walk a whole cycle after a wrap, over slots nothing has stamped
    /// since the wrap cleared them.
    ///
    /// The first walk stamps all of a large partition at generation 1, the next
    /// 254 touch only the three slots at the front of the buffer, and the large
    /// partition is walked again at the moment the generation runs out: a wrap
    /// that restarted the numbering without clearing would read every vertex
    /// past the third as reached already. Then the small partition takes the
    /// next wrap and a whole cycle more, and the large one is walked at the last
    /// generation: a wrap that cleared to anything but zero would read the
    /// slots it cleared as stamped by that generation.
    #[test]
    fn a_walk_on_the_wrap_of_the_generation_is_the_reference_walk() {
        const LARGE: usize = 64;
        let large = path_graph(LARGE);
        let small = path_graph(3);
        let large_table = Table((0..LARGE).map(|vertex| (vertex % 4) as f32).collect());
        let small_table = Table(vec![0.0; 3]);
        let mut scratch = SearchScratch::new(0);
        let small_walks = |scratch: &mut SearchScratch, count: usize| {
            for _ in 0..count {
                walk(&small, &small_table, 0, usize::MAX, scratch);
            }
        };

        let first = walk(&large, &large_table, 0, usize::MAX, &mut scratch);
        assert_eq!(
            first.1.len(),
            LARGE,
            "the first walk did not stamp the whole large partition, so the wrap has nothing \
             stale to trip over"
        );
        small_walks(&mut scratch, 254);
        assert_eq!(
            scratch.generation,
            u8::MAX,
            "the walks did not bring the generation to its last value"
        );
        assert_eq!(
            scratch.seen.len(),
            LARGE,
            "the small walks replaced the buffer, so its tail holds nothing stale"
        );
        let wrapped = walk(&large, &large_table, 0, usize::MAX, &mut scratch);
        assert_eq!(
            scratch.generation, 1,
            "the walk after the last generation did not restart the numbering"
        );
        assert_eq!(
            wrapped,
            reference_walk(&large, &large_table, 0, usize::MAX),
            "the walk on the wrap left a different list, expanded a different sequence or \
             measured a different number of distances"
        );

        // 254 walks to the last generation and one more to wrap, all of them
        // leaving the large partition's slots as the wrap cleared them.
        small_walks(&mut scratch, 255);
        assert_eq!(
            scratch.generation, 1,
            "the small partition did not take the second wrap"
        );
        small_walks(&mut scratch, 253);
        let a_cycle_later = walk(&large, &large_table, 0, usize::MAX, &mut scratch);
        assert_eq!(
            scratch.generation,
            u8::MAX,
            "the large partition was not walked at the last generation of the cycle"
        );
        assert_eq!(
            a_cycle_later,
            reference_walk(&large, &large_table, 0, usize::MAX),
            "a walk a whole cycle after a wrap left a different list, expanded a different \
             sequence or measured a different number of distances"
        );
    }

    /// [`PlainList`] ruled by a [`StopRule`] straight from its definition:
    /// nothing is thrown away, and the rule is consulted only when a vertex is
    /// asked for - expand the nearest unexpanded one if it is among the `rank`
    /// nearest or nearer than `(1 + margin)^2` times the `rank`-th's distance.
    struct RuledList {
        plain: PlainList,
        rank: usize,
        keep: usize,
        factor: f32,
    }

    impl RuledList {
        fn new(size: usize, stop: StopRule) -> Self {
            Self {
                plain: PlainList::new(size),
                rank: stop.rank,
                keep: stop.keep,
                factor: (1.0 + stop.margin).powi(2),
            }
        }

        fn offer(&mut self, id: u32, distance: f32) {
            self.plain.offer(id, distance);
        }

        fn bar(&self) -> Option<OrderedFloat> {
            self.plain
                .list
                .get(self.rank - 1)
                .map(|(node, _)| OrderedFloat(self.factor * node.dist.0))
        }

        fn next_unexpanded(&mut self) -> Option<OrderedNode> {
            let position = self.plain.list.iter().position(|(_, expanded)| !expanded)?;
            let beyond = self
                .bar()
                .is_some_and(|bar| bar <= self.plain.list[position].0.dist);
            if position >= self.rank && beyond {
                return None;
            }
            self.plain.list[position].1 = true;
            Some(self.plain.list[position].0.clone())
        }

        /// What the rule has any use for: the nearest `keep`, and after them
        /// whatever is nearer than the bar.
        fn kept(&self) -> Vec<(u32, u32, bool)> {
            let held = plainly_held(&self.plain);
            let cut = match self.bar() {
                Some(bar) if held.len() > self.keep => {
                    self.keep
                        + self.plain.list[self.keep..]
                            .iter()
                            .take_while(|(node, _)| node.dist < bar)
                            .count()
                }
                _ => held.len(),
            };
            held[..cut].to_vec()
        }
    }

    /// A list with a stop rule keeps the front of what the ruled plain list
    /// holds, and expands exactly what the rule allows, after every offer and
    /// every expansion.
    ///
    /// The front and not the whole: a list may throw away what the rule has no
    /// use for, and what it keeps has to be exactly the rest, or a candidate
    /// the re-score is owed - or one the walk should still expand - went
    /// missing. At a margin of zero it also expands exactly what the plain list
    /// of length `rank` does, whatever it keeps, and at `rank == keep` it is
    /// that list entry for entry: the walk as it was.
    ///
    /// The distances are chosen so that bars of 1.5625, 2.25 and 4 times one of
    /// them land exactly on another, and so exercise the strict side of every
    /// comparison. They are finite, as every distance a walk measures is.
    #[test]
    fn a_list_with_a_stop_rule_is_the_front_of_the_list_it_rules() {
        const DISTANCES: [f32; 12] = [
            -1.0, 0.0, 0.5, 1.0, 2.0, 2.25, 4.0, 4.5, 6.25, 9.0, 16.0, 25.0,
        ];
        let mut rng = SmallRng::seed_from_u64(13);
        // Where each half of the rule acted in the list itself: a stop with
        // unexpanded entries still held, and an insert that did not lengthen a
        // list the cap had room in. A fixture that never reached one of them
        // would pass the comparisons below without testing it.
        let (mut stopped, mut cut) = (0usize, 0usize);
        for (rank, keep) in [(1, 1), (1, 3), (2, 5), (3, 3), (5, 16), (16, 16)] {
            for size in [keep, keep + 1, 64, usize::MAX] {
                for margin in [0.0f32, 0.25, 0.5, 1.0] {
                    let stop = StopRule { margin, rank, keep };
                    for sequence in 0..20 {
                        let mut list = SearchList::with_margin(size, 256, stop);
                        let mut ruled = RuledList::new(size, stop);
                        let mut plain = (margin == 0.0).then(|| PlainList::new(rank));
                        for step in 0..100 {
                            let context = || {
                                format!(
                                    "rank {rank}, keep {keep}, L = {size}, margin {margin}, \
                                     sequence {sequence}, step {step}"
                                )
                            };
                            if rng.random_bool(0.7) {
                                let id = rng.random_range(0..256);
                                let distance = DISTANCES[rng.random_range(0..DISTANCES.len())];
                                let before = held(&list);
                                list.offer(id, distance);
                                let after = held(&list);
                                cut += usize::from(
                                    after != before
                                        && after.len() <= before.len()
                                        && before.len() < size,
                                );
                                ruled.offer(id, distance);
                                if let Some(plain) = plain.as_mut() {
                                    plain.offer(id, distance);
                                }
                            } else {
                                let expansions = if rng.random_bool(0.25) { 4 } else { 1 };
                                for _ in 0..expansions {
                                    let expected = ruled.next_unexpanded().map(bits);
                                    let expanded = list.next_unexpanded().map(bits);
                                    stopped += usize::from(
                                        expanded.is_none()
                                            && list.list.iter().any(|entry| !entry.expanded),
                                    );
                                    assert_eq!(
                                        expanded,
                                        expected,
                                        "{}: the lists expanded different vertices",
                                        context()
                                    );
                                    if let Some(plain) = plain.as_mut() {
                                        assert_eq!(
                                            plain.next_unexpanded().map(bits),
                                            expected,
                                            "{}: a margin of zero expanded what the list of \
                                             length rank does not",
                                            context()
                                        );
                                    }
                                }
                            }
                            let kept = ruled.kept();
                            assert_eq!(
                                held(&list),
                                kept,
                                "{}: the lists hold different entries",
                                context()
                            );
                            if let Some(plain) = plain.as_ref().filter(|_| rank == keep) {
                                assert_eq!(
                                    plainly_held(plain),
                                    kept,
                                    "{}: a margin of zero kept what the list of that length \
                                     does not",
                                    context()
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            stopped > 0 && cut > 0,
            "the list stopped {stopped} walks with entries left and cut {cut} times, so one half \
             of the rule was never exercised"
        );
    }

    /// Algorithm 1 with a stop rule, one vertex at a time: what a lazy walk of
    /// width one does with its list.
    fn ruled_walk(
        graph: &PartitionGraph,
        table: &Table,
        entry_point: u32,
        stop: StopRule,
        search_list_size: usize,
    ) -> (Vec<u32>, u64) {
        let mut scratch = SearchScratch::new(graph.len());
        scratch.begin();
        scratch.mark(entry_point);
        let mut list = SearchList::with_margin(search_list_size, graph.len(), stop);
        list.offer(entry_point, table.distance(entry_point));
        let mut comparisons = 1;
        let mut expanded = Vec::new();
        while let Some(node) = list.next_unexpanded() {
            expanded.push(node.id);
            for neighbor in graph.neighbors(node.id).unwrap() {
                if scratch.mark(*neighbor) {
                    comparisons += 1;
                    list.offer(*neighbor, table.distance(*neighbor));
                }
            }
        }
        (expanded, comparisons)
    }

    /// A wider margin walks on from where a narrower one stopped: the same
    /// vertices expanded in the same order, then more, and never fewer
    /// distances.
    ///
    /// This is what makes a sweep over the margin a sweep over one walk's
    /// stopping points rather than over different walks, so that recall and
    /// work move one way along it. It holds for one vertex a hop, under any cap
    /// both walks share - one just past `keep` binds all the time here - and for
    /// distances of at least zero: below zero a wider margin puts the bar lower,
    /// not higher.
    #[test]
    fn a_wider_margin_walks_on_from_where_a_narrower_one_stopped() {
        const DISTANCES: [f32; 6] = [0.0, 1.0, 1.0, 2.0, 3.0, 5.0];
        let mut rng = SmallRng::seed_from_u64(17);
        let mut further = 0usize;
        for num_vertices in [2, 17, 64, 200] {
            for _ in 0..24 {
                let graph = random_graph(num_vertices, 4, &mut rng);
                let table = Table(
                    (0..num_vertices)
                        .map(|_| DISTANCES[rng.random_range(0..DISTANCES.len())])
                        .collect(),
                );
                let entry_point = rng.random_range(0..num_vertices as u32);
                for (rank, keep) in [(1, 1), (2, 4), (5, 5)] {
                    for size in [keep + 1, usize::MAX] {
                        let mut narrower: Option<(Vec<u32>, u64)> = None;
                        for margin in [0.0f32, 0.25, 1.0, 4.0] {
                            let walked = ruled_walk(
                                &graph,
                                &table,
                                entry_point,
                                StopRule { margin, rank, keep },
                                size,
                            );
                            if let Some((expanded, comparisons)) = &narrower {
                                assert!(
                                    walked.0.starts_with(expanded),
                                    "over {num_vertices} vertices at rank {rank}, L = {size}, \
                                     margin {margin} expanded {:?} where a narrower margin \
                                     expanded {expanded:?}",
                                    walked.0
                                );
                                assert!(
                                    walked.1 >= *comparisons,
                                    "margin {margin} measured {} distances against \
                                     {comparisons} for a narrower one",
                                    walked.1
                                );
                                further += usize::from(walked.0.len() > expanded.len());
                            }
                            narrower = Some(walked);
                        }
                    }
                }
            }
        }
        assert!(
            further > 0,
            "no wider margin ever expanded more than a narrower one, so the fixture never let the \
             margin decide anything"
        );
    }
}
