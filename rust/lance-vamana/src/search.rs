// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Greedy search over one partition's graph.
//!
//! This is Algorithm 1 of the DiskANN paper, and it is used twice: a build runs
//! it once per vertex to collect the visited set a prune works from, and a query
//! runs it to answer. Both want the same thing, so it lives on its own.

use std::cell::Cell;
use std::sync::Arc;

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
#[derive(Debug)]
pub struct SearchScratch {
    seen: Vec<u32>,
    generation: u32,
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
            // Four billion searches later the stamps stop being unique, so the
            // marks are cleared once and numbering restarts.
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
        }
    }

    /// Offer a vertex at `distance`, keeping it if it beats the back of the list.
    ///
    /// Nothing here checks whether `id` is already in the list: a caller must
    /// have marked it in its [`SearchScratch`] first, which is what makes a
    /// vertex measured once and offered once.
    pub fn offer(&mut self, id: u32, distance: f32) {
        let distance = OrderedFloat(distance);
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
    }

    /// The nearest vertex whose out-edges have not been followed, marked as
    /// followed.
    ///
    /// Called `n` times in a row without an [`Self::offer`] between them, it
    /// yields the `n` nearest unexpanded vertices - which is exactly the
    /// frontier a lazy hop fetches in one request.
    pub fn next_unexpanded(&mut self) -> Option<OrderedNode> {
        let position = self.list.iter().position(|entry| !entry.expanded)?;
        self.list[position].expanded = true;
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
    use arrow_array::Float32Array;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::storage::VectorStore;
    use lance_linalg::distance::DistanceType;

    use super::*;

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
}
