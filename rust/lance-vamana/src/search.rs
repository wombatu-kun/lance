// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Greedy search over one partition's graph.
//!
//! This is Algorithm 1 of the DiskANN paper, and it is used twice: a build runs
//! it once per vertex to collect the visited set a prune works from, and a query
//! runs it to answer. Both want the same thing, so it lives on its own.

use std::cell::Cell;

use lance_core::{Error, Result};
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::storage::DistCalculator;

use crate::partition::PartitionGraph;

/// Counts distance computations.
///
/// Recall is only half of a graph index's story; the other half is what it cost
/// to get there. Lance measures neither for HNSW - its builder takes a metrics
/// argument and ignores it - so this crate carries its own from the first
/// algorithm, before there is anything to be tempted to flatter.
#[derive(Debug, Default)]
pub struct Comparisons(Cell<u64>);

impl Comparisons {
    #[inline]
    pub fn record(&self, count: u64) {
        self.0.set(self.0.get() + count);
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

    fn begin(&mut self) {
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
    fn mark(&mut self, id: u32) -> bool {
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
#[derive(Debug, Clone)]
struct Candidate {
    node: OrderedNode,
    expanded: bool,
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
    let mut list = Vec::with_capacity(search_list_size + 1);
    list.push(Candidate {
        node: OrderedNode::new(entry_point, OrderedFloat(query.distance(entry_point))),
        expanded: false,
    });
    let mut visited = Vec::new();

    while let Some(position) = list.iter().position(|candidate| !candidate.expanded) {
        list[position].expanded = true;
        let nearest_unexpanded = list[position].node.clone();
        visited.push(nearest_unexpanded.clone());

        for neighbor in graph.neighbors(nearest_unexpanded.id) {
            if !scratch.mark(*neighbor) {
                continue;
            }
            comparisons.record(1);
            let distance = OrderedFloat(query.distance(*neighbor));
            let at = list.partition_point(|candidate| candidate.node.dist <= distance);
            if at >= search_list_size {
                continue;
            }
            list.insert(
                at,
                Candidate {
                    node: OrderedNode::new(*neighbor, distance),
                    expanded: false,
                },
            );
            list.truncate(search_list_size);
        }
    }

    Ok(SearchResult {
        candidates: list.into_iter().map(|candidate| candidate.node).collect(),
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
    fn the_search_list_never_grows_past_its_bound() {
        let graph = path_graph(64);
        for search_list_size in [1, 2, 7, 64, 128] {
            let (result, _) = search(&graph, &line_storage(64), 63, 0, search_list_size);
            assert!(
                result.candidates.len() <= search_list_size,
                "list of {} exceeds the bound {search_list_size}",
                result.candidates.len()
            );
            assert!(
                result.candidates.windows(2).all(|pair| pair[0] <= pair[1]),
                "the search list came back unsorted at L = {search_list_size}"
            );
        }
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
}
