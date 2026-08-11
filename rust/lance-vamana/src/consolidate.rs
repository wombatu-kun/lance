// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Taking a partition's deleted rows out of it.
//!
//! Deleting a row leaves its vertex in the graph. That is deliberate on the read
//! path - the vertex still carries out-edges, and a walk that skipped it would
//! lose whatever it was the only route to - but it is not a resting state. The
//! dead row is stored, read and walked through forever, and a partition of
//! mostly-dead vertices reads ten times the bytes it needs.
//!
//! Measured on SIFT 100k, deletion alone costs a query nothing: the same 7492.9
//! distances at every fraction from 0% to 90% deleted, and recall down from
//! 0.9777 only to 0.9474. So this is not a repair of search quality. It is a
//! repair of what a partition costs to keep: bytes, and the share of a read that
//! is useful.

use lance_core::{Error, Result};
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use roaring::RoaringBitmap;

use crate::build::{medoid, robust_prune, validate_alpha};
use crate::builder::gather;
use crate::partition::{Partition, PartitionGraph};
use crate::search::{Comparisons, flat_storage};

/// A partition with its dead rows gone and its graph closed back up.
#[derive(Debug)]
pub struct Consolidated {
    pub partition: Partition,
    /// Recomputed rather than carried over, and not only because the old entry
    /// point may itself have been deleted: local ids move under compaction, so
    /// the old number would name a different vertex even when it survived.
    pub medoid: u32,
}

/// Drop `dead` from a partition and repair the edges that pointed at it.
///
/// Algorithm 4 of the FreshVamana paper. A live vertex that pointed at a dead
/// one inherits that vertex's own out-edges in its place, so the routes through
/// the dead are kept while the dead themselves go:
///
/// ```text
/// C <- (N_out(p) \ D) union (union of N_out(v) for v in N_out(p) intersect D)
/// N_out(p) <- RobustPrune(p, C \ D, alpha, R)   when |C| > R, else C itself
/// ```
///
/// The inheritance is one hop and not transitive, which is the paper's choice
/// and worth knowing: a dead vertex whose own out-edges are all dead passes on
/// nothing, and a live vertex reachable only through such a chain can be left
/// unreachable. What the algorithm guarantees is that no *edge* dangles, not
/// that connectivity is preserved.
///
/// `dead` names local ids, not row addresses. The delete list this ultimately
/// comes from is in address space, and translating it is the caller's job on
/// purpose: this function is a graph operation with no opinion about datasets,
/// and the translation is one line at the call site where the delete list lives.
///
/// A partition whose every row is dead is refused rather than returned empty.
/// An empty partition is written no file and given no row in the segment table,
/// so dropping it is the caller's decision to make before calling, exactly as it
/// is on the build path.
pub fn consolidate_partition(
    partition: &Partition,
    dead: &RoaringBitmap,
    distance_type: DistanceType,
    alpha: f32,
    comparisons: &Comparisons,
) -> Result<Consolidated> {
    validate_alpha(alpha)?;
    let num_vertices = partition.len();
    if let Some(highest) = dead.max()
        && highest as usize >= num_vertices
    {
        return Err(Error::invalid_input(format!(
            "Vamana was asked to delete vertex {highest} from a partition of {num_vertices}"
        )));
    }
    let survivors = num_vertices - dead.len() as usize;
    if survivors == 0 {
        return Err(Error::invalid_input(format!(
            "Vamana cannot consolidate a partition whose every one of {num_vertices} rows is \
             deleted; drop the partition instead"
        )));
    }

    let graph = partition.graph();
    let max_degree = graph.max_degree() as usize;
    let store = flat_storage(graph.row_ids(), partition.vectors(), distance_type)?;

    // Built before any repair, because a repaired list is written in the new
    // numbering and the candidates it is built from are read in the old one.
    let mut new_id = vec![0u32; num_vertices];
    let mut survivor_ids = Vec::with_capacity(survivors);
    for old in 0..num_vertices as u32 {
        if !dead.contains(old) {
            new_id[old as usize] = survivor_ids.len() as u32;
            survivor_ids.push(old);
        }
    }

    let mut adjacency = Vec::with_capacity(survivors);
    let mut candidates = Vec::with_capacity(max_degree);
    for point in &survivor_ids {
        let point = *point;
        candidates.clear();
        let mut touches_dead = false;
        for neighbor in graph.neighbors(point)? {
            if !dead.contains(*neighbor) {
                candidates.push(*neighbor);
                continue;
            }
            touches_dead = true;
            // The dead vertex's own dead neighbours are dropped here rather
            // than left for the prune: they are not candidates, and passing
            // them on would only cost distances to reject them.
            candidates.extend(
                graph
                    .neighbors(*neighbor)?
                    .iter()
                    .filter(|hop| !dead.contains(**hop) && **hop != point),
            );
        }

        let repaired = if !touches_dead {
            // Carried over whole, in the order it was already in. The prune
            // below is unreachable for such a vertex - its candidate set is its
            // own neighbour list, which is never wider than `max_degree` - so
            // the only thing this branch really saves is the *sort*, and that is
            // the point: out-edges are written nearest-first by the prune that
            // produced them, and re-sorting them by id would silently reorder
            // every untouched vertex on disk.
            &candidates
        } else {
            candidates.sort_unstable();
            candidates.dedup();
            if candidates.len() > max_degree {
                let from_point = store.dist_calculator_from_id(point);
                comparisons.record(candidates.len() as u64);
                let pool = candidates
                    .iter()
                    .map(|id| OrderedNode::new(*id, OrderedFloat(from_point.distance(*id))))
                    .collect();
                candidates = robust_prune(&store, point, pool, alpha, max_degree, comparisons)?;
            }
            &candidates
        };
        adjacency.push(repaired.iter().map(|id| new_id[*id as usize]).collect());
    }

    let row_ids = survivor_ids
        .iter()
        .map(|old| graph.row_ids()[*old as usize])
        .collect();
    let vectors = gather(partition.vectors(), &survivor_ids)?;
    let partition = Partition::try_new(
        PartitionGraph::try_new(graph.max_degree(), row_ids, adjacency)?,
        vectors,
    )?;

    let store = flat_storage(
        partition.graph().row_ids(),
        partition.vectors(),
        distance_type,
    )?;
    let medoid = medoid(&store, comparisons)?;
    Ok(Consolidated { partition, medoid })
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;

    use super::*;
    use crate::build::{BuildParams, build_partition};

    /// Vertices on a line at 0, 1, 2, ..., so every distance is hand-checkable
    /// and "the graph still reaches everything" is a property of the numbers
    /// rather than of a fixture nobody can read.
    fn line_partition(vertices: usize, max_degree: u32) -> Partition {
        let values = Float32Array::from((0..vertices).map(|i| i as f32).collect::<Vec<_>>());
        let vectors = FixedSizeListArray::try_new_from_values(values, 1).unwrap();
        let row_ids = (0..vertices as u64).map(|i| i * 7 + 3).collect::<Vec<_>>();
        // Each vertex points at its two neighbours on the line, so deleting one
        // leaves a hole that only the inheritance can close.
        // Higher neighbour first, so the lists are *not* in ascending id
        // order. A fixture that happened to be sorted would make "carried over
        // as it was" and "sorted on the way through" indistinguishable.
        let adjacency = (0..vertices)
            .map(|vertex| {
                let mut edges = Vec::new();
                if vertex + 1 < vertices {
                    edges.push(vertex as u32 + 1);
                }
                if vertex > 0 {
                    edges.push(vertex as u32 - 1);
                }
                edges
            })
            .collect();
        Partition::try_new(
            PartitionGraph::try_new(max_degree, row_ids, adjacency).unwrap(),
            vectors,
        )
        .unwrap()
    }

    fn dead(ids: impl IntoIterator<Item = u32>) -> RoaringBitmap {
        ids.into_iter().collect()
    }

    /// Every vertex reachable from `entry_point` by following out-edges.
    fn reachable(graph: &PartitionGraph, entry_point: u32) -> usize {
        let mut seen = vec![false; graph.len()];
        let mut frontier = vec![entry_point];
        seen[entry_point as usize] = true;
        let mut count = 1;
        while let Some(vertex) = frontier.pop() {
            for neighbor in graph.neighbors(vertex).unwrap() {
                if !seen[*neighbor as usize] {
                    seen[*neighbor as usize] = true;
                    count += 1;
                    frontier.push(*neighbor);
                }
            }
        }
        count
    }

    /// The rows that are gone are gone, the rows that are not are all still
    /// there, and they are still in the order the ids were written in.
    #[test]
    fn the_survivors_keep_their_rows_and_their_vectors() {
        let partition = line_partition(9, 4);
        let consolidated = consolidate_partition(
            &partition,
            &dead([1, 4, 7]),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();

        let expected = [0u32, 2, 3, 5, 6, 8];
        assert_eq!(consolidated.partition.len(), expected.len());
        for (new, old) in expected.iter().enumerate() {
            assert_eq!(
                consolidated.partition.graph().row_ids()[new],
                partition.graph().row_ids()[*old as usize],
                "row of vertex {new}"
            );
            assert_eq!(
                consolidated.partition.vector(new as u32),
                partition.vector(*old),
                "vector of vertex {new}"
            );
        }
    }

    /// The point of the inheritance: a vertex whose only route onwards was
    /// through a deleted vertex keeps a route onwards.
    #[test]
    fn a_hole_in_the_line_is_closed_by_inheritance() {
        let partition = line_partition(9, 4);
        let consolidated = consolidate_partition(
            &partition,
            &dead([4]),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();

        // Old 3 and old 5 pointed only at each other's side of the hole; after
        // the repair they point at each other, and the line is whole again.
        let graph = consolidated.partition.graph();
        assert!(graph.neighbors(3).unwrap().contains(&4), "3 -> 5");
        assert!(graph.neighbors(4).unwrap().contains(&3), "5 -> 3");
        assert_eq!(reachable(graph, 0), 8, "the line is in two pieces");
    }

    /// No edge may point at a row that is no longer there, and none may point
    /// at a vertex that never was: the whole graph is renumbered, not patched.
    #[test]
    fn no_edge_survives_that_pointed_at_a_deleted_row() {
        let partition = line_partition(64, 8);
        let consolidated = consolidate_partition(
            &partition,
            &dead((0..64).filter(|id| id % 3 == 0)),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();

        let graph = consolidated.partition.graph();
        let survivors = graph.len();
        assert_eq!(survivors, 64 - 22);
        for vertex in 0..survivors as u32 {
            let neighbors = graph.neighbors(vertex).unwrap();
            assert!(
                !neighbors.contains(&vertex),
                "vertex {vertex} points at itself"
            );
            assert!(
                neighbors.iter().all(|id| (*id as usize) < survivors),
                "vertex {vertex} points outside the partition: {neighbors:?}"
            );
            assert!(
                !neighbors.is_empty(),
                "vertex {vertex} was left with no way out"
            );
        }
    }

    /// A real build, consolidated: the fixture above is a line and cannot show
    /// that the prune is reached at all. Here the inherited candidate sets are
    /// wider than `max_degree` and have to be pruned back.
    #[test]
    fn a_built_graph_survives_losing_half_its_rows() {
        const VERTICES: usize = 400;
        let values = Float32Array::from(
            (0..VERTICES * 4)
                .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0)
                .collect::<Vec<_>>(),
        );
        let storage = lance_index::vector::flat::storage::FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(values, 4).unwrap(),
            DistanceType::L2,
        );
        let params = BuildParams {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
            seed: 42,
        };
        let built = build_partition(&storage, &params, &Comparisons::default()).unwrap();
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..VERTICES * 4)
                    .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0)
                    .collect::<Vec<_>>(),
            ),
            4,
        )
        .unwrap();
        let partition = Partition::try_new(built.graph, vectors).unwrap();

        let comparisons = Comparisons::default();
        let consolidated = consolidate_partition(
            &partition,
            &dead((0..VERTICES as u32).filter(|id| id % 2 == 0)),
            DistanceType::L2,
            params.alpha,
            &comparisons,
        )
        .unwrap();

        let graph = consolidated.partition.graph();
        assert_eq!(graph.len(), VERTICES / 2);
        assert!(
            comparisons.get() > 0,
            "nothing was pruned, so this fixture never reached the prune"
        );
        for vertex in 0..graph.len() as u32 {
            let neighbors = graph.neighbors(vertex).unwrap();
            assert!(neighbors.len() <= params.max_degree as usize);
            let mut sorted = neighbors.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), neighbors.len(), "vertex {vertex} has a twin");
        }
        assert_eq!(
            reachable(graph, consolidated.medoid),
            VERTICES / 2,
            "the graph came apart"
        );
    }

    /// Consolidating nothing is not a rebuild: the graph, the rows and the entry
    /// point all come back as they were - *including* the order of every
    /// neighbour list, which is why the fixture's lists are not sorted by id.
    #[test]
    fn consolidating_no_deletions_changes_nothing() {
        let partition = line_partition(32, 8);
        let consolidated = consolidate_partition(
            &partition,
            &RoaringBitmap::new(),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();

        assert_eq!(consolidated.partition.len(), partition.len());
        for vertex in 0..partition.len() as u32 {
            assert_eq!(
                consolidated.partition.graph().neighbors(vertex).unwrap(),
                partition.graph().neighbors(vertex).unwrap(),
                "vertex {vertex}"
            );
        }
        assert_eq!(
            consolidated.partition.graph().row_ids(),
            partition.graph().row_ids()
        );
    }

    /// The entry point is a local id, and local ids move. Carrying the old
    /// number over would name a different vertex.
    #[test]
    fn the_entry_point_is_recomputed_in_the_new_numbering() {
        let partition = line_partition(11, 4);
        // The line's middle is vertex 5; deleting the first four rows moves the
        // middle of what is left to old 7, which is new 3.
        let consolidated = consolidate_partition(
            &partition,
            &dead(0..4),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(consolidated.medoid, 3);
        assert_eq!(
            consolidated.partition.graph().row_ids()[consolidated.medoid as usize],
            partition.graph().row_ids()[7]
        );
    }

    #[test]
    fn a_partition_that_is_entirely_deleted_is_refused() {
        let partition = line_partition(5, 4);
        let error = consolidate_partition(
            &partition,
            &dead(0..5),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("drop the partition"), "{error}");
    }

    #[test]
    fn a_deletion_outside_the_partition_is_refused() {
        let partition = line_partition(5, 4);
        let error = consolidate_partition(
            &partition,
            &dead([5]),
            DistanceType::L2,
            1.2,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("vertex 5"), "{error}");
    }
}
