// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adding vertices to a graph that is already built.
//!
//! A build is repeated insertion, and this is the step it repeats: search the
//! graph as it currently stands for the point being added, prune what the search
//! visited into the point's out-edges, and give each chosen neighbour an edge
//! back - re-pruning it when its list was already full, which for a list that is
//! still exactly what a prune returned means checking the new point alone
//! against it, bar a zero separation (`build::admit`). [`crate::build`] calls
//! `insert_point` once per vertex twice over; this module's own
//! [`insert_into_partition`] calls it once per new vertex against a graph that
//! is already navigable, which is why it needs one pass rather than two.
//!
//! Sharing the step is not tidiness. A graph built one way and extended another
//! would drift apart silently: the two would agree on every invariant a test can
//! state - degree, no dangling edge, reachability - and disagree only on how good
//! the graph is, which is the one thing an assertion cannot check.
//!
//! Local ids of existing vertices never move. Edges are written in local ids, so
//! renumbering would mean rewriting every neighbour list in the partition to say
//! exactly what it already said; new vertices take the ids after the last one.

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray};
use arrow_schema::{DataType, Field};
use arrow_select::concat::concat;
use lance_core::{Error, Result};
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::build::{Admission, BuildParams, admit, medoid, prune, validate_alpha};
use crate::partition::{Partition, PartitionGraph};
use crate::search::{Comparisons, SearchScratch, flat_storage, greedy_search};

/// What one insertion runs under.
///
/// `alpha` is here rather than taken from [`BuildParams`] because a build makes
/// two passes at two values of it and changes nothing else between them.
pub(crate) struct Linking {
    pub alpha: f32,
    pub search_list_size: usize,
}

/// Reusable buffers for a run of insertions.
///
/// One insertion searches the graph and then rewrites up to `max_degree`
/// neighbour lists, and both need scratch. A build is one insertion per vertex
/// twice over, so allocating either per insertion would put millions of
/// allocations in the innermost loop of the crate.
pub(crate) struct InsertScratch {
    search: SearchScratch,
    /// One vertex's out-edges, read out so that the graph can be written back.
    edges: Vec<u32>,
    /// Whether each vertex's out-edges are still exactly what a prune returned
    /// without its coincident fill, the one kind of list [`admit`] can take a
    /// back-edge into. Such a list holds at the alpha it was pruned at and any
    /// larger one, never a smaller one, so the insertions one scratch serves must
    /// not lower alpha.
    ///
    /// False for the random graph a build starts from and for a graph read back
    /// from disk, whose history is unknown; set by a prune that did not use the
    /// fill; cleared by an unchecked append.
    is_prune_output: Vec<bool>,
}

impl InsertScratch {
    pub(crate) fn new(num_vertices: usize, max_degree: u32) -> Self {
        Self {
            search: SearchScratch::new(num_vertices),
            // Plus one: a full list has to hold the contender before the prune
            // decides which of the `max_degree + 1` of them stays.
            edges: Vec::with_capacity(max_degree as usize + 1),
            is_prune_output: vec![false; num_vertices],
        }
    }
}

/// Give `point` its out-edges, and give the vertices it chose an edge back.
///
/// The insertion of the DiskANN and FreshVamana papers. `point` must already be
/// a vertex of `graph` - with whatever out-edges it has, which is none for one
/// just appended and a full list for one a build is re-linking on its second
/// pass.
///
/// The back-edges are what make the graph navigable *to* the new point rather
/// than only *from* it. Each one is added outright while the neighbour has room,
/// and fought for through the prune when it does not: a neighbour at full degree
/// gives up a slot only if the new point beats an occupant on the same diversity
/// rule that filled the list in the first place.
///
/// A full list that is still exactly what a prune returned does not need the
/// prune run over it again: [`admit`] reaches the same answer by checking the new
/// point against the members alone, and leaves a zero separation to the prune. At
/// `alpha = 1.2` such a re-prune keeps almost every member, so the list stays full
/// and the next back-edge would pay the whole prune again - 2470-2548 distances
/// each time in builds over every tenth vector of the four benchmark datasets,
/// where it was 57% of GloVe's.
pub(crate) fn insert_point<S: VectorStore>(
    graph: &mut PartitionGraph,
    store: &S,
    scratch: &mut InsertScratch,
    linking: &Linking,
    point: u32,
    entry_point: u32,
    comparisons: &Comparisons,
) -> Result<()> {
    let max_degree = graph.max_degree() as usize;
    let from_point = store.dist_calculator_from_id(point);
    let mut candidates = greedy_search(
        graph,
        &from_point,
        entry_point,
        linking.search_list_size,
        &mut scratch.search,
        comparisons,
    )?
    .visited;
    // The paper folds the current out-edges into the candidate set inside the
    // prune; doing it here keeps the prune ignorant of the graph, which is what
    // lets the back-edge case below reuse it.
    comparisons.record(graph.neighbors(point)?.len() as u64);
    candidates.extend(
        graph.neighbors(point)?.iter().map(|neighbor| {
            OrderedNode::new(*neighbor, OrderedFloat(from_point.distance(*neighbor)))
        }),
    );

    let selected = prune(
        store,
        point,
        candidates,
        linking.alpha,
        max_degree,
        comparisons,
    )?;
    graph.set_neighbors(point, &selected.neighbors)?;
    scratch.is_prune_output[point as usize] = !selected.has_coincident_fill;

    for neighbor in &selected.neighbors {
        let neighbor = *neighbor;
        scratch.edges.clear();
        scratch.edges.extend_from_slice(graph.neighbors(neighbor)?);
        if scratch.edges.contains(&point) {
            continue;
        }
        if scratch.edges.len() < max_degree {
            scratch.edges.push(point);
            graph.set_neighbors(neighbor, &scratch.edges)?;
            // Appended without the rule, so no longer a prune's output.
            scratch.is_prune_output[neighbor as usize] = false;
            continue;
        }
        // Full: the back-edge has to earn its place against the rest.
        if scratch.is_prune_output[neighbor as usize] {
            match admit(
                store,
                neighbor,
                &scratch.edges,
                point,
                linking.alpha,
                max_degree,
                comparisons,
            )? {
                Admission::Refused => continue,
                Admission::Admitted(admitted) => {
                    graph.set_neighbors(neighbor, &admitted)?;
                    continue;
                }
                Admission::Undecided => {}
            }
        }
        let from_neighbor = store.dist_calculator_from_id(neighbor);
        comparisons.record(scratch.edges.len() as u64 + 1);
        let contenders = scratch
            .edges
            .iter()
            .chain(std::iter::once(&point))
            .map(|id| OrderedNode::new(*id, OrderedFloat(from_neighbor.distance(*id))))
            .collect();
        let repruned = prune(
            store,
            neighbor,
            contenders,
            linking.alpha,
            max_degree,
            comparisons,
        )?;
        graph.set_neighbors(neighbor, &repruned.neighbors)?;
        scratch.is_prune_output[neighbor as usize] = !repruned.has_coincident_fill;
    }
    Ok(())
}

/// A partition with new vertices linked into its graph.
#[derive(Debug)]
pub struct Inserted {
    pub partition: Partition,
    /// Recomputed over the enlarged partition.
    ///
    /// Unlike consolidation this does not have to be - local ids do not move, so
    /// the old entry point still names the vertex it always named. It is
    /// recomputed because the rule is that the entry point is the middle of what
    /// the partition holds, and what it holds has changed. The cost is `O(n*d)`
    /// against the batch's own `O(b*L*R*d)`, so it is negligible for a batch of
    /// any size and the dominant term for a batch of ten into a partition of ten
    /// thousand. It is counted in `comparisons` either way.
    pub medoid: u32,
}

/// Link `row_ids` and their `vectors` into `partition`, and say where a search
/// of the result should start.
///
/// One pass, where a build makes two. The first pass of a build exists to turn a
/// random graph into a navigable one before the pruning slack is applied to it;
/// here the graph the new points search is already navigable, and the paper
/// inserts into it at the full `alpha` directly.
///
/// `entry_point` is where each insertion's search starts, and it stays fixed for
/// the whole batch even as the medoid drifts - it is a starting position, not an
/// answer, and recomputing it per insertion would cost `O(n*d)` per new row.
///
/// The batch is inserted in an order drawn from `params.seed` rather than in the
/// order it arrives. Arrival order is scan order, and scan order is dataset
/// order, which for anything sorted or clustered on write would insert a whole
/// region of the space before its neighbours exist.
pub fn insert_into_partition(
    partition: &Partition,
    row_ids: &[u64],
    vectors: &FixedSizeListArray,
    entry_point: u32,
    distance_type: DistanceType,
    params: &BuildParams,
    comparisons: &Comparisons,
) -> Result<Inserted> {
    validate_alpha(params.alpha)?;
    if params.search_list_size == 0 {
        return Err(Error::invalid_input(
            "Vamana search list size must be greater than zero".to_string(),
        ));
    }
    if row_ids.len() != vectors.len() {
        return Err(Error::invalid_input(format!(
            "Vamana was given {} row ids and {} vectors to insert",
            row_ids.len(),
            vectors.len()
        )));
    }
    if row_ids.is_empty() {
        return Err(Error::invalid_input(
            "Vamana was asked to insert nothing; a partition that gains no rows is written as it \
             stands"
                .to_string(),
        ));
    }
    if partition.is_empty() {
        return Err(Error::invalid_input(
            "Vamana cannot insert into an empty partition; there is no vertex to search from, so \
             build one instead"
                .to_string(),
        ));
    }
    if entry_point as usize >= partition.len() {
        return Err(Error::invalid_input(format!(
            "Vamana was given entry point {entry_point} for a partition of {} vertices",
            partition.len()
        )));
    }
    if vectors.value_length() as u32 != partition.dimension() {
        return Err(Error::invalid_input(format!(
            "Vamana cannot insert vectors of dimension {} into a partition of dimension {}",
            vectors.value_length(),
            partition.dimension()
        )));
    }
    // Checked rather than taken from the graph, which is where the width really
    // comes from: a caller whose parameters disagree with the partition is a
    // caller about to write a segment that disagrees with its own table.
    if params.max_degree != partition.graph().max_degree() {
        return Err(Error::invalid_input(format!(
            "Vamana was asked to insert at degree {} into a partition built at degree {}",
            params.max_degree,
            partition.graph().max_degree()
        )));
    }
    if vectors.null_count() != 0 || vectors.values().null_count() != 0 {
        return Err(Error::invalid_input(
            "Vamana vectors must not contain nulls; a null breaks the fixed stride".to_string(),
        ));
    }

    let first_new = partition.len() as u32;
    let mut graph = partition.graph().clone();
    graph.extend(row_ids)?;
    let vectors = concat_vectors(&[partition.vectors().clone(), vectors.clone()])?;
    let store = flat_storage(graph.row_ids(), &vectors, distance_type)?;

    let mut scratch = InsertScratch::new(graph.len(), graph.max_degree());
    let linking = Linking {
        alpha: params.alpha,
        search_list_size: params.search_list_size,
    };
    let mut order = (first_new..graph.len() as u32).collect::<Vec<_>>();
    order.shuffle(&mut SmallRng::seed_from_u64(params.seed));
    for point in order {
        insert_point(
            &mut graph,
            &store,
            &mut scratch,
            &linking,
            point,
            entry_point,
            comparisons,
        )?;
    }

    let medoid = medoid(&store, comparisons)?;
    Ok(Inserted {
        partition: Partition::try_new(graph, vectors)?,
        medoid,
    })
}

/// One array holding every part's vectors, in the order they are given.
///
/// Through the value buffers rather than `arrow_select::concat` over the lists,
/// which insists the arrays carry identical field metadata down to the item
/// field's name and nullability - and one of them typically comes from a dataset
/// scan, where both are whatever the writer chose.
///
/// No offset arithmetic, which for a list type is usually the trap here:
/// `FixedSizeListArray::slice` slices the child buffer rather than moving a
/// window over it, so a sliced array reports `offset() == 0` and `values()` is
/// already exactly the rows it stands for. Should that ever stop holding, the
/// result comes out with more vectors than the graph has vertices and
/// `Partition::try_new` refuses it by the count.
pub(crate) fn concat_vectors(parts: &[FixedSizeListArray]) -> Result<FixedSizeListArray> {
    let width = parts
        .first()
        .map(FixedSizeListArray::value_length)
        .ok_or_else(|| {
            Error::internal("Vamana was asked to concatenate no vectors at all".to_string())
        })?;
    // Checked rather than assumed: the child buffers concatenate whatever their
    // widths, and a mismatch would come out as the right number of floats cut
    // into the wrong number of vectors.
    if let Some(odd) = parts.iter().find(|part| part.value_length() != width) {
        return Err(Error::invalid_input(format!(
            "Vamana cannot concatenate vectors of dimension {} with vectors of dimension {width}",
            odd.value_length()
        )));
    }
    let values = parts
        .iter()
        .map(|part| part.values().as_ref())
        .collect::<Vec<_>>();
    Ok(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        width,
        concat(&values)?,
        None,
    )?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use arrow_array::{ArrayRef, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use rand::Rng;

    use super::*;
    use lance_linalg::kernels::normalize_fsl;

    use crate::build::{Pruned, build_partition};

    const DIMENSION: usize = 32;
    const MAX_DEGREE: u32 = 16;

    /// Uniform noise, drawn from a seed derived from `offset` so that a batch
    /// and the partition it joins come from the same distribution and share no
    /// point.
    ///
    /// Noise rather than an arithmetic sequence, and that is not fussiness: a
    /// sequence of the `(i * prime) % m` kind is a lattice, and a graph over a
    /// lattice answers every query perfectly at any beam - which makes a recall
    /// comparison over it a comparison of two ones.
    fn scattered(count: usize, offset: usize) -> FixedSizeListArray {
        let mut rng = SmallRng::seed_from_u64(offset as u64 + 1);
        let values = (0..count * DIMENSION)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIMENSION as i32)
            .unwrap()
    }

    fn params() -> BuildParams {
        BuildParams {
            max_degree: MAX_DEGREE,
            search_list_size: 32,
            alpha: 1.2,
            seed: 42,
        }
    }

    /// A partition built over `count` points whose row ids are `1000 + id`, so
    /// that a local id read where a row id belongs is visible.
    fn built(count: usize) -> (Partition, u32) {
        let vectors = scattered(count, 0);
        let row_ids = (0..count as u64).map(|id| 1000 + id).collect::<Vec<_>>();
        let store = flat_storage(&row_ids, &vectors, DistanceType::L2).unwrap();
        let built = build_partition(&store, &params(), &Comparisons::default()).unwrap();
        (
            Partition::try_new(built.graph, vectors).unwrap(),
            built.medoid,
        )
    }

    fn insert(partition: &Partition, entry_point: u32, count: usize, from: usize) -> Inserted {
        let vectors = scattered(count, from);
        let row_ids = (0..count as u64).map(|id| 5000 + id).collect::<Vec<_>>();
        insert_into_partition(
            partition,
            &row_ids,
            &vectors,
            entry_point,
            DistanceType::L2,
            &params(),
            &Comparisons::default(),
        )
        .unwrap()
    }

    /// Recall@`k` of a walk from `entry_point` at beam `beam`, against brute
    /// force over the partition itself.
    fn recall(partition: &Partition, entry_point: u32, beam: usize, k: usize) -> f64 {
        let queries = scattered(64, 20_000);
        let store = flat_storage(
            partition.graph().row_ids(),
            partition.vectors(),
            DistanceType::L2,
        )
        .unwrap();
        let mut scratch = SearchScratch::new(partition.len());
        let mut hits = 0usize;
        for query in 0..queries.len() {
            let vector: ArrayRef = Arc::new(Float32Array::from(
                queries
                    .value(query)
                    .as_primitive::<Float32Type>()
                    .values()
                    .to_vec(),
            ));
            let calculator = store.dist_calculator(vector, 0.0);
            let mut scored = (0..partition.len() as u32)
                .map(|id| (OrderedFloat(calculator.distance(id)), id))
                .collect::<Vec<_>>();
            scored.sort_unstable();
            let truth = scored
                .iter()
                .take(k)
                .map(|(_, id)| *id)
                .collect::<HashSet<_>>();
            let found = greedy_search(
                partition.graph(),
                &calculator,
                entry_point,
                beam,
                &mut scratch,
                &Comparisons::default(),
            )
            .unwrap();
            hits += found
                .candidates
                .iter()
                .take(k)
                .filter(|node| truth.contains(&node.id))
                .count();
        }
        hits as f64 / (queries.len() * k) as f64
    }

    /// A new point is not linked in until something points *at* it. Out-edges
    /// alone would leave every inserted vertex stored, read and unreachable -
    /// and every invariant except this one would still hold.
    #[test]
    fn a_batch_is_linked_into_the_graph_in_both_directions() {
        const OLD: usize = 600;
        const NEW: usize = 200;
        let (partition, medoid) = built(OLD);
        let inserted = insert(&partition, medoid, NEW, OLD);

        let graph = inserted.partition.graph();
        assert_eq!(graph.len(), OLD + NEW);
        for vertex in 0..graph.len() as u32 {
            let neighbors = graph.neighbors(vertex).unwrap();
            assert!(!neighbors.is_empty(), "vertex {vertex} has no out-edges");
            assert!(
                neighbors.len() <= MAX_DEGREE as usize,
                "vertex {vertex} has degree {}",
                neighbors.len()
            );
        }

        let with_back_edges = (0..OLD as u32)
            .filter(|old| {
                graph
                    .neighbors(*old)
                    .unwrap()
                    .iter()
                    .any(|neighbor| *neighbor as usize >= OLD)
            })
            .count();
        assert!(
            with_back_edges > NEW / 2,
            "only {with_back_edges} of the {OLD} original vertices point at a new one"
        );
        assert_eq!(
            graph.reachable_from(inserted.medoid).unwrap(),
            graph.len(),
            "the enlarged graph is in pieces"
        );
    }

    /// Row ids ride along with their vectors. Getting this wrong would answer
    /// every query with somebody else's rows while every graph property held.
    ///
    /// The batch arrives as a **slice** of a longer array, which is the shape
    /// that catches the one arithmetic error here worth catching:
    /// `FixedSizeListArray::values` hands back the whole child buffer and says
    /// nothing about the window its parent is looking through, so a concatenation
    /// that forgot the offset would store the first fifty vectors of the source
    /// instead of the fifty asked for - and every other assertion in this module
    /// would still pass.
    #[test]
    fn the_inserted_rows_keep_their_ids_and_their_vectors() {
        const OLD: usize = 200;
        const NEW: usize = 50;
        const SKIP: usize = 17;
        let (partition, medoid) = built(OLD);
        let source = scattered(NEW + SKIP, OLD);
        let batch = source.slice(SKIP, NEW);
        let row_ids = (0..NEW as u64).map(|id| 5000 + id).collect::<Vec<_>>();

        let inserted = insert_into_partition(
            &partition,
            &row_ids,
            &batch,
            medoid,
            DistanceType::L2,
            &params(),
            &Comparisons::default(),
        )
        .unwrap();

        let graph = inserted.partition.graph();
        assert_eq!(
            graph.row_ids()[..OLD],
            partition.graph().row_ids()[..],
            "the original row ids moved"
        );
        assert_eq!(graph.row_ids()[OLD..], row_ids[..]);
        for new in 0..NEW {
            assert_eq!(
                inserted.partition.vector((OLD + new) as u32).unwrap(),
                source
                    .value(SKIP + new)
                    .as_primitive::<Float32Type>()
                    .values()
                    .to_vec(),
                "vector {new} landed on the wrong vertex"
            );
        }
    }

    /// The entry point follows the middle of the partition as the partition
    /// changes shape.
    ///
    /// Carrying the old one over would be legal - local ids do not move - and
    /// invisible to every other test here, because a batch drawn from the same
    /// cloud barely moves the centre. So the batch is placed off to one side,
    /// where an entry point that did not follow is a different vertex.
    #[test]
    fn the_entry_point_follows_the_middle_of_what_the_partition_now_holds() {
        const OLD: usize = 300;
        const NEW: usize = 300;
        let (partition, entry_point) = built(OLD);
        let shifted = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                scattered(NEW, OLD)
                    .values()
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .map(|value| value + 10.0)
                    .collect::<Vec<_>>(),
            ),
            DIMENSION as i32,
        )
        .unwrap();
        let row_ids = (0..NEW as u64).map(|id| 5000 + id).collect::<Vec<_>>();

        let inserted = insert_into_partition(
            &partition,
            &row_ids,
            &shifted,
            entry_point,
            DistanceType::L2,
            &params(),
            &Comparisons::default(),
        )
        .unwrap();

        let store = flat_storage(
            inserted.partition.graph().row_ids(),
            inserted.partition.vectors(),
            DistanceType::L2,
        )
        .unwrap();
        assert_eq!(
            inserted.medoid,
            medoid(&store, &Comparisons::default()).unwrap(),
            "the entry point is not the medoid of the enlarged partition"
        );
        // `built` returns the medoid of the old cloud, which is what carrying the
        // entry point over unchanged would produce.
        assert_ne!(
            inserted.medoid, entry_point,
            "the entry point did not move even though half the points are new and elsewhere"
        );
    }

    /// The number that says insertion is not a second-class way to build.
    ///
    /// Compared at a beam of exactly `k`, which leaves the walk no slack at all:
    /// at any wider beam over a fixture this size both arms answer perfectly and
    /// the comparison measures nothing. Measured here, built whole 0.6328 and
    /// grown 0.6531 - the grown graph is *ahead*, consistently and at two
    /// fixture sizes, which is worth noting and not worth concluding much from:
    /// the second half is inserted into a graph that is already navigable and at
    /// the full pruning slack, where a build's first pass works at `alpha = 1.0`
    /// over a random graph.
    #[test]
    fn a_grown_graph_answers_nearly_as_well_as_one_built_whole() {
        const TOTAL: usize = 2000;
        const OLD: usize = 1500;
        let (whole, whole_medoid) = built(TOTAL);
        let (partition, medoid) = built(OLD);
        let grown = insert(&partition, medoid, TOTAL - OLD, OLD);
        assert_eq!(grown.partition.len(), whole.len());

        let built_recall = recall(&whole, whole_medoid, 10, 10);
        let grown_recall = recall(&grown.partition, grown.medoid, 10, 10);
        assert!(
            grown_recall >= built_recall - 0.05,
            "a grown graph answers at {grown_recall} where one built whole answers at \
             {built_recall}"
        );
    }

    #[test]
    fn insertion_refuses_what_it_cannot_do() {
        let (partition, medoid) = built(100);
        let vectors = scattered(10, 100);
        let row_ids = (0..10u64).collect::<Vec<_>>();
        let attempt = |row_ids: &[u64],
                       vectors: &FixedSizeListArray,
                       entry_point: u32,
                       params: &BuildParams| {
            insert_into_partition(
                &partition,
                row_ids,
                vectors,
                entry_point,
                DistanceType::L2,
                params,
                &Comparisons::default(),
            )
            .unwrap_err()
            .to_string()
        };

        assert!(
            attempt(&row_ids[..9], &vectors, medoid, &params()).contains("9 row ids and 10"),
            "a row id per vector is what makes an answer point at the right row"
        );
        assert!(attempt(&[], &scattered(0, 0), medoid, &params()).contains("insert nothing"));
        assert!(
            attempt(&row_ids, &vectors, 100, &params()).contains("entry point 100"),
            "the entry point is a local id of the partition being inserted into"
        );
        let narrow =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 40]), 4)
                .unwrap();
        assert!(
            attempt(&row_ids, &narrow, medoid, &params())
                .contains("dimension 4 into a partition of dimension 32")
        );
        assert!(
            attempt(
                &row_ids,
                &vectors,
                medoid,
                &BuildParams {
                    max_degree: MAX_DEGREE + 1,
                    ..params()
                }
            )
            .contains("degree 17 into a partition built at degree 16")
        );
    }

    /// What a prune makes of `vertex`'s list as it stands. A list a prune returned
    /// comes back unchanged, and with the coincident fill exactly when it had it.
    fn reprune_of(
        store: &impl VectorStore,
        graph: &PartitionGraph,
        vertex: u32,
        alpha: f32,
    ) -> Pruned {
        let from_vertex = store.dist_calculator_from_id(vertex);
        let candidates = graph
            .neighbors(vertex)
            .unwrap()
            .iter()
            .map(|id| OrderedNode::new(*id, OrderedFloat(from_vertex.distance(*id))))
            .collect();
        prune(
            store,
            vertex,
            candidates,
            alpha,
            graph.max_degree() as usize,
            &Comparisons::default(),
        )
        .unwrap()
    }

    /// How often each change of the flag was met, so the test can insist it met all.
    #[derive(Debug, Default)]
    struct FlagChanges {
        appended_to_a_prune_output: usize,
        full_prune_output_kept: usize,
        full_prune_output_changed: usize,
        other_list_repruned: usize,
        other_list_filled: usize,
        own_list_filled: usize,
        flagged_own_list_filled: usize,
        prune_output_refilled: usize,
    }

    /// The flag `admit` relies on says what each list is after every insertion of
    /// a two-pass sweep over one scratch: set by a prune unless its coincident fill
    /// was used, cleared by an unchecked append, kept by a newcomer admitted or
    /// refused - and every list it is set on is one a prune returns unchanged.
    fn prune_output_flag_over(distance_type: DistanceType) -> FlagChanges {
        const DEGREE: u32 = 4;
        // Groups of `DEGREE + 2` copies, enough for the fill to fill a list. Under
        // `Cosine` a copy is a multiple of its point, normalised like the rest, and
        // no two multiples differ by a power of two, which would normalise to the
        // very same vector: the copies have to land ULPs apart.
        const SCALES: [f32; DEGREE as usize + 2] = [1.0, 1.3, 1.7, 2.3, 2.9, 3.1];
        let copies = scattered(24, 0)
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .chunks(DIMENSION)
            .flat_map(|point| {
                SCALES.iter().flat_map(move |scale| {
                    let scale = match distance_type {
                        DistanceType::Cosine => *scale,
                        _ => 1.0,
                    };
                    point.iter().map(move |value| value * scale)
                })
            })
            .collect::<Vec<_>>();
        let copies =
            FixedSizeListArray::try_new_from_values(Float32Array::from(copies), DIMENSION as i32)
                .unwrap();
        let vectors = concat_vectors(&[copies, scattered(80, 100)]).unwrap();
        let vectors = match distance_type {
            DistanceType::Cosine => normalize_fsl(&vectors).unwrap(),
            _ => vectors,
        };
        let row_ids = (0..vectors.len() as u64).collect::<Vec<_>>();
        let store = flat_storage(&row_ids, &vectors, distance_type).unwrap();
        let params = BuildParams {
            max_degree: DEGREE,
            search_list_size: 16,
            ..params()
        };
        let built = build_partition(&store, &params, &Comparisons::default()).unwrap();
        let mut graph = built.graph;
        let mut scratch = InsertScratch::new(graph.len(), DEGREE);
        let mut order = (0..graph.len() as u32).collect::<Vec<_>>();
        let mut rng = SmallRng::seed_from_u64(5);
        let mut changes = FlagChanges::default();

        for alpha in [1.0, params.alpha] {
            // The lists a pass hands on must hold at the alpha the next one uses.
            for vertex in 0..graph.len() as u32 {
                if scratch.is_prune_output[vertex as usize] {
                    let repruned = reprune_of(&store, &graph, vertex, alpha);
                    assert_eq!(repruned.neighbors, graph.neighbors(vertex).unwrap());
                    assert!(!repruned.has_coincident_fill, "vertex {vertex}");
                }
            }
            let linking = Linking {
                alpha,
                search_list_size: params.search_list_size,
            };
            order.shuffle(&mut rng);
            for &point in &order {
                let before = graph.clone();
                let flags = scratch.is_prune_output.clone();
                insert_point(
                    &mut graph,
                    &store,
                    &mut scratch,
                    &linking,
                    point,
                    built.medoid,
                    &Comparisons::default(),
                )
                .unwrap();

                let own = reprune_of(&store, &graph, point, alpha);
                assert_eq!(
                    own.neighbors,
                    graph.neighbors(point).unwrap(),
                    "vertex {point}"
                );
                assert_eq!(
                    scratch.is_prune_output[point as usize], !own.has_coincident_fill,
                    "vertex {point}'s own list"
                );
                if own.has_coincident_fill {
                    changes.own_list_filled += 1;
                    if flags[point as usize] {
                        changes.flagged_own_list_filled += 1;
                    }
                }
                let chosen = graph.neighbors(point).unwrap();
                for vertex in (0..graph.len() as u32).filter(|vertex| *vertex != point) {
                    let (was, is) = (
                        before.neighbors(vertex).unwrap(),
                        graph.neighbors(vertex).unwrap(),
                    );
                    let (flagged, flag) = (
                        flags[vertex as usize],
                        scratch.is_prune_output[vertex as usize],
                    );
                    let what = format!("vertex {vertex} after inserting {point} at {alpha}");
                    if !chosen.contains(&vertex) || was.contains(&point) {
                        assert_eq!(is, was, "{what}");
                        assert_eq!(flag, flagged, "{what}");
                    } else if was.len() < DEGREE as usize {
                        assert_eq!(is, [was, &[point]].concat(), "{what}");
                        assert!(!flag, "{what}: appended to, and still flagged");
                        if flagged {
                            changes.appended_to_a_prune_output += 1;
                        }
                    } else {
                        let repruned = reprune_of(&store, &graph, vertex, alpha);
                        assert_eq!(repruned.neighbors, is, "{what}");
                        assert_eq!(flag, !repruned.has_coincident_fill, "{what}");
                        match (flagged, repruned.has_coincident_fill) {
                            (true, true) => changes.prune_output_refilled += 1,
                            (true, _) if is == was => changes.full_prune_output_kept += 1,
                            (true, _) => changes.full_prune_output_changed += 1,
                            (false, false) => changes.other_list_repruned += 1,
                            (false, true) => changes.other_list_filled += 1,
                        }
                    }
                }
            }
        }
        changes
    }

    #[test]
    fn the_prune_output_flag_follows_every_list() {
        let changes = prune_output_flag_over(DistanceType::L2);
        assert!(
            changes.appended_to_a_prune_output > 0
                && changes.full_prune_output_kept > 0
                && changes.full_prune_output_changed > 0
                && changes.other_list_repruned > 0
                && changes.other_list_filled > 0
                && changes.own_list_filled > 0
                && changes.flagged_own_list_filled > 0,
            "{changes:?}"
        );
    }

    /// Under cosine a newcomer can sit at zero separation from two members that are
    /// not at zero from each other, so a list with the flag set can come back from
    /// the prune filled - and the flag has to clear.
    #[test]
    fn the_prune_output_flag_follows_every_list_under_cosine() {
        let changes = prune_output_flag_over(DistanceType::Cosine);
        assert!(changes.prune_output_refilled > 0, "{changes:?}");
    }
}
