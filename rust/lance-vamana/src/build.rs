// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building a partition's graph.

use std::collections::VecDeque;

use lance_core::{Error, Result};
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::partition::PartitionGraph;
use crate::search::{Comparisons, SearchScratch, greedy_search};

/// How a partition's graph is built.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildParams {
    /// `R`: the fixed width of every vertex's neighbour list.
    ///
    /// The parameter that decides whether high recall is reachable at all. On
    /// SIFT1M against Lance's HNSW at matched memory, `32` loses at every recall
    /// above 0.97 and `64` wins at every recall, by a margin that grows with it.
    pub max_degree: u32,
    /// `L`: the beam each build-time search keeps. Sets both build cost and
    /// graph quality, and is unrelated to the beam a query later uses.
    ///
    /// Raising it is not free improvement. At `max_degree = 32` on SIFT1M,
    /// doubling this to 200 made query cost at recall 0.99 *worse* by a quarter:
    /// a wider search hands the prune more candidates, and it spends the few
    /// slots there are on long, diverse edges rather than the local ones the
    /// last few points of recall need. At `max_degree = 64` doubling it changed
    /// nothing measurable while costing 2.5x the build.
    pub search_list_size: usize,
    /// `alpha` for the second pass. The first pass is always `1.0`.
    pub alpha: f32,
    /// How many vertices the entry point is chosen from. The true medoid costs
    /// `O(n^2)` distances; a uniform sample of this size costs `O(sample^2)`.
    pub medoid_sample_size: usize,
    /// Fixed rather than optional so that a build is reproducible by default.
    ///
    /// Lance's own vector index builds are random at half a dozen unseeded
    /// sites, which makes an A/B of two builds measure the dice as much as the
    /// change. Varying the seed here is a deliberate act.
    pub seed: u64,
}

impl Default for BuildParams {
    /// The best of the four `(max_degree, search_list_size)` pairs measured on
    /// SIFT1M: 18-26% cheaper per query than Lance's HNSW at matched memory,
    /// and cheaper to build than it too.
    fn default() -> Self {
        Self {
            max_degree: 64,
            search_list_size: 100,
            alpha: 1.2,
            medoid_sample_size: 256,
            seed: 42,
        }
    }
}

/// A partition's graph and the vertex a search of it should start from.
#[derive(Debug)]
pub struct BuiltPartition {
    pub graph: PartitionGraph,
    pub medoid: u32,
}

/// Build one partition's graph, in memory.
///
/// Algorithm 3 of the DiskANN paper: start from a random `R`-regular graph, then
/// sweep every vertex in random order, and for each one search the graph as it
/// currently stands, prune the visited set into its new neighbour list, and add
/// the back-edges. The sweep runs twice, first at `alpha = 1.0` and then at
/// `params.alpha`, because the second pass works on a graph that is already
/// navigable and can afford to keep the shorter edges the first pass dropped.
pub fn build_partition<S: VectorStore>(
    store: &S,
    params: &BuildParams,
    comparisons: &Comparisons,
) -> Result<BuiltPartition> {
    let num_vertices = store.len();
    if num_vertices == 0 {
        return Err(Error::invalid_input(
            "Vamana cannot build a graph over an empty partition".to_string(),
        ));
    }
    if params.search_list_size == 0 {
        return Err(Error::invalid_input(
            "Vamana search list size must be greater than zero".to_string(),
        ));
    }
    if params.medoid_sample_size == 0 {
        return Err(Error::invalid_input(
            "Vamana medoid sample size must be greater than zero".to_string(),
        ));
    }

    let mut rng = SmallRng::seed_from_u64(params.seed);
    // Indexed rather than iterated: a vertex's local id is its position here, so
    // this mapping is what turns a graph result back into a dataset row, and
    // `VectorStore` nowhere promises that `row_ids()` yields them in id order.
    let row_ids = (0..num_vertices as u32)
        .map(|id| store.row_id(id))
        .collect::<Vec<_>>();
    let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids)?;
    randomize(&mut graph, &mut rng)?;
    let medoid = medoid(store, params.medoid_sample_size, &mut rng, comparisons)?;

    let max_degree = params.max_degree as usize;
    let mut scratch = SearchScratch::new(num_vertices);
    let mut order = (0..num_vertices as u32).collect::<Vec<_>>();
    let mut existing = Vec::with_capacity(max_degree + 1);

    for alpha in [1.0, params.alpha] {
        order.shuffle(&mut rng);
        for point in &order {
            let point = *point;
            let from_point = store.dist_calculator_from_id(point);
            let mut candidates = greedy_search(
                &graph,
                &from_point,
                medoid,
                params.search_list_size,
                &mut scratch,
                comparisons,
            )?
            .visited;
            // The paper folds the current out-edges into the candidate set
            // inside the prune; doing it here keeps the prune ignorant of the
            // graph, which is what lets the back-edge case below reuse it.
            comparisons.record(graph.neighbors(point).len() as u64);
            candidates.extend(graph.neighbors(point).iter().map(|neighbor| {
                OrderedNode::new(*neighbor, OrderedFloat(from_point.distance(*neighbor)))
            }));

            let selected = robust_prune(store, point, candidates, alpha, max_degree, comparisons)?;
            graph.set_neighbors(point, &selected)?;

            for neighbor in &selected {
                let neighbor = *neighbor;
                existing.clear();
                existing.extend_from_slice(graph.neighbors(neighbor));
                if existing.contains(&point) {
                    continue;
                }
                if existing.len() < max_degree {
                    existing.push(point);
                    graph.set_neighbors(neighbor, &existing)?;
                    continue;
                }
                // Full: the back-edge has to earn its place against the rest.
                let from_neighbor = store.dist_calculator_from_id(neighbor);
                comparisons.record(existing.len() as u64 + 1);
                let contenders = existing
                    .iter()
                    .chain(std::iter::once(&point))
                    .map(|id| OrderedNode::new(*id, OrderedFloat(from_neighbor.distance(*id))))
                    .collect();
                let pruned =
                    robust_prune(store, neighbor, contenders, alpha, max_degree, comparisons)?;
                graph.set_neighbors(neighbor, &pruned)?;
            }
        }
    }

    Ok(BuiltPartition { graph, medoid })
}

/// Give every vertex `min(R, n - 1)` random out-edges.
///
/// The first pass has to be able to walk somewhere, and any connected starting
/// graph would do; random is the paper's choice because it has short diameter
/// and no structure for the pruning to inherit.
fn randomize(graph: &mut PartitionGraph, rng: &mut SmallRng) -> Result<()> {
    let num_vertices = graph.len();
    let degree = (graph.max_degree() as usize).min(num_vertices.saturating_sub(1));
    if degree == 0 {
        return Ok(());
    }

    let mut neighbors = Vec::with_capacity(degree);
    for point in 0..num_vertices as u32 {
        neighbors.clear();
        // Sampled from the other `n - 1` vertices, then shifted past `point`,
        // so a vertex can never draw itself and never draws a duplicate.
        neighbors.extend(
            rand::seq::index::sample(rng, num_vertices - 1, degree)
                .into_iter()
                .map(|drawn| {
                    let drawn = drawn as u32;
                    if drawn >= point { drawn + 1 } else { drawn }
                }),
        );
        graph.set_neighbors(point, &neighbors)?;
    }
    Ok(())
}

/// The vertex a search should start from: the sampled point most central to the
/// partition.
///
/// The true medoid needs every pairwise distance, which no partition can afford
/// at build time. A uniform sample is scored against itself instead, which is
/// enough for an entry point: the walk only has to start somewhere unbiased.
pub fn medoid<S: VectorStore>(
    store: &S,
    sample_size: usize,
    rng: &mut SmallRng,
    comparisons: &Comparisons,
) -> Result<u32> {
    let num_vertices = store.len();
    if num_vertices == 0 {
        return Err(Error::invalid_input(
            "Vamana cannot pick a medoid from an empty partition".to_string(),
        ));
    }
    let sample = if sample_size >= num_vertices {
        (0..num_vertices as u32).collect::<Vec<_>>()
    } else {
        let mut sample = (0..num_vertices as u32).collect::<Vec<_>>();
        sample.shuffle(rng);
        sample.truncate(sample_size);
        sample.sort_unstable();
        sample
    };

    let mut best = (f32::INFINITY, sample[0]);
    for candidate in &sample {
        let from_candidate = store.dist_calculator_from_id(*candidate);
        comparisons.record(sample.len() as u64);
        let total = sample
            .iter()
            .map(|other| from_candidate.distance(*other))
            .sum::<f32>();
        if total < best.0 {
            best = (total, *candidate);
        }
    }
    Ok(best.1)
}

/// Choose up to `max_degree` out-edges for `point` from `candidates`.
///
/// Algorithm 2 of the DiskANN paper. Candidates are taken nearest first, and
/// after each pick the rest are swept: a candidate that sits `alpha` times
/// closer to the vertex just picked than to `point` is dropped, because the
/// picked vertex already routes there. That sweep is what makes the result a
/// spread of directions rather than the `max_degree` nearest points, and it is
/// what lets a walk make progress instead of circling.
///
/// `alpha` is the slack in that test. At `1.0` it reduces exactly to the
/// diversity heuristic Lance applies to HNSW; above `1.0` fewer candidates are
/// dropped, so the graph keeps more of its short edges.
///
/// `candidates` carry their distance to `point`, and may contain `point` itself
/// and duplicates. The paper's version starts by folding `point`'s current
/// out-edges into that set; here the caller does it, which is what lets the same
/// function serve both a vertex's own prune and a back-edge that has to fight
/// for a slot.
pub fn robust_prune<S: VectorStore>(
    store: &S,
    point: u32,
    candidates: Vec<OrderedNode>,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Vec<u32>> {
    if alpha.is_nan() || alpha < 1.0 {
        return Err(Error::invalid_input(format!(
            "Vamana alpha must be at least 1.0, got {alpha}"
        )));
    }
    if max_degree == 0 {
        return Err(Error::invalid_input(
            "Vamana max_degree must be greater than zero".to_string(),
        ));
    }

    let mut pool = candidates;
    pool.retain(|candidate| candidate.id != point);
    pool.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
    pool.dedup_by_key(|candidate| candidate.id);
    let mut pool = VecDeque::from(pool);

    let mut selected = Vec::with_capacity(max_degree);
    while let Some(nearest) = pool.pop_front() {
        selected.push(nearest.id);
        if selected.len() == max_degree {
            break;
        }
        let from_nearest = store.dist_calculator_from_id(nearest.id);
        comparisons.record(pool.len() as u64);
        pool.retain(|candidate| alpha * from_nearest.distance(candidate.id) > candidate.dist.0);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::search::{SearchScratch, greedy_search};

    /// Deterministic pseudo-random vectors: a fixed multiplicative congruential
    /// sequence, so the cross-check against Lance runs on the same points every
    /// time without pulling in an RNG.
    fn scattered_storage(num_vertices: usize, dimension: usize) -> FlatFloatStorage {
        let mut state = 12345u64;
        let values = Float32Array::from(
            (0..num_vertices * dimension)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect::<Vec<_>>(),
        );
        FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap(),
            DistanceType::L2,
        )
    }

    fn all_candidates(
        storage: &FlatFloatStorage,
        point: u32,
        num_vertices: usize,
    ) -> Vec<OrderedNode> {
        let calculator = storage.dist_calculator_from_id(point);
        (0..num_vertices as u32)
            .map(|id| OrderedNode::new(id, OrderedFloat(calculator.distance(id))))
            .collect()
    }

    /// What Lance's own diversity heuristic would select, walking candidates
    /// nearest first. `prefers_candidate` is the same predicate at `alpha = 1`,
    /// so the two must agree vertex for vertex.
    fn lance_selection(
        storage: &FlatFloatStorage,
        point: u32,
        candidates: &[OrderedNode],
        max_degree: usize,
    ) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.retain(|candidate| candidate.id != point);
        sorted.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));

        let mut selected: Vec<OrderedNode> = Vec::with_capacity(max_degree);
        for candidate in sorted {
            if selected.len() == max_degree {
                break;
            }
            if storage.prefers_candidate(&candidate, &selected) {
                selected.push(candidate);
            }
        }
        selected.into_iter().map(|node| node.id).collect()
    }

    #[test]
    fn alpha_one_reproduces_lance_diversity_exactly() {
        const VERTICES: usize = 200;
        let storage = scattered_storage(VERTICES, 8);

        for point in [0u32, 7, 63, 199] {
            let candidates = all_candidates(&storage, point, VERTICES);
            let ours = robust_prune(
                &storage,
                point,
                candidates.clone(),
                1.0,
                16,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(
                ours,
                lance_selection(&storage, point, &candidates, 16),
                "vertex {point}"
            );
        }
    }

    /// With enough slack nothing is ever dropped, so the result must be plain
    /// nearest-neighbours. This pins which way the alpha test points: an
    /// inverted comparison would prune everything here instead of nothing.
    #[test]
    fn unbounded_alpha_selects_the_nearest_candidates() {
        const VERTICES: usize = 64;
        let storage = scattered_storage(VERTICES, 4);
        let candidates = all_candidates(&storage, 0, VERTICES);

        let mut nearest = candidates.clone();
        nearest.retain(|candidate| candidate.id != 0);
        nearest.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        let nearest = nearest
            .iter()
            .take(8)
            .map(|node| node.id)
            .collect::<Vec<_>>();

        let selected = robust_prune(
            &storage,
            0,
            candidates,
            f32::MAX,
            8,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(selected, nearest);
    }

    /// The other end of the same axis: at `alpha = 1` on scattered data the
    /// sweep must actually throw candidates away, or the test above is
    /// measuring nothing.
    #[test]
    fn alpha_one_prunes_more_than_unbounded_alpha() {
        const VERTICES: usize = 200;
        let storage = scattered_storage(VERTICES, 8);
        let candidates = all_candidates(&storage, 0, VERTICES);

        let diverse = robust_prune(
            &storage,
            0,
            candidates.clone(),
            1.0,
            VERTICES,
            &Comparisons::default(),
        )
        .unwrap();
        assert!(
            diverse.len() < VERTICES - 1,
            "nothing was pruned: {} of {} candidates survived",
            diverse.len(),
            VERTICES - 1
        );
    }

    #[test]
    fn the_nearest_candidate_is_always_kept() {
        const VERTICES: usize = 64;
        let storage = scattered_storage(VERTICES, 4);
        let candidates = all_candidates(&storage, 3, VERTICES);
        let nearest = candidates
            .iter()
            .filter(|candidate| candidate.id != 3)
            .min_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)))
            .unwrap()
            .id;

        for alpha in [1.0, 1.2, 2.0] {
            let selected = robust_prune(
                &storage,
                3,
                candidates.clone(),
                alpha,
                8,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(selected[0], nearest, "at alpha {alpha}");
        }
    }

    #[test]
    fn the_point_and_its_duplicates_never_become_edges() {
        const VERTICES: usize = 32;
        let storage = scattered_storage(VERTICES, 4);
        let mut candidates = all_candidates(&storage, 5, VERTICES);
        candidates.extend(all_candidates(&storage, 5, VERTICES));

        let selected =
            robust_prune(&storage, 5, candidates, 1.2, 16, &Comparisons::default()).unwrap();
        assert!(!selected.contains(&5), "the vertex selected itself");
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            selected.len(),
            "duplicate out-edge: {selected:?}"
        );
        assert!(selected.len() <= 16);
    }

    #[test]
    fn an_alpha_below_one_is_rejected() {
        let storage = scattered_storage(8, 2);
        let error = robust_prune(
            &storage,
            0,
            all_candidates(&storage, 0, 8),
            0.9,
            4,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least 1.0"), "{error}");
    }

    /// Vertices on a line at 0, 1, 2, ... so every distance is hand-checkable.
    fn line_storage(num_vertices: usize) -> FlatFloatStorage {
        let values = Float32Array::from((0..num_vertices).map(|i| i as f32).collect::<Vec<_>>());
        FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(values, 1).unwrap(),
            DistanceType::L2,
        )
    }

    fn small_params() -> BuildParams {
        BuildParams {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
            medoid_sample_size: 64,
            seed: 42,
        }
    }

    /// Every vertex reachable from `entry_point` by following out-edges.
    fn reachable(graph: &PartitionGraph, entry_point: u32) -> usize {
        let mut seen = vec![false; graph.len()];
        let mut frontier = vec![entry_point];
        seen[entry_point as usize] = true;
        let mut count = 1;
        while let Some(vertex) = frontier.pop() {
            for neighbor in graph.neighbors(vertex) {
                if !seen[*neighbor as usize] {
                    seen[*neighbor as usize] = true;
                    count += 1;
                    frontier.push(*neighbor);
                }
            }
        }
        count
    }

    #[test]
    fn a_built_graph_holds_its_shape() {
        const VERTICES: usize = 500;
        let storage = scattered_storage(VERTICES, 8);
        let params = small_params();
        let built = build_partition(&storage, &params, &Comparisons::default()).unwrap();

        for vertex in 0..VERTICES as u32 {
            let neighbors = built.graph.neighbors(vertex);
            assert!(
                !neighbors.is_empty(),
                "vertex {vertex} was pruned into a dead end"
            );
            assert!(neighbors.len() <= params.max_degree as usize);
            assert!(
                !neighbors.contains(&vertex),
                "vertex {vertex} points at itself"
            );

            let mut sorted = neighbors.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                neighbors.len(),
                "vertex {vertex}: {neighbors:?}"
            );
            assert!(sorted.last().unwrap() < &(VERTICES as u32), "dangling edge");
        }
        assert!((built.medoid as usize) < VERTICES);
    }

    /// A graph the entry point cannot reach all of is a graph whose unreachable
    /// half can never be returned, whatever the recall of the rest looks like.
    #[test]
    fn every_vertex_is_reachable_from_the_medoid() {
        const VERTICES: usize = 500;
        let storage = scattered_storage(VERTICES, 8);
        let built = build_partition(&storage, &small_params(), &Comparisons::default()).unwrap();
        assert_eq!(reachable(&built.graph, built.medoid), VERTICES);
    }

    /// Recall against brute force, on queries that are not in the index.
    ///
    /// A smoke test, and deliberately labelled as one. Measured against
    /// mutations of the build, it catches a graph with no back-edges (recall
    /// 0.718) but not a build stopped after one pass (0.998) and not a prune
    /// with the diversity test removed (0.998). Two thousand points in eight
    /// dimensions are simply too easy to tell a good graph from a mediocre one;
    /// that is what S2c on SIFT1M is for. What this does hold is the floor:
    /// the graph answers, and it answers without reading everything.
    #[test]
    fn recall_at_ten_beats_the_bar() {
        const VERTICES: usize = 2000;
        const QUERIES: usize = 50;
        const K: usize = 10;
        let storage = scattered_storage(VERTICES + QUERIES, 8);
        let indexed = scattered_storage(VERTICES, 8);
        let params = small_params();
        let comparisons = Comparisons::default();
        let built = build_partition(&indexed, &params, &comparisons).unwrap();

        let mut scratch = SearchScratch::new(VERTICES);
        let searching = Comparisons::default();
        let mut hits = 0;
        for query in VERTICES..VERTICES + QUERIES {
            // The extra points of `storage` are outside the index, so they are
            // honest out-of-sample queries against the same distribution.
            let vector = storage.vector(query as u32);
            let calculator = indexed.dist_calculator(vector, 0.0);

            let mut exact = (0..VERTICES as u32)
                .map(|id| OrderedNode::new(id, OrderedFloat(calculator.distance(id))))
                .collect::<Vec<_>>();
            exact.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
            let exact = exact
                .iter()
                .take(K)
                .map(|node| node.id)
                .collect::<std::collections::HashSet<_>>();

            let found = greedy_search(
                &built.graph,
                &calculator,
                built.medoid,
                params.search_list_size,
                &mut scratch,
                &searching,
            )
            .unwrap();
            hits += found
                .candidates
                .iter()
                .take(K)
                .filter(|node| exact.contains(&node.id))
                .count();
        }

        let recall = hits as f64 / (QUERIES * K) as f64;
        let scanned = searching.get() as f64 / QUERIES as f64;
        println!(
            "recall@{K} = {recall:.4}, {scanned:.1} distances per query = {:.1}% of {VERTICES} \
             vectors (build cost {} distances)",
            100.0 * scanned / VERTICES as f64,
            comparisons.get()
        );
        assert!(recall >= 0.95, "recall@{K} fell to {recall:.4}");
        // Recall means nothing without this: a walk that reaches most of the
        // partition would score perfectly while being brute force in disguise.
        assert!(
            scanned < 0.25 * VERTICES as f64,
            "the search touched {scanned:.1} of {VERTICES} vectors, so its recall is not the \
             graph's doing"
        );
    }

    /// A build that cannot be repeated cannot be A/B tested: two runs would
    /// differ by the dice as much as by whatever was changed between them.
    #[test]
    fn the_same_seed_builds_the_same_graph() {
        const VERTICES: usize = 300;
        let storage = scattered_storage(VERTICES, 6);
        let params = small_params();

        let first = build_partition(&storage, &params, &Comparisons::default()).unwrap();
        let second = build_partition(&storage, &params, &Comparisons::default()).unwrap();
        assert_eq!(first.graph, second.graph);
        assert_eq!(first.medoid, second.medoid);

        let other_seed = BuildParams {
            seed: params.seed + 1,
            ..params
        };
        let third = build_partition(&storage, &other_seed, &Comparisons::default()).unwrap();
        assert_ne!(
            third.graph, first.graph,
            "a different seed produced an identical graph, so the seed is not reaching the build"
        );
    }

    /// The second pass is the only place `alpha` acts, so a build that ignored
    /// it would come back identical.
    #[test]
    fn alpha_changes_the_graph_that_is_built() {
        const VERTICES: usize = 300;
        let storage = scattered_storage(VERTICES, 6);
        let params = small_params();
        let plain = BuildParams {
            alpha: 1.0,
            ..params.clone()
        };

        let with_slack = build_partition(&storage, &params, &Comparisons::default()).unwrap();
        let without = build_partition(&storage, &plain, &Comparisons::default()).unwrap();
        assert_ne!(with_slack.graph, without.graph);
        assert_eq!(
            with_slack.medoid, without.medoid,
            "alpha must not move the entry point"
        );
    }

    #[test]
    fn a_partition_smaller_than_the_degree_bound_builds() {
        for vertices in [1usize, 2, 3, 17] {
            let storage = scattered_storage(vertices, 4);
            let built =
                build_partition(&storage, &small_params(), &Comparisons::default()).unwrap();
            assert_eq!(built.graph.len(), vertices);
            assert_eq!(reachable(&built.graph, built.medoid), vertices);
            for vertex in 0..vertices as u32 {
                assert!(built.graph.neighbors(vertex).len() < vertices);
            }
        }
    }

    /// On a line the answer is arithmetic: minimising the sum of squared
    /// distances puts the entry point at the middle vertex, exactly.
    ///
    /// Worth pinning directly, because the build tests cannot see it. Replacing
    /// the medoid with vertex 0 changes no recall at unit scale - a graph this
    /// well connected is navigable from anywhere. The entry point earns its keep
    /// later, in consolidation, where it has to be recomputed because the old
    /// one may have been deleted.
    #[test]
    fn the_medoid_of_a_line_is_its_middle() {
        for vertices in [3usize, 11, 64] {
            let storage = line_storage(vertices);
            let chosen = medoid(
                &storage,
                vertices,
                &mut SmallRng::seed_from_u64(7),
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(
                chosen as usize,
                (vertices - 1) / 2,
                "medoid of a {vertices}-point line"
            );
        }
    }

    /// The entry point must follow the mass, not the extent: nine tenths of the
    /// points sit at one end, so that end is the middle.
    #[test]
    fn the_medoid_follows_the_dense_cluster() {
        let mut values = vec![0.0f32; 90];
        values.extend(std::iter::repeat_n(100.0f32, 10));
        let storage = FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), 1).unwrap(),
            DistanceType::L2,
        );

        let chosen = medoid(
            &storage,
            100,
            &mut SmallRng::seed_from_u64(7),
            &Comparisons::default(),
        )
        .unwrap();
        assert!(chosen < 90, "the entry point landed in the sparse cluster");
    }

    #[test]
    fn an_empty_partition_cannot_be_built() {
        let storage = scattered_storage(0, 4);
        let error =
            build_partition(&storage, &small_params(), &Comparisons::default()).unwrap_err();
        assert!(error.to_string().contains("empty partition"), "{error}");
    }

    /// A NaN alpha would make every prune test false and silently keep the
    /// nearest `max_degree` candidates, so it is rejected rather than compared.
    #[test]
    fn a_nan_alpha_is_rejected() {
        let storage = scattered_storage(8, 2);
        let error = robust_prune(
            &storage,
            0,
            all_candidates(&storage, 0, 8),
            f32::NAN,
            4,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least 1.0"), "{error}");
    }
}
