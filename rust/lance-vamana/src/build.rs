// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building a partition's graph.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, Float32Array, RecordBatch};
use lance_core::{Error, Result};
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::format::{IndexMetadata, MAX_PARTITION_ROWS};
use crate::insert::{InsertScratch, Linking, insert_point};
use crate::partition::PartitionGraph;
use crate::search::Comparisons;

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
            seed: 42,
        }
    }
}

/// The insertion order maintenance builds in.
///
/// Fixed rather than carried in the segment, unlike the degree, the beam and the
/// pruning slack, which are. Those three decide what the graph *is*, and a
/// partition a maintenance pass rebuilds or extends has to match its siblings on
/// them. The seed decides only which of the equally good graphs comes out, and
/// the crate's own position on it is that varying a seed is a deliberate act -
/// so maintenance takes the one it is given and stays reproducible.
pub const MAINTENANCE_SEED: u64 = 42;

impl BuildParams {
    /// The parameters a maintenance pass must work at, read off the segment it
    /// is rewriting.
    ///
    /// Every graph a rewrite produces sits beside siblings that were built with
    /// these three numbers, and a partition that disagrees with them disagrees
    /// with its own segment's table. The seed is the one thing not in the
    /// metadata, because the metadata deliberately does not record it: see
    /// [`MAINTENANCE_SEED`].
    pub fn maintenance(metadata: &IndexMetadata) -> Self {
        Self {
            max_degree: metadata.max_degree,
            search_list_size: metadata.search_list_size,
            alpha: metadata.alpha,
            seed: MAINTENANCE_SEED,
        }
    }
}

/// A partition's graph and the vertex a search of it should start from.
#[derive(Debug)]
pub struct BuiltPartition {
    pub graph: PartitionGraph,
    pub medoid: u32,
}

/// How many vertices a store holds, as the id space they will be addressed in.
///
/// A local id is a `u32` while `VectorStore::len` is a `usize`, and casting the
/// one to the other is the single place where a partition too large to address
/// turns into a partition of `len % 2^32` vertices instead of an error.
fn addressable_len(num_vertices: usize) -> Result<u32> {
    u32::try_from(num_vertices)
        .ok()
        .filter(|len| *len <= MAX_PARTITION_ROWS)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Vamana cannot build over {num_vertices} vectors, exceeding the addressable \
                 maximum {MAX_PARTITION_ROWS}"
            ))
        })
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
    let num_vertices = addressable_len(store.len())?;
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
    // Checked here and not left to the second pass: `robust_prune` would reject
    // it, but only after the whole first pass had already run.
    validate_alpha(params.alpha)?;

    let mut rng = SmallRng::seed_from_u64(params.seed);
    // Indexed rather than iterated: a vertex's local id is its position here, so
    // this mapping is what turns a graph result back into a dataset row, and
    // `VectorStore` nowhere promises that `row_ids()` yields them in id order.
    let row_ids = (0..num_vertices)
        .map(|id| store.row_id(id))
        .collect::<Vec<_>>();
    let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids)?;
    randomize(&mut graph, &mut rng)?;
    let medoid = medoid(store, comparisons)?;

    let mut scratch = InsertScratch::new(num_vertices as usize, params.max_degree);
    let mut order = (0..num_vertices).collect::<Vec<_>>();

    for alpha in [1.0, params.alpha] {
        let linking = Linking {
            alpha,
            search_list_size: params.search_list_size,
        };
        order.shuffle(&mut rng);
        for point in &order {
            insert_point(
                &mut graph,
                store,
                &mut scratch,
                &linking,
                *point,
                medoid,
                comparisons,
            )?;
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

/// The vertex a search should start from: the one nearest the partition's
/// centroid, which under this crate's metrics is its medoid exactly.
///
/// The medoid is the point minimising the summed distance to every other point,
/// and taken literally that is every pairwise distance. It is never computed
/// that way, because Lance's `L2` is the *squared* euclidean distance and a sum
/// of squares splits:
///
/// ```text
/// sum_j ||x_i - x_j||^2  =  n * ||x_i - c||^2  +  sum_j ||x_j - c||^2
/// ```
///
/// The right-hand term is the same for every `i`, so the vertex minimising the
/// left-hand side is the vertex nearest the centroid `c`, and the answer costs
/// `O(n*d)` rather than `O(n^2*d)`. A far outlier moves both sides of that
/// identity together: it drags the centroid, and it drags the summed distance
/// with it.
///
/// The same holds under `Cosine` for the vectors this crate builds over, because
/// [`crate::builder`] normalises them first and `1 - dot(x, y)` is
/// `||x - y||^2 / 2` on unit vectors. It does not hold for un-normalised vectors
/// under `Cosine`, where this returns the vertex most aligned with the centroid
/// instead.
///
/// It does not hold at all under Lance's `Dot`, spelled `1 - dot`: minimising
/// the summed distance there maximises the summed inner product, and the winner
/// is the vector of largest norm - the edge of the cloud, not its middle.
/// [`crate::builder::supported_distance_type`] refuses `Dot` for a related
/// reason.
///
/// What exactness buys at query time is small and shrinks with scale. Against a
/// sample of 256 at `R=64, L=100`: on SIFT 100k over three seeds it returned
/// 0.8-2.8% fewer distances per query at equal recall for 3.1% more build; on
/// SIFT 1M the same comparison landed inside the harness's own run-to-run noise
/// for 0.9% more build. An entry point is where a walk starts, and the longer
/// the walk the less of it that is.
///
/// What does survive is that the walk stopped starting wherever the dice landed:
/// at 100k the seed-to-seed spread of query cost fell from 3.8% to 0.1%. Recall
/// did not move at either scale.
pub fn medoid<S: VectorStore>(store: &S, comparisons: &Comparisons) -> Result<u32> {
    let num_vertices = addressable_len(store.len())?;
    if num_vertices == 0 {
        return Err(Error::invalid_input(
            "Vamana cannot pick a medoid from an empty partition".to_string(),
        ));
    }
    let centroid = centroid(store, num_vertices as usize)?;
    let from_centroid = store.dist_calculator(Arc::new(centroid) as ArrayRef, 0.0);
    // The averaging pass above is not charged, only the scan below. It computes
    // no distances, and counting its `n*d` additions as if it did would move the
    // build cost this crate publishes for a reason that has nothing to do with
    // the graph.
    comparisons.record(num_vertices as u64);

    let mut best = (f32::INFINITY, 0);
    for candidate in 0..num_vertices {
        let distance = from_centroid.distance(candidate);
        if distance < best.0 {
            best = (distance, candidate);
        }
    }
    Ok(best.1)
}

/// The mean of a store's vectors.
///
/// Accumulated in `f64` rather than the `f32` it is made of: a partition holds
/// up to `MAX_PARTITION_ROWS` vectors, and a naive `f32` sum of a million values
/// drifts by roughly a millionth of the total - enough to matter to a coordinate
/// whose spread is smaller than that.
fn centroid<S: VectorStore>(store: &S, num_vertices: usize) -> Result<Float32Array> {
    let mut sums: Vec<f64> = Vec::new();
    let mut counted = 0usize;
    for batch in store.to_batches()? {
        let (values, dimension) = vector_column(&batch)?;
        if dimension == 0 {
            return Err(Error::invalid_input(
                "Vamana cannot pick a medoid from vectors of no width".to_string(),
            ));
        }
        if sums.is_empty() {
            sums = vec![0.0; dimension];
        } else if sums.len() != dimension {
            return Err(Error::invalid_input(format!(
                "Vamana partition mixes vectors of {} and {dimension} dimensions",
                sums.len()
            )));
        }
        let wanted = batch.num_rows() * dimension;
        let Some(values) = values.values().get(..wanted) else {
            return Err(Error::invalid_input(format!(
                "Vamana partition holds {} values for {} vectors of {dimension} dimensions",
                values.len(),
                batch.num_rows()
            )));
        };
        for vector in values.chunks_exact(dimension) {
            for (sum, value) in sums.iter_mut().zip(vector) {
                *sum += *value as f64;
            }
        }
        counted += batch.num_rows();
    }
    // The store is addressed by local id `0..len` everywhere else, so batches
    // that do not add up to that length would put the centroid over a different
    // set of vectors than the scan that follows it.
    if counted != num_vertices {
        return Err(Error::invalid_input(format!(
            "Vamana partition reports {num_vertices} vectors but offers {counted}"
        )));
    }

    let scale = 1.0 / counted as f64;
    Ok(Float32Array::from(
        sums.iter()
            .map(|sum| (sum * scale) as f32)
            .collect::<Vec<_>>(),
    ))
}

/// The vectors of one batch of a store, and their width.
///
/// Found by type rather than by name because [`medoid`] is generic over the
/// store, and the column name is the storage implementation's business. The type
/// is also the check that keeps the centroid honest: a quantized store offers
/// codes, not vectors, and averaging those would produce a query that
/// [`VectorStore::dist_calculator`] cannot even accept - it dispatches on the
/// store's own value type and downcasts the query to it, so a `Float32` centroid
/// against any other store is a panic inside Arrow.
fn vector_column(batch: &RecordBatch) -> Result<(&Float32Array, usize)> {
    let mut columns = batch.columns().iter().filter_map(|column| {
        let vectors = column.as_fixed_size_list_opt()?;
        let values = vectors.values().as_primitive_opt::<Float32Type>()?;
        Some((values, vectors.value_length() as usize))
    });
    let Some(found) = columns.next() else {
        return Err(Error::invalid_input(format!(
            "Vamana needs a FixedSizeList<Float32> column to average and the store offers {}",
            batch.schema_ref()
        )));
    };
    if columns.next().is_some() {
        return Err(Error::invalid_input(format!(
            "Vamana cannot tell which of several FixedSizeList<Float32> columns holds the \
             vectors of {}",
            batch.schema_ref()
        )));
    }
    Ok(found)
}

/// Refuse a pruning slack the graph, or the manifest, could not survive.
///
/// Infinity is refused as well as NaN, and not for the arithmetic: `serde_json`
/// writes any non-finite float as `null`, so an infinite alpha serialises
/// cleanly, commits, and then fails every later `from_json` with "invalid type:
/// null" - an index that can be written once and never opened again.
pub(crate) fn validate_alpha(alpha: f32) -> Result<()> {
    if !alpha.is_finite() || alpha < 1.0 {
        return Err(Error::invalid_input(format!(
            "Vamana alpha must be a finite value of at least 1.0, got {alpha}"
        )));
    }
    Ok(())
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
    prune(store, point, candidates, alpha, max_degree, comparisons).map(|pruned| pruned.neighbors)
}

/// What [`prune`] chose.
pub(crate) struct Pruned {
    /// The out-edges in the order they were chosen: the picks nearest first, ties
    /// broken by id, then whatever the coincident fill added.
    pub(crate) neighbors: Vec<u32>,
    /// Whether the coincident fill supplied part of `neighbors`.
    ///
    /// A filled member was occluded - at zero separation, but occluded - so a list
    /// holding one is not the kind of list [`admit`] can take a newcomer into.
    pub(crate) has_coincident_fill: bool,
}

/// [`robust_prune`], saying as well whether the coincident fill was used.
pub(crate) fn prune<S: VectorStore>(
    store: &S,
    point: u32,
    candidates: Vec<OrderedNode>,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Pruned> {
    validate_alpha(alpha)?;
    if max_degree == 0 {
        return Err(Error::invalid_input(
            "Vamana max_degree must be greater than zero".to_string(),
        ));
    }
    // One pass over a set that is about to be sorted anyway, and it closes two
    // holes a caller can walk into. An id past the end of the store slices the
    // vector buffer out of bounds inside `dist_calculator_from_id`, which panics
    // rather than reporting - `greedy_search` guards this class of input and
    // this function did not. A non-finite distance is worse than a panic: every
    // comparison against NaN is false, so the diversity sweep drops candidate
    // after candidate and the vertex silently ends up with a single out-edge.
    let store_len = store.len();
    for candidate in &candidates {
        if candidate.id as usize >= store_len {
            return Err(Error::invalid_input(format!(
                "Vamana candidate {} is outside a store of {store_len} vectors",
                candidate.id
            )));
        }
        if !candidate.dist.0.is_finite() {
            return Err(Error::invalid_input(format!(
                "Vamana candidate {} has distance {}, which no comparison can order; \
                 the vectors are most likely not finite",
                candidate.id, candidate.dist.0
            )));
        }
    }
    if point as usize >= store_len {
        return Err(Error::invalid_input(format!(
            "Vamana vertex {point} is outside a store of {store_len} vectors"
        )));
    }

    let mut pool = candidates;
    pool.retain(|candidate| candidate.id != point);
    // Every distance the rule below compares is pinned at zero first. Under L2
    // that is a no-op, because an L2 distance is a sum of squares; under cosine
    // it is not, because `1 - dot` in f32 lands a few ULPs either side of zero
    // for a unit vector against itself - measured below zero for 30-44% of
    // random vectors, down to -2.4e-7. A negative distance inverts the rule:
    // multiplying it by `alpha > 1` moves the left-hand side *down*, so
    // `alpha * separation > candidate.dist` is false and the candidate is
    // dropped, while `separation == 0.0` misses it on the way out and the fill
    // below never sees it. A partition of duplicates then collapses into
    // single-edge vertices - which is the very failure this crate refuses `Dot`
    // for. Pinning repairs rounding around zero for a metric that is
    // mathematically non-negative; it is not a licence to take one that is
    // genuinely signed, and `supported_distance_type` still turns `Dot` away.
    for candidate in &mut pool {
        candidate.dist = OrderedFloat(candidate.dist.0.max(0.0));
    }
    // Deduplicated by id before being ordered by distance: `dedup_by_key` only
    // collapses neighbours, and the same id arriving twice with two different
    // distances would not be adjacent under a distance ordering.
    pool.sort_unstable_by(|a, b| a.id.cmp(&b.id).then(a.dist.cmp(&b.dist)));
    pool.dedup_by_key(|candidate| candidate.id);
    pool.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
    let mut pool = VecDeque::from(pool);

    let mut selected = Vec::with_capacity(max_degree);
    // Candidates that sit exactly on top of an already selected vertex. The
    // diversity rule occludes them, and for diversity it is right - they point
    // in no new direction. But they are the *same* point, not a worse one, and
    // dropping every one of them is what turns a partition of duplicates into a
    // chain: `alpha * 0 > 0` is false at every alpha, so the first selection
    // empties the pool and the vertex keeps a single out-edge.
    let mut coincident: Vec<OrderedNode> = Vec::new();
    while let Some(nearest) = pool.pop_front() {
        selected.push(nearest.id);
        if selected.len() == max_degree {
            break;
        }
        let from_nearest = store.dist_calculator_from_id(nearest.id);
        comparisons.record(pool.len() as u64);
        pool.retain(|candidate| {
            let separation = from_nearest.distance(candidate.id).max(0.0);
            if alpha * separation > candidate.dist.0 {
                return true;
            }
            if separation == 0.0 {
                coincident.push(candidate.clone());
            }
            false
        });
    }

    // Only reachable when the pool ran out before the slots did *and* something
    // was occluded at zero separation.
    //
    // "Zero separation" is not the same set of rows under every metric, and the
    // difference is worth naming. Under L2 it is exact duplicates, so ordinary
    // data never takes this path at all. Under cosine the builder stores unit
    // vectors, so it is rows that were *proportional* before normalisation - and
    // then a little more, because `1 - dot` in f32 lands at or below zero once the
    // inner product is within about 6e-8 of one, and the pinning above brings the
    // below back up to it. Those are still the same point
    // in the space the index measures, which is what makes filling the slots with
    // them right rather than a fallback: they point in no new direction, but they
    // are distinct rows a query has to be able to enumerate.
    //
    // Spending the leftover slots on the candidates the alpha rule occluded at a
    // *non-zero* separation was measured on SIFT1M and rejected. Filling every
    // slot that way (69% -> 100% at R=64) costs 3-5% more distances per query at
    // equal recall from 0.97 up, and 7x the distances to build - 78G against 11G.
    // The extra edges buy recall per beam, which is not the same thing as recall
    // per distance, and a denser graph makes every build-time search pay for them.
    let mut has_coincident_fill = false;
    if selected.len() < max_degree {
        coincident.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        for candidate in coincident {
            if selected.len() == max_degree {
                break;
            }
            selected.push(candidate.id);
            has_coincident_fill = true;
        }
    }
    Ok(Pruned {
        neighbors: selected,
        has_coincident_fill,
    })
}

/// What [`admit`] decided about a newcomer to a full list.
#[derive(Debug, PartialEq)]
pub(crate) enum Admission {
    /// A closer member occludes the newcomer, or it is the farthest of them all:
    /// the prune would return the list as it stands.
    Refused,
    /// The list the prune would return: the newcomer in its place, the members it
    /// occludes gone, cut at `max_degree`.
    Admitted(Vec<u32>),
    /// A separation of zero, or a distance the prune would refuse: left to the
    /// prune. A zero separation matters only against a farther member, where the
    /// coincident fill could take the freed slot, but it is rare - duplicates, and
    /// under cosine rows that were proportional before normalisation - so a closer
    /// member's goes to the prune as well.
    Undecided,
}

/// [`prune`] over a full list and one newcomer, when the list is itself what a
/// prune returned.
///
/// `members` must be the `max_degree` out-edges of `owner` exactly as a prune
/// without a coincident fill chose them, at `alpha` or at a smaller one: sorted by
/// distance to `owner`, pinned at zero as the prune pins it, and then by id, no
/// member occluded by a closer one. The prune picks nearest first and drops
/// whatever a pick occludes, so over such a list and a newcomer every member closer
/// than the newcomer is picked again, the newcomer survives only if none of them
/// occludes it, and a farther member drops only if the newcomer occludes it - no
/// member occludes another, so dropping one frees nothing. That takes the
/// newcomer's distance to each member, about `2 * max_degree` distances counting
/// the owner's, where the prune spends about `max_degree^2 / 2` on the whole list
/// again. The answer is the prune's bit for bit: the same distances, pinned and
/// compared the same way.
///
/// A smaller `alpha` for the list is fine because the rule only loosens as `alpha`
/// grows: a separation that kept two members apart at `1.0` keeps them apart at
/// `1.2`. That is what lets a build's first pass hand its lists to the second.
pub(crate) fn admit<S: VectorStore>(
    store: &S,
    owner: u32,
    members: &[u32],
    newcomer: u32,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Admission> {
    debug_assert_eq!(members.len(), max_degree, "vertex {owner} is not full");
    let from_owner = store.dist_calculator_from_id(owner);
    comparisons.record(members.len() as u64 + 1);
    let distances = members
        .iter()
        .map(|member| from_owner.distance(*member))
        .collect::<Vec<_>>();
    let newcomer_distance = from_owner.distance(newcomer);
    // The prune refuses a non-finite distance with an error of its own.
    if !newcomer_distance.is_finite() || distances.iter().any(|distance| !distance.is_finite()) {
        return Ok(Admission::Undecided);
    }
    let key = |distance: f32, id: u32| (OrderedFloat(distance.max(0.0)), id);
    let newcomer_key = key(newcomer_distance, newcomer);
    debug_assert!(
        distances
            .windows(2)
            .zip(members.windows(2))
            .all(|(pair, ids)| key(pair[0], ids[0]) < key(pair[1], ids[1])),
        "vertex {owner}: a list taken for a prune's output is not in prune order"
    );

    let place = members
        .iter()
        .zip(&distances)
        .position(|(member, distance)| key(*distance, *member) > newcomer_key)
        .unwrap_or(members.len());
    if place >= max_degree {
        return Ok(Admission::Refused);
    }
    for member in &members[..place] {
        comparisons.record(1);
        let separation = store
            .dist_calculator_from_id(*member)
            .distance(newcomer)
            .max(0.0);
        if alpha * separation > newcomer_key.0.0 {
            continue;
        }
        return Ok(if separation == 0.0 {
            Admission::Undecided
        } else {
            Admission::Refused
        });
    }

    let mut admitted = Vec::with_capacity(max_degree);
    admitted.extend_from_slice(&members[..place]);
    admitted.push(newcomer);
    let from_newcomer = store.dist_calculator_from_id(newcomer);
    for (member, distance) in members[place..].iter().zip(&distances[place..]) {
        if admitted.len() == max_degree {
            break;
        }
        comparisons.record(1);
        let separation = from_newcomer.distance(*member).max(0.0);
        if alpha * separation > distance.max(0.0) {
            admitted.push(*member);
        } else if separation == 0.0 {
            return Ok(Admission::Undecided);
        }
    }
    Ok(Admission::Admitted(admitted))
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_linalg::distance::DistanceType;
    use lance_linalg::kernels::normalize_fsl;

    use super::*;
    use crate::format::MAX_DEGREE;
    use crate::insert::{concat_vectors, insert_into_partition};
    use crate::partition::Partition;
    use crate::search::{SearchScratch, flat_storage, greedy_search};

    /// Deterministic pseudo-random vectors: a fixed multiplicative congruential
    /// sequence, so the cross-check against Lance runs on the same points every
    /// time without pulling in an RNG.
    fn scattered_values(count: usize) -> Vec<f32> {
        let mut state = 12345u64;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as f32 / (1u64 << 31) as f32
            })
            .collect()
    }

    fn storage_of(values: Vec<f32>, dimension: usize) -> FlatFloatStorage {
        FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension as i32)
                .unwrap(),
            DistanceType::L2,
        )
    }

    fn scattered_storage(num_vertices: usize, dimension: usize) -> FlatFloatStorage {
        storage_of(scattered_values(num_vertices * dimension), dimension)
    }

    /// The same cloud with one point far outside it, on the diagonal.
    fn scattered_storage_with_outlier(
        num_vertices: usize,
        dimension: usize,
        coordinate: f32,
    ) -> FlatFloatStorage {
        let mut values = scattered_values(num_vertices * dimension);
        values.extend(std::iter::repeat_n(coordinate, dimension));
        storage_of(values, dimension)
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

    /// The width is the stride of the on-disk neighbour list and the size of the
    /// allocation a build makes before it computes anything, so a `100_000`
    /// typed where `100` was meant is an allocator abort rather than an error -
    /// on a million-row partition, a request for 400 GB. Refused at the
    /// boundary, which is what the repository's own rule asks for.
    #[test]
    fn a_degree_past_the_ceiling_is_rejected() {
        let storage = scattered_storage(8, 2);
        let params = BuildParams {
            max_degree: MAX_DEGREE + 1,
            ..small_params()
        };
        let error = build_partition(&storage, &params, &Comparisons::default()).unwrap_err();
        assert!(
            error.to_string().contains("must be between 1 and"),
            "{error}"
        );

        // The ceiling itself is allowed, so the bound is not off by one.
        let params = BuildParams {
            max_degree: MAX_DEGREE,
            ..small_params()
        };
        build_partition(&storage, &params, &Comparisons::default()).unwrap();
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

    #[test]
    fn a_built_graph_holds_its_shape() {
        const VERTICES: usize = 500;
        let storage = scattered_storage(VERTICES, 8);
        let params = small_params();
        let built = build_partition(&storage, &params, &Comparisons::default()).unwrap();

        for vertex in 0..VERTICES as u32 {
            let neighbors = built.graph.neighbors(vertex).unwrap();
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
    /// A partition drawn from a handful of distinct values, which is what IVF
    /// routing produces from a dataset with duplicates - Lance's own k-means
    /// warns about exactly this data shape.
    ///
    /// Duplicates are the worst case for the diversity rule: every candidate
    /// occludes every other at zero separation, because `alpha * 0 > 0` is false
    /// at every alpha. Before the coincident fill each vertex kept a *single*
    /// out-edge and a walk from the medoid over 400 vertices reached four.
    ///
    /// Under both metrics, because zero separation is only exactly zero under
    /// L2. Under cosine `d(x, x)` rounds to either side of it, and a negative
    /// one inverted the alpha rule outright: before the distances were pinned at
    /// zero, vertex 35 of the two-value case came out of this build with two of
    /// its sixteen slots filled.
    ///
    /// Full reachability is neither restored nor the goal: with identical
    /// vectors every answer is equally correct, so what has to hold is that a
    /// walk can still enumerate enough distinct rows to answer a query.
    #[test]
    fn a_partition_of_duplicates_keeps_its_edges() {
        const VERTICES: usize = 400;
        const DIMENSION: usize = 8;
        let params = small_params();

        // Seven distinct values across the three cases, not one: whether
        // `d(x, x)` rounds below zero is a property of the vector, so a single
        // base vector could round the harmless way on another target and take
        // the cosine arm of this test with it. It stops at four values because
        // the fill can only spend what the beam collected: these vectors are
        // collinear, so `alpha * d(1, h) > d(0, h)` is false from the second
        // group out and every farther group is dropped, which leaves the
        // coincident copies of the two nearest groups to fill sixteen slots. At
        // eight values a beam of 32 holds about four copies of each and the
        // slots legitimately go unfilled - under L2 as much as under cosine.
        for distance_type in [DistanceType::L2, DistanceType::Cosine] {
            for distinct in [1usize, 2, 4] {
                let values = Float32Array::from(
                    (0..VERTICES)
                        .flat_map(|vertex| {
                            (0..DIMENSION)
                                .map(move |axis| ((vertex % distinct) * DIMENSION + axis) as f32)
                        })
                        .collect::<Vec<_>>(),
                );
                let vectors =
                    FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
                // A cosine build stores unit vectors, so the duplicates a cosine
                // graph is built over are the normalised ones.
                let vectors = if distance_type == DistanceType::Cosine {
                    normalize_fsl(&vectors).unwrap()
                } else {
                    vectors
                };
                let store = FlatFloatStorage::new(vectors, distance_type);
                let built = build_partition(&store, &params, &Comparisons::default()).unwrap();

                for vertex in 0..VERTICES as u32 {
                    assert_eq!(
                        built.graph.neighbors(vertex).unwrap().len(),
                        params.max_degree as usize,
                        "{distance_type}, distinct={distinct}: vertex {vertex} was left short \
                         of its slots"
                    );
                }
                assert!(
                    reachable(&built.graph, built.medoid) > params.max_degree as usize,
                    "{distance_type}, distinct={distinct}: a walk reached {} vertices, so the \
                     graph closed over the medoid's own neighbourhood",
                    reachable(&built.graph, built.medoid)
                );
            }
        }
    }

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
                // Not `< vertices`, which is structural - self-edges and
                // duplicates are refused, so a shorter list is the only kind
                // there is. What is worth asserting is that no vertex was pruned
                // into a dead end. Saturation is *not* the bar even here: at
                // three vertices the diversity rule already occludes one of the
                // two candidates, so the middle vertex keeps a single edge.
                assert!(
                    !built.graph.neighbors(vertex).unwrap().is_empty() || vertices == 1,
                    "vertex {vertex} of a {vertices}-vertex partition was left with no edges"
                );
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
            let chosen = medoid(&storage, &Comparisons::default()).unwrap();
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

        let chosen = medoid(&storage, &Comparisons::default()).unwrap();
        assert!(chosen < 90, "the entry point landed in the sparse cluster");
    }

    /// `argmin_i` of the summed distance to every other vertex, by the
    /// definition of the medoid rather than by any shortcut.
    fn exhaustive_medoid(storage: &FlatFloatStorage) -> u32 {
        let summed = |point: u32| {
            let from = storage.dist_calculator_from_id(point);
            (0..storage.len() as u32)
                .map(|other| from.distance(other))
                .sum::<f32>()
        };
        (0..storage.len() as u32)
            .min_by(|left, right| summed(*left).total_cmp(&summed(*right)))
            .unwrap()
    }

    /// The entry point is the vertex nearest the centroid, and the claim that
    /// makes it the *medoid* is that under a squared metric the two are the same
    /// vertex. Checked against the definition, on clouds far larger than any
    /// sample the old implementation would have drawn.
    ///
    /// The outlier case is here because the docs used to call it the counter-
    /// example: a far point was said to drag the centroid where the true medoid
    /// would not follow. Squared distances do follow, which is why the shortcut
    /// is exact rather than merely convenient.
    #[test]
    fn the_medoid_minimises_the_summed_distance() {
        for (storage, what) in [
            (scattered_storage(200, 4), "a scattered cloud"),
            (line_storage(41), "a line"),
            (
                scattered_storage_with_outlier(64, 4, 100.0),
                "a cloud with one far outlier",
            ),
        ] {
            let chosen = medoid(&storage, &Comparisons::default()).unwrap();
            assert_eq!(chosen, exhaustive_medoid(&storage), "{what}");
        }
    }

    #[test]
    fn an_empty_partition_cannot_be_built() {
        let storage = scattered_storage(0, 4);
        let error =
            build_partition(&storage, &small_params(), &Comparisons::default()).unwrap_err();
        assert!(error.to_string().contains("empty partition"), "{error}");
    }

    /// Checked as arithmetic because the store it protects against cannot be
    /// built in a test: the first count that overflows a `u32` is 16GB of
    /// vectors. Left to the cast, that store builds a graph over
    /// `len % 2^32` vertices, and at exactly `2^32` it takes the medoid of an
    /// empty sample and panics.
    #[test]
    fn a_partition_larger_than_the_id_space_is_refused() {
        assert_eq!(addressable_len(0).unwrap(), 0);
        assert_eq!(
            addressable_len(MAX_PARTITION_ROWS as usize).unwrap(),
            MAX_PARTITION_ROWS
        );
        for num_vertices in [MAX_PARTITION_ROWS as usize + 1, u32::MAX as usize + 1] {
            let error = addressable_len(num_vertices).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains(&num_vertices.to_string()),
                "{error}"
            );
        }
    }

    /// A NaN alpha would make every prune test false and silently keep the
    /// nearest `max_degree` candidates. An infinite one is worse than useless
    /// rather than merely wrong: `serde_json` writes any non-finite float as
    /// `null`, so it commits and then fails every later open.
    #[test]
    fn an_alpha_that_is_not_finite_is_rejected() {
        let storage = scattered_storage(8, 2);
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.9] {
            let error = robust_prune(
                &storage,
                0,
                all_candidates(&storage, 0, 8),
                alpha,
                4,
                &Comparisons::default(),
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("finite value of at least 1.0"),
                "alpha {alpha}: {error}"
            );
        }
    }

    /// `dist_calculator_from_id` slices the vector buffer without checking, so an
    /// id past the end panics the process instead of being reported.
    /// `greedy_search` guards this class of input; this function did not.
    #[test]
    fn a_candidate_outside_the_store_is_rejected() {
        let storage = scattered_storage(8, 2);
        let mut candidates = all_candidates(&storage, 0, 8);
        candidates.push(OrderedNode::new(99, OrderedFloat(0.5)));
        let error =
            robust_prune(&storage, 0, candidates, 1.2, 4, &Comparisons::default()).unwrap_err();
        assert!(error.to_string().contains("outside a store"), "{error}");

        let error = robust_prune(
            &storage,
            99,
            all_candidates(&storage, 0, 8),
            1.2,
            4,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("outside a store"), "{error}");
    }

    /// Every comparison against NaN is false, so the diversity sweep drops each
    /// candidate in turn and the vertex ends up with one out-edge - a graph
    /// silently degenerating instead of a build that stops.
    #[test]
    fn a_non_finite_distance_is_rejected_rather_than_pruned_away() {
        let storage = scattered_storage(8, 2);
        let mut candidates = all_candidates(&storage, 0, 8);
        candidates[3] = OrderedNode::new(3, OrderedFloat(f32::NAN));
        let error =
            robust_prune(&storage, 0, candidates, 1.2, 4, &Comparisons::default()).unwrap_err();
        assert!(
            error.to_string().contains("no comparison can order"),
            "{error}"
        );
    }

    /// The same failure reached through the whole build rather than through one
    /// prune: `create_index` filters non-finite vectors out at assignment, but
    /// `build_partition` is public and the example calls it directly.
    #[test]
    fn a_build_over_non_finite_vectors_stops() {
        const VERTICES: usize = 64;
        let mut values = (0..VERTICES * 2).map(|i| i as f32).collect::<Vec<_>>();
        values[7] = f32::NAN;
        let storage = FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), 2).unwrap(),
            DistanceType::L2,
        );
        let error =
            build_partition(&storage, &small_params(), &Comparisons::default()).unwrap_err();
        assert!(
            error.to_string().contains("no comparison can order"),
            "{error}"
        );
    }

    /// `groups` points `copies` times each, then `unique` more. Under `Cosine` a
    /// copy is a multiple of its point, and everything is normalised, as a cosine
    /// build stores it: the copies then sit at `1 - dot` a few ULPs either side of
    /// zero rather than at zero.
    fn storage_with_copies(
        groups: usize,
        copies: usize,
        unique: usize,
        dimension: usize,
        distance_type: DistanceType,
    ) -> FlatFloatStorage {
        let values = scattered_values((groups + unique) * dimension);
        let (grouped, rest) = values.split_at(groups * dimension);
        let mut all = Vec::with_capacity((groups * copies + unique) * dimension);
        for point in grouped.chunks(dimension) {
            for copy in 0..copies {
                let scale = match distance_type {
                    DistanceType::Cosine => (copy + 1) as f32,
                    _ => 1.0,
                };
                all.extend(point.iter().map(|value| value * scale));
            }
        }
        all.extend_from_slice(rest);
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(all), dimension as i32)
                .unwrap();
        let vectors = match distance_type {
            DistanceType::Cosine => normalize_fsl(&vectors).unwrap(),
            _ => vectors,
        };
        FlatFloatStorage::new(vectors, distance_type)
    }

    /// Every point of `{0, 1, 2, 3}^3`. Squared distances are exact integers, so
    /// they tie at different ids, and `alpha * separation` lands exactly on a
    /// distance at both alphas a build uses: `5 = 5`, and `1.2 * 5` rounds to `6`.
    fn lattice_storage() -> FlatFloatStorage {
        let values = (0..64)
            .flat_map(|point| [point % 4, point / 4 % 4, point / 16].map(|axis| axis as f32))
            .collect::<Vec<_>>();
        storage_of(values, 3)
    }

    fn prune_over(
        storage: &FlatFloatStorage,
        owner: u32,
        ids: &[u32],
        alpha: f32,
        max_degree: usize,
    ) -> Pruned {
        let from_owner = storage.dist_calculator_from_id(owner);
        let candidates = ids
            .iter()
            .map(|id| OrderedNode::new(*id, OrderedFloat(from_owner.distance(*id))))
            .collect();
        prune(
            storage,
            owner,
            candidates,
            alpha,
            max_degree,
            &Comparisons::default(),
        )
        .unwrap()
    }

    /// How often each way through `admit` was taken over a fixture, so that a test
    /// can insist its fixture reached the ways it is there for.
    #[derive(Debug, Default)]
    struct AdmitPaths {
        refused_by_a_closer_member: usize,
        /// Farther than every member and occluded by none of them: refused only
        /// because the list has no room past its last member.
        refused_as_the_farthest: usize,
        admitted_with_the_last_member_cut: usize,
        admitted_with_a_member_dropped: usize,
        admitted_into_a_shorter_list: usize,
        admitted_into_the_last_slot: usize,
        undecided: usize,
        /// Undecided at a farther member whose distance to the owner rounded below
        /// zero: the one case where the pin on that distance decides between
        /// undecided and admitted.
        undecided_at_a_member_below_zero: usize,
        /// Admitted ahead of a member at the same distance, which has a higher id.
        admitted_on_the_id: usize,
        /// Refused by closer members all at exactly `alpha * separation == distance`.
        refused_at_a_tie: usize,
        /// A farther member dropped at exactly that equality.
        dropped_at_a_tie: usize,
        /// Decided while a distance to the owner had rounded below zero.
        decided_below_zero: usize,
    }

    /// `admit` against the prune it stands in for. Each list is a prune's own output
    /// over half the partition, taken only when full and fill-free, and every
    /// vertex outside that half comes in as the newcomer. The prune must return the
    /// list `admit` decided, without the fill, and `admit` must have counted what
    /// it computed: the owner's distances, then one per member it checked.
    fn admit_against_prune(
        storage: &FlatFloatStorage,
        max_degree: usize,
        owners: impl Iterator<Item = u32>,
    ) -> AdmitPaths {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut paths = AdmitPaths::default();
        for owner in owners {
            // A list pruned at 1.0 taking a newcomer at 1.2 is a build's handover
            // from its first pass to its second.
            for (list_alpha, alpha) in [(1.0, 1.0), (1.0, 1.2), (1.2, 1.2)] {
                let mut others = (0..storage.len() as u32)
                    .filter(|id| *id != owner)
                    .collect::<Vec<_>>();
                others.shuffle(&mut rng);
                let (pool, outside) = others.split_at(others.len() / 2);
                let listed = prune_over(storage, owner, pool, list_alpha, max_degree);
                if listed.neighbors.len() < max_degree || listed.has_coincident_fill {
                    continue;
                }
                let members = listed.neighbors;
                let from_owner = storage.dist_calculator_from_id(owner);
                let key = |id: u32| (OrderedFloat(from_owner.distance(id).max(0.0)), id);
                let member_keys = members.iter().map(|id| key(*id)).collect::<Vec<_>>();
                let separation = |from: u32, to: u32| {
                    storage.dist_calculator_from_id(from).distance(to).max(0.0)
                };
                let slack = |from: u32, to: u32| alpha * separation(from, to);

                for &newcomer in outside {
                    let counted = Comparisons::default();
                    let admission = admit(
                        storage, owner, &members, newcomer, alpha, max_degree, &counted,
                    )
                    .unwrap();
                    let expected = prune_over(
                        storage,
                        owner,
                        &[members.as_slice(), &[newcomer]].concat(),
                        alpha,
                        max_degree,
                    );
                    let what = format!(
                        "owner {owner}, newcomer {newcomer}, list at {list_alpha}, newcomer at \
                         {alpha}: {admission:?}"
                    );
                    let newcomer_key = key(newcomer);
                    let distance = newcomer_key.0.0;
                    let place = member_keys
                        .iter()
                        .position(|member| *member > newcomer_key)
                        .unwrap_or(max_degree);

                    let (list, checked) = match &admission {
                        Admission::Undecided => {
                            paths.undecided += 1;
                            // Where `admit` stops: at the first closer member that
                            // occludes the newcomer, or at the first farther one the
                            // newcomer drops while the list still has room.
                            let mut length = place + 1;
                            let stopped = members[..place]
                                .iter()
                                .position(|member| slack(*member, newcomer) <= distance)
                                .or_else(|| {
                                    (place..max_degree).find(|at| {
                                        let member = members[*at];
                                        if length == max_degree {
                                            false
                                        } else if slack(newcomer, member) > key(member).0.0 {
                                            length += 1;
                                            false
                                        } else {
                                            separation(newcomer, member) == 0.0
                                        }
                                    })
                                })
                                .unwrap_or_else(|| panic!("{what}: nowhere to stop"));
                            let zero = if stopped < place {
                                separation(members[stopped], newcomer)
                            } else {
                                separation(newcomer, members[stopped])
                            };
                            assert_eq!(zero, 0.0, "{what}: undecided at member {stopped}");
                            assert_eq!(
                                counted.get(),
                                (max_degree + 2 + stopped) as u64,
                                "{what}: counted distances"
                            );
                            if stopped >= place && from_owner.distance(members[stopped]) < 0.0 {
                                paths.undecided_at_a_member_below_zero += 1;
                            }
                            continue;
                        }
                        Admission::Refused if place == max_degree => {
                            if members
                                .iter()
                                .all(|member| slack(*member, newcomer) > distance)
                            {
                                paths.refused_as_the_farthest += 1;
                            }
                            (members.clone(), 0)
                        }
                        Admission::Refused => {
                            paths.refused_by_a_closer_member += 1;
                            let occluders = members[..place]
                                .iter()
                                .filter(|member| slack(**member, newcomer) <= distance)
                                .collect::<Vec<_>>();
                            if occluders
                                .iter()
                                .all(|member| slack(**member, newcomer) == distance)
                            {
                                paths.refused_at_a_tie += 1;
                            }
                            let first = members
                                .iter()
                                .position(|member| Some(&member) == occluders.first())
                                .unwrap_or_else(|| panic!("{what}: no closer member occludes"));
                            assert!(
                                separation(members[first], newcomer) > 0.0,
                                "{what}: refused at a zero separation, which is the prune's to settle"
                            );
                            (members.clone(), first + 1)
                        }
                        Admission::Admitted(list) => {
                            let examined = if list.len() < max_degree {
                                max_degree - place
                            } else if list.last() == Some(&newcomer) {
                                0
                            } else {
                                1 + members[place..]
                                    .iter()
                                    .position(|member| Some(member) == list.last())
                                    .unwrap_or_else(|| panic!("{what}: ends on a stranger"))
                            };
                            let cut = [
                                &members[..place],
                                &[newcomer],
                                &members[place..max_degree - 1],
                            ]
                            .concat();
                            if list.len() < max_degree {
                                paths.admitted_into_a_shorter_list += 1;
                            } else if *list == cut {
                                paths.admitted_with_the_last_member_cut += 1;
                            } else {
                                paths.admitted_with_a_member_dropped += 1;
                            }
                            if place == max_degree - 1 {
                                paths.admitted_into_the_last_slot += 1;
                            }
                            if member_keys.iter().any(|(member_distance, id)| {
                                *member_distance == newcomer_key.0 && *id > newcomer
                            }) {
                                paths.admitted_on_the_id += 1;
                            }
                            if members[place..place + examined]
                                .iter()
                                .any(|member| slack(newcomer, *member) == key(*member).0.0)
                            {
                                paths.dropped_at_a_tie += 1;
                            }
                            (list.clone(), place + examined)
                        }
                    };
                    assert_eq!(list, expected.neighbors, "{what}");
                    assert!(
                        !expected.has_coincident_fill,
                        "{what}: the prune filled a list admit decided"
                    );
                    assert_eq!(
                        counted.get(),
                        (max_degree + 1 + checked) as u64,
                        "{what}: counted distances"
                    );
                    if from_owner.distance(newcomer) < 0.0
                        || members
                            .iter()
                            .any(|member| from_owner.distance(*member) < 0.0)
                    {
                        paths.decided_below_zero += 1;
                    }
                }
            }
        }
        paths
    }

    #[test]
    fn admitting_a_newcomer_agrees_with_the_prune_over_scattered_points() {
        let storage = storage_with_copies(40, 2, 80, 8, DistanceType::L2);
        let paths = admit_against_prune(&storage, 8, (0..storage.len() as u32).step_by(2));
        assert!(
            paths.refused_by_a_closer_member > 0
                && paths.refused_as_the_farthest > 0
                && paths.admitted_with_the_last_member_cut > 0
                && paths.admitted_with_a_member_dropped > 0
                && paths.admitted_into_a_shorter_list > 0
                && paths.admitted_into_the_last_slot > 0
                && paths.undecided > 0,
            "{paths:?}"
        );
    }

    #[test]
    fn admitting_a_newcomer_agrees_with_the_prune_on_a_lattice() {
        let paths = admit_against_prune(&lattice_storage(), 4, 0..64);
        assert!(
            paths.admitted_on_the_id > 0
                && paths.refused_at_a_tie > 0
                && paths.dropped_at_a_tie > 0,
            "{paths:?}"
        );
    }

    /// Groups of six copies, so that the owner, a member and the newcomer can all be
    /// copies of one point: only then can a member behind the newcomer sit below
    /// zero.
    #[test]
    fn admitting_a_newcomer_agrees_with_the_prune_under_cosine() {
        let storage = storage_with_copies(25, 6, 0, 8, DistanceType::Cosine);
        let paths = admit_against_prune(&storage, 8, (0..storage.len() as u32).step_by(2));
        assert!(
            paths.decided_below_zero > 0
                && paths.undecided > 0
                && paths.undecided_at_a_member_below_zero > 0,
            "{paths:?}"
        );
    }

    /// `insert::insert_point` from before a full list could take a newcomer without
    /// a prune: every back-edge into a full list re-prunes all of it.
    fn insert_with_full_reprunes(
        graph: &mut PartitionGraph,
        storage: &FlatFloatStorage,
        scratch: &mut SearchScratch,
        linking: &Linking,
        point: u32,
        entry_point: u32,
        comparisons: &Comparisons,
    ) {
        let max_degree = graph.max_degree() as usize;
        let from_point = storage.dist_calculator_from_id(point);
        let mut candidates = greedy_search(
            graph,
            &from_point,
            entry_point,
            linking.search_list_size,
            scratch,
            comparisons,
        )
        .unwrap()
        .visited;
        comparisons.record(graph.neighbors(point).unwrap().len() as u64);
        candidates.extend(graph.neighbors(point).unwrap().iter().map(|neighbor| {
            OrderedNode::new(*neighbor, OrderedFloat(from_point.distance(*neighbor)))
        }));
        let selected = robust_prune(
            storage,
            point,
            candidates,
            linking.alpha,
            max_degree,
            comparisons,
        )
        .unwrap();
        graph.set_neighbors(point, &selected).unwrap();

        for neighbor in selected {
            let mut edges = graph.neighbors(neighbor).unwrap().to_vec();
            if edges.contains(&point) {
                continue;
            }
            if edges.len() < max_degree {
                edges.push(point);
                graph.set_neighbors(neighbor, &edges).unwrap();
                continue;
            }
            let from_neighbor = storage.dist_calculator_from_id(neighbor);
            comparisons.record(edges.len() as u64 + 1);
            let contenders = edges
                .iter()
                .chain(std::iter::once(&point))
                .map(|id| OrderedNode::new(*id, OrderedFloat(from_neighbor.distance(*id))))
                .collect();
            let pruned = robust_prune(
                storage,
                neighbor,
                contenders,
                linking.alpha,
                max_degree,
                comparisons,
            )
            .unwrap();
            graph.set_neighbors(neighbor, &pruned).unwrap();
        }
    }

    /// `build_partition` over [`insert_with_full_reprunes`], drawing the random
    /// start and both insertion orders from the same generator in the same order.
    fn build_with_full_reprunes(
        storage: &FlatFloatStorage,
        params: &BuildParams,
        comparisons: &Comparisons,
    ) -> BuiltPartition {
        let num_vertices = storage.len() as u32;
        let mut rng = SmallRng::seed_from_u64(params.seed);
        let row_ids = (0..num_vertices).map(|id| storage.row_id(id)).collect();
        let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids).unwrap();
        randomize(&mut graph, &mut rng).unwrap();
        let medoid = medoid(storage, comparisons).unwrap();
        let mut scratch = SearchScratch::new(num_vertices as usize);
        let mut order = (0..num_vertices).collect::<Vec<_>>();
        for alpha in [1.0, params.alpha] {
            let linking = Linking {
                alpha,
                search_list_size: params.search_list_size,
            };
            order.shuffle(&mut rng);
            for point in &order {
                insert_with_full_reprunes(
                    &mut graph,
                    storage,
                    &mut scratch,
                    &linking,
                    *point,
                    medoid,
                    comparisons,
                );
            }
        }
        BuiltPartition { graph, medoid }
    }

    fn assert_same_graph(built: &PartitionGraph, reference: &PartitionGraph) {
        assert_eq!(built.len(), reference.len());
        for vertex in 0..built.len() as u32 {
            assert_eq!(
                built.neighbors(vertex).unwrap(),
                reference.neighbors(vertex).unwrap(),
                "vertex {vertex}"
            );
        }
    }

    /// Small lists fill up, which is when a back-edge has to fight for a slot.
    fn narrow_params() -> BuildParams {
        BuildParams {
            max_degree: 4,
            search_list_size: 16,
            ..small_params()
        }
    }

    /// The distances each build counted, after insisting they built the same
    /// graph from the same entry point: the build's, then the reference's.
    fn build_both_ways(storage: &FlatFloatStorage) -> (u64, u64) {
        let params = narrow_params();
        let counted = Comparisons::default();
        let built = build_partition(storage, &params, &counted).unwrap();
        let reference_counted = Comparisons::default();
        let reference = build_with_full_reprunes(storage, &params, &reference_counted);
        assert_eq!(built.medoid, reference.medoid);
        assert_same_graph(&built.graph, &reference.graph);
        (counted.get(), reference_counted.get())
    }

    #[test]
    fn a_build_equals_one_that_re_prunes_every_full_list() {
        let (counted, reference) = build_both_ways(&scattered_storage(256, 8));
        assert!(
            counted < reference,
            "{counted} distances against {reference}: the short path never paid"
        );
    }

    /// Groups of `max_degree + 2` copies, so the coincident fill can fill a list.
    /// The counts need only differ, which says `admit` ran: a zero separation pays
    /// for the check and the prune, so over copies it can cost more.
    #[test]
    fn a_build_over_copies_equals_one_that_re_prunes_every_full_list() {
        let copies = narrow_params().max_degree as usize + 2;
        let (counted, reference) =
            build_both_ways(&storage_with_copies(32, copies, 64, 8, DistanceType::L2));
        assert_ne!(counted, reference);
    }

    #[test]
    fn a_cosine_build_equals_one_that_re_prunes_every_full_list() {
        let copies = narrow_params().max_degree as usize + 2;
        let (counted, reference) = build_both_ways(&storage_with_copies(
            32,
            copies,
            64,
            8,
            DistanceType::Cosine,
        ));
        assert_ne!(counted, reference);
    }

    /// Insertion into a graph read back whole, where no list starts out known to be
    /// a prune's output, against the same insertion re-pruning every full list.
    #[test]
    fn an_insertion_equals_one_that_re_prunes_every_full_list() {
        const OLD: usize = 200;
        const NEW: usize = 100;
        const DIMENSION: usize = 8;
        let params = narrow_params();
        let values = scattered_values((OLD + NEW) * DIMENSION);
        let (old_values, new_values) = values.split_at(OLD * DIMENSION);
        let vectors_of = |values: &[f32]| {
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(values.to_vec()),
                DIMENSION as i32,
            )
            .unwrap()
        };
        let (old, new) = (vectors_of(old_values), vectors_of(new_values));
        let old_row_ids = (0..OLD as u64).collect::<Vec<_>>();
        let new_row_ids = (OLD as u64..(OLD + NEW) as u64).collect::<Vec<_>>();
        let storage = flat_storage(&old_row_ids, &old, DistanceType::L2).unwrap();
        let built = build_partition(&storage, &params, &Comparisons::default()).unwrap();
        let partition = Partition::try_new(built.graph, old).unwrap();

        let counted = Comparisons::default();
        let inserted = insert_into_partition(
            &partition,
            &new_row_ids,
            &new,
            built.medoid,
            DistanceType::L2,
            &params,
            &counted,
        )
        .unwrap();

        let reference_counted = Comparisons::default();
        let mut graph = partition.graph().clone();
        graph.extend(&new_row_ids).unwrap();
        let vectors = concat_vectors(&[partition.vectors().clone(), new]).unwrap();
        let storage = flat_storage(graph.row_ids(), &vectors, DistanceType::L2).unwrap();
        let mut scratch = SearchScratch::new(graph.len());
        let linking = Linking {
            alpha: params.alpha,
            search_list_size: params.search_list_size,
        };
        let mut order = (OLD as u32..(OLD + NEW) as u32).collect::<Vec<_>>();
        order.shuffle(&mut SmallRng::seed_from_u64(params.seed));
        for point in order {
            insert_with_full_reprunes(
                &mut graph,
                &storage,
                &mut scratch,
                &linking,
                point,
                built.medoid,
                &reference_counted,
            );
        }
        let reference_medoid = medoid(&storage, &reference_counted).unwrap();

        assert_eq!(inserted.medoid, reference_medoid);
        assert_same_graph(inserted.partition.graph(), &graph);
        assert!(
            counted.get() < reference_counted.get(),
            "{} distances against {}: the short path never paid",
            counted.get(),
            reference_counted.get()
        );
    }
}
