// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Starting a lazy walk near its query rather than at the medoid.
//!
//! Every walk of a partition starts at one vertex, its medoid, whatever it was
//! asked - so every walk first covers the same ground, from the middle of the
//! data out to wherever its query is. Entry points spread the start over the
//! partition: the vertex nearest each of `K` k-means centroids. A walk asked to
//! start at them ([`crate::query::WalkStart::NearestEntry`]) measures its query
//! against each of them by code and starts at the nearest.
//!
//! Counted at equal recall on four million-vector datasets at one partition,
//! `R = 70`, eight-bit scalar codes and the stop margin
//! (`examples/entry_points_walk.rs`), 64 entry points measured 15, 6, 16 and 9
//! per cent fewer distances at top-10 on SIFT, GloVe-200, Cohere and GIST - the
//! 64 that choosing costs included - and 2 to 8 per cent at top-100. A walk
//! started at the query's own nearest neighbour measured 19 to 51 per cent fewer
//! at top-10, so most of the approach is still walked. Random vertices in place
//! of the k-means ones measured 3 to 5 points less, and seeding the walk with
//! all 64 rather than starting at the nearest lost at every `K`: how near the
//! start is matters more than how many starts there are.
//!
//! Timed on the same indexes (`examples/ivf_rq_ab.rs`), that saving became 8
//! to 11 per cent of the search phase at top-10 with one query in flight, and
//! 6 to 10 per cent of the time a query at twelve in flight on GloVe-200,
//! Cohere and GIST; on SIFT, whose query takes 32 us there, the difference was
//! inside the noise.
//!
//! Nothing here is stored. [`crate::query::VamanaIndex::train_entry_points`]
//! reads every partition's vectors once and trains them, and
//! [`crate::query::VamanaIndex::with_entry_points`] hands the result to an
//! opened index - or to several openings of the same one, which is what an
//! [`EntryPoints`] behind an `Arc` is for.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array};
use futures::future::try_join_all;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::kmeans::{KMeans, KMeansParams};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use uuid::Uuid;

use crate::builder::{MAX_KMEANS_SAMPLE_RATE, gather, routing_distance_type};
use crate::search::flat_storage;

/// Vectors sampled per entry point when [`EntryPointParams::sample_size`] is
/// unset: the rate the router trains at by default
/// ([`crate::IndexParams::kmeans_sample_rate`]).
const SAMPLE_RATE: usize = 256;

/// Iteration bound of the k-means, the router's default
/// ([`crate::IndexParams::kmeans_max_iters`]).
const MAX_ITERS: u32 = 50;

/// Fewest vertices one piece of the nearest-vertex scan takes, so that a small
/// partition is not cut into pieces too small to be worth handing to the CPU
/// pool: at sixteen dimensions and eight centroids this many is still a few
/// hundred microseconds.
const MIN_CHUNK: usize = 4096;

/// How [`crate::query::VamanaIndex::train_entry_points`] picks a partition's
/// entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPointParams {
    /// `K`: how many centroids the k-means trains in each partition.
    ///
    /// Each centroid contributes the live vertex nearest it, and two that share
    /// a nearest vertex contribute it once, so this is a bound rather than a
    /// count: GloVe-200 trained 54 to 56 distinct entry points of 64 under
    /// every seed tried. A partition with no more live vertices than this gets
    /// none, and its walks start at its medoid - choosing among that many would
    /// cost as much as the partition, and the walk would then measure the ones
    /// it did not choose a second time.
    ///
    /// Every walk that starts at entry points pays one coded distance for each
    /// of its partition's, so this is a cost on every query as well as a
    /// training one: 1024 lost to the medoid on every dataset measured.
    ///
    /// At a million centroid values (`K` times the dimension) and over, Lance's
    /// k-means stops assigning exhaustively and searches an HNSW over the
    /// centroids that it builds in parallel, so training stops being
    /// reproducible; `LANCE_USE_HNSW_SPEEDUP_INDEXING` does the same at any
    /// size. At 64 that is first reached at 15 625 dimensions.
    pub num_entries: usize,
    /// How many of a partition's live vectors the k-means trains on, drawn at
    /// random. `None` takes 256 per entry point.
    ///
    /// At least [`Self::num_entries`], and at most [`MAX_KMEANS_SAMPLE_RATE`]
    /// per entry point, refused above it rather than clamped: Lance's k-means
    /// keeps only the front `512 * K` of what it is given. A partition with no
    /// more live vectors than this trains on all of them.
    pub sample_size: Option<usize>,
    /// Seeds the sample and the starting centroids, so that a partition trains
    /// to the same entry points every time.
    pub seed: u64,
}

impl Default for EntryPointParams {
    fn default() -> Self {
        Self::new(64)
    }
}

impl EntryPointParams {
    /// `num_entries` centroids a partition, the default sample and seed 42.
    pub fn new(num_entries: usize) -> Self {
        Self {
            num_entries,
            sample_size: None,
            seed: 42,
        }
    }

    /// Train on at most `sample_size` of a partition's live vectors rather than
    /// 256 per entry point.
    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = Some(sample_size);
        self
    }

    /// Draw the sample and the starting centroids from `seed` rather than 42.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The most vectors a partition trains on: [`Self::sample_size`], or 256
    /// per entry point when it is unset. A partition with fewer live vectors
    /// trains on all of them.
    pub fn resolved_sample_size(&self) -> usize {
        self.sample_size
            .unwrap_or_else(|| self.num_entries.saturating_mul(SAMPLE_RATE))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.num_entries == 0 {
            return Err(Error::invalid_input(
                "entry point num_entries must be at least 1: it is how many k-means centroids \
                 each partition is clustered into"
                    .to_string(),
            ));
        }
        let sample_size = self.resolved_sample_size();
        if sample_size < self.num_entries {
            return Err(Error::invalid_input(format!(
                "entry point sample_size {sample_size} is smaller than num_entries {}: k-means \
                 cannot train {} centroids on fewer vectors",
                self.num_entries, self.num_entries
            )));
        }
        let ceiling = self.num_entries.saturating_mul(MAX_KMEANS_SAMPLE_RATE);
        if sample_size > ceiling {
            return Err(Error::invalid_input(format!(
                "entry point sample_size {sample_size} is over {MAX_KMEANS_SAMPLE_RATE} vectors \
                 per entry point, {ceiling} at num_entries {}: Lance's k-means trains on the \
                 first {ceiling} vectors it is given and never sees the rest",
                self.num_entries
            )));
        }
        Ok(())
    }
}

/// One partition's entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionEntryPoints {
    /// The segment the partition belongs to: every segment of an index has its
    /// own partition 0.
    pub segment: Uuid,
    /// The partition's id within its segment: its IVF centroid's.
    pub partition_id: u32,
    /// How many vertices the partition held when its entry points were chosen.
    /// An index whose partition holds another number refuses them.
    pub num_rows: u32,
    /// Local ids, ascending and distinct. Empty for a partition whose walks
    /// start at its medoid: one with no more live vertices than
    /// [`EntryPointParams::num_entries`].
    pub entries: Vec<u32>,
}

/// Entry points for every non-empty partition of one index.
///
/// From [`crate::query::VamanaIndex::train_entry_points`], or from
/// [`Self::from_partitions`] for entry points chosen elsewhere; read by a walk
/// once [`crate::query::VamanaIndex::with_entry_points`] has checked them
/// against the index they are given to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoints {
    params: EntryPointParams,
    partitions: Vec<PartitionEntryPoints>,
    /// Where each partition is in `partitions`, by segment and partition id.
    positions: HashMap<(Uuid, u32), usize>,
}

impl EntryPoints {
    /// Entry points chosen elsewhere - read back from a file a training pass
    /// wrote, say.
    ///
    /// Each partition's list has to be what training would have produced the
    /// shape of: ascending, distinct, inside the partition, and no longer than
    /// [`EntryPointParams::num_entries`]. Whether the partitions are an index's
    /// is for [`crate::query::VamanaIndex::with_entry_points`] to check, once
    /// there is an index to check them against.
    pub fn from_partitions(
        params: EntryPointParams,
        partitions: Vec<PartitionEntryPoints>,
    ) -> Result<Self> {
        params.validate()?;
        let mut positions = HashMap::with_capacity(partitions.len());
        for (position, partition) in partitions.iter().enumerate() {
            let PartitionEntryPoints {
                segment,
                partition_id,
                num_rows,
                entries,
            } = partition;
            if entries.len() > params.num_entries {
                return Err(Error::invalid_input(format!(
                    "partition {partition_id} of segment {segment} has {} entry points, more \
                     than num_entries {}",
                    entries.len(),
                    params.num_entries
                )));
            }
            if let Some(pair) = entries.windows(2).find(|pair| pair[0] >= pair[1]) {
                return Err(Error::invalid_input(format!(
                    "the entry points of partition {partition_id} of segment {segment} must be \
                     ascending and distinct, but {} comes before {}",
                    pair[0], pair[1]
                )));
            }
            if let Some(&last) = entries.last()
                && last >= *num_rows
            {
                return Err(Error::invalid_input(format!(
                    "entry point {last} is outside partition {partition_id} of segment \
                     {segment}, which has {num_rows} vertices"
                )));
            }
            if positions
                .insert((*segment, *partition_id), position)
                .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "partition {partition_id} of segment {segment} is listed twice"
                )));
            }
        }
        Ok(Self {
            params,
            partitions,
            positions,
        })
    }

    /// What the entry points were trained with, or said to have been by
    /// [`Self::from_partitions`].
    pub fn params(&self) -> &EntryPointParams {
        &self.params
    }

    /// Every partition's entry points, in the order they were given: as
    /// [`crate::query::VamanaIndex::train_entry_points`] gives them, the
    /// index's segments in manifest order and each one's partitions by id.
    /// Two sets equal partition for partition but listed in another order are
    /// not equal.
    pub fn partitions(&self) -> &[PartitionEntryPoints] {
        &self.partitions
    }

    pub(crate) fn of(&self, segment: Uuid, partition_id: u32) -> Option<&PartitionEntryPoints> {
        self.positions
            .get(&(segment, partition_id))
            .map(|&position| &self.partitions[position])
    }
}

/// One partition's entry points, ascending: the live vertex nearest each
/// centroid of a k-means over a sample of its live vectors.
///
/// The router's recipe ([`crate::builder`]'s `train_router`) under the entry
/// points' own seed: the sample is drawn with `rand::seq::index::sample` and
/// kept in the order it draws it, the starting centroids are `K` rows of that
/// sample drawn with the same generator, and the training is assigned by the
/// routing metric. The nearest vertex is by the partition's own metric, the
/// first of the nearest in id order on a tie, so the answer does not depend on
/// how the scan below is divided.
///
/// `vectors` and `row_ids` are the whole partition in local-id order, `live`
/// the local ids of its live vertices, ascending.
pub(crate) async fn train_partition(
    vectors: FixedSizeListArray,
    row_ids: Vec<u64>,
    live: Vec<u32>,
    distance_type: DistanceType,
    params: &EntryPointParams,
) -> Result<Vec<u32>> {
    let num_entries = params.num_entries;
    if live.len() <= num_entries {
        return Ok(Vec::new());
    }
    let (sample_size, seed) = (params.resolved_sample_size(), params.seed);
    let live = Arc::new(live);

    // The store on the CPU pool too: under cosine it measures every vector's
    // norm as it is built. The k-means parallelises with rayon, which has a
    // pool of its own, so a CPU-pool worker parked on it cannot starve the pool
    // it is running in.
    let (store, centroids) = spawn_cpu({
        let live = live.clone();
        move || {
            let store = flat_storage(&row_ids, &vectors, distance_type)?;
            let centroids = centroids(
                &vectors,
                &live,
                num_entries,
                sample_size,
                seed,
                distance_type,
            )?;
            Ok::<_, Error>((Arc::new(store), Arc::new(centroids)))
        }
    })
    .await?;

    let chunk = live
        .len()
        .div_ceil(get_num_compute_intensive_cpus().max(1))
        .max(MIN_CHUNK);
    let by_chunk = try_join_all((0..live.len()).step_by(chunk).map(|start| {
        let (store, live, centroids) = (store.clone(), live.clone(), centroids.clone());
        spawn_cpu(move || {
            let end = (start + chunk).min(live.len());
            Ok::<_, Error>(nearest_vertices(&store, &live[start..end], &centroids))
        })
    }))
    .await?;

    let mut entries = (0..centroids.len())
        .map(|centroid| {
            by_chunk
                .iter()
                .map(|nearest| nearest[centroid])
                .filter(|&(_, local_id)| local_id != u32::MAX)
                .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
                .map(|(_, local_id)| local_id)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "no live vertex of {} measured a comparable distance to entry point \
                         centroid {centroid}",
                        live.len()
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort_unstable();
    entries.dedup();
    Ok(entries)
}

/// The k-means centroids, one array a centroid.
fn centroids(
    vectors: &FixedSizeListArray,
    live: &[u32],
    num_entries: usize,
    sample_size: usize,
    seed: u64,
    distance_type: DistanceType,
) -> Result<Vec<ArrayRef>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let training = if live.len() > sample_size {
        let picked = rand::seq::index::sample(&mut rng, live.len(), sample_size)
            .into_iter()
            .map(|at| live[at])
            .collect::<Vec<_>>();
        gather(vectors, &picked)?
    } else {
        gather(vectors, live)?
    };
    let init = gather(
        &training,
        &rand::seq::index::sample(&mut rng, training.len(), num_entries)
            .into_iter()
            .map(|row| row as u32)
            .collect::<Vec<_>>(),
    )?;
    let kmeans_params = KMeansParams::new(
        Some(Arc::new(init)),
        MAX_ITERS,
        1,
        routing_distance_type(distance_type),
    )
    .with_hierarchical_k(1);
    let kmeans = KMeans::new_with_params(&training, num_entries, &kmeans_params)?;
    let dimension = vectors.value_length() as usize;
    let values = kmeans
        .centroids
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| {
            Error::internal(format!(
                "k-means returned {} centroids, not the f32 it was trained on",
                kmeans.centroids.data_type()
            ))
        })?
        .values();
    if values.len() != num_entries * dimension {
        return Err(Error::internal(format!(
            "k-means returned {} centroid values for {num_entries} centroids of {dimension} \
             dimensions",
            values.len()
        )));
    }
    Ok(values
        .chunks_exact(dimension)
        .map(|centroid| Arc::new(Float32Array::from(centroid.to_vec())) as ArrayRef)
        .collect())
}

/// For each centroid, the nearest of `local_ids` and its distance, or
/// `(inf, u32::MAX)` when none of them measured a comparable one.
///
/// A block of vertices at a time against every centroid, so that each vector
/// comes out of memory once rather than once a centroid: at `d = 960` a
/// partition of a million is 3.8 GB, and 64 passes over it would be most of
/// the training.
fn nearest_vertices(
    store: &FlatFloatStorage,
    local_ids: &[u32],
    centroids: &[ArrayRef],
) -> Vec<(f32, u32)> {
    const BLOCK: usize = 512;
    let calculators = centroids
        .iter()
        .map(|centroid| store.dist_calculator(centroid.clone(), 0.0))
        .collect::<Vec<_>>();
    let mut nearest = vec![(f32::INFINITY, u32::MAX); centroids.len()];
    for block in local_ids.chunks(BLOCK) {
        for (calculator, nearest) in calculators.iter().zip(nearest.iter_mut()) {
            for &local_id in block {
                let distance = calculator.distance(local_id);
                if distance < nearest.0 {
                    *nearest = (distance, local_id);
                }
            }
        }
    }
    nearest
}

#[cfg(test)]
mod tests {
    use arrow_array::UInt32Array;
    use arrow_select::take::take;
    use lance_arrow::FixedSizeListArrayExt;
    use rand::Rng;

    use super::*;

    const WIDTH: i32 = 8;

    fn random(rows: usize, seed: u64) -> FixedSizeListArray {
        let mut rng = SmallRng::seed_from_u64(seed);
        let values = (0..rows * WIDTH as usize)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), WIDTH).unwrap()
    }

    /// The counted experiment's training over every row of `vectors`, as
    /// `examples/entry_points_walk.rs` wrote it (`kmeans_entries` and
    /// `nearest_vertices`): what a partition with nothing deleted has to train.
    fn counted(vectors: &FixedSizeListArray, count: usize, seed: u64) -> Vec<u32> {
        let take_rows = |from: &FixedSizeListArray, rows: &[u32]| {
            take(from, &UInt32Array::from(rows.to_vec()), None)
                .unwrap()
                .as_fixed_size_list()
                .clone()
        };
        let rows = vectors.len();
        let mut rng = SmallRng::seed_from_u64(seed);
        let sample_size = 256 * count;
        let training = if rows > sample_size {
            let picked = rand::seq::index::sample(&mut rng, rows, sample_size)
                .into_iter()
                .map(|row| row as u32)
                .collect::<Vec<_>>();
            take_rows(vectors, &picked)
        } else {
            vectors.clone()
        };
        let init = take_rows(
            &training,
            &rand::seq::index::sample(&mut rng, training.len(), count)
                .into_iter()
                .map(|row| row as u32)
                .collect::<Vec<_>>(),
        );
        let params =
            KMeansParams::new(Some(Arc::new(init)), 50, 1, DistanceType::L2).with_hierarchical_k(1);
        let kmeans = KMeans::new_with_params(&training, count, &params).unwrap();
        let store = FlatFloatStorage::new(vectors.clone(), DistanceType::L2);
        let mut entries = kmeans
            .centroids
            .as_primitive::<Float32Type>()
            .values()
            .chunks_exact(WIDTH as usize)
            .map(|centroid| {
                let target = Arc::new(Float32Array::from(centroid.to_vec())) as ArrayRef;
                let calculator = store.dist_calculator(target, 0.0);
                let mut nearest = (f32::INFINITY, u32::MAX);
                for row in 0..rows as u32 {
                    let distance = calculator.distance(row);
                    if distance < nearest.0 {
                        nearest = (distance, row);
                    }
                }
                nearest.1
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        entries.dedup();
        entries
    }

    async fn trained(
        vectors: &FixedSizeListArray,
        live: Vec<u32>,
        params: &EntryPointParams,
    ) -> Vec<u32> {
        let row_ids = (0..vectors.len() as u64).collect();
        train_partition(vectors.clone(), row_ids, live, DistanceType::L2, params)
            .await
            .unwrap()
    }

    /// A partition trains the entry points the counted experiment trained, to
    /// the last vertex: drawing more rows than the sample and fewer, and over
    /// a scan cut into several pieces, which is what a partition of more than
    /// [`MIN_CHUNK`] vertices on more than one core gets.
    ///
    /// The measurement the entry points were adopted on is that experiment's,
    /// and only this pins the recipe against it without its million-vector
    /// datasets: the order of the two draws, the sample size, the iteration
    /// bound and the hierarchy switch would each train other entry points.
    #[tokio::test]
    async fn a_partition_trains_the_entry_points_the_counted_experiment_trained() {
        for rows in [10_000, 1500] {
            let vectors = random(rows, 7);
            let all = (0..rows as u32).collect::<Vec<_>>();
            assert_eq!(
                trained(&vectors, all, &EntryPointParams::new(8)).await,
                counted(&vectors, 8, 42),
                "{rows} rows"
            );
        }
    }

    /// A partition trains as the partition of its live vertices alone would,
    /// renumbered: the sample, the starting centroids and the scan all draw
    /// from the live vertices and from nothing else.
    #[tokio::test]
    async fn a_partition_trains_as_its_live_vertices_alone_would() {
        // Live vertices more than the sample, so it is drawn, and fewer.
        for rows in [4000, 2400] {
            let vectors = random(rows, 7);
            let live = (0..rows as u32)
                .filter(|id| id % 3 != 0)
                .collect::<Vec<_>>();
            let alone = gather(&vectors, &live).unwrap();
            let params = EntryPointParams::new(8);
            let renumbered = trained(&alone, (0..live.len() as u32).collect(), &params)
                .await
                .into_iter()
                .map(|at| live[at as usize])
                .collect::<Vec<_>>();
            assert_eq!(
                trained(&vectors, live, &params).await,
                renumbered,
                "{rows} rows"
            );
        }
    }

    /// The seed reaches the training: another one draws another sample and
    /// other starting centroids, and here trains other entry points.
    #[tokio::test]
    async fn another_seed_trains_other_entry_points() {
        let vectors = random(3000, 7);
        let all = (0..3000).collect::<Vec<_>>();
        assert_ne!(
            trained(&vectors, all.clone(), &EntryPointParams::new(8)).await,
            trained(&vectors, all, &EntryPointParams::new(8).with_seed(43)).await
        );
    }
}
