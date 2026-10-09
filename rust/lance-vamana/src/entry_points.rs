// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Starting a lazy walk near its query rather than at the medoid.
//!
//! A walk that starts at its partition's medoid starts at the same vertex
//! whatever it was asked - so every such walk first covers the same ground,
//! from the middle of the data out to wherever its query is. Entry points
//! spread the start over the partition: the vertex nearest each of `K` k-means
//! centroids. A lazy walk starts at them by default
//! ([`crate::query::WalkStart::PreferNearestEntry`]): it measures its query
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
//! At 8 192 and 65 536 rows a partition of SIFT1M, probing 20 and 6 of them,
//! entry points measured 8 to 10 per cent fewer distances at top-10, and at
//! top-100 1.5 to 5 per cent at the narrowest walk and within 1 per cent at the
//! widest. 16 measured as few as 64 at 8 192 rows for a quarter of the
//! training, which is where [`EntryPointParams::default`]'s rule comes from.
//!
//! An index built with codes stores each partition's entry points beside its
//! medoid, trained when the partition is built under the parameters its
//! metadata records, and every maintenance pass that rewrites a partition
//! trains them again. [`crate::query::VamanaIndex::train_entry_points`] trains
//! a set at open instead, under any parameters, by reading every partition's
//! vectors once, and [`crate::query::VamanaIndex::with_entry_points`] hands it
//! to an opened index in place of the stored ones - for one opening, or for
//! several, which is what an [`EntryPoints`] behind an `Arc` is for.

use std::collections::HashMap;
use std::fmt;
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
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::builder::{MAX_KMEANS_SAMPLE_RATE, gather, routing_distance_type};
use crate::partition::Partition;
use crate::search::flat_storage;

/// Vectors sampled per entry point by default: the rate the router trains at
/// by default ([`crate::IndexParams::kmeans_sample_rate`]).
const SAMPLE_RATE: usize = 256;

/// Entry points of a partition larger than [`SMALL_PARTITION_ROWS`] by
/// default: the count that measured best on four million-vector datasets at
/// one partition.
const NUM_ENTRIES: usize = 64;

/// The largest partition, in live vertices, that trains
/// [`SMALL_PARTITION_ENTRIES`] by default: the geometric middle of the two
/// partition sizes the rule was measured at, 8 192 and 65 536 rows of SIFT1M.
const SMALL_PARTITION_ROWS: usize = 23_170;

/// Entry points of a partition of at most [`SMALL_PARTITION_ROWS`] by
/// default. At 8 192 rows a partition, 16 measured as few distances as 64 and
/// trained in a quarter of the time, 2.8 per cent of the build against 11.5.
const SMALL_PARTITION_ENTRIES: usize = 16;

/// Iteration bound of the k-means, the router's default
/// ([`crate::IndexParams::kmeans_max_iters`]).
const MAX_ITERS: u32 = 50;

/// Fewest vertices one piece of the nearest-vertex scan takes, so that a small
/// partition is not cut into pieces too small to be worth handing to the CPU
/// pool: at sixteen dimensions and eight centroids this many is still a few
/// hundred microseconds.
const MIN_CHUNK: usize = 4096;

/// How a partition's entry points are trained: how many for a partition of
/// its size, on how large a sample, from which seed.
///
/// The default is the rule measured on SIFT1M at 8 192, 65 536 and a million
/// rows a partition: 16 entry points for a partition of at most 23 170 live
/// vertices and 64 for a larger one, 256 sampled vectors per entry point, seed
/// 42. An index built with codes trains its entry points under these and
/// records them ([`crate::IndexParams::entry_points`]); every maintenance pass
/// that rewrites a partition trains its entry points again under the recorded
/// ones, and [`crate::query::VamanaIndex::train_entry_points`] trains a set at
/// open under any.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPointParams {
    /// `K` for a partition of more than [`Self::small_partition_rows`] live
    /// vertices: how many centroids its k-means trains.
    ///
    /// Each centroid contributes the live vertex nearest it, two that share a
    /// nearest vertex contribute it once, and one no live vertex measures a
    /// comparable distance to contributes none, so this is a bound rather than
    /// a count: GloVe-200 trained 54 to 56 distinct entry points of 64 under
    /// every seed tried. A partition with no more live vertices than its `K`
    /// gets none, and its walks start at its medoid - choosing among that many
    /// would cost as much as the partition, and the walk would then measure the
    /// ones it did not choose a second time.
    ///
    /// Every walk that starts at entry points pays one coded distance for each
    /// of its partition's, so this is a cost on every query as well as a
    /// training one: 1024 lost to the medoid on every dataset measured.
    ///
    /// At a million centroid values (`K` times the dimension) and over, Lance's
    /// k-means stops assigning exhaustively and searches an HNSW over the
    /// centroids that it builds in parallel, so training stops being
    /// reproducible; `LANCE_USE_HNSW_SPEEDUP_INDEXING=enabled` does the same at
    /// any size, and `disabled` keeps it exhaustive at every size. At 64 that
    /// is first reached at 15 625 dimensions, and a build that gets there says
    /// so in a warning.
    pub num_entries: usize,
    /// The largest partition, in live vertices, that trains
    /// [`Self::small_partition_entries`] rather than [`Self::num_entries`].
    pub small_partition_rows: usize,
    /// `K` for a partition of at most [`Self::small_partition_rows`] live
    /// vertices. No more than [`Self::num_entries`]: a partition's `K` never
    /// falls as it grows, which is what lets a stored list be checked against
    /// the size of its partition alone.
    pub small_partition_entries: usize,
    /// How many live vectors the k-means trains on per entry point, drawn at
    /// random: 4 096 for 16 entry points at the default 256, 16 384 for 64. A
    /// partition with no more live vectors than that trains on all of them.
    ///
    /// From 1 to [`MAX_KMEANS_SAMPLE_RATE`], refused above it rather than
    /// clamped: Lance's k-means keeps only the front `512 * K` of what it is
    /// given.
    pub sample_rate: usize,
    /// Seeds the sample and the starting centroids, so that a partition trains
    /// to the same entry points every time.
    pub seed: u64,
}

impl Default for EntryPointParams {
    /// The measured rule: 16 entry points up to 23 170 live vertices and 64
    /// above, 256 sampled vectors per entry point, seed 42.
    fn default() -> Self {
        Self {
            num_entries: NUM_ENTRIES,
            small_partition_rows: SMALL_PARTITION_ROWS,
            small_partition_entries: SMALL_PARTITION_ENTRIES,
            sample_rate: SAMPLE_RATE,
            seed: 42,
        }
    }
}

impl EntryPointParams {
    /// `num_entries` centroids in every partition whatever its size, 256
    /// sampled vectors per entry point and seed 42.
    pub fn new(num_entries: usize) -> Self {
        Self {
            num_entries,
            small_partition_rows: 0,
            small_partition_entries: num_entries,
            ..Self::default()
        }
    }

    /// Train `num_entries` centroids in a partition of at most `rows` live
    /// vertices, and [`Self::num_entries`] in a larger one.
    pub fn with_small_partitions(mut self, rows: usize, num_entries: usize) -> Self {
        self.small_partition_rows = rows;
        self.small_partition_entries = num_entries;
        self
    }

    /// Sample `sample_rate` live vectors per entry point rather than 256.
    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Draw the sample and the starting centroids from `seed` rather than 42.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// `K` of a partition whose k-means trains on `live_vertices`.
    pub fn entries_for(&self, live_vertices: usize) -> usize {
        if live_vertices <= self.small_partition_rows {
            self.small_partition_entries
        } else {
            self.num_entries
        }
    }

    /// The most live vectors a partition of `num_entries` entry points trains
    /// on. A partition with fewer trains on all of them.
    pub fn sample_size(&self, num_entries: usize) -> usize {
        num_entries.saturating_mul(self.sample_rate)
    }

    /// Why `entry_points` cannot be what training under these produced in a
    /// partition of `num_rows` vertices, if they cannot: a trained list is
    /// ascending, distinct, inside its partition, and no longer than the `K` a
    /// partition of its size trains - which bounds a list trained over fewer
    /// live vertices too, since the rule never gives a smaller partition more.
    ///
    /// Nothing stronger is checked. How many vertices were live when a list
    /// was trained is recorded nowhere, and any ids inside the partition start
    /// a walk correctly.
    pub(crate) fn list_problem(&self, entry_points: &[u32], num_rows: u32) -> Option<String> {
        let allowed = self.entries_for(num_rows as usize);
        if entry_points.len() > allowed {
            return Some(format!(
                "lists {} entry points, more than the {allowed} its parameters train in a \
                 partition of {num_rows} vertices ({self})",
                entry_points.len()
            ));
        }
        if let Some(pair) = entry_points.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Some(format!(
                "lists entry points that are not ascending and distinct: {} comes before {}",
                pair[0], pair[1]
            ));
        }
        entry_points
            .last()
            .filter(|&&last| last >= num_rows)
            .map(|last| format!("lists entry point {last} but holds only {num_rows} vertices"))
    }

    /// Whether Lance's k-means trains these approximately over vectors of
    /// `dimension`, `switch` being what `LANCE_USE_HNSW_SPEEDUP_INDEXING`
    /// says: as `SimpleIndex::may_train_index` in `lance-index` decides,
    /// whose `disabled` never builds the HNSW.
    pub(crate) fn trains_approximately(&self, dimension: usize, switch: Option<&str>) -> bool {
        match switch {
            Some("enabled") => true,
            Some("disabled") => false,
            _ => self.num_entries.saturating_mul(dimension) >= 1_000_000,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.num_entries == 0 || self.small_partition_entries == 0 {
            return Err(Error::invalid_input(format!(
                "entry point num_entries {} and small_partition_entries {} must both be at least \
                 1: each is how many k-means centroids a partition is clustered into",
                self.num_entries, self.small_partition_entries
            )));
        }
        if self.small_partition_entries > self.num_entries {
            return Err(Error::invalid_input(format!(
                "entry point small_partition_entries {} is more than num_entries {}: a partition \
                 of at most {} live vertices would train more entry points than a larger one, \
                 and a stored list could no longer be checked against the size of its partition",
                self.small_partition_entries, self.num_entries, self.small_partition_rows
            )));
        }
        if self.sample_rate == 0 || self.sample_rate > MAX_KMEANS_SAMPLE_RATE {
            return Err(Error::invalid_input(format!(
                "entry point sample_rate {} is outside 1..={MAX_KMEANS_SAMPLE_RATE}: k-means \
                 cannot train a centroid on no vectors, and Lance's trains on the first \
                 {MAX_KMEANS_SAMPLE_RATE} per centroid it is given and never sees the rest",
                self.sample_rate
            )));
        }
        Ok(())
    }
}

impl fmt::Display for EntryPointParams {
    /// `16 up to 23170 rows, else 64; 256 per entry; seed 42`, or
    /// `64; 256 per entry; seed 42` for the one count at every size that
    /// [`Self::new`] makes. Every field is printed that equality compares, so
    /// two parameters a refusal tells apart print differently.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.small_partition_rows == 0 && self.small_partition_entries == self.num_entries {
            write!(f, "{}", self.num_entries)?;
        } else {
            write!(
                f,
                "{} up to {} rows, else {}",
                self.small_partition_entries, self.small_partition_rows, self.num_entries
            )?;
        }
        write!(f, "; {} per entry; seed {}", self.sample_rate, self.seed)
    }
}

/// A segment's entry point parameters as an error message names them:
/// `none` for a segment that keeps no entry points.
pub(crate) fn describe_entry_point_params(params: Option<&EntryPointParams>) -> String {
    params.map_or_else(|| "none".to_string(), ToString::to_string)
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
    /// start at its medoid: one with no more live vertices than the `K`
    /// [`EntryPointParams::entries_for`] gives it.
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
    /// the `K` [`EntryPointParams::entries_for`] gives a partition of its size,
    /// as a stored list has to be. Whether the partitions are an index's
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
            if let Some(problem) = params.list_problem(entries, *num_rows) {
                return Err(Error::invalid_input(format!(
                    "partition {partition_id} of segment {segment} {problem}"
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
    let num_entries = params.entries_for(live.len());
    if live.len() <= num_entries {
        return Ok(Vec::new());
    }
    let (sample_size, seed) = (params.sample_size(num_entries), params.seed);
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

    // A centroid no live vertex measures a comparable distance to - every one
    // of them a zero vector a cosine index normalised to NaN, say - gives no
    // entry point rather than failing the pass: entry points only choose where
    // a walk starts, and a partition left with none starts at its medoid.
    let mut entries = (0..centroids.len())
        .filter_map(|centroid| {
            by_chunk
                .iter()
                .map(|nearest| nearest[centroid])
                .filter(|&(_, local_id)| local_id != u32::MAX)
                .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
                .map(|(_, local_id)| local_id)
        })
        .collect::<Vec<_>>();
    entries.sort_unstable();
    entries.dedup();
    Ok(entries)
}

/// The entry points a partition about to be written stores: trained under
/// `params` over the `live` local ids of `partition`, ascending, or none for a
/// segment that keeps none.
///
/// What every pass that writes a partition from its vectors calls - a build, a
/// consolidation or merge that rewrites one, an insert that creates or grows
/// one - with the vectors its graph was built over: in local-id order and,
/// under cosine, normalised, the same ones an index keeping its vectors stores
/// and [`crate::query::VamanaIndex::train_entry_points`] reads back.
pub(crate) async fn train_stored(
    partition: &Partition,
    distance_type: DistanceType,
    params: Option<&EntryPointParams>,
    live: Vec<u32>,
) -> Result<Arc<[u32]>> {
    let Some(params) = params else {
        return Ok(Arc::from(Vec::new()));
    };
    debug_assert!(
        live.windows(2).all(|pair| pair[0] < pair[1])
            && live
                .last()
                .is_none_or(|&last| (last as usize) < partition.len()),
        "live local ids must be ascending, distinct and inside a partition of {}",
        partition.len()
    );
    let entries = train_partition(
        partition.vectors().clone(),
        partition.graph().row_ids().to_vec(),
        live,
        distance_type,
        params,
    )
    .await?;
    Ok(entries.into())
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
    /// The experiment sampled 256 vectors per entry point; `rate` is that 256.
    fn counted(vectors: &FixedSizeListArray, count: usize, rate: usize, seed: u64) -> Vec<u32> {
        let take_rows = |from: &FixedSizeListArray, rows: &[u32]| {
            take(from, &UInt32Array::from(rows.to_vec()), None)
                .unwrap()
                .as_fixed_size_list()
                .clone()
        };
        let rows = vectors.len();
        let mut rng = SmallRng::seed_from_u64(seed);
        let sample_size = rate * count;
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
            .as_chunks::<{ WIDTH as usize }>()
            .0
            .iter()
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
                counted(&vectors, 8, 256, 42),
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

    /// The rule decides `K` by the live vertices a partition trains on: a
    /// partition above the small-partition bound trains the larger count, and
    /// the same partition with a third of it deleted, at the bound, the
    /// smaller one - as the partition of its live vertices alone would.
    #[tokio::test]
    async fn a_partition_trains_the_count_its_live_vertices_call_for() {
        let vectors = random(3000, 7);
        let rule = EntryPointParams::new(8).with_small_partitions(2000, 4);
        let all = (0..3000).collect::<Vec<_>>();
        assert_eq!(
            trained(&vectors, all, &rule).await,
            counted(&vectors, 8, 256, 42)
        );

        let live = (0..3000).filter(|id| id % 3 != 0).collect::<Vec<u32>>();
        assert_eq!(live.len(), 2000, "the bound itself is the small count's");
        let alone = gather(&vectors, &live).unwrap();
        let renumbered = counted(&alone, 4, 256, 42)
            .into_iter()
            .map(|at| live[at as usize])
            .collect::<Vec<_>>();
        assert_eq!(trained(&vectors, live, &rule).await, renumbered);
    }

    /// The sample rate reaches the training: at each rate a partition trains
    /// what the counted experiment trains on a sample of that size, and at 128
    /// that is not what it trains at 256. Both ends of the range train - at 1
    /// the k-means gets one vector per centroid, the fewest it takes, and at
    /// 512 more than this partition holds, so all of it.
    #[tokio::test]
    async fn the_sample_rate_sizes_the_sample() {
        let vectors = random(3000, 7);
        let all = (0..3000).collect::<Vec<_>>();
        for sample_rate in [1, 128, 512] {
            assert_eq!(
                trained(
                    &vectors,
                    all.clone(),
                    &EntryPointParams::new(8).with_sample_rate(sample_rate),
                )
                .await,
                counted(&vectors, 8, sample_rate, 42),
                "sample_rate {sample_rate}"
            );
        }
        assert_ne!(counted(&vectors, 8, 128, 42), counted(&vectors, 8, 256, 42));
    }

    #[test]
    fn a_partition_trains_sixteen_up_to_the_bound_and_sixty_four_above_it() {
        let rule = EntryPointParams::default();
        for (live_vertices, num_entries) in [(0, 16), (23_170, 16), (23_171, 64), (1_000_000, 64)] {
            assert_eq!(
                rule.entries_for(live_vertices),
                num_entries,
                "{live_vertices}"
            );
        }
        assert_eq!(rule.sample_size(16), 4096);
        assert_eq!(rule.sample_size(64), 16_384);
        let fixed = EntryPointParams::new(8);
        for live_vertices in [0, 1, 23_170, 23_171, 1_000_000] {
            assert_eq!(fixed.entries_for(live_vertices), 8, "{live_vertices}");
        }
    }

    #[test]
    fn parameters_out_of_shape_are_refused() {
        for (params, says) in [
            (EntryPointParams::new(0), "num_entries 0"),
            (
                EntryPointParams::default().with_small_partitions(100, 0),
                "small_partition_entries 0",
            ),
            (
                EntryPointParams::new(8).with_small_partitions(100, 9),
                "small_partition_entries 9 is more than num_entries 8",
            ),
            (
                EntryPointParams::default().with_sample_rate(0),
                "sample_rate 0",
            ),
            (
                EntryPointParams::default().with_sample_rate(513),
                "sample_rate 513",
            ),
        ] {
            let error = params.validate().unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains(says), "{error}");
        }
        for params in [
            EntryPointParams::default(),
            EntryPointParams::default().with_sample_rate(1),
            EntryPointParams::default().with_sample_rate(512),
            EntryPointParams::new(1),
        ] {
            params.validate().unwrap();
        }
    }

    /// What an index records, and the line `info` prints. A change to either
    /// changes what a stored index says it was trained with, which is a new
    /// format version.
    #[test]
    fn the_default_rule_is_recorded_and_printed_as_it_is() {
        let rule = EntryPointParams::default();
        let json = serde_json::to_string(&rule).unwrap();
        assert_eq!(
            json,
            r#"{"num_entries":64,"small_partition_rows":23170,"small_partition_entries":16,"sample_rate":256,"seed":42}"#
        );
        assert_eq!(
            serde_json::from_str::<EntryPointParams>(&json).unwrap(),
            rule
        );
        assert_eq!(
            rule.to_string(),
            "16 up to 23170 rows, else 64; 256 per entry; seed 42"
        );
        assert_eq!(
            EntryPointParams::new(64).with_seed(7).to_string(),
            "64; 256 per entry; seed 7"
        );
        // Unequal to `new(64)` though it trains the same, and printed so: a
        // refusal between the two must not name the same parameters twice.
        assert_eq!(
            EntryPointParams::new(64)
                .with_small_partitions(23_170, 64)
                .to_string(),
            "64 up to 23170 rows, else 64; 256 per entry; seed 42"
        );
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

    /// A vertex no distance reaches is never an entry point: under cosine, a
    /// zero vector, which normalises to NaN. A partition of nothing else
    /// trains none, and its walks start at its medoid, rather than failing
    /// the build or the pass that trains it.
    #[tokio::test]
    async fn a_vertex_no_distance_reaches_is_never_an_entry_point() {
        let trained = |values: Vec<f32>| async move {
            let vectors =
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), WIDTH).unwrap();
            let normalised = lance_linalg::kernels::normalize_fsl(&vectors).unwrap();
            let rows = normalised.len();
            train_partition(
                normalised,
                (0..rows as u64).collect(),
                (0..rows as u32).collect(),
                DistanceType::Cosine,
                &EntryPointParams::new(8),
            )
            .await
            .unwrap()
        };
        let every_third_zero = random(3000, 7)
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .as_chunks::<{ WIDTH as usize }>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(row, vector)| {
                vector
                    .iter()
                    .map(move |&value| if row % 3 == 0 { 0.0 } else { value })
            })
            .collect();
        let entries = trained(every_third_zero).await;
        assert!(!entries.is_empty());
        assert!(entries.iter().all(|row| row % 3 != 0), "{entries:?}");

        assert_eq!(
            trained(vec![0.0; 100 * WIDTH as usize]).await,
            Vec::<u32>::new()
        );
    }

    /// Lance's k-means assigns through an HNSW over the centroids, which it
    /// builds in parallel, from a million centroid values on - 64 entry
    /// points at 15 625 dimensions - and at any size when told to; told not
    /// to, it never does. Anything else it reads as its own default.
    #[test]
    fn training_is_approximate_where_lances_kmeans_searches_an_hnsw() {
        let rule = EntryPointParams::default();
        for (dimension, switch, approximate) in [
            (15_624, None, false),
            (15_625, None, true),
            (15_625, Some("auto"), true),
            (15_625, Some("disabled"), false),
            (8, Some("enabled"), true),
        ] {
            assert_eq!(
                rule.trains_approximately(dimension, switch),
                approximate,
                "{dimension} dimensions, {switch:?}"
            );
        }
    }
}
