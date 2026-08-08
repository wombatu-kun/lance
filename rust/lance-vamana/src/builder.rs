// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building a Vamana index over a Lance dataset, and committing it.
//!
//! Every step goes through Lance's published API: the column is read with the
//! ordinary scanner, the router is trained with Lance's own k-means, and the
//! finished segment is committed with `commit_existing_index_segments`. No patch
//! to Lance is involved anywhere, which is the point of this stage.
//!
//! The whole vector column is held in memory for the duration of a build. That
//! is a property of the builder, not of the index: a query reads one partition
//! at a time. Streaming the build is a later concern.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, FixedSizeListArray, UInt32Array};
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::kmeans::{KMeans, KMeansParams, compute_partitions_arrow_array};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl;
use object_store::path::Path;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use uuid::Uuid;

use crate::build::{BuildParams, build_partition};
use crate::format::{FORMAT_VERSION, IndexMetadata, RowIdMode};
use crate::io::SegmentWriter;
use crate::partition::Partition;
use crate::search::{Comparisons, flat_storage};
use crate::segment::SegmentManifest;

/// On-disk index version recorded in the dataset manifest.
pub const INDEX_VERSION: i32 = 1;

/// The `type_url` of the details blob that travels with a committed segment.
///
/// Deliberately ours and deliberately unresolvable by Lance. A url Lance can
/// resolve - `VectorIndexDetails` - puts the segment under a version ceiling
/// this crate does not control, and a segment above that ceiling disappears
/// *silently* when the dataset is reopened. An unresolvable one is kept as is.
/// The payload is empty because the segment's own `index.idx` is the only
/// source of truth about its contents.
pub const INDEX_DETAILS_TYPE_URL: &str = "type.googleapis.com/lance.vamana.VamanaIndexDetails";

/// How to build one Vamana index segment.
#[derive(Debug, Clone)]
pub struct IndexParams {
    /// Vector column to index. Must be `FixedSizeList<Float32, dim>`.
    pub column: String,
    /// Number of IVF partitions, i.e. how many k-means centroids to train.
    pub num_partitions: u32,
    pub distance_type: DistanceType,
    /// Graph parameters, applied to every partition.
    pub graph: BuildParams,
    /// Iteration bound for the router's k-means.
    pub kmeans_max_iters: u32,
    /// Vectors sampled per centroid when training the router.
    ///
    /// Capped at 512 by Lance, which re-slices the training set to `512 * k`
    /// before it starts, so anything above that has no effect.
    pub kmeans_sample_rate: usize,
}

impl IndexParams {
    pub fn new(column: impl Into<String>, num_partitions: u32) -> Self {
        Self {
            column: column.into(),
            num_partitions,
            distance_type: DistanceType::L2,
            graph: BuildParams::default(),
            kmeans_max_iters: 50,
            kmeans_sample_rate: 256,
        }
    }

    pub fn with_distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = distance_type;
        self
    }

    pub fn with_graph_params(mut self, graph: BuildParams) -> Self {
        self.graph = graph;
        self
    }

    pub fn with_kmeans_max_iters(mut self, kmeans_max_iters: u32) -> Self {
        self.kmeans_max_iters = kmeans_max_iters;
        self
    }
}

/// Reject the metrics this crate cannot answer correctly.
///
/// `Hamming` does not apply to the Float32 vectors the format stores. `Dot` is
/// refused for a subtler reason: Lance spells dot distance as `1 - dot`, which
/// goes negative for any pair whose inner product exceeds one - the ordinary
/// case for the unnormalised vectors `Dot` exists to serve. `RobustPrune` keeps
/// a candidate when `alpha * d(selected, c) > d(point, c)`, and multiplying a
/// negative left-hand side by `alpha > 1` *lowers* it, so the pruning slack
/// tightens the diversity rule instead of relaxing it and the second pass drops
/// a strict superset of what the first pass drops. The graph comes out sparser
/// than an `alpha = 1` build, silently. Until that is reworked and measured,
/// refusing beats shipping a metric that quietly builds a worse index.
pub fn supported_distance_type(distance_type: DistanceType) -> Result<()> {
    match distance_type {
        DistanceType::L2 | DistanceType::Cosine => Ok(()),
        DistanceType::Hamming => Err(Error::not_supported(
            "Vamana stores Float32 vectors, which Hamming distance does not apply to".to_string(),
        )),
        DistanceType::Dot => Err(Error::not_supported(
            "Vamana does not support dot distance yet: Lance's dot distance is `1 - dot`, which \
             is negative for vectors of norm above one, and a negative distance makes the \
             pruning slack tighten the diversity rule instead of relaxing it"
                .to_string(),
        )),
    }
}

/// The distance type the IVF router works in.
///
/// Cosine is not one of them: `IvfModel::find_partitions` reaches k-means code
/// that handles only L2 and dot and **panics** on anything else. Lance's own
/// index sidesteps this by normalising and routing by L2, and so do we - which
/// is why a cosine build normalises the vectors it stores.
pub fn routing_distance_type(distance_type: DistanceType) -> DistanceType {
    match distance_type {
        DistanceType::Cosine => DistanceType::L2,
        other => other,
    }
}

/// Build a Vamana index over every live fragment of `dataset` and commit it.
pub async fn create_index(
    dataset: &mut Dataset,
    index_name: &str,
    params: &IndexParams,
) -> Result<()> {
    let fragments = live_fragments(dataset);
    let segment = build_index_segment(dataset, params, &fragments).await?;
    dataset
        .commit_existing_index_segments(index_name, &params.column, vec![segment])
        .await
}

pub fn live_fragments(dataset: &Dataset) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect()
}

/// Build a segment over `fragments` and describe it, ready to commit.
///
/// Separate from [`create_index`] because a segment is the unit of maintenance:
/// coverage has to be chosen by the caller, and a segment naming a subset of the
/// fragments is how new data is indexed without rewriting what is already there.
pub async fn build_index_segment(
    dataset: &Dataset,
    params: &IndexParams,
    fragments: &[u32],
) -> Result<IndexSegment> {
    let field = dataset.schema().field(&params.column).ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{}' does not exist in the dataset",
            params.column
        ))
    })?;
    let field_id = field.id;
    let dataset_version = dataset.manifest.version;

    let uuid = Uuid::new_v4();
    let dir = dataset.indices_dir().join(uuid.to_string());
    build_segment(dataset, params, &dir, fragments).await?;

    let details = prost_types::Any {
        type_url: INDEX_DETAILS_TYPE_URL.to_string(),
        value: Vec::new(),
    };
    Ok(IndexSegment::new(
        uuid,
        fragments.to_vec(),
        [field_id],
        Arc::new(details),
        INDEX_VERSION,
        dataset_version,
    ))
}

/// Build one segment into `dir` without committing it.
///
/// Only the rows of `fragments` are indexed. That has to be the caller's choice
/// rather than "everything": a segment's committed coverage is what Lance trusts
/// it to hold, and a segment naming two fragments while physically holding the
/// whole dataset would put every other row into two segments at once.
pub async fn build_segment(
    dataset: &Dataset,
    params: &IndexParams,
    dir: &Path,
    fragments: &[u32],
) -> Result<SegmentManifest> {
    if dataset.manifest().uses_stable_row_ids() {
        // The delete list of stage C is derived from deletion vectors, which are
        // always in address space. Applying it to logical ids would filter out
        // live rows and return deleted ones, silently - so the mode is refused
        // here rather than discovered later.
        return Err(Error::not_supported(
            "Vamana requires a dataset with address-style row ids; \
             this dataset was created with stable row ids enabled"
                .to_string(),
        ));
    }
    if params.num_partitions == 0 {
        return Err(Error::invalid_input(
            "Vamana num_partitions must be greater than zero".to_string(),
        ));
    }
    supported_distance_type(params.distance_type)?;
    if fragments.is_empty() {
        return Err(Error::invalid_input(
            "Vamana cannot build a segment over no fragments".to_string(),
        ));
    }

    let (row_ids, vectors) = read_vectors(dataset, &params.column, fragments).await?;
    let dimension = u32::try_from(vectors.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "column '{}' has a negative vector dimension {}",
            params.column,
            vectors.value_length()
        ))
    })?;
    // Cosine is routed and stored as L2 over unit vectors, exactly as Lance does
    // it. Cosine distance is scale invariant, so the stored answer is unchanged.
    let vectors = if params.distance_type == DistanceType::Cosine {
        normalize_fsl(&vectors)?
    } else {
        vectors
    };

    let mut rng = SmallRng::seed_from_u64(params.graph.seed);
    let ivf = train_router(&vectors, params, &mut rng)?;
    let assignment = assign(&ivf, &vectors, params)?;

    let metadata = IndexMetadata {
        format_version: FORMAT_VERSION,
        max_degree: params.graph.max_degree,
        alpha: params.graph.alpha,
        dimension,
        distance_type: params.distance_type,
        row_id_mode: RowIdMode::Address,
    };
    let mut writer = SegmentWriter::new(
        dataset.object_store(None).await?,
        dir.clone(),
        metadata,
        ivf,
    );

    let comparisons = Comparisons::default();
    for (partition_id, members) in group_by_partition(&assignment, params.num_partitions)
        .into_iter()
        .enumerate()
    {
        if members.is_empty() {
            continue;
        }
        let (partition, medoid) = build_one(&members, &row_ids, &vectors, params, &comparisons)?;
        writer
            .write_partition(partition_id as u32, medoid, &partition)
            .await?;
    }
    writer.finish().await
}

/// Read the vector column and the row id of every row that has a vector.
///
/// Rows whose vector is null are dropped: they have nothing to index, and Lance's
/// own vector indices skip them too. The index therefore covers a subset of the
/// dataset's rows, which is exactly what `fragment_bitmap` already allows for.
async fn read_vectors(
    dataset: &Dataset,
    column: &str,
    fragments: &[u32],
) -> Result<(Vec<u64>, FixedSizeListArray)> {
    let selected = fragments
        .iter()
        .map(|id| {
            dataset
                .get_fragment(*id as usize)
                .map(|fragment| fragment.metadata().clone())
                .ok_or_else(|| {
                    Error::invalid_input(format!("the dataset has no fragment {id} to index"))
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    scanner.with_row_id();
    scanner.with_fragments(selected);
    let batches = scanner
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let schema = batches
        .first()
        .ok_or_else(|| Error::invalid_input("the dataset has no rows to index".to_string()))?
        .schema();
    let batch = concat_batches(&schema, batches.iter())?;

    let row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| {
            Error::internal("a scan with row ids returned no row id column".to_string())
        })?
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| Error::internal("the row id column is not UInt64".to_string()))?
        .values()
        .to_vec();
    let vectors = batch
        .column_by_name(column)
        .ok_or_else(|| Error::invalid_input(format!("column '{column}' does not exist")))?;
    let vectors = vectors.as_fixed_size_list_opt().ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{column}' has type {}, expected a fixed size list of Float32",
            vectors.data_type()
        ))
    })?;
    if vectors.value_type() != arrow_schema::DataType::Float32 {
        return Err(Error::not_supported(format!(
            "column '{column}' holds {} vectors; Vamana indexes Float32 only",
            vectors.value_type()
        )));
    }

    let live = (0..vectors.len() as u32)
        .filter(|row| vectors.is_valid(*row as usize))
        .collect::<Vec<_>>();
    if live.is_empty() {
        return Err(Error::invalid_input(format!(
            "column '{column}' has no non-null vectors to index"
        )));
    }
    if live.len() == vectors.len() {
        return Ok((row_ids, vectors.clone()));
    }
    let kept_row_ids = live.iter().map(|row| row_ids[*row as usize]).collect();
    let kept = gather(vectors, &live)?;
    Ok((kept_row_ids, kept))
}

fn train_router(
    vectors: &FixedSizeListArray,
    params: &IndexParams,
    rng: &mut SmallRng,
) -> Result<IvfModel> {
    let k = params.num_partitions as usize;
    if vectors.len() < k {
        return Err(Error::invalid_input(format!(
            "Vamana cannot train {k} IVF partitions over {} vectors; use fewer partitions",
            vectors.len()
        )));
    }

    // The sample is drawn at random rather than off the front. Lance's own
    // `train_kmeans` slices a prefix, and dataset order is rarely unrelated to
    // the vectors. Seeded from the same seed as the graph build, so that a whole
    // build is reproducible - the alternative measures the dice, not the change.
    let sample_size = params.kmeans_sample_rate.saturating_mul(k);
    let training = if vectors.len() > sample_size {
        let picked = rand::seq::index::sample(rng, vectors.len(), sample_size).into_vec();
        gather(
            vectors,
            &picked.iter().map(|row| *row as u32).collect::<Vec<_>>(),
        )?
    } else {
        vectors.clone()
    };

    // Lance's own k-means seeds its random init from the OS - `SmallRng::from_os_rng`,
    // with its own `TODO: use seed for Rng` beside it - so leaving the init to it
    // makes a build unreproducible, and an A/B over two such builds measures the
    // dice. Handing k sampled rows in as the starting centroids uses the public
    // `Incremental` init and puts the build back under one seed. Hierarchical
    // clustering is switched off for the same reason: above k = 256 it takes over
    // the training and reproducibility would silently stop holding.
    //
    // One hole remains and is not ours to close: whenever an iteration leaves a
    // cluster empty, Lance splits it using an RNG it seeds from the OS as well.
    // So a build is reproducible while every centroid keeps at least one member,
    // which is the normal case but not a guarantee - Lance itself warns about
    // the data shapes that break it. An A/B at high partition counts should
    // check that the trained centroids match before trusting the comparison.
    let init = gather(
        &training,
        &rand::seq::index::sample(rng, training.len(), k)
            .into_iter()
            .map(|row| row as u32)
            .collect::<Vec<_>>(),
    )?;
    let kmeans_params = KMeansParams::new(
        Some(Arc::new(init)),
        params.kmeans_max_iters,
        1,
        routing_distance_type(params.distance_type),
    )
    .with_hierarchical_k(1);
    let kmeans = KMeans::new_with_params(&training, k, &kmeans_params)?;
    let centroids =
        FixedSizeListArray::try_new_from_values(kmeans.centroids, vectors.value_length())?;
    // `IvfModel::new` leaves `offsets` and `lengths` empty, which is what the
    // segment manifest requires: partition sizes live in its own table, and a
    // second copy of them would be a second thing to disagree with.
    Ok(IvfModel::new(centroids, Some(kmeans.loss)))
}

fn assign(ivf: &IvfModel, vectors: &FixedSizeListArray, params: &IndexParams) -> Result<Vec<u32>> {
    let centroids = ivf
        .centroids
        .as_ref()
        .ok_or_else(|| Error::internal("the trained router has no centroids".to_string()))?;
    let (partitions, _) = compute_partitions_arrow_array(
        centroids,
        vectors,
        routing_distance_type(params.distance_type),
    )?;
    partitions
        .into_iter()
        .enumerate()
        .map(|(row, partition)| {
            partition.ok_or_else(|| {
                Error::invalid_input(format!(
                    "Vamana could not assign row {row} to a partition; \
                     the vector is most likely not finite"
                ))
            })
        })
        .collect()
}

fn group_by_partition(assignment: &[u32], num_partitions: u32) -> Vec<Vec<u32>> {
    let mut members = vec![Vec::new(); num_partitions as usize];
    for (row, partition) in assignment.iter().enumerate() {
        members[*partition as usize].push(row as u32);
    }
    members
}

/// Build the graph of one partition over the rows assigned to it.
fn build_one(
    members: &[u32],
    row_ids: &[u64],
    vectors: &FixedSizeListArray,
    params: &IndexParams,
    comparisons: &Comparisons,
) -> Result<(Partition, u32)> {
    let taken = gather(vectors, members)?;
    let member_row_ids = members
        .iter()
        .map(|row| row_ids[*row as usize])
        .collect::<Vec<_>>();

    let store = flat_storage(&member_row_ids, &taken, params.distance_type)?;
    let built = build_partition(&store, &params.graph, comparisons)?;
    Ok((Partition::try_new(built.graph, taken)?, built.medoid))
}

fn gather(vectors: &FixedSizeListArray, rows: &[u32]) -> Result<FixedSizeListArray> {
    let taken = take(vectors, &UInt32Array::from(rows.to_vec()), None)?;
    Ok(taken.as_fixed_size_list().clone())
}
