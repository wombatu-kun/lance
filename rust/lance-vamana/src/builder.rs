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

/// The `type_url` of the details blob that travels with a committed segment.
///
/// Deliberately ours and deliberately unresolvable by Lance. A url Lance can
/// resolve - `VectorIndexDetails` - puts the segment under a version ceiling
/// this crate does not control, and a segment above that ceiling disappears
/// *silently* when the dataset is reopened. An unresolvable one is kept as is.
/// The payload is empty because the segment's own `index.idx` is the only
/// source of truth about its contents.
///
/// "Kept as is" rests on one upstream line: `retain_supported_indices` resolves
/// an unknown url to a maximum supported version of `i32::MAX`, under a comment
/// reading "If we don't know how to read the index, it isn't supported". The
/// fail-open is what keeps this crate's segments visible to their own driver -
/// and if it is ever tightened, `load_indices` will drop the segment with a
/// warning, `VamanaIndex::open` will report that no such index exists, and a
/// rebuild will add a *second* segment beside the invisible first rather than
/// replacing it.
pub const INDEX_DETAILS_TYPE_URL: &str = "type.googleapis.com/lance.vamana.VamanaIndexDetails";

/// How to build one Vamana index segment.
#[derive(Debug, Clone)]
pub struct IndexParams {
    /// Vector column to index. Must be `FixedSizeList<Float32, dim>`.
    pub column: String,
    /// Number of IVF partitions, i.e. how many k-means centroids to train.
    ///
    /// This is also a cost the *dataset* carries, not only the index. Every
    /// non-empty partition is its own file, and Lance records one `IndexFile`
    /// entry per file of a committed index in the manifest - which is then
    /// re-serialised into every manifest written afterwards. At 4096 partitions
    /// that is 4097 entries paid for by each later append, delete or update and
    /// by every `Dataset::open`. Lance's own IVF indices are one or two files, so
    /// nothing upstream is sized for a per-partition list.
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

    pub fn with_kmeans_sample_rate(mut self, kmeans_sample_rate: usize) -> Self {
        self.kmeans_sample_rate = kmeans_sample_rate;
        self
    }
}

/// What building a segment cost.
///
/// The counterpart of [`crate::query::QueryResult::comparisons`]. A graph is a
/// trade between what a build pays and what a query pays, so a change that
/// halves one by tripling the other is not an improvement - and the only way to
/// see that is for both numbers to leave the crate. This one is returned rather
/// than logged for the same reason the query's is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BuildStats {
    /// Distance computations across every partition's graph construction.
    ///
    /// Routing is not in here: assignment measures every vector against every
    /// centroid inside Lance's own k-means, which reports nothing.
    pub comparisons: u64,
    /// Vectors indexed, which is rows of the covered fragments minus those whose
    /// vector is null.
    pub vectors: usize,
    /// Partitions that came out non-empty and were therefore written.
    pub partitions: usize,
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
) -> Result<BuildStats> {
    let fragments = live_fragments(dataset);
    let (segment, stats) = build_index_segment(dataset, params, &fragments).await?;
    dataset
        .commit_existing_index_segments(index_name, &params.column, vec![segment])
        .await?;
    Ok(stats)
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
///
/// Commit it promptly. A segment records the dataset version it was built at,
/// and `prune_stale_segment_coverage` runs over any segment older than the
/// manifest it is committed against: it checks out that version - which fails
/// outright once `cleanup_old_versions` has removed it - and silently drops from
/// the coverage any fragment whose data file has been rewritten since. The
/// commit then succeeds with a narrower bitmap than the segment was built over,
/// and [`crate::query::VamanaIndex::open`] refuses the result, because that is
/// exactly the shape of an index whose data moved underneath it.
pub async fn build_index_segment(
    dataset: &Dataset,
    params: &IndexParams,
    fragments: &[u32],
) -> Result<(IndexSegment, BuildStats)> {
    // Refused before the graph is built rather than discovered on the commit
    // that follows it: Lance would open this index while committing, and cannot.
    if writer_predates_bitmap_recalculation(dataset) {
        return Err(Error::not_supported(format!(
            "Vamana cannot index a dataset whose manifest was written by {}: Lance recalculates \
             every index's fragment coverage on the next commit, and it does that by opening the \
             index, which fails for this format. Commit any change with a current Lance build \
             first - an append or a compaction rewrites the manifest with a current writer \
             version - and then build the index",
            dataset.manifest().writer_version.as_ref().map_or(
                "no recorded writer".to_string(),
                |version| format!("{} {}", version.library, version.version)
            )
        )));
    }

    let field = dataset.schema().field(&params.column).ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{}' does not exist in the dataset",
            params.column
        ))
    })?;
    // `Schema::field` resolves a dotted path, so a nested leaf gets this far and
    // then fails three lines into the build with "column does not exist":
    // `Scanner::project` on a nested leaf yields a column named after its
    // top-level parent, which `read_vectors` looks for by the full path. Refused
    // here, where the reason can be stated.
    if !dataset
        .schema()
        .fields
        .iter()
        .any(|top_level| top_level.name == params.column)
    {
        return Err(Error::not_supported(format!(
            "column '{}' is nested; Vamana indexes top-level vector columns only",
            params.column
        )));
    }
    let field_id = field.id;
    let dataset_version = dataset.manifest.version;

    let uuid = Uuid::new_v4();
    let dir = dataset.indices_dir().join(uuid.to_string());
    let (_, stats) = build_segment(dataset, params, &dir, fragments).await?;

    let details = prost_types::Any {
        type_url: INDEX_DETAILS_TYPE_URL.to_string(),
        value: Vec::new(),
    };
    Ok((
        IndexSegment::new(
            uuid,
            fragments.to_vec(),
            [field_id],
            Arc::new(details),
            // The manifest records the version of the files it points at, and
            // there is only one such number. A second one, counted separately
            // and checked nowhere, would be a version this crate believed in
            // and nothing enforced.
            FORMAT_VERSION as i32,
            dataset_version,
        ),
        stats,
    ))
}

/// Whether Lance will recompute every index's fragment coverage on the next
/// commit of this dataset.
///
/// It does that by *opening* each index - `migrate_indices` ->
/// `open_generic_index`, propagated with `?` and no fallback - so for this
/// crate's segments the commit fails outright. The condition mirrors Lance's own
/// `must_recalculate_fragment_bitmap`: a manifest with no recorded writer, or
/// one written by a Lance older than 0.8.15, whose fragment bitmaps could be
/// corrupt. A manifest written by any other library is left alone by Lance and
/// so is left alone here.
///
/// The version compared is the one on the manifest the commit *starts from*, so
/// a single commit by a current Lance build clears it permanently.
fn writer_predates_bitmap_recalculation(dataset: &Dataset) -> bool {
    match dataset.manifest().writer_version.as_ref() {
        None => true,
        Some(version) if version.library != "lance" => false,
        // Unparseable counts as old, which is what Lance concludes too.
        Some(version) => version
            .lance_lib_version()
            .is_none_or(|parsed| (parsed.major, parsed.minor, parsed.patch) < (0, 8, 15)),
    }
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
) -> Result<(SegmentManifest, BuildStats)> {
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
    // Zero draws an empty training set, and sampling k centroids from nothing
    // panics inside `rand` rather than returning an error.
    if params.kmeans_sample_rate == 0 {
        return Err(Error::invalid_input(
            "Vamana kmeans_sample_rate must be greater than zero; it is how many vectors are \
             sampled per centroid to train the router"
                .to_string(),
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
    let assignment = assign(&ivf, &vectors, &row_ids, params)?;

    let metadata = IndexMetadata {
        format_version: FORMAT_VERSION,
        max_degree: params.graph.max_degree,
        alpha: params.graph.alpha,
        dimension,
        distance_type: params.distance_type,
        row_id_mode: RowIdMode::Address,
        fragments: fragments.to_vec(),
    };
    let mut writer = SegmentWriter::new(
        dataset.object_store(None).await?,
        dir.clone(),
        metadata,
        ivf,
    );

    let comparisons = Comparisons::default();
    let mut stats = BuildStats {
        vectors: vectors.len(),
        ..Default::default()
    };
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
        stats.partitions += 1;
    }
    stats.comparisons = comparisons.get();
    Ok((writer.finish().await?, stats))
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

    let num_rows = u32::try_from(vectors.len()).map_err(|_| {
        Error::invalid_input(format!(
            "column '{column}' holds {} rows, more than one segment can address",
            vectors.len()
        ))
    })?;
    let live = (0..num_rows)
        .filter(|row| vectors.is_valid(*row as usize))
        .collect::<Vec<_>>();
    if live.is_empty() {
        return Err(Error::invalid_input(format!(
            "column '{column}' has no non-null vectors to index"
        )));
    }
    if live.len() == vectors.len() {
        return reject_item_nulls(column, vectors.clone()).map(|vectors| (row_ids, vectors));
    }
    let kept_row_ids = live.iter().map(|row| row_ids[*row as usize]).collect();
    let kept = reject_item_nulls(column, gather(vectors, &live)?)?;
    Ok((kept_row_ids, kept))
}

/// Refuse vectors with a null *inside* them, as opposed to a null vector.
///
/// A list-level null is a row with nothing to index and is skipped above. A null
/// coordinate is a row whose vector is partly unknown, and `Partition::try_new`
/// refuses it - but only under L2. A cosine build normalises first, and
/// `normalize_fsl` rebuilds the child through `from_iter_values`, which keeps the
/// list-level nulls and drops the item-level ones. The same column would then be
/// an error under one metric and silently indexed with whatever byte sat under
/// the null - usually `0.0` - as a coordinate under the other.
fn reject_item_nulls(column: &str, vectors: FixedSizeListArray) -> Result<FixedSizeListArray> {
    if vectors.values().null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "column '{column}' has nulls inside its vectors; a partly null vector has no \
             position to index and the byte under a null is not a coordinate"
        )));
    }
    Ok(vectors)
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
    // Two holes remain and neither is ours to close. Whenever an iteration
    // leaves a cluster empty, Lance splits it using an RNG it seeds from the OS
    // as well, so a build is reproducible while every centroid keeps at least
    // one member - the normal case, but not a guarantee, and Lance itself warns
    // about the data shapes that break it.
    //
    // The second is size-dependent and therefore easy to miss in a small test:
    // every k-means iteration calls `SimpleIndex::may_train_index`, which
    // switches assignment from exhaustive to an *approximate* HNSW search over
    // the centroids once the flattened centroid array reaches a million values -
    // `num_partitions * dimension`, so 4096 partitions of 256 dimensions is
    // exactly at it - or at any size when `LANCE_USE_HNSW_SPEEDUP_INDEXING` is
    // set. That HNSW is built in parallel into shared state, so its answers
    // depend on thread interleaving.
    //
    // Both bite at the scale an A/B is worth running at, so an A/B at high
    // partition counts should check that the trained centroids match before
    // trusting anything downstream of them.
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

fn assign(
    ivf: &IvfModel,
    vectors: &FixedSizeListArray,
    row_ids: &[u64],
    params: &IndexParams,
) -> Result<Vec<u32>> {
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
                // Named by row id, not by position: the position is into the
                // array left after null vectors were dropped, which nothing the
                // caller has can be matched against.
                Error::invalid_input(format!(
                    "Vamana could not assign row {} to a partition; \
                     the vector is most likely not finite",
                    row_ids.get(row).copied().unwrap_or_default()
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
