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
use arrow_schema::DataType;
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::kmeans::{KMeans, KMeansParams, compute_partitions_arrow_array};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;
use lance_table::format::WriterVersion;
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

/// Most vectors per centroid the router's k-means will actually train on.
///
/// Lance's own ceiling, applied here so that a caller learns about it: at
/// `data.len() >= k * 512` its k-means slices the training set down with
/// `data.slice(0, k * 512)`. A prefix, not a sample - so above this the
/// randomness of our own sampling would be spent and the router would be trained
/// on the front of the dataset.
pub const MAX_KMEANS_SAMPLE_RATE: usize = 512;

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
    /// At most [`MAX_KMEANS_SAMPLE_RATE`], and refused above it rather than
    /// clamped: Lance re-slices the training set to `512 * k` before it starts,
    /// and it takes the *front* of it, so a larger rate would quietly stop being
    /// a random sample of the dataset.
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
///
/// The fragment list is taken once, before a build that can run for minutes, and
/// a compaction landing in between is not a commit conflict: `(CreateIndex,
/// Rewrite)` do not conflict in Lance's table, so the index commits over
/// fragments the dataset no longer has. That costs coverage rather than
/// correctness - [`crate::query::VamanaIndex::open`] narrows the index to the
/// fragments that survived, and the rows of the rest are the caller's to scan
/// until the index is rebuilt.
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
/// the coverage every fragment that has since been rewritten or has gone
/// altogether. The commit then succeeds with a narrower bitmap than the segment
/// was built over, and [`crate::query::VamanaIndex::open`] takes those two apart:
/// a fragment that is gone narrows the index and is logged, while one that is
/// still there and was rewritten is refused, because that is the shape of an
/// index whose data moved underneath it.
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
    predates_bitmap_recalculation(dataset.manifest().writer_version.as_ref())
}

/// The comparison itself, over the value rather than over a dataset, because a
/// manifest carrying a prerelease writer is not something this crate's fixtures
/// can produce and the ordering is exactly where this can go wrong.
///
/// `semver::Version` rather than the `(major, minor, patch)` triple: semver
/// orders a prerelease *below* the release it leads to, so `0.8.15-beta.1` is
/// old to Lance and would be new to a triple comparison - and this crate would
/// then build an index over a manifest whose next commit recalculates every
/// fragment bitmap by opening it.
fn predates_bitmap_recalculation(version: Option<&WriterVersion>) -> bool {
    match version {
        None => true,
        Some(version) if version.library != "lance" => false,
        // Unparseable counts as old, which is what Lance concludes too.
        Some(version) => version
            .lance_lib_version()
            .is_none_or(|parsed| parsed < semver::Version::new(0, 8, 15)),
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
    // Zero iterations leaves the centroids exactly where the initialisation put
    // them - k rows drawn at random - so the router would route by a sample
    // rather than by a clustering, and nothing downstream would look wrong.
    if params.kmeans_max_iters == 0 {
        return Err(Error::invalid_input(
            "Vamana kmeans_max_iters must be greater than zero; at zero the router keeps the \
             centroids it was initialised with"
                .to_string(),
        ));
    }
    supported_distance_type(params.distance_type)?;
    if params.kmeans_sample_rate > MAX_KMEANS_SAMPLE_RATE {
        return Err(Error::invalid_input(format!(
            "Vamana kmeans_sample_rate must be at most {MAX_KMEANS_SAMPLE_RATE}, got {}; above \
             that Lance re-slices the training set down to {MAX_KMEANS_SAMPLE_RATE} vectors per \
             centroid, and it takes a *prefix* - so a larger rate would not train on more data, \
             it would train on the front of the dataset",
            params.kmeans_sample_rate
        )));
    }
    if fragments.is_empty() {
        return Err(Error::invalid_input(
            "Vamana cannot build a segment over no fragments".to_string(),
        ));
    }
    // A fragment named twice would be read twice and indexed twice, and nothing
    // downstream could tell: the coverage bitmap collapses the duplicate, so the
    // segment would look ordinary while holding every one of that fragment's
    // rows in two partitions.
    let mut seen = fragments.to_vec();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != fragments.len() {
        return Err(Error::invalid_input(format!(
            "Vamana was asked to index {} fragments but only {} of them are distinct; a fragment \
             named twice would have its rows indexed twice",
            fragments.len(),
            seen.len()
        )));
    }
    // Asked of the schema, not of the data. `read_vectors` checks the same thing
    // on the array it decoded, which is the last line of defence and far too
    // late to be the first one: a `FixedSizeList<Float64>` column of five
    // million rows would be read into memory in full and only then refused.
    let field = dataset.schema().field(&params.column).ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{}' does not exist in the dataset",
            params.column
        ))
    })?;
    match field.data_type() {
        // Width included, because zero is a type the schema can hold and no
        // layer below is ready for it: k-means divides by the dimension and
        // `l2_distance_batch` takes a chunk size of zero, both of which end the
        // process rather than the call.
        DataType::FixedSizeList(item, width)
            if item.data_type() == &DataType::Float32 && width > 0 => {}
        other => {
            return Err(Error::not_supported(format!(
                "column '{}' has type {other}; Vamana indexes FixedSizeList<Float32> of a \
                 positive width only",
                params.column
            )));
        }
    }

    let (row_ids, vectors) = read_vectors(dataset, &params.column, fragments).await?;
    let dimension = u32::try_from(vectors.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "column '{}' has a negative vector dimension {}",
            params.column,
            vectors.value_length()
        ))
    })?;

    // Everything from here to the last partition is arithmetic, and it runs on
    // the CPU pool rather than here. A build is minutes of it with no await to
    // yield at, and the runtime it would otherwise hold is the one the scan
    // scheduler runs its io loop on - so on a runtime with few workers, and on
    // the single-threaded one an ordinary `#[tokio::test]` gives, reads across
    // the whole process would stop for the duration. The query side already
    // moves a walk off for the same reason, and a walk is milliseconds.
    //
    // The k-means inside `train_router` parallelises with rayon, which has a
    // thread pool of its own, so a pool worker parked on it is waiting on
    // something outside the pool and cannot starve it - the deadlock `spawn_cpu`
    // warns about needs a closure waiting on the pool it is running in.
    let row_ids = Arc::new(row_ids);
    let params = Arc::new(params.clone());
    let (vectors, ivf, assignment) = {
        let row_ids = row_ids.clone();
        let params = params.clone();
        // Moved in rather than cloned, so that the normalisation below has the
        // only reference to the buffer and can work in place.
        spawn_cpu(move || {
            // Cosine is routed and stored as L2 over unit vectors, exactly as
            // Lance does it. Cosine distance is scale invariant, so the stored
            // answer is unchanged.
            //
            // `_owned` and not `normalize_fsl`, which allocates a second copy of
            // the whole column: this one hands the values to `into_builder`,
            // which writes through them when the buffer is unshared.
            let vectors = if params.distance_type == DistanceType::Cosine {
                normalize_fsl_owned(vectors)?
            } else {
                vectors
            };
            let mut rng = SmallRng::seed_from_u64(params.graph.seed);
            let ivf = train_router(&vectors, &params, &mut rng)?;
            let assignment = assign(&ivf, &vectors, &row_ids, &params)?;
            Ok::<_, Error>((vectors, ivf, assignment))
        })
        .await?
    };

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

    let members_by_partition = group_by_partition(&assignment, params.num_partitions);
    let stats =
        write_partitions(&mut writer, members_by_partition, row_ids, vectors, params).await?;
    Ok((writer.finish().await?, stats))
}

/// Build the graph of every occupied partition and write it, in id order.
///
/// Its own function so that the empty ones can be tested. A partition with
/// nothing assigned to it is written no file and given no row in the segment
/// table, and skipping it is what makes the id a partition is written under the
/// id of its *centroid* rather than its position among the ones that were
/// written. The two numbers agree in every fixture whose partitions are all
/// occupied, and whether k-means leaves one empty is decided by an RNG Lance
/// seeds from the OS - so a build cannot be asked for an empty partition on
/// purpose, and nothing that goes through one can tell the two apart.
async fn write_partitions(
    writer: &mut SegmentWriter,
    members_by_partition: Vec<Vec<u32>>,
    row_ids: Arc<Vec<u64>>,
    vectors: FixedSizeListArray,
    params: Arc<IndexParams>,
) -> Result<BuildStats> {
    let mut stats = BuildStats {
        vectors: vectors.len(),
        ..Default::default()
    };
    for (partition_id, members) in members_by_partition.into_iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let built = {
            let vectors = vectors.clone();
            let row_ids = row_ids.clone();
            let params = params.clone();
            spawn_cpu(move || build_one(&members, &row_ids, &vectors, &params)).await?
        };
        writer
            .write_partition(partition_id as u32, built.medoid, &built.partition)
            .await?;
        stats.partitions += 1;
        stats.comparisons = stats.comparisons.saturating_add(built.comparisons);
    }
    Ok(stats)
}

/// Read the vector column and the row id of every row that has a vector.
///
/// Rows whose vector is null are dropped: they have nothing to index, and Lance's
/// own vector indices skip them too. The index therefore holds a subset of the
/// rows of the fragments it covers, and nothing records which subset - coverage
/// is per fragment, never per row. That is the same position Lance's own vector
/// indices are in, and it is why a caller cannot treat "the index covers this
/// fragment" as "every row of it is in the index".
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
    // The concatenation copied every value out of them, so from here they are a
    // second copy of the vector column and nothing reads it. A build already
    // holds the whole column twice at this line; holding it twice for the rest
    // of the function is what this drop is about.
    drop(batches);

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
    if vectors.value_type() != DataType::Float32 {
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

/// One partition's graph, ready to write, and what building it cost.
struct BuiltOne {
    partition: Partition,
    medoid: u32,
    comparisons: u64,
}

/// Build the graph of one partition over the rows assigned to it.
///
/// The comparison count is returned rather than accumulated into a counter the
/// caller holds: [`Comparisons`] is a `Cell`, deliberately, because it is
/// written once per candidate in the innermost loop of the build - and a `Cell`
/// cannot cross the thread boundary this runs behind.
fn build_one(
    members: &[u32],
    row_ids: &[u64],
    vectors: &FixedSizeListArray,
    params: &IndexParams,
) -> Result<BuiltOne> {
    let taken = gather(vectors, members)?;
    let member_row_ids = members
        .iter()
        .map(|row| row_ids[*row as usize])
        .collect::<Vec<_>>();

    let comparisons = Comparisons::default();
    let store = flat_storage(&member_row_ids, &taken, params.distance_type)?;
    let built = build_partition(&store, &params.graph, &comparisons)?;
    Ok(BuiltOne {
        partition: Partition::try_new(built.graph, taken)?,
        medoid: built.medoid,
        comparisons: comparisons.get(),
    })
}

fn gather(vectors: &FixedSizeListArray, rows: &[u32]) -> Result<FixedSizeListArray> {
    let taken = take(vectors, &UInt32Array::from(rows.to_vec()), None)?;
    Ok(taken.as_fixed_size_list().clone())
}

#[cfg(test)]
mod tests {
    use arrow_array::Float32Array;
    use lance_io::object_store::ObjectStore;

    use super::*;
    use crate::format::partition_file_name;

    fn written_by(library: &str, version: &str, prerelease: Option<&str>) -> WriterVersion {
        WriterVersion {
            library: library.to_string(),
            version: version.to_string(),
            prerelease: prerelease.map(str::to_string),
            build_metadata: None,
        }
    }

    /// The version gate is the one thing standing between a build and a commit
    /// that fails inside Lance, and the case it can get wrong is the one no
    /// fixture in this crate can produce: semver puts a prerelease *below* the
    /// release it leads to, so `0.8.15-beta.1` is old to Lance and would be new
    /// to a comparison of `(major, minor, patch)`.
    #[test]
    fn a_prerelease_writer_counts_as_older_than_its_release() {
        for (version, prerelease, expected, what) in [
            ("0.8.14", None, true, "older than the fix"),
            ("0.8.15", None, false, "the fix itself"),
            ("0.8.15", Some("beta.1"), true, "a prerelease of the fix"),
            (
                "0.9.0",
                Some("rc.1"),
                false,
                "a prerelease of a later release",
            ),
            ("1.2.3", None, false, "current"),
        ] {
            assert_eq!(
                predates_bitmap_recalculation(Some(&written_by("lance", version, prerelease))),
                expected,
                "{what}"
            );
        }

        assert!(
            predates_bitmap_recalculation(None),
            "a manifest with no recorded writer has to count as old, as it does upstream"
        );
        assert!(
            !predates_bitmap_recalculation(Some(&written_by("something-else", "0.1.0", None))),
            "Lance leaves another library's manifest alone, and so does this"
        );
        assert!(
            predates_bitmap_recalculation(Some(&written_by("lance", "not a version", None))),
            "an unparseable version counts as old, as it does upstream"
        );
    }

    /// A partition whose centroid drew nothing gets no file and no row in the
    /// segment table, and the partitions after it keep their own ids. Writing
    /// them under a running count instead would produce a segment that routes a
    /// query to a centroid and answers it with somebody else's vectors - and
    /// every fixture that goes through k-means would still pass, because the two
    /// numbers only differ once a partition has come out empty.
    #[tokio::test]
    async fn an_empty_partition_does_not_shift_the_ids_after_it() {
        const DIMENSION: i32 = 4;
        const VERTICES: usize = 12;
        const PARTITIONS: u32 = 3;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path()).unwrap();

        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..VERTICES * DIMENSION as usize)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>(),
            ),
            DIMENSION,
        )
        .unwrap();
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..PARTITIONS as usize * DIMENSION as usize)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>(),
            ),
            DIMENSION,
        )
        .unwrap();
        let params = Arc::new(IndexParams::new("vector", PARTITIONS).with_graph_params(
            BuildParams {
                max_degree: 4,
                search_list_size: 8,
                ..Default::default()
            },
        ));
        let metadata = IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: params.graph.max_degree,
            alpha: params.graph.alpha,
            dimension: DIMENSION as u32,
            distance_type: params.distance_type,
            row_id_mode: RowIdMode::Address,
            fragments: vec![0],
        };
        let mut writer =
            SegmentWriter::new(store, path, metadata, IvfModel::new(centroids, Some(0.0)));

        // The first centroid drew nothing, the other two split the rows.
        let stats = write_partitions(
            &mut writer,
            vec![Vec::new(), (0..6).collect(), (6..12).collect()],
            Arc::new((0..VERTICES as u64).collect()),
            vectors,
            params,
        )
        .await
        .unwrap();
        let manifest = writer.finish().await.unwrap();

        assert_eq!(stats.partitions, 2, "two partitions had rows to write");
        assert_eq!(stats.vectors, VERTICES);
        assert_eq!(
            manifest
                .partitions()
                .iter()
                .map(|entry| (entry.partition_id, entry.file.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (1, partition_file_name(1).as_str()),
                (2, partition_file_name(2).as_str())
            ],
            "the occupied partitions were written under the wrong ids"
        );
        assert!(
            manifest.partition(0).is_none(),
            "the empty partition was given a row in the segment table"
        );
    }
}
