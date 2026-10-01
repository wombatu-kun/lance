// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the integration tests.
//!
//! Every integration binary compiles this module whole, so a fixture only one of
//! them needs still lands in the others.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::optimize::{CompactionMetrics, CompactionOptions, compact_files};
use lance::dataset::transaction::{DataOverlayGroup, Operation};
use lance::dataset::{WriteDestination, WriteMode, WriteParams};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_io::utils::CachedFileSize;
use lance_table::format::DataFile;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::codes::CodeSpec;
use lance_vamana::format::{IndexMetadata, VectorSource};
use lance_vamana::io::{
    open_file, read_partition, read_partition_batch, read_segment, scan_scheduler,
};
use lance_vamana::partition::{Partition, PartitionGraph};
use lance_vamana::query::committed_segments;
use lance_vamana::segment::SegmentManifest;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;

/// A graph whose vertices have deliberately unequal degrees.
///
/// Uniform degrees would hide both ends of the layout: nothing would exercise
/// the sentinel padding, and nothing would exercise a saturated vertex.
pub fn sample_graph(max_degree: u32, vertices: usize) -> PartitionGraph {
    let row_ids = (0..vertices as u64).map(|i| i * 3 + 1).collect::<Vec<_>>();
    let adjacency = (0..vertices)
        .map(|local_id| {
            let degree = local_id % (max_degree as usize + 1);
            (0..degree)
                .map(|k| ((local_id + k + 1) % vertices) as u32)
                .collect()
        })
        .collect();
    PartitionGraph::try_new(max_degree, row_ids, adjacency).unwrap()
}

/// Vectors whose every value is distinct, so a vertex read from the wrong offset
/// cannot compare equal to the right one.
pub fn sample_vectors(vertices: usize, dimension: u32) -> FixedSizeListArray {
    let values = (0..vertices * dimension as usize)
        .map(|i| i as f32)
        .collect::<Vec<_>>();
    FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        dimension as i32,
        Arc::new(Float32Array::from(values)),
        None,
    )
    .unwrap()
}

pub fn sample_partition(max_degree: u32, vertices: usize, dimension: u32) -> Partition {
    let graph = sample_graph(max_degree, vertices);
    Partition::try_new(graph, sample_vectors(vertices, dimension)).unwrap()
}

pub const VECTOR_COLUMN: &str = "vec";
pub const VECTOR_DIM: i32 = 16;

/// A dataset with a vector column, spread over several fragments.
///
/// The vectors are uniform noise rather than clustered blobs. That is the wrong
/// data for judging graph *quality* - uniform noise has maximal intrinsic
/// dimension - but it is the right data for judging *routing*: k-means cuts an
/// unclustered cloud into arbitrary cells, so a true neighbour lands outside the
/// nearest cell often enough that a narrow probe is visibly worse than a wide
/// one. Well-separated blobs would let a broken router look perfect.
pub struct DatasetFixture {
    pub fragments: usize,
    pub rows_per_fragment: usize,
    pub stable_row_ids: bool,
    /// Make every n-th vector null, to exercise the skip path.
    pub null_every: Option<usize>,
    pub seed: u64,
    /// The width of every vector. Lance lays a column out full-zip, which is
    /// what a re-score can read out of a data file by offset, only from 256
    /// bytes a value on: 64 and up here, and never at the default.
    pub dimension: i32,
    /// The data file format the dataset is written in; `None` for Lance's
    /// default.
    pub storage_version: Option<LanceFileVersion>,
}

impl Default for DatasetFixture {
    fn default() -> Self {
        Self {
            fragments: 3,
            rows_per_fragment: 512,
            stable_row_ids: false,
            null_every: None,
            seed: 11,
            dimension: VECTOR_DIM,
            storage_version: None,
        }
    }
}

impl DatasetFixture {
    pub fn rows(&self) -> usize {
        self.fragments * self.rows_per_fragment
    }

    /// How many rows carry a vector, and therefore how many the index covers.
    pub fn indexed_rows(&self) -> usize {
        match self.null_every {
            None => self.rows(),
            Some(every) => (0..self.rows()).filter(|row| row % every != 0).count(),
        }
    }

    pub async fn write(&self, uri: &str) -> Dataset {
        self.write_with_mode(uri, WriteMode::Create).await
    }

    /// Add another round of the same rows as fresh fragments.
    pub async fn append(&self, uri: &str) -> Dataset {
        self.write_with_mode(uri, WriteMode::Append).await
    }

    async fn write_with_mode(&self, uri: &str, mode: WriteMode) -> Dataset {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            VECTOR_COLUMN,
            DataType::FixedSizeList(item, self.dimension),
            true,
        )]));

        let mut rng = SmallRng::seed_from_u64(self.seed);
        let pool = (0..self.rows())
            .map(|_| {
                (0..self.dimension)
                    .map(|_| Some(rng.random::<f32>()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..self.rows())
                .map(|row| match self.null_every {
                    Some(every) if row % every == 0 => None,
                    _ => Some(pool[row].clone()),
                })
                .collect::<Vec<_>>(),
            self.dimension,
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                mode,
                max_rows_per_file: self.rows_per_fragment,
                max_rows_per_group: self.rows_per_fragment,
                enable_stable_row_ids: self.stable_row_ids,
                data_storage_version: self.storage_version,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }
}

/// Lance's own exhaustive k-NN over the vector column.
///
/// `use_index(false)` is not optional: the scanner picks a vector index by field
/// id alone, so with one of this crate's segments committed the ordinary path
/// would try to open it as one of Lance's own.
pub async fn brute_force(dataset: &Dataset, query: &[f32], k: usize) -> Vec<u64> {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.nearest(VECTOR_COLUMN, &key, k).unwrap();
    scanner.use_index(false);
    scanner.with_row_id();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

/// Compact the dataset, fragments under a Vamana index included.
///
/// Since upstream #8427 Lance's default planner holds back every fragment an
/// index it has no reader for covers, because it cannot remap that index onto
/// rewritten fragments, so a default compaction of a fully indexed dataset
/// rewrites nothing. A deferred remap rewrites them and records where every row
/// went, and the index follows that record.
pub async fn compact_indexed(dataset: &mut Dataset) -> CompactionMetrics {
    compact_files(
        dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap()
}

/// Every `_rowid` the dataset still has, in scan order.
pub async fn live_row_ids(dataset: &Dataset) -> Vec<u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project::<&str>(&[]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

/// One committed segment as it is on disk: what it says about itself, and every
/// partition it holds.
pub struct ReadSegment {
    pub uuid: uuid::Uuid,
    pub manifest: SegmentManifest,
    pub partitions: HashMap<u32, Partition>,
}

/// Read every committed segment of `index_name` back off disk, in manifest
/// order.
pub async fn read_committed_segments(dataset: &Dataset, index_name: &str) -> Vec<ReadSegment> {
    let indices = committed_segments(dataset, index_name).await.unwrap();
    let store = dataset.object_store(None).await.unwrap();
    let scheduler = scan_scheduler(&store);

    let mut segments = Vec::with_capacity(indices.len());
    for index in indices.iter() {
        let dir = dataset.indices_dir().join(index.uuid.to_string());
        let manifest = read_segment(&scheduler, &dir, None).await.unwrap();
        let mut partitions = HashMap::new();
        for entry in manifest.partitions() {
            let reader = open_file(
                &scheduler,
                &dir.clone().join(entry.file.as_str()),
                None,
                None,
            )
            .await
            .unwrap();
            partitions.insert(
                entry.partition_id,
                read_partition(&reader, entry.num_rows).await.unwrap(),
            );
        }
        segments.push(ReadSegment {
            uuid: index.uuid,
            manifest,
            partitions,
        });
    }
    segments
}

/// Locate the one committed segment of `index_name` and read every partition of
/// it back off disk.
pub async fn read_committed_segment(
    dataset: &Dataset,
    index_name: &str,
) -> (SegmentManifest, HashMap<u32, Partition>) {
    let mut segments = read_committed_segments(dataset, index_name).await;
    assert_eq!(
        segments.len(),
        1,
        "expected exactly one committed segment, found {}",
        segments.len()
    );
    let segment = segments.remove(0);
    (segment.manifest, segment.partitions)
}

/// Vectors of 64 floats, 256 bytes: the narrowest Lance lays out full-zip, and
/// so the narrowest a re-score can read out of a data file by offset - and an
/// index may leave to the dataset.
pub const WIDE_DIM: i32 = 64;

/// A dataset an index may leave its vectors to, small enough to build twice in
/// a test.
pub fn wide_fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 3,
        rows_per_fragment: 200,
        dimension: WIDE_DIM,
        ..Default::default()
    }
}

/// What twins are built with: the codes the stand measures, which also carry
/// no random rotation for the twins to differ by, over a narrow graph.
pub fn twin_params(num_partitions: u32) -> IndexParams {
    IndexParams::new(VECTOR_COLUMN, num_partitions)
        .with_codes(CodeSpec::Scalar { num_bits: 8 })
        .with_graph_params(BuildParams {
            max_degree: 16,
            search_list_size: 32,
            ..Default::default()
        })
}

/// The same rows written twice, under `dir`, each with `params` built over them:
/// first an index that keeps its vectors, then one that leaves them to the
/// dataset. Returned with their URIs, in that order.
pub async fn twins(
    dir: &std::path::Path,
    fixture: &DatasetFixture,
    index_name: &str,
    params: &IndexParams,
) -> Vec<(String, Dataset)> {
    let mut twins = Vec::with_capacity(2);
    for vector_source in [VectorSource::Index, VectorSource::Dataset] {
        let uri = dir
            .join(vector_source.to_string())
            .to_str()
            .unwrap()
            .to_string();
        let mut dataset = fixture.write(&uri).await;
        create_index(
            &mut dataset,
            index_name,
            &params.clone().with_vector_source(vector_source),
        )
        .await
        .unwrap();
        twins.push((uri, dataset));
    }
    twins
}

/// Every committed segment of `index_name` with each partition's batch as its
/// file holds it, which reads an index whether or not it keeps its vectors.
pub async fn read_committed_batches(
    dataset: &Dataset,
    index_name: &str,
) -> Vec<(SegmentManifest, Vec<RecordBatch>)> {
    let indices = committed_segments(dataset, index_name).await.unwrap();
    let store = dataset.object_store(None).await.unwrap();
    let scheduler = scan_scheduler(&store);

    let mut segments = Vec::with_capacity(indices.len());
    for index in indices.iter() {
        let dir = dataset.indices_dir().join(index.uuid.to_string());
        let manifest = read_segment(&scheduler, &dir, None).await.unwrap();
        let mut batches = Vec::with_capacity(manifest.partitions().len());
        for entry in manifest.partitions() {
            let reader = open_file(
                &scheduler,
                &dir.clone().join(entry.file.as_str()),
                None,
                None,
            )
            .await
            .unwrap();
            batches.push(read_partition_batch(&reader, entry.num_rows).await.unwrap());
        }
        segments.push((manifest, batches));
    }
    segments
}

/// Twins hold the same index: segment for segment the same routing, table,
/// graphs and codes, and in every partition file the same columns but the
/// vectors, which only the first keeps.
pub async fn assert_twins_hold_the_same(with: &Dataset, without: &Dataset, index_name: &str) {
    let with = read_committed_batches(with, index_name).await;
    let without = read_committed_batches(without, index_name).await;
    assert_eq!(
        with.len(),
        without.len(),
        "the twins hold different segments"
    );
    for ((with, with_batches), (without, without_batches)) in with.iter().zip(&without) {
        assert_eq!(with.metadata().vector_source, VectorSource::Index);
        assert_eq!(without.metadata().vector_source, VectorSource::Dataset);
        let relabelled = SegmentManifest::try_new(
            IndexMetadata {
                vector_source: VectorSource::Index,
                ..without.metadata().clone()
            },
            without.ivf().clone(),
            without.partitions().to_vec(),
        )
        .unwrap();
        assert_eq!(
            &relabelled, with,
            "the routing, the table or the codes drifted"
        );
        assert_eq!(without_batches.len(), with_batches.len());
        for (without, with) in without_batches.iter().zip(with_batches) {
            let kept = (0..with.num_columns())
                .filter(|column| {
                    with.schema().field(*column).name() != lance_vamana::format::VECTOR_COLUMN
                })
                .collect::<Vec<_>>();
            assert_eq!(
                kept.len(),
                with.num_columns() - 1,
                "no vector column to leave out"
            );
            assert_eq!(
                *without,
                with.project(&kept).unwrap(),
                "a partition drifted"
            );
        }
    }
}

/// Replace one fragment's vectors with an overlay, the way Lance's own overlay
/// tests do: write a file holding the new values for the indexed field alone,
/// then commit `Operation::DataOverlay` naming the offsets it covers.
///
/// `committed_version` is stamped by the commit, not by this caller, so an
/// overlay is newer than every index built before it and older than every index
/// built after it - which is the whole basis of the version gate under test.
pub async fn commit_overlay(
    dataset: Dataset,
    fragment_id: u64,
    offsets: &[u32],
    name: &str,
) -> Dataset {
    let field = dataset.schema().field(VECTOR_COLUMN).unwrap();
    let DataType::FixedSizeList(_, width) = field.data_type() else {
        panic!("the vector column is {}", field.data_type());
    };
    // A constant vector, so an answer ranked on the pre-overlay values is
    // distinguishable from one ranked on these.
    let replacement = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        offsets
            .iter()
            .map(|_| Some(vec![Some(9.0f32); width as usize]))
            .collect::<Vec<_>>(),
        width,
    );
    commit_overlay_of(dataset, fragment_id, offsets, name, replacement).await
}

/// [`commit_overlay`] with `replacement` as the new values, one for each of
/// `offsets`.
pub async fn commit_overlay_of(
    dataset: Dataset,
    fragment_id: u64,
    offsets: &[u32],
    name: &str,
    replacement: FixedSizeListArray,
) -> Dataset {
    assert_eq!(replacement.len(), offsets.len());
    let read_version = dataset.version().version;
    let field = dataset.schema().field(VECTOR_COLUMN).unwrap();
    let overlay_schema = dataset.schema().project_by_ids(&[field.id], true);

    let file = format!("{name}.lance");
    let store = dataset.object_store(None).await.unwrap();
    let mut writer = create_writer(
        ConcreteFileVersion::V2_1,
        store
            .create(&dataset.data_dir().join(file.as_str()))
            .await
            .unwrap(),
        overlay_schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_column(0, Arc::new(replacement)).await.unwrap();
    let summary = writer.finish().await.unwrap();

    let mut data_file = DataFile::new_unstarted(file, ConcreteFileVersion::V2_1);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(field_id, _)| *field_id as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, column_index)| *column_index as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id,
                overlays: vec![DataOverlayFile {
                    data_file,
                    coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter(
                        offsets.iter().copied(),
                    ))),
                    committed_version: 0,
                }],
            }],
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

/// Query vectors drawn the same way as the dataset's, from a different seed.
pub fn random_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
    random_vectors_of(count, VECTOR_DIM, seed)
}

pub fn random_vectors_of(count: usize, dimension: i32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dimension).map(|_| rng.random::<f32>()).collect())
        .collect()
}

/// The fraction of `truth` that `found` recovered.
///
/// The two arguments are not interchangeable, and with equal-length inputs the
/// arithmetic cannot tell them apart - so the lengths are asserted rather than
/// assumed. A `found` shorter than `truth` is a real result and would otherwise
/// be scored as if the missing answers had simply not been asked for.
pub fn recall(found: &[u64], truth: &[u64]) -> f64 {
    assert_eq!(
        found.len(),
        truth.len(),
        "recall compares a k-long answer against a k-long ground truth"
    );
    let found_set = found
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    truth.iter().filter(|row| found_set.contains(row)).count() as f64 / truth.len() as f64
}
