// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Does a whole segment survive the trip to disk and back?
//!
//! A segment is three things stored three different ways - parameters in schema
//! metadata, the IVF routing model in a global buffer, the partition table in
//! columns - and each of them has its own way of going missing. So the test
//! writes one and reads it back through the public reader, rather than
//! inspecting the pieces it just wrote.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_arrow::FixedSizeListArrayExt;
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_index::vector::ivf::storage::IvfModel;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::DistanceType;
use lance_vamana::format::{
    FORMAT_VERSION, INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, IndexMetadata,
    RowIdMode, partition_file_name,
};
use lance_vamana::io::{
    SEGMENT_FILE_VERSION, SegmentWriter, open_file, read_partition, read_segment,
};
use lance_vamana::partition::Partition;
use object_store::path::Path;

mod common;
use common::sample_partition;

const MAX_DEGREE: u32 = 32;
const DIMENSION: u32 = 6;
const PARTITIONS: usize = 8;

/// Sparse on purpose: partitions 1, 4, 6 and 7 hold nothing. Sizes and medoids
/// all differ, so a table that mixed its rows up could not still pass.
const POPULATED: [(u32, usize); 4] = [(0, 40), (2, 7), (3, 128), (5, 1)];

fn index_metadata() -> IndexMetadata {
    IndexMetadata {
        format_version: FORMAT_VERSION,
        max_degree: MAX_DEGREE,
        alpha: 1.2,
        dimension: DIMENSION,
        distance_type: DistanceType::Cosine,
        row_id_mode: RowIdMode::Address,
    }
}

/// Centroids that are all different from each other, so a model that came back
/// transposed or truncated cannot compare equal.
fn ivf_model() -> IvfModel {
    let values = Float32Array::from(
        (0..PARTITIONS * DIMENSION as usize)
            .map(|i| i as f32 * 0.25)
            .collect::<Vec<_>>(),
    );
    IvfModel::new(
        FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap(),
        Some(1.5),
    )
}

fn segment_dir(dir: &tempfile::TempDir) -> (Arc<ObjectStore>, Path) {
    (
        Arc::new(ObjectStore::local()),
        Path::from_absolute_path(dir.path()).unwrap(),
    )
}

/// Writes the fixture segment and hands back what each partition was given.
async fn write_sample_segment(
    store: Arc<ObjectStore>,
    dir: &Path,
) -> (
    lance_vamana::SegmentManifest,
    HashMap<u32, (u32, Partition)>,
) {
    let mut writer = SegmentWriter::new(store, dir.clone(), index_metadata(), ivf_model());
    let mut written = HashMap::new();
    for (partition_id, vertices) in POPULATED {
        let partition = sample_partition(MAX_DEGREE, vertices, DIMENSION);
        let medoid = (vertices / 3) as u32;
        writer
            .write_partition(partition_id, medoid, &partition)
            .await
            .unwrap();
        written.insert(partition_id, (medoid, partition));
    }
    (writer.finish().await.unwrap(), written)
}

#[tokio::test]
async fn segment_round_trips_through_a_directory() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let (written, partitions) = write_sample_segment(store.clone(), &path).await;

    let read = read_segment(store.clone(), &path).await.unwrap();
    assert_eq!(
        read, written,
        "the segment read back is not the one written"
    );
    assert_eq!(read.metadata(), &index_metadata());
    assert_eq!(read.ivf(), &ivf_model());

    for (partition_id, (medoid, partition)) in &partitions {
        let entry = read
            .partition(*partition_id)
            .unwrap_or_else(|| panic!("partition {partition_id} is missing from the table"));
        assert_eq!(entry.medoid, *medoid);
        assert_eq!(entry.num_rows as usize, partition.len());

        let reader = open_file(store.clone(), &path.clone().join(entry.file.as_str()), None)
            .await
            .unwrap();
        assert_eq!(
            &read_partition(&reader).await.unwrap(),
            partition,
            "partition {partition_id} did not round trip"
        );
    }
}

/// An empty partition is absent, not present-and-empty: no row, no file.
#[tokio::test]
async fn empty_partitions_leave_no_trace() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    write_sample_segment(store.clone(), &path).await;

    let read = read_segment(store.clone(), &path).await.unwrap();
    for partition_id in 0..PARTITIONS as u32 {
        let expected = POPULATED.iter().any(|(id, _)| *id == partition_id);
        assert_eq!(
            read.partition(partition_id).is_some(),
            expected,
            "partition {partition_id} is listed when it should not be, or the other way round"
        );
    }

    // Lance fills an index's file list by listing this directory, so what is on
    // disk is exactly what Lance will believe the index consists of.
    let mut found = store.read_dir(path).await.unwrap();
    found.sort();
    let mut expected = POPULATED
        .iter()
        .map(|(partition_id, _)| partition_file_name(*partition_id))
        .collect::<Vec<_>>();
    expected.push(INDEX_FILE_NAME.to_string());
    expected.sort();
    assert_eq!(found, expected);
}

/// Degenerate but reachable: every partition empty means `index.idx` has no
/// rows at all, and the parameters still have to survive.
#[tokio::test]
async fn a_segment_with_no_partitions_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    SegmentWriter::new(store.clone(), path.clone(), index_metadata(), ivf_model())
        .finish()
        .await
        .unwrap();

    let read = read_segment(store, &path).await.unwrap();
    assert!(read.partitions().is_empty());
    assert_eq!(read.metadata(), &index_metadata());
    assert_eq!(read.ivf(), &ivf_model());
}

#[tokio::test]
async fn the_writer_rejects_an_empty_partition() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let mut writer = SegmentWriter::new(store, path, index_metadata(), ivf_model());

    let error = writer
        .write_partition(0, 0, &sample_partition(MAX_DEGREE, 0, DIMENSION))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("is empty"), "{error}");
}

/// The stride is a segment-wide constant, so a partition at another degree would
/// be unreadable by anything that trusted `index.idx`.
#[tokio::test]
async fn the_writer_rejects_a_partition_of_the_wrong_degree() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let mut writer = SegmentWriter::new(store, path, index_metadata(), ivf_model());

    let error = writer
        .write_partition(0, 0, &sample_partition(MAX_DEGREE * 2, 4, DIMENSION))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("max_degree 64"), "{error}");
}

/// So is the dimension: the routing model and every other partition assume it.
#[tokio::test]
async fn the_writer_rejects_a_partition_of_the_wrong_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let mut writer = SegmentWriter::new(store, path, index_metadata(), ivf_model());

    let error = writer
        .write_partition(0, 0, &sample_partition(MAX_DEGREE, 4, DIMENSION + 1))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("dimension 7"), "{error}");
}

#[tokio::test]
async fn the_writer_rejects_partitions_out_of_order() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let mut writer = SegmentWriter::new(store, path, index_metadata(), ivf_model());

    let partition = sample_partition(MAX_DEGREE, 4, DIMENSION);
    writer.write_partition(3, 0, &partition).await.unwrap();
    let error = writer.write_partition(1, 0, &partition).await.unwrap_err();
    assert!(error.to_string().contains("ascending order"), "{error}");
    let error = writer.write_partition(3, 0, &partition).await.unwrap_err();
    assert!(error.to_string().contains("ascending order"), "{error}");
}

/// Pointing the reader at somebody else's `index.idx` must say so, not decode
/// whatever happens to be there. Lance's own vector indices use the same name.
#[tokio::test]
async fn a_foreign_index_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::UInt32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from(vec![1u32]))],
    )
    .unwrap();
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store
            .create(&path.clone().join(INDEX_FILE_NAME))
            .await
            .unwrap(),
        lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();

    let error = read_segment(store, &path).await.unwrap_err();
    assert!(error.to_string().contains(INDEX_METADATA_KEY), "{error}");
}

/// The parameters and the routing model are stored two different ways, so
/// losing one must not look like losing the other.
#[tokio::test]
async fn a_segment_without_its_ivf_model_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::UInt32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from(vec![1u32]))],
    )
    .unwrap();
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store
            .create(&path.clone().join(INDEX_FILE_NAME))
            .await
            .unwrap(),
        lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.add_schema_metadata(INDEX_METADATA_KEY, index_metadata().to_json().unwrap());
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();

    let error = read_segment(store, &path).await.unwrap_err();
    assert!(error.to_string().contains(IVF_POSITION_KEY), "{error}");
}
