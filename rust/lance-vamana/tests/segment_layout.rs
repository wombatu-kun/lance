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
use lance_file::version::ConcreteFileVersion;
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
    SEGMENT_FILE_VERSION, SegmentWriter, open_file, read_partition, read_segment, scan_scheduler,
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
        fragments: vec![0],
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

    let read = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap();
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

        let reader = open_file(
            &scan_scheduler(&store),
            &path.clone().join(entry.file.as_str()),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            &read_partition(&reader, entry.num_rows).await.unwrap(),
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

    let read = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap();
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

    let read = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap();
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

/// The entry point is a local id, so it has to be one. A medoid past the end of
/// the partition would be read back and handed straight to `greedy_search`,
/// which refuses it - one query at a time, forever, on a segment that was
/// already written.
#[tokio::test]
async fn the_writer_rejects_a_medoid_outside_the_partition() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let mut writer = SegmentWriter::new(store, path, index_metadata(), ivf_model());

    let error = writer
        .write_partition(0, 4, &sample_partition(MAX_DEGREE, 4, DIMENSION))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("medoid 4"), "{error}");
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

    let error = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap_err();
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

    let error = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains(IVF_POSITION_KEY), "{error}");
}

/// Write a segment's `index.idx` from a manifest, without writing its partitions.
///
/// [`SegmentWriter`] names each partition file itself, so this is what makes a
/// table naming something else possible to write at all.
async fn write_index_file(
    store: &ObjectStore,
    dir: &Path,
    manifest: &lance_vamana::SegmentManifest,
) {
    let batch = manifest.to_batch().unwrap();
    let schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap();
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store
            .create(&dir.clone().join(INDEX_FILE_NAME))
            .await
            .unwrap(),
        schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.add_schema_metadata(INDEX_METADATA_KEY, manifest.metadata().to_json().unwrap());
    let position = writer
        .add_global_buffer(
            prost::Message::encode_to_vec(&lance_index::pb::Ivf::try_from(manifest.ivf()).unwrap())
                .into(),
        )
        .await
        .unwrap();
    writer.add_schema_metadata(IVF_POSITION_KEY, position.to_string());
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();
}

/// A partition is found through the `__file` column, not through the
/// `part_%05d.idx` convention every writer in this crate happens to follow.
///
/// The two agree in everything the crate produces, so the only way to tell which
/// one the reader actually uses is to move a file out from under the convention
/// and leave the table pointing at where it went.
#[tokio::test]
async fn a_partition_is_found_through_the_table_not_the_naming_convention() {
    const RENAMED: &str = "somewhere-else.bin";
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let partition = sample_partition(MAX_DEGREE, 40, DIMENSION);

    let mut writer = SegmentWriter::new(store.clone(), path.clone(), index_metadata(), ivf_model());
    writer.write_partition(2, 7, &partition).await.unwrap();
    writer.finish().await.unwrap();

    std::fs::rename(
        dir.path().join(partition_file_name(2)),
        dir.path().join(RENAMED),
    )
    .unwrap();
    let retitled = lance_vamana::SegmentManifest::try_new(
        index_metadata(),
        ivf_model(),
        vec![lance_vamana::PartitionEntry {
            partition_id: 2,
            medoid: 7,
            num_rows: partition.len() as u32,
            file: RENAMED.to_string(),
        }],
    )
    .unwrap();
    write_index_file(&store, &path, &retitled).await;

    let read = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap();
    let entry = read.partition(2).expect("partition 2 is missing");
    assert_eq!(entry.file, RENAMED);
    let reader = open_file(
        &scan_scheduler(&store),
        &path.clone().join(entry.file.as_str()),
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        &read_partition(&reader, entry.num_rows).await.unwrap(),
        &partition
    );
}

/// Write an `index.idx` by hand with whatever schema metadata is asked for.
async fn write_hand_made_index(
    store: &ObjectStore,
    dir: &Path,
    metadata: &[(&str, String)],
    global_buffer: Option<Vec<u8>>,
    version: ConcreteFileVersion,
    rows: usize,
) {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::UInt32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from(
            (0..rows as u32).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let mut writer = create_writer(
        version,
        store
            .create(&dir.clone().join(INDEX_FILE_NAME))
            .await
            .unwrap(),
        lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    if let Some(bytes) = global_buffer {
        writer.add_global_buffer(bytes.into()).await.unwrap();
    }
    for (key, value) in metadata {
        writer.add_schema_metadata(*key, value.clone());
    }
    // An empty batch would be written as a column of no values, which is not
    // the same file as one with no batch at all.
    if rows > 0 {
        writer.write_batch(&batch).await.unwrap();
    }
    writer.finish().await.unwrap();
}

/// Global buffer indices are one-based; buffer 0 is the file's own descriptor.
#[tokio::test]
async fn a_segment_pointing_at_the_descriptor_buffer_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    write_hand_made_index(
        &store,
        &path,
        &[
            (INDEX_METADATA_KEY, index_metadata().to_json().unwrap()),
            (IVF_POSITION_KEY, "0".to_string()),
        ],
        None,
        SEGMENT_FILE_VERSION,
        1,
    )
    .await;

    let error = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("file descriptor"), "{error}");
}

/// `IvfModel::try_from` asserts these agree; without a guard here a malformed
/// buffer aborts the process instead of being reported.
#[tokio::test]
async fn an_ivf_model_with_mismatched_offsets_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let error = read_segment_carrying(
        store,
        &path,
        lance_index::pb::Ivf {
            offsets: vec![0, 1, 2],
            lengths: vec![],
            ..Default::default()
        },
    )
    .await;
    assert!(error.to_string().contains("3 offsets"), "{error}");
}

/// Writes an `index.idx` whose global buffer is `ivf`, and reads it back.
async fn read_segment_carrying(
    store: Arc<ObjectStore>,
    dir: &Path,
    ivf: lance_index::pb::Ivf,
) -> lance_core::Error {
    write_hand_made_index(
        &store,
        dir,
        &[
            (INDEX_METADATA_KEY, index_metadata().to_json().unwrap()),
            (IVF_POSITION_KEY, "1".to_string()),
        ],
        Some(prost::Message::encode_to_vec(&ivf)),
        SEGMENT_FILE_VERSION,
        1,
    )
    .await;
    read_segment(&scan_scheduler(&store), dir, None)
        .await
        .unwrap_err()
}

/// The v1 centroid layout recovers its width by dividing by the number of
/// partitions, which it reads from `lengths`; empty, that is a division by zero.
#[tokio::test]
async fn an_ivf_model_with_legacy_centroids_and_no_lengths_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let error = read_segment_carrying(
        store,
        &path,
        lance_index::pb::Ivf {
            centroids: vec![0.5; DIMENSION as usize * PARTITIONS],
            lengths: vec![],
            ..Default::default()
        },
    )
    .await;
    assert!(
        error.to_string().contains("no partition lengths"),
        "{error}"
    );
}

/// `FixedSizeListArray::try_from(&Tensor)` unwraps the enum conversion, so a
/// data type outside the range aborts the process instead of being reported.
#[tokio::test]
async fn an_ivf_model_with_an_unknown_centroid_type_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let error = read_segment_carrying(
        store,
        &path,
        lance_index::pb::Ivf {
            centroids_tensor: Some(lance_index::pb::Tensor {
                data_type: 99,
                shape: vec![PARTITIONS as u32, DIMENSION],
                data: vec![],
            }),
            ..Default::default()
        },
    )
    .await;
    assert!(
        error.to_string().contains("unknown data type 99"),
        "{error}"
    );
}

/// Not a crash but a deferred failure: routing dispatches on the pair of
/// centroid and query types, and this crate only ever builds an f32 query, so a
/// model of another width opens cleanly and then fails on every search.
#[tokio::test]
async fn an_ivf_model_of_another_float_width_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let values = (0..PARTITIONS * DIMENSION as usize)
        .flat_map(|i| (i as f64).to_le_bytes())
        .collect::<Vec<u8>>();
    let error = read_segment_carrying(
        store,
        &path,
        lance_index::pb::Ivf {
            centroids_tensor: Some(lance_index::pb::Tensor {
                data_type: lance_index::pb::tensor::DataType::Float64 as i32,
                shape: vec![PARTITIONS as u32, DIMENSION],
                data: values,
            }),
            ..Default::default()
        },
    )
    .await;
    assert!(error.to_string().contains("expected Float32"), "{error}");
}

/// The rule "an empty partition gets no file" belongs to the format, not to one
/// writer: the free function is public and is what `SegmentWriter` delegates to.
#[tokio::test]
async fn the_free_writer_refuses_an_empty_partition() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let error = lance_vamana::io::write_partition(
        &store,
        &path.clone().join("part_00000.idx"),
        &sample_partition(MAX_DEGREE, 0, DIMENSION),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("empty partition"), "{error}");
}

/// A segment that breaks a rule of the format is a corrupt file, not a caller's
/// bad input. The distinction is not cosmetic: the same constructor serves the
/// writer, where the caller *is* the one who got it wrong, so the two have to be
/// told apart by where the value came from rather than by what was wrong with
/// it.
///
/// Reached with a table of no rows, so the columns are beyond reproach and the
/// refusal comes from the parameters - which is the half of the constructor that
/// still reported bad input.
#[tokio::test]
async fn a_segment_breaking_a_rule_of_the_format_is_reported_as_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    let broken = IndexMetadata {
        max_degree: 0,
        ..index_metadata()
    };
    write_hand_made_index(
        &store,
        &path,
        &[
            (INDEX_METADATA_KEY, broken.to_json().unwrap()),
            (IVF_POSITION_KEY, "1".to_string()),
        ],
        Some(prost::Message::encode_to_vec(
            &lance_index::pb::Ivf::try_from(&ivf_model()).unwrap(),
        )),
        SEGMENT_FILE_VERSION,
        0,
    )
    .await;

    let error = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, lance_core::Error::CorruptFile { .. }),
        "a value that arrived out of a file was reported as bad input: {error:?}"
    );
    assert!(error.to_string().contains("max_degree 0"), "{error}");
}

/// The writer pins the file version, so the reader has to check it. A projection
/// is computed against the structural grammar of one version, and a file written
/// under another lays its columns out differently: the read would come back with
/// the wrong bytes rather than fail.
#[tokio::test]
async fn a_segment_file_of_another_lance_version_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = segment_dir(&dir);
    write_hand_made_index(
        &store,
        &path,
        &[
            (INDEX_METADATA_KEY, index_metadata().to_json().unwrap()),
            (IVF_POSITION_KEY, "1".to_string()),
        ],
        Some(prost::Message::encode_to_vec(
            &lance_index::pb::Ivf::try_from(&ivf_model()).unwrap(),
        )),
        ConcreteFileVersion::V2_0,
        1,
    )
    .await;

    let error = read_segment(&scan_scheduler(&store), &path, None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, lance_core::Error::CorruptFile { .. }),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("2.0") && error.to_string().contains("2.1"),
        "the error should name both versions: {error}"
    );
}
