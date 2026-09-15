// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What an index type Lance cannot open does to the dataset that carries it.
//!
//! Nothing here builds an index of this crate's own. The segment committed is one
//! arbitrary file plus an index-details type url Lance does not know, which is
//! all that any out-of-tree index type has in common, so what happens to it is a
//! property of Lance rather than of this crate.
//!
//! Until upstream #8529 the damage did not stay on the column the foreign index
//! is on: it took `optimize_indices` down for every index of the dataset. Since
//! then Lance's read paths skip such an index and its write paths keep it. The
//! assertions pin both halves, on a first-party BTree over a different column
//! whose maintenance has to commit with the foreign segment in place.

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_io::object_store::ObjectStore;
use lance_table::io::manifest::read_manifest_indexes;
use object_store::path::Path;
use uuid::Uuid;

const VECTOR_COLUMN: &str = "vec";
const NUMBER_COLUMN: &str = "n";
const DIMENSION: i32 = 8;
const FOREIGN_INDEX: &str = "foreign_idx";
const SCALAR_INDEX: &str = "n_idx";

/// The name Lance classifies a vector index by, whatever the index really is:
/// `metadata_is_vector_index` in `rust/lance/src/index/append.rs`.
const INDEX_FILE_NAME: &str = "index.idx";

/// A type url no plugin claims. Lance validates that a segment set agrees on one
/// (`validate_segment_index_details`) and, since #8529, leaves an index whose
/// type has no reader out of its reader-side listing.
const FOREIGN_DETAILS_TYPE_URL: &str = "type.googleapis.com/example.MyIndexDetails";

#[tokio::test]
async fn an_index_type_lance_cannot_open_is_skipped_and_kept() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_rows(uri, 0..64, WriteMode::Create).await;
    dataset
        .create_index(
            &[NUMBER_COLUMN],
            IndexType::Scalar,
            Some(SCALAR_INDEX.to_owned()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Rows the BTree has not seen, so `optimize_indices` has real work to do and
    // really commits.
    let mut dataset = write_rows(uri, 64..96, WriteMode::Append).await;
    assert_eq!(
        unindexed_rows(&dataset).await,
        32,
        "the BTree has to start out with work pending"
    );

    commit_foreign_segment(&mut dataset).await;
    assert!(
        manifest_index_names(&dataset)
            .await
            .contains(&FOREIGN_INDEX.to_owned())
    );

    // Read paths skip it: the scanner answers the vector column as if no index
    // were there, and asking for the foreign index by name finds nothing.
    let query = Float32Array::from(vec![0.5f32; DIMENSION as usize]);
    let mut scanner = dataset.scan();
    scanner.nearest(VECTOR_COLUMN, &query, 5).unwrap();
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 5);
    let error = dataset.index_statistics(FOREIGN_INDEX).await.unwrap_err();
    assert!(
        matches!(error, lance_core::Error::IndexNotFound { .. }),
        "expected Lance not to see the foreign index, got: {error}"
    );

    // Write paths keep it: the BTree's maintenance commits, and the manifest that
    // commit wrote still names the foreign segment.
    dataset.optimize_indices(&Default::default()).await.unwrap();
    assert_eq!(unindexed_rows(&dataset).await, 0);
    assert!(
        manifest_index_names(&dataset)
            .await
            .contains(&FOREIGN_INDEX.to_owned()),
        "maintenance of another index erased the foreign one"
    );

    // Dropping it by name still works, which is how an operator gets rid of one.
    dataset.drop_index(FOREIGN_INDEX).await.unwrap();
    assert!(
        !manifest_index_names(&dataset)
            .await
            .contains(&FOREIGN_INDEX.to_owned())
    );
}

/// Rows of the BTree's column that the index has not covered yet, which is the
/// work `optimize_indices` exists to do.
async fn unindexed_rows(dataset: &Dataset) -> u64 {
    let stats = dataset.index_statistics(SCALAR_INDEX).await.unwrap();
    serde_json::from_str::<serde_json::Value>(&stats).unwrap()["num_unindexed_rows"]
        .as_u64()
        .unwrap()
}

/// Every index name the manifest records, including those Lance's own listing
/// leaves out.
async fn manifest_index_names(dataset: &Dataset) -> Vec<String> {
    let store = dataset.object_store(None).await.unwrap();
    read_manifest_indexes(&store, dataset.manifest_location(), dataset.manifest())
        .await
        .unwrap()
        .into_iter()
        .map(|index| index.name)
        .collect()
}

/// Commit an index segment whose type Lance has no reader for, the way any
/// out-of-tree index type has to: write the files, then name them in a commit.
async fn commit_foreign_segment(dataset: &mut Dataset) {
    let uuid = Uuid::new_v4();
    let store = dataset.object_store(None).await.unwrap();
    let index_file = dataset
        .indices_dir()
        .join(uuid.to_string())
        .join(INDEX_FILE_NAME);
    write_stub_index_file(&store, &index_file).await;

    let fragments = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let field_id = dataset.schema().field(VECTOR_COLUMN).unwrap().id;
    let dataset_version = dataset.manifest.version;
    let segment = IndexSegment::new(
        uuid,
        fragments,
        [field_id],
        Arc::new(prost_types::Any {
            type_url: FOREIGN_DETAILS_TYPE_URL.to_owned(),
            value: Vec::new(),
        }),
        1,
        dataset_version,
        vec![],
    );
    dataset
        .commit_existing_index_segments(FOREIGN_INDEX, VECTOR_COLUMN, vec![segment])
        .await
        .unwrap();
}

/// A perfectly well formed Lance file that simply is not one of Lance's own
/// vector indices, so it carries no `lance:index` schema metadata. That is the
/// shape of every out-of-tree index type: valid files, unknown contents.
async fn write_stub_index_file(store: &ObjectStore, path: &Path) {
    let schema = ArrowSchema::new(vec![Field::new("anything", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from_iter_values(0..4))],
    )
    .unwrap();
    let mut writer = create_writer(
        ConcreteFileVersion::V2_1,
        store.create(path).await.unwrap(),
        lance_core::datatypes::Schema::try_from(&schema).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();
}

async fn write_rows(uri: &str, rows: std::ops::Range<i32>, mode: WriteMode) -> Dataset {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(
            VECTOR_COLUMN,
            DataType::FixedSizeList(item, DIMENSION),
            false,
        ),
        Field::new(NUMBER_COLUMN, DataType::Int32, false),
    ]));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.clone()
            .map(|row| Some((0..DIMENSION).map(move |axis| Some(row as f32 + axis as f32)))),
        DIMENSION,
    );
    let numbers = Int32Array::from_iter_values(rows);
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors), Arc::new(numbers)]).unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            mode,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}
