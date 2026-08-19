// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What an index type Lance cannot open does to the dataset that carries it.
//!
//! Nothing here builds an index of this crate's own. The segment committed is one
//! arbitrary file plus an index-details type url Lance does not know, which is
//! all that any out-of-tree index type has in common, so what breaks is a
//! property of the missing plugin surface rather than of this crate.
//!
//! The damage does not stay on the column the foreign index is on: the last
//! assertions are about a first-party BTree over a different column, whose
//! maintenance the foreign segment blocks.

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
/// (`validate_segment_index_details`) and never that the one it agrees on is a
/// type Lance can open.
const FOREIGN_DETAILS_TYPE_URL: &str = "type.googleapis.com/example.MyIndexDetails";

#[tokio::test]
async fn an_index_type_lance_cannot_open_degrades_the_whole_dataset() {
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

    // Rows the BTree has not seen, so `optimize_indices` has real work to do.
    let mut dataset = write_rows(uri, 64..96, WriteMode::Append).await;

    let query = Float32Array::from(vec![0.5f32; DIMENSION as usize]);
    let nearest = |dataset: &Dataset, use_index: bool| {
        let mut scanner = dataset.scan();
        scanner.nearest(VECTOR_COLUMN, &query, 5).unwrap();
        scanner.use_index(use_index);
        async move { scanner.try_into_batch().await }
    };

    assert_eq!(nearest(&dataset, true).await.unwrap().num_rows(), 5);
    let pending = unindexed_rows(&dataset).await;
    assert_eq!(pending, 32, "the BTree has to start out with work pending");

    commit_foreign_segment(&mut dataset).await;

    // The scanner picks a vector index by field id alone, without checking that
    // the index it found is one it can open: `indices.iter().find(|i|
    // i.fields.contains(&column_id))` in `rust/lance/src/dataset/scanner.rs`.
    let shadowed = nearest(&dataset, true).await.unwrap_err();
    assert!(
        shadowed.to_string().contains("Index Metadata not found"),
        "expected the vector path to fail on an index it cannot open, got: {shadowed}"
    );
    assert_eq!(
        nearest(&dataset, false).await.unwrap().num_rows(),
        5,
        "the exhaustive path is the only escape hatch left, it has to stay open"
    );
    dataset.index_statistics(FOREIGN_INDEX).await.unwrap_err();

    // The blast radius. `optimize_indices` walks every index of the dataset and
    // classifies each as vector by the presence of `index.idx`, so the foreign
    // segment takes the BTree over `n` down with it.
    let blocked = dataset
        .optimize_indices(&Default::default())
        .await
        .unwrap_err();
    assert!(
        blocked.to_string().contains("Index Metadata not found"),
        "expected optimizing the BTree to fail because of the foreign segment, got: {blocked}"
    );
    assert_eq!(
        unindexed_rows(&dataset).await,
        pending,
        "the BTree's pending rows are still pending, so the failure lost the work"
    );

    // Dropping the foreign index restores every one of them, which is what makes
    // it the cause rather than a coincidence.
    dataset.drop_index(FOREIGN_INDEX).await.unwrap();
    assert_eq!(nearest(&dataset, true).await.unwrap().num_rows(), 5);
    dataset.optimize_indices(&Default::default()).await.unwrap();
    assert_eq!(unindexed_rows(&dataset).await, 0);
}

/// Rows of the BTree's column that the index has not covered yet, which is the
/// work `optimize_indices` exists to do.
async fn unindexed_rows(dataset: &Dataset) -> u64 {
    let stats = dataset.index_statistics(SCALAR_INDEX).await.unwrap();
    serde_json::from_str::<serde_json::Value>(&stats).unwrap()["num_unindexed_rows"]
        .as_u64()
        .unwrap()
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
