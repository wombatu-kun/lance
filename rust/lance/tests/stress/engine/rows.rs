// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rows with a `gen` column, and the writes of them that `concurrency` and `soak` share.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use lance::Dataset;
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::write::merge_insert::{
    MergeInsertWriteMode, MergeStats, WhenMatched, WhenNotMatched,
};
use lance::dataset::{MergeInsertBuilder, MergeInsertJob, UpdateBuilder};
use lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY;

use crate::engine::data::{Scale, VECTOR_DIM, category, text, vector};
use crate::engine::ops::OpOutput;

/// A hold point can eat the default 30 s, and an attempt past the limit is cut
/// off even if its commit has landed.
const RETRY_TIMEOUT: Duration = Duration::from_secs(120);

/// The rows of one class: each id with the `gen` it holds.
pub type ClassRows = BTreeMap<i64, i64>;

pub fn id_run(first_id: i64, rows: i64, stride: i64) -> impl Iterator<Item = i64> {
    (0..rows).map(move |slot| first_id + slot * stride)
}

/// The columns of the rows, with `gen`, which tells the operations apart.
pub fn schema(declared_key: bool) -> SchemaRef {
    let mut id = Field::new("id", DataType::Int64, false);
    if declared_key {
        id = id.with_metadata(HashMap::from([(
            LANCE_UNENFORCED_PRIMARY_KEY.to_string(),
            "true".to_string(),
        )]));
    }
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![
        id,
        Field::new("category", DataType::Int32, false),
        Field::new("vec", DataType::FixedSizeList(item, VECTOR_DIM), false),
        Field::new("text", DataType::Utf8, false),
        Field::new("gen", DataType::Int64, false),
    ]))
}

pub fn source_of(batch: RecordBatch) -> impl RecordBatchReader + Send + 'static {
    let schema = batch.schema();
    RecordBatchIterator::new(vec![Ok(batch)], schema)
}

/// Full rows, every column a function of the id but `gen`.
pub fn full_batch(
    schema: &SchemaRef,
    data_seed: u64,
    scale: &Scale,
    ids: &[i64],
    stamp: i64,
) -> RecordBatch {
    let vectors = Float32Array::from_iter_values(ids.iter().flat_map(|id| vector(data_seed, *id)));
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ids.to_vec())),
        Arc::new(Int32Array::from_iter_values(
            ids.iter().map(|id| category(*id)),
        )),
        Arc::new(
            FixedSizeListArray::try_new(item, VECTOR_DIM, Arc::new(vectors), None)
                .expect("build the vectors"),
        ),
        Arc::new(StringArray::from_iter_values(
            ids.iter().map(|id| text(*id, scale)),
        )),
        Arc::new(Int64Array::from(vec![stamp; ids.len()])),
    ];
    RecordBatch::try_new(schema.clone(), columns).expect("build a batch")
}

/// How many rows an operation says it touched.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowCounts {
    pub inserted: u64,
    pub updated: u64,
    pub deleted: u64,
}

/// What a try of an operation reports.
#[derive(Default)]
pub struct Attempted {
    pub output: OpOutput,
    pub counts: Option<RowCounts>,
    /// How many tries Lance itself made inside, where it says.
    pub tries_inside: u32,
}

/// The rows of a worker's class in a range of ids.
pub fn class_range(stride: i64, worker: usize, lo: i64, hi: i64) -> String {
    format!("id % {stride} = {worker} AND id >= {lo} AND id < {hi}")
}

pub async fn delete(ds: &mut Dataset, predicate: String) -> lance::Result<Attempted> {
    let deleted = DeleteBuilder::new(Arc::new(ds.clone()), predicate)
        .retry_timeout(RETRY_TIMEOUT)
        .execute()
        .await?;
    *ds = deleted.new_dataset.as_ref().clone();
    Ok(Attempted {
        counts: Some(RowCounts {
            deleted: deleted.num_deleted_rows,
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub async fn update(ds: &mut Dataset, predicate: &str, value: &str) -> lance::Result<Attempted> {
    let job = UpdateBuilder::new(Arc::new(ds.clone()))
        .update_where(predicate)?
        .set("gen", value)?
        .retry_timeout(RETRY_TIMEOUT)
        .build()?;
    let updated = Box::pin(job.execute()).await?;
    *ds = updated.new_dataset.as_ref().clone();
    Ok(Attempted {
        counts: Some(RowCounts {
            updated: updated.rows_updated,
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub fn upsert_job(ds: &Dataset) -> lance::Result<MergeInsertJob> {
    MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])?
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .retry_timeout(RETRY_TIMEOUT)
        .try_build()
}

pub async fn upsert(
    ds: &mut Dataset,
    source: Box<dyn RecordBatchReader + Send>,
) -> lance::Result<Attempted> {
    let (merged, stats) = Box::pin(upsert_job(ds)?.execute_reader(source)).await?;
    *ds = merged.as_ref().clone();
    Ok(merge_report(&stats))
}

/// Rewrites `gen` in place for the rows that exist, and inserts none.
pub fn stamp_job(ds: &Dataset) -> lance::Result<MergeInsertJob> {
    MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])?
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .retry_timeout(RETRY_TIMEOUT)
        .try_build()
}

pub fn stamp_source(
    schema: &SchemaRef,
    ids: Vec<i64>,
    number: i64,
) -> Box<dyn RecordBatchReader + Send> {
    let columns = Arc::new(Schema::new(vec![
        schema.field(0).clone(),
        Field::new("gen", DataType::Int64, false),
    ]));
    let stamps = vec![number; ids.len()];
    let batch = RecordBatch::try_new(
        columns,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(stamps)),
        ],
    )
    .expect("build a batch");
    Box::new(source_of(batch))
}

pub fn merge_report(stats: &MergeStats) -> Attempted {
    Attempted {
        counts: Some(RowCounts {
            inserted: stats.num_inserted_rows,
            updated: stats.num_updated_rows,
            deleted: stats.num_deleted_rows,
        }),
        tries_inside: stats.num_attempts,
        ..Default::default()
    }
}
