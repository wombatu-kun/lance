// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The rows of a seed: every column a function of the id and the data seed.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use super::splitmix64;

pub const VECTOR_DIM: i32 = 32;
pub const IVF_PARTITIONS: usize = 2;
/// The probes of an exact IVF query, far more partitions than an index of a seed
/// reaches. A default `optimize_indices` splits a partition whose entries, those
/// of deleted and moved rows included, pass four times the rows per partition the
/// index was built with, so an index may come to have more than `IVF_PARTITIONS`;
/// Lance probes no more partitions than a segment has.
pub const ALL_PARTITIONS: usize = 1 << 16;
pub const NEIGHBORS: usize = 10;
pub const CATEGORIES: i32 = 7;
/// The rows of one generated batch: the reader builds them only when asked, so
/// the source of a write holds almost no memory of its own.
const CHUNK_ROWS: i64 = 1024;

/// How much data a scenario works on.
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    pub rows_per_file: usize,
    pub initial_rows: (u64, u64),
    pub added_rows: (u64, u64),
    pub min_live_rows: usize,
    doc_tokens: u64,
    pub vocabulary: u64,
}

pub const SMALL: Scale = Scale {
    rows_per_file: 200,
    initial_rows: (600, 1200),
    added_rows: (100, 400),
    min_live_rows: 300,
    doc_tokens: 3,
    vocabulary: 16,
};

/// Enough data for what a build retains to show, in cycles short enough to repeat.
pub const MEDIUM: Scale = Scale {
    rows_per_file: 500,
    initial_rows: (5000, 8000),
    added_rows: (500, 1000),
    min_live_rows: 1000,
    doc_tokens: 8,
    vocabulary: 256,
};

/// Enough data for the memory of an operation to stand out from the process.
pub const LARGE: Scale = Scale {
    rows_per_file: 2000,
    initial_rows: (20_000, 40_000),
    added_rows: (2000, 8000),
    min_live_rows: 10_000,
    doc_tokens: 16,
    vocabulary: 4096,
};

pub fn category(id: i64) -> i32 {
    (splitmix64(id as u64 ^ 0xCA7E_9087) % CATEGORIES as u64) as i32
}

pub fn tokens(id: i64, scale: &Scale) -> impl Iterator<Item = u64> {
    let (doc_tokens, vocabulary) = (scale.doc_tokens, scale.vocabulary);
    (0..doc_tokens).map(move |slot| splitmix64(((id as u64) << 8) | slot) % vocabulary)
}

pub fn text(id: i64, scale: &Scale) -> String {
    tokens(id, scale)
        .map(|token| format!("tok{token:04}"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn vector(data_seed: u64, id: i64) -> [f32; VECTOR_DIM as usize] {
    std::array::from_fn(|lane| {
        let bits =
            splitmix64(data_seed ^ splitmix64((id as u64) * VECTOR_DIM as u64 + lane as u64));
        (bits >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    })
}

pub fn squared_l2(a: &[f32; VECTOR_DIM as usize], b: &[f32; VECTOR_DIM as usize]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub fn schema() -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", DataType::Int32, false),
        Field::new("vec", DataType::FixedSizeList(item, VECTOR_DIM), false),
        Field::new("text", DataType::Utf8, false),
    ]))
}

pub fn rows(
    data_seed: u64,
    ids: Range<i64>,
    scale: Scale,
) -> impl RecordBatchReader + Send + 'static {
    let end = ids.end;
    let chunks = ids.step_by(CHUNK_ROWS as usize).map(move |start| {
        Ok(batch_of_rows(
            data_seed,
            start..(start + CHUNK_ROWS).min(end),
            &scale,
        ))
    });
    RecordBatchIterator::new(chunks, schema())
}

fn batch_of_rows(data_seed: u64, ids: Range<i64>, scale: &Scale) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let vectors = Float32Array::from_iter_values(ids.iter().flat_map(|id| vector(data_seed, *id)));
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ids.clone())),
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
    ];
    RecordBatch::try_new(schema(), columns).expect("build a batch")
}
