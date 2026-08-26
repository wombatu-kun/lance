// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build one Lance `IVF_HNSW_SQ` index in its shipped default shape, so that a
//! later measurement can read it beside this crate's own index without paying
//! the build again.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/gist VECTORS=0 DATASET_DIR=~/vamana-runs \
//!   cargo run --profile release-no-lto --example hnsw_index
//! ```
//!
//! Environment: `SIFT_DIR` (required), `DATASET_DIR` (required), `VECTORS`
//! (default 100000, `0` for all) and `PARTITIONS` (default 1).
//!
//! Lance has no bare HNSW index type: the graph always sits under an IVF and
//! over a quantizer. `IVF_HNSW_SQ` at one partition is the shape that leaves
//! only the graph moving, and it is what upstream's own default
//! `target_partition_size` resolves to on a million rows.
//!
//! The rows are written exactly as `ivf_rq_ab` writes its own, so the arms read
//! the same table in the same layout. The index gets its own dataset rather
//! than a second index on a shared one, because the scanner picks a vector
//! index by field id alone and would not be given a choice.

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::DistanceType;

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const ID_COLUMN: &str = "id";
const VECTOR_FIELD: &str = "vector";
const HNSW_INDEX: &str = "hnsw_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;

#[tokio::main]
async fn main() {
    let dir = std::env::var("SIFT_DIR").expect("set SIFT_DIR to the extracted dataset directory");
    let home =
        std::env::var("DATASET_DIR").expect("set DATASET_DIR to where the index should live");
    let prefix = std::path::Path::new(&dir)
        .file_name()
        .and_then(|name| name.to_str())
        .expect("SIFT_DIR must end in the dataset name")
        .to_string();

    let (base, dim, total) = read_fvecs(&format!("{dir}/{prefix}_base.fvecs"));
    let requested = env_usize("VECTORS", 100_000);
    let rows = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let partitions = env_usize("PARTITIONS", 1);

    let uri = format!("{home}/{prefix}-{rows}-p{partitions}-hnswsq.lance");
    assert!(
        std::fs::metadata(&uri).is_err(),
        "{uri} already exists: delete it or point DATASET_DIR elsewhere"
    );

    let hnsw = HnswBuildParams::default();
    let sq = SQBuildParams::default();
    println!(
        "{prefix} {rows} x {dim}, {partitions} partitions, m={} ef_construction={} \
         max_level={}, sq {} bits",
        hnsw.m, hnsw.ef_construction, hnsw.max_level, sq.num_bits
    );

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();
    drop(base);

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_FIELD, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
            Arc::new(vectors),
        ],
    )
    .unwrap();

    let started = Instant::now();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &uri,
        Some(WriteParams::default()),
    )
    .await
    .unwrap();
    println!("rows written in {:.1}s", started.elapsed().as_secs_f64());

    let started = Instant::now();
    dataset
        .create_index(
            &[VECTOR_FIELD],
            IndexType::IvfHnswSq,
            Some(HNSW_INDEX.to_string()),
            &VectorIndexParams::with_ivf_hnsw_sq_params(
                DISTANCE_TYPE,
                IvfBuildParams::new(partitions),
                hnsw,
                sq,
            ),
            false,
        )
        .await
        .unwrap();
    println!(
        "IVF_HNSW_SQ indexed in {:.1}s at {uri}",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
}
