// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the integration tests.
//!
//! Every integration binary compiles this module whole, so a fixture only one of
//! them needs still lands in the others.
#![allow(dead_code)]

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance_vamana::partition::{Partition, PartitionGraph};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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
}

impl Default for DatasetFixture {
    fn default() -> Self {
        Self {
            fragments: 3,
            rows_per_fragment: 512,
            stable_row_ids: false,
            null_every: None,
            seed: 11,
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
            DataType::FixedSizeList(item, VECTOR_DIM),
            true,
        )]));

        let mut rng = SmallRng::seed_from_u64(self.seed);
        let pool = (0..self.rows())
            .map(|_| {
                (0..VECTOR_DIM)
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
            VECTOR_DIM,
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
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }
}

/// Query vectors drawn the same way as the dataset's, from a different seed.
pub fn random_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..VECTOR_DIM).map(|_| rng.random::<f32>()).collect())
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
