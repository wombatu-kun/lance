// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the integration tests.

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use lance_vamana::partition::{Partition, PartitionGraph};

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
