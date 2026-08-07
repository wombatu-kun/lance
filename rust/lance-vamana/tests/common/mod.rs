// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the integration tests.

use lance_vamana::partition::PartitionGraph;

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
