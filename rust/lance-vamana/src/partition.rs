// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One partition's graph, in memory, in the shape it has on disk.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::DataType;
use lance_core::{Error, Result};

use crate::format::{
    MAX_PARTITION_ROWS, NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, partition_schema,
};

/// The out-edges of one IVF partition, plus the row id of each vertex.
///
/// Vertices are addressed by *partition-local* id, which is simply the row's
/// position in this structure. Edges therefore never name a row id and never
/// leave the partition, which is what makes both consolidation and dataset
/// compaction rewrite only [`Self::row_ids`] and leave the adjacency untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionGraph {
    max_degree: u32,
    row_ids: Vec<u64>,
    /// `max_degree` slots per vertex, tail-padded with [`NO_NEIGHBOR`].
    neighbors: Vec<u32>,
}

impl PartitionGraph {
    /// Build a partition from one adjacency list per vertex.
    ///
    /// Lists shorter than `max_degree` are padded; the padding is what lets a
    /// later insert or prune change a vertex's degree without moving any other
    /// vertex on disk.
    pub fn try_new(max_degree: u32, row_ids: Vec<u64>, adjacency: Vec<Vec<u32>>) -> Result<Self> {
        if max_degree == 0 {
            return Err(Error::invalid_input(
                "Vamana max_degree must be greater than zero".to_string(),
            ));
        }
        if row_ids.len() != adjacency.len() {
            return Err(Error::invalid_input(format!(
                "Vamana partition has {} row ids but {} adjacency lists",
                row_ids.len(),
                adjacency.len()
            )));
        }
        if row_ids.len() as u64 > MAX_PARTITION_ROWS as u64 {
            return Err(Error::invalid_input(format!(
                "Vamana partition holds {} rows, exceeding the addressable maximum {}",
                row_ids.len(),
                MAX_PARTITION_ROWS
            )));
        }

        let num_rows = row_ids.len();
        let width = max_degree as usize;
        let mut neighbors = vec![NO_NEIGHBOR; num_rows * width];
        for (local_id, out_edges) in adjacency.iter().enumerate() {
            if out_edges.len() > width {
                return Err(Error::invalid_input(format!(
                    "Vamana vertex {local_id} has degree {} which exceeds max_degree {max_degree}",
                    out_edges.len()
                )));
            }
            for (slot, neighbor) in out_edges.iter().enumerate() {
                if *neighbor as usize >= num_rows {
                    return Err(Error::invalid_input(format!(
                        "Vamana vertex {local_id} points at local id {neighbor}, \
                         but the partition holds only {num_rows} vertices"
                    )));
                }
                neighbors[local_id * width + slot] = *neighbor;
            }
        }

        Ok(Self {
            max_degree,
            row_ids,
            neighbors,
        })
    }

    pub fn max_degree(&self) -> u32 {
        self.max_degree
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    pub fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }

    /// Out-edges of `local_id`, with the padding trimmed off.
    pub fn neighbors(&self, local_id: usize) -> &[u32] {
        let width = self.max_degree as usize;
        let slots = &self.neighbors[local_id * width..(local_id + 1) * width];
        let degree = slots
            .iter()
            .position(|neighbor| *neighbor == NO_NEIGHBOR)
            .unwrap_or(width);
        &slots[..degree]
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(partition_schema(self.max_degree)?);
        let DataType::FixedSizeList(item, width) =
            schema.field_with_name(NEIGHBORS_COLUMN)?.data_type()
        else {
            unreachable!("partition_schema always produces a fixed size list");
        };
        let neighbors = FixedSizeListArray::try_new(
            item.clone(),
            *width,
            Arc::new(UInt32Array::from(self.neighbors.clone())),
            None,
        )?;
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(self.row_ids.clone())),
                Arc::new(neighbors),
            ],
        )?)
    }

    pub fn try_from_batch(batch: &RecordBatch) -> Result<Self> {
        let row_ids = batch
            .column_by_name(ROW_ID_COLUMN)
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    ROW_ID_COLUMN,
                    "Vamana partition file is missing the row id column".to_string(),
                )
            })?
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    ROW_ID_COLUMN,
                    "Vamana row id column is not UInt64".to_string(),
                )
            })?
            .values()
            .to_vec();

        let neighbors_column = batch.column_by_name(NEIGHBORS_COLUMN).ok_or_else(|| {
            Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                "Vamana partition file is missing the neighbours column".to_string(),
            )
        })?;
        let neighbors = neighbors_column.as_fixed_size_list_opt().ok_or_else(|| {
            Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!(
                    "Vamana neighbours column has type {}, expected a fixed size list",
                    neighbors_column.data_type()
                ),
            )
        })?;
        let max_degree = u32::try_from(neighbors.value_length()).map_err(|_| {
            Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!(
                    "Vamana neighbours column has a negative width {}",
                    neighbors.value_length()
                ),
            )
        })?;

        if row_ids.len() != neighbors.len() {
            return Err(Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!(
                    "Vamana partition file has {} row ids but {} adjacency rows",
                    row_ids.len(),
                    neighbors.len()
                ),
            ));
        }

        Ok(Self {
            max_degree,
            row_ids,
            neighbors: neighbors
                .values()
                .as_primitive_opt::<UInt32Type>()
                .ok_or_else(|| {
                    Error::corrupt_file_named(
                        NEIGHBORS_COLUMN,
                        "Vamana neighbour ids are not UInt32".to_string(),
                    )
                })?
                .values()
                .to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph(max_degree: u32) -> PartitionGraph {
        PartitionGraph::try_new(
            max_degree,
            vec![100, 200, 300, 400],
            vec![vec![1, 2], vec![0], vec![0, 1, 3], vec![]],
        )
        .unwrap()
    }

    #[test]
    fn neighbours_are_trimmed_at_the_padding() {
        let graph = sample_graph(4);
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.neighbors(1), &[0]);
        assert_eq!(graph.neighbors(2), &[0, 1, 3]);
        assert_eq!(graph.neighbors(3), &[] as &[u32]);
    }

    #[test]
    fn a_saturated_vertex_uses_every_slot() {
        let graph = PartitionGraph::try_new(2, vec![7, 8], vec![vec![1, 0], vec![0, 1]]).unwrap();
        assert_eq!(graph.neighbors(0), &[1, 0]);
        assert_eq!(graph.neighbors(1), &[0, 1]);
    }

    #[test]
    fn batch_round_trip_preserves_the_graph() {
        let graph = sample_graph(4);
        let restored = PartitionGraph::try_from_batch(&graph.to_batch().unwrap()).unwrap();
        assert_eq!(restored, graph);
    }

    #[test]
    fn dangling_edges_are_rejected() {
        let error = PartitionGraph::try_new(4, vec![1, 2], vec![vec![5], vec![]]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("local id 5"), "{error}");
    }

    #[test]
    fn overfull_vertices_are_rejected() {
        let error = PartitionGraph::try_new(2, vec![1, 2, 3], vec![vec![1, 2, 0], vec![], vec![]])
            .unwrap_err();
        assert!(error.to_string().contains("exceeds max_degree"), "{error}");
    }

    #[test]
    fn mismatched_row_id_and_adjacency_counts_are_rejected() {
        let error = PartitionGraph::try_new(4, vec![1, 2], vec![vec![]]).unwrap_err();
        assert!(error.to_string().contains("adjacency lists"), "{error}");
    }
}
