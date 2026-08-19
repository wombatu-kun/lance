// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One partition's graph, in memory, in the shape it has on disk.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};

use crate::format::{
    MAX_PARTITION_ROWS, NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, VECTOR_COLUMN,
    partition_schema,
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

    /// A partition whose vertices have no out-edges yet.
    ///
    /// This is what a build starts from and mutates in place through
    /// [`Self::set_neighbors`]: the padded layout is already the one a builder
    /// wants, so there is no separate in-memory graph type to convert from.
    pub fn edgeless(max_degree: u32, row_ids: Vec<u64>) -> Result<Self> {
        let adjacency = vec![Vec::new(); row_ids.len()];
        Self::try_new(max_degree, row_ids, adjacency)
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
    pub fn neighbors(&self, local_id: u32) -> &[u32] {
        let slots = self.slots(local_id);
        let degree = slots
            .iter()
            .position(|neighbor| *neighbor == NO_NEIGHBOR)
            .unwrap_or(slots.len());
        &slots[..degree]
    }

    /// Replace the out-edges of `local_id`.
    ///
    /// The vertex keeps its slot, so a prune that shortens a neighbour list
    /// moves nothing else on disk. That is the whole reason the width is fixed.
    pub fn set_neighbors(&mut self, local_id: u32, neighbors: &[u32]) -> Result<()> {
        let num_rows = self.row_ids.len();
        if local_id as usize >= num_rows {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {local_id} is outside a partition of {num_rows} vertices"
            )));
        }
        if neighbors.len() > self.max_degree as usize {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {local_id} was given degree {} which exceeds max_degree {}",
                neighbors.len(),
                self.max_degree
            )));
        }
        for neighbor in neighbors {
            if *neighbor as usize >= num_rows {
                return Err(Error::invalid_input(format!(
                    "Vamana vertex {local_id} points at local id {neighbor}, \
                     but the partition holds only {num_rows} vertices"
                )));
            }
            if *neighbor == local_id {
                return Err(Error::invalid_input(format!(
                    "Vamana vertex {local_id} points at itself"
                )));
            }
        }
        debug_assert!(
            {
                let mut sorted = neighbors.to_vec();
                sorted.sort_unstable();
                sorted.windows(2).all(|pair| pair[0] != pair[1])
            },
            "vertex {local_id} was given a duplicate out-edge: {neighbors:?}"
        );

        let width = self.max_degree as usize;
        let start = local_id as usize * width;
        self.neighbors[start..start + neighbors.len()].copy_from_slice(neighbors);
        self.neighbors[start + neighbors.len()..start + width].fill(NO_NEIGHBOR);
        Ok(())
    }

    fn slots(&self, local_id: u32) -> &[u32] {
        let width = self.max_degree as usize;
        let start = local_id as usize * width;
        &self.neighbors[start..start + width]
    }
}

/// One partition exactly as it is stored: the graph plus the vectors it walks.
///
/// The vectors live beside the edges rather than in the dataset because a graph
/// walk needs a distance for every candidate it considers - fetching them from
/// the dataset would be one `take` per hop. Co-locating them also makes the
/// index self-contained: a query reads its own segment and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    graph: PartitionGraph,
    vectors: FixedSizeListArray,
}

impl Partition {
    pub fn try_new(graph: PartitionGraph, vectors: FixedSizeListArray) -> Result<Self> {
        if vectors.len() != graph.len() {
            return Err(Error::invalid_input(format!(
                "Vamana partition has {} vertices but {} vectors",
                graph.len(),
                vectors.len()
            )));
        }
        if vectors.value_type() != DataType::Float32 {
            return Err(Error::invalid_input(format!(
                "Vamana vectors have item type {}, expected Float32",
                vectors.value_type()
            )));
        }
        if vectors.value_length() <= 0 {
            return Err(Error::invalid_input(format!(
                "Vamana vectors have dimension {}, which must be positive",
                vectors.value_length()
            )));
        }
        // A null on either level adds a control word to every value, and then the
        // stride stops being `dimension * 4` - which is the entire layout.
        if vectors.null_count() != 0 || vectors.values().null_count() != 0 {
            return Err(Error::invalid_input(
                "Vamana vectors must not contain nulls; a null breaks the fixed stride".to_string(),
            ));
        }

        // Normalise the item field so that a partition built from an array whose
        // child is spelled differently still matches `partition_schema` on write.
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            vectors.value_length(),
            vectors.values().clone(),
            None,
        )?;
        Ok(Self { graph, vectors })
    }

    pub fn graph(&self) -> &PartitionGraph {
        &self.graph
    }

    pub fn vectors(&self) -> &FixedSizeListArray {
        &self.vectors
    }

    pub fn len(&self) -> usize {
        self.graph.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    pub fn dimension(&self) -> u32 {
        self.vectors.value_length() as u32
    }

    /// The vector of one vertex, as a slice of the backing buffer.
    pub fn vector(&self, local_id: u32) -> &[f32] {
        let dim = self.dimension() as usize;
        let start = local_id as usize * dim;
        &self.values()[start..start + dim]
    }

    /// Every vector end to end, `dimension` values per vertex.
    pub fn values(&self) -> &[f32] {
        self.vectors.values().as_primitive::<Float32Type>().values()
    }

    pub fn into_parts(self) -> (PartitionGraph, FixedSizeListArray) {
        (self.graph, self.vectors)
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let schema = Arc::new(partition_schema(self.graph.max_degree, self.dimension())?);
        let DataType::FixedSizeList(item, width) =
            schema.field_with_name(NEIGHBORS_COLUMN)?.data_type()
        else {
            unreachable!("partition_schema always produces a fixed size list");
        };
        let neighbors = FixedSizeListArray::try_new(
            item.clone(),
            *width,
            Arc::new(UInt32Array::from(self.graph.neighbors.clone())),
            None,
        )?;
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(self.graph.row_ids.clone())),
                Arc::new(neighbors),
                Arc::new(self.vectors.clone()),
            ],
        )?)
    }

    pub fn try_from_batch(batch: &RecordBatch) -> Result<Self> {
        let graph = graph_from_batch(batch)?;
        let vectors = vectors_from_batch(batch)?;
        if graph.len() != vectors.len() {
            return Err(Error::corrupt_file_named(
                VECTOR_COLUMN,
                format!(
                    "Vamana partition file has {} vertices but {} vectors",
                    graph.len(),
                    vectors.len()
                ),
            ));
        }
        Self::try_new(graph, vectors)
    }
}

fn graph_from_batch(batch: &RecordBatch) -> Result<PartitionGraph> {
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
            Error::corrupt_file_named(ROW_ID_COLUMN, "Vamana row id column is not UInt64")
        })?
        .values()
        .to_vec();

    let neighbors = fixed_size_list(batch, NEIGHBORS_COLUMN)?;
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

    Ok(PartitionGraph {
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

fn vectors_from_batch(batch: &RecordBatch) -> Result<FixedSizeListArray> {
    let vectors = fixed_size_list(batch, VECTOR_COLUMN)?;
    if vectors.value_type() != DataType::Float32 {
        return Err(Error::corrupt_file_named(
            VECTOR_COLUMN,
            format!(
                "Vamana vector column has item type {}, expected Float32",
                vectors.value_type()
            ),
        ));
    }
    Ok(vectors.clone())
}

fn fixed_size_list<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a FixedSizeListArray> {
    let array = batch.column_by_name(column).ok_or_else(|| {
        Error::corrupt_file_named(
            column,
            format!("Vamana partition file is missing the {column} column"),
        )
    })?;
    array.as_fixed_size_list_opt().ok_or_else(|| {
        Error::corrupt_file_named(
            column,
            format!(
                "Vamana {column} column has type {}, expected a fixed size list",
                array.data_type()
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float32Array, Float64Array};

    const DIMENSION: i32 = 3;

    fn sample_graph(max_degree: u32) -> PartitionGraph {
        PartitionGraph::try_new(
            max_degree,
            vec![100, 200, 300, 400],
            vec![vec![1, 2], vec![0], vec![0, 1, 3], vec![]],
        )
        .unwrap()
    }

    fn vectors(rows: usize, dimension: i32, nullable_item: bool) -> FixedSizeListArray {
        let values = (0..rows as i32 * dimension)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, nullable_item)),
            dimension,
            Arc::new(Float32Array::from(values)),
            None,
        )
        .unwrap()
    }

    fn sample_partition(max_degree: u32) -> Partition {
        let graph = sample_graph(max_degree);
        let vectors = vectors(graph.len(), DIMENSION, false);
        Partition::try_new(graph, vectors).unwrap()
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
    fn batch_round_trip_preserves_the_partition() {
        let partition = sample_partition(4);
        let restored = Partition::try_from_batch(&partition.to_batch().unwrap()).unwrap();
        assert_eq!(restored, partition);
    }

    #[test]
    fn a_vertex_vector_is_its_own_slice() {
        let partition = sample_partition(4);
        assert_eq!(partition.dimension(), DIMENSION as u32);
        assert_eq!(partition.vector(0), &[0.0, 1.0, 2.0]);
        assert_eq!(partition.vector(2), &[6.0, 7.0, 8.0]);
    }

    #[test]
    fn a_vector_count_disagreeing_with_the_graph_is_rejected() {
        let graph = sample_graph(4);
        let error = Partition::try_new(graph, vectors(3, DIMENSION, false)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains("4 vertices but 3 vectors"),
            "{error}"
        );
    }

    #[test]
    fn a_vector_column_that_is_not_float32_is_rejected() {
        let graph = sample_graph(4);
        let wide = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float64, false)),
            DIMENSION,
            Arc::new(Float64Array::from(vec![
                0.0;
                graph.len() * DIMENSION as usize
            ])),
            None,
        )
        .unwrap();
        let error = Partition::try_new(graph, wide).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("expected Float32"), "{error}");
    }

    /// A null anywhere adds a control word to every value, so the stride is gone.
    #[test]
    fn null_vectors_are_rejected() {
        let graph = sample_graph(4);
        let mut values = (0..graph.len() * DIMENSION as usize)
            .map(|i| Some(i as f32))
            .collect::<Vec<_>>();
        values[5] = None;
        let holed = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            DIMENSION,
            Arc::new(Float32Array::from(values)),
            None,
        )
        .unwrap();
        let error = Partition::try_new(graph, holed).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("fixed stride"), "{error}");
    }

    /// A partition read back must carry the encoding hint that makes the layout
    /// addressable, or `to_batch` has quietly stopped producing a usable file.
    #[test]
    fn the_batch_schema_is_the_partition_schema() {
        let partition = sample_partition(4);
        let batch = partition.to_batch().unwrap();
        assert_eq!(
            batch.schema().as_ref(),
            &partition_schema(4, DIMENSION as u32).unwrap()
        );
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
