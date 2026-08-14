// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One partition's graph, in memory, in the shape it has on disk.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};

use crate::format::{
    MAX_DEGREE, MAX_PARTITION_ROWS, NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, VECTOR_COLUMN,
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
        if max_degree == 0 || max_degree > MAX_DEGREE {
            return Err(Error::invalid_input(format!(
                "Vamana max_degree must be between 1 and {MAX_DEGREE}, got {max_degree}"
            )));
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
            check_adjacency(local_id as u32, out_edges, num_rows, max_degree)?;
            neighbors[local_id * width..local_id * width + out_edges.len()]
                .copy_from_slice(out_edges);
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

    /// Append vertices with no out-edges yet, leaving every existing vertex and
    /// its local id exactly where it was.
    ///
    /// What insertion starts from. The existing ids have to survive because the
    /// edges already in the graph are written in them: renumbering here would
    /// mean rewriting every neighbour list in the partition to say the same
    /// thing.
    pub fn extend(&mut self, row_ids: &[u64]) -> Result<()> {
        let total = self
            .row_ids
            .len()
            .checked_add(row_ids.len())
            .filter(|total| *total <= MAX_PARTITION_ROWS as usize)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Vamana cannot add {} vertices to a partition of {}, exceeding the \
                     addressable maximum {MAX_PARTITION_ROWS}",
                    row_ids.len(),
                    self.row_ids.len()
                ))
            })?;
        self.row_ids.extend_from_slice(row_ids);
        self.neighbors
            .resize(total * self.max_degree as usize, NO_NEIGHBOR);
        Ok(())
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
    ///
    /// Fallible for the same reason [`Partition::vector`] is: local ids arrive
    /// from `__neighbors`, which is read off disk, so an id past the end is a
    /// corrupt file rather than a caller's mistake and must not be an index out
    /// of bounds. `Result` rather than `Option` because every caller of this one
    /// wants the same message, where `vector`'s callers decide for themselves.
    ///
    /// Reported as such, and not as bad input: the one caller that passes an id
    /// of its own rather than one read out of the file is a build, whose ids are
    /// its own loop bounds.
    pub fn neighbors(&self, local_id: u32) -> Result<&[u32]> {
        let slots = self.slots(local_id).ok_or_else(|| {
            Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!(
                    "Vamana vertex {local_id} is outside a partition of {} vertices",
                    self.len()
                ),
            )
        })?;
        let degree = slots
            .iter()
            .position(|neighbor| *neighbor == NO_NEIGHBOR)
            .unwrap_or(slots.len());
        Ok(&slots[..degree])
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
        check_adjacency(local_id, neighbors, num_rows, self.max_degree)?;

        let width = self.max_degree as usize;
        let start = local_id as usize * width;
        self.neighbors[start..start + neighbors.len()].copy_from_slice(neighbors);
        self.neighbors[start + neighbors.len()..start + width].fill(NO_NEIGHBOR);
        Ok(())
    }

    /// How many vertices a walk from `entry_point` can reach along out-edges.
    ///
    /// Not a search: no distances and no order, only whether the graph is in one
    /// piece. Consolidation asks because the one-hop repair it runs guarantees
    /// that no *edge* dangles and not that the graph stays connected, and a
    /// partition that came apart can only ever answer from the island holding
    /// its entry point. Asking costs `len * max_degree` pointer chases against
    /// the `len * max_degree` *distances* the repair itself spends, so the check
    /// disappears next to the thing it checks.
    ///
    /// Every id in `neighbors` is in range for anything built through this
    /// type's constructors or read through [`Partition::try_from_batch`], both
    /// of which refuse one that is not - the same invariant a graph walk
    /// already indexes the visit marks with.
    pub fn reachable_from(&self, entry_point: u32) -> Result<usize> {
        let mut seen = vec![false; self.len()];
        let Some(start) = seen.get_mut(entry_point as usize) else {
            return Err(Error::invalid_input(format!(
                "Vamana cannot walk from vertex {entry_point} of a partition of {} vertices",
                self.len()
            )));
        };
        *start = true;

        let mut reached = 1;
        let mut frontier = vec![entry_point];
        while let Some(vertex) = frontier.pop() {
            for neighbor in self.neighbors(vertex)? {
                if !seen[*neighbor as usize] {
                    seen[*neighbor as usize] = true;
                    reached += 1;
                    frontier.push(*neighbor);
                }
            }
        }
        Ok(reached)
    }

    fn slots(&self, local_id: u32) -> Option<&[u32]> {
        let width = self.max_degree as usize;
        let start = (local_id as usize).checked_mul(width)?;
        self.neighbors.get(start..start.checked_add(width)?)
    }
}

/// What one vertex's trimmed out-edge list must satisfy, for every constructor.
///
/// Shared so that the two ways of building a graph cannot disagree: whichever
/// one a caller reaches, an edge that leaves the partition, points at its own
/// vertex or repeats is refused.
fn check_adjacency(
    local_id: u32,
    neighbors: &[u32],
    num_rows: usize,
    max_degree: u32,
) -> Result<()> {
    if neighbors.len() > max_degree as usize {
        return Err(Error::invalid_input(format!(
            "Vamana vertex {local_id} has degree {} which exceeds max_degree {max_degree}",
            neighbors.len()
        )));
    }
    for (position, neighbor) in neighbors.iter().enumerate() {
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
        // Quadratic rather than sorted: `max_degree` is tens of slots, and this
        // runs beside a prune that spends `pool * max_degree` distances on the
        // same vertex, so the scan is free where the sort's allocation is not.
        if neighbors[..position].contains(neighbor) {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {local_id} has a duplicate out-edge {neighbor}"
            )));
        }
    }
    Ok(())
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
    ///
    /// `None` when `local_id` is not a vertex of this partition. Local ids reach
    /// this method from `__neighbors`, which is read off disk, so an id past the
    /// end is a corrupt file rather than a caller's mistake and must not be an
    /// index out of bounds.
    pub fn vector(&self, local_id: u32) -> Option<&[f32]> {
        let dim = self.dimension() as usize;
        let start = (local_id as usize).checked_mul(dim)?;
        self.values().get(start..start.checked_add(dim)?)
    }

    /// Every vector end to end, `dimension` values per vertex.
    pub fn values(&self) -> &[f32] {
        self.vectors.values().as_primitive::<Float32Type>().values()
    }

    pub fn into_parts(self) -> (PartitionGraph, FixedSizeListArray) {
        (self.graph, self.vectors)
    }

    /// The batch this partition is written as.
    ///
    /// `codes` is passed in rather than held on the partition because a code is
    /// a projection of a vector taken at write time: every maintenance pass
    /// moves vertices between local ids, and none of them has to move a code
    /// with one when there is no code to move. See [`crate::codes`].
    pub fn to_batch(&self, codes: Option<&FixedSizeListArray>) -> Result<RecordBatch> {
        let stride = codes
            .map(|codes| {
                u32::try_from(codes.value_length()).map_err(|_| {
                    Error::invalid_input(format!(
                        "Vamana codes have a negative stride {}",
                        codes.value_length()
                    ))
                })
            })
            .transpose()?;
        if let Some(codes) = codes
            && codes.len() != self.len()
        {
            return Err(Error::invalid_input(format!(
                "Vamana partition has {} vertices but {} codes",
                self.len(),
                codes.len()
            )));
        }
        let schema = Arc::new(partition_schema(
            self.graph.max_degree,
            self.dimension(),
            stride,
        )?);
        // Built to the shape `partition_schema` gives `__neighbors` rather than
        // read back out of it: the values are a `u32` array either way, so
        // matching the schema would buy nothing but an impossible arm to panic
        // on. `RecordBatch::try_new` below is what holds the two together.
        let neighbors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            self.graph.max_degree as i32,
            Arc::new(UInt32Array::from(self.graph.neighbors.clone())),
            None,
        )?;
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(self.graph.row_ids.clone())),
            Arc::new(neighbors),
            Arc::new(self.vectors.clone()),
        ];
        if let Some(codes) = codes {
            columns.push(Arc::new(codes.clone()));
        }
        Ok(RecordBatch::try_new(schema, columns)?)
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

/// Decode [`ROW_ID_COLUMN`], which a batch may carry on its own.
///
/// Also reached without the rest of the partition: deciding whether a partition
/// holds any deleted row needs this column and nothing else, and the schema
/// keeps the three columns apart so that asking for it alone is a projection.
pub(crate) fn row_ids_from_batch(batch: &RecordBatch) -> Result<Vec<u64>> {
    let column = batch.column_by_name(ROW_ID_COLUMN).ok_or_else(|| {
        Error::corrupt_file_named(
            ROW_ID_COLUMN,
            "Vamana partition file is missing the row id column".to_string(),
        )
    })?;
    // `values()` reads through the null mask, and this is the one column of a
    // partition file that reaches the caller's answer: a null slot would come
    // back as row address 0, a real live row of fragment 0, indistinguishable
    // from a correct answer. The segment table guards its own columns the same
    // way, and the file's schema is never checked against `partition_schema`,
    // so declaring the field non-nullable on write buys nothing here.
    if column.null_count() != 0 {
        return Err(Error::corrupt_file_named(
            ROW_ID_COLUMN,
            format!("Vamana partition column {ROW_ID_COLUMN} holds nulls"),
        ));
    }
    Ok(column
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| {
            Error::corrupt_file_named(ROW_ID_COLUMN, "Vamana row id column is not UInt64")
        })?
        .values()
        .to_vec())
}

fn graph_from_batch(batch: &RecordBatch) -> Result<PartitionGraph> {
    let row_ids = row_ids_from_batch(batch)?;

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
    if max_degree == 0 {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            "Vamana neighbours column has zero width".to_string(),
        ));
    }
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
    if neighbors.null_count() != 0 || neighbors.values().null_count() != 0 {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            "Vamana neighbours column holds nulls".to_string(),
        ));
    }

    let slots = neighbors
        .values()
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| {
            Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                "Vamana neighbour ids are not UInt32".to_string(),
            )
        })?
        .values();

    // Everything below is what `try_new` enforces on the write path. It has to
    // be enforced here too and cannot be delegated to it, because the sentinel
    // padding is legal on disk and `try_new` takes trimmed lists. Without these
    // checks an out-of-range id read off disk indexes straight into the visit
    // marks and the vector buffer during a search, so a single flipped byte in
    // a partition file panics the process instead of being reported.
    let num_rows = row_ids.len();
    let width = max_degree as usize;
    // `NO_NEIGHBOR` is the top id, so a partition that reached it would have a
    // vertex whose id reads back as padding. Unreachable through this crate's
    // own writer - the segment table counts rows in a `u32` and is checked
    // against this file - but `try_from_batch` is public and takes a batch.
    if num_rows as u64 > MAX_PARTITION_ROWS as u64 {
        return Err(Error::corrupt_file_named(
            ROW_ID_COLUMN,
            format!(
                "Vamana partition file holds {num_rows} rows, exceeding the addressable \
                 maximum {MAX_PARTITION_ROWS}"
            ),
        ));
    }
    // Arrow already guarantees `values.len() == len * size`, but the slicing
    // below is what keeps the search in bounds, so it is checked rather than
    // assumed.
    let expected = num_rows.checked_mul(width).ok_or_else(|| {
        Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            format!("Vamana partition claims {num_rows} rows of width {width}"),
        )
    })?;
    if slots.len() != expected {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            format!(
                "Vamana adjacency holds {} ids, expected {expected} for {num_rows} vertices of \
                 width {width}",
                slots.len()
            ),
        ));
    }
    for (local_id, out_edges) in slots.chunks_exact(width).enumerate() {
        checked_neighbors(out_edges, local_id as u32, num_rows)?;
    }
    // Duplicate out-edges are deliberately not checked here, unlike on the write
    // path. A repeat is harmless to a walk - `SearchScratch` marks a vertex the
    // first time and skips the second - while the check would cost
    // `max_degree^2` per vertex on every partition of every query.

    Ok(PartitionGraph {
        max_degree,
        row_ids,
        neighbors: slots.to_vec(),
    })
}

/// One vertex's adjacency slots, trimmed at the padding and checked.
///
/// Shared between the two ways a walk can get at an edge list, because the
/// checks are what stand between a flipped byte on disk and a panic: an id past
/// the end of the partition indexes straight into a walk's visit marks. A whole
/// partition runs this over every vertex on the way in; a lazy walk runs it over
/// the vertices it fetches, as it fetches them, and no other reader of
/// [`NEIGHBORS_COLUMN`] exists.
///
/// `local_id` is the vertex the slots belong to, which the caller knows and the
/// slots do not: without it a self-edge is unrecognisable.
pub(crate) fn checked_neighbors(slots: &[u32], local_id: u32, num_rows: usize) -> Result<&[u32]> {
    // The padding is a suffix, because a vertex's degree is the index of its
    // first sentinel. An id sitting after one is not read at all: the vertex
    // silently becomes a dead end, and a dead-end medoid reduces its whole
    // partition to a single answer.
    let mut degree = None;
    for (position, neighbor) in slots.iter().enumerate() {
        if *neighbor == NO_NEIGHBOR {
            degree.get_or_insert(position);
            continue;
        }
        if degree.is_some() {
            return Err(Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!(
                    "Vamana vertex {local_id} holds neighbour {neighbor} after its padding, so \
                     its degree cannot be read"
                ),
            ));
        }
        if *neighbor as usize >= num_rows {
            return Err(Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!(
                    "Vamana vertex {local_id} points at local id {neighbor}, but the partition \
                     holds only {num_rows} vertices"
                ),
            ));
        }
        if *neighbor == local_id {
            return Err(Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                format!("Vamana vertex {local_id} points at itself"),
            ));
        }
    }
    Ok(&slots[..degree.unwrap_or(slots.len())])
}

/// [`NEIGHBORS_COLUMN`] of a batch of rows read on their own, as one flat slice.
///
/// `max_degree` comes from the segment rather than from the column, which is the
/// opposite of what [`graph_from_batch`] does and is the point: a lazy walk
/// never holds a whole partition, so nothing else is in a position to notice a
/// file whose stride disagrees with the segment that lists it.
pub(crate) fn neighbor_slots(batch: &RecordBatch, max_degree: u32) -> Result<&[u32]> {
    let neighbors = fixed_size_list(batch, NEIGHBORS_COLUMN)?;
    if neighbors.value_length() != max_degree as i32 {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            format!(
                "Vamana neighbours column is {} slots wide but its segment declares max_degree \
                 {max_degree}",
                neighbors.value_length()
            ),
        ));
    }
    if neighbors.null_count() != 0 || neighbors.values().null_count() != 0 {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            "Vamana neighbours column holds nulls".to_string(),
        ));
    }
    let slots = neighbors
        .values()
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| {
            Error::corrupt_file_named(
                NEIGHBORS_COLUMN,
                "Vamana neighbour ids are not UInt32".to_string(),
            )
        })?
        .values();
    let expected = neighbors.len() * max_degree as usize;
    if slots.len() != expected {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            format!(
                "Vamana adjacency holds {} ids, expected {expected} for {} vertices of width \
                 {max_degree}",
                slots.len(),
                neighbors.len()
            ),
        ));
    }
    Ok(slots)
}

/// [`VECTOR_COLUMN`] of a batch of rows read on their own.
///
/// The width against the segment for the same reason as [`neighbor_slots`], and
/// the nulls because `flat_storage` reads values straight through a validity
/// mask - a null vector would come back as a distance of zero, the nearest
/// answer there is.
pub(crate) fn vectors_of(batch: &RecordBatch, dimension: u32) -> Result<FixedSizeListArray> {
    let vectors = vectors_from_batch(batch)?;
    if vectors.value_length() != dimension as i32 {
        return Err(Error::corrupt_file_named(
            VECTOR_COLUMN,
            format!(
                "Vamana vector column is {} wide but its segment declares dimension {dimension}",
                vectors.value_length()
            ),
        ));
    }
    if vectors.null_count() != 0 || vectors.values().null_count() != 0 {
        return Err(Error::corrupt_file_named(
            VECTOR_COLUMN,
            "Vamana vector column holds nulls".to_string(),
        ));
    }
    Ok(vectors)
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
    use arrow_schema::Schema as ArrowSchema;

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
        assert_eq!(graph.neighbors(0).unwrap(), &[1, 2]);
        assert_eq!(graph.neighbors(1).unwrap(), &[0]);
        assert_eq!(graph.neighbors(2).unwrap(), &[0, 1, 3]);
        assert_eq!(graph.neighbors(3).unwrap(), &[] as &[u32]);
    }

    #[test]
    fn a_saturated_vertex_uses_every_slot() {
        let graph =
            PartitionGraph::try_new(2, vec![7, 8, 9], vec![vec![1, 2], vec![2, 0], vec![0, 1]])
                .unwrap();
        assert_eq!(graph.neighbors(0).unwrap(), &[1, 2]);
        assert_eq!(graph.neighbors(1).unwrap(), &[2, 0]);
        assert_eq!(graph.neighbors(2).unwrap(), &[0, 1]);
    }

    /// The only mutator the builder has. Its whole contract is that a shortened
    /// list moves nothing else, which is the reason the width is fixed at all.
    #[test]
    fn set_neighbors_rewrites_one_vertex_and_repads_it() {
        let mut graph = sample_graph(4);
        let untouched = graph.neighbors(2).unwrap().to_vec();

        graph.set_neighbors(0, &[3, 1, 2]).unwrap();
        assert_eq!(graph.neighbors(0).unwrap(), &[3, 1, 2]);
        graph.set_neighbors(0, &[1]).unwrap();
        assert_eq!(
            graph.neighbors(0).unwrap(),
            &[1],
            "the tail of a shortened list was not re-padded, so a stale edge survived"
        );
        assert_eq!(graph.neighbors(2).unwrap(), untouched.as_slice());
    }

    #[test]
    fn set_neighbors_refuses_what_it_cannot_store() {
        let mut graph = sample_graph(4);
        for (neighbors, expected) in [
            (vec![0u32], "points at itself"),
            (vec![9], "local id 9"),
            (vec![1, 2, 3, 1, 2], "exceeds max_degree"),
        ] {
            let error = graph.set_neighbors(0, &neighbors).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains(expected), "{error}");
        }
        let error = graph.set_neighbors(9, &[1]).unwrap_err();
        assert!(error.to_string().contains("outside a partition"), "{error}");
    }

    #[test]
    fn a_self_edge_is_rejected() {
        let error = PartitionGraph::try_new(2, vec![7, 8], vec![vec![0], vec![]]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("points at itself"), "{error}");
    }

    #[test]
    fn batch_round_trip_preserves_the_partition() {
        let partition = sample_partition(4);
        let restored = Partition::try_from_batch(&partition.to_batch(None).unwrap()).unwrap();
        assert_eq!(restored, partition);
    }

    #[test]
    fn a_vertex_vector_is_its_own_slice() {
        let partition = sample_partition(4);
        assert_eq!(partition.dimension(), DIMENSION as u32);
        assert_eq!(partition.vector(0), Some([0.0, 1.0, 2.0].as_slice()));
        assert_eq!(partition.vector(2), Some([6.0, 7.0, 8.0].as_slice()));
    }

    /// Local ids come out of `__neighbors`, so one past the end is a corrupt
    /// file arriving at a public method, not a caller slipping. Neither the
    /// vector nor the edges of such an id may be an index out of bounds.
    #[test]
    fn a_vertex_beyond_the_partition_has_neither_vector_nor_edges() {
        let partition = sample_partition(4);
        assert_eq!(partition.len(), 4);
        for local_id in [4, u32::MAX] {
            assert!(partition.vector(local_id).is_none());
            let error = partition.graph().neighbors(local_id).unwrap_err();
            assert!(error.to_string().contains("outside a partition"), "{error}");
        }
    }

    /// Rebuild a partition's batch with one adjacency slot overwritten.
    fn with_slot(partition: &Partition, slot: usize, value: u32) -> RecordBatch {
        let batch = partition.to_batch(None).unwrap();
        let neighbors = batch[NEIGHBORS_COLUMN].as_fixed_size_list();
        let mut values = neighbors
            .values()
            .as_primitive::<UInt32Type>()
            .values()
            .to_vec();
        values[slot] = value;
        let patched = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            neighbors.value_length(),
            Arc::new(UInt32Array::from(values)),
            None,
        )
        .unwrap();
        RecordBatch::try_new(
            batch.schema(),
            vec![
                batch.column(0).clone(),
                Arc::new(patched),
                batch.column(2).clone(),
            ],
        )
        .unwrap()
    }

    /// The read path has to enforce what the write path does. An id past the end
    /// of the partition indexes straight into the visit marks during a search, so
    /// without this check a single flipped byte panics the process.
    #[test]
    fn an_out_of_range_edge_read_back_is_rejected() {
        let partition = sample_partition(4);
        let error = Partition::try_from_batch(&with_slot(&partition, 0, 99)).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("local id 99"), "{error}");
    }

    /// The sentinel is legal, but only as a suffix: a degree is the index of the
    /// first sentinel, so an id behind one is never read. The vertex becomes a
    /// silent dead end, and a dead-end medoid answers its whole partition with
    /// one row.
    ///
    /// Vertex 0 holds `[1, 2, pad, pad]`, so blanking its first slot leaves the
    /// edge to vertex 2 stranded behind the padding; shortening vertex 2's list
    /// from three edges to two is the same edit made legally.
    #[test]
    fn an_edge_behind_the_padding_is_rejected() {
        let partition = sample_partition(4);
        let error = Partition::try_from_batch(&with_slot(&partition, 0, NO_NEIGHBOR)).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("after its padding"), "{error}");

        let restored = Partition::try_from_batch(&with_slot(&partition, 10, NO_NEIGHBOR)).unwrap();
        assert_eq!(restored.graph().neighbors(2).unwrap(), &[0, 1]);
    }

    /// `try_new` refuses a self-edge, so the read path has to as well - the two
    /// constructors describing different graphs is the whole class of bug the
    /// checks in `graph_from_batch` exist for.
    #[test]
    fn a_self_edge_read_back_is_rejected() {
        let partition = sample_partition(4);
        let error = Partition::try_from_batch(&with_slot(&partition, 0, 0)).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("points at itself"), "{error}");
    }

    /// The row id column is the only one that reaches the caller's answer, and
    /// `values()` reads straight through the null mask. A null slot holds 0,
    /// which is a perfectly resolvable address - row 0 of fragment 0 - so the
    /// answer would name a real, live, wrong row.
    #[test]
    fn a_null_row_id_read_back_is_rejected() {
        let partition = sample_partition(4);
        let batch = partition.to_batch(None).unwrap();
        let mut row_ids = batch[ROW_ID_COLUMN]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>();
        row_ids[1] = None;
        let fields = batch
            .schema()
            .fields()
            .iter()
            .map(|field| {
                if field.name() == ROW_ID_COLUMN {
                    Arc::new(Field::new(ROW_ID_COLUMN, DataType::UInt64, true))
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        let holed = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(fields)),
            vec![
                Arc::new(UInt64Array::from(row_ids)),
                batch.column(1).clone(),
                batch.column(2).clone(),
            ],
        )
        .unwrap();

        let error = Partition::try_from_batch(&holed).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("holds nulls"), "{error}");
    }

    /// The neighbour column is read through `values()` as well, and a null slot
    /// reads back as 0 - a perfectly ordinary local id, so a walk would follow
    /// the edge to vertex 0 of the partition and never know. Nulls are guarded
    /// on both levels because either one can carry them: the list may be null,
    /// or a slot inside it may.
    #[test]
    fn a_null_neighbour_read_back_is_rejected() {
        let partition = sample_partition(4);
        let batch = partition.to_batch(None).unwrap();
        let width = partition.graph().max_degree() as i32;

        for (what, list_nulls, slot_nulls) in
            [("a null slot", false, true), ("a null list", true, false)]
        {
            let mut slots = batch[NEIGHBORS_COLUMN]
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>();
            if slot_nulls {
                slots[1] = None;
            }
            let lists = if list_nulls {
                Some(vec![true, false, true, true].into())
            } else {
                None
            };
            let holed = RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![
                    batch.schema().field(0).clone(),
                    Field::new(
                        NEIGHBORS_COLUMN,
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::UInt32, true)),
                            width,
                        ),
                        true,
                    ),
                    batch.schema().field(2).clone(),
                ])),
                vec![
                    batch.column(0).clone(),
                    Arc::new(
                        FixedSizeListArray::try_new(
                            Arc::new(Field::new("item", DataType::UInt32, true)),
                            width,
                            Arc::new(UInt32Array::from(slots)),
                            lists,
                        )
                        .unwrap(),
                    ),
                    batch.column(2).clone(),
                ],
            )
            .unwrap();

            let error = Partition::try_from_batch(&holed).unwrap_err();
            assert!(
                matches!(error, Error::CorruptFile { .. }),
                "{what}: {error}"
            );
            assert!(error.to_string().contains("holds nulls"), "{what}: {error}");
        }
    }

    /// A repeated out-edge was only a `debug_assert`, so a release build took
    /// it, and both constructors then reported a degree the walk cannot deliver.
    #[test]
    fn duplicate_out_edges_are_rejected() {
        let error = PartitionGraph::try_new(4, vec![1, 2, 3], vec![vec![2, 2], vec![], vec![]])
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("duplicate out-edge"), "{error}");

        let mut graph = sample_graph(4);
        let error = graph.set_neighbors(0, &[1, 2, 1]).unwrap_err();
        assert!(error.to_string().contains("duplicate out-edge"), "{error}");
    }

    /// A lazy walk holds no [`Partition`], so `check_partition_shape` never runs
    /// for it and these two accessors are the only place a partition file whose
    /// stride disagrees with its segment can be caught. Without the check the
    /// walk would read a neighbour list striding by the wrong number of slots
    /// and follow edges assembled out of two vertices' halves.
    #[test]
    fn a_column_disagreeing_with_the_segment_is_rejected() {
        let partition = sample_partition(4);
        let batch = partition.to_batch(None).unwrap();

        assert_eq!(
            neighbor_slots(&batch, 4).unwrap().len(),
            4 * partition.len()
        );
        let error = neighbor_slots(&batch, 8).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("max_degree 8"), "{error}");

        assert_eq!(
            vectors_of(&batch, DIMENSION as u32).unwrap().len(),
            partition.len()
        );
        let error = vectors_of(&batch, DIMENSION as u32 + 1).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("dimension 4"), "{error}");
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
        let batch = partition.to_batch(None).unwrap();
        assert_eq!(
            batch.schema().as_ref(),
            &partition_schema(4, DIMENSION as u32, None).unwrap()
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
