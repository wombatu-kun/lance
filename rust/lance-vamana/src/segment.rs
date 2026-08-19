// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What `index.idx` says about a segment.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{Array, RecordBatch, StringArray, UInt32Array};
use lance_core::{Error, Result};
use lance_index::vector::ivf::storage::IvfModel;

use crate::format::{
    FILE_COLUMN, IndexMetadata, MEDOID_COLUMN, NUM_ROWS_COLUMN, PARTITION_ID_COLUMN, index_schema,
};

/// One non-empty partition of a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionEntry {
    pub partition_id: u32,
    /// Local id of the vertex a search starts from, recomputed on every
    /// consolidation: after deletions the old entry point may be gone.
    pub medoid: u32,
    pub num_rows: u32,
    /// File name within the segment directory, never a path.
    ///
    /// Stored rather than derived from `partition_id` so that a reader can find
    /// the partitions without knowing this crate's naming convention.
    pub file: String,
}

/// The contents of `index.idx`: segment-wide parameters, the IVF routing model
/// and the partition table.
///
/// Not to be confused with a Lance manifest; this describes one index segment,
/// which Lance itself treats as an opaque directory of files.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentManifest {
    metadata: IndexMetadata,
    ivf: IvfModel,
    /// Sorted by `partition_id`, and holding only partitions that have vertices.
    partitions: Vec<PartitionEntry>,
}

impl SegmentManifest {
    pub fn try_new(
        metadata: IndexMetadata,
        ivf: IvfModel,
        partitions: Vec<PartitionEntry>,
    ) -> Result<Self> {
        // Lance packs every partition into one file, so its own `IvfModel`
        // doubles as a row-count table. Ours does not: the partition table is
        // the only record of what a partition holds, and a model arriving with
        // that half filled in would be a second answer to the same question.
        if !ivf.offsets.is_empty() || !ivf.lengths.is_empty() {
            return Err(Error::invalid_input(format!(
                "Vamana takes an IVF model for routing only, but this one carries {} offsets and \
                 {} lengths; partition sizes belong to the partition table",
                ivf.offsets.len(),
                ivf.lengths.len()
            )));
        }

        if let Some(centroids) = ivf.centroids.as_ref() {
            let dimension = u32::try_from(centroids.value_length()).unwrap_or(u32::MAX);
            if dimension != metadata.dimension {
                return Err(Error::invalid_input(format!(
                    "Vamana index metadata declares dimension {} but its IVF centroids have \
                     dimension {dimension}",
                    metadata.dimension
                )));
            }
        }

        let mut previous: Option<u32> = None;
        for entry in &partitions {
            if previous.is_some_and(|last| last >= entry.partition_id) {
                return Err(Error::invalid_input(format!(
                    "Vamana partition table is not sorted: partition {} follows {}",
                    entry.partition_id,
                    previous.unwrap_or_default()
                )));
            }
            previous = Some(entry.partition_id);

            if entry.num_rows == 0 {
                return Err(Error::invalid_input(format!(
                    "Vamana partition {} is listed with no rows; empty partitions are omitted",
                    entry.partition_id
                )));
            }
            if entry.medoid >= entry.num_rows {
                return Err(Error::invalid_input(format!(
                    "Vamana partition {} has medoid {} but holds only {} vertices",
                    entry.partition_id, entry.medoid, entry.num_rows
                )));
            }
            if entry.file.is_empty() || entry.file.contains('/') {
                return Err(Error::invalid_input(format!(
                    "Vamana partition {} names file {:?}, which is not a plain file name",
                    entry.partition_id, entry.file
                )));
            }
            if ivf.centroids.is_some() && entry.partition_id as usize >= ivf.num_partitions() {
                return Err(Error::invalid_input(format!(
                    "Vamana partition table names partition {} but the IVF model has only {} \
                     partitions",
                    entry.partition_id,
                    ivf.num_partitions()
                )));
            }
        }

        Ok(Self {
            metadata,
            ivf,
            partitions,
        })
    }

    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    pub fn ivf(&self) -> &IvfModel {
        &self.ivf
    }

    /// Every non-empty partition, in ascending partition id.
    pub fn partitions(&self) -> &[PartitionEntry] {
        &self.partitions
    }

    /// The entry for `partition_id`, or `None` when that partition is empty.
    pub fn partition(&self, partition_id: u32) -> Option<&PartitionEntry> {
        self.partitions
            .binary_search_by_key(&partition_id, |entry| entry.partition_id)
            .ok()
            .map(|position| &self.partitions[position])
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let partition_ids = self
            .partitions
            .iter()
            .map(|entry| entry.partition_id)
            .collect::<Vec<_>>();
        let medoids = self
            .partitions
            .iter()
            .map(|entry| entry.medoid)
            .collect::<Vec<_>>();
        let num_rows = self
            .partitions
            .iter()
            .map(|entry| entry.num_rows)
            .collect::<Vec<_>>();
        let files = self
            .partitions
            .iter()
            .map(|entry| entry.file.as_str())
            .collect::<Vec<_>>();

        Ok(RecordBatch::try_new(
            Arc::new(index_schema()),
            vec![
                Arc::new(UInt32Array::from(partition_ids)),
                Arc::new(UInt32Array::from(medoids)),
                Arc::new(UInt32Array::from(num_rows)),
                Arc::new(StringArray::from(files)),
            ],
        )?)
    }

    pub fn try_from_batch(
        metadata: IndexMetadata,
        ivf: IvfModel,
        batch: &RecordBatch,
    ) -> Result<Self> {
        let partition_ids = u32_column(batch, PARTITION_ID_COLUMN)?;
        let medoids = u32_column(batch, MEDOID_COLUMN)?;
        let num_rows = u32_column(batch, NUM_ROWS_COLUMN)?;
        let files = batch
            .column_by_name(FILE_COLUMN)
            .ok_or_else(|| missing_column(FILE_COLUMN))?;
        let files = files.as_string_opt::<i32>().ok_or_else(|| {
            Error::corrupt_file_named(
                FILE_COLUMN,
                format!(
                    "Vamana partition table column {FILE_COLUMN} has type {}, expected Utf8",
                    files.data_type()
                ),
            )
        })?;

        let partitions = (0..batch.num_rows())
            .map(|row| PartitionEntry {
                partition_id: partition_ids[row],
                medoid: medoids[row],
                num_rows: num_rows[row],
                file: files.value(row).to_string(),
            })
            .collect();
        Self::try_new(metadata, ivf, partitions)
    }
}

fn missing_column(name: &str) -> Error {
    Error::corrupt_file_named(
        name,
        format!("Vamana partition table is missing column {name}"),
    )
}

fn u32_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a [u32]> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| missing_column(name))?;
    Ok(column
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| {
            Error::corrupt_file_named(
                name,
                format!(
                    "Vamana partition table column {name} has type {}, expected UInt32",
                    column.data_type()
                ),
            )
        })?
        .values())
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::format::{FORMAT_VERSION, RowIdMode, partition_file_name};

    fn metadata(dimension: u32) -> IndexMetadata {
        IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: 32,
            alpha: 1.2,
            dimension,
            distance_type: DistanceType::L2,
            row_id_mode: RowIdMode::Address,
        }
    }

    fn ivf(num_partitions: usize, dimension: usize) -> IvfModel {
        let values = Float32Array::from(vec![0.5; num_partitions * dimension]);
        IvfModel::new(
            FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap(),
            None,
        )
    }

    fn entry(partition_id: u32, num_rows: u32) -> PartitionEntry {
        PartitionEntry {
            partition_id,
            medoid: num_rows / 2,
            num_rows,
            file: partition_file_name(partition_id),
        }
    }

    #[test]
    fn batch_round_trip_preserves_the_table() {
        let manifest = SegmentManifest::try_new(
            metadata(4),
            ivf(8, 4),
            vec![entry(0, 10), entry(3, 7), entry(7, 1)],
        )
        .unwrap();
        let restored =
            SegmentManifest::try_from_batch(metadata(4), ivf(8, 4), &manifest.to_batch().unwrap())
                .unwrap();
        assert_eq!(restored.partitions(), manifest.partitions());
    }

    #[test]
    fn an_absent_partition_is_an_empty_one() {
        let manifest =
            SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![entry(0, 10), entry(7, 1)])
                .unwrap();
        assert_eq!(manifest.partition(7).unwrap().num_rows, 1);
        assert!(manifest.partition(3).is_none());
        assert!(manifest.partition(8).is_none());
    }

    #[test]
    fn an_out_of_order_table_is_rejected() {
        let error =
            SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![entry(3, 5), entry(1, 5)])
                .unwrap_err();
        assert!(error.to_string().contains("not sorted"), "{error}");
    }

    #[test]
    fn a_duplicated_partition_is_rejected() {
        let error =
            SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![entry(3, 5), entry(3, 5)])
                .unwrap_err();
        assert!(error.to_string().contains("not sorted"), "{error}");
    }

    #[test]
    fn an_empty_partition_may_not_be_listed() {
        let error =
            SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![entry(0, 0)]).unwrap_err();
        assert!(error.to_string().contains("empty partitions"), "{error}");
    }

    #[test]
    fn a_medoid_outside_the_partition_is_rejected() {
        let mut broken = entry(0, 4);
        broken.medoid = 4;
        let error = SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![broken]).unwrap_err();
        assert!(error.to_string().contains("medoid 4"), "{error}");
    }

    #[test]
    fn a_file_name_that_escapes_the_segment_is_rejected() {
        let mut broken = entry(0, 4);
        broken.file = "../other/part_00000.idx".to_string();
        let error = SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![broken]).unwrap_err();
        assert!(error.to_string().contains("plain file name"), "{error}");
    }

    #[test]
    fn a_partition_beyond_the_ivf_model_is_rejected() {
        let error =
            SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![entry(8, 3)]).unwrap_err();
        assert!(error.to_string().contains("only 8"), "{error}");
    }

    /// The metadata and the centroids are two records of the same fact, so they
    /// are checked against each other rather than trusted one at a time.
    #[test]
    fn a_dimension_disagreeing_with_the_centroids_is_rejected() {
        let error = SegmentManifest::try_new(metadata(4), ivf(8, 16), vec![]).unwrap_err();
        assert!(error.to_string().contains("dimension 16"), "{error}");
    }

    /// A model trained by Lance arrives with its partition sizes filled in, and
    /// they would then be a second, unmaintained copy of `num_rows`.
    #[test]
    fn an_ivf_model_carrying_partition_sizes_is_rejected() {
        let mut sized = ivf(8, 4);
        sized.lengths = vec![10, 0, 0, 0, 0, 0, 0, 0];
        let error = SegmentManifest::try_new(metadata(4), sized, vec![entry(0, 10)]).unwrap_err();
        assert!(error.to_string().contains("routing only"), "{error}");
    }
}
