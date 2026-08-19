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
    FILE_COLUMN, FORMAT_VERSION, INDEX_FILE_NAME, IndexMetadata, MAX_PARTITION_ROWS, MEDOID_COLUMN,
    NUM_ROWS_COLUMN, PARTITION_ID_COLUMN, index_schema,
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
    /// Stored rather than derived from `partition_id` so that a reader follows
    /// the table instead of a naming convention. Today every writer in this
    /// crate fills it from [`crate::format::partition_file_name`], so the two
    /// always agree and nothing yet exercises the difference.
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
        // The metadata is checked as well as the table, and on the way out as
        // well as in: `SegmentWriter` is public, so without this a caller could
        // write an `index.idx` whose JSON declares one format version while the
        // dataset manifest records another - two records of one number, kept
        // apart on purpose, made to disagree at the source.
        if metadata.format_version != FORMAT_VERSION {
            return Err(Error::invalid_input(format!(
                "Vamana segment declares format version {} but this build reads and writes \
                 version {FORMAT_VERSION}",
                metadata.format_version
            )));
        }
        if metadata.max_degree == 0 {
            return Err(Error::invalid_input(
                "Vamana segment declares max_degree 0, so its vertices could hold no edges"
                    .to_string(),
            ));
        }
        // Not a formality: a centroid tensor of zero width passes
        // `validate_ivf_model`, which checks offsets, lengths and data type but
        // never the shape, and then matches a zero `dimension` here because
        // `0 == 0`. A query of zero length would clear the dimension guard and
        // reach `l2_distance_batch(&[], &[], 0)`, whose `to.len() % dimension`
        // divides by zero and takes the process down.
        if metadata.dimension == 0 {
            return Err(Error::invalid_input(
                "Vamana segment declares dimension 0, which no query could be measured against"
                    .to_string(),
            ));
        }

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

        // Without centroids there is nothing to route with, and every remaining
        // check below would silently evaporate: `IvfModel::num_partitions` falls
        // back to `offsets.len()`, which the rule above has just required to be
        // empty. Worse, `IvfModel::find_partitions` unwraps the centroids, so a
        // segment accepted here would abort the process on its first query.
        let Some(centroids) = ivf.centroids.as_ref() else {
            return Err(Error::invalid_input(
                "Vamana takes an IVF model for routing, and a model without centroids cannot route"
                    .to_string(),
            ));
        };
        let dimension = u32::try_from(centroids.value_length()).unwrap_or(u32::MAX);
        if dimension != metadata.dimension {
            return Err(Error::invalid_input(format!(
                "Vamana index metadata declares dimension {} but its IVF centroids have \
                 dimension {dimension}",
                metadata.dimension
            )));
        }
        let num_partitions = ivf.num_partitions();

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
            // `NO_NEIGHBOR` takes the top local id, and the bound keeps a
            // partition's ids clear of it with one row to spare - the ids of an
            // `n`-row partition stop at `n - 1`, so this refuses one count
            // earlier than it strictly has to. The partition file is checked
            // against this claim on read, so refusing an unmeetable claim here
            // is what keeps that check meaningful.
            if entry.num_rows > MAX_PARTITION_ROWS {
                return Err(Error::invalid_input(format!(
                    "Vamana partition {} claims {} rows, exceeding the addressable maximum {}",
                    entry.partition_id, entry.num_rows, MAX_PARTITION_ROWS
                )));
            }
            if !is_plain_file_name(&entry.file) {
                return Err(Error::invalid_input(format!(
                    "Vamana partition {} names file {:?}, which is not a plain file name",
                    entry.partition_id, entry.file
                )));
            }
            if entry.partition_id as usize >= num_partitions {
                return Err(Error::invalid_input(format!(
                    "Vamana partition table names partition {} but the IVF model has only \
                     {num_partitions} partitions",
                    entry.partition_id
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
        // Symmetrical with `u32_column`: `value()` reads through the null mask,
        // and a null slot's offsets are only equal by convention, so a null file
        // name would read as whatever bytes the offsets happen to bracket.
        if files.null_count() != 0 {
            return Err(Error::corrupt_file_named(
                FILE_COLUMN,
                format!("Vamana partition table column {FILE_COLUMN} holds nulls"),
            ));
        }
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

/// A name a segment may give one of its partition files.
///
/// An allow-list rather than a list of things to reject. The name is joined onto
/// the segment directory and handed to the object store, so the question is not
/// "does it contain a slash" - `..`, `.`, a NUL byte and percent-escapes all
/// answer no while still being something other than a file of this segment.
/// Today the join is made safe by `object_store::path::PathPart` sanitising each
/// segment, which is an implementation detail of a dependency and pins nothing.
///
/// [`INDEX_FILE_NAME`] is excluded because it is the segment's own manifest: a
/// partition claiming it would have the reader open the table as a graph.
fn is_plain_file_name(name: &str) -> bool {
    name != INDEX_FILE_NAME
        && !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
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
    // `values()` ignores the null mask, so a null would read as whatever the
    // buffer happens to hold - a null medoid would silently become vertex 0.
    if column.null_count() != 0 {
        return Err(Error::corrupt_file_named(
            name,
            format!("Vamana partition table column {name} holds nulls"),
        ));
    }
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
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::format::{FORMAT_VERSION, RowIdMode, partition_file_name};

    fn metadata(dimension: u32) -> IndexMetadata {
        IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: 32,
            search_list_size: 64,
            alpha: 1.2,
            dimension,
            distance_type: DistanceType::L2,
            row_id_mode: RowIdMode::Address,
            fragments: vec![0],
            codes: None,
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
    /// A routing model with no centroids cannot route, and every other check in
    /// `try_new` is defined in terms of them - so accepting one would disable the
    /// lot and hand `find_partitions` an unwrap it would abort on.
    #[test]
    fn an_ivf_model_without_centroids_is_rejected() {
        let error = SegmentManifest::try_new(metadata(4), IvfModel::empty(), vec![entry(0, 4)])
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("without centroids"), "{error}");
    }

    /// One valid row, with a null put in one column. The values are chosen so
    /// that dropping a null check does not merely change the error: a null
    /// partition id or medoid reads back as 0, which is a *valid* entry, so the
    /// table would be accepted with a row it never held.
    fn table_with_a_null_in(column: &str) -> RecordBatch {
        let value = |name: &str, valid: u32| (name != column).then_some(valid);
        let nullable = |name: &str, data_type: DataType| Field::new(name, data_type, true);
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                nullable(PARTITION_ID_COLUMN, DataType::UInt32),
                nullable(MEDOID_COLUMN, DataType::UInt32),
                nullable(NUM_ROWS_COLUMN, DataType::UInt32),
                nullable(FILE_COLUMN, DataType::Utf8),
            ])),
            vec![
                Arc::new(UInt32Array::from(vec![value(PARTITION_ID_COLUMN, 0)])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![value(MEDOID_COLUMN, 0)])),
                Arc::new(UInt32Array::from(vec![value(NUM_ROWS_COLUMN, 4)])),
                Arc::new(StringArray::from(vec![
                    (column != FILE_COLUMN).then(|| partition_file_name(0)),
                ])),
            ],
        )
        .unwrap()
    }

    /// `try_from_batch` is handed a batch, not a file, so it cannot lean on
    /// `index_schema` having declared every column non-nullable.
    #[test]
    fn a_null_in_the_partition_table_is_rejected() {
        for column in [
            PARTITION_ID_COLUMN,
            MEDOID_COLUMN,
            NUM_ROWS_COLUMN,
            FILE_COLUMN,
        ] {
            let error = SegmentManifest::try_from_batch(
                metadata(4),
                ivf(8, 4),
                &table_with_a_null_in(column),
            )
            .unwrap_err();
            assert!(
                error.to_string().contains(&format!("{column} holds nulls")),
                "{column}: {error}"
            );
        }
    }

    /// The metadata is the other half of what a segment says about itself, and
    /// `SegmentWriter` will write whatever it is handed. A zero dimension is the
    /// one that costs a crash rather than a wrong answer: a zero-width centroid
    /// tensor clears every cross-check by matching it, and the query that
    /// follows divides by it.
    #[test]
    fn segment_metadata_that_describes_nothing_is_rejected() {
        // Each expectation is a phrase only the guard under test produces. A
        // zero dimension also trips the centroid cross-check one line below, and
        // its message says "dimension 0" too - so matching on that would pass
        // with the guard removed.
        for (metadata, expected) in [
            (
                IndexMetadata {
                    format_version: FORMAT_VERSION + 1,
                    ..metadata(4)
                },
                "this build reads and writes",
            ),
            (
                IndexMetadata {
                    max_degree: 0,
                    ..metadata(4)
                },
                "vertices could hold no edges",
            ),
            (
                IndexMetadata {
                    dimension: 0,
                    ..metadata(4)
                },
                "no query could be measured against",
            ),
        ] {
            let error =
                SegmentManifest::try_new(metadata, ivf(8, 4), vec![entry(0, 4)]).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    /// The file name is joined onto the segment directory and handed to the
    /// object store, so "no slash" is not the question. Every name here reaches
    /// something other than a file of this segment - including `index.idx`,
    /// which would have the reader open the partition table as a graph.
    #[test]
    fn a_file_name_that_is_not_a_plain_name_is_rejected() {
        for name in [
            "..",
            ".",
            "",
            "%2E%2E",
            "a/b",
            "a\\b",
            "a\0b",
            INDEX_FILE_NAME,
        ] {
            let mut broken = entry(0, 4);
            broken.file = name.to_string();
            let error = SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![broken]).unwrap_err();
            assert!(
                error.to_string().contains("plain file name"),
                "{name:?} was accepted: {error}"
            );
        }
        for name in ["part_00000.idx", "p-1_2.bin"] {
            let mut entry = entry(0, 4);
            entry.file = name.to_string();
            SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![entry])
                .unwrap_or_else(|error| panic!("{name:?} was refused: {error}"));
        }
    }

    /// `NO_NEIGHBOR` owns the top local id, so a partition of `u32::MAX` rows
    /// would have a vertex whose id reads back as padding.
    #[test]
    fn a_partition_claiming_more_rows_than_are_addressable_is_rejected() {
        let mut broken = entry(0, 4);
        broken.num_rows = u32::MAX;
        let error = SegmentManifest::try_new(metadata(4), ivf(8, 4), vec![broken]).unwrap_err();
        assert!(error.to_string().contains("addressable maximum"), "{error}");
    }

    #[test]
    fn an_ivf_model_carrying_partition_sizes_is_rejected() {
        let mut sized = ivf(8, 4);
        sized.lengths = vec![10, 0, 0, 0, 0, 0, 0, 0];
        let error = SegmentManifest::try_new(metadata(4), sized, vec![entry(0, 10)]).unwrap_err();
        assert!(error.to_string().contains("routing only"), "{error}");
    }
}
