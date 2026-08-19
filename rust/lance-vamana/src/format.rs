// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! On-disk shape of a Vamana index segment.
//!
//! A segment directory holds one `index.idx` describing the segment and one
//! file per partition holding that partition's graph.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use lance_core::{Error, Result};
use lance_encoding::constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

/// Name of the file describing a segment.
///
/// Not a free choice: Lance decides whether an index is a vector index or a
/// scalar one by looking for this exact name among the segment's files.
pub const INDEX_FILE_NAME: &str = "index.idx";

/// Id of the IVF partition a row of `index.idx` describes.
pub const PARTITION_ID_COLUMN: &str = "__partition_id";

/// Local id of the vertex a search of that partition starts from.
pub const MEDOID_COLUMN: &str = "__medoid";

/// Number of vertices in that partition.
pub const NUM_ROWS_COLUMN: &str = "__num_rows";

/// Name of that partition's file within the segment directory.
pub const FILE_COLUMN: &str = "__file";

/// Row ids of the vertices, in the space named by [`RowIdMode`].
pub const ROW_ID_COLUMN: &str = "__row_id";

/// Out-edges of each vertex as partition-local ids, padded with [`NO_NEIGHBOR`].
pub const NEIGHBORS_COLUMN: &str = "__neighbors";

/// The vector of each vertex, in the same order as [`NEIGHBORS_COLUMN`].
///
/// A graph walk needs a distance for every candidate it considers, so the
/// vectors have to be reachable at query time. Keeping them in the partition
/// makes the index self-contained - a query reads its own segment and never the
/// dataset's data files - and it is the layout the disk-resident traversal
/// wants: one vertex is one stride of this column plus one stride of
/// [`NEIGHBORS_COLUMN`], with nothing else read.
pub const VECTOR_COLUMN: &str = "__vector";

/// Padding slot in [`NEIGHBORS_COLUMN`].
///
/// A vertex's degree is the index of its first padding slot, so degree is not
/// stored separately. A degree column would cost a second ranged read per
/// vertex, and reading one vertex in one read is the entire reason this layout
/// has a fixed stride.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Highest partition-local id addressable, given [`NO_NEIGHBOR`] takes the top.
pub const MAX_PARTITION_ROWS: u32 = u32::MAX - 1;

pub const FORMAT_VERSION: u32 = 2;

/// Schema metadata key under which [`IndexMetadata`] is stored as JSON.
pub const INDEX_METADATA_KEY: &str = "lance-vamana:index";

/// Schema metadata key holding the index of the global buffer with the IVF model.
///
/// The routing model is a protobuf blob rather than a column because it is read
/// in full or not at all, and a global buffer is exactly one ranged read.
pub const IVF_POSITION_KEY: &str = "lance-vamana:ivf";

/// Which identifier space [`ROW_ID_COLUMN`] is expressed in.
///
/// Lance hands out row addresses by default and stable logical ids when the
/// dataset enables them, and the two are not interchangeable: deletion vectors
/// are always in address space, so a delete list built from them can only be
/// applied to stored ids when the index was built in [`RowIdMode::Address`].
/// Applying it in the wrong space would filter out live rows and return deleted
/// ones, silently. Hence the mode travels with the index and is checked on open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowIdMode {
    /// Fragment id in the high 32 bits, row offset within the fragment in the low 32.
    Address,
    /// A logical id with no relation to fragment layout.
    Stable,
}

/// Segment-wide parameters, stored in the schema metadata of `index.idx`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub format_version: u32,
    /// `R` in the Vamana papers: the fixed width of [`NEIGHBORS_COLUMN`].
    pub max_degree: u32,
    /// Pruning slack. `1.0` reproduces the HNSW diversity heuristic exactly.
    pub alpha: f32,
    pub dimension: u32,
    #[serde(with = "distance_type_as_name")]
    pub distance_type: DistanceType,
    pub row_id_mode: RowIdMode,
}

/// `DistanceType` carries no serde impls, and its `Display` / `TryFrom<&str>`
/// pair is the spelling Lance already persists everywhere else.
mod distance_type_as_name {
    use lance_linalg::distance::DistanceType;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(
        distance_type: &DistanceType,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&distance_type.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<DistanceType, D::Error> {
        let name = String::deserialize(deserializer)?;
        DistanceType::try_from(name.as_str()).map_err(D::Error::custom)
    }
}

impl IndexMetadata {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| {
            Error::invalid_input(format!("failed to serialize Vamana index metadata: {e}"))
        })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let metadata: Self = serde_json::from_str(json).map_err(|e| {
            Error::corrupt_file_named(
                INDEX_METADATA_KEY,
                format!("failed to parse Vamana index metadata: {e}"),
            )
        })?;
        if metadata.format_version != FORMAT_VERSION {
            return Err(Error::not_supported(format!(
                "Vamana index format version {} is not supported by this build (expected {})",
                metadata.format_version, FORMAT_VERSION
            )));
        }
        Ok(metadata)
    }
}

/// Arrow schema of one partition file.
///
/// Both fixed-size-list columns are laid out so that vertex `local_id` sits at
/// `base + local_id * stride` with no read amplification: `max_degree * 4` bytes
/// for `__neighbors`, `dimension * 4` for `__vector`. Two details make that hold
/// and neither is the default:
///
/// - The full-zip encoding is requested explicitly. Left to the heuristic, Lance
///   picks full-zip only once a value reaches 256 bytes - `max_degree >= 64`, or
///   `dimension >= 64` - and quietly falls back to mini-block below that, which
///   reintroduces chunk amplification and destroys the addressing.
/// - Both the column and its item are non-nullable. A null anywhere adds a
///   control word to every value, so the stride stops being a clean multiple.
///
/// The two columns stay separate rather than being interleaved into one wide
/// value because their access patterns differ: consolidation rewrites the edges
/// and not the vectors, and a graph walk reads the vectors and edges but never
/// the row ids. Separate columns are what makes each of those a projection.
pub fn partition_schema(max_degree: u32, dimension: u32) -> Result<Schema> {
    let neighbors = addressable_list(NEIGHBORS_COLUMN, DataType::UInt32, max_degree, "max_degree")?;
    let vector = addressable_list(VECTOR_COLUMN, DataType::Float32, dimension, "dimension")?;
    Ok(Schema::new(vec![
        Field::new(ROW_ID_COLUMN, DataType::UInt64, false),
        neighbors,
        vector,
    ]))
}

/// A non-nullable `FixedSizeList` field that is explicitly full-zip encoded.
fn addressable_list(name: &str, item_type: DataType, width: u32, what: &str) -> Result<Field> {
    if width == 0 {
        return Err(Error::invalid_input(format!(
            "Vamana {what} must be greater than zero"
        )));
    }
    let width = i32::try_from(width).map_err(|_| {
        Error::invalid_input(format!(
            "Vamana {what} {width} exceeds the maximum Arrow list width {}",
            i32::MAX
        ))
    })?;
    Ok(Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", item_type, false)), width),
        false,
    )
    .with_metadata(HashMap::from([(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_FULLZIP.to_string(),
    )])))
}

/// Arrow schema of `index.idx`: one row per *non-empty* partition.
///
/// No column is nullable because an empty partition is not listed at all. It has
/// no vertices, so it has no entry point and no file, and leaving the row out is
/// the only encoding of that which cannot disagree with itself.
pub fn index_schema() -> Schema {
    Schema::new(vec![
        Field::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
        Field::new(MEDOID_COLUMN, DataType::UInt32, false),
        Field::new(NUM_ROWS_COLUMN, DataType::UInt32, false),
        Field::new(FILE_COLUMN, DataType::Utf8, false),
    ])
}

/// Canonical file name of a partition within its segment directory.
pub fn partition_file_name(partition_id: u32) -> String {
    format!("part_{partition_id:05}.idx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_round_trips_through_json() {
        let metadata = IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: 64,
            alpha: 1.2,
            dimension: 128,
            distance_type: DistanceType::Cosine,
            row_id_mode: RowIdMode::Address,
        };
        let parsed = IndexMetadata::from_json(&metadata.to_json().unwrap()).unwrap();
        assert_eq!(parsed, metadata);
    }

    #[test]
    fn metadata_rejects_an_unknown_distance_type() {
        let json = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "max_degree": 64,
            "alpha": 1.2,
            "dimension": 128,
            "distance_type": "manhattan",
            "row_id_mode": "address",
        })
        .to_string();
        let error = IndexMetadata::from_json(&json).unwrap_err();
        assert!(error.to_string().contains("manhattan"), "{error}");
    }

    #[test]
    fn metadata_rejects_a_future_format_version() {
        let json = serde_json::json!({
            "format_version": FORMAT_VERSION + 1,
            "max_degree": 64,
            "alpha": 1.2,
            "dimension": 128,
            "distance_type": "l2",
            "row_id_mode": "address",
        })
        .to_string();
        let error = IndexMetadata::from_json(&json).unwrap_err();
        assert!(
            matches!(error, Error::NotSupported { .. }),
            "unexpected error: {error}"
        );
        assert!(error.to_string().contains("format version"));
    }

    #[test]
    fn partition_schema_requests_fullzip_and_stays_non_nullable() {
        let schema = partition_schema(32, 24).unwrap();
        // Both widths are under the 256-byte threshold at which Lance would pick
        // full-zip unprompted, so both columns depend on the explicit hint.
        for (column, expected_width, expected_item) in [
            (NEIGHBORS_COLUMN, 32, DataType::UInt32),
            (VECTOR_COLUMN, 24, DataType::Float32),
        ] {
            let field = schema.field_with_name(column).unwrap();
            assert!(
                !field.is_nullable(),
                "{column}: a control word would break the stride"
            );
            assert_eq!(
                field.metadata().get(STRUCTURAL_ENCODING_META_KEY),
                Some(&STRUCTURAL_ENCODING_FULLZIP.to_string()),
                "{column}: below 64 the heuristic would choose mini-block on its own"
            );
            match field.data_type() {
                DataType::FixedSizeList(item, width) => {
                    assert_eq!(*width, expected_width, "{column}");
                    assert_eq!(*item.data_type(), expected_item, "{column}");
                    assert!(!item.is_nullable(), "{column}");
                }
                other => panic!("unexpected {column} type: {other}"),
            }
        }
    }

    #[test]
    fn partition_schema_rejects_a_zero_degree() {
        let error = partition_schema(0, 8).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("max_degree"), "{error}");
    }

    #[test]
    fn partition_schema_rejects_a_zero_dimension() {
        let error = partition_schema(32, 0).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("dimension"), "{error}");
    }
}
