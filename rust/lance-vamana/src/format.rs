// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! On-disk shape of a Vamana index segment.
//!
//! A segment directory holds one `index.idx` describing the segment and one
//! file per partition holding that partition's graph.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef, Schema};
use lance_core::{Error, Result};
use lance_encoding::constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use crate::codes::{CODE_COLUMN, CodeParams};
use crate::entry_points::EntryPointParams;

/// Name of the file describing a segment.
///
/// Not a free choice: Lance decides whether an index is a vector index or a
/// scalar one by looking for this exact name among the segment's files.
pub const INDEX_FILE_NAME: &str = "index.idx";

/// Id of the IVF partition a row of `index.idx` describes.
pub const PARTITION_ID_COLUMN: &str = "__partition_id";

/// Local id of the vertex a search of that partition starts from.
pub const MEDOID_COLUMN: &str = "__medoid";

/// Local ids of that partition's entry points, ascending: where a walk may
/// start instead, at the one nearest its query ([`crate::entry_points`]).
/// Empty for a partition that trained none.
pub const ENTRY_POINTS_COLUMN: &str = "__entry_points";

/// Number of vertices in that partition.
pub const NUM_ROWS_COLUMN: &str = "__num_rows";

/// Name of that partition's file within the segment directory.
pub const FILE_COLUMN: &str = "__file";

/// Row ids of the vertices, in the space named by [`RowIdMode`].
pub const ROW_ID_COLUMN: &str = "__row_id";

/// Out-edges of each vertex as partition-local ids, padded with [`NO_NEIGHBOR`].
pub const NEIGHBORS_COLUMN: &str = "__neighbors";

/// The vector of each vertex, in the same order as [`NEIGHBORS_COLUMN`], in a
/// segment that keeps a copy of them: [`VectorSource::Index`].
///
/// One vertex is one stride of this column, so the re-score that ends a coded
/// walk reads a candidate's vector in one ranged read, and a segment holding it
/// is self-contained: a query reads its own files and never the dataset's,
/// unless it asks to with [`crate::query::SearchParams::rescore_from_dataset`].
/// A segment of [`VectorSource::Dataset`] has no such column and reads the same
/// vectors out of the dataset's data files.
pub const VECTOR_COLUMN: &str = "__vector";

/// Padding slot in [`NEIGHBORS_COLUMN`].
///
/// A vertex's degree is the index of its first padding slot, so degree is not
/// stored separately. A degree column would cost a second ranged read per
/// vertex, and reading one vertex in one read is the entire reason this layout
/// has a fixed stride.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// Most rows one partition may hold.
///
/// A *count*, and every use of it is a count - not, despite the arithmetic
/// looking the same, the highest local id. The ids of an `n`-row partition run
/// to `n - 1`, so the widest partition whose ids all stay clear of
/// [`NO_NEIGHBOR`] holds `u32::MAX` rows, and this is one below that: the check
/// errs on the safe side by a single row, deliberately, so that no arithmetic
/// anywhere has to be exact about the boundary.
///
/// Do not read a maximum local id out of it. That number is `MAX_PARTITION_ROWS
/// - 1`, and using this constant as one would put a vertex on the sentinel.
pub const MAX_PARTITION_ROWS: u32 = u32::MAX - 1;

/// Widest neighbour list a partition may be built with.
///
/// Not a property of the format, which stores the width as a `u32` and would
/// take any of them, but a bound on what a typo can cost. The width is the
/// stride of `__neighbors`, so it is both `4 * max_degree` bytes per vertex on
/// disk and `4 * max_degree * rows` bytes allocated up front by
/// [`crate::partition::PartitionGraph`] - a `100_000` typed where `100` was
/// meant asks the allocator for 400 GB over a million-row partition and aborts
/// the process instead of returning an error.
///
/// `1024` puts one vertex's list at 4 KiB, a page, which is already far past
/// anything the literature builds: DiskANN's `R` is tens, and this crate's own
/// measured working point is 64.
pub const MAX_DEGREE: u32 = 1024;

/// The one version number of this format.
///
/// Written twice, to two independently corruptible places - the dataset
/// manifest's `index_version` and the segment's own [`IndexMetadata`] - and
/// checked against both on open. Two *different* numbers is what this replaced,
/// and the one recorded in the manifest was checked nowhere at all.
pub const FORMAT_VERSION: u32 = 7;

/// Narrowest vectors a segment may leave to the dataset.
///
/// The re-score reads a row of the dataset's vector column by offset, which
/// only a full-zip column allows: there row `r` sits at `base + r * stride`. A
/// partition file asks Lance for full-zip explicitly ([`partition_schema`]); a
/// dataset's column does not, and Lance then lays out any value narrower than
/// 256 bytes as mini-block (`is_narrow` in `lance-encoding`'s
/// `encodings/logical/primitive.rs`, over a constant it does not export), which
/// packs rows into chunks no offset reaches. At four bytes a dimension, 256
/// bytes is 64 dimensions.
pub const MIN_DATASET_VECTOR_DIMENSION: u32 = 64;

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

/// Where a segment keeps the full vectors its re-score measures against.
///
/// A walk over codes reads a vector only for the few candidates it ends with,
/// so the copy in [`VECTOR_COLUMN`] is a choice rather than a necessity: without
/// it a segment is `4 * dimension` bytes a vertex smaller, and the re-score reads
/// the same vectors out of the dataset's data files by row address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorSource {
    /// The partitions hold a copy of every vector, and every walk mode can
    /// search the segment without reading the dataset.
    Index,
    /// The partitions hold no vectors, so only the walks that steer by codes and
    /// read the vectors of the candidates they end with -
    /// [`crate::query::WalkMode::Lazy`] and [`crate::query::WalkMode::Flat`] -
    /// can search the segment; `Exact` and `Coded` read partitions whole,
    /// vectors included, and are refused. The segment must have codes,
    /// and the dataset's vectors must be at least [`MIN_DATASET_VECTOR_DIMENSION`]
    /// wide. A query needs the data files of the dataset version the index was
    /// opened at, so once a compaction and a cleanup have removed them a
    /// re-score that has to open one fails until the index is opened again -
    /// the rule Lance's own indices live by.
    Dataset,
}

impl std::fmt::Display for VectorSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Index => "index",
            Self::Dataset => "dataset",
        })
    }
}

/// Segment-wide parameters, stored in the schema metadata of `index.idx`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub format_version: u32,
    /// `R` in the Vamana papers: the fixed width of [`NEIGHBORS_COLUMN`].
    pub max_degree: u32,
    /// `L`: the beam the build that produced this segment searched with.
    ///
    /// Nothing reads a segment with it, which is why it is worth saying what it
    /// is here for: a partition can have to be *rebuilt* after the fact -
    /// consolidation does it when the one-hop repair leaves the graph in pieces -
    /// and a rebuild is a build, so it needs the same beam its siblings were
    /// built with. Taking that from the caller instead would be a second record
    /// of one number with nothing to check it against, and the number decides
    /// graph quality: at `max_degree = 32` doubling it made query cost at recall
    /// 0.99 worse by a quarter.
    pub search_list_size: usize,
    /// Pruning slack. `1.0` reproduces the HNSW diversity heuristic exactly.
    pub alpha: f32,
    pub dimension: u32,
    #[serde(with = "distance_type_as_name")]
    pub distance_type: DistanceType,
    pub row_id_mode: RowIdMode,
    /// Ids of the fragments whose rows this segment physically holds.
    ///
    /// The same set is handed to Lance as the segment's committed coverage, so
    /// the two start out equal - and Lance then edits its copy without telling
    /// anyone. An in-place column update prunes the rewritten fragments out of
    /// the manifest's `fragment_bitmap`; a pure row rewrite adds fragments back
    /// into it. Both leave this list untouched, which is exactly what makes it
    /// useful: it is the only record of what the segment was built from, so
    /// comparing the two on open is what turns a silent edit into a refusal.
    pub fragments: Vec<u32>,
    /// How this segment's [`CODE_COLUMN`] was built, when it has one.
    ///
    /// `None` says the partitions carry no codes, so only
    /// [`crate::query::WalkMode::Exact`], which measures against the stored
    /// vectors, can search them. It is not a version marker: a build writes
    /// eight-bit scalar codes unless it is told not to
    /// ([`crate::IndexParams::without_codes`]).
    ///
    /// Inherited wholesale by every maintenance pass, which is what makes the
    /// rotation inside it one per index rather than one per segment.
    pub codes: Option<CodeParams>,
    /// What every partition's [`ENTRY_POINTS_COLUMN`] was trained under, when
    /// the segment keeps entry points.
    ///
    /// `None` says no partition has any, and every walk of the segment starts
    /// at the partition's medoid. Never set without [`Self::codes`]: a walk
    /// chooses among entry points by code.
    ///
    /// Inherited by every maintenance pass like [`Self::codes`], which trains a
    /// partition it rewrites under these, so that every list of a segment was
    /// trained under the one set of parameters recorded here.
    pub entry_point_params: Option<EntryPointParams>,
    /// Whether the partitions hold the vectors or leave them to the dataset.
    ///
    /// Inherited by every maintenance pass like [`Self::codes`], and for the
    /// same reason: a partition file copied between two segments either holds
    /// [`VECTOR_COLUMN`] or does not.
    pub vector_source: VectorSource,
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

    /// The metadata in `json`, refused unless it is of [`FORMAT_VERSION`].
    ///
    /// The version is read on its own first: a field a later format made
    /// required is missing from every earlier one, and what a reader of an
    /// earlier index has to be told is its format, not that its file is corrupt.
    pub fn from_json(json: &str) -> Result<Self> {
        let corrupt = |error: serde_json::Error| {
            Error::corrupt_file_named(
                INDEX_METADATA_KEY,
                format!("failed to parse Vamana index metadata: {error}"),
            )
        };
        let Versioned { format_version } = serde_json::from_str(json).map_err(corrupt)?;
        if format_version != FORMAT_VERSION {
            return Err(Error::not_supported(format!(
                "Vamana index format version {format_version} is not supported by this build \
                 (expected {FORMAT_VERSION})"
            )));
        }
        serde_json::from_str(json).map_err(corrupt)
    }
}

/// The one field of [`IndexMetadata`] every format version shares.
#[derive(Deserialize)]
struct Versioned {
    format_version: u32,
}

/// Arrow schema of one partition file.
///
/// The fixed-size-list columns are laid out so that vertex `local_id` sits at
/// `base + local_id * stride` with no read amplification: `max_degree * 4` bytes
/// for `__neighbors`, `dimension * 4` for `__vector`. Two details make that hold
/// and neither is the default:
///
/// - The full-zip encoding is requested explicitly. Left to the heuristic, Lance
///   picks full-zip only once a value reaches 256 bytes - `max_degree >= 64`, or
///   `dimension >= 64` - and quietly falls back to mini-block below that, which
///   reintroduces chunk amplification and destroys the addressing.
/// - Both the column and its item are non-nullable. What costs is an actual
///   null, not the flag: a null anywhere adds a control word to *every* value in
///   the column and the stride stops being a clean multiple (measured at 133
///   bytes against a stride of 128). Lance drops a validity bitmap that holds no
///   nulls, so a merely nullable column keeps its stride - declaring the field
///   non-nullable is what makes a null impossible to write at all.
///
/// The columns stay separate rather than being interleaved into one wide value
/// because their access patterns differ: consolidation rewrites the edges and
/// not the vectors, a graph walk reads the vectors and edges but never the row
/// ids, and a coded walk reads [`CODE_COLUMN`] and nothing else until it has an
/// answer to re-score. Separate columns are what makes each of those a
/// projection.
///
/// `code_stride` is `None` for a segment built without codes
/// ([`crate::IndexParams::without_codes`]). The code
/// column goes last so that a reader projecting the others by name is
/// unaffected by its presence. `__vector` is there only for
/// [`VectorSource::Index`].
pub fn partition_schema(
    max_degree: u32,
    dimension: u32,
    code_stride: Option<u32>,
    vector_source: VectorSource,
) -> Result<Schema> {
    let mut fields = vec![
        Field::new(ROW_ID_COLUMN, DataType::UInt64, false),
        addressable_list(NEIGHBORS_COLUMN, DataType::UInt32, max_degree, "max_degree")?,
    ];
    if vector_source == VectorSource::Index {
        fields.push(addressable_list(
            VECTOR_COLUMN,
            DataType::Float32,
            dimension,
            "dimension",
        )?);
    }
    if let Some(stride) = code_stride {
        fields.push(addressable_list(
            CODE_COLUMN,
            DataType::UInt8,
            stride,
            "code stride",
        )?);
    }
    Ok(Schema::new(fields))
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
/// the only encoding of that which cannot disagree with itself. A partition
/// with no entry points of its own lists none, an empty list.
pub fn index_schema() -> Schema {
    Schema::new(vec![
        Field::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
        Field::new(MEDOID_COLUMN, DataType::UInt32, false),
        Field::new(
            ENTRY_POINTS_COLUMN,
            DataType::List(entry_point_item()),
            false,
        ),
        Field::new(NUM_ROWS_COLUMN, DataType::UInt32, false),
        Field::new(FILE_COLUMN, DataType::Utf8, false),
    ])
}

/// Item of an [`ENTRY_POINTS_COLUMN`] list: a local id, never null.
pub fn entry_point_item() -> FieldRef {
    Arc::new(Field::new("item", DataType::UInt32, false))
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
        let scalar = CodeParams::Scalar {
            num_bits: 8,
            bounds: -1.0..1.0,
        };
        for (vector_source, spelling) in [
            (VectorSource::Index, "\"vector_source\":\"index\""),
            (VectorSource::Dataset, "\"vector_source\":\"dataset\""),
        ] {
            for (codes, entry_point_params) in [
                (None, None),
                (Some(scalar.clone()), None),
                (Some(scalar.clone()), Some(EntryPointParams::default())),
            ] {
                let metadata = IndexMetadata {
                    format_version: FORMAT_VERSION,
                    max_degree: 64,
                    search_list_size: 100,
                    alpha: 1.2,
                    dimension: 128,
                    distance_type: DistanceType::Cosine,
                    row_id_mode: RowIdMode::Address,
                    fragments: vec![0, 3, 7],
                    codes,
                    entry_point_params,
                    vector_source,
                };
                let json = metadata.to_json().unwrap();
                assert!(json.contains(spelling), "{json}");
                assert_eq!(IndexMetadata::from_json(&json).unwrap(), metadata);
            }
        }
    }

    /// A segment that does not say where its vectors are is not read as one
    /// that holds them: guessing wrong either way sends the re-score to a
    /// column that is not there.
    #[test]
    fn metadata_without_a_vector_source_is_rejected() {
        let json = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "max_degree": 64,
            "search_list_size": 100,
            "alpha": 1.2,
            "dimension": 128,
            "distance_type": "l2",
            "row_id_mode": "address",
            "fragments": [0],
        })
        .to_string();
        let error = IndexMetadata::from_json(&json).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("vector_source"), "{error}");
    }

    #[test]
    fn metadata_round_trips_a_stable_row_id_mode() {
        // The builder refuses to produce this mode, so serde is the only place
        // the spelling can be exercised at all - and `query::VamanaIndex::open`
        // rejects an index by reading it back.
        let metadata = IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.0,
            dimension: 8,
            distance_type: DistanceType::L2,
            row_id_mode: RowIdMode::Stable,
            fragments: vec![0],
            codes: None,
            entry_point_params: None,
            vector_source: VectorSource::Index,
        };
        let json = metadata.to_json().unwrap();
        assert!(json.contains("\"stable\""), "{json}");
        assert_eq!(IndexMetadata::from_json(&json).unwrap(), metadata);
    }

    /// Why `validate_alpha` refuses a non-finite value, stated as a fact about
    /// the format rather than as a comment: JSON has no spelling for infinity,
    /// `serde_json` writes `null`, and the index becomes one that was written
    /// once and can never be opened. The guard lives on the build path, so this
    /// is the only place the consequence itself can be pinned.
    #[test]
    fn a_non_finite_alpha_would_serialise_to_an_unreadable_index() {
        let metadata = IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: 64,
            search_list_size: 100,
            alpha: f32::INFINITY,
            dimension: 128,
            distance_type: DistanceType::L2,
            row_id_mode: RowIdMode::Address,
            fragments: vec![0],
            codes: None,
            entry_point_params: None,
            vector_source: VectorSource::Index,
        };
        let json = metadata.to_json().unwrap();
        assert!(json.contains("\"alpha\":null"), "{json}");
        let error = IndexMetadata::from_json(&json).unwrap_err();
        assert!(error.to_string().contains("invalid type: null"), "{error}");
    }

    #[test]
    fn metadata_rejects_an_unknown_distance_type() {
        let json = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "max_degree": 64,
            "search_list_size": 100,
            "alpha": 1.2,
            "dimension": 128,
            "distance_type": "manhattan",
            "row_id_mode": "address",
            "fragments": [0],
            "vector_source": "index",
        })
        .to_string();
        let error = IndexMetadata::from_json(&json).unwrap_err();
        assert!(error.to_string().contains("manhattan"), "{error}");
    }

    /// Another format is refused for its version, including the one before it,
    /// whose metadata would otherwise read: format 6 wrote no
    /// `entry_point_params`, so its segments would pass for ones keeping no
    /// entry points, and its partition table has no [`ENTRY_POINTS_COLUMN`].
    /// Its reader is told which format it has rather than that its file is
    /// corrupt.
    #[test]
    fn metadata_rejects_another_format_version() {
        let earlier = serde_json::json!({
            "format_version": 6,
            "max_degree": 64,
            "search_list_size": 100,
            "alpha": 1.2,
            "dimension": 128,
            "distance_type": "l2",
            "row_id_mode": "address",
            "fragments": [0],
            "codes": null,
            "vector_source": "index",
        });
        let mut future = earlier.clone();
        future["format_version"] = serde_json::json!(FORMAT_VERSION + 1);
        future["entry_point_params"] = serde_json::to_value(EntryPointParams::default()).unwrap();
        assert_eq!(
            FORMAT_VERSION, 7,
            "the earlier format here is the one before this"
        );
        for (version, json) in [(FORMAT_VERSION + 1, future), (6, earlier)] {
            let error = IndexMetadata::from_json(&json.to_string()).unwrap_err();
            assert!(
                matches!(error, Error::NotSupported { .. }),
                "unexpected error: {error}"
            );
            assert!(
                error
                    .to_string()
                    .contains(&format!("format version {version} is not supported")),
                "{error}"
            );
        }
    }

    #[test]
    fn partition_schema_requests_fullzip_and_stays_non_nullable() {
        let schema = partition_schema(32, 24, None, VectorSource::Index).unwrap();
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
    fn partition_schema_leaves_the_vectors_to_the_dataset() {
        let schema = partition_schema(32, 64, Some(64), VectorSource::Dataset).unwrap();
        let names = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, [ROW_ID_COLUMN, NEIGHBORS_COLUMN, CODE_COLUMN]);
    }

    #[test]
    fn partition_schema_rejects_a_zero_degree() {
        let error = partition_schema(0, 8, None, VectorSource::Index).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("max_degree"), "{error}");
    }

    #[test]
    fn partition_schema_rejects_a_zero_dimension() {
        let error = partition_schema(32, 0, None, VectorSource::Index).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("dimension"), "{error}");
    }
}
