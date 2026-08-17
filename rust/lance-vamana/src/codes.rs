// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Quantised codes: what a walk steers by when it will not read a vector.
//!
//! A partition file may carry one [`CODE_COLUMN`] value per vertex beside its
//! vector and its edges, and a walk given one measures its distances against
//! that column instead of `__vector`. What the column is *for* is the disk
//! traversal: codes small enough to keep resident are what leaves a walk with
//! only the edges of the vertices it expands to fetch. On their own they buy
//! nothing - a partition is still read whole - and cost thirteen per cent of the
//! index at `d = 128`.
//!
//! The measurement behind the parameters is `examples/coded_walk.rs`. A walk on
//! RaBitQ residual codes reaches the exact walk's recall for two to thirteen per
//! cent more comparisons from **three bits a dimension**, 68 bytes a vertex at
//! `d = 128`; one bit needs a beam one and a half to three and a half times
//! wider, which multiplies the very reads the codes were there to save. The
//! answer has to be re-scored from the whole candidate list rather than its
//! nearest `k`, because a coded walk's own ordering tops out around 0.95 recall
//! at any width.
//!
//! The quantiser is Lance's own (`lance_index::vector::bq`) rather than one of
//! ours: what is being asked of it is to steer a walk, and a hand-rolled
//! estimator would steer by its own bugs instead. What is ours is the *layout*.
//! Lance spreads one RaBitQ code across seven columns and holds it in memory in
//! a kernel layout that interleaves thirty-two vectors at a time; here one
//! vertex is one contiguous stride, so a partition's codes are one ranged read,
//! and Lance's own loader rebuilds the kernel layout from it.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt8Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::ex_dot::blocked_ex_code_bytes;
use lance_index::vector::bq::rotation::random_fast_rotation_signs;
use lance_index::vector::bq::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN, RabitQuantizationMetadata,
    RabitQuantizationStorage, RabitQueryEstimator, rabit_binary_code_field, rabit_ex_code_field,
};
use lance_index::vector::bq::transform::{
    ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
    RQTransformer, SCALE_FACTORS_COLUMN,
};
use lance_index::vector::bq::{
    RQBuildParams, RQRotationType, rabit_binary_code_bytes, rabit_ex_bits, validate_rq_num_bits,
};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::transform::Transformer;
use lance_index::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use crate::format::INDEX_FILE_NAME;

/// One vertex's code, as many bytes as [`CodeParams::stride`] says.
///
/// A `UInt8` blob rather than the seven typed columns Lance writes, because a
/// walk wants every piece of one vertex at once and none of one it skips: seven
/// columns would be seven reads of a partition where this is one, and seven
/// things to keep resident where this is one.
pub const CODE_COLUMN: &str = "__code";

/// The column name the RaBitQ transform is handed its residuals under.
///
/// Local to this module: nothing else sees the batch it names, which lives for
/// exactly as long as one call to [`encode`].
const RESIDUAL_COLUMN: &str = "residual";

/// RaBitQ's `add`, `scale` and `error` factors, which every code carries.
const BINARY_FACTORS: usize = 3;

/// The two more a code carries once it has extended bits to correct with.
const EX_FACTORS: usize = 2;

/// Factors are `f32`, written little-endian - the byte order of every other
/// number in a Lance file.
const FACTOR_BYTES: usize = std::mem::size_of::<f32>();

/// How a segment's codes were built.
///
/// Travels inside [`crate::format::IndexMetadata`], which every maintenance pass
/// inherits from the segment it rewrites, and that inheritance is what makes
/// "one rotation per index" true rather than merely intended: a pass would have
/// to go out of its way to mint a second one, and
/// [`crate::io::SegmentWriter::copy_partition`] refuses a partition carried in
/// from a segment whose codes disagree with these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeParams {
    /// Bits a dimension.
    ///
    /// Three is the measured working point: at one the walk needs a beam wide
    /// enough to cost more reads than the codes save, and at five the last two
    /// to thirteen per cent of comparisons cost another 32 bytes a vertex.
    pub num_bits: u8,
    /// The one rotation every code of this index was built under.
    ///
    /// RaBitQ quantises a *rotated* residual, so a code read back under another
    /// rotation is not a worse distance but a meaningless one. Stored rather
    /// than regenerated because it is random, and it can be stored at all
    /// because it is small: 480 bytes at `d = 960`.
    pub rotation_signs: Vec<u8>,
}

impl CodeParams {
    /// Mint a fresh rotation for an index of `dimension` columns.
    ///
    /// Refuses a dimension RaBitQ cannot quantise rather than quietly building
    /// an index without codes: a caller who asked for codes and got none would
    /// find out from a query that was slower than it expected.
    pub fn mint(num_bits: u8, dimension: u32) -> Result<Self> {
        validate_rq_num_bits(num_bits)?;
        check_dimension(dimension)?;
        Ok(Self {
            num_bits,
            rotation_signs: random_fast_rotation_signs(dimension as usize),
        })
    }

    /// Bytes one vertex's code occupies.
    pub fn stride(&self, dimension: u32) -> Result<u32> {
        let stride = self.layout(dimension)?.stride;
        u32::try_from(stride).map_err(|_| {
            Error::invalid_input(format!(
                "Vamana codes of {} bits over {dimension} dimensions are {stride} bytes a vertex, \
                 which no Arrow list can hold",
                self.num_bits
            ))
        })
    }

    /// What Lance's quantiser is told about these codes.
    ///
    /// `packed: false` because the column holds one vertex per stride; Lance
    /// repacks into its thirty-two-vector kernel layout as it loads the batch.
    fn quantization(&self, dimension: u32) -> RabitQuantizationMetadata {
        RabitQuantizationMetadata {
            rotate_mat: None,
            rotate_mat_position: None,
            fast_rotation_signs: Some(self.rotation_signs.clone()),
            rotation_type: RQRotationType::Fast,
            code_dim: dimension,
            num_bits: self.num_bits,
            packed: false,
            query_estimator: RabitQueryEstimator::RawQuery,
        }
    }

    fn layout(&self, dimension: u32) -> Result<Layout> {
        validate_rq_num_bits(self.num_bits)?;
        check_dimension(dimension)?;
        let dimension = dimension as usize;
        let binary = 0..rabit_binary_code_bytes(dimension);
        let ex_bits = rabit_ex_bits(self.num_bits)?;
        let (ex, factors) = if ex_bits == 0 {
            (None, BINARY_FACTORS)
        } else {
            let end = binary.end + blocked_ex_code_bytes(dimension, ex_bits);
            (Some(binary.end..end), BINARY_FACTORS + EX_FACTORS)
        };
        let first_factor = ex.as_ref().map_or(binary.end, |ex| ex.end);
        Ok(Layout {
            binary,
            ex,
            factors,
            first_factor,
            stride: first_factor + factors * FACTOR_BYTES,
        })
    }
}

/// Where each piece of one vertex's code sits inside its stride.
///
/// Described once and read by both directions, so that moving anything in the
/// encoder cannot leave the decoder reading the old place.
#[derive(Debug)]
struct Layout {
    binary: Range<usize>,
    /// Absent at one bit a dimension, where there is nothing to correct with.
    ex: Option<Range<usize>>,
    factors: usize,
    first_factor: usize,
    stride: usize,
}

impl Layout {
    fn factor(&self, index: usize) -> Range<usize> {
        let start = self.first_factor + index * FACTOR_BYTES;
        start..start + FACTOR_BYTES
    }
}

/// RaBitQ packs a bit a dimension and requires a whole number of bytes.
fn check_dimension(dimension: u32) -> Result<()> {
    if !(dimension as usize).is_multiple_of(u8::BITS as usize) {
        return Err(Error::invalid_input(format!(
            "Vamana cannot build codes over {dimension} dimensions: RaBitQ packs a bit a dimension \
             and requires a multiple of 8"
        )));
    }
    Ok(())
}

/// The metric the codes are built and read in.
///
/// Cosine is stored as unit vectors and routed by L2 - see
/// [`crate::builder::routing_distance_type`] - and RaBitQ's factors are defined
/// for L2 and dot only, so the codes work in the same L2 the router does. Over
/// unit vectors that orders exactly as cosine does, and an answer's distances
/// come from the exact re-scoring rather than from a code, so nothing downstream
/// sees the substitution.
///
/// Mapped here rather than at the call sites because getting it wrong produces
/// distances that are wrong rather than approximate, and a recall number reports
/// that as "the codes are poor".
fn coding_distance_type(distance_type: DistanceType) -> Result<DistanceType> {
    match distance_type {
        DistanceType::L2 | DistanceType::Cosine => Ok(DistanceType::L2),
        other => Err(Error::not_supported(format!(
            "Vamana cannot build codes under {other} distance"
        ))),
    }
}

/// Encode one partition's vectors against the centroid they were routed to.
///
/// The codes are a projection of the vectors taken as they are written, never a
/// field of [`crate::partition::Partition`]. That is what keeps them in step:
/// consolidation, insertion and merge all move vertices between local ids, and
/// none of them has to move a code with one, because there is no code to move
/// until the partition is written again.
pub(crate) fn encode(
    params: &CodeParams,
    distance_type: DistanceType,
    vectors: &FixedSizeListArray,
    centroid: &ArrayRef,
) -> Result<FixedSizeListArray> {
    let dimension = u32::try_from(vectors.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "Vamana cannot code vectors of dimension {}",
            vectors.value_length()
        ))
    })?;
    // One row, because the transform indexes its centroids by partition id and
    // every vector here belongs to the same one.
    let centroid = FixedSizeListArray::try_new_from_values(
        centroid
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Vamana codes want a Float32 centroid, got {}",
                    centroid.data_type()
                ))
            })?
            .clone(),
        vectors.value_length(),
    )?;
    let layout = params.layout(dimension)?;
    let distance_type = coding_distance_type(distance_type)?;

    let (residuals, norms) = residuals(vectors, &centroid)?;
    let quantizer = RabitQuantizer::build(
        &residuals,
        distance_type,
        &RQBuildParams {
            num_bits: params.num_bits,
            rotation_type: RQRotationType::Fast,
            rotation: Some(params.quantization(dimension)),
        },
    )?;
    let rows = vectors.len();
    let batch = RecordBatch::try_from_iter_with_nullable(vec![
        (RESIDUAL_COLUMN, Arc::new(residuals) as ArrayRef, false),
        (CENTROID_DIST_COLUMN, Arc::new(norms) as ArrayRef, false),
        (
            PART_ID_COLUMN,
            Arc::new(UInt32Array::from(vec![0u32; rows])) as ArrayRef,
            false,
        ),
    ])?;
    let coded = RQTransformer::new(quantizer, distance_type, centroid, RESIDUAL_COLUMN)?
        .transform(&batch)?;

    interleave(&coded, &layout, rows)
}

/// Read a partition's code column back into the store a walk measures against.
///
/// The row ids come from the partition rather than from the code column: a code
/// is addressed by local id, like every other per-vertex column here, and Lance's
/// storage wants a row id per row only because it is written for an index whose
/// answers come out of it. Ours come out of the exact re-scoring.
/// The code column of a partition file, as read.
///
/// Split out from [`storage`] so that a caller can keep the column and drop the
/// rest of the batch: the vectors and the edges are already held by the
/// [`crate::partition::Partition`] beside it, and holding the batch through a
/// walk would keep a second copy of both.
pub(crate) fn column(batch: &RecordBatch) -> Result<FixedSizeListArray> {
    let codes = batch.column_by_name(CODE_COLUMN).ok_or_else(|| {
        Error::corrupt_file_named(
            CODE_COLUMN,
            "Vamana segment declares codes but its partition file has none".to_string(),
        )
    })?;
    Ok(codes
        .as_fixed_size_list_opt()
        .ok_or_else(|| {
            Error::corrupt_file_named(
                CODE_COLUMN,
                format!(
                    "Vamana code column has type {}, expected a fixed size list",
                    codes.data_type()
                ),
            )
        })?
        .clone())
}

pub(crate) fn storage(
    params: &CodeParams,
    distance_type: DistanceType,
    dimension: u32,
    row_ids: &[u64],
    codes: &FixedSizeListArray,
) -> Result<RabitQuantizationStorage> {
    let layout = params.layout(dimension)?;
    if codes.value_length() as usize != layout.stride {
        return Err(Error::corrupt_file_named(
            CODE_COLUMN,
            format!(
                "Vamana code column is {} bytes a vertex but {} bits over {dimension} dimensions \
                 are {} bytes",
                codes.value_length(),
                params.num_bits,
                layout.stride
            ),
        ));
    }
    if codes.len() != row_ids.len() {
        return Err(Error::corrupt_file_named(
            CODE_COLUMN,
            format!(
                "Vamana partition file has {} vertices but {} codes",
                row_ids.len(),
                codes.len()
            ),
        ));
    }
    if codes.null_count() != 0 || codes.values().null_count() != 0 {
        return Err(Error::corrupt_file_named(
            CODE_COLUMN,
            "Vamana code column holds nulls; a null breaks the fixed stride".to_string(),
        ));
    }

    let batch = split(params, &layout, dimension, row_ids, codes)?;
    RabitQuantizationStorage::try_from_batch(
        batch,
        &params.quantization(dimension),
        coding_distance_type(distance_type)?,
        None,
    )
}

/// `|q - c|^2` between a routing query and one partition's centroid.
///
/// The term RaBitQ's raw-query estimator wants beside the query itself, because
/// the centroid is already folded into every vertex's own factors. Written here
/// rather than at the query path so that the two things a coded distance is
/// assembled from are described in one place.
///
/// The *routing* query rather than the raw one, because that is the space the
/// codes were built in: a cosine index stores unit vectors and its centroids are
/// unit-space too.
///
/// Getting it wrong is invisible to a walk. For L2 this is one additive constant
/// across a whole partition, so it shifts every distance equally and reorders
/// nothing; the answer is re-scored exactly, so the shift never leaves the walk
/// either. What it does reach is Lance's own error-bound gate above one bit, and
/// anything later that treats a coded distance as a number rather than as a
/// rank - which is why the calibration is a test of its own.
pub(crate) fn centroid_distance(
    ivf: &IvfModel,
    partition_id: u32,
    routing_query: &ArrayRef,
) -> Result<f32> {
    let centroid = ivf.centroid(partition_id as usize).ok_or_else(|| {
        Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "Vamana partition {partition_id} has no centroid in a routing model of {}",
                ivf.num_partitions()
            ),
        )
    })?;
    let centroid = centroid.as_primitive_opt::<Float32Type>().ok_or_else(|| {
        Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "Vamana segment carries {} centroids, expected Float32",
                centroid.data_type()
            ),
        )
    })?;
    let query = routing_query
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::internal("a Vamana routing query is not Float32".to_string()))?;
    if query.len() != centroid.len() {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "Vamana partition {partition_id} has a centroid of {} values against a query of {}",
                centroid.len(),
                query.len()
            ),
        ));
    }
    Ok(query
        .values()
        .iter()
        .zip(centroid.values())
        .map(|(value, center)| (value - center) * (value - center))
        .sum())
}

/// One partition's vectors as residuals against its centroid, with `|v - c|^2`.
///
/// Both are what the transform wants, and the residual is the whole reason a
/// code of a few bits a dimension can steer anything: it quantises the offset
/// from a point the query knows exactly, not the vector.
fn residuals(
    vectors: &FixedSizeListArray,
    centroid: &FixedSizeListArray,
) -> Result<(FixedSizeListArray, Float32Array)> {
    let dimension = vectors.value_length() as usize;
    let values = vectors.values().as_primitive::<Float32Type>().values();
    let centroid = centroid.values().as_primitive::<Float32Type>().values();
    let mut residuals = Vec::with_capacity(values.len());
    let mut norms = Vec::with_capacity(vectors.len());
    for vector in values.chunks_exact(dimension) {
        let mut norm = 0.0f32;
        for (value, center) in vector.iter().zip(centroid) {
            let residual = value - center;
            residuals.push(residual);
            norm += residual * residual;
        }
        norms.push(norm);
    }
    Ok((
        FixedSizeListArray::try_new_from_values(Float32Array::from(residuals), dimension as i32)?,
        Float32Array::from(norms),
    ))
}

/// Lay the transform's columns out one vertex at a time.
fn interleave(coded: &RecordBatch, layout: &Layout, rows: usize) -> Result<FixedSizeListArray> {
    let binary = code_bytes(coded, RABIT_CODE_COLUMN, layout.binary.len())?;
    let ex = layout
        .ex
        .as_ref()
        .map(|ex| code_bytes(coded, RABIT_BLOCKED_EX_CODE_COLUMN, ex.len()))
        .transpose()?;
    let factors = factor_columns(coded, layout.factors)?;

    let mut packed = vec![0u8; rows * layout.stride];
    for (row, slot) in packed.chunks_exact_mut(layout.stride).enumerate() {
        copy_row(&mut slot[layout.binary.clone()], binary, row);
        if let (Some(range), Some(ex)) = (layout.ex.clone(), ex) {
            copy_row(&mut slot[range], ex, row);
        }
        for (index, column) in factors.iter().enumerate() {
            slot[layout.factor(index)].copy_from_slice(&column[row].to_le_bytes());
        }
    }
    // Not `try_new_from_values`, which makes the item nullable: the column has
    // to match the non-nullable field `crate::format::partition_schema` declares
    // for it, and a fixed size list's item nullability is part of its type.
    Ok(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        layout.stride as i32,
        Arc::new(UInt8Array::from(packed)),
        None,
    )?)
}

/// Split one vertex per stride back into the columns Lance's loader wants.
///
/// The field shapes are Lance's own rather than ours, because a fixed size
/// list's item nullability is part of its type and its loader would reject a
/// column that merely holds the right bytes.
fn split(
    params: &CodeParams,
    layout: &Layout,
    dimension: u32,
    row_ids: &[u64],
    codes: &FixedSizeListArray,
) -> Result<RecordBatch> {
    let rows = row_ids.len();
    // Bounded by the row count the segment table gave rather than by the length
    // of the buffer: `values()` on a fixed size list reads through to the whole
    // child array, so a longer one would silently produce more codes than there
    // are vertices and be reported as mismatched column lengths further down.
    let packed = codes
        .values()
        .as_primitive::<UInt8Type>()
        .values()
        .get(..rows * layout.stride)
        .ok_or_else(|| {
            Error::corrupt_file_named(
                CODE_COLUMN,
                format!(
                    "Vamana code column holds {} bytes, short of the {} that {rows} vertices of \
                     {} bytes need",
                    codes.values().len(),
                    rows * layout.stride,
                    layout.stride
                ),
            )
        })?;

    let mut binary = Vec::with_capacity(rows * layout.binary.len());
    let mut ex = layout
        .ex
        .as_ref()
        .map(|ex| Vec::with_capacity(rows * ex.len()));
    let mut factors = vec![Vec::with_capacity(rows); layout.factors];
    for slot in packed.chunks_exact(layout.stride) {
        binary.extend_from_slice(&slot[layout.binary.clone()]);
        if let (Some(range), Some(ex)) = (layout.ex.clone(), ex.as_mut()) {
            ex.extend_from_slice(&slot[range]);
        }
        for (index, column) in factors.iter_mut().enumerate() {
            // The slice is `FACTOR_BYTES` long by construction, so the array
            // conversion cannot fail; `expect` rather than `?` keeps the loop
            // free of an error path nothing can take.
            let bytes = slot[layout.factor(index)]
                .try_into()
                .expect("a factor is four bytes");
            column.push(f32::from_le_bytes(bytes));
        }
    }

    let dimension = dimension as usize;
    let mut fields = vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        rabit_binary_code_field(dimension),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(row_ids.to_vec())),
        Arc::new(nullable_bytes(binary, layout.binary.len())?),
    ];
    if let (Some(range), Some(values)) = (layout.ex.clone(), ex) {
        fields.push(
            rabit_ex_code_field(dimension, params.num_bits)?
                .ok_or_else(|| Error::internal("RaBitQ lost its ex-code field".to_string()))?,
        );
        columns.push(Arc::new(nullable_bytes(values, range.len())?));
    }
    for (name, values) in factor_names(layout.factors).into_iter().zip(factors) {
        fields.push(Field::new(name, DataType::Float32, true));
        columns.push(Arc::new(Float32Array::from(values)));
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// A code column in the shape Lance's own writer produces: nullable throughout.
fn nullable_bytes(values: Vec<u8>, width: usize) -> Result<FixedSizeListArray> {
    Ok(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt8, true)),
        width as i32,
        Arc::new(UInt8Array::from(values)),
        None,
    )?)
}

fn copy_row(slot: &mut [u8], values: &[u8], row: usize) {
    let width = slot.len();
    slot.copy_from_slice(&values[row * width..(row + 1) * width]);
}

fn code_bytes<'a>(coded: &'a RecordBatch, column: &str, width: usize) -> Result<&'a [u8]> {
    let array = coded
        .column_by_name(column)
        .ok_or_else(|| Error::internal(format!("RaBitQ produced no {column} column")))?
        .as_fixed_size_list();
    if array.value_length() as usize != width {
        return Err(Error::internal(format!(
            "RaBitQ produced {column} of {} bytes a vertex where the layout expects {width}",
            array.value_length()
        )));
    }
    Ok(array.values().as_primitive::<UInt8Type>().values())
}

fn factor_names(factors: usize) -> Vec<&'static str> {
    let mut names = vec![
        ADD_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
        ERROR_FACTORS_COLUMN,
    ];
    if factors > BINARY_FACTORS {
        names.push(EX_ADD_FACTORS_COLUMN);
        names.push(EX_SCALE_FACTORS_COLUMN);
    }
    names
}

fn factor_columns(coded: &RecordBatch, factors: usize) -> Result<Vec<&[f32]>> {
    factor_names(factors)
        .into_iter()
        .map(|name| {
            Ok(coded
                .column_by_name(name)
                .ok_or_else(|| Error::internal(format!("RaBitQ produced no {name} column")))?
                .as_primitive::<Float32Type>()
                .values()
                .as_ref())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BinaryHeap;

    use lance_index::vector::storage::{DistCalculator, VectorStore};
    use rand::rngs::SmallRng;
    use rand::{Rng, RngCore, SeedableRng};

    const DIMENSION: u32 = 64;
    const ROWS: usize = 50;

    #[test]
    fn a_stride_is_the_bits_plus_the_factors() {
        for (num_bits, expected) in [(1u8, 8 + 12), (3, 8 + 16 + 20)] {
            let params = CodeParams::mint(num_bits, DIMENSION).unwrap();
            assert_eq!(
                params.stride(DIMENSION).unwrap(),
                expected,
                "{num_bits} bits"
            );
        }
    }

    /// The width the measurement reported, pinned so that a change in Lance's
    /// blocked ex-code layout shows up here rather than as an index that grew.
    #[test]
    fn three_bits_are_sixty_eight_bytes_at_d_128() {
        assert_eq!(CodeParams::mint(3, 128).unwrap().stride(128).unwrap(), 68);
    }

    #[test]
    fn a_dimension_rabit_cannot_pack_is_refused() {
        let error = CodeParams::mint(3, 100).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("multiple of 8"), "{error}");
    }

    fn sample(seed: u64) -> (FixedSizeListArray, ArrayRef, Vec<u64>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let values = (0..ROWS * DIMENSION as usize)
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIMENSION as i32)
                .unwrap();
        let centroid: ArrayRef = Arc::new(Float32Array::from(
            (0..DIMENSION)
                .map(|_| rng.random::<f32>())
                .collect::<Vec<_>>(),
        ));
        (vectors, centroid, (0..ROWS as u64).collect())
    }

    /// The storage Lance would build from its own seven columns, for the same
    /// vectors and the same rotation.
    ///
    /// The reference the blob is checked against. Reproducing Lance's own
    /// assembly here rather than checking column against column is what makes
    /// this catch more than a layout slip: `packed`, the estimator and the
    /// rotation type all reach the distances through it, and every one of them
    /// is a value this crate chooses.
    fn reference(
        params: &CodeParams,
        vectors: &FixedSizeListArray,
        centroid: &ArrayRef,
        row_ids: &[u64],
    ) -> RabitQuantizationStorage {
        let centroid_list = FixedSizeListArray::try_new_from_values(
            centroid.as_primitive::<Float32Type>().clone(),
            DIMENSION as i32,
        )
        .unwrap();
        let (residuals, norms) = residuals(vectors, &centroid_list).unwrap();
        let quantizer = RabitQuantizer::build(
            &residuals,
            DistanceType::L2,
            &RQBuildParams {
                num_bits: params.num_bits,
                rotation_type: RQRotationType::Fast,
                rotation: Some(params.quantization(DIMENSION)),
            },
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter_with_nullable(vec![
            (
                ROW_ID,
                Arc::new(UInt64Array::from(row_ids.to_vec())) as ArrayRef,
                false,
            ),
            (RESIDUAL_COLUMN, Arc::new(residuals) as ArrayRef, false),
            (CENTROID_DIST_COLUMN, Arc::new(norms) as ArrayRef, false),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0u32; row_ids.len()])) as ArrayRef,
                false,
            ),
        ])
        .unwrap();
        let coded = RQTransformer::new(quantizer, DistanceType::L2, centroid_list, RESIDUAL_COLUMN)
            .unwrap()
            .transform(&batch)
            .unwrap();
        let kept = coded
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| {
                !matches!(
                    field.name().as_str(),
                    RESIDUAL_COLUMN | CENTROID_DIST_COLUMN | PART_ID_COLUMN
                )
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        RabitQuantizationStorage::try_from_batch(
            coded.project(&kept).unwrap(),
            &params.quantization(DIMENSION),
            DistanceType::L2,
            None,
        )
        .unwrap()
    }

    /// One vertex per stride and Lance's seven columns have to be the same code.
    ///
    /// Bit for bit, over the whole partition, because the failure this guards
    /// against is silent: a factor read from the wrong offset gives distances
    /// that are wrong rather than absent, and the only thing downstream that
    /// would notice is recall.
    #[test]
    fn a_blob_and_lances_own_columns_give_the_same_distances() {
        for num_bits in [1u8, 3, 5] {
            let params = CodeParams::mint(num_bits, DIMENSION).unwrap();
            let (vectors, centroid, row_ids) = sample(7);
            let column = encode(&params, DistanceType::L2, &vectors, &centroid).unwrap();
            assert_eq!(
                column.value_length() as u32,
                params.stride(DIMENSION).unwrap(),
                "{num_bits} bits"
            );

            let ours = storage(&params, DistanceType::L2, DIMENSION, &row_ids, &column).unwrap();
            let theirs = reference(&params, &vectors, &centroid, &row_ids);

            let (query, _, _) = sample(99);
            let key: ArrayRef = Arc::new(Float32Array::from(
                query.values().as_primitive::<Float32Type>().values()[..DIMENSION as usize]
                    .to_vec(),
            ));
            let dist_q_c = key
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .zip(centroid.as_primitive::<Float32Type>().values())
                .map(|(value, center)| (value - center) * (value - center))
                .sum::<f32>();
            let ours = ours.dist_calculator(key.clone(), dist_q_c);
            let theirs = theirs.dist_calculator(key, dist_q_c);
            for id in 0..ROWS as u32 {
                assert_eq!(
                    ours.distance(id),
                    theirs.distance(id),
                    "{num_bits} bits, vertex {id}"
                );
            }
        }
    }

    /// The distances have to *be* distances, not merely be ordered like them.
    ///
    /// A walk is blind to this: RaBitQ's raw-query factor is one additive
    /// constant per partition per query, so getting it wrong shifts every
    /// distance in a partition equally and reorders nothing - and the answer is
    /// re-scored exactly, so the shift never leaves the walk either. Recall
    /// would not notice. What would notice is anything that compares a coded
    /// distance against a *number*: the error-bound gate Lance already applies
    /// above one bit, and any future use of one as a bound on whether to expand
    /// a vertex at all.
    #[test]
    fn a_coded_distance_estimates_the_real_one() {
        let params = CodeParams::mint(3, DIMENSION).unwrap();
        let (vectors, centroid, row_ids) = sample(7);
        let column = encode(&params, DistanceType::L2, &vectors, &centroid).unwrap();
        let store = storage(&params, DistanceType::L2, DIMENSION, &row_ids, &column).unwrap();

        let (queries, _, _) = sample(31);
        let query =
            queries.values().as_primitive::<Float32Type>().values()[..DIMENSION as usize].to_vec();
        // Through the production helper, over a one-partition routing model, so
        // that a mistake in the term itself fails here and not only in recall.
        let key: ArrayRef = Arc::new(Float32Array::from(query.clone()));
        let ivf = IvfModel::new(
            FixedSizeListArray::try_new_from_values(
                centroid.as_primitive::<Float32Type>().clone(),
                DIMENSION as i32,
            )
            .unwrap(),
            None,
        );
        let dist_q_c = centroid_distance(&ivf, 0, &key).unwrap();
        let coded = store.dist_calculator(key, dist_q_c);

        let values = vectors.values().as_primitive::<Float32Type>().values();
        let mut errors = values
            .chunks_exact(DIMENSION as usize)
            .enumerate()
            .map(|(id, vector)| {
                let exact = query
                    .iter()
                    .zip(vector)
                    .map(|(left, right)| (left - right) * (left - right))
                    .sum::<f32>();
                ((coded.distance(id as u32) - exact) / exact).abs()
            })
            .collect::<Vec<_>>();
        errors.sort_by(f32::total_cmp);
        let median = errors[errors.len() / 2];
        // Loose on purpose: the bound this pins is "a code is an estimate", and
        // a three-bit code of a 64-dimensional residual is a coarse one. The
        // failure it exists for - a factor left out of the query - is off by the
        // distance to the centroid, which on this fixture is several times the
        // distance being estimated.
        assert!(
            median < 0.2,
            "the median coded distance is off by {median:.3} of the real one"
        );
    }

    /// A rotation drawn from `seed` rather than from the machine.
    ///
    /// [`CodeParams::mint`] takes a fresh random rotation every call, which is
    /// right for an index and wrong for a test that measures how often an
    /// estimate built on one misses: the answer would be a different number
    /// every run, and any bar tight enough to be worth setting would fail on
    /// some of them. How many sign bits a fast rotation wants stays Lance's
    /// business - the bytes are minted and then overwritten, not counted here.
    fn code_params(num_bits: u8, seed: u64) -> CodeParams {
        let mut rotation_signs = random_fast_rotation_signs(DIMENSION as usize);
        SmallRng::seed_from_u64(seed).fill_bytes(&mut rotation_signs);
        CodeParams {
            num_bits,
            rotation_signs,
        }
    }

    /// Where a query sits relative to the partition it is scored against.
    ///
    /// Both placements are needed to exercise the gate. One *on a vertex* makes
    /// the heap threshold tight from the first group, which is what gets the
    /// bound consulted at all; one *elsewhere* puts the `L`-th and the `L + 1`-th
    /// within a hair of each other, which is where a bound off by a slack decides
    /// the wrong one.
    #[derive(Debug, Clone, Copy)]
    enum Placement {
        OnAVertex,
        Elsewhere,
    }

    /// One coded partition, and one query scored against it both ways.
    ///
    /// Returns what [`DistCalculator::accumulate_topk_with_scratch`] kept, as
    /// `(vertex, distance)`, and every distance
    /// [`DistCalculator::distance_all`] measured. Between them those answer the
    /// two questions the tests below ask: whether the gate kept the right
    /// vertices, and whether it gave them the right distances.
    fn gate(
        num_bits: u8,
        seed: u64,
        placement: Placement,
        search_list_size: usize,
    ) -> (Vec<(u64, f32)>, Vec<f32>) {
        let params = code_params(num_bits, seed);
        let (vectors, centroid, row_ids) = sample(seed % 17 + 1);
        let column = encode(&params, DistanceType::L2, &vectors, &centroid).unwrap();
        let store = storage(&params, DistanceType::L2, DIMENSION, &row_ids, &column).unwrap();

        let query = match placement {
            Placement::OnAVertex => vectors.values().as_primitive::<Float32Type>().values()
                [..DIMENSION as usize]
                .to_vec(),
            Placement::Elsewhere => {
                let (elsewhere, _, _) = sample(seed + 1000);
                elsewhere.values().as_primitive::<Float32Type>().values()[..DIMENSION as usize]
                    .to_vec()
            }
        };
        let key: ArrayRef = Arc::new(Float32Array::from(query));
        let dist_q_c = key
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .zip(centroid.as_primitive::<Float32Type>().values())
            .map(|(value, center)| (value - center) * (value - center))
            .sum::<f32>();
        let coded = store.dist_calculator(key, dist_q_c);

        let mut nearest = BinaryHeap::new();
        coded.accumulate_topk_with_scratch(
            search_list_size,
            None,
            None,
            u64::from,
            &mut nearest,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        let kept = nearest
            .into_iter()
            .map(|node| (node.id, node.dist.0))
            .collect::<Vec<_>>();
        (kept, coded.distance_all(0))
    }

    /// The gate's distances, ascending, having checked each against the vertex
    /// it was credited to.
    ///
    /// That check is the one exactness that holds wherever the gate runs: a mask
    /// evaluated at the wrong lane offset would hand back a real distance
    /// attached to the wrong vertex, and every other assertion here looks only at
    /// distances and would let it through.
    fn kept_distances(kept: &[(u64, f32)], all: &[f32], what: &str) -> Vec<f32> {
        let mut distances = kept
            .iter()
            .map(|(vertex, distance)| {
                assert_eq!(
                    *distance, all[*vertex as usize],
                    "{what}: vertex {vertex} came back at a distance the exhaustive scan does not \
                     give it"
                );
                *distance
            })
            .collect::<Vec<_>>();
        distances.sort_by(f32::total_cmp);
        distances
    }

    /// Where the gate cannot prune, it has to be the exhaustive scan exactly.
    ///
    /// There are two such places, and they are the two ways
    /// [`DistCalculator::accumulate_topk_with_scratch`] can decline to gate. At
    /// one bit there is no extra-bit refinement to skip, so Lance bypasses the
    /// mask altogether and runs `distance_all`; at a list as wide as the
    /// partition the heap never fills, so the threshold the mask compares against
    /// is never set. Both are exact, and an equality is what a mutation to either
    /// guard fails - a recall bar over a whole index would not notice.
    #[test]
    fn a_gate_with_nothing_to_prune_is_the_exhaustive_scan() {
        for placement in [Placement::OnAVertex, Placement::Elsewhere] {
            for search_list_size in [1usize, 5, 17, ROWS] {
                // One bit, every width: bypassed.
                let cases = [(1u8, search_list_size)]
                    .into_iter()
                    // Three and five bits, but only at the full width: gated,
                    // with a threshold that is never set.
                    .chain((search_list_size == ROWS).then_some((3, ROWS)))
                    .chain((search_list_size == ROWS).then_some((5, ROWS)));
                for (num_bits, size) in cases {
                    let what = format!("{num_bits} bits, query {placement:?}, list of {size}");
                    let (kept, all) = gate(num_bits, 7, placement, size);
                    let mut exhaustive = all.clone();
                    exhaustive.sort_by(f32::total_cmp);
                    assert_eq!(
                        kept_distances(&kept, &all, &what),
                        exhaustive[..size],
                        "{what}"
                    );
                }
            }
        }
    }

    /// Where the gate can prune, it prunes by a bound that is only *probably*
    /// right, and this is how wrong it is allowed to be.
    ///
    /// [`crate::lazy::LazyProbe::scan`] asks for a top-`L` rather than for every
    /// distance because the bound `estimate - error_factor * query_error` lets a
    /// vertex already worse than the `L`-th best skip its extra-bit refinement.
    /// The error term is a confidence interval and not a guarantee, so the answer
    /// is an approximation of the top-`L` rather than the top-`L`: measured over
    /// this fixture the two differ in about one run in a hundred, and when they
    /// do it is one vertex, seated a couple of per cent of the partition's own
    /// spread too far out.
    ///
    /// That is worth pinning precisely because the failure it has to be told
    /// apart from looks identical and is far larger. A bound with its slack
    /// removed prunes candidates by the dozen, and neither a single query nor a
    /// recall figure over a whole index separates the two - the first sees
    /// nothing most of the time, and the second averages it away.
    #[test]
    fn a_gate_with_something_to_prune_rarely_misses() {
        const SEEDS: u64 = 100;

        for num_bits in [3u8, 5] {
            for search_list_size in [1usize, 5, 17] {
                for placement in [Placement::OnAVertex, Placement::Elsewhere] {
                    let what =
                        format!("{num_bits} bits, query {placement:?}, list of {search_list_size}");
                    let mut differed = 0u64;
                    let mut worst = 0.0f32;
                    for seed in 0..SEEDS {
                        let (kept, all) = gate(num_bits, seed, placement, search_list_size);
                        let mut exhaustive = all.clone();
                        exhaustive.sort_by(f32::total_cmp);
                        let gated = kept_distances(&kept, &all, &what);
                        if gated[..] == exhaustive[..search_list_size] {
                            continue;
                        }
                        differed += 1;
                        // As a share of the whole partition's spread rather than
                        // of the distance itself, which passes through zero.
                        let spread = exhaustive[ROWS - 1] - exhaustive[0];
                        worst = worst.max(
                            (gated[search_list_size - 1] - exhaustive[search_list_size - 1])
                                / spread,
                        );
                    }
                    // Deterministic, so these are the measured numbers with a
                    // little air rather than a confidence interval: one run in a
                    // hundred differs, by 0.0055 of the spread at the worst. A
                    // bound with its slack removed misses twenty-seven runs in a
                    // hundred, so the two are nowhere near each other.
                    assert!(
                        differed <= 2,
                        "{what}: the gate kept a different list in {differed} runs of {SEEDS}"
                    );
                    assert!(
                        worst <= 0.02,
                        "{what}: the gate's candidate {search_list_size} sat {worst:.4} of the \
                         partition's spread beyond the one it should have kept"
                    );
                }
            }
        }
    }

    /// A code column the partition's own width disagrees with is a corrupt file,
    /// not a caller's mistake, and it must not be read as if it were shorter.
    #[test]
    fn a_code_column_of_the_wrong_stride_is_rejected() {
        let params = CodeParams::mint(3, DIMENSION).unwrap();
        let (vectors, centroid, row_ids) = sample(7);
        let column = encode(&params, DistanceType::L2, &vectors, &centroid).unwrap();

        let error = storage(
            &CodeParams::mint(1, DIMENSION).unwrap(),
            DistanceType::L2,
            DIMENSION,
            &row_ids,
            &column,
        )
        .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("bytes a vertex"), "{error}");
    }

    #[test]
    fn a_partition_file_without_the_column_is_reported_as_corrupt() {
        let batch = RecordBatch::try_from_iter_with_nullable(vec![(
            ROW_ID,
            Arc::new(UInt64Array::from(vec![0u64; ROWS])) as ArrayRef,
            false,
        )])
        .unwrap();
        let error = column(&batch).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("declares codes but"), "{error}");
    }
}
