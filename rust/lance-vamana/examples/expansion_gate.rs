// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Is RaBitQ's error bound informative enough to gate an expansion?
//!
//! After the cross-query cache a query reads two things and nothing else: the
//! out-edges of the vertices it expands, and the vectors of the candidates it
//! ends with. The first is a chain of *dependent* round trips - a hop cannot be
//! issued until the previous hop's neighbours have been scored - so it is what
//! the mode's latency is made of, and cutting expansions is the only way to cut
//! it. RaBitQ carries a per-vector error factor that bounds how far a coded
//! distance can be from the true one, which suggests a gate: do not follow a
//! vertex whose distance, at its most optimistic, still cannot reach the answer.
//!
//! Three things have to be true before that gate is worth building, and this
//! stand measures all three rather than assuming them.
//!
//! - **The bound has to be a bound for the estimate we walk by.** Lance's
//!   `raw_query_lower_bound` subtracts `error_factor * |q - c|` from the *binary*
//!   estimate, and uses it to decide whether computing the multi-bit one is
//!   worth the cycles. We walk by the multi-bit estimate already, so the bound
//!   we would need is one on that. Applying the binary bound to a three-bit
//!   estimate is valid only if the extra bits never move an estimate further
//!   from the truth, which is a claim about data, not about arithmetic.
//! - **It has to be tight enough to fire.** A bound that is ten times the actual
//!   error never excludes anything, and a gate that never fires is a branch in
//!   the hot loop and nothing else.
//! - **It has to say something per vertex.** Within one partition and one query
//!   `|q - c|` is a constant, so every vertex-to-vertex difference in the bound
//!   comes from its error factor. If those are near-constant, the gate is a
//!   threshold on the estimate wearing a bound's clothes - which is a smaller
//!   search list, and a smaller search list is already a knob.
//!
//! The gate itself is not built here and no walk runs. The search list holds the
//! `L` nearest vertices by coded distance and the walk expands all of them, so
//! "which expansions would the gate skip" is answerable from the top `L` of a
//! partition directly, against the tightest threshold any walk could converge
//! to. That makes this an *upper bound* on what the gate could save: a real walk
//! meets each vertex earlier, with a looser threshold, and skips fewer.
//!
//! # The partition gate, and what it costs to decide
//!
//! The first pass found that over half of what the gate could skip is partitions
//! it skips *entirely*, which needs no gate inside the walk at all - only a check
//! before a partition is opened. That check is a different problem, and the
//! second half of this stand measures it.
//!
//! It is different because of what it costs. The quantity the first pass
//! reported is a minimum over a partition's vertices, and a driver that wants it
//! has to spend a coded distance on every one of them: at 8192 rows and 25
//! probes that is 204,800 distances to decide something, against the 4,543 a
//! whole query spends today. So the useful question is not how much the exact
//! minimum would skip, but how much a *cheap* stand-in for it would, and there
//! are three worth asking about:
//!
//! - **A radius bound.** `|q - v|` is at least `|q - c| - max|v - c|`, so a
//!   partition whose nearest possible vertex loses to the answer so far cannot
//!   contribute. It costs nothing: routing has already computed `|q - c|`. Its
//!   catch is `max`, which one outlying vertex is enough to ruin, so two
//!   percentiles are measured beside it.
//! - **A sample.** The minimum over evenly spaced vertices, which bounds nothing
//!   but costs a hundredth of the scan.
//! - **The scan itself**, as the reference the other two are judged against.
//!
//! Whether the radius is even reachable is its own question, and the answer is
//! yes: [`recovered_norm_square`] gets `|v - c|^2` back out of two factors every
//! resident code already carries, so a partition's radius is a pass at load time
//! rather than a new field in the segment table.
//!
//! Two things the first pass left implicit are made explicit here. The threshold
//! a driver actually has is built from *exact* distances, not coded ones, since
//! every walk re-scores its candidates before returning them. And a driver that
//! keeps `PARTITIONS_IN_FLIGHT` probes in flight cannot see the results of the
//! ones still running, so the threshold lags by that many probes - which is
//! measured rather than assumed.
//!
//! ```text
//! SIFT_DIR=~/sift cargo run --release-no-lto --example expansion_gate
//! VECTORS=100000 QUERIES=100 LIST=100 ROWS_PER_PARTITION=8192 ...
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::vector::ApproxMode;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::RabitQuantizationStorage;
use lance_index::vector::bq::transform::{
    ERROR_FACTORS_COLUMN, RQTransformer, SCALE_FACTORS_COLUMN,
};
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::storage::{DistCalculator, DistanceCalculatorOptions, VectorStore};
use lance_index::vector::transform::Transformer;
use lance_index::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};
use lance_io::scheduler::ScanScheduler;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::format::INDEX_FILE_NAME;
use lance_vamana::io::{open_file, read_partition, read_segment, scan_scheduler};
use lance_vamana::partition::Partition;
use lance_vamana::search::flat_storage;
use lance_vamana::segment::{PartitionEntry, SegmentManifest};
use object_store::path::Path;

mod common;
use common::{env_usize, read_fvecs};

const K: usize = 10;
const VECTOR_FIELD: &str = "vector";
const ID_COLUMN: &str = "id";
const INDEX_NAME: &str = "vamana_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;

/// How much of the bound to believe: `skip if est - lambda * err >= threshold`.
///
/// One is the bound as RaBitQ states it. Below one the gate is no longer a
/// bound but a tunable, and has to earn its place against a smaller search list
/// at equal recall. Zero is the degenerate case and exists as a self-check: it
/// skips exactly the vertices ranked at or below the threshold.
const LAMBDAS: [f32; 6] = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0];

/// The constant Lance scales a RaBitQ error factor by, mirrored here because it
/// is private to `bq::transform`.
///
/// Mirroring it is what makes [`recovered_norm_square`] possible and is also its
/// one weakness: the recovery is tied to the exact arithmetic of a module that
/// owes us no stability. A gate built on it either re-derives the constant from
/// the data or stores the radius itself.
const RABIT_ERROR_EPSILON: f32 = 1.9;

/// Vertices a sampling gate measures before deciding, against a partition of
/// thousands.
///
/// Swept rather than fixed because the whole question about a sample is where
/// it stops being cheap and starts being right, and taking that curve in a
/// second run would mean rebuilding the index to get it.
const SAMPLES: [usize; 3] = [16, 64, 256];

/// Whether the answer a probe is measured against is assembled from coded
/// distances or from exact ones.
///
/// Step one measured the coded threshold. The driver's is exact - every walk
/// re-scores its candidate list before returning it - so the exact column is the
/// one an implementation would see, and the coded one is here to line up against
/// what was already reported.
const THRESHOLD_KINDS: [&str; 2] = ["coded", "exact"];

/// How far behind the threshold runs, as a count of probes whose results are not
/// yet visible.
///
/// `buffer_unordered(F)` starts `F` probes at once, so when probe `i` starts,
/// `i - F + 1` have finished and no more. One is the fully serial driver, four
/// is today's `PARTITIONS_IN_FLIGHT`, and the oracle sees every probe including
/// the ones after it.
const LAGS: [(&str, usize); 3] = [("serial", 1), ("in flight 4", 4), ("oracle", 0)];

/// Gate forms that spend a coded distance per vertex they look at, and so carry
/// a lambda: the full scan, then one per entry of [`SAMPLES`].
const SCAN_FORMS: usize = 1 + SAMPLES.len();

fn scan_form_name(form: usize) -> String {
    match form {
        0 => "scan all".to_string(),
        form => format!("sample {}", SAMPLES[form - 1]),
    }
}

/// Gate forms that spend nothing at all: the query-to-centroid distance is
/// already computed at routing, and the radius is a property of the partition.
const RADIUS_FORMS: [&str; 3] = ["radius max", "radius p99", "radius p90"];

fn env_list(name: &str, fallback: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .map(|raw| {
            raw.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a comma-separated list of numbers"))
        })
        .collect()
}

async fn write_dataset(
    uri: &str,
    vectors: FixedSizeListArray,
    rows_per_fragment: usize,
) -> Dataset {
    let rows = vectors.len() as u64;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_FIELD, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            max_rows_per_group: rows_per_fragment.min(8192),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// Which partitions a query would read, in the order the driver would pick them.
fn probe_plan(manifest: &SegmentManifest, query: &ArrayRef, nprobes: usize) -> Vec<u32> {
    let (ranked, _) = manifest
        .ivf()
        .find_partitions(
            query.as_ref(),
            manifest.ivf().num_partitions(),
            DISTANCE_TYPE,
        )
        .unwrap();
    ranked
        .values()
        .iter()
        .filter(|id| manifest.partition(**id).is_some())
        .take(nprobes)
        .copied()
        .collect()
}

/// One partition, with everything a coded distance and an exact one need.
struct Probe {
    partition: Partition,
    exact: FlatFloatStorage,
    centroid: Vec<f32>,
}

async fn load_probes(
    scheduler: &Arc<ScanScheduler>,
    dir: &Path,
    file_sizes: &HashMap<String, u64>,
    manifest: &SegmentManifest,
    wanted: &HashSet<u32>,
) -> HashMap<u32, Probe> {
    let mut probes = HashMap::with_capacity(wanted.len());
    for partition_id in wanted {
        let entry: PartitionEntry = manifest.partition(*partition_id).unwrap().clone();
        let path = dir.clone().join(entry.file.as_str());
        let reader = open_file(scheduler, &path, None, file_sizes.get(&entry.file).copied())
            .await
            .unwrap();
        let partition = read_partition(&reader, entry.num_rows).await.unwrap();
        let exact = flat_storage(
            partition.graph().row_ids(),
            partition.vectors(),
            DISTANCE_TYPE,
        )
        .unwrap();
        let centroid = manifest
            .ivf()
            .centroid(*partition_id as usize)
            .expect("a probed partition has a centroid");
        let centroid = centroid.as_primitive::<Float32Type>().values().to_vec();
        probes.insert(
            *partition_id,
            Probe {
                partition,
                exact,
                centroid,
            },
        );
    }
    probes
}

/// A partition's vectors as residuals against its centroid, with `|v - c|^2`.
fn residuals(probe: &Probe) -> (FixedSizeListArray, Float32Array) {
    let vectors = probe.partition.vectors();
    let dim = vectors.value_length() as usize;
    let values = vectors.values().as_primitive::<Float32Type>().values();
    let mut residuals = Vec::with_capacity(values.len());
    let mut norms = Vec::with_capacity(vectors.len());
    for row in 0..vectors.len() {
        let vector = &values[row * dim..(row + 1) * dim];
        let mut norm = 0.0f32;
        for (value, center) in vector.iter().zip(&probe.centroid) {
            let residual = value - center;
            residuals.push(residual);
            norm += residual * residual;
        }
        norms.push(norm);
    }
    (
        FixedSizeListArray::try_new_from_values(Float32Array::from(residuals), dim as i32).unwrap(),
        Float32Array::from(norms),
    )
}

/// `|v - c|^2` recovered from two of the factors a code already carries.
///
/// A partition-level gate wants the farthest a vertex sits from the centroid,
/// and the cheapest place to get it would be data already resident. RaBitQ's L2
/// factors are `scale = -2n / b` and
/// `error = 2 sqrt(n) EPSILON sqrt((a - 1) / (d - 1))` with `a = n (d/4) / b^2`,
/// where `n` is the norm square and `b` the inner product between the residual
/// and its own binary code. Substituting the first into the second cancels `b` -
/// the one term a reader does not have - and leaves
/// `n = d scale^2 / 16 - error^2 (d - 1) / (4 EPSILON^2)`.
///
/// If that holds numerically, a partition's radius is a pass over the resident
/// codes at load time rather than a new field in the segment table, and the gate
/// needs no format change at all.
fn recovered_norm_square(scale: f32, error: f32, code_dim: usize) -> f32 {
    let dimension = code_dim as f32;
    let from_scale = dimension * scale * scale / 16.0;
    let from_error =
        error * error * (dimension - 1.0) / (4.0 * RABIT_ERROR_EPSILON * RABIT_ERROR_EPSILON);
    (from_scale - from_error).max(0.0)
}

/// How far a partition's vertices sit from its centroid.
///
/// The maximum is the only one of these that bounds anything: `|q - v|` is at
/// least `|q - c| - max`, so a partition whose nearest possible vertex loses to
/// the answer so far cannot contribute to it. The percentiles are the same
/// arithmetic with the guarantee traded away, and they are here because a single
/// outlying vertex is enough to make the maximum useless.
struct Radii {
    max: f32,
    p99: f32,
    p90: f32,
}

impl Radii {
    fn of(norm_squares: &[f32]) -> Self {
        let mut radii = norm_squares
            .iter()
            .map(|norm_square| norm_square.max(0.0).sqrt())
            .collect::<Vec<_>>();
        radii.sort_unstable_by(f32::total_cmp);
        Self {
            max: percentile(&radii, 1.0),
            p99: percentile(&radii, 0.99),
            p90: percentile(&radii, 0.90),
        }
    }
}

/// The nearest a vertex of this partition could possibly be, given `|q - c|^2`.
fn radius_bound(dist_q_c: f32, radius: f32) -> f32 {
    let gap = dist_q_c.max(0.0).sqrt() - radius;
    if gap <= 0.0 { 0.0 } else { gap * gap }
}

/// One partition's codes, its error factors, and what they say about its radius.
struct Quantised {
    store: RabitQuantizationStorage,
    errors: Vec<f32>,
    radii: Radii,
    /// The largest relative disagreement between a recovered `|v - c|^2` and the
    /// measured one over this partition, which is the self-check the radius
    /// forms stand on.
    recovery_error: f32,
}

/// RaBitQ codes for one partition, and the error factor of every vertex.
///
/// The factors come out of the transform's own column rather than out of our
/// stride, so that what is measured is the quantiser's bound and not our
/// packing of it. They are the same bytes either way: a stride carries the
/// binary code, the extended one and five factors, of which this is the third.
fn rabit_store(probe: &Probe, num_bits: u8) -> Quantised {
    let (residuals, norms) = residuals(probe);
    let dim = residuals.value_length();
    let rows = residuals.len();
    let quantizer =
        RabitQuantizer::build(&residuals, DISTANCE_TYPE, &RQBuildParams::new(num_bits)).unwrap();
    let centroid =
        FixedSizeListArray::try_new_from_values(Float32Array::from(probe.centroid.clone()), dim)
            .unwrap();
    let batch = RecordBatch::try_from_iter_with_nullable(vec![
        (
            ROW_ID,
            Arc::new(UInt64Array::from(
                probe.partition.graph().row_ids().to_vec(),
            )) as ArrayRef,
            false,
        ),
        (VECTOR_FIELD, Arc::new(residuals) as ArrayRef, false),
        (
            CENTROID_DIST_COLUMN,
            Arc::new(norms.clone()) as ArrayRef,
            false,
        ),
        (
            PART_ID_COLUMN,
            Arc::new(UInt32Array::from(vec![0u32; rows])) as ArrayRef,
            false,
        ),
    ])
    .unwrap();
    let coded = RQTransformer::new(quantizer.clone(), DISTANCE_TYPE, centroid, VECTOR_FIELD)
        .unwrap()
        .transform(&batch)
        .unwrap();

    let factors = coded
        .column_by_name(ERROR_FACTORS_COLUMN)
        .unwrap_or_else(|| panic!("a {num_bits}-bit RaBitQ code carries an error factor"))
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    let scales = coded
        .column_by_name(SCALE_FACTORS_COLUMN)
        .unwrap_or_else(|| panic!("a {num_bits}-bit RaBitQ code carries a scale factor"))
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();

    // Against the norms measured off the vectors, which is the only thing that
    // says whether the closed form above survives being computed in f32.
    let code_dim = quantizer.metadata(None).code_dim as usize;
    let recovered = scales
        .iter()
        .zip(&factors)
        .map(|(scale, error)| recovered_norm_square(*scale, *error, code_dim))
        .collect::<Vec<_>>();
    let recovery_error = recovered
        .iter()
        .zip(norms.values())
        .map(|(recovered, measured)| {
            if *measured <= 0.0 {
                0.0
            } else {
                (recovered - measured).abs() / measured
            }
        })
        .fold(0.0f32, f32::max);

    let kept = coded
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| {
            !matches!(
                field.name().as_str(),
                VECTOR_FIELD | CENTROID_DIST_COLUMN | PART_ID_COLUMN
            )
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let coded = coded.project(&kept).unwrap();
    let store = RabitQuantizationStorage::try_from_batch(
        coded,
        &quantizer.metadata(None),
        DISTANCE_TYPE,
        None,
    )
    .unwrap();
    Quantised {
        store,
        errors: factors,
        radii: Radii::of(&recovered),
        recovery_error,
    }
}

/// Nanoseconds one coded distance costs, both ways a caller can spend it.
///
/// The number the whole question turns on. A gate that is sound has to take the
/// minimum over every vertex, so its price is a coded distance per vertex per
/// probe, and whether that is worth paying depends entirely on what one costs
/// against the round trip it saves.
///
/// Both ways, because they are not the same price and the two callers are
/// different. A walk asks for the vertices its hop uncovered, one id at a time,
/// which is what `distance` is for. A scan wants all of them, which is what
/// `distance_all` is for - and that one is free to work over the quantiser's
/// blocked layout rather than gathering a vertex at a time, so it should be the
/// cheaper of the two by whatever that layout is worth. Building the calculator
/// is charged to both.
fn coded_distance_cost(probes: &HashMap<u32, Probed>, query: &ArrayRef) -> (f64, f64, usize) {
    let mut vertices = 0usize;
    let single = Instant::now();
    let mut sink = 0.0f32;
    for probed in probes.values() {
        let dist_q_c = centroid_distance(&probed.probe, query);
        let calculator = probed.coded.store.dist_calculator(query.clone(), dist_q_c);
        let count = probed.probe.partition.len();
        for id in 0..count as u32 {
            sink += calculator.distance(id);
        }
        vertices += count;
    }
    let single = single.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    let batched = Instant::now();
    let mut sink = 0usize;
    for probed in probes.values() {
        let dist_q_c = centroid_distance(&probed.probe, query);
        let calculator = probed.coded.store.dist_calculator(query.clone(), dist_q_c);
        sink += calculator.distance_all(K).len();
    }
    let batched = batched.elapsed().as_secs_f64();
    std::hint::black_box(sink);

    (
        single * 1e9 / vertices as f64,
        batched * 1e9 / vertices as f64,
        vertices,
    )
}

/// `|q - c|^2` between a query and one partition's centroid.
///
/// The raw-query estimator folds the centroid into every vertex's own factors,
/// so what the calculator wants is the query itself plus this - and it is also
/// the term the error bound is scaled by, which is why getting it wrong here
/// would show up as a bound that never holds rather than as lost recall.
fn centroid_distance(probe: &Probe, query: &ArrayRef) -> f32 {
    let values = query.as_primitive::<Float32Type>().values();
    values
        .iter()
        .zip(&probe.centroid)
        .map(|(value, center)| (value - center) * (value - center))
        .sum()
}

fn percentile(sorted: &[f32], fraction: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let at = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[at]
}

/// What the gate measures a candidate against: the `K`th best coded distance,
/// but over how much.
///
/// A query walks several partitions, and a candidate that would survive its own
/// partition's threshold can be hopeless against the answer being assembled from
/// all of them. Which of these is reachable is a driver question rather than a
/// quantiser one, so all three are measured.
const THRESHOLDS: [&str; 3] = ["partition", "running", "oracle"];

/// What one granularity's measurement adds up to.
#[derive(Default)]
struct Tally {
    /// `est - err > exact`: the bound did not hold.
    binary_violations: usize,
    coded_violations: usize,
    vertices: usize,
    /// `err`, and the absolute error of both estimates, over every vertex.
    errs: Vec<f32>,
    binary_errors: Vec<f32>,
    coded_errors: Vec<f32>,
    /// Expansions the gate would skip: `[threshold][list size][lambda]`.
    skipped: Vec<Vec<[usize; LAMBDAS.len()]>>,
    /// Of those, the ones in a partition the gate skips *entirely* - the share
    /// of the win that needs no gate in the walk at all, only a check before
    /// the partition is opened.
    skipped_whole: Vec<Vec<[usize; LAMBDAS.len()]>>,
    expansions: Vec<usize>,
    probes: usize,
}

impl Tally {
    fn new(list_sizes: usize) -> Self {
        Self {
            skipped: vec![vec![[0; LAMBDAS.len()]; list_sizes]; THRESHOLDS.len()],
            skipped_whole: vec![vec![[0; LAMBDAS.len()]; list_sizes]; THRESHOLDS.len()],
            expansions: vec![0; list_sizes],
            ..Default::default()
        }
    }

    fn report(&mut self, factors: &[f32], list_sizes: &[usize]) {
        self.errs.sort_unstable_by(f32::total_cmp);
        self.binary_errors.sort_unstable_by(f32::total_cmp);
        self.coded_errors.sort_unstable_by(f32::total_cmp);
        let mut factors = factors.to_vec();
        factors.sort_unstable_by(f32::total_cmp);
        let mean = factors.iter().sum::<f32>() / factors.len() as f32;
        let variance = factors
            .iter()
            .map(|factor| (factor - mean) * (factor - mean))
            .sum::<f32>()
            / factors.len() as f32;

        println!(
            "  error factors   min {:.5} p50 {:.5} max {:.5}, CV {:.3}",
            percentile(&factors, 0.0),
            percentile(&factors, 0.5),
            percentile(&factors, 1.0),
            variance.sqrt() / mean
        );
        println!(
            "  bound holds     binary estimate {:.4}% violated | 3-bit estimate {:.4}% violated",
            100.0 * self.binary_violations as f64 / self.vertices as f64,
            100.0 * self.coded_violations as f64 / self.vertices as f64,
        );
        let err = percentile(&self.errs, 0.5);
        let binary = percentile(&self.binary_errors, 0.5);
        let coded = percentile(&self.coded_errors, 0.5);
        println!(
            "  tightness (p50) err {err:.1} | |binary - exact| {binary:.1} ({:.1}x) | \
             |3-bit - exact| {coded:.1} ({:.1}x)",
            err / binary,
            err / coded,
        );
        println!("  gate: share of expansions skipped, threshold = {K}th best coded distance");
        print!("    {:<11}{:<6}", "threshold", "L");
        for lambda in LAMBDAS {
            print!("{:>10}", format!("λ={lambda}"));
        }
        println!("{:>12}", "expansions");
        for (which, name) in THRESHOLDS.iter().enumerate() {
            for (row, list_size) in list_sizes.iter().enumerate() {
                print!("    {name:<11}{list_size:<6}");
                for index in 0..LAMBDAS.len() {
                    let share = 100.0 * self.skipped[which][row][index] as f64
                        / self.expansions[row] as f64;
                    print!("{share:9.2}%");
                }
                println!("{:>12}", self.expansions[row]);
            }
        }
        println!(
            "  of which in a partition skipped whole, over {} probes",
            self.probes
        );
        for (which, name) in THRESHOLDS.iter().enumerate() {
            for (row, list_size) in list_sizes.iter().enumerate() {
                print!("    {name:<11}{list_size:<6}");
                for index in 0..LAMBDAS.len() {
                    let share = 100.0 * self.skipped_whole[which][row][index] as f64
                        / self.expansions[row] as f64;
                    print!("{share:9.2}%");
                }
                println!();
            }
        }
        println!(
            "    partition = this partition's own {K}th best, running = every partition probed so \
             far, oracle = all of them."
        );
        println!(
            "    λ=0 against the partition threshold is the self-check: (L - k + 1) / L exactly. \
             λ=1 is the bound as RaBitQ states it."
        );
    }
}

/// What a check *before* a partition is opened could skip.
///
/// Separate from [`Tally`] because it counts a different thing. That one counts
/// expansions, and its "skipped whole" row asks whether a partition's top `L`
/// all fail the threshold - which is the right question for a walk that has
/// already opened the partition, and the wrong one for a gate that decides
/// before it does. A gate that decides first has to be sound over *every*
/// vertex, since it has no way to know which ones a walk would have reached.
struct ProbeTally {
    probes: usize,
    /// `[threshold kind][lag][form][lambda]`, over [`SCAN_FORMS`].
    scanned: [[[[usize; LAMBDAS.len()]; SCAN_FORMS]; LAGS.len()]; THRESHOLD_KINDS.len()],
    /// `[threshold kind][lag][form]`, over [`RADIUS_FORMS`], which carry no
    /// lambda: there is no error term in a distance to a centroid.
    radius: [[[usize; RADIUS_FORMS.len()]; LAGS.len()]; THRESHOLD_KINDS.len()],
    /// Probes where the radius bound is a positive number at all, per form.
    ///
    /// A bound of zero excludes nothing, and it is zero whenever the query sits
    /// closer to the centroid than the partition's own edge. Counted separately
    /// so that a radius form reading zero says *why* it reads zero rather than
    /// leaving it to be guessed.
    reachable: [usize; RADIUS_FORMS.len()],
    /// `|q - c|` and the radius it is measured against, over every probe.
    centroid_distances: Vec<f32>,
    radii: Vec<f32>,
    /// The worst relative disagreement between a recovered `|v - c|^2` and the
    /// measured one, over every partition probed.
    recovery_error: f32,
}

impl ProbeTally {
    fn new() -> Self {
        Self {
            probes: 0,
            scanned: [[[[0; LAMBDAS.len()]; SCAN_FORMS]; LAGS.len()]; THRESHOLD_KINDS.len()],
            radius: [[[0; RADIUS_FORMS.len()]; LAGS.len()]; THRESHOLD_KINDS.len()],
            reachable: [0; RADIUS_FORMS.len()],
            centroid_distances: Vec::new(),
            radii: Vec::new(),
            recovery_error: 0.0,
        }
    }

    fn report(&mut self, working: usize) {
        self.centroid_distances.sort_unstable_by(f32::total_cmp);
        self.radii.sort_unstable_by(f32::total_cmp);
        println!(
            "  |v - c|^2 recovered from (scale, error): worst relative error {:.3e}",
            self.recovery_error
        );
        println!(
            "  |q - c| (p50) {:.1} against a partition radius (p50) of {:.1}; the radius bound is \
             a positive number on {:.2}% of probes at max, {:.2}% at p99, {:.2}% at p90",
            percentile(&self.centroid_distances, 0.5),
            percentile(&self.radii, 0.5),
            self.share(self.reachable[0]),
            self.share(self.reachable[1]),
            self.share(self.reachable[2]),
        );
        println!(
            "  partition gate: share of {} probes a check before opening would skip, L = {working}",
            self.probes
        );
        print!("    {:<10}{:<13}{:<11}", "threshold", "lag", "form");
        for lambda in LAMBDAS {
            print!("{:>10}", format!("λ={lambda}"));
        }
        println!();
        for (kind, kind_name) in THRESHOLD_KINDS.iter().enumerate() {
            for (lag, (lag_name, _)) in LAGS.iter().enumerate() {
                for form in 0..SCAN_FORMS {
                    print!(
                        "    {kind_name:<10}{lag_name:<13}{:<11}",
                        scan_form_name(form)
                    );
                    for index in 0..LAMBDAS.len() {
                        print!("{:9.2}%", self.share(self.scanned[kind][lag][form][index]));
                    }
                    println!();
                }
            }
        }
        println!("  the same for the forms that cost nothing, and so carry no λ");
        print!("    {:<10}{:<13}", "threshold", "lag");
        for name in RADIUS_FORMS {
            print!("{name:>13}");
        }
        println!();
        for (kind, kind_name) in THRESHOLD_KINDS.iter().enumerate() {
            for (lag, (lag_name, _)) in LAGS.iter().enumerate() {
                print!("    {kind_name:<10}{lag_name:<13}");
                for form in 0..RADIUS_FORMS.len() {
                    print!("{:12.2}%", self.share(self.radius[kind][lag][form]));
                }
                println!();
            }
        }
        println!(
            "    serial is a driver that probes one partition at a time, in flight 4 is today's \
             PARTITIONS_IN_FLIGHT, oracle sees every probe including the ones after it."
        );
        println!(
            "    coded is the threshold step one measured; exact is the one a driver has, because \
             every walk re-scores its candidates before returning them."
        );
    }

    fn share(&self, count: usize) -> f64 {
        100.0 * count as f64 / self.probes as f64
    }
}

/// One partition, its codes, and the error factor of each of its vertices.
///
/// The store and the factors have to come from the *same* build: a RaBitQ
/// rotation is minted at random, so a second quantiser over the same vectors
/// produces codes and factors that are each self-consistent and mean nothing
/// together.
struct Probed {
    probe: Probe,
    coded: Quantised,
}

/// One vertex as the gate sees it: its coded distance, its error factor and the
/// truth the two are approximating.
type Scored = (f32, f32, f32);

/// The `K`th smallest distance in a set of lists, under whichever of the two a
/// caller asks for, or `None` before there are `K` of them - a threshold nobody
/// has reached yet gates nothing.
fn kth_best(lists: &[&[Scored]], of: fn(&Scored) -> f32) -> Option<f32> {
    let mut all = lists
        .iter()
        .flat_map(|list| list.iter().map(of))
        .collect::<Vec<_>>();
    if all.len() < K {
        return None;
    }
    all.select_nth_unstable_by(K - 1, f32::total_cmp);
    Some(all[K - 1])
}

/// What one probe's partition offers a gate that runs before the walk.
struct Bounds {
    /// One per [`SCAN_FORMS`]: the minimum of `est - lambda * err` over every
    /// vertex, then over evenly spaced samples of it.
    ///
    /// Only the first bounds anything. Evenly spaced rather than drawn at
    /// random so that a rerun of this stand reproduces its own numbers.
    scanned: [[f32; LAMBDAS.len()]; SCAN_FORMS],
    /// `(|q - c| - r)^2` for each radius in [`RADIUS_FORMS`], which costs nothing
    /// beyond the centroid distance routing has already computed.
    radius: [f32; RADIUS_FORMS.len()],
}

/// Measure one query against every partition it would probe.
fn measure(
    probes: &HashMap<u32, Probed>,
    plan: &[u32],
    query: &ArrayRef,
    list_sizes: &[usize],
    tally: &mut Tally,
    gate: &mut ProbeTally,
) {
    let mut scratch = Vec::new();
    let mut scored_by_probe = Vec::with_capacity(plan.len());
    let mut bounds_by_probe = Vec::with_capacity(plan.len());
    for partition_id in plan {
        let Probed { probe, coded } = &probes[partition_id];
        let Quantised {
            store,
            errors,
            radii,
            ..
        } = coded;
        let vertices = probe.partition.len();
        let dist_q_c = centroid_distance(probe, query);
        let exact = probe.exact.dist_calculator(query.clone(), 0.0);
        let coded = store.dist_calculator(query.clone(), dist_q_c);
        let mut scored = Vec::with_capacity(vertices);
        let mut scanned = [[f32::INFINITY; LAMBDAS.len()]; SCAN_FORMS];
        let steps = SAMPLES.map(|sample| (vertices / sample).max(1));
        {
            let binary = store.dist_calculator_with_scratch(
                query.clone(),
                dist_q_c,
                None,
                &mut scratch,
                DistanceCalculatorOptions {
                    approx_mode: ApproxMode::Fast,
                },
            );
            for id in 0..vertices as u32 {
                let truth = exact.distance(id);
                let binary = binary.distance(id);
                let coded = coded.distance(id);
                let err = errors[id as usize] * dist_q_c.max(0.0).sqrt();
                tally.vertices += 1;
                tally.binary_violations += usize::from(binary - err > truth);
                tally.coded_violations += usize::from(coded - err > truth);
                tally.errs.push(err);
                tally.binary_errors.push((binary - truth).abs());
                tally.coded_errors.push((coded - truth).abs());
                for (index, lambda) in LAMBDAS.iter().enumerate() {
                    let bound = coded - lambda * err;
                    scanned[0][index] = scanned[0][index].min(bound);
                    for (form, (step, sample)) in steps.iter().zip(SAMPLES).enumerate() {
                        if (id as usize).is_multiple_of(*step) && (id as usize) / step < sample {
                            scanned[form + 1][index] = scanned[form + 1][index].min(bound);
                        }
                    }
                }
                scored.push((coded, err, truth));
            }
        }
        let bounds = Bounds {
            scanned,
            radius: [
                radius_bound(dist_q_c, radii.max),
                radius_bound(dist_q_c, radii.p99),
                radius_bound(dist_q_c, radii.p90),
            ],
        };
        gate.centroid_distances.push(dist_q_c.max(0.0).sqrt());
        gate.radii.push(radii.max);
        for (form, bound) in bounds.radius.iter().enumerate() {
            gate.reachable[form] += usize::from(*bound > 0.0);
        }
        bounds_by_probe.push(bounds);
        // The search list is the `L` nearest by coded distance and the walk
        // expands all of them, so the head of this *is* the expansion set - at
        // the threshold a walk converges to rather than the looser ones it
        // passes through.
        scored.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
        scored_by_probe.push(scored);
    }

    for (row, list_size) in list_sizes.iter().enumerate() {
        let lists = scored_by_probe
            .iter()
            .map(|scored| &scored[..(*list_size).min(scored.len())])
            .collect::<Vec<_>>();
        let oracle = kth_best(&lists, |scored| scored.0);
        for (at, list) in lists.iter().enumerate() {
            let thresholds = [
                kth_best(&lists[at..=at], |scored| scored.0),
                kth_best(&lists[..=at], |scored| scored.0),
                oracle,
            ];
            tally.expansions[row] += list.len();
            tally.probes += usize::from(row == 0);
            for (which, threshold) in thresholds.iter().enumerate() {
                let Some(threshold) = threshold else {
                    continue;
                };
                for (index, lambda) in LAMBDAS.iter().enumerate() {
                    let skipped = list
                        .iter()
                        .filter(|(coded, err, _)| coded - lambda * err >= *threshold)
                        .count();
                    tally.skipped[which][row][index] += skipped;
                    if skipped == list.len() {
                        tally.skipped_whole[which][row][index] += skipped;
                    }
                }
            }
        }
    }

    // The partition gate is measured at one list size rather than swept: the
    // threshold is the `K`th best and `K` is under every `L` in the sweep, so
    // the coded threshold does not move with `L` at all, and the exact one moves
    // only through which rows a longer list re-scores.
    let working = list_sizes[0];
    let lists = scored_by_probe
        .iter()
        .map(|scored| &scored[..working.min(scored.len())])
        .collect::<Vec<_>>();
    let projections: [fn(&Scored) -> f32; THRESHOLD_KINDS.len()] =
        [|scored| scored.0, |scored| scored.2];
    for (at, bounds) in bounds_by_probe.iter().enumerate() {
        gate.probes += 1;
        for (kind, of) in projections.iter().enumerate() {
            for (lag, (_, behind)) in LAGS.iter().enumerate() {
                let visible = match behind {
                    0 => &lists[..],
                    behind => &lists[..at.saturating_sub(behind - 1)],
                };
                let Some(threshold) = kth_best(visible, *of) else {
                    continue;
                };
                for (form, bounds) in bounds.scanned.iter().enumerate() {
                    for (index, bound) in bounds.iter().enumerate() {
                        gate.scanned[kind][lag][form][index] += usize::from(*bound >= threshold);
                    }
                }
                for (form, bound) in bounds.radius.iter().enumerate() {
                    gate.radius[kind][lag][form] += usize::from(*bound >= threshold);
                }
            }
        }
    }
}

async fn granularity(
    vectors: &FixedSizeListArray,
    queries: &[ArrayRef],
    rows_per_partition: usize,
    list_sizes: &[usize],
    degree: u32,
    rows_per_fragment: usize,
    probe_counts: &[usize],
) {
    let rows = vectors.len();
    let partitions = rows.div_ceil(rows_per_partition).max(1) as u32;
    // Deduplicated after capping, because a budget larger than the index has
    // partitions is the same measurement twice.
    let mut probe_counts = probe_counts
        .iter()
        .map(|nprobes| (*nprobes).clamp(1, partitions as usize))
        .collect::<Vec<_>>();
    probe_counts.sort_unstable();
    probe_counts.dedup();
    let widest = *probe_counts.last().expect("at least one probe count");
    let temp = tempfile::tempdir().unwrap();
    let uri = temp.path().to_str().unwrap();
    let mut dataset = write_dataset(uri, vectors.clone(), rows_per_fragment).await;
    let started = Instant::now();
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_FIELD, partitions)
            .with_distance_type(DISTANCE_TYPE)
            .with_graph_params(BuildParams {
                max_degree: degree,
                ..Default::default()
            }),
    )
    .await
    .unwrap();
    println!(
        "\n=== {rows_per_partition} rows a partition: {partitions} partitions, {probe_counts:?} \
         probed, built in {:.1}s ===",
        started.elapsed().as_secs_f64()
    );

    let committed = dataset
        .load_indices_by_name(INDEX_NAME)
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let segment_dir = dataset.indices_dir().join(committed.uuid.to_string());
    let file_sizes = committed
        .files
        .iter()
        .flatten()
        .map(|file| (file.path.clone(), file.size_bytes))
        .collect::<HashMap<_, _>>();
    let scheduler = scan_scheduler(&dataset.object_store(None).await.unwrap());
    let manifest = read_segment(
        &scheduler,
        &segment_dir,
        file_sizes.get(INDEX_FILE_NAME).copied(),
    )
    .await
    .unwrap();

    // Planned once at the widest budget and truncated per sweep point, so that
    // every budget probes the same partitions in the same order and the sweep
    // measures the budget rather than a reshuffle.
    let plans = queries
        .iter()
        .map(|query| probe_plan(&manifest, query, widest))
        .collect::<Vec<_>>();
    let wanted = plans.iter().flatten().copied().collect::<HashSet<_>>();
    let probes = load_probes(&scheduler, &segment_dir, &file_sizes, &manifest, &wanted).await;
    let started = Instant::now();
    let probes = probes
        .into_iter()
        .map(|(partition_id, probe)| {
            let coded = rabit_store(&probe, 3);
            (partition_id, Probed { probe, coded })
        })
        .collect::<HashMap<_, _>>();
    println!(
        "  {} partitions probed, coded in {:.1}s",
        probes.len(),
        started.elapsed().as_secs_f64()
    );

    let (single, batched, vertices) = coded_distance_cost(&probes, &queries[0]);
    println!(
        "  one coded distance costs {single:.1} ns one at a time, {batched:.1} ns a whole \
         partition at once, over {vertices} vertices"
    );
    let nanoseconds = batched;

    let recovery_error = probes
        .values()
        .map(|probed| probed.coded.recovery_error)
        .fold(0.0f32, f32::max);
    let factors = probes
        .values()
        .flat_map(|probed| probed.coded.errors.iter().copied())
        .collect::<Vec<_>>();

    for nprobes in probe_counts {
        let scan = rows_per_partition * nprobes;
        println!(
            "\n-- {nprobes} probes: a sound gate scans {scan} vertices a query, {:.0} us of \
             arithmetic at the batched price --",
            scan as f64 * nanoseconds / 1000.0
        );
        let mut tally = Tally::new(list_sizes.len());
        let mut gate = ProbeTally::new();
        gate.recovery_error = recovery_error;
        for (query, plan) in queries.iter().zip(&plans) {
            measure(
                &probes,
                &plan[..nprobes.min(plan.len())],
                query,
                list_sizes,
                &mut tally,
                &mut gate,
            );
        }
        tally.report(&factors, list_sizes);
        gate.report(list_sizes[0]);
    }
}

#[tokio::main]
async fn main() {
    let dir =
        std::env::var("SIFT_DIR").expect("set SIFT_DIR to the directory holding sift_*.fvecs");
    let (base, dim, total) = read_fvecs(&format!("{dir}/sift_base.fvecs"));
    let (query_values, query_dim, total_queries) = read_fvecs(&format!("{dir}/sift_query.fvecs"));
    assert_eq!(dim, query_dim);

    let requested = env_usize("VECTORS", 100_000);
    let rows = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let num_queries = env_usize("QUERIES", 100).min(total_queries);
    let sweep = env_list("ROWS_PER_PARTITION", "1000,8192,65536");
    let list_sizes = env_list("LIST", "20,40,100,200");
    let degree = env_usize("DEGREE", 64) as u32;
    let rows_per_fragment = env_usize("ROWS_PER_FRAGMENT", 10_000);
    let probe_percent = env_usize("PROBE_PERCENT", 20);
    // A budget in partitions rather than a share of them, because the point that
    // has to be measured is the one a driver would set, and that is read off a
    // recall curve rather than off the partition count. Absent, the share
    // stands, which is what every earlier run of this stand used.
    let probe_counts = std::env::var("NPROBES")
        .is_ok()
        .then(|| env_list("NPROBES", ""));

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();
    let queries = (0..num_queries)
        .map(|q| {
            Arc::new(Float32Array::from(
                query_values[q * dim..(q + 1) * dim].to_vec(),
            )) as ArrayRef
        })
        .collect::<Vec<_>>();
    println!(
        "SIFT {rows} x {dim}, {num_queries} queries, k = {K}, R = {degree}, L = {list_sizes:?}, \
         3-bit codes"
    );

    for rows_per_partition in sweep {
        let partitions = rows.div_ceil(rows_per_partition).max(1);
        let counts = probe_counts
            .clone()
            .unwrap_or_else(|| vec![((probe_percent * partitions).div_ceil(100)).max(1)]);
        granularity(
            &vectors,
            &queries,
            rows_per_partition,
            &list_sizes,
            degree,
            rows_per_fragment,
            &counts,
        )
        .await;
    }
}
