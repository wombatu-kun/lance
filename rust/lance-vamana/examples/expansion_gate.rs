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
use lance_index::vector::bq::transform::{ERROR_FACTORS_COLUMN, RQTransformer};
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

/// RaBitQ codes for one partition, and the error factor of every vertex.
///
/// The factors come out of the transform's own column rather than out of our
/// stride, so that what is measured is the quantiser's bound and not our
/// packing of it. They are the same bytes either way: a stride carries the
/// binary code, the extended one and five factors, of which this is the third.
fn rabit_store(probe: &Probe, num_bits: u8) -> (RabitQuantizationStorage, Vec<f32>) {
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
        (CENTROID_DIST_COLUMN, Arc::new(norms) as ArrayRef, false),
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
    (store, factors)
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

/// One partition, its codes, and the error factor of each of its vertices.
///
/// The store and the factors have to come from the *same* build: a RaBitQ
/// rotation is minted at random, so a second quantiser over the same vectors
/// produces codes and factors that are each self-consistent and mean nothing
/// together.
struct Probed {
    probe: Probe,
    store: RabitQuantizationStorage,
    factors: Vec<f32>,
}

/// The `K`th smallest coded distance in a set of lists, or `None` before there
/// are `K` of them - a threshold nobody has reached yet gates nothing.
fn threshold_of(lists: &[&[(f32, f32)]]) -> Option<f32> {
    let mut all = lists
        .iter()
        .flat_map(|list| list.iter().map(|(coded, _)| *coded))
        .collect::<Vec<_>>();
    if all.len() < K {
        return None;
    }
    all.select_nth_unstable_by(K - 1, f32::total_cmp);
    Some(all[K - 1])
}

/// Measure one query against every partition it would probe.
fn measure(
    probes: &HashMap<u32, Probed>,
    plan: &[u32],
    query: &ArrayRef,
    list_sizes: &[usize],
    tally: &mut Tally,
) {
    let mut scratch = Vec::new();
    let mut scored_by_probe = Vec::with_capacity(plan.len());
    for partition_id in plan {
        let Probed {
            probe,
            store,
            factors,
        } = &probes[partition_id];
        let vertices = probe.partition.len();
        let dist_q_c = centroid_distance(probe, query);
        let exact = probe.exact.dist_calculator(query.clone(), 0.0);
        let coded = store.dist_calculator(query.clone(), dist_q_c);
        let mut scored = Vec::with_capacity(vertices);
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
                let err = factors[id as usize] * dist_q_c.max(0.0).sqrt();
                tally.vertices += 1;
                tally.binary_violations += usize::from(binary - err > truth);
                tally.coded_violations += usize::from(coded - err > truth);
                tally.errs.push(err);
                tally.binary_errors.push((binary - truth).abs());
                tally.coded_errors.push((coded - truth).abs());
                scored.push((coded, err));
            }
        }
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
        let oracle = threshold_of(&lists);
        for (at, list) in lists.iter().enumerate() {
            let thresholds = [
                threshold_of(&lists[at..=at]),
                threshold_of(&lists[..=at]),
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
                        .filter(|(coded, err)| coded - lambda * err >= *threshold)
                        .count();
                    tally.skipped[which][row][index] += skipped;
                    if skipped == list.len() {
                        tally.skipped_whole[which][row][index] += skipped;
                    }
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
    probe_percent: usize,
) {
    let rows = vectors.len();
    let partitions = rows.div_ceil(rows_per_partition).max(1) as u32;
    let nprobes = ((probe_percent * partitions as usize).div_ceil(100)).max(1);
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
        "\n=== {rows_per_partition} rows a partition: {partitions} partitions, {nprobes} probed, \
         built in {:.1}s ===",
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

    let plans = queries
        .iter()
        .map(|query| probe_plan(&manifest, query, nprobes))
        .collect::<Vec<_>>();
    let wanted = plans.iter().flatten().copied().collect::<HashSet<_>>();
    let probes = load_probes(&scheduler, &segment_dir, &file_sizes, &manifest, &wanted).await;
    let started = Instant::now();
    let probes = probes
        .into_iter()
        .map(|(partition_id, probe)| {
            let (store, factors) = rabit_store(&probe, 3);
            (
                partition_id,
                Probed {
                    probe,
                    store,
                    factors,
                },
            )
        })
        .collect::<HashMap<_, _>>();
    println!(
        "  {} partitions probed, coded in {:.1}s",
        probes.len(),
        started.elapsed().as_secs_f64()
    );

    let mut tally = Tally::new(list_sizes.len());
    for (query, plan) in queries.iter().zip(&plans) {
        measure(&probes, plan, query, list_sizes, &mut tally);
    }
    let factors = probes
        .values()
        .flat_map(|probed| probed.factors.iter().copied())
        .collect::<Vec<_>>();
    tally.report(&factors, list_sizes);
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
        granularity(
            &vectors,
            &queries,
            rows_per_partition,
            &list_sizes,
            degree,
            rows_per_fragment,
            probe_percent,
        )
        .await;
    }
}
