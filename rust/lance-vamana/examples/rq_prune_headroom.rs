// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How much of `IVF_RQ`'s ex-code rerank Lance's lower-bound gate already
//! throws away, swept over `k` and `refine_factor`.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift VECTORS=0 DATASET_DIR=~/vamana-bench-prune \
//!   LANCE_RQ_PRUNE_STATS=1 LANCE_RQ_PRUNE_STATS_INTERVAL=1 \
//!   cargo run --profile release-no-lto --example rq_prune_headroom
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 200), `ROWS_PER_PARTITION` (default 8192), `NPROBES`
//! (default 7), `CODE_BITS` (default 3), `KS` (default `10,100,1000,10000`),
//! `REFINES` (default `1,2,4,8,16`), `CACHE_MB` (default 4096), `WARMUP`
//! (default 8), `DATASET_DIR` (unset: a temporary directory thrown away at the
//! end), `ORACLE` (default 0) and `ORACLE_SLACK` (default 1.000001).
//!
//! **What this measures, and what it does not.** Not time. Multi-bit `IVF_RQ`
//! answers a partition in two steps: a binary FastScan ranks every row by its
//! one-bit code, and only rows that survive a per-row distance lower bound go
//! on to the ex-code rerank, which reads the extra bits. The gate is that
//! survival test, and Lance tallies it itself behind `LANCE_RQ_PRUNE_STATS`.
//! The column that answers "is there anything left to win" is `exact_ratio`,
//! the share of scanned rows that pay for the rerank today. It is a count, not
//! a duration, so it reproduces to the digit; a duration here would be the
//! wrong instrument twice over, because nothing has been changed yet and
//! because the rerank reads extra bits of an already open partition rather
//! than original vectors.
//!
//! **Why the sweep is over these two knobs.** `rust/lance/src/index/vector/ivf/v2.rs`
//! asks the sub-index for `query.k * refine_factor` neighbours, so the gate's
//! heap holds `k_eff = k * refine_factor` entries and its threshold is the
//! `k_eff`-th best distance seen so far. Both knobs move one quantity, and the
//! grid is here to check whether the tallies depend on anything but that
//! product.
//!
//! **The warm-up that bounds every cell.** `heap_threshold` is `None` until the
//! heap is full, so the first `k_eff` rows of a scan are reranked
//! unconditionally, in storage order rather than in any useful order. No
//! threshold can prune them, which caps the prune ratio at `1 - k_eff / rows
//! scanned` before any question about the gate's quality is asked. A cell whose
//! `k_eff` exceeds the rows a query scans is therefore degenerate by
//! construction, and the table says so rather than reporting a zero that reads
//! like a verdict.
//!
//! **`ORACLE` measures the ceiling without patching Lance.** A range query's
//! `upper_bound` reaches the exact field a computed threshold would occupy
//! (`Query.upper_bound` -> `FlatQueryParams` -> `accumulate_topk_with_scratch`
//! -> `RawQueryTopkContext::query_upper_bound`), so handing it the true k-th
//! neighbour distance simulates a perfect threshold and moves rows into the
//! `pruned_upper_bound` column. That is an optimistic bound rather than an
//! achievable one: a real threshold can never be tighter than the k-th distance
//! of the set actually searched, and this one is the k-th distance of the whole
//! dataset. `ORACLE_SLACK` scales it; the default clears the strict `<` test at
//! the k-th neighbour itself, and a value like `1.1` asks how much a loose
//! threshold gives up.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::scanner::Scanner;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::IndexType;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use std::sync::Arc;

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const ID_COLUMN: &str = "id";
const VECTOR_FIELD: &str = "vector";
const RQ_INDEX: &str = "rq_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;

/// Keeps Lance's prune tallies instead of printing them, so a cell's numbers
/// are the difference between two snapshots of one cumulative counter.
///
/// Printing them would not do: the counters are process-global and never reset,
/// `LANCE_RQ_PRUNE_STATS_INTERVAL=1` emits a line per partition scan, and the
/// measurement's own table has to share the same stdout.
struct PruneStatsCapture;

static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

impl log::Log for PruneStatsCapture {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.target() == "lance_index::vector::bq::prune_stats"
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            CAPTURED.lock().unwrap().push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

#[derive(Clone, Copy, Default)]
struct Tallies {
    candidates: u64,
    pruned_upper_bound: u64,
    pruned_heap: u64,
    exact: u64,
}

impl Tallies {
    fn since(self, before: Self) -> Self {
        Self {
            candidates: self.candidates - before.candidates,
            pruned_upper_bound: self.pruned_upper_bound - before.pruned_upper_bound,
            pruned_heap: self.pruned_heap - before.pruned_heap,
            exact: self.exact - before.exact,
        }
    }

    fn ratio(numerator: u64, denominator: u64) -> f64 {
        match denominator {
            0 => 0.0,
            _ => numerator as f64 / denominator as f64,
        }
    }
}

/// The cumulative tallies as of the last line captured, and how many bypass
/// lines have been seen.
///
/// A bypass line is not a detail: it says the gate was not running at all, and
/// a zero in the table would then mean the opposite of what it reads.
fn snapshot() -> (Tallies, usize) {
    let captured = CAPTURED.lock().unwrap();
    let mut tallies = Tallies::default();
    let mut bypasses = 0;
    for line in captured.iter() {
        if line.starts_with("ivf_rq_prune_stats_bypass") {
            bypasses += 1;
        } else if let Some(fields) = line.strip_prefix("ivf_rq_prune_stats ") {
            tallies = parse_tallies(fields);
        }
    }
    (tallies, bypasses)
}

fn parse_tallies(fields: &str) -> Tallies {
    let mut tallies = Tallies::default();
    for field in fields.split_whitespace() {
        let Some((name, value)) = field.split_once('=') else {
            continue;
        };
        let count = || {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("{name} is not a count: {value}"))
        };
        match name {
            "candidates" => tallies.candidates = count(),
            "pruned_upper_bound" => tallies.pruned_upper_bound = count(),
            "pruned_heap" => tallies.pruned_heap = count(),
            "exact" => tallies.exact = count(),
            _ => {}
        }
    }
    tallies
}

/// Exact nearest `k` of one query by brute force, ascending, as
/// `(distance, base position)`.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef, k: usize) -> Vec<(f32, u64)> {
    let calculator = store.dist_calculator(query, 0.0);
    let mut scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id as u64))
        .collect::<Vec<_>>();
    let nth = k.min(scored.len() - 1);
    scored.select_nth_unstable_by(nth, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(k.min(scored.len()));
    scored.sort_by(|left, right| left.0.total_cmp(&right.0));
    scored
}

async fn write_dataset(uri: &str, vectors: &FixedSizeListArray) -> Dataset {
    let rows = vectors.len() as u64;
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt64, false),
        Field::new(VECTOR_FIELD, vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(vectors.clone()),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams::default()),
    )
    .await
    .unwrap()
}

/// The base-vector position of every row, keyed by the address a search answers
/// in.
async fn positions_by_address(dataset: &Dataset) -> HashMap<u64, u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project(&[ID_COLUMN]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .zip(batch[ID_COLUMN].as_primitive::<UInt64Type>().values())
        .map(|(address, id)| (*address, *id))
        .collect()
}

fn rq_scanner(
    dataset: &Dataset,
    query: &[f32],
    k: usize,
    nprobes: usize,
    refine: u32,
    upper_bound: Option<f32>,
) -> Scanner {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.empty_project().unwrap();
    scanner.nearest(VECTOR_FIELD, &key, k).unwrap();
    scanner.nprobes(nprobes);
    scanner.fast_search();
    // Set unconditionally, `1` included: `refine_factor` of `Some(1)` and `None`
    // reach the sub-index as the same `k_eff`, and a cell that set it one way
    // and its neighbours the other would differ by a plan node rather than by
    // the knob the column names.
    scanner.refine(refine);
    if let Some(bound) = upper_bound {
        scanner.distance_range(None, Some(bound));
    }
    scanner.with_row_id();
    scanner
}

async fn neighbors(
    dataset: &Dataset,
    query: &[f32],
    k: usize,
    nprobes: usize,
    refine: u32,
    upper_bound: Option<f32>,
) -> Vec<u64> {
    let batch = rq_scanner(dataset, query, k, nprobes, refine, upper_bound)
        .try_into_batch()
        .await
        .unwrap();
    batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec()
}

fn env_f64(name: &str, fallback: f64) -> f64 {
    std::env::var(name)
        .ok()
        .map(|raw| {
            raw.parse()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(fallback)
}

fn env_list(name: &str, fallback: &str) -> Vec<usize> {
    let raw = std::env::var(name).unwrap_or_else(|_| fallback.to_string());
    raw.split(',')
        .map(|item| {
            item.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} is a comma-separated list of numbers"))
        })
        .collect()
}

#[tokio::main]
async fn main() {
    assert!(
        std::env::var("LANCE_RQ_PRUNE_STATS").is_ok_and(|value| !matches!(
            value.to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        )),
        "set LANCE_RQ_PRUNE_STATS=1: this example reads Lance's own tallies and has nothing to \
         report without them"
    );
    assert_eq!(
        std::env::var("LANCE_RQ_PRUNE_STATS_INTERVAL")
            .ok()
            .as_deref(),
        Some("1"),
        "set LANCE_RQ_PRUNE_STATS_INTERVAL=1: the default emits one line per 1024 partition \
         scans, so a cell shorter than that would be read as the previous cell's totals"
    );
    static LOGGER: PruneStatsCapture = PruneStatsCapture;
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    let dir = std::env::var("SIFT_DIR").expect("set SIFT_DIR to the extracted dataset directory");
    let prefix = std::path::Path::new(&dir)
        .file_name()
        .and_then(|name| name.to_str())
        .expect("SIFT_DIR must end in the dataset name")
        .to_string();
    let (base, dim, total) = read_fvecs(&format!("{dir}/{prefix}_base.fvecs"));
    let (query_values, query_dim, total_queries) =
        read_fvecs(&format!("{dir}/{prefix}_query.fvecs"));
    assert_eq!(dim, query_dim);

    let requested = env_usize("VECTORS", 100_000);
    let rows = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let num_queries = env_usize("QUERIES", 200).min(total_queries);
    let rows_per_partition = env_usize("ROWS_PER_PARTITION", 8192);
    let partitions = rows.div_ceil(rows_per_partition).max(1) as u32;
    let nprobes = env_usize("NPROBES", 7);
    let code_bits = env_usize("CODE_BITS", 3) as u8;
    let ks = env_list("KS", "10,100,1000,10000");
    let refines = env_list("REFINES", "1,2,4,8,16");
    let cache_bytes = env_usize("CACHE_MB", 4096) << 20;
    let warmup = env_usize("WARMUP", 8).min(num_queries);
    let oracle = env_usize("ORACLE", 0) != 0;
    let oracle_slack = env_f64("ORACLE_SLACK", 1.000_001);
    let k_max = *ks.iter().max().expect("KS names at least one k");
    assert!(k_max < rows, "every k must be smaller than the row count");

    // A probe budget the largest cell cannot fill makes every number in that row
    // a statement about the warm-up rather than about the gate, so the shape is
    // printed beside the table and each cell carries its own ceiling.
    let scanned_estimate = (nprobes.min(partitions as usize) * rows_per_partition).min(rows);

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();
    let queries = (0..num_queries)
        .map(|i| query_values[i * dim..(i + 1) * dim].to_vec())
        .collect::<Vec<_>>();

    let scratch = std::env::var("DATASET_DIR").ok();
    let temporary = scratch.is_none().then(|| tempfile::tempdir().unwrap());
    let root = scratch.unwrap_or_else(|| {
        temporary
            .as_ref()
            .unwrap()
            .path()
            .to_string_lossy()
            .into_owned()
    });
    let uri = format!("{root}/{prefix}-{rows}-p{rows_per_partition}-rq{code_bits}.lance");

    println!(
        "{prefix} {rows} x {dim}, IVF_RQ on {code_bits} bits, {partitions} partitions of about \
         {rows_per_partition}, {nprobes} probes, {num_queries} queries, cache {} MB, warmup {}",
        cache_bytes >> 20,
        warmup
    );
    println!(
        "about {scanned_estimate} rows scanned per query{}",
        match oracle {
            true => format!(", oracle upper bound at {oracle_slack} x the true k-th distance"),
            false => String::new(),
        }
    );

    if std::fs::metadata(&uri).is_ok() {
        let dataset = Dataset::open(&uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), rows);
        println!("reusing the IVF_RQ index at {uri}");
    } else {
        let mut dataset = write_dataset(&uri, &vectors).await;
        let started = Instant::now();
        dataset
            .create_index(
                &[VECTOR_FIELD],
                IndexType::IvfRq,
                Some(RQ_INDEX.to_string()),
                &VectorIndexParams::with_ivf_rq_params(
                    DISTANCE_TYPE,
                    IvfBuildParams::new(partitions as usize),
                    RQBuildParams {
                        num_bits: code_bits,
                        ..Default::default()
                    },
                ),
                false,
            )
            .await
            .unwrap();
        println!(
            "IVF_RQ indexed in {:.1}s at {uri}",
            started.elapsed().as_secs_f64()
        );
    }

    let store = FlatFloatStorage::new(vectors.clone(), DISTANCE_TYPE);
    let started = Instant::now();
    let truth = queries
        .iter()
        .map(|query| {
            exact_top(
                &store,
                Arc::new(Float32Array::from(query.clone())) as ArrayRef,
                k_max,
            )
        })
        .collect::<Vec<_>>();
    println!(
        "brute force ground truth to k = {k_max} in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let dataset = DatasetBuilder::from_uri(&uri)
        .with_index_cache_size_bytes(cache_bytes)
        .load()
        .await
        .unwrap();
    let positions = positions_by_address(&dataset).await;

    for query in queries.iter().take(warmup) {
        neighbors(&dataset, query, *ks.first().unwrap(), nprobes, 1, None).await;
    }

    println!(
        "\n{:>7} {:>7} {:>9} {:>8} {:>12} {:>10} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "k",
        "refine",
        "k_eff",
        "ceiling",
        "candidates",
        "pruned_ub",
        "pruned_heap",
        "prune_ratio",
        "exact",
        "exact_r",
        "recall"
    );
    // Flushed per row: a cell of the widest sweep runs for minutes, and a
    // block-buffered stdout would hold the whole table until the process ends.
    std::io::stdout().flush().unwrap();

    for k in &ks {
        // One set per query per `k`, reused across the refine row: the truth
        // prefix does not depend on how wide the candidate list is.
        let wanted = truth
            .iter()
            .map(|exact| {
                exact[..*k]
                    .iter()
                    .map(|(_, position)| *position)
                    .collect::<HashSet<_>>()
            })
            .collect::<Vec<_>>();

        for refine in &refines {
            let k_eff = k * refine;
            let (before, bypasses_before) = snapshot();
            let started = Instant::now();
            let mut found = 0.0;
            for (index, (query, wanted)) in queries.iter().zip(&wanted).enumerate() {
                let bound = oracle.then(|| truth[index][*k - 1].0 * oracle_slack as f32);
                let answer = neighbors(&dataset, query, *k, nprobes, *refine as u32, bound).await;
                found += answer
                    .iter()
                    .filter_map(|address| positions.get(address))
                    .filter(|position| wanted.contains(position))
                    .count() as f64;
            }
            let (after, bypasses_after) = snapshot();
            assert_eq!(
                bypasses_before, bypasses_after,
                "the gate was bypassed during k = {k}, refine = {refine}: a zero in this row \
                 would mean the gate never ran, not that it found nothing to prune"
            );
            let cell = after.since(before);
            let pruned = cell.pruned_upper_bound + cell.pruned_heap;
            println!(
                "{:>7} {:>7} {:>9} {:>8.4} {:>12} {:>10} {:>12} {:>12.6} {:>12} {:>8.6} {:>8.4} {:>7.1}s{}",
                k,
                refine,
                k_eff,
                1.0 - Tallies::ratio(k_eff as u64, scanned_estimate as u64),
                cell.candidates,
                cell.pruned_upper_bound,
                cell.pruned_heap,
                Tallies::ratio(pruned, cell.candidates),
                cell.exact,
                Tallies::ratio(cell.exact, cell.candidates),
                found / (num_queries * k) as f64,
                started.elapsed().as_secs_f64(),
                match k_eff >= scanned_estimate {
                    true =>
                        "  <- k_eff exceeds the rows a query scans: the heap never fills and \
                             no threshold exists",
                    false => "",
                }
            );
            std::io::stdout().flush().unwrap();
        }
    }
}
