// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! B0: does the in-partition graph earn its keep?
//!
//! Sweeps `target_partition_size` across `IVF_PQ` / `IVF_RQ` / `IVF_HNSW_SQ` on
//! a texmex-format dataset and reports recall@k against the work each family
//! actually does: bytes read from the index files, distance comparisons, and
//! wall-clock latency, under a ladder of index-cache budgets.
//!
//! The shipped defaults differ by 128-256x (`IVF_PQ` 8192, `IVF_RQ` 4096,
//! `IVF_HNSW_*` 1 << 20), so on a 1M-row table the default `IVF_HNSW_SQ`
//! resolves to a single partition. This sweep walks the graph index down to
//! where the scan indexes live and back up again.
//!
//! Research harness: not a PR, not production code.
//!
//! Run:
//!   cargo run --release -p lance --example b0_partition_size_sweep -- \
//!     --data-dir ~/datasets/sift --work-dir /tmp/b0 --out /tmp/b0/results.json

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::types::UInt32Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, UInt32Array, cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use clap::Parser;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
use lance_index::IndexType;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use serde_json::{Value, json};

const INDEX_NAME: &str = "b0_vec_idx";
const VECTOR_COLUMN: &str = "vector";
const ID_COLUMN: &str = "id";

#[derive(Parser, Debug)]
#[command(about = "B0 target_partition_size sweep: graph vs scan inside an IVF partition")]
struct Args {
    /// Directory holding <prefix>_base.fvecs / _query.fvecs / _groundtruth.ivecs
    #[arg(long)]
    data_dir: String,

    /// Directory for the generated Lance dataset (created if absent)
    #[arg(long)]
    work_dir: String,

    /// Where to write the JSON results
    #[arg(long)]
    out: String,

    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Queries used for the recall pass (recall is cache-independent, so this
    /// runs once per configuration at an unbounded cache).
    #[arg(long, default_value_t = 10_000)]
    recall_queries: usize,

    /// Queries used per (configuration, cache budget) cell in the cost pass.
    #[arg(long, default_value_t = 500)]
    cost_queries: usize,

    /// Leading cost-pass queries discarded so the cache reaches steady state.
    #[arg(long, default_value_t = 100)]
    cost_warmup: usize,

    /// target_partition_size rungs, comma separated
    #[arg(long, value_delimiter = ',')]
    rungs: Option<Vec<usize>>,

    /// Index families to sweep, comma separated: pq, rq, sq, hnsw_sq
    #[arg(long, value_delimiter = ',')]
    families: Option<Vec<String>>,

    /// ef values swept for the HNSW family, comma separated
    #[arg(long, value_delimiter = ',')]
    efs: Option<Vec<usize>>,

    /// Rebuild the Lance dataset even if it already exists
    #[arg(long, default_value_t = false)]
    rewrite_dataset: bool,
}

/// (flat values, dim, count). Same reader as `acorn_bench_sift`.
fn read_fvecs(path: &Path) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert_eq!(bytes.len() % record, 0);
    let count = bytes.len() / record;
    let mut values = Vec::with_capacity(count * dim);
    for row in 0..count {
        let start = row * record + 4;
        for i in 0..dim {
            let offset = start + i * 4;
            values.push(f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }
    }
    (values, dim, count)
}

fn read_ivecs(path: &Path) -> Vec<Vec<u32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert_eq!(bytes.len() % record, 0);
    (0..bytes.len() / record)
        .map(|row| {
            let start = row * record + 4;
            (0..dim)
                .map(|i| {
                    let offset = start + i * 4;
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
                })
                .collect()
        })
        .collect()
}

fn texmex_prefix(dir: &Path) -> String {
    dir.file_name()
        .and_then(|name| name.to_str())
        .expect("data-dir must have a file name")
        .to_string()
}

fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| {
            let path = entry.path();
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => dir_bytes(&path),
                Ok(_) => std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                Err(_) => 0,
            }
        })
        .sum()
}

/// Bytes belonging to *this* index only. Scoping by uuid matters because
/// `create_index(replace = true)` leaves the superseded configuration's files
/// behind in `_indices`, and a whole-directory sum would carry every previously
/// swept configuration into the cache-budget arithmetic.
async fn index_bytes_on_disk(dataset: &Dataset, dataset_dir: &Path, name: &str) -> u64 {
    let indices = dataset.load_indices_by_name(name).await.unwrap();
    assert!(!indices.is_empty(), "index {name} vanished after creation");
    indices
        .iter()
        .map(|meta| dir_bytes(&dataset_dir.join("_indices").join(meta.uuid.to_string())))
        .sum()
}

/// The statistics blob's shape is not part of any stability contract, so find
/// the key rather than assume a path; a silently missing count would collapse
/// every nprobe ladder to `[1]`.
fn find_usize(value: &Value, key: &str) -> Option<usize> {
    match value {
        Value::Object(map) => {
            if let Some(found) = map.get(key).and_then(Value::as_u64) {
                return Some(found as usize);
            }
            map.values().find_map(|v| find_usize(v, key))
        }
        Value::Array(items) => items.iter().find_map(|v| find_usize(v, key)),
        _ => None,
    }
}

async fn ensure_dataset(args: &Args, dir: &Path) -> (Dataset, PathBuf, usize, usize) {
    let prefix = texmex_prefix(dir);
    let base_path = dir.join(format!("{prefix}_base.fvecs"));
    let dataset_uri = PathBuf::from(&args.work_dir).join(format!("{prefix}.lance"));

    if args.rewrite_dataset && dataset_uri.exists() {
        std::fs::remove_dir_all(&dataset_uri).expect("remove existing dataset");
    }

    if let Ok(dataset) = Dataset::open(dataset_uri.to_str().unwrap()).await {
        let rows = dataset.count_rows(None).await.unwrap();
        let dim = match dataset.schema().field(VECTOR_COLUMN).unwrap().data_type() {
            DataType::FixedSizeList(_, d) => d as usize,
            other => panic!("unexpected vector column type {other:?}"),
        };
        println!(
            "reusing dataset at {} ({rows} rows, dim {dim})",
            dataset_uri.display()
        );
        return (dataset, dataset_uri, rows, dim);
    }

    println!("reading {}", base_path.display());
    let (values, dim, count) = read_fvecs(&base_path);

    let vector_field = Field::new(
        VECTOR_COLUMN,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        false,
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ID_COLUMN, DataType::UInt32, false),
        vector_field,
    ]));

    // 100k-row batches keep the writer's working set modest without making the
    // fragment layout an interesting variable.
    const BATCH_ROWS: usize = 100_000;
    let mut batches = Vec::new();
    for start in (0..count).step_by(BATCH_ROWS) {
        let end = (start + BATCH_ROWS).min(count);
        let ids = UInt32Array::from_iter_values(start as u32..end as u32);
        let slice = Float32Array::from(values[start * dim..end * dim].to_vec());
        let vectors = FixedSizeListArray::try_new_from_values(slice, dim as i32).unwrap();
        batches.push(
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)]).unwrap(),
        );
    }
    drop(values);

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    println!("writing {} rows to {}", count, dataset_uri.display());
    let dataset = Dataset::write(reader, dataset_uri.to_str().unwrap(), Some(write_params))
        .await
        .unwrap();

    let rows = dataset.count_rows(None).await.unwrap();
    assert_eq!(rows, count, "row count must survive the write");
    (dataset, dataset_uri, rows, dim)
}

fn index_params(family: &str, rung: usize, dim: usize) -> VectorIndexParams {
    // `with_target_partition_size` leaves `num_partitions` unset, which is the
    // only branch in `build_ivf_model` that consults the target size at all.
    let ivf = IvfBuildParams::with_target_partition_size(rung);
    match family {
        "pq" => {
            let pq = PQBuildParams {
                num_bits: 8,
                num_sub_vectors: (dim / 8).max(1),
                ..Default::default()
            };
            VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf, pq)
        }
        "rq" => {
            VectorIndexParams::with_ivf_rq_params(MetricType::L2, ivf, RQBuildParams::default())
        }
        // The controlled arm. `sq` and `hnsw_sq` share a quantizer, a partition
        // layout and a byte budget for codes, so the only thing that differs is
        // flat scan versus graph inside the partition -- which is the question.
        // Comparing `hnsw_sq` against `pq`/`rq` instead confounds the graph with
        // an 8x difference in bytes retained per vector.
        "sq" => {
            VectorIndexParams::with_ivf_sq_params(MetricType::L2, ivf, SQBuildParams::default())
        }
        "hnsw_sq" => VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::L2,
            ivf,
            HnswBuildParams::default(),
            SQBuildParams::default(),
        ),
        other => panic!("unknown family {other}"),
    }
}

#[derive(Default)]
struct QueryStats {
    bytes_read: usize,
    iops: usize,
    requests: usize,
    parts_loaded: usize,
    partitions_searched: usize,
    comparisons: usize,
    cache_hits: usize,
    cache_misses: usize,
}

/// The promoted metrics live in typed fields; `all_counts` holds only the ones
/// added after the struct was frozen. Reading everything out of `all_counts`
/// silently yields zeros for exactly the numbers this sweep is about.
fn counts_to_stats(counts: &ExecutionSummaryCounts) -> QueryStats {
    QueryStats {
        bytes_read: counts.bytes_read,
        iops: counts.iops,
        requests: counts.requests,
        parts_loaded: counts.parts_loaded,
        partitions_searched: counts
            .all_counts
            .get(PARTITIONS_SEARCHED_METRIC)
            .copied()
            .unwrap_or_default(),
        comparisons: counts.index_comparisons,
        cache_hits: counts.index_cache_hits(),
        cache_misses: counts.index_cache_misses(),
    }
}

/// Runs one query and returns (returned ids, per-query stats, elapsed micros).
#[allow(clippy::too_many_arguments)]
async fn run_query(
    dataset: &Dataset,
    query: &Float32Array,
    k: usize,
    nprobe: usize,
    ef: Option<usize>,
) -> (Vec<u32>, QueryStats, f64) {
    let holder: Arc<Mutex<Option<ExecutionSummaryCounts>>> = Arc::new(Mutex::new(None));
    let sink = holder.clone();
    let callback: ExecutionStatsCallback = Arc::new(move |stats| {
        *sink.lock().unwrap() = Some(stats.clone());
    });

    let mut scan = dataset.scan();
    scan.nearest(VECTOR_COLUMN, query, k).unwrap();
    // Pin the probed set: `minimum_nprobes` alone fixes it when no prefilter is
    // present, but pinning both removes any doubt about adaptive widening.
    scan.minimum_nprobes(nprobe);
    scan.maximum_nprobes(nprobe);
    if let Some(ef) = ef {
        scan.ef(ef);
    }
    scan.project(&[ID_COLUMN]).unwrap();
    scan.scan_stats_callback(callback);

    let started = Instant::now();
    let batch = scan.try_into_batch().await.unwrap();
    let elapsed_us = started.elapsed().as_secs_f64() * 1e6;

    let ids = batch[ID_COLUMN]
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    let stats = holder
        .lock()
        .unwrap()
        .take()
        .map(|counts| counts_to_stats(&counts))
        .unwrap_or_default();
    (ids, stats, elapsed_us)
}

/// Exact KNN through the scan path, bypassing every index. Used only as the
/// id-mapping gate: it must reproduce the official ground truth exactly, and if
/// it does not then `id` is not the write ordinal and every recall below is
/// meaningless.
async fn exact_topk(dataset: &Dataset, query: &Float32Array, k: usize) -> Vec<u32> {
    let mut scan = dataset.scan();
    scan.nearest(VECTOR_COLUMN, query, k).unwrap();
    scan.use_index(false);
    scan.project(&[ID_COLUMN]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    batch[ID_COLUMN]
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec()
}

fn recall_at_k(got: &[u32], truth: &[u32], k: usize) -> f64 {
    let hits = truth[..k.min(truth.len())]
        .iter()
        .filter(|id| got.contains(id))
        .count();
    hits as f64 / k as f64
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// nprobe ladder for a rung, capped at the resolved partition count so no arm
/// silently asks for more partitions than exist.
fn nprobe_ladder(num_partitions: usize) -> Vec<usize> {
    let mut ladder: Vec<usize> = [1usize, 2, 4, 8, 16, 32, 64]
        .into_iter()
        .filter(|n| *n < num_partitions)
        .collect();
    ladder.push(num_partitions.min(64));
    ladder.dedup();
    ladder
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let data_dir = PathBuf::from(shellexpand_home(&args.data_dir));
    let prefix = texmex_prefix(&data_dir);

    std::fs::create_dir_all(&args.work_dir).expect("create work dir");

    let (base_dataset, dataset_uri, num_rows, dim) = ensure_dataset(&args, &data_dir).await;

    let (query_values, query_dim, query_count) =
        read_fvecs(&data_dir.join(format!("{prefix}_query.fvecs")));
    assert_eq!(query_dim, dim, "query dim must match base dim");
    let ground_truth = read_ivecs(&data_dir.join(format!("{prefix}_groundtruth.ivecs")));
    assert_eq!(ground_truth.len(), query_count);

    let queries: Vec<Float32Array> = (0..query_count)
        .map(|i| Float32Array::from(query_values[i * dim..(i + 1) * dim].to_vec()))
        .collect();

    let rungs = args
        .rungs
        .clone()
        .unwrap_or_else(|| vec![8192, 32768, 131_072, 1_048_576]);
    let families = args
        .families
        .clone()
        .unwrap_or_else(|| vec!["pq".into(), "rq".into(), "sq".into(), "hnsw_sq".into()]);
    let efs = args.efs.clone().unwrap_or_else(|| vec![15, 64, 256]);

    let recall_n = args.recall_queries.min(query_count);
    let cost_n = args.cost_queries.min(query_count);

    println!(
        "dataset: {num_rows} rows, dim {dim}, {query_count} queries, k={}",
        args.k
    );
    println!("rungs: {rungs:?}  families: {families:?}  efs (hnsw): {efs:?}");

    for qi in [0usize, 1, 2] {
        let got = exact_topk(&base_dataset, &queries[qi], args.k).await;
        let exact_recall = recall_at_k(&got, &ground_truth[qi], args.k);
        assert!(
            (exact_recall - 1.0).abs() < 1e-9,
            "id-mapping gate failed on query {qi}: exact search recall@{} = {exact_recall}, \
             so the `id` column is not the fvecs ordinal",
            args.k
        );
    }
    println!("id-mapping gate: exact search reproduces the official ground truth");
    drop(base_dataset);

    let mut records: Vec<Value> = Vec::new();

    for family in &families {
        for &rung in &rungs {
            let params = index_params(family, rung, dim);
            let mut dataset = Dataset::open(dataset_uri.to_str().unwrap()).await.unwrap();

            let build_started = Instant::now();
            dataset
                .create_index(
                    &[VECTOR_COLUMN],
                    IndexType::Vector,
                    Some(INDEX_NAME.to_string()),
                    &params,
                    true,
                )
                .await
                .unwrap();
            let build_secs = build_started.elapsed().as_secs_f64();

            let stats_json = dataset.index_statistics(INDEX_NAME).await.unwrap();
            let stats: Value = serde_json::from_str(&stats_json).unwrap();
            let num_partitions = find_usize(&stats, "num_partitions")
                .unwrap_or_else(|| panic!("no num_partitions in index statistics: {stats_json}"));
            assert!(
                num_partitions > 0,
                "index statistics reported zero partitions: {stats_json}"
            );
            let idx_bytes = index_bytes_on_disk(&dataset, &dataset_uri, INDEX_NAME).await;
            assert!(idx_bytes > 0, "index reported zero bytes on disk");

            println!(
                "\n== {family} @ target_partition_size={rung}: {num_partitions} partitions, \
                 {:.1} MiB on disk, built in {build_secs:.1}s",
                idx_bytes as f64 / (1024.0 * 1024.0)
            );

            let ladder = nprobe_ladder(num_partitions.max(1));
            let ef_ladder: Vec<Option<usize>> = if family == "hnsw_sq" {
                efs.iter().map(|e| Some(*e)).collect()
            } else {
                vec![None]
            };

            // ---- recall pass: cache-independent, so run it once, warm.
            let warm = DatasetBuilder::from_uri(dataset_uri.to_str().unwrap())
                .with_index_cache_size_bytes(idx_bytes as usize * 2 + (1 << 20))
                .load()
                .await
                .unwrap();

            for &nprobe in &ladder {
                for &ef in &ef_ladder {
                    let mut recalls = Vec::with_capacity(recall_n);
                    let mut comparisons = Vec::with_capacity(recall_n);
                    let mut latencies = Vec::with_capacity(recall_n);
                    let mut searched = Vec::with_capacity(recall_n);
                    for qi in 0..recall_n {
                        let (ids, qstats, us) =
                            run_query(&warm, &queries[qi], args.k, nprobe, ef).await;
                        recalls.push(recall_at_k(&ids, &ground_truth[qi], args.k));
                        comparisons.push(qstats.comparisons as f64);
                        searched.push(qstats.partitions_searched as f64);
                        latencies.push(us);
                    }
                    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    records.push(json!({
                        "pass": "recall",
                        "family": family,
                        "target_partition_size": rung,
                        "num_partitions": num_partitions,
                        "index_bytes_on_disk": idx_bytes,
                        "build_secs": build_secs,
                        "nprobe": nprobe,
                        "ef": ef,
                        "queries": recall_n,
                        "recall": mean(&recalls),
                        "comparisons_mean": mean(&comparisons),
                        "partitions_searched_mean": mean(&searched),
                        "latency_p50_us": percentile(&latencies, 0.50),
                        "latency_p99_us": percentile(&latencies, 0.99),
                    }));
                    println!(
                        "  recall  nprobe={nprobe:<3} ef={:<5} recall@{}={:.4} cmp={:.0} \
                         searched={:.2} p50={:.0}us",
                        ef.map(|e| e.to_string()).unwrap_or_else(|| "-".into()),
                        args.k,
                        mean(&recalls),
                        mean(&comparisons),
                        mean(&searched),
                        percentile(&latencies, 0.50),
                    );
                }
            }
            drop(warm);

            // ---- cost pass: the cache ladder, bytes and latency only.
            let budgets: Vec<(&str, usize)> = vec![
                ("full", idx_bytes as usize * 2 + (1 << 20)),
                ("quarter", (idx_bytes / 4) as usize),
                ("sixteenth", (idx_bytes / 16) as usize),
                ("zero", 0),
            ];

            for (budget_name, budget_bytes) in budgets {
                let ds = DatasetBuilder::from_uri(dataset_uri.to_str().unwrap())
                    .with_index_cache_size_bytes(budget_bytes)
                    .load()
                    .await
                    .unwrap();

                for &nprobe in &ladder {
                    for &ef in &ef_ladder {
                        // One full-width probe first: without it the "full"
                        // budget still pays first-touch loads for partitions the
                        // warm-up stream happened to miss, and the arm that is
                        // supposed to mean "cache never evicts" would report
                        // residual bytes that are really cold-start cost.
                        run_query(&ds, &queries[0], args.k, num_partitions, ef).await;

                        let mut bytes = Vec::with_capacity(cost_n);
                        let mut iops = Vec::with_capacity(cost_n);
                        let mut requests = Vec::with_capacity(cost_n);
                        let mut parts = Vec::with_capacity(cost_n);
                        let mut hits = Vec::with_capacity(cost_n);
                        let mut misses = Vec::with_capacity(cost_n);
                        let mut latencies = Vec::with_capacity(cost_n);
                        for qi in 0..cost_n {
                            let (_ids, qstats, us) =
                                run_query(&ds, &queries[qi], args.k, nprobe, ef).await;
                            if qi < args.cost_warmup {
                                continue;
                            }
                            bytes.push(qstats.bytes_read as f64);
                            iops.push(qstats.iops as f64);
                            requests.push(qstats.requests as f64);
                            parts.push(qstats.parts_loaded as f64);
                            hits.push(qstats.cache_hits as f64);
                            misses.push(qstats.cache_misses as f64);
                            latencies.push(us);
                        }
                        assert!(
                            !bytes.is_empty(),
                            "cost-warmup {} consumed every one of {cost_n} queries",
                            args.cost_warmup
                        );
                        // With no index cache every query must reload its
                        // partitions, so zero bytes here means the instrument is
                        // broken rather than the I/O being free. This is the gate
                        // that catches reading a promoted metric out of the
                        // wrong bucket.
                        if budget_name == "zero" {
                            assert!(
                                mean(&bytes) > 0.0 && mean(&parts) > 0.0,
                                "zero-cache arm reported no I/O: bytes={} parts={}",
                                mean(&bytes),
                                mean(&parts)
                            );
                        }
                        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        records.push(json!({
                            "pass": "cost",
                            "family": family,
                            "target_partition_size": rung,
                            "num_partitions": num_partitions,
                            "index_bytes_on_disk": idx_bytes,
                            "cache_budget": budget_name,
                            "cache_budget_bytes": budget_bytes,
                            "nprobe": nprobe,
                            "ef": ef,
                            "queries": bytes.len(),
                            "bytes_read_mean": mean(&bytes),
                            "iops_mean": mean(&iops),
                            "requests_mean": mean(&requests),
                            "parts_loaded_mean": mean(&parts),
                            "cache_hits_mean": mean(&hits),
                            "cache_misses_mean": mean(&misses),
                            "latency_p50_us": percentile(&latencies, 0.50),
                            "latency_p99_us": percentile(&latencies, 0.99),
                        }));
                        println!(
                            "  cost    cache={budget_name:<10} nprobe={nprobe:<3} ef={:<5} \
                             bytes={:>9.0} iops={:>5.1} parts={:>5.2} hit/miss={:.1}/{:.1} \
                             p50={:.0}us",
                            ef.map(|e| e.to_string()).unwrap_or_else(|| "-".into()),
                            mean(&bytes),
                            mean(&iops),
                            mean(&parts),
                            mean(&hits),
                            mean(&misses),
                            percentile(&latencies, 0.50),
                        );
                    }
                }
            }

            std::fs::write(
                &args.out,
                serde_json::to_string_pretty(&json!({
                    "dataset": prefix,
                    "num_rows": num_rows,
                    "dim": dim,
                    "k": args.k,
                    "records": records,
                }))
                .unwrap(),
            )
            .expect("write results");
        }
    }

    println!("\nwrote {} records to {}", records.len(), args.out);
}

/// `--data-dir ~/datasets/sift` is convenient enough to be worth the four lines.
fn shellexpand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => path.to_string(),
    }
}
