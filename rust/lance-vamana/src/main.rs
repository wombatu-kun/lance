// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A command line for the Vamana index.
//!
//! Argument parsing over calls the library exports, and nothing else. A binary
//! target is a separate crate, so what this reaches is the public API an
//! out-of-tree consumer sees.

mod fvecs;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, UInt64Array,
};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use clap::{Args, Parser, Subcommand, ValueEnum};
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::{CacheStats, LanceCache};
use lance_core::{Error, ROW_ID, Result};
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::consolidator::consolidate_index;
use lance_vamana::inserter::{insert_as_segment, insert_in_place};
use lance_vamana::merger::merge_index;
use lance_vamana::query::{QueryResult, SearchParams, VamanaIndex, WalkMode};
use serde::Serialize;

use fvecs::{Fvecs, read_ivecs};

#[derive(Parser)]
#[command(
    name = "vamana",
    version,
    about = "Build, query and maintain a Vamana vector index over a Lance dataset"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write a `.fvecs` file out as a Lance dataset.
    Ingest(IngestArgs),
    /// Build an index over a dataset's vector column and commit it.
    Build(BuildArgs),
    /// Answer queries against a committed index.
    Search(SearchArgs),
    /// Index the rows the index does not cover yet, as a segment beside the base.
    Insert(InsertArgs),
    /// Apply everything pending in one pass, leaving one segment. The only call
    /// that removes delta segments.
    Merge(TargetArgs),
    /// Take the dataset's deleted rows out of the graphs that still hold them.
    /// Cheaper than a merge, and right when deletions are all that is pending.
    Consolidate(TargetArgs),
    /// Print what a committed index is made of.
    Info(InfoArgs),
}

#[derive(Args)]
struct TargetArgs {
    /// The dataset holding the index.
    #[arg(long, value_name = "URI")]
    dataset: String,
    /// Name the index was committed under.
    #[arg(long, value_name = "NAME")]
    index_name: String,
}

#[derive(Args)]
struct IngestArgs {
    /// The `.fvecs` file to read.
    #[arg(long, value_name = "PATH")]
    fvecs: PathBuf,
    /// Where to write the dataset.
    #[arg(long, value_name = "URI")]
    dataset: String,
    /// Name of the vector column to write.
    #[arg(long, default_value = "vector", value_name = "NAME")]
    column: String,
    /// Name of the position column to write, which `search --truth` scores by.
    #[arg(long, default_value = "id", value_name = "NAME")]
    id_column: String,
    /// Stop after this many vectors, rather than at the end of the file.
    #[arg(long, value_name = "N")]
    rows: Option<usize>,
    /// Rows per record batch handed to the writer.
    #[arg(long, default_value_t = 8192, value_name = "N")]
    batch_rows: usize,
}

#[derive(Args)]
#[command(group(clap::ArgGroup::new("granularity").required(true).args(["partitions", "rows_per_partition"])))]
struct BuildArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// Vector column to index. Must be `FixedSizeList<Float32, dim>`.
    #[arg(long, default_value = "vector", value_name = "NAME")]
    column: String,
    /// IVF partitions to train, i.e. how many k-means centroids.
    #[arg(long, value_name = "N")]
    partitions: Option<u32>,
    /// Partitions to train, given as rows each should hold on average.
    #[arg(long, value_name = "N")]
    rows_per_partition: Option<usize>,
    #[arg(long, value_enum, default_value_t = MetricArg::L2)]
    metric: MetricArg,
    /// Bits a dimension for the resident code column. Omitted, no codes are
    /// written and only `--mode exact` can search the index.
    #[arg(long, value_name = "BITS")]
    code_bits: Option<u8>,
    /// `R`: the fixed width of every vertex's neighbour list.
    #[arg(short = 'R', long, default_value_t = BuildParams::default().max_degree, value_name = "N")]
    max_degree: u32,
    /// `L`: the beam each build-time search keeps.
    #[arg(long, default_value_t = BuildParams::default().search_list_size, value_name = "N")]
    search_list_size: usize,
    /// Pruning slack for the second pass. `1.0` reproduces the HNSW heuristic.
    #[arg(long, default_value_t = BuildParams::default().alpha, value_name = "F")]
    alpha: f32,
    /// Insertion order. Varying it is a deliberate act; a build is reproducible
    /// by default.
    #[arg(long, default_value_t = BuildParams::default().seed, value_name = "N")]
    seed: u64,
    #[arg(long, default_value_t = IndexParams::new("", 1).kmeans_max_iters, value_name = "N")]
    kmeans_iters: u32,
    #[arg(long, default_value_t = IndexParams::new("", 1).kmeans_sample_rate, value_name = "N")]
    kmeans_sample_rate: usize,
}

#[derive(Args)]
#[command(group(clap::ArgGroup::new("queries").required(true).args(["fvecs", "vector"])))]
struct SearchArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// A `.fvecs` file of queries.
    #[arg(long, value_name = "PATH")]
    fvecs: Option<PathBuf>,
    /// One query, as comma-separated numbers.
    #[arg(long, value_name = "X,Y,...", allow_hyphen_values = true)]
    vector: Option<String>,
    /// Answer only the first this many queries of the file.
    #[arg(long, value_name = "N", conflicts_with = "vector")]
    limit: Option<usize>,
    /// Neighbours to return.
    #[arg(short = 'k', long, default_value_t = 10, value_name = "N")]
    k: usize,
    /// Partitions to open per segment.
    #[arg(long, default_value_t = 1, value_name = "N")]
    nprobes: usize,
    /// `L`: how wide a search list a walk keeps, or how many candidates a scan
    /// keeps out of the whole partition. Defaults to `k + k/2`.
    #[arg(short = 'L', long, value_name = "N")]
    search_list_size: Option<usize>,
    #[arg(long, value_enum, default_value_t = ModeArg::Exact)]
    mode: ModeArg,
    /// `W`: vertices one hop of a lazy walk expands at a time.
    #[arg(short = 'W', long, default_value_t = 4, value_name = "N")]
    beam_width: usize,
    /// Candidates measured exactly, counted across every partition a query
    /// probes rather than within each of them.
    #[arg(long, value_name = "N")]
    rescore_budget: Option<usize>,
    /// Megabytes of partition codes to keep across queries. Zero holds nothing.
    #[arg(long, default_value_t = 0, value_name = "MB")]
    cache_mb: usize,
    /// Queries to answer and discard before measuring, so that a cache is
    /// reported warm rather than in its first second.
    #[arg(long, default_value_t = 0, value_name = "N")]
    warmup: usize,
    /// An `.ivecs` ground truth to score recall against. Its rows are
    /// base-vector positions, so `--id-column` must name them, and its `n`th
    /// row is the `n`th query of `--fvecs`.
    #[arg(long, value_name = "PATH", conflicts_with = "vector")]
    truth: Option<PathBuf>,
    /// Column holding each row's position in the base file.
    #[arg(long, default_value = "id", value_name = "NAME")]
    id_column: String,
    /// Also fetch these columns for every neighbour returned.
    #[arg(long, value_name = "COL,COL")]
    take: Option<String>,
    /// Print one JSON object instead of a table. It carries every answer; the
    /// table prints answers only for a single query.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct InsertArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// Grow the base segment's own graphs instead of adding a segment beside it.
    #[arg(long)]
    in_place: bool,
}

#[derive(Args)]
struct InfoArgs {
    #[command(flatten)]
    target: TargetArgs,
    #[arg(long)]
    json: bool,
}

/// `Dot` and `Hamming` are refused by `builder::supported_distance_type`, so
/// they are not offered: better a parser error than one after an open.
#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    L2,
    Cosine,
}

impl From<MetricArg> for DistanceType {
    fn from(metric: MetricArg) -> Self {
        match metric {
            MetricArg::L2 => Self::L2,
            MetricArg::Cosine => Self::Cosine,
        }
    }
}

/// Mirrors [`WalkMode`]; deriving on that would put `clap` in the library's API.
#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Exact,
    Coded,
    Lazy,
    Flat,
}

impl From<ModeArg> for WalkMode {
    fn from(mode: ModeArg) -> Self {
        match mode {
            ModeArg::Exact => Self::Exact,
            ModeArg::Coded => Self::Coded,
            ModeArg::Lazy => Self::Lazy,
            ModeArg::Flat => Self::Flat,
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vamana: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Ingest(args) => ingest(args).await,
        Command::Build(args) => build(args).await,
        Command::Search(args) => search(args).await,
        Command::Insert(args) => insert(args).await,
        Command::Merge(target) => merge(target).await,
        Command::Consolidate(target) => consolidate(target).await,
        Command::Info(args) => info(args).await,
    }
}

async fn ingest(args: IngestArgs) -> Result<()> {
    if args.batch_rows == 0 {
        return Err(Error::invalid_input(
            "--batch-rows must be at least one".to_string(),
        ));
    }
    if args.rows == Some(0) {
        return Err(Error::invalid_input(
            "--rows must be at least one; a dataset of no rows cannot be indexed".to_string(),
        ));
    }
    let mut file = Fvecs::open(&args.fvecs)?;
    if let Some(rows) = args.rows {
        file = file.take(rows);
    }
    let (dim, rows) = (file.dim(), file.rows());
    let schema = dataset_schema(&args.id_column, &args.column, dim);

    let batches = Batches {
        file,
        schema: schema.clone(),
        batch_rows: args.batch_rows,
        written: 0,
    };
    let dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        args.dataset.as_str(),
        None,
    )
    .await?;

    let written = dataset.count_rows(None).await?;
    if written != rows {
        return Err(Error::io(format!(
            "{} holds {written} rows, but {rows} were read from {}",
            args.dataset,
            args.fvecs.display()
        )));
    }
    println!(
        "wrote {written} rows of {dim} dimensions to {}",
        args.dataset
    );
    Ok(())
}

async fn build(args: BuildArgs) -> Result<()> {
    if args.rows_per_partition == Some(0) {
        return Err(Error::invalid_input(
            "--rows-per-partition must be at least one".to_string(),
        ));
    }
    let mut dataset = Dataset::open(&args.target.dataset).await?;
    let partitions = match (args.partitions, args.rows_per_partition) {
        (Some(partitions), None) => partitions,
        (None, Some(rows_per_partition)) => {
            let rows = dataset.count_rows(None).await?;
            u32::try_from(rows.div_ceil(rows_per_partition).max(1)).map_err(|_| {
                Error::invalid_input(format!(
                    "{rows} rows at {rows_per_partition} a partition needs more partitions than \
                     an index can have"
                ))
            })?
        }
        (None, None) | (Some(_), Some(_)) => {
            return Err(Error::invalid_input(
                "exactly one of --partitions and --rows-per-partition is required".to_string(),
            ));
        }
    };

    let mut params = IndexParams::new(&args.column, partitions)
        .with_distance_type(args.metric.into())
        .with_kmeans_max_iters(args.kmeans_iters)
        .with_kmeans_sample_rate(args.kmeans_sample_rate)
        .with_graph_params(BuildParams {
            max_degree: args.max_degree,
            search_list_size: args.search_list_size,
            alpha: args.alpha,
            seed: args.seed,
        });
    if let Some(code_bits) = args.code_bits {
        params = params.with_code_bits(code_bits);
    }

    let started = Instant::now();
    let stats = create_index(&mut dataset, &args.target.index_name, &params).await?;
    println!(
        "indexed {} vectors into {} partitions in {:.1}s, {} distance computations",
        stats.vectors,
        stats.partitions,
        started.elapsed().as_secs_f64(),
        stats.comparisons
    );
    Ok(())
}

async fn search(args: SearchArgs) -> Result<()> {
    let dataset = Dataset::open(&args.target.dataset).await?;
    let index = VamanaIndex::open(&dataset, &args.target.index_name).await?;
    // Zero is no cache at all, not a cache of no capacity: one of those still
    // serves what it holds before it reclaims.
    let index = match args.cache_mb {
        0 => index,
        megabytes => {
            let bytes = megabytes.checked_mul(1 << 20).ok_or_else(|| {
                Error::invalid_input(format!("--cache-mb {megabytes} is not a byte count"))
            })?;
            index.with_cache(LanceCache::with_capacity(bytes))
        }
    };

    let queries = load_queries(&args)?;
    if queries.is_empty() {
        return Err(Error::invalid_input(
            "there are no queries to answer".to_string(),
        ));
    }
    let dimension = index.metadata().dimension as usize;
    if let Some(query) = queries.iter().find(|query| query.len() != dimension) {
        return Err(Error::invalid_input(format!(
            "the index holds {dimension}-dimensional vectors, but a query has {}",
            query.len()
        )));
    }

    let mut params = SearchParams::new(args.k)
        .with_nprobes(args.nprobes)
        .with_mode(args.mode.into())
        .with_beam_width(args.beam_width);
    if let Some(search_list_size) = args.search_list_size {
        params = params.with_search_list_size(search_list_size);
    }
    if let Some(budget) = args.rescore_budget {
        params = params.with_rescore_budget(budget);
    }

    for query in queries.iter().take(args.warmup) {
        index.search(query, &params).await?;
    }

    let before = index.io_stats();
    let cache_before = index.cache_stats().await;
    let started = Instant::now();
    let mut results = Vec::with_capacity(queries.len());
    for query in &queries {
        results.push(index.search(query, &params).await?);
    }
    let micros = started.elapsed().as_micros() as f64;
    let after = index.io_stats();

    let addresses = answered_addresses(&results);
    let recall = match &args.truth {
        None => None,
        Some(path) => Some(recall_against(&dataset, &args, &results, &addresses, path).await?),
    };
    let taken = match &args.take {
        None => None,
        Some(columns) => Some(take_columns(&dataset, columns, &addresses).await?),
    };

    let count = queries.len() as f64;
    let report = SearchReport {
        queries: results.len(),
        settings: Settings {
            k: params.k,
            mode: format!("{:?}", params.mode).to_lowercase(),
            nprobes: params.nprobes,
            search_list_size: params.search_list_size,
            beam_width: params.beam_width,
            rescore_budget: params.rescore_budget,
            cache_mb: args.cache_mb,
            warmup: args.warmup.min(queries.len()),
        },
        recall,
        per_query: PerQuery {
            bytes: (after.bytes_read - before.bytes_read) as f64 / count,
            iops: (after.iops - before.iops) as f64 / count,
            requests: (after.requests - before.requests) as f64 / count,
            micros: micros / count,
            comparisons: results.iter().map(|r| r.comparisons).sum::<u64>() as f64 / count,
            partitions_read: results.iter().map(|r| r.partitions_read).sum::<usize>() as f64
                / count,
        },
        cache: cache_report(cache_before, index.cache_stats().await),
        answers: results
            .iter()
            .map(|result| {
                result
                    .neighbors
                    .iter()
                    .map(|neighbor| Answer {
                        row_addr: neighbor.row_addr,
                        distance: neighbor.distance,
                        row: taken
                            .as_ref()
                            .and_then(|rows| rows.get(&neighbor.row_addr).cloned()),
                    })
                    .collect()
            })
            .collect(),
    };
    report.print(args.json)
}

/// What the cache served over the measured pass, and what it holds now.
///
/// Reported because a budget smaller than the working set makes the byte
/// figure mean something else, and nothing else in the output would say so.
fn cache_report(before: Option<CacheStats>, after: Option<CacheStats>) -> Option<CacheReport> {
    let (before, after) = (before?, after?);
    let hits = after.hits - before.hits;
    let lookups = hits + (after.misses - before.misses);
    Some(CacheReport {
        hit_ratio: match lookups {
            0 => 0.0,
            lookups => hits as f64 / lookups as f64,
        },
        held_bytes: after.size_bytes,
    })
}

async fn insert(args: InsertArgs) -> Result<()> {
    let mut dataset = Dataset::open(&args.target.dataset).await?;
    let stats = match args.in_place {
        true => insert_in_place(&mut dataset, &args.target.index_name).await?,
        false => insert_as_segment(&mut dataset, &args.target.index_name).await?,
    };
    println!(
        "indexed {} vectors of {} fragments: {} partitions created, {} grown, {} copied, {} \
         distance computations",
        stats.vectors,
        stats.fragments_indexed,
        stats.partitions_created,
        stats.partitions_grown,
        stats.partitions_copied,
        stats.comparisons
    );
    Ok(())
}

async fn merge(target: TargetArgs) -> Result<()> {
    let mut dataset = Dataset::open(&target.dataset).await?;
    let stats = merge_index(&mut dataset, &target.index_name).await?;
    println!(
        "folded {} segments: {} vectors inserted, {} vertices folded, {} removed",
        stats.segments_folded,
        stats.vectors_inserted,
        stats.vertices_folded,
        stats.vertices_removed
    );
    println!(
        "partitions: {} written ({} rebuilt), {} copied, {} dropped, {} distance computations",
        stats.partitions_written,
        stats.partitions_rebuilt,
        stats.partitions_copied,
        stats.partitions_dropped,
        stats.comparisons
    );
    Ok(())
}

async fn consolidate(target: TargetArgs) -> Result<()> {
    let mut dataset = Dataset::open(&target.dataset).await?;
    let stats = consolidate_index(&mut dataset, &target.index_name).await?;
    println!(
        "segments: {} rewritten, {} untouched, {} abandoned",
        stats.segments_rewritten, stats.segments_untouched, stats.segments_abandoned
    );
    println!(
        "partitions: {} consolidated ({} rebuilt), {} copied, {} dropped, {} vertices removed, {} \
         distance computations",
        stats.partitions_consolidated,
        stats.partitions_rebuilt,
        stats.partitions_copied,
        stats.partitions_dropped,
        stats.vertices_removed,
        stats.comparisons
    );
    Ok(())
}

async fn info(args: InfoArgs) -> Result<()> {
    let dataset = Dataset::open(&args.target.dataset).await?;
    let index = VamanaIndex::open(&dataset, &args.target.index_name).await?;
    let metadata = index.metadata();

    if args.json {
        let report = serde_json::json!({
            "index": args.target.index_name,
            "segments": index.num_segments(),
            "fragments": index.covered_fragments().len(),
            "first_segment": metadata,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!("index      {}", args.target.index_name);
    println!("segments   {}", index.num_segments());
    println!("fragments  {}", index.covered_fragments().len());
    // The rotation the full metadata carries is 480 bytes at d = 960, which is
    // a wall of numbers here and belongs to `--json`.
    println!("first segment:");
    println!("  dimension      {}", metadata.dimension);
    println!("  metric         {}", metadata.distance_type);
    println!(
        "  row ids        {}",
        format!("{:?}", metadata.row_id_mode).to_lowercase()
    );
    println!("  max degree     {}", metadata.max_degree);
    println!("  build beam     {}", metadata.search_list_size);
    println!("  alpha          {}", metadata.alpha);
    match &metadata.codes {
        Some(codes) => println!("  code bits      {}", codes.num_bits),
        None => println!("  code bits      none"),
    }
    println!("  format version {}", metadata.format_version);
    Ok(())
}

fn load_queries(args: &SearchArgs) -> Result<Vec<Vec<f32>>> {
    match (&args.fvecs, &args.vector) {
        (Some(path), None) => {
            let mut file = Fvecs::open(path)?;
            if let Some(limit) = args.limit {
                file = file.take(limit);
            }
            file.rest()
        }
        (None, Some(literal)) => literal
            .split(',')
            .map(|value| {
                value.trim().parse::<f32>().map_err(|e| {
                    Error::invalid_input(format!(
                        "--vector holds {value:?}, which is not a number: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(|query| vec![query]),
        (None, None) | (Some(_), Some(_)) => Err(Error::invalid_input(
            "exactly one of --fvecs and --vector is required".to_string(),
        )),
    }
}

/// Every row address any query answered with, sorted and without repeats.
fn answered_addresses(results: &[QueryResult]) -> Vec<u64> {
    let mut addresses = results
        .iter()
        .flat_map(|result| result.neighbors.iter().map(|neighbor| neighbor.row_addr))
        .collect::<Vec<_>>();
    addresses.sort_unstable();
    addresses.dedup();
    addresses
}

/// The share of each query's true `k` nearest that came back.
async fn recall_against(
    dataset: &Dataset,
    args: &SearchArgs,
    results: &[QueryResult],
    addresses: &[u64],
    truth_path: &std::path::Path,
) -> Result<f64> {
    let truth = read_ivecs(truth_path)?;
    if truth.len() < results.len() {
        return Err(Error::invalid_input(format!(
            "{} holds {} rows of ground truth for {} queries",
            truth_path.display(),
            truth.len(),
            results.len()
        )));
    }
    if let Some(row) = truth
        .iter()
        .take(results.len())
        .find(|row| row.len() < args.k)
    {
        return Err(Error::invalid_input(format!(
            "{} names {} neighbours a query, fewer than the {} asked for",
            truth_path.display(),
            row.len(),
            args.k
        )));
    }

    let positions = positions_of(dataset, &args.id_column, addresses).await?;
    let mut found = 0usize;
    for (result, expected) in results.iter().zip(&truth) {
        let expected = &expected[..args.k];
        for neighbor in &result.neighbors {
            let position = positions.get(&neighbor.row_addr).ok_or_else(|| {
                Error::invalid_input(format!(
                    "row {} is not in the dataset's {} column",
                    neighbor.row_addr, args.id_column
                ))
            })?;
            found += usize::from(expected.contains(position));
        }
    }
    Ok(found as f64 / (results.len() * args.k) as f64)
}

/// Each answered row's position in the base file, keyed by its address.
///
/// Only the answered rows: scanning the whole `id` column would be the one part
/// of scoring recall that does not fit in memory at a billion rows.
async fn positions_of(
    dataset: &Dataset,
    id_column: &str,
    addresses: &[u64],
) -> Result<HashMap<u64, u32>> {
    let batch = dataset
        .take_rows(addresses, projection(dataset, &[id_column])?)
        .await?;
    let taken = row_ids(&batch)?;
    let positions = batch
        .column_by_name(id_column)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "recall needs {id_column} to be a UInt64 column of base-file positions"
            ))
        })?;
    if positions.null_count() > 0 {
        return Err(Error::invalid_input(format!(
            "{id_column} is null for {} of the rows answered, so their position is unknown",
            positions.null_count()
        )));
    }

    let mut by_address = HashMap::with_capacity(taken.len());
    let mut seen = HashSet::with_capacity(taken.len());
    for (row, address) in taken.values().iter().enumerate() {
        let position = positions.value(row);
        // A ground truth names positions as `u32`, so anything wider is not one,
        // and scoring it as a miss would report a broken index instead.
        let position = u32::try_from(position).map_err(|_| {
            Error::invalid_input(format!(
                "{id_column} holds {position}, which is too large to be a base-file position"
            ))
        })?;
        if !seen.insert(position) {
            return Err(Error::invalid_input(format!(
                "{id_column} holds {position} more than once, so recall cannot be scored by it"
            )));
        }
        by_address.insert(*address, position);
    }
    Ok(by_address)
}

/// The named columns of every answered row, keyed by row address.
///
/// Joined on `_rowid` and never on position: `take_rows` drops rows it cannot
/// find rather than erroring, so position would shift every later answer.
async fn take_columns(
    dataset: &Dataset,
    columns: &str,
    addresses: &[u64],
) -> Result<HashMap<u64, HashMap<String, String>>> {
    let wanted = columns
        .split(',')
        .map(str::trim)
        .filter(|column| !column.is_empty())
        .collect::<Vec<_>>();
    if wanted.is_empty() {
        return Err(Error::invalid_input("--take names no columns".to_string()));
    }
    let batch = dataset
        .take_rows(addresses, projection(dataset, &wanted)?)
        .await?;

    let taken = row_ids(&batch)?;
    let options = FormatOptions::default();
    let formatters = batch
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name() != ROW_ID)
        .map(|field| {
            let column = batch
                .column_by_name(field.name())
                .expect("a field of the batch's own schema");
            ArrayFormatter::try_new(column.as_ref(), &options)
                .map(|formatter| (field.name().clone(), formatter))
        })
        .collect::<std::result::Result<Vec<_>, ArrowError>>()?;

    Ok(taken
        .values()
        .iter()
        .enumerate()
        .map(|(row, address)| {
            let values = formatters
                .iter()
                .map(|(name, formatter)| (name.clone(), formatter.value(row).to_string()))
                .collect();
            (*address, values)
        })
        .collect())
}

/// `columns` plus the row id, checked against the schema first.
///
/// [`ProjectionRequest::from_columns`] unwraps the projection it builds, so an
/// unknown column would abort the process rather than report itself.
fn projection(dataset: &Dataset, columns: &[&str]) -> Result<ProjectionRequest> {
    for column in columns {
        if dataset.schema().field(column).is_none() {
            return Err(Error::invalid_input(format!(
                "the dataset has no column named {column}"
            )));
        }
    }
    Ok(ProjectionRequest::from_columns(
        columns.iter().copied().chain([ROW_ID]),
        dataset.schema(),
    ))
}

fn row_ids(batch: &RecordBatch) -> Result<&UInt64Array> {
    batch
        .column_by_name(ROW_ID)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| Error::invalid_input("the take returned no row ids".to_string()))
}

#[derive(Serialize)]
struct SearchReport {
    queries: usize,
    settings: Settings,
    recall: Option<f64>,
    per_query: PerQuery,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache: Option<CacheReport>,
    answers: Vec<Vec<Answer>>,
}

/// What the figures were taken at.
///
/// Carried in the report because a byte count means nothing without them: the
/// same index and the same queries cost wildly different amounts under another
/// mode, beam or cache, and two reports that differ thirtyfold would otherwise
/// be indistinguishable.
#[derive(Serialize)]
struct Settings {
    k: usize,
    mode: String,
    nprobes: usize,
    search_list_size: usize,
    beam_width: usize,
    rescore_budget: Option<usize>,
    cache_mb: usize,
    warmup: usize,
}

#[derive(Serialize)]
struct PerQuery {
    bytes: f64,
    iops: f64,
    requests: f64,
    micros: f64,
    comparisons: f64,
    partitions_read: f64,
}

#[derive(Serialize)]
struct CacheReport {
    hit_ratio: f64,
    held_bytes: usize,
}

#[derive(Serialize)]
struct Answer {
    row_addr: u64,
    distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    row: Option<HashMap<String, String>>,
}

impl SearchReport {
    fn print(&self, json: bool) -> Result<()> {
        if json {
            println!("{}", serde_json::to_string_pretty(self)?);
            return Ok(());
        }
        let settings = &self.settings;
        print!(
            "{} probes, mode {}, L = {}, W = {}",
            settings.nprobes, settings.mode, settings.search_list_size, settings.beam_width
        );
        if let Some(budget) = settings.rescore_budget {
            print!(", budget {budget}");
        }
        if settings.cache_mb > 0 {
            print!(", cache {} MB", settings.cache_mb);
        }
        if settings.warmup > 0 {
            print!(", {} warmed", settings.warmup);
        }
        println!();

        let queries = match self.queries {
            1 => "1 query".to_string(),
            many => format!("{many} queries"),
        };
        match self.recall {
            Some(recall) => println!("recall {recall:.4} over {queries} at k = {}", settings.k),
            None => println!("{queries} at k = {}", settings.k),
        }
        let cost = &self.per_query;
        println!(
            "per query: {:.0} bytes, {:.0} iops, {:.1} requests, {:.0} us, {:.0} distances, {:.1} \
             partitions",
            cost.bytes,
            cost.iops,
            cost.requests,
            cost.micros,
            cost.comparisons,
            cost.partitions_read
        );
        if let Some(cache) = &self.cache {
            println!(
                "cache: {:.0}% of lookups served, holding {:.1} MB",
                cache.hit_ratio * 100.0,
                cache.held_bytes as f64 / (1 << 20) as f64
            );
        }

        match self.answers.as_slice() {
            [answers] => {
                for (rank, answer) in answers.iter().enumerate() {
                    print!(
                        "{rank:>4} {:>20} {:>12.4}",
                        answer.row_addr, answer.distance
                    );
                    if let Some(row) = &answer.row {
                        let mut columns = row.iter().collect::<Vec<_>>();
                        columns.sort_unstable();
                        for (name, value) in columns {
                            print!("  {name}={value}");
                        }
                    }
                    println!();
                }
            }
            _ => println!("answers to every query are in --json"),
        }
        Ok(())
    }
}

/// The item field is nullable because [`FixedSizeListArray::try_new_from_values`]
/// builds it that way, and a disagreeing schema is refused on the first batch.
fn dataset_schema(id_column: &str, vector_column: &str, dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(id_column, DataType::UInt64, false),
        Field::new(
            vector_column,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

/// The file, handed to the writer a batch at a time rather than collected:
/// GIST1M is 3.8 GB of `.fvecs`.
struct Batches {
    file: Fvecs,
    schema: SchemaRef,
    batch_rows: usize,
    written: u64,
}

impl Iterator for Batches {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let dim = self.file.dim();
        let values = match self.file.next_batch(self.batch_rows) {
            Ok(None) => return None,
            Ok(Some(values)) => values,
            Err(error) => return Some(Err(ArrowError::ExternalError(Box::new(error)))),
        };
        let rows = (values.len() / dim) as u64;
        let batch = FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32)
            .and_then(|vectors| {
                RecordBatch::try_new(
                    self.schema.clone(),
                    vec![
                        Arc::new(UInt64Array::from_iter_values(
                            self.written..self.written + rows,
                        )),
                        Arc::new(vectors) as Arc<dyn Array>,
                    ],
                )
            });
        self.written += rows;
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::CommandFactory;

    #[test]
    fn the_parser_is_well_formed() {
        Cli::command().debug_assert();
    }
}
