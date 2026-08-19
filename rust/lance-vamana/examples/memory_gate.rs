// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whether reading a partition vertex by vertex could ever beat reading it whole.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --release --example memory_gate
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 500), `GRID_QUERIES` (default 200), `CENSUS_QUERIES`
//! (default 50), `ROWS_PER_PARTITION` (default `1000,8192,65536`),
//! `PROBE_PERCENTS` (default `5,10,20,40,80`), `BEAMS` (default
//! `10,20,40,80,160`), `TUNING_BEAM` (default 200), `TARGET_RECALL` (default 95),
//! `CACHE_PERCENT` (default `100,25,10,5,1`), `DEGREE` (default 64), `CODE_BYTES`
//! (default 32), `LAZY_BEAM` (default 4), `ROWS_PER_FRAGMENT` (default 10000).
//!
//! This is the gate the whole of phase D hangs on, and it is a measurement rather
//! than a design: the crate reads partitions whole and caches nothing between
//! queries, and the alternative - a beam search that reads only the vertices it
//! touches - is weeks of work that is only worth starting if the arithmetic
//! favours it. Three arms are compared at the same operating point:
//!
//! - **whole**, what the driver does today: `nprobes` partition files, read
//!   entire, per query.
//! - **lazy**, a traversal that keeps nothing resident: it must read the vector
//!   of every vertex it measures a distance against, and the edges of every
//!   vertex it expands.
//! - **coded**, DiskANN proper: quantised codes for every vector stay resident,
//!   so the only disk reads are the edges of expanded vertices plus the full
//!   vectors of the final candidates, for re-ranking.
//!
//! **The two lazy arms are measured, not modelled.** The read set of a walk is
//! recovered exactly - `greedy_search` marks a vertex once, immediately before
//! counting a distance against it, so the vertices whose vectors it needs are
//! `{medoid} ∪ ⋃ neighbours(v)` over the visited `v`, and that set's size must
//! equal the walk's own comparison count, which is asserted below. Those row
//! numbers are then handed to the reader as a single [`ReadBatchParams::Ranges`]
//! request and the bytes come off the scheduler's counters. So the columns
//! compare requests to requests.
//!
//! What is modelled is only the conversion into time, because this machine cannot
//! measure it: the files are written by this process and every read is served
//! from the page cache, and dropping it needs root. So `Q*` - the number of
//! queries one whole-partition load has to serve before it is the cheaper arm -
//! is printed over a range of device parameters, with the plan's own assumption
//! (100 us) among them.
//!
//! The other half of the gate is `Q` itself, which is a property of the workload
//! rather than of the index: routing every query is pure arithmetic, so the exact
//! sequence of partitions a stream of queries would read is replayed through an
//! LRU of a stated capacity, and `Q` is probes divided by loads. Reported for the
//! query file's own order and for a clustered order, which is the hot-working-set
//! case and therefore the one hostile to reading lazily.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_io::ReadBatchParams;
use lance_io::scheduler::ScanScheduler;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::format::{INDEX_FILE_NAME, NEIGHBORS_COLUMN, ROW_ID_COLUMN, VECTOR_COLUMN};
use lance_vamana::io::{open_file, read_partition, read_segment, scan_scheduler};
use lance_vamana::partition::Partition;
use lance_vamana::query::{SearchParams, VamanaIndex};
use lance_vamana::search::{Comparisons, SearchScratch, flat_storage, greedy_search};
use lance_vamana::segment::{PartitionEntry, SegmentManifest};
use object_store::path::Path;

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const K: usize = 10;
const VECTOR_FIELD: &str = "vector";
const ID_COLUMN: &str = "id";
const INDEX_NAME: &str = "vamana_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;
/// Round trips the time model is printed for, in microseconds. The middle one is
/// what the plan assumed for a local NVMe without measuring it.
const ROUND_TRIPS_US: [f64; 3] = [20.0, 100.0, 500.0];
/// Bytes a second the time model assumes. Only the whole-partition arm is
/// bandwidth-bound, so one value is enough to place the crossover.
const BANDWIDTH: f64 = 2e9;
/// What the driver holds at once (`query.rs::PARTITIONS_IN_FLIGHT`): the only
/// bound on a query's working set, and so what "resident" means for the arm that
/// reads partitions whole.
const PARTITIONS_IN_FLIGHT: f64 = 4.0;
/// The block a local store reads in, from `lance-io`'s `infer_block_size`.
///
/// [`ScanStats`] counts the bytes a reader *asked* for, which is the right input
/// to a model of a better layout but not what a device does: a read of 776 bytes
/// still moves a page. So every arm is reported twice - what it requested, and
/// `iops` times this, which is the floor on what storage actually moved. For the
/// arm that reads a partition whole the two are the same number; for one that
/// reads scattered vertices they are not close.
const PAGE_BYTES: f64 = 4096.0;

fn env_list(name: &str, fallback: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be a comma-separated list of numbers"))
        })
        .collect()
}

/// What one read asked storage for, and what it took.
///
/// The time is warm-cache and therefore not storage latency. It is still worth
/// having: the layout probe measured the reader decoding full-zip data at about
/// 0.7 GB/s, which is slower than the NVMe under it, so what a partition costs to
/// *decode* is a real per-byte price that reading less avoids as surely as it
/// avoids the IO. Both arms are measured the same warm way, so the comparison
/// between them is a CPU comparison and holds.
#[derive(Default, Clone, Copy)]
struct Cost {
    bytes: u64,
    iops: u64,
    micros: u128,
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, other: Self) {
        self.bytes += other.bytes;
        self.iops += other.iops;
        self.micros += other.micros;
    }
}

async fn cost<F, T>(scheduler: &Arc<ScanScheduler>, read: F) -> Cost
where
    F: std::future::Future<Output = T>,
{
    let before = scheduler.stats();
    let started = Instant::now();
    let value = read.await;
    let micros = started.elapsed().as_micros();
    let after = scheduler.stats();
    drop(value);
    Cost {
        bytes: after.bytes_read - before.bytes_read,
        iops: after.iops - before.iops,
        micros,
    }
}

/// Read exactly `rows` as one request, the way a lazy traversal would.
///
/// The scheduler coalesces adjacent ranges in a single pass and does not sort
/// them, so `rows` has to be ascending - which the callers guarantee by sorting
/// the read set they recovered.
async fn read_vertices(reader: &FileReader, rows: &[u32]) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let ranges = rows
        .iter()
        .map(|row| *row as u64..*row as u64 + 1)
        .collect::<Vec<Range<u64>>>();
    reader
        .read_stream(
            ReadBatchParams::Ranges(ranges.into()),
            u32::MAX,
            1,
            FilterExpression::no_filter(),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

/// Exact nearest `K` ids of one query, by brute force over every row.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef) -> Vec<u64> {
    let calculator = store.dist_calculator(query, 0.0);
    let mut scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id))
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(K, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(K);
    scored.into_iter().map(|(_, id)| id as u64).collect()
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

/// The `id` of every row, keyed by the address the index answers in.
async fn ids_by_address(dataset: &Dataset) -> HashMap<u64, u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project(&[ID_COLUMN]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let ids = batch[ID_COLUMN].as_primitive::<UInt64Type>();
    let addresses = batch[ROW_ID].as_primitive::<UInt64Type>();
    addresses
        .values()
        .iter()
        .zip(ids.values())
        .map(|(address, id)| (*address, *id))
        .collect()
}

/// Which partitions a query would read, in the order the driver would pick them.
///
/// The same selection as `query.rs::route`: every centroid is ranked and empty
/// partitions are skipped rather than counted, because a probe spent on one would
/// read nothing.
fn probe_plan(manifest: &SegmentManifest, query: &ArrayRef, nprobes: usize) -> Vec<PartitionEntry> {
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
        .filter_map(|id| manifest.partition(*id))
        .take(nprobes)
        .cloned()
        .collect()
}

/// Probes served per physical load, replaying `plans` through an LRU of
/// `capacity` partitions.
///
/// A property of the workload rather than of the index: it is the number the
/// whole gate turns on, because a load that serves many queries is a load whose
/// cost per query is small however large it is.
fn amortization(plans: &[Vec<u32>], capacity: usize) -> f64 {
    let mut resident: Vec<u32> = Vec::with_capacity(capacity + 1);
    let mut probes = 0usize;
    let mut loads = 0usize;
    for plan in plans {
        for partition in plan {
            probes += 1;
            if let Some(position) = resident.iter().position(|held| held == partition) {
                let held = resident.remove(position);
                resident.push(held);
                continue;
            }
            loads += 1;
            if resident.len() == capacity {
                resident.remove(0);
            }
            resident.push(*partition);
        }
    }
    probes as f64 / loads.max(1) as f64
}

/// What a walk touched, and what reading only that asked storage for.
#[derive(Default)]
struct Census {
    /// Vertices whose vector a walk needed a distance against.
    read_set: f64,
    /// Vertices whose out-edges it followed.
    visited: f64,
    /// The longest such chain among the partitions of one query, which is what
    /// its latency would be, the partitions being read against each other.
    chain: f64,
    /// Vertices in the partitions one query read.
    partition_rows: f64,
    /// Reading the probed partitions whole, measured the same way as the two lazy
    /// arms so that the three are comparable in time as well as in bytes. The
    /// driver's own per-query latency is measured separately and includes routing,
    /// the walk and the concurrency between reads; this is the read alone.
    whole: Cost,
    lazy: Cost,
    coded: Cost,
    /// What opening a partition file costs, per probe.
    ///
    /// Measured because the arms would otherwise not be comparable: the driver
    /// opens a partition file on every query, while the readers a lazy traversal
    /// would keep are opened once here. Without this line the lazy arms would be
    /// flattered by exactly the difference.
    open: Cost,
    queries: usize,
}

impl Census {
    fn mean(&self, value: f64) -> f64 {
        value / self.queries as f64
    }
}

struct Measured {
    recall: f64,
    bytes: f64,
    iops: f64,
    median_micros: u128,
    partitions: f64,
}

async fn measure(
    index: &VamanaIndex,
    queries: &[ArrayRef],
    truth: &[Vec<u64>],
    ids: &HashMap<u64, u64>,
    params: &SearchParams,
) -> Measured {
    let mut hits = 0usize;
    let mut partitions = 0usize;
    let mut latencies = Vec::with_capacity(queries.len());
    let before = index.io_stats();
    for (query, exact) in queries.iter().zip(truth) {
        let expected = exact.iter().copied().collect::<HashSet<_>>();
        let started = Instant::now();
        let result = index
            .search(query.as_primitive::<Float32Type>().values(), params)
            .await
            .unwrap();
        latencies.push(started.elapsed().as_micros());
        partitions += result.partitions_read;
        for neighbor in &result.neighbors {
            let id = ids[&neighbor.row_addr];
            hits += usize::from(expected.contains(&id));
        }
    }
    latencies.sort_unstable();
    let after = index.io_stats();
    let per_query = queries.len() as f64;
    Measured {
        recall: hits as f64 / (queries.len() * K) as f64,
        bytes: (after.bytes_read - before.bytes_read) as f64 / per_query,
        iops: (after.iops - before.iops) as f64 / per_query,
        median_micros: latencies[latencies.len() / 2],
        partitions: partitions as f64 / per_query,
    }
}

/// Measure the first `count` queries, after an uncounted pass over a prefix.
///
/// The warm-up is not politeness: the driver caches nothing between queries, so a
/// first pass over a freshly written index measures the page cache filling rather
/// than the index.
async fn graded(
    index: &VamanaIndex,
    queries: &[ArrayRef],
    truth: &[Vec<u64>],
    ids: &HashMap<u64, u64>,
    params: &SearchParams,
    count: usize,
) -> Measured {
    let warm = count.min(50);
    measure(index, &queries[..warm], &truth[..warm], ids, params).await;
    measure(index, &queries[..count], &truth[..count], ids, params).await
}

/// One partition, held for as long as the census keeps probing it.
///
/// The graph and the readers both, because the census walks the same partition
/// once per query that routes to it and re-reading it each time would cost more
/// than everything being measured.
struct Held {
    partition: Partition,
    storage: FlatFloatStorage,
    /// Every column, which is what the driver reads and therefore what the arm it
    /// stands for costs.
    all: FileReader,
    /// The vector of every vertex a distance is measured against, which is what a
    /// traversal with nothing resident has to fetch and one with codes resident
    /// does not.
    vectors: FileReader,
    /// The out-edges of every vertex a walk expands, which both need.
    edges: FileReader,
    /// Vector and row id of the candidates that come back, for re-ranking and for
    /// naming the rows. Both arms pay it.
    answer: FileReader,
}

#[allow(clippy::too_many_arguments)]
async fn census(
    scheduler: &Arc<ScanScheduler>,
    manifest: &SegmentManifest,
    dir: &Path,
    file_sizes: &HashMap<String, u64>,
    queries: &[ArrayRef],
    nprobes: usize,
    beam: usize,
    distance_type: DistanceType,
) -> Census {
    let mut held: HashMap<String, Held> = HashMap::new();
    let mut census = Census {
        queries: queries.len(),
        ..Default::default()
    };
    for query in queries {
        let mut chain = 0usize;
        for entry in probe_plan(manifest, query, nprobes) {
            if !held.contains_key(&entry.file) {
                let path = dir.clone().join(entry.file.as_str());
                let size = file_sizes.get(&entry.file).copied();
                let all = open_file(scheduler, &path, None, size).await.unwrap();
                let partition = read_partition(&all, entry.num_rows).await.unwrap();
                let storage = flat_storage(
                    partition.graph().row_ids(),
                    partition.vectors(),
                    distance_type,
                )
                .unwrap();
                held.insert(
                    entry.file.clone(),
                    Held {
                        partition,
                        storage,
                        all,
                        vectors: open_file(
                            scheduler,
                            &path,
                            Some([VECTOR_COLUMN].as_slice()),
                            size,
                        )
                        .await
                        .unwrap(),
                        edges: open_file(
                            scheduler,
                            &path,
                            Some([NEIGHBORS_COLUMN].as_slice()),
                            size,
                        )
                        .await
                        .unwrap(),
                        answer: open_file(
                            scheduler,
                            &path,
                            Some([ROW_ID_COLUMN, VECTOR_COLUMN].as_slice()),
                            size,
                        )
                        .await
                        .unwrap(),
                    },
                );
            }
            let state = &held[&entry.file];
            let partition = &state.partition;
            let calculator = state.storage.dist_calculator(query.clone(), 0.0);
            let mut scratch = SearchScratch::new(partition.len());
            let comparisons = Comparisons::default();
            let walk = greedy_search(
                partition.graph(),
                &calculator,
                entry.medoid,
                beam,
                &mut scratch,
                &comparisons,
            )
            .unwrap();

            // The set a lazy traversal would have to read, recovered rather than
            // instrumented: a vertex is marked exactly once, immediately before a
            // distance is counted against it.
            let mut read_set = vec![entry.medoid];
            for node in &walk.visited {
                read_set.extend_from_slice(partition.graph().neighbors(node.id).unwrap());
            }
            read_set.sort_unstable();
            read_set.dedup();
            assert_eq!(
                read_set.len() as u64,
                comparisons.get(),
                "the recovered read set and the walk's own counter disagree, so one of them does \
                 not mean what this harness assumes"
            );
            let mut visited = walk.visited.iter().map(|node| node.id).collect::<Vec<_>>();
            visited.sort_unstable();
            visited.dedup();
            let mut candidates = walk
                .candidates
                .iter()
                .take(K)
                .map(|node| node.id)
                .collect::<Vec<_>>();
            candidates.sort_unstable();
            candidates.dedup();

            census.read_set += read_set.len() as f64;
            census.visited += visited.len() as f64;
            census.partition_rows += partition.len() as f64;
            chain = chain.max(visited.len());
            // What the driver does, measured here so that all three arms are
            // measured by one instrument. The partition itself is already in hand;
            // this read is the cost of having it.
            census.whole += cost(scheduler, read_partition(&state.all, entry.num_rows)).await;

            // Three reads, shared between two arms rather than measured twice:
            // both have to follow the edges of what they expand and re-rank what
            // comes back, and only the one without codes has to fetch a vector to
            // measure a distance at all.
            let edges = cost(scheduler, read_vertices(&state.edges, &visited)).await;
            let answer = cost(scheduler, read_vertices(&state.answer, &candidates)).await;
            let mut coded = edges;
            coded += answer;
            let mut lazy = coded;
            lazy += cost(scheduler, read_vertices(&state.vectors, &read_set)).await;
            census.coded += coded;
            census.lazy += lazy;
            let path = dir.clone().join(entry.file.as_str());
            let size = file_sizes.get(&entry.file).copied();
            census.open += cost(scheduler, async {
                open_file(scheduler, &path, None, size).await.unwrap()
            })
            .await;
        }
        census.chain += chain as f64;
    }
    census
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
    let num_queries = env_usize("QUERIES", 500).min(total_queries);
    // The beam grid is only there to pick an operating point, and 2000 samples
    // place recall to within half a point - enough to choose a beam, and a
    // fraction of the reads that measuring every beam over the full stream would
    // cost at the coarse end of the sweep, where one probe is tens of megabytes.
    let grid_queries = env_usize("GRID_QUERIES", 200).min(num_queries);
    let census_queries = env_usize("CENSUS_QUERIES", 50).min(num_queries);
    let sweep = env_list("ROWS_PER_PARTITION", "1000,8192,65536");
    let probe_percents = env_list("PROBE_PERCENTS", "5,10,20,40,80");
    let beams = env_list("BEAMS", "10,20,40,80,160");
    // Wide enough that the first stage measures what routing can reach rather
    // than what the walk can: recall has two ceilings, and only one of them moves
    // with the beam.
    let tuning_beam = env_usize("TUNING_BEAM", 200);
    let target_recall = env_usize("TARGET_RECALL", 95) as f64 / 100.0;
    let cache_percent = env_list("CACHE_PERCENT", "100,25,10,5,1");
    let degree = env_usize("DEGREE", 64) as u32;
    let code_bytes = env_usize("CODE_BYTES", 32);
    let lazy_beam = env_usize("LAZY_BEAM", 4).max(1);
    let rows_per_fragment = env_usize("ROWS_PER_FRAGMENT", 10_000);

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
        "SIFT {rows} x {dim}, {num_queries} queries ({census_queries} in the census), k = {K}, \
         R = {degree}, target recall {target_recall:.2}"
    );
    let started = Instant::now();
    let store = FlatFloatStorage::new(vectors.clone(), DISTANCE_TYPE);
    let truth = queries
        .iter()
        .map(|query| exact_top(&store, query.clone()))
        .collect::<Vec<_>>();
    println!(
        "exact top-{K} by brute force in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    for rows_per_partition in sweep {
        let partitions = rows.div_ceil(rows_per_partition).max(1) as u32;
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
        let build = started.elapsed().as_secs_f64();
        let ids = ids_by_address(&dataset).await;
        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        assert_eq!(index.num_segments(), 1);

        println!(
            "\n=== {rows_per_partition} rows a partition: {partitions} partitions, built in \
             {build:.1}s ==="
        );

        // Recall has two ceilings and they are not interchangeable. `nprobes`
        // decides whether the partition holding the answer is opened at all, and
        // no beam recovers a partition that was never read; the beam decides
        // whether a walk inside an opened partition finds what is there. So the
        // operating point is found in that order, cheapest first, and both stages
        // stop at the first setting that reaches the target - which is the point
        // the arms are then compared at.
        let mut probing = None;
        let mut tried = HashSet::new();
        println!(
            "  {:>7} {:>8} {:>10} {:>14}",
            "probe %", "nprobes", "recall@10", "bytes/query"
        );
        for percent in &probe_percents {
            let nprobes = ((percent * partitions as usize).div_ceil(100)).max(1);
            if !tried.insert(nprobes) {
                continue;
            }
            let params = SearchParams::new(K)
                .with_nprobes(nprobes)
                .with_search_list_size(tuning_beam);
            let measured = graded(&index, &queries, &truth, &ids, &params, grid_queries).await;
            println!(
                "  {percent:>7} {nprobes:>8} {:>10.4} {:>14.0}",
                measured.recall, measured.bytes
            );
            if measured.recall >= target_recall {
                probing = Some(nprobes);
                break;
            }
        }
        let Some(nprobes) = probing else {
            println!(
                "  probing up to {}% of {partitions} partitions never reaches recall \
                 {target_recall:.2} at beam {tuning_beam}: this granularity cannot be gated \
                 against the others",
                probe_percents.last().copied().unwrap_or_default()
            );
            continue;
        };

        let mut chosen = None;
        println!(
            "  {:>7} {:>8} {:>10} {:>14} {:>10} {:>10}",
            "beam", "nprobes", "recall@10", "bytes/query", "iops/qry", "p50 (us)"
        );
        for beam in &beams {
            let params = SearchParams::new(K)
                .with_nprobes(nprobes)
                .with_search_list_size(*beam);
            let measured = graded(&index, &queries, &truth, &ids, &params, grid_queries).await;
            println!(
                "  {beam:>7} {nprobes:>8} {:>10.4} {:>14.0} {:>10.1} {:>10}",
                measured.recall, measured.bytes, measured.iops, measured.median_micros
            );
            if measured.recall >= target_recall {
                chosen = Some(*beam);
                break;
            }
        }
        let beam = chosen.unwrap_or(tuning_beam);

        // The chosen beam over the whole stream, which is the arm the lazy ones
        // are compared against.
        let params = SearchParams::new(K)
            .with_nprobes(nprobes)
            .with_search_list_size(beam);
        let whole = measure(&index, &queries, &truth, &ids, &params).await;

        // Everything below is at that one operating point.
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

        let taken = census(
            &scheduler,
            &manifest,
            &segment_dir,
            &file_sizes,
            &queries[..census_queries],
            nprobes,
            beam,
            DISTANCE_TYPE,
        )
        .await;

        let read_set = taken.mean(taken.read_set);
        let visited = taken.mean(taken.visited);
        let partition_rows = taken.mean(taken.partition_rows);
        let lazy_bytes = taken.mean(taken.lazy.bytes as f64);
        let lazy_iops = taken.mean(taken.lazy.iops as f64);
        let coded_bytes = taken.mean(taken.coded.bytes as f64);
        let coded_iops = taken.mean(taken.coded.iops as f64);
        let chain = taken.mean(taken.chain);
        let partition_bytes = whole.bytes / whole.partitions;

        // What every query would read, routed but not read: the plan a query
        // follows is pure arithmetic, so both the byte check below and the
        // amortisation table further down come off the same routing pass over the
        // full stream rather than off the census sample.
        let plans = queries
            .iter()
            .map(|query| probe_plan(&manifest, query, nprobes))
            .collect::<Vec<_>>();
        let probed_files = plans
            .iter()
            .flatten()
            .map(|entry| file_sizes.get(&entry.file).copied().unwrap_or_default())
            .sum::<u64>() as f64
            / num_queries as f64;

        println!(
            "\n  at beam {beam}, recall {:.4}: a query reads {:.1} partitions of {:.0} vertices, \
             {:.1}% of the dataset",
            whole.recall,
            whole.partitions,
            partition_rows / whole.partitions,
            100.0 * partition_rows / rows as f64,
        );
        let census_whole_bytes = taken.mean(taken.whole.bytes as f64);
        let census_whole_iops = taken.mean(taken.whole.iops as f64);
        let pages = |bytes: f64, iops: f64| bytes.max(iops * PAGE_BYTES);
        let whole_pages = pages(census_whole_bytes, census_whole_iops);
        let lazy_pages = pages(lazy_bytes, lazy_iops);
        let coded_pages = pages(coded_bytes, coded_iops);
        println!(
            "  {:<8} {:>16} {:>8} {:>14} {:>10} {:>9}",
            "arm", "requested bytes", "iops", "pages moved", "read us", "touched"
        );
        for (label, bytes, iops, moved, micros, touched) in [
            (
                "whole",
                census_whole_bytes,
                census_whole_iops,
                whole_pages,
                taken.mean(taken.whole.micros as f64),
                100.0,
            ),
            (
                "lazy",
                lazy_bytes,
                lazy_iops,
                lazy_pages,
                taken.mean(taken.lazy.micros as f64),
                100.0 * read_set / partition_rows,
            ),
            (
                "coded",
                coded_bytes,
                coded_iops,
                coded_pages,
                taken.mean(taken.coded.micros as f64),
                100.0 * visited / partition_rows,
            ),
        ] {
            println!(
                "  {label:<8} {bytes:>16.0} {iops:>8.1} {moved:>14.0} {micros:>10.0} \
                 {touched:>8.1}%"
            );
        }
        println!(
            "  against pages moved whole: lazy {:.2}x, coded {:.2}x; a walk expands {:.0} vertices \
             and would stall {:.0} times at beam {lazy_beam}",
            lazy_pages / whole_pages,
            coded_pages / whole_pages,
            visited,
            (chain / lazy_beam as f64).ceil(),
        );
        println!(
            "  the driver's own p50 is {} us a query, of which opening files is {:.0} bytes and \
             {:.1} iops",
            whole.median_micros,
            taken.mean(taken.open.bytes as f64),
            taken.mean(taken.open.iops as f64),
        );
        println!(
            "  the probed files take {probed_files:.0} bytes on disk, so the driver reads {:.3}x \
             what they hold",
            whole.bytes / probed_files,
        );
        println!(
            "  resident: whole arm {:.1} MiB ({PARTITIONS_IN_FLIGHT:.0} partitions), coded arm \
             {:.1} MiB of codes at {code_bytes} bytes a vector plus a beam",
            PARTITIONS_IN_FLIGHT * partition_bytes / (1024.0 * 1024.0),
            (rows * code_bytes) as f64 / (1024.0 * 1024.0),
        );

        // Q*: how many queries one whole-partition load must serve before it is
        // the cheaper arm. Above it, reading lazily loses whatever the bytes say.
        //
        // Over pages rather than requested bytes, because this is a model of a
        // device, and with the chain of dependent reads a lazy walk cannot avoid:
        // the next vertex to read is only known once the last one has arrived, so
        // its round trips add up where the whole-partition arm's overlap.
        println!(
            "\n  {:>10} {:>12} {:>12}",
            "round trip", "Q* lazy", "Q* coded"
        );
        let stalls = (chain / lazy_beam as f64).ceil();
        for round_trip in ROUND_TRIPS_US {
            let rtt = round_trip * 1e-6;
            let load = whole_pages / BANDWIDTH + census_whole_iops * rtt;
            let lazy = lazy_pages / BANDWIDTH + stalls * rtt;
            let coded = coded_pages / BANDWIDTH + stalls * rtt;
            println!(
                "  {round_trip:>9.0}us {:>12.2} {:>12.2}",
                load / lazy,
                load / coded
            );
        }

        // And Q as the workload actually produces it.
        let plans = plans
            .into_iter()
            .map(|plan| {
                plan.into_iter()
                    .map(|entry| entry.partition_id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut clustered = plans.clone();
        clustered.sort_by_key(|plan| plan.first().copied().unwrap_or(0));
        println!(
            "\n  Q over {num_queries} queries, {:>13} {:>13}",
            "as queried", "clustered"
        );
        for percent in &cache_percent {
            let capacity = ((percent * partitions as usize).div_ceil(100)).max(1);
            println!(
                "  cache {percent:>3}% ({capacity:>5} partitions) {:>13.1} {:>13.1}",
                amortization(&plans, capacity),
                amortization(&clustered, capacity),
            );
        }
    }
}
