// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How many distances a walk saves at equal recall by starting somewhere other than the medoid.
//!
//! The kill test for entry points. The stored graphs are walked as production walks them,
//! stop margin included, from the medoid (production), from the query's true nearest
//! neighbour (the ceiling of any single start), and from `K` entry vertices picked by
//! k-means or at random - either the one nearest the query, which costs `K` coded
//! distances to choose, or all `K` seeded at once. Each arm's margin is swept until its
//! recall crosses the bar on both halves of the queries (even and odd), and every query's
//! counters are written out for `ep.py` to interpolate, cross-fit and bootstrap. Counts
//! only: nothing is timed.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift DATASET_DIR=~/vamana-runs EP_DIR=<dir> TARGET=99 \
//!     cargo run --profile release-no-lto --example entry_points_walk
//! ```
//!
//! `SIFT_DIR` holds `<prefix>_{base,query}.fvecs`; `DATASET_DIR` the index
//! (`<prefix>-<rows>-p1-r70-sq8.lance`); `EP_DIR` the round logs
//! `<prefix>-wnk{10,100}c1repperf-index-r1.log` and the cache `a3-gt-<prefix>-q1000.bin`,
//! and receives `ep-gt-<prefix>-q<n>.bin`, `ep-<prefix>[-p2].json` and
//! `ep-<prefix>[-p2]-q.bin`. `TARGET` is the recall bar in per cent; `QUERIES` (default
//! 1000) and `THREADS` (default 10). `EP_PHASE2=<kmeans|random>,<nearest|all>,<K>` walks
//! only the extra seeds of that one configuration. `EP_GATE=1` runs the gates alone and
//! adds V3: the crate's own entry points (`VamanaIndex::train_entry_points`, K = 64, seed
//! 42) are this harness's, and its walk from the nearest of them equals the kmeans-nearest-64
//! arm bit for bit, at margins 0, the round logs' pair, the pair in `EP_DIR/et-points.txt`
//! if there is one, and the ceiling. Nothing is written but the log.
//!
//! No arm is read before the gates hold: the medoid arm, through the same seeding path as
//! every other arm, equals `VamanaIndex::search` bit for bit (V1) and the round logs to
//! the digit (V2).

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float32Type, UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array, UInt64Array,
};
use lance::Dataset;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::vector::SQ_CODE_COLUMN;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::kmeans::{KMeans, KMeansParams};
use lance_index::vector::quantizer::QuantizerBuildParams;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::sq::storage::ScalarQuantizationStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_vamana::codes::{CODE_COLUMN, CodeParams};
use lance_vamana::entry_points::EntryPointParams;
use lance_vamana::format::{
    INDEX_FILE_NAME, NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, VECTOR_COLUMN,
};
use lance_vamana::io::{PartitionFile, read_partition_batch, read_segment, scan_scheduler};
use lance_vamana::query::{
    Neighbor, SearchParams, VamanaIndex, WalkMode, WalkStart, committed_segments,
};
use lance_vamana::search::{Comparisons, SearchList};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const BEAM: usize = 4;
const PREFETCH: usize = 2;
const DEGREE: u32 = 70;
const LOGGED_QUERIES: usize = 200;
const INDEX_NAME: &str = "vamana_idx";
const ID_COLUMN: &str = "id";
const ENTRY_COUNTS: [usize; 4] = [16, 64, 256, 1024];
/// The crate's own k-means recipe (`src/builder.rs:857-932`) and the graph's default seed.
const KMEANS_SAMPLE_RATE: usize = 256;
const KMEANS_MAX_ITERS: u32 = 50;
const KMEANS_SEED: u64 = 42;
/// The entry points V3 holds the crate's training and walk to: the configuration the kill
/// test chose.
const GATE_ENTRIES: usize = 64;
const KMEANS_EXTRA_SEEDS: [u64; 2] = [43, 44];
/// One random draw the nested sets are cut from, and the independent draws of phase 2.
const RANDOM_SEED: u64 = 4242;
const RANDOM_EXTRA_SEEDS: [u64; 4] = [4243, 4244, 4245, 4246];
const RANDOM_POOL: usize = 1024;
/// Every this many queries a walk records its offers for the prefix check.
const REPLAY_EVERY: usize = 50;
/// Margins are integers of this many units per 1.0, so every grid point is an exact
/// decimal string and parses to the `f32` the stand would have run at.
const MARGIN_UNITS: u32 = 100_000;
const COARSE_BATCH: usize = 4;
const RECORD_BYTES: usize = 23;

/// One working point: `k`, its re-score budget and list cap as in round WN, and the
/// margin grid swept to find each arm's crossing.
#[derive(Clone, Copy)]
struct Setting {
    k: usize,
    budget: usize,
    cap: usize,
    coarse: u32,
    fine: u32,
    ceiling: u32,
    /// Step of the ranks below `k` walked at margin zero, when zero already clears the bar.
    rank_step: usize,
    log_tag: &'static str,
}

const SETTINGS: [Setting; 2] = [
    Setting {
        k: 10,
        budget: 20,
        cap: 2048,
        coarse: 1000,
        fine: 50,
        ceiling: 30_000,
        rank_step: 1,
        log_tag: "wnk10c1repperf",
    },
    Setting {
        k: 100,
        budget: 200,
        cap: 4096,
        coarse: 500,
        fine: 25,
        ceiling: 10_000,
        rank_step: 10,
        log_tag: "wnk100c1repperf",
    },
];

/// A point of the effort axis, in walking order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Point {
    /// Margin zero with the stop rule's rank below `k`: the axis continued under zero.
    Rank(usize),
    /// A stop margin, in `MARGIN_UNITS` per 1.0.
    Margin(u32),
}

impl Point {
    fn label(self) -> String {
        match self {
            Self::Rank(rank) => format!("r{rank}"),
            Self::Margin(units) => {
                format!("{}.{:05}", units / MARGIN_UNITS, units % MARGIN_UNITS)
            }
        }
    }

    fn margin(self) -> f32 {
        match self {
            Self::Rank(_) => 0.0,
            Self::Margin(_) => self.label().parse().unwrap(),
        }
    }

    fn stop(self, setting: &Setting) -> StopRule {
        StopRule {
            margin: self.margin(),
            rank: match self {
                Self::Rank(rank) => rank,
                Self::Margin(_) => setting.k,
            },
            keep: setting.budget,
        }
    }
}

fn env_string(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

/// `(dimension, rows)` of an `.fvecs` file, from its first header and its length.
fn fvecs_shape(path: &str) -> (usize, usize) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut header = [0u8; 4];
    file.read_exact(&mut header).unwrap();
    let dimension = i32::from_le_bytes(header) as usize;
    let bytes = file.metadata().unwrap().len() as usize;
    let record = 4 + 4 * dimension;
    assert_eq!(
        bytes % record,
        0,
        "{path} is not whole {dimension}-wide rows"
    );
    (dimension, bytes / record)
}

/// The values of a fixed size list, row-major, whatever its offset.
fn flat_values<T: ArrowPrimitiveType>(list: &FixedSizeListArray) -> &[T::Native] {
    let width = list.value_length() as usize;
    &list.values().as_primitive::<T>().values()[list.offset() * width..][..list.len() * width]
}

fn fnv(words: impl Iterator<Item = u64>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325, |hash, word| {
        (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// One segment's partition as the production walk sees it, in local-id order.
struct Built {
    width: usize,
    medoid: u32,
    bounds: Range<f64>,
    row_ids: Vec<u64>,
    edges: Vec<u32>,
    codes: FixedSizeListArray,
    vectors: FixedSizeListArray,
}

async fn load(dataset: &Dataset, dimension: usize) -> Built {
    let committed = committed_segments(dataset, INDEX_NAME).await.unwrap();
    let [segment] = committed.as_slice() else {
        panic!(
            "expected one committed segment of {INDEX_NAME}, found {}",
            committed.len()
        );
    };
    let dir = dataset.indices_dir().join(segment.uuid.to_string());
    let sizes = segment
        .files
        .iter()
        .flatten()
        .map(|file| (file.path.clone(), file.size_bytes))
        .collect::<HashMap<_, _>>();
    let scheduler = scan_scheduler(&dataset.object_store(None).await.unwrap());
    let manifest = read_segment(&scheduler, &dir, sizes.get(INDEX_FILE_NAME).copied())
        .await
        .unwrap();
    let metadata = manifest.metadata();
    assert_eq!(metadata.distance_type, DistanceType::L2);
    assert_eq!(metadata.max_degree, DEGREE);
    assert_eq!(metadata.dimension as usize, dimension);
    let Some(CodeParams::Scalar {
        num_bits: 8,
        bounds,
    }) = metadata.codes.clone()
    else {
        panic!("expected 8-bit scalar codes, found {:?}", metadata.codes);
    };
    let [entry] = manifest.partitions() else {
        panic!(
            "expected one partition, found {}",
            manifest.partitions().len()
        );
    };
    let file = PartitionFile::open(
        &scheduler,
        &dir.clone().join(entry.file.as_str()),
        sizes.get(&entry.file).copied(),
        None,
    )
    .await
    .unwrap();
    let reader = file
        .project(&[ROW_ID_COLUMN, NEIGHBORS_COLUMN, CODE_COLUMN])
        .await
        .unwrap();
    let batch = read_partition_batch(&reader, entry.num_rows).await.unwrap();
    let row_ids = batch[ROW_ID_COLUMN]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    let neighbors = batch[NEIGHBORS_COLUMN].as_fixed_size_list();
    let width = neighbors.value_length() as usize;
    let edges = flat_values::<UInt32Type>(neighbors).to_vec();
    let codes = batch[CODE_COLUMN].as_fixed_size_list().clone();
    drop(batch);
    let reader = file.project(&[VECTOR_COLUMN]).await.unwrap();
    let batch = read_partition_batch(&reader, entry.num_rows).await.unwrap();
    let vectors = batch[VECTOR_COLUMN].as_fixed_size_list().clone();
    Built {
        width,
        medoid: entry.medoid,
        bounds,
        row_ids,
        edges,
        codes,
        vectors,
    }
}

/// `checked_neighbors`' rules over every row at once: padding only at the end, ids
/// inside the partition, no vertex its own neighbour.
fn check_edges(edges: &[u32], width: usize, rows: usize) {
    for (vertex, slots) in edges.chunks_exact(width).enumerate() {
        let degree = degree_of(slots);
        assert!(
            slots[degree..].iter().all(|slot| *slot == NO_NEIGHBOR),
            "vertex {vertex} holds a neighbour after its padding"
        );
        for neighbor in &slots[..degree] {
            assert!((*neighbor as usize) < rows && *neighbor as usize != vertex);
        }
    }
}

fn degree_of(slots: &[u32]) -> usize {
    slots
        .iter()
        .position(|slot| *slot == NO_NEIGHBOR)
        .unwrap_or(slots.len())
}

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

/// Row `p` of the base file is the stored vector of local id `p`, bit for bit.
fn check_against_fvecs(path: &str, stored: &[f32], dimension: usize) {
    let mut reader = BufReader::with_capacity(1 << 24, File::open(path).unwrap());
    let mut record = vec![0u8; 4 + 4 * dimension];
    for (position, row) in stored.chunks_exact(dimension).enumerate() {
        reader.read_exact(&mut record).unwrap();
        assert_eq!(
            i32::from_le_bytes(record[..4].try_into().unwrap()) as usize,
            dimension
        );
        for (value, bytes) in row.iter().zip(record[4..].as_chunks::<4>().0) {
            assert_eq!(
                value.to_bits(),
                u32::from_le_bytes(*bytes),
                "base row {position} differs from the stored vector of local id {position}"
            );
        }
    }
}

/// The rows `sq_sample` takes (`src/codes.rs:287-304`): a strided sample in position
/// order, which is local-id order here.
fn sample_rows(rows: usize) -> Vec<u32> {
    let wanted = SQBuildParams {
        num_bits: 8,
        ..Default::default()
    }
    .sample_size();
    if rows <= wanted {
        return (0..rows as u32).collect();
    }
    let stride = rows / wanted;
    (0..wanted).map(|row| (row * stride) as u32).collect()
}

fn take_rows(list: &FixedSizeListArray, rows: &[u32]) -> FixedSizeListArray {
    arrow_select::take::take(list, &UInt32Array::from(rows.to_vec()), None)
        .unwrap()
        .as_fixed_size_list()
        .clone()
}

fn scalar_store(row_ids: &[u64], codes: ArrayRef, bounds: Range<f64>) -> ScalarQuantizationStorage {
    let batch = RecordBatch::try_from_iter_with_nullable(vec![
        (
            ROW_ID,
            Arc::new(UInt64Array::from(row_ids.to_vec())) as ArrayRef,
            false,
        ),
        (SQ_CODE_COLUMN, codes, false),
    ])
    .unwrap();
    ScalarQuantizationStorage::try_new(8, DistanceType::L2, bounds, [batch], None).unwrap()
}

/// The exact truth of one query: its nearest 10 and 100 by the stand's own rule
/// (`select_nth_unstable` over the ids in order, each depth from its own copy, so a tie at
/// the boundary falls exactly as in the stand), and the nearest vertex, lowest id among
/// equals, with its distance.
struct Truth {
    top10: Vec<u32>,
    top100: Vec<u32>,
    nearest: u32,
    nearest_distance: f32,
}

fn exact_truth(store: &FlatFloatStorage, query: ArrayRef) -> Truth {
    let calculator = store.dist_calculator(query, 0.0);
    let scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id))
        .collect::<Vec<_>>();
    let mut nearest = (f32::INFINITY, u32::MAX);
    for (distance, id) in &scored {
        if *distance < nearest.0 {
            nearest = (*distance, *id);
        }
    }
    let top = |depth: usize| {
        let mut copy = scored.clone();
        copy.select_nth_unstable_by(depth, |left, right| left.0.total_cmp(&right.0));
        // From the slice, not `copy.into_iter()`: collecting in place would keep the
        // whole row's buffer, 8 MB a query, behind a vector of `depth` ids.
        let mut ids = copy[..depth].iter().map(|(_, id)| *id).collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    };
    Truth {
        top10: top(10),
        top100: top(100),
        nearest: nearest.1,
        nearest_distance: nearest.0,
    }
}

/// `exact_truth` for every query, cached in `path` under a key of the base and the queries.
fn ground_truth(
    store: &FlatFloatStorage,
    queries: &[ArrayRef],
    key: [u64; 2],
    threads: usize,
    path: &str,
) -> Vec<Truth> {
    let header = [
        u64::from_le_bytes(*b"EPTRUTH1"),
        store.len() as u64,
        queries.len() as u64,
        key[0],
        key[1],
    ];
    let per_query = 10 + 100 + 2;
    if let Ok(bytes) = std::fs::read(path)
        && bytes.len() == header.len() * 8 + queries.len() * per_query * 4
        && bytes[..header.len() * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .eq(header)
    {
        println!("ground truth from {path}");
        return bytes[header.len() * 8..]
            .chunks_exact(per_query * 4)
            .map(|record| {
                let words = record
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word))
                    .collect::<Vec<_>>();
                Truth {
                    top10: words[..10].to_vec(),
                    top100: words[10..110].to_vec(),
                    nearest: words[110],
                    nearest_distance: f32::from_bits(words[111]),
                }
            })
            .collect();
    }
    let started = Instant::now();
    let found = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || {
                    (thread..queries.len())
                        .step_by(threads)
                        .map(|query| (query, exact_truth(store, queries[query].clone())))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut truth = (0..queries.len()).map(|_| None).collect::<Vec<_>>();
    for (query, found) in found {
        truth[query] = Some(found);
    }
    let truth = truth.into_iter().map(Option::unwrap).collect::<Vec<_>>();
    let mut sink = Vec::with_capacity(header.len() * 8 + queries.len() * per_query * 4);
    for word in header {
        sink.extend_from_slice(&word.to_le_bytes());
    }
    for found in &truth {
        for id in found.top10.iter().chain(&found.top100) {
            sink.extend_from_slice(&id.to_le_bytes());
        }
        sink.extend_from_slice(&found.nearest.to_le_bytes());
        sink.extend_from_slice(&found.nearest_distance.to_bits().to_le_bytes());
    }
    std::fs::write(path, sink).unwrap();
    println!(
        "brute force ground truth in {:.1}s, cached in {path}",
        started.elapsed().as_secs_f64()
    );
    truth
}

/// The depth-10 truth `two_level_walk` cached, as sorted sets, if it exists for these queries.
fn a3_truth(path: &str, rows: usize, queries: usize, query_hash: u64) -> Option<Vec<Vec<u32>>> {
    let bytes = std::fs::read(path).ok()?;
    let words = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| u64::from_le_bytes(*word))
        .collect::<Vec<_>>();
    let header = [
        u64::from_le_bytes(*b"A3TRUTH1"),
        rows as u64,
        queries as u64,
        10,
        query_hash,
    ];
    (words.len() == header.len() + queries * 10 && words[..header.len()] == header).then(|| {
        words[header.len()..]
            .as_chunks::<10>()
            .0
            .iter()
            .map(|ids| {
                let mut ids = ids.iter().map(|id| *id as u32).collect::<Vec<_>>();
                ids.sort_unstable();
                ids
            })
            .collect()
    })
}

/// Visited marks for one thread's walks: a generation stamp per vertex.
struct Scratch {
    seen: Vec<u32>,
    generation: u32,
    frontier: Vec<u32>,
    fresh: Vec<u32>,
}

impl Scratch {
    fn new(rows: usize) -> Self {
        Self {
            seen: vec![0; rows],
            generation: 0,
            frontier: Vec::with_capacity(BEAM),
            fresh: Vec::new(),
        }
    }

    fn is_marked(&self, id: u32) -> bool {
        self.seen[id as usize] == self.generation
    }
}

fn mark(seen: &mut [u32], generation: u32, id: u32) -> bool {
    let slot = &mut seen[id as usize];
    if *slot == generation {
        false
    } else {
        *slot = generation;
        true
    }
}

/// `StopRule` (`src/search.rs:264-269`), crate-private there.
#[derive(Clone, Copy, Debug)]
struct StopRule {
    margin: f32,
    rank: usize,
    keep: usize,
}

#[derive(Clone, Copy)]
struct Margin {
    rank: usize,
    keep: usize,
    factor: f32,
    bar: OrderedFloat,
}

#[derive(Clone)]
struct Entry {
    node: OrderedNode,
    expanded: bool,
}

/// `SearchList` under a stop rule (`src/search.rs:303-527`), copied because
/// `SearchList::with_margin` is crate-private. It also keeps the longest length it
/// reached and, when asked, every offer in order.
struct MarginList {
    list: Vec<Entry>,
    size: usize,
    cursor: usize,
    margin: Margin,
    peak: usize,
    offers: Option<Vec<(u32, f32)>>,
}

impl MarginList {
    fn new(size: usize, rows: usize, stop: StopRule, record: bool) -> Self {
        assert!(
            1 <= stop.rank && stop.rank <= stop.keep && stop.keep <= size,
            "a stop rule at rank {} keeping {} does not fit a list of {size}",
            stop.rank,
            stop.keep
        );
        let widened = 1.0 + stop.margin;
        Self {
            list: Vec::with_capacity(size.min(rows).saturating_add(1)),
            size,
            cursor: 0,
            margin: Margin {
                rank: stop.rank,
                keep: stop.keep,
                factor: widened * widened,
                bar: OrderedFloat(f32::INFINITY),
            },
            peak: 0,
            offers: record.then(Vec::new),
        }
    }

    fn offer(&mut self, id: u32, distance: f32) {
        if let Some(offers) = &mut self.offers {
            offers.push((id, distance));
        }
        let distance = OrderedFloat(distance);
        if self.list.len() >= self.size
            && self
                .list
                .last()
                .is_some_and(|back| back.node.dist <= distance)
        {
            return;
        }
        if self.list.len() >= self.margin.keep
            && self.margin.bar <= distance
            && self.list[self.margin.keep - 1].node.dist <= distance
        {
            return;
        }
        let at = self
            .list
            .partition_point(|entry| entry.node.dist <= distance);
        if at >= self.size {
            return;
        }
        self.list.insert(
            at,
            Entry {
                node: OrderedNode::new(id, distance),
                expanded: false,
            },
        );
        self.list.truncate(self.size);
        self.peak = self.peak.max(self.list.len());
        self.cursor = self.cursor.min(at);
        let margin = &mut self.margin;
        if at < margin.rank && self.list.len() >= margin.rank {
            margin.bar = OrderedFloat(margin.factor * self.list[margin.rank - 1].node.dist.0);
        }
        if at < margin.keep && self.list.len() > margin.keep {
            let nearer =
                self.list[margin.keep..].partition_point(|entry| entry.node.dist < margin.bar);
            self.list.truncate(margin.keep + nearer);
        }
    }

    fn offer_all<C: DistCalculator>(
        &mut self,
        ids: &[u32],
        calculator: &Counting<C>,
        ahead: usize,
    ) {
        for id in ids.iter().take(ahead) {
            calculator.prefetch(*id);
        }
        for (position, id) in ids.iter().enumerate() {
            if ahead != 0
                && let Some(later) = ids.get(position.saturating_add(ahead))
            {
                calculator.prefetch(*later);
            }
            self.offer(*id, calculator.distance(*id));
        }
    }

    fn next_unexpanded(&mut self) -> Option<u32> {
        let position = self.cursor
            + self.list[self.cursor..]
                .iter()
                .position(|entry| !entry.expanded)?;
        if position >= self.margin.rank && self.margin.bar <= self.list[position].node.dist {
            self.cursor = position;
            return None;
        }
        self.list[position].expanded = true;
        self.cursor = position + 1;
        Some(self.list[position].node.id)
    }
}

/// The walk's coded distances, counted, with the count at which the first vertex of the
/// query's true top `k` was measured.
struct Counting<'a, C> {
    inner: &'a C,
    truth: &'a [u32],
    calls: Cell<u32>,
    first_hit: Cell<u32>,
}

impl<'a, C: DistCalculator> Counting<'a, C> {
    fn new(inner: &'a C, truth: &'a [u32]) -> Self {
        Self {
            inner,
            truth,
            calls: Cell::new(0),
            first_hit: Cell::new(u32::MAX),
        }
    }

    fn distance(&self, id: u32) -> f32 {
        let calls = self.calls.get() + 1;
        self.calls.set(calls);
        if self.first_hit.get() == u32::MAX && self.truth.binary_search(&id).is_ok() {
            self.first_hit.set(calls);
        }
        self.inner.distance(id)
    }

    fn prefetch(&self, id: u32) {
        self.inner.prefetch(id);
    }
}

#[derive(Clone, Copy)]
struct Graph<'a> {
    edges: &'a [u32],
    width: usize,
    rows: usize,
}

/// Where a walk starts: one vertex, the nearest of a set (each measured to choose it), or
/// a whole set offered before the first hop. Sets are sorted and free of repeats.
#[derive(Clone, Copy)]
enum Seeds<'a> {
    One(u32),
    Nearest(&'a [u32]),
    All(&'a [u32]),
}

struct Walked {
    list: Vec<OrderedNode>,
    seed_cost: u32,
    fresh: u32,
    expansions: u32,
    hops: u32,
    first_hit: u32,
    remeasured: u32,
    capped: bool,
}

/// Production's lazy walk over resident edges (`src/lazy.rs:176-293`, `fresh_neighbours`
/// at `:499-526`) from `seeds`: hops of up to `BEAM` vertices taken in list order and
/// sorted by id, their unseen out-edges offered in hop order then slot order, a vertex
/// marked when it is collected. `Seeds::One(medoid)` is exactly production's start, and
/// `Seeds::Nearest` its start at the nearest entry point (`WalkStart::NearestEntry`).
fn walk<C: DistCalculator>(
    graph: &Graph,
    calculator: &Counting<C>,
    seeds: Seeds,
    stop: StopRule,
    cap: usize,
    scratch: &mut Scratch,
    record: bool,
) -> Walked {
    scratch.generation = scratch.generation.checked_add(1).unwrap();
    let generation = scratch.generation;
    let mut list = MarginList::new(cap, graph.rows, stop, record);
    let (chosen, seed_cost) = match seeds {
        Seeds::One(id) => {
            mark(&mut scratch.seen, generation, id);
            list.offer(id, calculator.distance(id));
            (id, 1)
        }
        Seeds::Nearest(entries) => {
            let mut best = (f32::INFINITY, u32::MAX);
            for entry in entries {
                let distance = calculator.distance(*entry);
                if distance < best.0 {
                    best = (distance, *entry);
                }
            }
            assert_ne!(best.1, u32::MAX, "no entry vertex has a finite distance");
            if record {
                // The choice is the argmin of its own distances, lowest id among equals.
                let check = entries
                    .iter()
                    .map(|entry| (calculator.inner.distance(*entry), *entry))
                    .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
                    .unwrap();
                assert_eq!(check.1, best.1, "the nearest entry is not the argmin");
            }
            mark(&mut scratch.seen, generation, best.1);
            list.offer(best.1, best.0);
            (best.1, entries.len() as u32)
        }
        Seeds::All(entries) => {
            for entry in entries {
                assert!(
                    mark(&mut scratch.seen, generation, *entry),
                    "entry {entry} is seeded twice"
                );
                list.offer(*entry, calculator.distance(*entry));
            }
            (u32::MAX, entries.len() as u32)
        }
    };
    let (mut fresh, mut expansions, mut hops) = (0u32, 0u32, 0u32);
    loop {
        scratch.frontier.clear();
        while scratch.frontier.len() < BEAM {
            let Some(id) = list.next_unexpanded() else {
                break;
            };
            scratch.frontier.push(id);
        }
        if scratch.frontier.is_empty() {
            break;
        }
        hops += 1;
        expansions += scratch.frontier.len() as u32;
        scratch.frontier.sort_unstable();
        scratch.fresh.clear();
        for vertex in &scratch.frontier {
            let slots = &graph.edges[*vertex as usize * graph.width..][..graph.width];
            for neighbor in &slots[..degree_of(slots)] {
                if mark(&mut scratch.seen, generation, *neighbor) {
                    scratch.fresh.push(*neighbor);
                }
            }
        }
        fresh += scratch.fresh.len() as u32;
        list.offer_all(&scratch.fresh, calculator, PREFETCH);
    }
    assert_eq!(
        calculator.calls.get(),
        seed_cost + fresh,
        "the walk measured other distances than it counted"
    );
    let remeasured = match seeds {
        Seeds::Nearest(entries) => entries
            .iter()
            .filter(|entry| **entry != chosen && scratch.is_marked(**entry))
            .count() as u32,
        _ => 0,
    };
    let capped = list.peak >= cap;
    if let Some(offers) = list.offers.take() {
        // The list under its rule is always the front of the plain list fed the same
        // offers under the same cap (`src/search.rs:312-319`).
        let mut plain = SearchList::new(cap, graph.rows);
        for (id, distance) in offers {
            plain.offer(id, distance);
        }
        let plain = plain.into_candidates();
        assert!(
            list.list.len() <= plain.len()
                && list.list.iter().zip(&plain).all(|(ours, theirs)| {
                    ours.node.id == theirs.id
                        && ours.node.dist.0.to_bits() == theirs.dist.0.to_bits()
                }),
            "the list under its stop rule is not the front of the plain list"
        );
    }
    let list = list
        .list
        .into_iter()
        .map(|entry| entry.node)
        .collect::<Vec<_>>();
    let mut ids = list.iter().map(|node| node.id).collect::<Vec<_>>();
    ids.sort_unstable();
    assert!(
        ids.windows(2).all(|pair| pair[0] != pair[1]),
        "a vertex is in the list twice"
    );
    Walked {
        list,
        seed_cost,
        fresh,
        expansions,
        hops,
        first_hit: calculator.first_hit.get(),
        remeasured,
        capped,
    }
}

#[derive(Clone)]
struct Outcome {
    answer: Vec<Neighbor>,
    coded: Vec<Neighbor>,
    kept: u32,
}

/// Production's `merge` (`src/query.rs:2837-2851`).
fn merge(mut found: Vec<Neighbor>, k: usize) -> Vec<Neighbor> {
    found.sort_by(|left, right| {
        left.row_addr
            .cmp(&right.row_addr)
            .then(left.distance.total_cmp(&right.distance))
    });
    found.dedup_by_key(|neighbor| neighbor.row_addr);
    found.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then(left.row_addr.cmp(&right.row_addr))
    });
    found.truncate(k);
    found
}

/// Production's post-walk for one partition: `allocate` (`src/query.rs:2788-2825`),
/// `coded_answer` (`:2631-2649`), `measure` (`src/lazy.rs:453-478`) and `answer`
/// (`src/query.rs:2575-2615`), then `merge`.
fn finish(
    list: &[OrderedNode],
    exact: &impl DistCalculator,
    row_ids: &[u64],
    k: usize,
    budget: usize,
) -> Outcome {
    let mut candidates = list
        .iter()
        .map(|node| (node.id, row_ids[node.id as usize], node.dist.0))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|candidate| candidate.0);
    if candidates.len() > budget {
        let mut ranked = (0..candidates.len()).collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            let (left, right) = (&candidates[*left], &candidates[*right]);
            left.2.total_cmp(&right.2).then(left.1.cmp(&right.1))
        });
        ranked.truncate(budget);
        ranked.sort_unstable();
        candidates = ranked.into_iter().map(|at| candidates[at]).collect();
    }
    let coded = merge(
        candidates
            .iter()
            .map(|(_, row_addr, coded)| Neighbor {
                row_addr: *row_addr,
                distance: *coded,
            })
            .collect(),
        k,
    );
    let mut rescored = candidates
        .iter()
        .map(|(id, row_addr, _)| Neighbor {
            row_addr: *row_addr,
            distance: exact.distance(*id),
        })
        .collect::<Vec<_>>();
    rescored.sort_by(|left, right| left.distance.total_cmp(&right.distance));
    assert!(
        rescored
            .iter()
            .all(|neighbor| neighbor.distance.is_finite()),
        "a re-scored distance is not finite"
    );
    rescored.truncate(k);
    Outcome {
        answer: merge(rescored, k),
        coded,
        kept: candidates.len() as u32,
    }
}

/// One query's counters at one point of one arm, as `ep.py` reads them.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Record {
    hits: u8,
    coded: u8,
    seed: u16,
    fresh: u32,
    expansions: u32,
    hops: u16,
    first: u32,
    remeasured: u16,
    capped: u8,
    kept: u16,
}

impl Record {
    fn write(&self, sink: &mut Vec<u8>) {
        let start = sink.len();
        sink.push(self.hits);
        sink.push(self.coded);
        sink.extend_from_slice(&self.seed.to_le_bytes());
        sink.extend_from_slice(&self.fresh.to_le_bytes());
        sink.extend_from_slice(&self.expansions.to_le_bytes());
        sink.extend_from_slice(&self.hops.to_le_bytes());
        sink.extend_from_slice(&self.first.to_le_bytes());
        sink.extend_from_slice(&self.remeasured.to_le_bytes());
        sink.push(self.capped);
        sink.extend_from_slice(&self.kept.to_le_bytes());
        debug_assert_eq!(sink.len() - start, RECORD_BYTES);
    }

    /// The distance count production reports: routing, the walk, the re-score.
    fn comparisons(&self) -> u64 {
        1 + u64::from(self.seed) + u64::from(self.fresh) + u64::from(self.kept)
    }
}

#[derive(Clone)]
enum Start {
    Medoid,
    Oracle,
    Nearest(Arc<Vec<u32>>),
    All(Arc<Vec<u32>>),
}

#[derive(Clone)]
struct Arm {
    label: String,
    family: &'static str,
    mode: &'static str,
    count: usize,
    seed: u64,
    start: Start,
}

#[derive(Clone, Copy)]
struct Context<'a> {
    graph: Graph<'a>,
    medoid: u32,
    row_ids: &'a [u64],
    positions: &'a HashMap<u64, u64>,
    truth: &'a [Truth],
    queries: &'a [ArrayRef],
    sq8: &'a ScalarQuantizationStorage,
    exact: &'a FlatFloatStorage,
    threads: usize,
}

impl Context<'_> {
    fn truth_of(&self, query: usize, k: usize) -> &[u32] {
        match k {
            10 => &self.truth[query].top10,
            100 => &self.truth[query].top100,
            _ => unreachable!("no truth at depth {k}"),
        }
    }

    fn hits(&self, found: &[Neighbor], query: usize, k: usize) -> u8 {
        let truth = self.truth_of(query, k);
        found
            .iter()
            .filter(|neighbor| {
                truth
                    .binary_search(&(self.positions[&neighbor.row_addr] as u32))
                    .is_ok()
            })
            .count() as u8
    }

    /// Every query walked from `arm` at every point of `points`, indexed
    /// `[point][query]`, each with its outcome when `outcomes` asks for it.
    fn run(
        &self,
        arm: &Arm,
        setting: &Setting,
        points: &[Point],
        outcomes: bool,
    ) -> Vec<Vec<(Record, Option<Outcome>)>> {
        let queries = self.queries.len();
        let found = std::thread::scope(|scope| {
            let handles = (0..self.threads)
                .map(|thread| {
                    scope.spawn(move || {
                        let mut scratch = Scratch::new(self.graph.rows);
                        let mut mine = Vec::new();
                        for query in (thread..queries).step_by(self.threads) {
                            let coded = self.sq8.dist_calculator(self.queries[query].clone(), 0.0);
                            let exact =
                                self.exact.dist_calculator(self.queries[query].clone(), 0.0);
                            let seeds = match &arm.start {
                                Start::Medoid => Seeds::One(self.medoid),
                                Start::Oracle => Seeds::One(self.truth[query].nearest),
                                Start::Nearest(entries) => Seeds::Nearest(entries),
                                Start::All(entries) => Seeds::All(entries),
                            };
                            let truth = self.truth_of(query, setting.k);
                            for (at, point) in points.iter().enumerate() {
                                let counting = Counting::new(&coded, truth);
                                let walked = walk(
                                    &self.graph,
                                    &counting,
                                    seeds,
                                    point.stop(setting),
                                    setting.cap,
                                    &mut scratch,
                                    query % REPLAY_EVERY == 0,
                                );
                                let outcome = finish(
                                    &walked.list,
                                    &exact,
                                    self.row_ids,
                                    setting.k,
                                    setting.budget,
                                );
                                let record = Record {
                                    hits: self.hits(&outcome.answer, query, setting.k),
                                    coded: self.hits(&outcome.coded, query, setting.k),
                                    seed: u16::try_from(walked.seed_cost).unwrap(),
                                    fresh: walked.fresh,
                                    expansions: walked.expansions,
                                    hops: u16::try_from(walked.hops).unwrap(),
                                    first: walked.first_hit,
                                    remeasured: u16::try_from(walked.remeasured).unwrap(),
                                    capped: u8::from(walked.capped),
                                    kept: u16::try_from(outcome.kept).unwrap(),
                                };
                                mine.push((at, query, record, outcomes.then_some(outcome)));
                            }
                        }
                        mine
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut slots = (0..points.len())
            .map(|_| (0..queries).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for (at, query, record, outcome) in found {
            slots[at][query] = Some((record, outcome));
        }
        slots
            .into_iter()
            .map(|row| row.into_iter().map(Option::unwrap).collect())
            .collect()
    }

    fn records(&self, arm: &Arm, setting: &Setting, points: &[Point]) -> Vec<Vec<Record>> {
        self.run(arm, setting, points, false)
            .into_iter()
            .map(|row| row.into_iter().map(|(record, _)| record).collect())
            .collect()
    }
}

/// Recall of one half of the queries (`fold` 0 the even ones, 1 the odd ones).
fn fold_recall(records: &[Record], fold: usize, k: usize) -> f64 {
    let (hits, count) = records
        .iter()
        .skip(fold)
        .step_by(2)
        .fold((0u64, 0u64), |(hits, count), record| {
            (hits + u64::from(record.hits), count + 1)
        });
    hits as f64 / (count as f64 * k as f64)
}

/// Pool-adjacent-violators with equal weights: the non-decreasing fit `ep.py` reads its
/// crossings from, step for step, so the sweep stops where the reading needs it to.
fn isotonic(values: &[f64]) -> Vec<f64> {
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(values.len());
    for value in values {
        blocks.push((*value, 1));
        while let [.., (low_sum, low_count), (high_sum, high_count)] = blocks[..]
            && low_sum / low_count as f64 > high_sum / high_count as f64
        {
            blocks.pop();
            let low = blocks.last_mut().unwrap();
            low.0 += high_sum;
            low.1 += high_count;
        }
    }
    blocks
        .iter()
        .flat_map(|(sum, count)| std::iter::repeat_n(sum / *count as f64, *count))
        .collect()
}

/// The margin points walked so far and their recall on `fold`, in axis order.
fn margin_curve(
    walked: &BTreeMap<Point, Vec<Record>>,
    fold: usize,
    k: usize,
) -> (Vec<Point>, Vec<f64>) {
    walked
        .iter()
        .filter(|(point, _)| matches!(point, Point::Margin(_)))
        .map(|(point, records)| (*point, fold_recall(records, fold, k)))
        .unzip()
}

/// Every point an arm needs for its crossing on both halves: the coarse margins until two
/// past the crossing of the isotonic fit on each, ranks under zero when zero already
/// clears the bar, and the fine margins inside each half's bracket - the isotonic one,
/// which `ep.py` reads, and the first raw one, which it checks against.
fn sweep(
    context: &Context,
    arm: &Arm,
    setting: &Setting,
    bar: f64,
) -> BTreeMap<Point, Vec<Record>> {
    let k = setting.k;
    let mut walked = BTreeMap::new();
    let past_crossing = |walked: &BTreeMap<Point, Vec<Record>>, fold: usize| {
        let (_, recall) = margin_curve(walked, fold, k);
        isotonic(&recall)
            .iter()
            .position(|fitted| *fitted >= bar)
            .is_some_and(|first| recall.len() >= first + 3)
    };
    let mut next = 0u32;
    while next <= setting.ceiling && !(past_crossing(&walked, 0) && past_crossing(&walked, 1)) {
        let batch = (0..COARSE_BATCH as u32)
            .map(|step| next + step * setting.coarse)
            .filter(|units| *units <= setting.ceiling)
            .map(Point::Margin)
            .collect::<Vec<_>>();
        next += COARSE_BATCH as u32 * setting.coarse;
        for (point, records) in batch.iter().zip(context.records(arm, setting, &batch)) {
            walked.insert(*point, records);
        }
    }
    let zero_clears = |walked: &BTreeMap<Point, Vec<Record>>, fold: usize| {
        let lowest = walked.iter().next().unwrap();
        fold_recall(lowest.1, fold, k) >= bar
    };
    let mut rank = k;
    while rank > setting.rank_step && (zero_clears(&walked, 0) || zero_clears(&walked, 1)) {
        rank -= setting.rank_step;
        let point = Point::Rank(rank);
        let records = context.records(arm, setting, &[point]).pop().unwrap();
        walked.insert(point, records);
    }
    let mut fine = BTreeSet::new();
    for fold in 0..2 {
        let (points, recall) = margin_curve(&walked, fold, k);
        let fitted = isotonic(&recall);
        for curve in [&fitted, &recall] {
            if let Some(high) = curve.iter().position(|value| *value >= bar)
                && high > 0
                && let (Point::Margin(low), Point::Margin(high)) = (points[high - 1], points[high])
            {
                let mut units = low + setting.fine;
                while units < high {
                    fine.insert(Point::Margin(units));
                    units += setting.fine;
                }
            }
        }
    }
    let fine = fine
        .into_iter()
        .filter(|point| !walked.contains_key(point))
        .collect::<Vec<_>>();
    if !fine.is_empty() {
        for (point, records) in fine.iter().zip(context.records(arm, setting, &fine)) {
            walked.insert(*point, records);
        }
    }
    walked
}

/// The vertex nearest each target by the exact store's own distance, lowest id among
/// equals: rows are taken in blocks and each block is measured against every target, so it
/// stays in cache while the targets pass over it.
fn nearest_vertices(store: &FlatFloatStorage, targets: &[ArrayRef], threads: usize) -> Vec<u32> {
    const BLOCK: usize = 512;
    let rows = store.len();
    let blocks = rows.div_ceil(BLOCK);
    let found = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || {
                    let calculators = targets
                        .iter()
                        .map(|target| store.dist_calculator(target.clone(), 0.0))
                        .collect::<Vec<_>>();
                    let mut best = vec![(f32::INFINITY, u32::MAX); targets.len()];
                    for block in (thread..blocks).step_by(threads) {
                        let rows = (block * BLOCK) as u32..((block + 1) * BLOCK).min(rows) as u32;
                        for (calculator, best) in calculators.iter().zip(best.iter_mut()) {
                            for row in rows.clone() {
                                let distance = calculator.distance(row);
                                if distance < best.0 {
                                    *best = (distance, row);
                                }
                            }
                        }
                    }
                    best
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    (0..targets.len())
        .map(|target| {
            let (distance, row) = found
                .iter()
                .map(|best| best[target])
                .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
                .unwrap();
            assert!(distance.is_finite() && row != u32::MAX);
            row
        })
        .collect()
}

fn rows_of(values: &[f32], dimension: usize) -> Vec<ArrayRef> {
    values
        .chunks_exact(dimension)
        .map(|row| Arc::new(Float32Array::from(row.to_vec())) as ArrayRef)
        .collect()
}

/// `train_router`'s k-means (`src/builder.rs:857-932`) with `count` centroids under
/// `seed`, and the vertex nearest each centroid; returned sorted, repeats removed.
fn kmeans_entries(
    vectors: &FixedSizeListArray,
    store: &FlatFloatStorage,
    count: usize,
    seed: u64,
    threads: usize,
) -> Vec<u32> {
    let rows = vectors.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    let sample_size = KMEANS_SAMPLE_RATE * count;
    let training = if rows > sample_size {
        let picked = rand::seq::index::sample(&mut rng, rows, sample_size)
            .into_iter()
            .map(|row| row as u32)
            .collect::<Vec<_>>();
        take_rows(vectors, &picked)
    } else {
        vectors.clone()
    };
    let init = take_rows(
        &training,
        &rand::seq::index::sample(&mut rng, training.len(), count)
            .into_iter()
            .map(|row| row as u32)
            .collect::<Vec<_>>(),
    );
    let params = KMeansParams::new(Some(Arc::new(init)), KMEANS_MAX_ITERS, 1, DistanceType::L2)
        .with_hierarchical_k(1);
    let kmeans = KMeans::new_with_params(&training, count, &params).unwrap();
    let dimension = vectors.value_length() as usize;
    let centroids = kmeans.centroids.as_primitive::<Float32Type>().values();
    assert_eq!(centroids.len(), count * dimension);
    let mut entries = nearest_vertices(store, &rows_of(centroids, dimension), threads);
    entries.sort_unstable();
    entries.dedup();
    entries
}

fn random_entries(rows: usize, count: usize, seed: u64) -> Vec<u32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut entries = rand::seq::index::sample(&mut rng, rows, count)
        .into_iter()
        .map(|row| row as u32)
        .collect::<Vec<_>>();
    entries.sort_unstable();
    entries
}

/// The stand's per-row work line (`examples/ivf_rq_ab.rs:827-848`).
struct Work {
    mean: f64,
    median: u64,
    p99: u64,
    most: u64,
    worst_tenth: f64,
}

impl Work {
    fn of(each: &[(u64, f64)]) -> Self {
        let mut distances = each.iter().map(|(count, _)| *count).collect::<Vec<_>>();
        distances.sort_unstable();
        let mut recalls = each.iter().map(|(_, recall)| *recall).collect::<Vec<_>>();
        recalls.sort_by(f64::total_cmp);
        let rank = |share: f64| {
            let at = (share * distances.len() as f64).ceil() as usize;
            distances[at.clamp(1, distances.len()) - 1]
        };
        let tenth = recalls.len().div_ceil(10);
        Self {
            mean: distances.iter().sum::<u64>() as f64 / distances.len() as f64,
            median: rank(0.5),
            p99: rank(0.99),
            most: distances[distances.len() - 1],
            worst_tenth: recalls[..tenth].iter().sum::<f64>() / tenth as f64,
        }
    }
}

/// A walk row of a round log with the work line under it, every field as printed.
struct LoggedRow {
    margin: String,
    recall: String,
    coded: String,
    work: String,
}

fn logged_rows(path: &str, budget: usize) -> Vec<LoggedRow> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let label = format!("vamana walk margin b={budget} ");
    let mut rows = Vec::new();
    let mut pending = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&label) {
            let words = rest.split_whitespace().collect::<Vec<_>>();
            pending = Some((
                words[0].to_string(),
                words[1].to_string(),
                words[2].to_string(),
            ));
        } else if let Some(rest) = line.strip_prefix("# work of the row above: ")
            && let Some((margin, recall, coded)) = pending.take()
        {
            rows.push(LoggedRow {
                margin,
                recall,
                coded,
                work: rest.to_string(),
            });
        }
    }
    assert!(!rows.is_empty(), "no margin walk rows in {path}");
    rows
}

/// The grid point of a margin printed in a log, which has to be one exactly.
fn grid_point(margin: &str) -> Point {
    let value = margin.parse::<f64>().unwrap();
    let units = (value * f64::from(MARGIN_UNITS)).round() as u32;
    let point = Point::Margin(units);
    assert_eq!(
        point.margin().to_bits(),
        margin.parse::<f32>().unwrap().to_bits(),
        "margin {margin} has no exact grid point"
    );
    point
}

/// `(k, margin)` of every margin the locator wrote for this dataset's entry-point arm
/// (`<dataset> <k> <budget> <margin>,<margin> <cap>` a line); none without the file.
fn located_margins(path: &str, prefix: &str) -> Vec<(usize, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let words = line.split_whitespace().collect::<Vec<_>>();
            (words.len() == 5 && words[0] == prefix).then(|| {
                let k = words[1].parse::<usize>().unwrap();
                words[3]
                    .split(',')
                    .map(|margin| (k, margin.to_string()))
                    .collect::<Vec<_>>()
            })
        })
        .flatten()
        .collect()
}

/// `VamanaIndex::search` from `start` at every point, against the walk of the arm whose
/// runs these are: the same answer and coded answer to the last bit and the same
/// comparisons, query by query. `gate` names the gate a difference fails.
async fn production_walks_it(
    index: &VamanaIndex,
    setting: &Setting,
    points: &[Point],
    runs: &[Vec<(Record, Option<Outcome>)>],
    (query_values, dimension): (&[f32], usize),
    start: WalkStart,
    gate: &str,
) {
    for (point, outcomes) in points.iter().zip(runs) {
        let params = SearchParams::new(setting.k)
            .with_nprobes(1)
            .with_search_list_size(setting.cap)
            .with_mode(WalkMode::Lazy)
            .with_beam_width(BEAM)
            .with_prefetch_ahead(PREFETCH)
            .with_resident_edges(true)
            .with_rescore_budget(setting.budget)
            .with_report_coded(true)
            .with_stop_margin(point.margin())
            .with_start(start);
        for (query, (record, outcome)) in outcomes.iter().enumerate() {
            let outcome = outcome.as_ref().unwrap();
            let theirs = index
                .search(&query_values[query * dimension..][..dimension], &params)
                .await
                .unwrap();
            assert!(
                same_neighbors(&outcome.answer, &theirs.neighbors),
                "{gate}: query {query} at k = {}, margin {} answers differently from the index",
                setting.k,
                point.label()
            );
            assert!(
                same_neighbors(&outcome.coded, &theirs.coded_neighbors),
                "{gate}: query {query} at k = {}, margin {} has other coded neighbours than the \
                 index",
                setting.k,
                point.label()
            );
            assert_eq!(
                record.comparisons(),
                theirs.comparisons,
                "{gate}: query {query} at k = {}, margin {} counts other comparisons than the \
                 index",
                setting.k,
                point.label()
            );
        }
    }
}

fn same_neighbors(ours: &[Neighbor], theirs: &[Neighbor]) -> bool {
    ours.len() == theirs.len()
        && ours.iter().zip(theirs).all(|(left, right)| {
            left.row_addr == right.row_addr && left.distance.to_bits() == right.distance.to_bits()
        })
}

fn git_head() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let dirty = run(&["status", "--porcelain", "--", "."]);
    format!(
        "{}{}",
        run(&["rev-parse", "--short", "HEAD"]),
        if dirty.is_empty() {
            String::new()
        } else {
            format!(" + uncommitted: {}", dirty.replace('\n', "; "))
        }
    )
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn ids_fnv(ids: &[u32]) -> String {
    format!("{:016x}", fnv(ids.iter().map(|id| u64::from(*id))))
}

#[tokio::main]
async fn main() {
    let started = Instant::now();
    let data_dir = env_string("SIFT_DIR");
    let prefix = std::path::Path::new(&data_dir)
        .file_name()
        .and_then(|name| name.to_str())
        .expect("SIFT_DIR must end in the dataset name")
        .to_string();
    let dataset_dir = env_string("DATASET_DIR");
    let ep_dir = env_string("EP_DIR");
    let bar = env_usize("TARGET", 0) as f64 / 100.0;
    assert!(
        bar > 0.0 && bar < 1.0,
        "set TARGET to the recall bar in per cent"
    );
    let threads = env_usize("THREADS", 10);
    let phase2 = std::env::var("EP_PHASE2").ok().map(|raw| {
        let parts = raw.split(',').collect::<Vec<_>>();
        let [family, mode, count] = parts.as_slice() else {
            panic!("EP_PHASE2 is <kmeans|random>,<nearest|all>,<K>, not {raw}");
        };
        assert!(matches!(*family, "kmeans" | "random") && matches!(*mode, "nearest" | "all"));
        (
            family.to_string(),
            mode.to_string(),
            count.parse::<usize>().unwrap(),
        )
    });
    let gate_only = std::env::var("EP_GATE").is_ok();
    assert!(
        !(gate_only && phase2.is_some()),
        "EP_GATE runs the phase-1 gates; it has no phase 2"
    );
    assert!(
        std::env::var("LANCE_USE_HNSW_SPEEDUP_INDEXING").is_err(),
        "LANCE_USE_HNSW_SPEEDUP_INDEXING makes k-means assign approximately and by thread order"
    );
    let base_path = format!("{data_dir}/{prefix}_base.fvecs");
    let (dimension, rows) = fvecs_shape(&base_path);
    let uri = format!("{dataset_dir}/{prefix}-{rows}-p1-r{DEGREE}-sq8.lance");
    let source_fnv = fnv(include_bytes!("entry_points_walk.rs")
        .iter()
        .map(|byte| u64::from(*byte)));
    let git = git_head();
    println!("# unix time {}", unix_now());
    println!("# git {git}, source fnv {source_fnv:016x}");
    println!(
        "{prefix} {rows} x {dimension}, bar {bar}, index {uri}, threads {threads}, phase {}",
        if phase2.is_some() { 2 } else { 1 }
    );

    let dataset = Dataset::open(&uri).await.unwrap();
    let built = load(&dataset, dimension).await;
    assert_eq!(built.row_ids.len(), rows);
    assert_eq!(
        dataset.count_rows(None).await.unwrap(),
        rows,
        "rows are deleted"
    );
    check_edges(&built.edges, built.width, rows);
    let positions = positions_by_address(&dataset).await;
    for (local, row_addr) in built.row_ids.iter().enumerate() {
        assert_eq!(
            positions[row_addr] as usize, local,
            "local id {local} is not position order, which the whole harness relies on"
        );
    }
    let stored = flat_values::<Float32Type>(&built.vectors);
    check_against_fvecs(&base_path, stored, dimension);
    let replayed = ScalarQuantizer::with_bounds(8, dimension, built.bounds.clone())
        .transform::<Float32Type>(&built.vectors)
        .unwrap();
    assert!(
        flat_values::<UInt8Type>(replayed.as_fixed_size_list())
            == flat_values::<UInt8Type>(&built.codes),
        "the stored codes are not Lance's SQ8 of the stored vectors"
    );
    drop(replayed);
    let resampled = ScalarQuantizer::new(8, dimension)
        .update_bounds::<Float32Type>(&take_rows(&built.vectors, &sample_rows(rows)))
        .unwrap();
    assert_eq!(
        resampled, built.bounds,
        "the sample rule does not give the stored bounds"
    );
    let exact = FlatFloatStorage::new(built.vectors.clone(), DistanceType::L2);
    // The medoid is the vertex nearest the f64 mean (`src/build.rs:279-358`): the stored
    // one, `build::medoid` and this harness's own nearest-vertex search must all agree.
    let mean = {
        let mut sums = vec![0f64; dimension];
        for row in stored.chunks_exact(dimension) {
            for (sum, value) in sums.iter_mut().zip(row) {
                *sum += f64::from(*value);
            }
        }
        let scale = 1.0 / rows as f64;
        sums.iter()
            .map(|sum| (sum * scale) as f32)
            .collect::<Vec<_>>()
    };
    let ours = nearest_vertices(&exact, &rows_of(&mean, dimension), threads)[0];
    let theirs = lance_vamana::build::medoid(&exact, &Comparisons::default()).unwrap();
    assert!(
        ours == built.medoid && theirs == built.medoid,
        "medoid: stored {}, build::medoid {theirs}, nearest-vertex search {ours}",
        built.medoid
    );
    println!(
        "premise: local ids are position order, base file = stored vectors, SQ8 replay = stored \
         codes, sample rule = stored bounds {:?}, medoid {} = build::medoid = nearest to the mean",
        built.bounds, built.medoid
    );

    let (query_values, query_dimension, available) =
        read_fvecs(&format!("{data_dir}/{prefix}_query.fvecs"));
    assert_eq!(query_dimension, dimension);
    let count = env_usize("QUERIES", 1000).min(available);
    let query_values = query_values[..count * dimension].to_vec();
    let queries = rows_of(&query_values, dimension);
    let query_hash = fnv(query_values.iter().map(|value| u64::from(value.to_bits())));
    let base_hash = fnv(stored
        .chunks_exact(dimension)
        .step_by(4099)
        .flatten()
        .map(|value| u64::from(value.to_bits())));
    let truth = ground_truth(
        &exact,
        &queries,
        [query_hash, base_hash],
        threads,
        &format!("{ep_dir}/ep-gt-{prefix}-q{count}.bin"),
    );
    // Required at the thousand queries the cache was made for: it is the truth round A3
    // was scored by, computed by another program.
    let a3_path = format!("{ep_dir}/a3-gt-{prefix}-q{count}.bin");
    match a3_truth(&a3_path, rows, count, query_hash) {
        Some(a3) => {
            assert!(
                truth
                    .iter()
                    .zip(&a3)
                    .all(|(ours, theirs)| ours.top10 == *theirs),
                "the depth-10 truth differs from two_level_walk's cache"
            );
            println!("truth: depth 10 equals two_level_walk's cache on all {count} queries");
        }
        None => assert_ne!(
            count, 1000,
            "{a3_path} is missing or keyed to other rows or queries"
        ),
    }
    for (query, found) in truth.iter().enumerate() {
        assert!(
            found
                .top10
                .iter()
                .all(|id| found.top100.binary_search(id).is_ok()),
            "query {query}: the top 10 is not inside the top 100"
        );
        let calculator = exact.dist_calculator(queries[query].clone(), 0.0);
        let nearest_of_ten = found
            .top10
            .iter()
            .map(|id| calculator.distance(*id))
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            nearest_of_ten.to_bits(),
            found.nearest_distance.to_bits(),
            "query {query}: the oracle's start is not as near as the nearest of the top 10"
        );
        assert_eq!(
            calculator.distance(found.nearest).to_bits(),
            found.nearest_distance.to_bits()
        );
    }
    let truth_fnv = fnv(truth.iter().flat_map(|found| {
        found
            .top10
            .iter()
            .chain(&found.top100)
            .chain(std::iter::once(&found.nearest))
            .map(|id| u64::from(*id))
            .collect::<Vec<_>>()
    }));
    println!(
        "{count} queries, query fnv {query_hash:016x}, base fnv {base_hash:016x}, truth fnv \
         {truth_fnv:016x}"
    );

    let sq8 = scalar_store(
        &built.row_ids,
        Arc::new(built.codes.clone()),
        built.bounds.clone(),
    );
    let context = Context {
        graph: Graph {
            edges: &built.edges,
            width: built.width,
            rows,
        },
        medoid: built.medoid,
        row_ids: &built.row_ids,
        positions: &positions,
        truth: &truth,
        queries: &queries,
        sq8: &sq8,
        exact: &exact,
        threads,
    };
    let out_degree =
        |vertex: u32| degree_of(&built.edges[vertex as usize * built.width..][..built.width]);
    assert!(out_degree(built.medoid) > 0, "the medoid has no out-edges");
    for found in &truth {
        assert!(
            out_degree(found.nearest) > 0,
            "the oracle start {} has no out-edges",
            found.nearest
        );
    }

    let mut gates = Vec::new();
    if phase2.is_none() {
        // V1 and V2: the medoid arm, through the generic seeding path in all three of its
        // forms, is production's walk before any other arm is read.
        let index = VamanaIndex::open(&dataset, INDEX_NAME)
            .await
            .unwrap()
            .with_cache(LanceCache::with_capacity(4 << 30));
        let medoid_set = Arc::new(vec![built.medoid]);
        let forms = [
            Start::Medoid,
            Start::Nearest(medoid_set.clone()),
            Start::All(medoid_set),
        ];
        for setting in &SETTINGS {
            let logged = logged_rows(
                &format!("{ep_dir}/{prefix}-{}-index-r1.log", setting.log_tag),
                setting.budget,
            );
            let mut points = vec![Point::Margin(0), Point::Margin(setting.ceiling)];
            points.extend(logged.iter().map(|row| grid_point(&row.margin)));
            points.sort_unstable();
            points.dedup();
            let runs = forms
                .iter()
                .map(|start| {
                    let arm = Arm {
                        label: "medoid".to_string(),
                        family: "medoid",
                        mode: "one",
                        count: 1,
                        seed: 0,
                        start: start.clone(),
                    };
                    context.run(&arm, setting, &points, true)
                })
                .collect::<Vec<_>>();
            for (form, run) in runs.iter().enumerate().skip(1) {
                for (point, (theirs, ours)) in points.iter().zip(runs[0].iter().zip(run)) {
                    for (query, ((their_record, their), (our_record, our))) in
                        theirs.iter().zip(ours).enumerate()
                    {
                        let (their, our) = (their.as_ref().unwrap(), our.as_ref().unwrap());
                        assert!(
                            their_record == our_record
                                && same_neighbors(&their.answer, &our.answer)
                                && same_neighbors(&their.coded, &our.coded),
                            "V1: seeding form {form} walks query {query} at {} unlike the medoid",
                            point.label()
                        );
                    }
                }
            }
            production_walks_it(
                &index,
                setting,
                &points,
                &runs[0],
                (&query_values, dimension),
                WalkStart::Medoid,
                "V1",
            )
            .await;
            let labels = points.iter().map(|point| point.label()).collect::<Vec<_>>();
            let line = format!(
                "V1: k = {}, the medoid arm in all three seeding forms equals VamanaIndex::search \
                 bit for bit on {count} queries at margins {labels:?}",
                setting.k
            );
            println!("{line}");
            gates.push(line);
            for row in &logged {
                let at = points
                    .iter()
                    .position(|point| point.margin() == row.margin.parse::<f32>().unwrap())
                    .unwrap();
                let first = &runs[0][at][..LOGGED_QUERIES];
                let k = setting.k as f64;
                let recall = first
                    .iter()
                    .fold(0.0, |sum, (record, _)| sum + f64::from(record.hits) / k)
                    / LOGGED_QUERIES as f64;
                let coded = first
                    .iter()
                    .fold(0.0, |sum, (record, _)| sum + f64::from(record.coded) / k)
                    / LOGGED_QUERIES as f64;
                let work = Work::of(
                    &first
                        .iter()
                        .map(|(record, _)| (record.comparisons(), f64::from(record.hits) / k))
                        .collect::<Vec<_>>(),
                );
                let ours = (
                    format!("{recall:.4}"),
                    format!("{coded:.4}"),
                    format!(
                        "distances a query mean {:.3}, p50 {}, p99 {}, max {}; worst tenth of \
                         queries at recall {:.4}",
                        work.mean, work.median, work.p99, work.most, work.worst_tenth
                    ),
                );
                assert_eq!(
                    (&ours.0, &ours.1, &ours.2),
                    (&row.recall, &row.coded, &row.work),
                    "V2: k = {}, margin {} does not reproduce the round log",
                    setting.k,
                    row.margin
                );
            }
            let line = format!(
                "V2: k = {}, the first {LOGGED_QUERIES} queries reproduce {} rows of the round log \
                 to the digit (recall, coded, mean, p50, p99, max, worst tenth)",
                setting.k,
                logged.len()
            );
            println!("{line}");
            gates.push(line);
        }
    }

    if gate_only {
        // V3: the crate's own entry points, trained by the crate, are this harness's, and
        // its walk from the nearest of them is this harness's kmeans-nearest-64 arm.
        let set = Arc::new(kmeans_entries(
            &built.vectors,
            &exact,
            GATE_ENTRIES,
            KMEANS_SEED,
            threads,
        ));
        let training = Instant::now();
        let trained = VamanaIndex::open(&dataset, INDEX_NAME)
            .await
            .unwrap()
            .train_entry_points(&EntryPointParams::new(GATE_ENTRIES).with_seed(KMEANS_SEED))
            .await
            .unwrap();
        let trained_in = training.elapsed().as_secs_f64();
        let [partition] = trained.partitions() else {
            panic!(
                "V3: the crate trained {} partitions, not the index's one",
                trained.partitions().len()
            );
        };
        assert_eq!(
            partition.entries, *set,
            "V3: the crate trained other entry points than this harness"
        );
        let line = format!(
            "V3: the crate trains this harness's kmeans K = {GATE_ENTRIES} seed {KMEANS_SEED} \
             entry points: {} distinct, fnv {}, trained in {trained_in:.2}s",
            set.len(),
            ids_fnv(&set)
        );
        println!("{line}");
        gates.push(line);

        let index = VamanaIndex::open(&dataset, INDEX_NAME)
            .await
            .unwrap()
            .with_cache(LanceCache::with_capacity(4 << 30))
            .with_entry_points(Arc::new(trained))
            .unwrap();
        let arm = Arm {
            label: format!("kmeans-nearest-{GATE_ENTRIES}"),
            family: "kmeans",
            mode: "nearest",
            count: GATE_ENTRIES,
            seed: KMEANS_SEED,
            start: Start::Nearest(set.clone()),
        };
        let located_path = format!("{ep_dir}/et-points.txt");
        let located = located_margins(&located_path, &prefix);
        if std::fs::metadata(&located_path).is_ok() {
            for setting in &SETTINGS {
                assert!(
                    located.iter().filter(|(k, _)| *k == setting.k).count() == 2,
                    "{located_path} has no pair for {prefix} at k = {}",
                    setting.k
                );
            }
        }
        for setting in &SETTINGS {
            let logged = logged_rows(
                &format!("{ep_dir}/{prefix}-{}-index-r1.log", setting.log_tag),
                setting.budget,
            );
            let mut points = vec![Point::Margin(0), Point::Margin(setting.ceiling)];
            points.extend(logged.iter().map(|row| grid_point(&row.margin)));
            points.extend(
                located
                    .iter()
                    .filter(|(k, _)| *k == setting.k)
                    .map(|(_, margin)| grid_point(margin)),
            );
            points.sort_unstable();
            points.dedup();
            let runs = context.run(&arm, setting, &points, true);
            production_walks_it(
                &index,
                setting,
                &points,
                &runs,
                (&query_values, dimension),
                WalkStart::NearestEntry,
                "V3",
            )
            .await;
            let labels = points.iter().map(|point| point.label()).collect::<Vec<_>>();
            let line = format!(
                "V3: k = {}, the crate's walk from the nearest entry point equals the \
                 kmeans-nearest-{GATE_ENTRIES} arm bit for bit on {count} queries at margins \
                 {labels:?}",
                setting.k
            );
            println!("{line}");
            gates.push(line);
        }
        println!(
            "gates only: {} held in {:.1}s",
            gates.len(),
            started.elapsed().as_secs_f64()
        );
        return;
    }

    // The entry sets.
    let mut entries: Vec<(&'static str, usize, u64, Arc<Vec<u32>>)> = Vec::new();
    let kmeans_started = Instant::now();
    match &phase2 {
        None => {
            for count in ENTRY_COUNTS {
                let set = kmeans_entries(&built.vectors, &exact, count, KMEANS_SEED, threads);
                entries.push(("kmeans", count, KMEANS_SEED, Arc::new(set)));
            }
            let mut rng = SmallRng::seed_from_u64(RANDOM_SEED);
            let pool = rand::seq::index::sample(&mut rng, rows, RANDOM_POOL)
                .into_iter()
                .map(|row| row as u32)
                .collect::<Vec<_>>();
            for count in ENTRY_COUNTS {
                let mut set = pool[..count].to_vec();
                set.sort_unstable();
                entries.push(("random", count, RANDOM_SEED, Arc::new(set)));
            }
            if prefix == "sift" {
                for count in [16, 1024] {
                    let again = kmeans_entries(&built.vectors, &exact, count, KMEANS_SEED, threads);
                    let first = &entries
                        .iter()
                        .find(|(family, k, _, _)| *family == "kmeans" && *k == count)
                        .unwrap()
                        .3;
                    assert_eq!(
                        **first, again,
                        "k-means with K = {count} is not reproducible"
                    );
                }
                gates.push(
                    "premise: k-means at K = 16 and 1024 trained twice gives the same entries"
                        .to_string(),
                );
            }
        }
        Some((_, _, count)) => {
            for seed in KMEANS_EXTRA_SEEDS {
                let set = kmeans_entries(&built.vectors, &exact, *count, seed, threads);
                entries.push(("kmeans", *count, seed, Arc::new(set)));
            }
            for seed in RANDOM_EXTRA_SEEDS {
                entries.push((
                    "random",
                    *count,
                    seed,
                    Arc::new(random_entries(rows, *count, seed)),
                ));
            }
        }
    }
    for (family, count, seed, set) in &entries {
        assert!(set.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            set.iter().all(|vertex| out_degree(*vertex) > 0),
            "an entry vertex of {family} K = {count} has no out-edges"
        );
        println!(
            "entries: {family} K = {count} seed {seed}: {} distinct vertices, fnv {}",
            set.len(),
            ids_fnv(set)
        );
    }
    println!(
        "entry sets built in {:.1}s",
        kmeans_started.elapsed().as_secs_f64()
    );

    let mut arms = Vec::new();
    if phase2.is_none() {
        for (family, start) in [("medoid", Start::Medoid), ("oracle", Start::Oracle)] {
            arms.push(Arm {
                label: family.to_string(),
                family,
                mode: "one",
                count: 1,
                seed: 0,
                start,
            });
        }
    }
    for &(family, count, seed, ref set) in &entries {
        for mode in ["nearest", "all"] {
            if let Some((_, wanted, _)) = &phase2
                && wanted != mode
            {
                continue;
            }
            let suffix = if phase2.is_some() {
                format!("-s{seed}")
            } else {
                String::new()
            };
            arms.push(Arm {
                label: format!("{family}-{mode}-{count}{suffix}"),
                family,
                mode,
                count,
                seed,
                start: if mode == "nearest" {
                    Start::Nearest(set.clone())
                } else {
                    Start::All(set.clone())
                },
            });
        }
    }

    let mut blocks = Vec::new();
    let mut payload = Vec::new();
    for setting in &SETTINGS {
        for arm in &arms {
            let arm_started = Instant::now();
            let walked = sweep(&context, arm, setting, bar);
            let mut line = format!("{} k={}:", arm.label, setting.k);
            for (point, records) in &walked {
                let capped = records.iter().filter(|record| record.capped != 0).count();
                let cost = records
                    .iter()
                    .map(|record| u64::from(record.seed) + u64::from(record.fresh))
                    .sum::<u64>() as f64
                    / records.len() as f64;
                line.push_str(&format!(
                    " {} {:.4}/{:.4} {cost:.0}{}",
                    point.label(),
                    fold_recall(records, 0, setting.k),
                    fold_recall(records, 1, setting.k),
                    if capped > 0 {
                        format!(" capped {capped}")
                    } else {
                        String::new()
                    }
                ));
                blocks.push(serde_json::json!({
                    "arm": arm.label,
                    "family": arm.family,
                    "mode": arm.mode,
                    "count": arm.count,
                    "seed": arm.seed,
                    "k": setting.k,
                    "budget": setting.budget,
                    "cap": setting.cap,
                    "point": point.label(),
                    "kind": match point { Point::Rank(_) => "rank", Point::Margin(_) => "margin" },
                    "rank": match point { Point::Rank(rank) => *rank, Point::Margin(_) => setting.k },
                    "margin": format!("{}", point.margin()),
                    "order": match point { Point::Rank(rank) => *rank as i64 - setting.k as i64, Point::Margin(units) => i64::from(*units) },
                    "block": blocks.len(),
                }));
                for record in records {
                    record.write(&mut payload);
                }
            }
            println!(
                "{line} ({} points, {:.1}s)",
                walked.len(),
                arm_started.elapsed().as_secs_f64()
            );
        }
    }

    let tag = if phase2.is_some() { "-p2" } else { "" };
    let payload_fnv = fnv(payload.iter().map(|byte| u64::from(*byte)));
    std::fs::write(format!("{ep_dir}/ep-{prefix}{tag}-q.bin"), &payload).unwrap();
    let manifest = serde_json::json!({
        "prefix": prefix,
        "rows": rows,
        "dimension": dimension,
        "queries": count,
        "bar": bar,
        "phase": if phase2.is_some() { 2 } else { 1 },
        "phase2": phase2.as_ref().map(|(family, mode, count)| format!("{family},{mode},{count}")),
        "git": git,
        "source_fnv": format!("{source_fnv:016x}"),
        "query_fnv": format!("{query_hash:016x}"),
        "base_fnv": format!("{base_hash:016x}"),
        "truth_fnv": format!("{truth_fnv:016x}"),
        "payload_fnv": format!("{payload_fnv:016x}"),
        "medoid": built.medoid,
        "settings": SETTINGS.iter().map(|setting| serde_json::json!({
            "k": setting.k, "budget": setting.budget, "cap": setting.cap,
            "coarse": setting.coarse, "fine": setting.fine, "ceiling": setting.ceiling,
            "rank_step": setting.rank_step, "units": MARGIN_UNITS,
        })).collect::<Vec<_>>(),
        "entries": entries.iter().map(|(family, count, seed, set)| serde_json::json!({
            "family": family, "count": count, "seed": seed, "distinct": set.len(), "fnv": ids_fnv(set),
        })).collect::<Vec<_>>(),
        "record": {
            "bytes": RECORD_BYTES,
            "fields": ["hits u1", "coded u1", "seed u2", "fresh u4", "expansions u4", "hops u2",
                       "first u4", "remeasured u2", "capped u1", "kept u2"],
        },
        "blocks": blocks,
        "gates": gates,
        "seconds": started.elapsed().as_secs_f64(),
        "complete": true,
    });
    std::fs::write(
        format!("{ep_dir}/ep-{prefix}{tag}.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    println!(
        "# end ep {prefix}{tag}: {} blocks, payload fnv {payload_fnv:016x}, {:.1}s, unix time {}",
        blocks.len(),
        started.elapsed().as_secs_f64(),
        unix_now()
    );
    std::io::stdout().flush().unwrap();
}
