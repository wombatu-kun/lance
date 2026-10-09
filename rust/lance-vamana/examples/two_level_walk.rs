// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whether a walk navigated by a short PCA code keeps the recall of the full-code walk.
//!
//! Two-level codes on an index that is already built: the walk measures its
//! neighbours by the first `m` principal components of each vector (unquantized, at
//! 8 bits under one bound, or at 8 bits under per-dimension bounds), with or without
//! the squared norm of the remaining components; the whole final list is then
//! re-ranked by the index's own SQ8 codes and the nearest 20 are re-scored exactly,
//! as production does. The graph does not depend on the codes, so nothing is rebuilt.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/gist DATASET_DIR=~/vamana-runs A3_DIR=<dir> TARGET=95 \
//!     cargo run --profile release-no-lto --example two_level_walk
//! ```
//!
//! `SIFT_DIR` holds `<prefix>_{base,query}.fvecs`; `DATASET_DIR` the index `ivf_rq_ab`
//! built (`<prefix>-<rows>-p1-r70-sq8.lance`); `A3_DIR` the PCA (`a3-pca-<prefix>.
//! {mean,components,eigenvalues}.f64`) and the round log `<prefix>-aduperf-fixed-r1.log`,
//! and receives the ground-truth cache and `a3-<prefix>-curves.csv`. `TARGET` is the
//! recall bar in per cent; `QUERIES` (default 1000), `THREADS` (default 10) and `MS`
//! (default `64,128,256`, only those below the dimension run).
//!
//! No arm is read before three gates hold: the harness's walk equals
//! `VamanaIndex::search` bit for bit (V1) and the round log to the digit (V2), and the
//! two-level path given the SQ8 code as its head and zero tails equals it too (V3).

use std::collections::HashMap;
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
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::vector::SQ_CODE_COLUMN;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::quantizer::QuantizerBuildParams;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::sq::storage::ScalarQuantizationStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::l2::l2;
use lance_vamana::codes::{CODE_COLUMN, CodeParams};
use lance_vamana::format::{
    INDEX_FILE_NAME, NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, VECTOR_COLUMN,
};
use lance_vamana::io::{PartitionFile, read_partition_batch, read_segment, scan_scheduler};
use lance_vamana::query::{
    Neighbor, SearchParams, VamanaIndex, WalkMode, WalkStart, committed_segments,
};
use lance_vamana::search::SearchList;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const K: usize = 10;
const BUDGET: usize = 20;
const BEAM: usize = 4;
const PREFETCH: usize = 2;
const DEGREE: u32 = 70;
const SMALLEST_LIST: usize = 20;
const LOGGED_QUERIES: usize = 200;
const INDEX_NAME: &str = "vamana_idx";
const ID_COLUMN: &str = "id";
const QUEUE_LIMIT: f64 = 1.25;
const BYTE_FACTOR: f64 = 3.0;
const RESAMPLES: usize = 1000;
const SEED: u64 = 20_260_925;

fn env_string(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

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
        let degree = slots
            .iter()
            .position(|slot| *slot == NO_NEIGHBOR)
            .unwrap_or(width);
        assert!(
            slots[degree..].iter().all(|slot| *slot == NO_NEIGHBOR),
            "vertex {vertex} holds a neighbour after its padding"
        );
        for neighbor in &slots[..degree] {
            assert!((*neighbor as usize) < rows && *neighbor as usize != vertex);
        }
    }
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

/// Row `p` of the base file is the stored vector of `local_of_position[p]`, bit for bit.
fn check_against_fvecs(path: &str, stored: &[f32], dimension: usize, local_of_position: &[u32]) {
    let mut reader = BufReader::with_capacity(1 << 24, File::open(path).unwrap());
    let mut record = vec![0u8; 4 + 4 * dimension];
    for (position, local) in local_of_position.iter().enumerate() {
        reader.read_exact(&mut record).unwrap();
        assert_eq!(
            i32::from_le_bytes(record[..4].try_into().unwrap()) as usize,
            dimension
        );
        let row = &stored[*local as usize * dimension..][..dimension];
        for (value, bytes) in row.iter().zip(record[4..].as_chunks::<4>().0) {
            assert_eq!(
                value.to_bits(),
                u32::from_le_bytes(*bytes),
                "base row {position} differs from the stored vector of local id {local}"
            );
        }
    }
}

/// Exact nearest `K` positions of one query, by brute force over every row.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef) -> Vec<u64> {
    let calculator = store.dist_calculator(query, 0.0);
    let mut scored = (0..store.len() as u32)
        .map(|id| (calculator.distance(id), id))
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(K, |left, right| left.0.total_cmp(&right.0));
    scored.truncate(K);
    scored.into_iter().map(|(_, id)| id as u64).collect()
}

/// `exact_top` for every query, cached in `path` under a key of the base and the queries.
fn ground_truth(
    store: &FlatFloatStorage,
    queries: &[ArrayRef],
    query_hash: u64,
    threads: usize,
    path: &str,
) -> Vec<Vec<u64>> {
    let header = [
        u64::from_le_bytes(*b"A3TRUTH1"),
        store.len() as u64,
        queries.len() as u64,
        K as u64,
        query_hash,
    ];
    if let Ok(bytes) = std::fs::read(path) {
        let words = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .collect::<Vec<_>>();
        if words.len() == header.len() + queries.len() * K && words[..header.len()] == header {
            println!("ground truth from {path}");
            return words[header.len()..]
                .as_chunks::<K>()
                .0
                .iter()
                .map(|ids| ids.to_vec())
                .collect();
        }
    }
    let started = Instant::now();
    let found = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || {
                    (thread..queries.len())
                        .step_by(threads)
                        .map(|query| (query, exact_top(store, queries[query].clone())))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut truth = vec![Vec::new(); queries.len()];
    for (query, top) in found {
        truth[query] = top;
    }
    let mut sink = File::create(path).unwrap();
    for word in header.iter().chain(truth.iter().flatten()) {
        sink.write_all(&word.to_le_bytes()).unwrap();
    }
    println!(
        "brute force ground truth in {:.1}s, cached in {path}",
        started.elapsed().as_secs_f64()
    );
    truth
}

/// The rows `sq_sample` takes (`src/codes.rs:286-304`), as local ids: a strided sample
/// in the order the build read the column, which is position order.
fn sample_rows(local_of_position: &[u32]) -> Vec<u32> {
    let wanted = SQBuildParams {
        num_bits: 8,
        ..Default::default()
    }
    .sample_size();
    let rows = local_of_position.len();
    if rows <= wanted {
        return local_of_position.to_vec();
    }
    let stride = rows / wanted;
    (0..wanted)
        .map(|row| local_of_position[row * stride])
        .collect()
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

#[derive(Clone, Copy)]
struct Graph<'a> {
    edges: &'a [u32],
    width: usize,
    medoid: u32,
    rows: usize,
}

/// Production's lazy walk over resident edges (`src/lazy.rs:194-262` with
/// `fresh_neighbours` at `:454-481`), measured by `nav`: hops of up to `BEAM` vertices
/// taken in list order and sorted by id, their unseen out-edges offered in hop order
/// then slot order, a vertex marked when it is collected.
fn walk(
    graph: &Graph,
    nav: &impl DistCalculator,
    list_size: usize,
    scratch: &mut Scratch,
) -> (Vec<OrderedNode>, u32) {
    scratch.generation = scratch.generation.checked_add(1).unwrap();
    let generation = scratch.generation;
    let mut list = SearchList::new(list_size, graph.rows);
    mark(&mut scratch.seen, generation, graph.medoid);
    let mut distances = 1u32;
    list.offer(graph.medoid, nav.distance(graph.medoid));
    loop {
        scratch.frontier.clear();
        while scratch.frontier.len() < BEAM {
            let Some(node) = list.next_unexpanded() else {
                break;
            };
            scratch.frontier.push(node.id);
        }
        if scratch.frontier.is_empty() {
            break;
        }
        scratch.frontier.sort_unstable();
        scratch.fresh.clear();
        for vertex in &scratch.frontier {
            let slots = &graph.edges[*vertex as usize * graph.width..][..graph.width];
            let degree = slots
                .iter()
                .position(|slot| *slot == NO_NEIGHBOR)
                .unwrap_or(graph.width);
            for neighbor in &slots[..degree] {
                if mark(&mut scratch.seen, generation, *neighbor) {
                    scratch.fresh.push(*neighbor);
                }
            }
        }
        distances += scratch.fresh.len() as u32;
        list.offer_all(&scratch.fresh, nav, PREFETCH);
    }
    (list.into_candidates(), distances)
}

#[derive(Clone, Copy)]
struct Candidate {
    id: u32,
    row_addr: u64,
    navigated: f32,
    coded: f32,
}

#[derive(Clone)]
struct Outcome {
    answer: Vec<Neighbor>,
    coded: Vec<Neighbor>,
    navigated: Vec<Neighbor>,
    walk: u32,
    reranked: u32,
    kept: u32,
}

/// Production's `merge` (`src/query.rs:2242-2256`).
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

/// Production's post-walk for one partition: `allocate` (`src/query.rs:2193-2230`),
/// `coded_answer` (`:2036-2054`), `rescore` (`src/lazy.rs:394-433`) and `answer`
/// (`src/query.rs:1980-2020`), then `merge`. With `rerank` the list is first given
/// its SQ8 distances, because the walk measured it by another code.
fn finish(
    list: Vec<OrderedNode>,
    walk: u32,
    rerank: Option<&impl DistCalculator>,
    exact: &impl DistCalculator,
    row_ids: &[u64],
) -> Outcome {
    let reranked = if rerank.is_some() {
        list.len() as u32
    } else {
        0
    };
    let mut candidates = list
        .iter()
        .map(|node| Candidate {
            id: node.id,
            row_addr: row_ids[node.id as usize],
            navigated: node.dist.0,
            coded: rerank.map_or(node.dist.0, |sq8| sq8.distance(node.id)),
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|candidate| candidate.id);
    let navigated = merge(
        candidates
            .iter()
            .map(|candidate| Neighbor {
                row_addr: candidate.row_addr,
                distance: candidate.navigated,
            })
            .collect(),
        K,
    );
    if candidates.len() > BUDGET {
        let mut ranked = (0..candidates.len()).collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            let (left, right) = (&candidates[*left], &candidates[*right]);
            left.coded
                .total_cmp(&right.coded)
                .then(left.row_addr.cmp(&right.row_addr))
        });
        ranked.truncate(BUDGET);
        ranked.sort_unstable();
        candidates = ranked.into_iter().map(|at| candidates[at]).collect();
    }
    let coded = merge(
        candidates
            .iter()
            .map(|candidate| Neighbor {
                row_addr: candidate.row_addr,
                distance: candidate.coded,
            })
            .collect(),
        K,
    );
    let mut rescored = candidates
        .iter()
        .map(|candidate| Neighbor {
            row_addr: candidate.row_addr,
            distance: exact.distance(candidate.id),
        })
        .collect::<Vec<_>>();
    rescored.sort_by(|left, right| left.distance.total_cmp(&right.distance));
    assert!(
        rescored
            .iter()
            .all(|neighbor| neighbor.distance.is_finite()),
        "a re-scored distance is not finite"
    );
    rescored.truncate(K);
    Outcome {
        answer: merge(rescored, K),
        coded,
        navigated,
        walk,
        reranked,
        kept: candidates.len() as u32,
    }
}

#[derive(Clone, Copy, Default)]
struct Tally {
    answer: u8,
    coded: u8,
    navigated: u8,
    walk: u32,
    reranked: u32,
}

#[derive(Clone, Copy)]
struct Context<'a> {
    graph: Graph<'a>,
    row_ids: &'a [u64],
    positions: &'a HashMap<u64, u64>,
    truth: &'a [Vec<u64>],
    queries: &'a [ArrayRef],
    sq8: &'a ScalarQuantizationStorage,
    exact: &'a FlatFloatStorage,
    threads: usize,
}

impl Context<'_> {
    fn hits(&self, found: &[Neighbor], query: usize) -> u8 {
        let truth = &self.truth[query];
        found
            .iter()
            .filter(|neighbor| truth.contains(&self.positions[&neighbor.row_addr]))
            .count() as u8
    }

    fn tally(&self, query: usize, outcome: &Outcome) -> Tally {
        Tally {
            answer: self.hits(&outcome.answer, query),
            coded: self.hits(&outcome.coded, query),
            navigated: self.hits(&outcome.navigated, query),
            walk: outcome.walk,
            reranked: outcome.reranked,
        }
    }

    /// Every query walked at every list size in `sizes`, `nav` building each query's
    /// navigation distance, `keep` reducing each outcome; indexed `[size][query]`.
    fn run<C: DistCalculator, T: Send>(
        &self,
        sizes: &[usize],
        rerank: bool,
        nav: &(impl Fn(usize) -> C + Sync),
        keep: &(impl Fn(usize, &Outcome) -> T + Sync),
    ) -> Vec<Vec<T>> {
        let queries = self.queries.len();
        let found = std::thread::scope(|scope| {
            let handles = (0..self.threads)
                .map(|thread| {
                    scope.spawn(move || {
                        let mut scratch = Scratch::new(self.graph.rows);
                        let mut mine = Vec::new();
                        for query in (thread..queries).step_by(self.threads) {
                            let navigator = nav(query);
                            let sq8 = self.sq8.dist_calculator(self.queries[query].clone(), 0.0);
                            let exact =
                                self.exact.dist_calculator(self.queries[query].clone(), 0.0);
                            for (at, size) in sizes.iter().enumerate() {
                                let (list, walked) =
                                    walk(&self.graph, &navigator, *size, &mut scratch);
                                let outcome = finish(
                                    list,
                                    walked,
                                    rerank.then_some(&sq8),
                                    &exact,
                                    self.row_ids,
                                );
                                mine.push((at, query, keep(query, &outcome)));
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
        let mut slots = (0..sizes.len())
            .map(|_| (0..queries).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for (at, query, kept) in found {
            slots[at][query] = Some(kept);
        }
        slots
            .into_iter()
            .map(|row| row.into_iter().map(Option::unwrap).collect())
            .collect()
    }

    fn with_queries(&self, count: usize) -> Self {
        Self {
            queries: &self.queries[..count],
            truth: &self.truth[..count],
            ..*self
        }
    }
}

/// The navigation distance of a two-level arm: its head, plus the squared norm of the
/// candidate's tail and of the query's when the arm counts the tail.
struct TwoLevel<'a, H> {
    head: H,
    tails: Option<(&'a [f32], f32)>,
}

impl<H: DistCalculator> DistCalculator for TwoLevel<'_, H> {
    fn distance(&self, id: u32) -> f32 {
        let head = self.head.distance(id);
        match self.tails {
            Some((tails, query_tail)) => head + tails[id as usize] + query_tail,
            None => head,
        }
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        unreachable!("a walk measures one vertex at a time")
    }

    fn prefetch(&self, id: u32) {
        self.head.prefetch(id);
    }
}

/// The unquantized head: plain L2 over the first `m` components.
struct FloatHead<'a> {
    heads: &'a [f32],
    stride: usize,
    query: Vec<f32>,
}

impl DistCalculator for FloatHead<'_> {
    fn distance(&self, id: u32) -> f32 {
        l2(
            &self.query,
            &self.heads[id as usize * self.stride..][..self.query.len()],
        )
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        unreachable!("a walk measures one vertex at a time")
    }
}

/// The head at eight bits under per-dimension bounds, measured asymmetrically: the
/// query stays f32 and each code is decoded at the middle of its bin.
struct PerDimensionHead<'a> {
    codes: &'a [u8],
    step: &'a [f32],
    /// `query - low - step / 2`, so that a code only has to be scaled.
    shifted: Vec<f32>,
}

impl DistCalculator for PerDimensionHead<'_> {
    fn distance(&self, id: u32) -> f32 {
        let width = self.shifted.len();
        let code = &self.codes[id as usize * width..][..width];
        let mut lanes = [0f32; 8];
        for ((shifted, step), code) in self
            .shifted
            .as_chunks::<8>()
            .0
            .iter()
            .zip(self.step.as_chunks::<8>().0)
            .zip(code.as_chunks::<8>().0)
        {
            for (((lane, shifted), step), code) in lanes.iter_mut().zip(shifted).zip(step).zip(code)
            {
                let gap = shifted - f32::from(*code) * step;
                *lane += gap * gap;
            }
        }
        lanes.iter().sum()
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        unreachable!("a walk measures one vertex at a time")
    }
}

struct Pca {
    mean: Vec<f64>,
    components: Vec<f64>,
    eigenvalues: Vec<f64>,
}

fn read_f64s(path: &str) -> Vec<f64> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| f64::from_le_bytes(*word))
        .collect()
}

fn load_pca(prefix: &str, dimension: usize) -> Pca {
    let pca = Pca {
        mean: read_f64s(&format!("{prefix}.mean.f64")),
        components: read_f64s(&format!("{prefix}.components.f64")),
        eigenvalues: read_f64s(&format!("{prefix}.eigenvalues.f64")),
    };
    assert_eq!(pca.mean.len(), dimension);
    assert_eq!(pca.components.len(), dimension * dimension);
    assert_eq!(pca.eigenvalues.len(), dimension);
    assert!(pca.eigenvalues.windows(2).all(|pair| pair[0] >= pair[1]));
    let mut worst = 0f64;
    for left in 0..dimension {
        let row = &pca.components[left * dimension..][..dimension];
        for right in left..dimension {
            let other = &pca.components[right * dimension..][..dimension];
            let dot = row.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
            let expected = if left == right { 1.0 } else { 0.0 };
            worst = worst.max((dot - expected).abs());
        }
    }
    assert!(
        worst < 1e-9,
        "the components are not orthonormal: {worst:e}"
    );
    let sum = pca.eigenvalues.iter().sum::<f64>();
    let squares = pca
        .eigenvalues
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let hash = fnv(pca
        .mean
        .iter()
        .chain(&pca.components)
        .chain(&pca.eigenvalues)
        .map(|value| value.to_bits()));
    println!(
        "pca: participation ratio {:.4}, orthonormality error {worst:.2e}, fnv {hash:016x}",
        sum * sum / squares
    );
    pca
}

/// Rows projected onto the first `width` components: the heads (f32, `width` a row),
/// the squared tail past each `m` of `ms` (f32, one array per `m`), and the f64 totals
/// of head energy per `m` and of all energy, for the premise check.
struct Projected {
    heads: Vec<f32>,
    width: usize,
    tails: Vec<Vec<f32>>,
    head_energy: Vec<f64>,
    energy: f64,
}

fn project(
    values: &[f32],
    dimension: usize,
    pca: &Pca,
    width: usize,
    ms: &[usize],
    threads: usize,
) -> Projected {
    let rows = values.len() / dimension;
    // Transposed so that each input value scales one contiguous row: the inner loop
    // is then an axpy over independent accumulators, which vectorises in f64.
    let mut transposed = vec![0f64; dimension * width];
    for component in 0..width {
        for input in 0..dimension {
            transposed[input * width + component] = pca.components[component * dimension + input];
        }
    }
    let mut heads = vec![0f32; rows * width];
    let mut tails = vec![0f32; rows * ms.len()];
    let chunk = rows.div_ceil(threads).max(1);
    let partials = std::thread::scope(|scope| {
        let handles = heads
            .chunks_mut(chunk * width)
            .zip(tails.chunks_mut(chunk * ms.len()))
            .enumerate()
            .map(|(at, (heads, tails))| {
                let transposed = &transposed;
                scope.spawn(move || {
                    let mut accumulated = vec![0f64; width];
                    let mut head_energy = vec![0f64; ms.len()];
                    let mut energy = 0f64;
                    for (row, (head, tail)) in heads
                        .chunks_exact_mut(width)
                        .zip(tails.chunks_exact_mut(ms.len()))
                        .enumerate()
                    {
                        let vector = &values[(at * chunk + row) * dimension..][..dimension];
                        accumulated.fill(0.0);
                        let mut norm = 0f64;
                        for (input, value) in vector.iter().enumerate() {
                            let centred = f64::from(*value) - pca.mean[input];
                            norm += centred * centred;
                            for (slot, weight) in accumulated
                                .iter_mut()
                                .zip(&transposed[input * width..][..width])
                            {
                                *slot += centred * weight;
                            }
                        }
                        for (slot, value) in head.iter_mut().zip(&accumulated) {
                            *slot = *value as f32;
                        }
                        for (at_m, m) in ms.iter().enumerate() {
                            let kept = accumulated[..*m].iter().map(|v| v * v).sum::<f64>();
                            let rest = norm - kept;
                            assert!(rest > -1e-9 * norm.max(1.0), "a negative tail: {rest}");
                            tail[at_m] = rest.max(0.0) as f32;
                            head_energy[at_m] += kept;
                        }
                        energy += norm;
                    }
                    (head_energy, energy)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut head_energy = vec![0f64; ms.len()];
    let mut energy = 0f64;
    for (partial, total) in partials {
        for (sum, part) in head_energy.iter_mut().zip(partial) {
            *sum += part;
        }
        energy += total;
    }
    let tails = (0..ms.len())
        .map(|at_m| tails.iter().skip(at_m).step_by(ms.len()).copied().collect())
        .collect();
    Projected {
        heads,
        width,
        tails,
        head_energy,
        energy,
    }
}

/// Every stored vector's norm survives a projection onto all components, on a sample.
fn check_norms(values: &[f32], dimension: usize, pca: &Pca) {
    let rows = values.len() / dimension;
    let mut worst = 0f64;
    for row in (0..rows).step_by((rows / 1000).max(1)) {
        let centred = values[row * dimension..][..dimension]
            .iter()
            .zip(&pca.mean)
            .map(|(value, mean)| f64::from(*value) - mean)
            .collect::<Vec<_>>();
        let before = centred.iter().map(|value| value * value).sum::<f64>();
        let after = pca
            .components
            .chunks_exact(dimension)
            .map(|component| {
                let dot = component
                    .iter()
                    .zip(&centred)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                dot * dot
            })
            .sum::<f64>();
        worst = worst.max((after - before).abs() / before.max(f64::MIN_POSITIVE));
    }
    assert!(worst < 1e-9, "a projection loses norm: {worst:e}");
    println!("pca: norm preserved to {worst:.2e} on a sample of stored vectors");
}

/// A head of `m` components as its own contiguous rows.
fn head_rows(projected: &Projected, m: usize) -> Vec<f32> {
    projected
        .heads
        .chunks_exact(projected.width)
        .flat_map(|row| row[..m].iter().copied())
        .collect()
}

fn outside(values: &[f32], low: f64, high: f64) -> f64 {
    let count = values
        .iter()
        .filter(|value| f64::from(**value) < low || f64::from(**value) > high)
        .count();
    count as f64 / values.len() as f64
}

/// The head at eight bits under Lance's SQ8 math: one bound over every head value of
/// the index's own sample, symmetric integer L2.
struct GlobalCodes {
    store: ScalarQuantizationStorage,
    base_outside: f64,
    query_outside: f64,
}

fn global_codes(
    projected: &Projected,
    queries: &Projected,
    m: usize,
    sample: &[u32],
    row_ids: &[u64],
) -> GlobalCodes {
    let list = FixedSizeListArray::try_new_from_values(
        Float32Array::from(head_rows(projected, m)),
        m as i32,
    )
    .unwrap();
    let bounds = ScalarQuantizer::new(8, m)
        .update_bounds::<Float32Type>(&take_rows(&list, sample))
        .unwrap();
    let codes = ScalarQuantizer::with_bounds(8, m, bounds.clone())
        .transform::<Float32Type>(&list)
        .unwrap();
    GlobalCodes {
        base_outside: outside(flat_values::<Float32Type>(&list), bounds.start, bounds.end),
        query_outside: outside(&head_rows(queries, m), bounds.start, bounds.end),
        store: scalar_store(row_ids, codes, bounds),
    }
}

/// The head at eight bits under per-dimension bounds from the index's own sample,
/// coded by Lance's truncating rule applied per dimension.
struct PerDimensionCodes {
    codes: Vec<u8>,
    low: Vec<f32>,
    step: Vec<f32>,
    base_outside: f64,
    query_outside: f64,
}

fn per_dimension_codes(
    projected: &Projected,
    queries: &Projected,
    m: usize,
    sample: &[u32],
) -> PerDimensionCodes {
    let heads = head_rows(projected, m);
    let mut low = vec![f64::MAX; m];
    let mut high = vec![f64::MIN; m];
    for row in sample {
        for (at, value) in heads[*row as usize * m..][..m].iter().enumerate() {
            low[at] = low[at].min(f64::from(*value));
            high[at] = high[at].max(f64::from(*value));
        }
    }
    let codes = heads
        .chunks_exact(m)
        .flat_map(|row| {
            row.iter().enumerate().map(|(at, value)| {
                if low[at] == high[at] {
                    0
                } else {
                    ((f64::from(*value) - low[at]) * 255.0 / (high[at] - low[at])) as u8
                }
            })
        })
        .collect::<Vec<_>>();
    let count_outside = |values: &[f32]| {
        let count = values
            .chunks_exact(m)
            .flat_map(|row| row.iter().enumerate())
            .filter(|(at, value)| f64::from(**value) < low[*at] || f64::from(**value) > high[*at])
            .count();
        count as f64 / values.len() as f64
    };
    PerDimensionCodes {
        codes,
        low: low.iter().map(|value| *value as f32).collect(),
        step: low
            .iter()
            .zip(&high)
            .map(|(low, high)| ((high - low) / 255.0) as f32)
            .collect(),
        base_outside: count_outside(&heads),
        query_outside: count_outside(&head_rows(queries, m)),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Float,
    Global,
    PerDimension,
}

#[derive(Clone, Copy)]
struct Arm {
    kind: Kind,
    m: usize,
    tail: bool,
}

impl Arm {
    fn name(&self) -> String {
        let kind = match self.kind {
            Kind::Float => "F",
            Kind::Global => "QG",
            Kind::PerDimension => "QD",
        };
        format!("{kind}{}", if self.tail { "" } else { "0" })
    }

    fn bytes(&self) -> usize {
        let head = match self.kind {
            Kind::Float => 4 * self.m,
            Kind::Global | Kind::PerDimension => self.m,
        };
        head + if self.tail { 4 } else { 0 }
    }
}

/// Sums over the queries of every tally column at every list size, each query counted
/// `weights[query]` times.
struct Curve {
    sizes: Vec<usize>,
    answer: Vec<f64>,
    coded: Vec<f64>,
    navigated: Vec<f64>,
    walk: Vec<f64>,
    reranked: Vec<f64>,
    queries: f64,
}

impl Curve {
    fn of(sizes: &[usize], tallies: &[Vec<Tally>], weights: &[u32]) -> Self {
        let sum = |column: &dyn Fn(&Tally) -> f64| {
            tallies
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(weights)
                        .map(|(tally, weight)| column(tally) * f64::from(*weight))
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };
        Self {
            sizes: sizes.to_vec(),
            answer: sum(&|tally| f64::from(tally.answer)),
            coded: sum(&|tally| f64::from(tally.coded)),
            navigated: sum(&|tally| f64::from(tally.navigated)),
            walk: sum(&|tally| f64::from(tally.walk)),
            reranked: sum(&|tally| f64::from(tally.reranked)),
            queries: weights.iter().map(|weight| f64::from(*weight)).sum(),
        }
    }

    fn recall(&self, at: usize) -> f64 {
        self.answer[at] / (K as f64 * self.queries)
    }

    /// Where recall first reaches `bar`, every column interpolated with one weight.
    fn cross(&self, bar: f64) -> Crossing {
        let Some(hi) = (0..self.sizes.len()).find(|at| self.recall(*at) >= bar) else {
            let best = (0..self.sizes.len())
                .map(|at| self.recall(at))
                .fold(0.0, f64::max);
            return Crossing::Never { best };
        };
        let recall_mean = |column: &[f64], at: usize| column[at] / (K as f64 * self.queries);
        let mean = |column: &[f64], at: usize| column[at] / self.queries;
        let (lo, weight) = if hi == 0 {
            (0, 0.0)
        } else {
            let (below, above) = (self.recall(hi - 1), self.recall(hi));
            (hi - 1, (bar - below) / (above - below))
        };
        let blend = |value: &dyn Fn(usize) -> f64| value(lo) + weight * (value(hi) - value(lo));
        let dips = (hi + 1..self.sizes.len())
            .filter(|at| self.recall(*at) < bar)
            .count();
        Crossing::At(Point {
            floor: hi == 0,
            bracket: (self.sizes[lo], self.sizes[hi]),
            size: self.sizes[lo] as f64 + weight * (self.sizes[hi] - self.sizes[lo]) as f64,
            walk: blend(&|at| mean(&self.walk, at)),
            reranked: blend(&|at| mean(&self.reranked, at)),
            coded: blend(&|at| recall_mean(&self.coded, at)),
            navigated: blend(&|at| recall_mean(&self.navigated, at)),
            dips,
        })
    }
}

#[derive(Clone, Copy)]
struct Point {
    /// Already at or above the bar at the smallest list size: an upper bound.
    floor: bool,
    bracket: (usize, usize),
    size: f64,
    walk: f64,
    reranked: f64,
    coded: f64,
    navigated: f64,
    /// List sizes past the crossing whose recall falls back under the bar.
    dips: usize,
}

#[derive(Clone, Copy)]
enum Crossing {
    At(Point),
    Never { best: f64 },
}

/// Code bytes a query runs through distance kernels at a crossing.
fn code_bytes(point: &Point, arm: Option<&Arm>, dimension: usize) -> f64 {
    match arm {
        None => point.walk * dimension as f64,
        Some(arm) => point.walk * arm.bytes() as f64 + point.reranked * dimension as f64,
    }
}

/// `(queue ratio, byte ratio)` of an arm against the baseline, when both cross.
fn ratios(base: &Crossing, arm: &Crossing, spec: &Arm, dimension: usize) -> Option<(f64, f64)> {
    match (base, arm) {
        (Crossing::At(base), Crossing::At(point)) => Some((
            point.size / base.size,
            code_bytes(base, None, dimension) / code_bytes(point, Some(spec), dimension),
        )),
        _ => None,
    }
}

fn passes(spec: &Arm, ratios: Option<(f64, f64)>) -> bool {
    spec.kind != Kind::Float
        && ratios.is_some_and(|(queue, bytes)| queue <= QUEUE_LIMIT && bytes >= BYTE_FACTOR)
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * share).round() as usize]
}

/// The round log's walk rows: `(queue, recall, coded recall, mean distances)` as printed.
fn logged_rows(path: &str) -> Vec<(usize, String, String, String)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut rows = Vec::new();
    let mut pending = None;
    for line in text.lines() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if line.starts_with("vamana walk b=20") && words.len() > 5 {
            pending = words[3]
                .parse::<usize>()
                .ok()
                .map(|queue| (queue, words[4].to_string(), words[5].to_string()));
        } else if let Some(rest) =
            line.strip_prefix("# work of the row above: distances a query mean ")
            && let Some((queue, recall, coded)) = pending.take()
        {
            let mean = rest.split(',').next().unwrap().to_string();
            rows.push((queue, recall, coded, mean));
        }
    }
    assert!(!rows.is_empty(), "no walk rows in {path}");
    rows
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
    let a3_dir = env_string("A3_DIR");
    let bar = env_usize("TARGET", 0) as f64 / 100.0;
    assert!(
        bar > 0.0 && bar < 1.0,
        "set TARGET to the recall bar in per cent"
    );
    let threads = env_usize("THREADS", 10);
    let base_path = format!("{data_dir}/{prefix}_base.fvecs");
    let (dimension, rows) = fvecs_shape(&base_path);
    let ms = env_list("MS", "64,128,256")
        .into_iter()
        .filter(|m| *m < dimension)
        .collect::<Vec<_>>();
    assert!(
        ms.iter().all(|m| m % 8 == 0),
        "every m must be a multiple of 8"
    );
    let widest = *ms.iter().max().expect("no m below the dimension");
    let uri = format!("{dataset_dir}/{prefix}-{rows}-p1-r{DEGREE}-sq8.lance");
    println!("# {}", unix_now());
    println!("# git {}", git_head());
    println!("{prefix} {rows} x {dimension}, bar {bar}, ms {ms:?}, index {uri}, threads {threads}");

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
    let mut local_of_position = vec![u32::MAX; rows];
    for (local, row_addr) in built.row_ids.iter().enumerate() {
        let position = positions[row_addr] as usize;
        assert_eq!(
            local_of_position[position],
            u32::MAX,
            "two vertices at {position}"
        );
        local_of_position[position] = local as u32;
    }
    let identity = local_of_position
        .iter()
        .enumerate()
        .all(|(position, local)| *local as usize == position);
    println!(
        "loaded: medoid {}, degree {}, local ids {} position order",
        built.medoid,
        built.width,
        if identity { "are" } else { "are NOT" }
    );

    let stored = flat_values::<Float32Type>(&built.vectors);
    check_against_fvecs(&base_path, stored, dimension, &local_of_position);
    let replayed = ScalarQuantizer::with_bounds(8, dimension, built.bounds.clone())
        .transform::<Float32Type>(&built.vectors)
        .unwrap();
    assert!(
        flat_values::<UInt8Type>(replayed.as_fixed_size_list())
            == flat_values::<UInt8Type>(&built.codes),
        "the stored codes are not Lance's SQ8 of the stored vectors"
    );
    drop(replayed);
    let sample = sample_rows(&local_of_position);
    let resampled = ScalarQuantizer::new(8, dimension)
        .update_bounds::<Float32Type>(&take_rows(&built.vectors, &sample))
        .unwrap();
    assert_eq!(
        resampled, built.bounds,
        "the sample rule does not give the stored bounds"
    );
    println!(
        "premise: base file = stored vectors, SQ8 replay = stored codes, sample rule = stored \
         bounds {:?}",
        built.bounds
    );

    let (query_values, query_dimension, available) =
        read_fvecs(&format!("{data_dir}/{prefix}_query.fvecs"));
    assert_eq!(query_dimension, dimension);
    let count = env_usize("QUERIES", 1000).min(available);
    let query_values = query_values[..count * dimension].to_vec();
    let queries = query_values
        .chunks_exact(dimension)
        .map(|query| Arc::new(Float32Array::from(query.to_vec())) as ArrayRef)
        .collect::<Vec<_>>();
    let query_hash = fnv(query_values.iter().map(|value| u64::from(value.to_bits())));
    let exact = FlatFloatStorage::new(built.vectors.clone(), DistanceType::L2);
    let truth = {
        let ordered = if identity {
            exact.clone()
        } else {
            FlatFloatStorage::new(
                take_rows(&built.vectors, &local_of_position),
                DistanceType::L2,
            )
        };
        ground_truth(
            &ordered,
            &queries,
            query_hash,
            threads,
            &format!("{a3_dir}/a3-gt-{prefix}-q{count}.bin"),
        )
    };
    let sq8 = scalar_store(
        &built.row_ids,
        Arc::new(built.codes.clone()),
        built.bounds.clone(),
    );
    let context = Context {
        graph: Graph {
            edges: &built.edges,
            width: built.width,
            medoid: built.medoid,
            rows,
        },
        row_ids: &built.row_ids,
        positions: &positions,
        truth: &truth,
        queries: &queries,
        sq8: &sq8,
        exact: &exact,
        threads,
    };
    println!("{count} queries, query fnv {query_hash:016x}");

    // Gates V1-V3: the harness's walk is production's before any arm is read.
    let logged = logged_rows(&format!("{a3_dir}/{prefix}-aduperf-fixed-r1.log"));
    let mut gate_sizes = logged.iter().map(|row| row.0).collect::<Vec<_>>();
    gate_sizes.push(SMALLEST_LIST);
    gate_sizes.sort_unstable();
    gate_sizes.dedup();
    let base_nav = |query: usize| sq8.dist_calculator(queries[query].clone(), 0.0);
    let ours = context.run(&gate_sizes, false, &base_nav, &|_, outcome: &Outcome| {
        outcome.clone()
    });
    {
        let index = VamanaIndex::open(&dataset, INDEX_NAME)
            .await
            .unwrap()
            .with_cache(LanceCache::with_capacity(4 << 30));
        for (size, outcomes) in gate_sizes.iter().zip(&ours) {
            let params = SearchParams::new(K)
                .with_nprobes(1)
                .with_search_list_size(*size)
                .with_mode(WalkMode::Lazy)
                .with_start(WalkStart::Medoid)
                .with_beam_width(BEAM)
                .with_prefetch_ahead(PREFETCH)
                .with_resident_edges(true)
                .with_rescore_budget(BUDGET)
                .with_report_coded(true);
            for (query, outcome) in outcomes.iter().enumerate() {
                let theirs = index
                    .search(&query_values[query * dimension..][..dimension], &params)
                    .await
                    .unwrap();
                assert!(
                    same_neighbors(&outcome.answer, &theirs.neighbors),
                    "V1: query {query} at L = {size} answers differently from the index"
                );
                assert!(
                    same_neighbors(&outcome.coded, &theirs.coded_neighbors),
                    "V1: query {query} at L = {size} has other coded neighbours than the index"
                );
                assert_eq!(
                    1 + u64::from(outcome.walk) + u64::from(outcome.kept),
                    theirs.comparisons,
                    "V1: query {query} at L = {size} counts other comparisons than the index"
                );
            }
        }
    }
    println!(
        "V1: the harness equals VamanaIndex::search bit for bit, {count} queries at L = {gate_sizes:?}"
    );
    let logged_context = context.with_queries(LOGGED_QUERIES);
    for (queue, recall, coded, mean) in &logged {
        let at = gate_sizes.iter().position(|size| size == queue).unwrap();
        let outcomes = &ours[at][..LOGGED_QUERIES];
        let hits = |pick: &dyn Fn(&Outcome) -> &[Neighbor]| {
            outcomes
                .iter()
                .enumerate()
                .map(|(query, outcome)| u64::from(logged_context.hits(pick(outcome), query)))
                .sum::<u64>() as f64
                / (K * LOGGED_QUERIES) as f64
        };
        let ours_recall = format!("{:.4}", hits(&|outcome| outcome.answer.as_slice()));
        let ours_coded = format!("{:.4}", hits(&|outcome| outcome.coded.as_slice()));
        let distances = outcomes
            .iter()
            .map(|outcome| 1 + u64::from(outcome.walk) + u64::from(outcome.kept))
            .sum::<u64>() as f64
            / LOGGED_QUERIES as f64;
        let ours_mean = format!("{distances:.3}");
        assert_eq!(
            (&ours_recall, &ours_coded, &ours_mean),
            (recall, coded, mean),
            "V2: L = {queue} does not reproduce the round log"
        );
    }
    println!(
        "V2: the first {LOGGED_QUERIES} queries reproduce the round log to the digit at {} rows",
        logged.len()
    );
    let zeros = vec![0f32; rows];
    let plumbing_nav = |query: usize| TwoLevel {
        head: sq8.dist_calculator(queries[query].clone(), 0.0),
        tails: Some((zeros.as_slice(), 0.0)),
    };
    let plumbed = context.run(&gate_sizes, true, &plumbing_nav, &|_, outcome: &Outcome| {
        outcome.clone()
    });
    for (size, (theirs, mine)) in gate_sizes.iter().zip(ours.iter().zip(&plumbed)) {
        for (query, (theirs, mine)) in theirs.iter().zip(mine).enumerate() {
            assert!(
                same_neighbors(&mine.answer, &theirs.answer)
                    && same_neighbors(&mine.coded, &theirs.coded)
                    && mine.walk == theirs.walk,
                "V3: the two-level path changes query {query} at L = {size}"
            );
        }
    }
    println!("V3: the two-level path with the SQ8 head and zero tails equals the baseline");
    drop((ours, plumbed));

    let pca = load_pca(&format!("{a3_dir}/a3-pca-{prefix}"), dimension);
    let projected = project(stored, dimension, &pca, widest, &ms, threads);
    let query_projected = project(&query_values, dimension, &pca, widest, &ms, threads);
    check_norms(stored, dimension, &pca);
    let total = pca.eigenvalues.iter().sum::<f64>();
    for (at_m, m) in ms.iter().enumerate() {
        let expected = pca.eigenvalues[..*m].iter().sum::<f64>() / total;
        let measured = projected.head_energy[at_m] / projected.energy;
        assert!(
            (measured - expected).abs() < 1e-7,
            "m = {m}: the heads hold {measured} of the energy, the eigenvalues {expected}"
        );
        println!(
            "pca: m = {m} holds {:.4}% of the variance (eigenvalues {:.4}%)",
            100.0 * measured,
            100.0 * expected
        );
    }
    let globals = ms
        .iter()
        .map(|m| global_codes(&projected, &query_projected, *m, &sample, &built.row_ids))
        .collect::<Vec<_>>();
    let per_dimensions = ms
        .iter()
        .map(|m| per_dimension_codes(&projected, &query_projected, *m, &sample))
        .collect::<Vec<_>>();
    for (at_m, m) in ms.iter().enumerate() {
        println!(
            "codes m = {m}: QG outside its bound {:.4}% of base values, {:.4}% of query values; \
             QD {:.4}% and {:.4}%",
            100.0 * globals[at_m].base_outside,
            100.0 * globals[at_m].query_outside,
            100.0 * per_dimensions[at_m].base_outside,
            100.0 * per_dimensions[at_m].query_outside
        );
    }

    // The baseline first, far enough to find its crossing and look past it.
    let weights = vec![1u32; count];
    let mut base_sizes = Vec::new();
    let mut base_tallies = Vec::new();
    let base_point = loop {
        let from = base_sizes.last().map_or(SMALLEST_LIST, |last| last + 1);
        let next = (from..from + 100).collect::<Vec<_>>();
        base_tallies.extend(context.run(&next, false, &base_nav, &|query, outcome| {
            context.tally(query, outcome)
        }));
        base_sizes.extend(next);
        if let Crossing::At(point) = Curve::of(&base_sizes, &base_tallies, &weights).cross(bar)
            && *base_sizes.last().unwrap() >= point.bracket.1 + 20
        {
            break point;
        }
        assert!(base_sizes.len() < 2000, "the baseline never reaches {bar}");
    };
    let arm_sizes = (SMALLEST_LIST..=(2.0 * base_point.size).ceil() as usize).collect::<Vec<_>>();
    println!(
        "baseline crosses {bar} at L = {:.2}; arms walk L = {}..={}",
        base_point.size,
        arm_sizes[0],
        arm_sizes.last().unwrap()
    );

    let mut arms = Vec::new();
    for (at_m, m) in ms.iter().enumerate() {
        for kind in [Kind::Float, Kind::Global, Kind::PerDimension] {
            for tail in [true, false] {
                let spec = Arm { kind, m: *m, tail };
                let tails = |query: usize| {
                    tail.then(|| {
                        (
                            projected.tails[at_m].as_slice(),
                            query_projected.tails[at_m][query],
                        )
                    })
                };
                let query_head =
                    |query: usize| query_projected.heads[query * widest..][..*m].to_vec();
                let keep = |query: usize, outcome: &Outcome| context.tally(query, outcome);
                let arm_started = Instant::now();
                let tallies = match kind {
                    Kind::Float => context.run(
                        &arm_sizes,
                        true,
                        &|query| TwoLevel {
                            head: FloatHead {
                                heads: &projected.heads,
                                stride: widest,
                                query: query_head(query),
                            },
                            tails: tails(query),
                        },
                        &keep,
                    ),
                    Kind::Global => context.run(
                        &arm_sizes,
                        true,
                        &|query| TwoLevel {
                            head: globals[at_m].store.dist_calculator(
                                Arc::new(Float32Array::from(query_head(query))) as ArrayRef,
                                0.0,
                            ),
                            tails: tails(query),
                        },
                        &keep,
                    ),
                    Kind::PerDimension => {
                        let codes = &per_dimensions[at_m];
                        context.run(
                            &arm_sizes,
                            true,
                            &|query| TwoLevel {
                                head: PerDimensionHead {
                                    codes: &codes.codes,
                                    step: &codes.step,
                                    shifted: query_head(query)
                                        .iter()
                                        .zip(codes.low.iter().zip(&codes.step))
                                        .map(|(value, (low, step))| value - low - 0.5 * step)
                                        .collect(),
                                },
                                tails: tails(query),
                            },
                            &keep,
                        )
                    }
                };
                println!(
                    "arm {} m = {m} walked in {:.1}s",
                    spec.name(),
                    arm_started.elapsed().as_secs_f64()
                );
                arms.push((spec, tallies));
            }
        }
    }

    // The table: every arm at the bar against the baseline on the same queries.
    let base_curve = Curve::of(&base_sizes, &base_tallies, &weights);
    let base_cross = base_curve.cross(bar);
    let base_bytes = code_bytes(&base_point, None, dimension);
    println!(
        "\n{prefix}: {count} queries, bar {bar}, baseline L* {:.2} [{}, {}], walk {:.1}, \
         {:.0} code bytes a query, coded recall {:.4}, dips past it {}",
        base_point.size,
        base_point.bracket.0,
        base_point.bracket.1,
        base_point.walk,
        base_bytes,
        base_point.coded,
        base_point.dips
    );
    println!(
        "{:<4} {:>4} {:>6} {:>8} {:>11} {:>9} {:>8} {:>11} {:>7} {:>7} {:>6} {:>6} {:>7} \
         {:>7} {:>7} {:>5}",
        "arm",
        "m",
        "B/vtx",
        "L*",
        "bracket",
        "walk",
        "rerank",
        "bytes/q",
        "L x",
        "bytes /",
        "L<=1.25",
        "B>=3",
        "verdict",
        "coded",
        "nav",
        "dips"
    );
    let mut csv = String::from("arm,m,tail,L,recall,coded,navigated,walk,reranked\n");
    for (at, size) in base_sizes.iter().enumerate() {
        csv.push_str(&format!(
            "base,{dimension},0,{size},{:.6},{:.6},{:.6},{:.3},{:.3}\n",
            base_curve.recall(at),
            base_curve.coded[at] / (K * count) as f64,
            base_curve.navigated[at] / (K * count) as f64,
            base_curve.walk[at] / count as f64,
            base_curve.reranked[at] / count as f64
        ));
    }
    let mut crossings = Vec::new();
    for (spec, tallies) in &arms {
        let curve = Curve::of(&arm_sizes, tallies, &weights);
        for (at, size) in arm_sizes.iter().enumerate() {
            csv.push_str(&format!(
                "{},{},{},{size},{:.6},{:.6},{:.6},{:.3},{:.3}\n",
                spec.name(),
                spec.m,
                u8::from(spec.tail),
                curve.recall(at),
                curve.coded[at] / (K * count) as f64,
                curve.navigated[at] / (K * count) as f64,
                curve.walk[at] / count as f64,
                curve.reranked[at] / count as f64
            ));
        }
        let cross = curve.cross(bar);
        let ratio = ratios(&base_cross, &cross, spec, dimension);
        crossings.push(cross);
        match cross {
            Crossing::Never { best } => println!(
                "{:<4} {:>4} {:>6} never reaches {bar} by L = {} (best {best:.4})",
                spec.name(),
                spec.m,
                spec.bytes(),
                arm_sizes.last().unwrap()
            ),
            Crossing::At(point) => {
                let (queue, bytes) = ratio.unwrap();
                println!(
                    "{:<4} {:>4} {:>6} {:>7.2}{} {:>11} {:>9.1} {:>8.1} {:>11.0} {:>7.3} {:>7.2} \
                     {:>6} {:>6} {:>7} {:>7.4} {:>7.4} {:>5}",
                    spec.name(),
                    spec.m,
                    spec.bytes(),
                    point.size,
                    if point.floor { "<" } else { " " },
                    format!("[{}, {}]", point.bracket.0, point.bracket.1),
                    point.walk,
                    point.reranked,
                    code_bytes(&point, Some(spec), dimension),
                    queue,
                    bytes,
                    if queue <= QUEUE_LIMIT { "yes" } else { "no" },
                    if bytes >= BYTE_FACTOR { "yes" } else { "no" },
                    match (spec.kind, passes(spec, ratio)) {
                        (Kind::Float, _) => "control",
                        (_, true) => "PASS",
                        (_, false) => "fail",
                    },
                    point.coded,
                    point.navigated,
                    point.dips
                );
            }
        }
    }
    let csv_path = format!("{a3_dir}/a3-{prefix}-curves.csv");
    std::fs::write(&csv_path, csv).unwrap();

    // Query resampling: how often each arm's verdict survives another draw of queries.
    let resampled = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                let (arms, base_sizes, base_tallies, arm_sizes) =
                    (&arms, &base_sizes, &base_tallies, &arm_sizes);
                scope.spawn(move || {
                    (thread..RESAMPLES)
                        .step_by(threads)
                        .map(|resample| {
                            let mut rng = SmallRng::seed_from_u64(SEED + resample as u64);
                            let mut weights = vec![0u32; count];
                            for _ in 0..count {
                                weights[rng.random_range(0..count)] += 1;
                            }
                            let base = Curve::of(base_sizes, base_tallies, &weights).cross(bar);
                            arms.iter()
                                .map(|(spec, tallies)| {
                                    let arm = Curve::of(arm_sizes, tallies, &weights).cross(bar);
                                    ratios(&base, &arm, spec, dimension)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    println!("\nresampled queries ({RESAMPLES} draws, seed {SEED}): 90% intervals and pass share");
    for (at, (spec, _)) in arms.iter().enumerate() {
        let mut queue = Vec::new();
        let mut bytes = Vec::new();
        let mut passed = 0;
        for draw in &resampled {
            let (q, b) = draw[at].unwrap_or((f64::INFINITY, 0.0));
            queue.push(q);
            bytes.push(b);
            passed += usize::from(passes(spec, draw[at]));
        }
        queue.sort_by(f64::total_cmp);
        bytes.sort_by(f64::total_cmp);
        println!(
            "{:<4} {:>4}  L x [{:.3}, {:.3}]  bytes / [{:.2}, {:.2}]  passes in {:.1}%",
            spec.name(),
            spec.m,
            percentile(&queue, 0.05),
            percentile(&queue, 0.95),
            percentile(&bytes, 0.05),
            percentile(&bytes, 0.95),
            100.0 * passed as f64 / RESAMPLES as f64
        );
    }
    let verdict = arms
        .iter()
        .zip(&crossings)
        .filter(|((spec, _), cross)| passes(spec, ratios(&base_cross, cross, spec, dimension)))
        .map(|((spec, _), _)| format!("{} m = {}", spec.name(), spec.m))
        .collect::<Vec<_>>();
    println!(
        "\nverdict {prefix}: {} (curves in {csv_path}; {:.0}s)",
        if verdict.is_empty() {
            "no quantized arm passes".to_string()
        } else {
            format!("PASS by {}", verdict.join(", "))
        },
        started.elapsed().as_secs_f64()
    );
}

/// Seconds since the epoch, for the log header, without a date crate.
fn unix_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    format!("unix time {}", now.as_secs())
}
