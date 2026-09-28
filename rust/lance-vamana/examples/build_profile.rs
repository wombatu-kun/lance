// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Where a Vamana build spends its time, and what the incremental re-prune saves.
//!
//! Replays `build_partition` over a built index's own vectors, in the index's own
//! local-id order and at the parameters its metadata records, with a timer and a
//! distance counter on each phase of every insertion: the search of the graph as it
//! stands, the point's own prune, and the back-edges its choice sends out. The loop
//! is copied, with the crate-private pieces it calls, and each arm is trusted only
//! after it builds the same graph as `build_partition` does; the incremental arm,
//! the build as it is, must count the same distances too.
//!
//! Two arms. `incremental` is the build as it is: a back-edge into a full list whose
//! members are still exactly what a prune returned checks only the newcomer against
//! them. `build.rs` does that with `prune` and `admit`, which are crate-private, so
//! they are copied here as `select` and `reprune_incremental`. `full` is the build
//! before that, re-pruning every full list through the library's `robust_prune`.
//! Both must build the same graph, byte for byte, and the gates below hold them to
//! that.
//!
//! ```text
//! cd rust/lance-vamana
//! cargo build --profile release-no-lto --example build_profile
//! INDEX=~/vamana-runs/sift-1000000-p1-r70-sq8.lance STRIDE=10 REPRUNE=both OUT=profile.json \
//!     target/release-no-lto/examples/build_profile
//! ```
//!
//! `STRIDE` (default 1) builds over every `STRIDE`-th vertex; at 1 every arm's graph
//! and medoid must also equal the stored ones. `REPRUNE` is `incremental` (default),
//! `full`, `both` or `both-reversed` (both arms, in that order, whose graphs must be
//! identical), or `none` for the gate alone. `GATE_STRIDE` (default 100) is the
//! subset on which every arm is first checked against `build_partition`,
//! `GATE_ROUNDS` (default 2) times; at 1 `build_partition` must also build the
//! stored graph. `BUCKETS` (default 20) slices each pass into a timeline. `OUT` receives
//! every total, the timelines and each pass's offsets from the start of the
//! process, which is what lines them up with a `perf stat -I` log of the same run.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, UInt32Array};
use lance::Dataset;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use lance_vamana::PartitionGraph;
use lance_vamana::build::{BuildParams, build_partition, medoid, robust_prune};
use lance_vamana::format::{
    INDEX_FILE_NAME, IndexMetadata, NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, VECTOR_COLUMN,
};
use lance_vamana::io::{PartitionFile, read_partition_batch, read_segment, scan_scheduler};
use lance_vamana::query::committed_segments;
use lance_vamana::search::{Comparisons, SearchScratch, flat_storage, greedy_search};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use serde_json::json;

#[path = "common/mod.rs"]
mod common;
use common::env_usize;

const INDEX_NAME: &str = "vamana_idx";

fn env_string(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
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

/// One segment's partition as the build that wrote it saw its input.
struct Stored {
    metadata: IndexMetadata,
    medoid: u32,
    row_ids: Vec<u64>,
    vectors: FixedSizeListArray,
    width: usize,
    edges: Option<Vec<u32>>,
}

async fn load(dataset: &Dataset, with_edges: bool) -> Stored {
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
    let metadata = manifest.metadata().clone();
    assert_eq!(metadata.distance_type, DistanceType::L2);
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
    let columns: &[&str] = if with_edges {
        &[ROW_ID_COLUMN, NEIGHBORS_COLUMN]
    } else {
        &[ROW_ID_COLUMN]
    };
    let reader = file.project(columns).await.unwrap();
    let batch = read_partition_batch(&reader, entry.num_rows).await.unwrap();
    let row_ids = batch[ROW_ID_COLUMN]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    let (width, edges) = if with_edges {
        let neighbors = batch[NEIGHBORS_COLUMN].as_fixed_size_list();
        let width = neighbors.value_length() as usize;
        let slots = neighbors.values().as_primitive::<UInt32Type>().values();
        let slots = slots[neighbors.offset() * width..][..neighbors.len() * width].to_vec();
        (width, Some(slots))
    } else {
        (metadata.max_degree as usize, None)
    };
    drop(batch);
    let reader = file.project(&[VECTOR_COLUMN]).await.unwrap();
    let batch = read_partition_batch(&reader, entry.num_rows).await.unwrap();
    let vectors = batch[VECTOR_COLUMN].as_fixed_size_list().clone();
    assert_eq!(vectors.len(), row_ids.len());
    assert_eq!(vectors.value_length() as u32, metadata.dimension);
    Stored {
        metadata,
        medoid: entry.medoid,
        row_ids,
        vectors,
        width,
        edges,
    }
}

/// The flat store over every `stride`-th stored vertex, in local-id order.
fn store_of(stored: &Stored, stride: usize) -> FlatFloatStorage {
    if stride == 1 {
        return flat_storage(&stored.row_ids, &stored.vectors, DistanceType::L2).unwrap();
    }
    let rows = (0..stored.row_ids.len() as u32)
        .step_by(stride)
        .collect::<Vec<_>>();
    let vectors = arrow_select::take::take(&stored.vectors, &UInt32Array::from(rows.clone()), None)
        .unwrap()
        .as_fixed_size_list()
        .clone();
    let row_ids = rows
        .iter()
        .map(|row| stored.row_ids[*row as usize])
        .collect::<Vec<_>>();
    flat_storage(&row_ids, &vectors, DistanceType::L2).unwrap()
}

/// How a back-edge into a full list is settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Reprune {
    /// `robust_prune` over the list and the newcomer: the build before `admit`.
    Full,
    /// Only the newcomer checked, when the list is still a prune's output, as
    /// `build_partition` does.
    Incremental,
}

/// What one slice of a pass cost, phase by phase.
#[derive(Debug, Default, Clone, Serialize)]
struct Tally {
    insertions: u64,
    search_nanos: u64,
    own_nanos: u64,
    back_nanos: u64,
    search_distances: u64,
    own_distances: u64,
    back_distances: u64,
    /// Vertices the search expanded: its visited list, the prune's main input.
    expansions: u64,
    /// The visited list plus the point's current out-edges, as the prune takes them.
    own_candidates: u64,
    selected: u64,
    /// A chosen neighbour that already pointed back.
    back_present: u64,
    /// A back-edge that went into a free slot.
    back_added: u64,
    /// A back-edge that found the list full and made it re-prune.
    back_pruned: u64,
    /// Of those, settled by the incremental check.
    reprune_fast: u64,
    /// Of the fast ones, the list left as it was: the newcomer occluded, or farther
    /// than every member.
    reprune_unchanged: u64,
    /// A stable list whose check met a zero separation and went to the full prune.
    reprune_fallback: u64,
    /// A list that was not a prune's output: appended to, filled by the coincident
    /// fill, or never pruned.
    reprune_unstable: u64,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.insertions += other.insertions;
        self.search_nanos += other.search_nanos;
        self.own_nanos += other.own_nanos;
        self.back_nanos += other.back_nanos;
        self.search_distances += other.search_distances;
        self.own_distances += other.own_distances;
        self.back_distances += other.back_distances;
        self.expansions += other.expansions;
        self.own_candidates += other.own_candidates;
        self.selected += other.selected;
        self.back_present += other.back_present;
        self.back_added += other.back_added;
        self.back_pruned += other.back_pruned;
        self.reprune_fast += other.reprune_fast;
        self.reprune_unchanged += other.reprune_unchanged;
        self.reprune_fallback += other.reprune_fallback;
        self.reprune_unstable += other.reprune_unstable;
    }

    fn distances(&self) -> u64 {
        self.search_distances + self.own_distances + self.back_distances
    }
}

#[derive(Debug, Serialize)]
struct Pass {
    alpha: f32,
    /// Seconds from the start of the process.
    start: f64,
    end: f64,
    total: Tally,
    timeline: Vec<Tally>,
}

struct Profiled {
    arm: Reprune,
    graph: PartitionGraph,
    medoid: u32,
    randomize_nanos: u64,
    medoid_nanos: u64,
    medoid_distances: u64,
    passes: Vec<Pass>,
}

impl Profiled {
    fn distances(&self) -> u64 {
        self.medoid_distances
            + self
                .passes
                .iter()
                .map(|pass| pass.total.distances())
                .sum::<u64>()
    }
}

/// `build.rs::randomize`, verbatim.
fn randomize(graph: &mut PartitionGraph, rng: &mut SmallRng) {
    let num_vertices = graph.len();
    let degree = (graph.max_degree() as usize).min(num_vertices.saturating_sub(1));
    if degree == 0 {
        return;
    }
    let mut neighbors = Vec::with_capacity(degree);
    for point in 0..num_vertices as u32 {
        neighbors.clear();
        neighbors.extend(
            rand::seq::index::sample(rng, num_vertices - 1, degree)
                .into_iter()
                .map(|drawn| {
                    let drawn = drawn as u32;
                    if drawn >= point { drawn + 1 } else { drawn }
                }),
        );
        graph.set_neighbors(point, &neighbors).unwrap();
    }
}

/// `build.rs::prune` with its refusals of bad candidates as asserts: `robust_prune`,
/// returning as well whether the coincident fill supplied part of the list. Such a
/// list is not a prune's output in the sense the incremental check relies on,
/// because a filled member was occluded.
fn select(
    store: &FlatFloatStorage,
    point: u32,
    candidates: Vec<OrderedNode>,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> (Vec<u32>, bool) {
    let store_len = store.len();
    for candidate in &candidates {
        assert!((candidate.id as usize) < store_len && candidate.dist.0.is_finite());
    }
    assert!((point as usize) < store_len && max_degree > 0);

    let mut pool = candidates;
    pool.retain(|candidate| candidate.id != point);
    for candidate in &mut pool {
        candidate.dist = OrderedFloat(candidate.dist.0.max(0.0));
    }
    pool.sort_unstable_by(|a, b| a.id.cmp(&b.id).then(a.dist.cmp(&b.dist)));
    pool.dedup_by_key(|candidate| candidate.id);
    pool.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
    let mut pool = VecDeque::from(pool);

    let mut selected = Vec::with_capacity(max_degree);
    let mut coincident: Vec<OrderedNode> = Vec::new();
    while let Some(nearest) = pool.pop_front() {
        selected.push(nearest.id);
        if selected.len() == max_degree {
            break;
        }
        let from_nearest = store.dist_calculator_from_id(nearest.id);
        comparisons.record(pool.len() as u64);
        pool.retain(|candidate| {
            let separation = from_nearest.distance(candidate.id).max(0.0);
            if alpha * separation > candidate.dist.0 {
                return true;
            }
            if separation == 0.0 {
                coincident.push(candidate.clone());
            }
            false
        });
    }

    let mut filled = false;
    if selected.len() < max_degree {
        coincident.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        for candidate in coincident {
            if selected.len() == max_degree {
                break;
            }
            selected.push(candidate.id);
            filled = true;
        }
    }
    (selected, filled)
}

/// What the incremental check decided.
enum Checked {
    /// A closer member occludes the newcomer, or it is farther than every member: the
    /// prune would return the list as it is.
    Unchanged,
    /// The list the prune would return.
    Pruned(Vec<u32>),
    /// A zero separation, left to the full prune as `admit` leaves it.
    Fallback,
}

/// `build.rs::admit`: `robust_prune(members + newcomer)` for a full list that is itself
/// a prune's output.
///
/// Such a list is sorted by (distance to the owner, id) and no member is occluded by
/// a closer one, at the alpha it was pruned with and so at any larger one. The prune
/// picks candidates nearest first and drops whatever a pick occludes, so the members
/// before the newcomer are picked again unchanged, the newcomer survives only if none
/// of them occludes it, and a member after it drops only if the newcomer does. The
/// list stops at `max_degree`, as the prune's own loop does.
fn reprune_incremental(
    store: &FlatFloatStorage,
    owner: u32,
    members: &[u32],
    newcomer: u32,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Checked {
    let from_owner = store.dist_calculator_from_id(owner);
    comparisons.record(members.len() as u64 + 1);
    let distances = members
        .iter()
        .map(|member| from_owner.distance(*member).max(0.0))
        .collect::<Vec<_>>();
    let newcomer_distance = from_owner.distance(newcomer).max(0.0);
    let key = |distance: f32, id: u32| (OrderedFloat(distance), id);
    assert!(
        distances
            .windows(2)
            .zip(members.windows(2))
            .all(|(pair, ids)| key(pair[0], ids[0]) < key(pair[1], ids[1])),
        "vertex {owner}: a list marked stable is not in prune order"
    );

    let at = members
        .iter()
        .zip(&distances)
        .position(|(member, distance)| key(*distance, *member) > key(newcomer_distance, newcomer))
        .unwrap_or(members.len());
    if at >= max_degree {
        return Checked::Unchanged;
    }
    for member in &members[..at] {
        comparisons.record(1);
        let separation = store
            .dist_calculator_from_id(*member)
            .distance(newcomer)
            .max(0.0);
        if alpha * separation > newcomer_distance {
            continue;
        }
        return if separation == 0.0 {
            Checked::Fallback
        } else {
            Checked::Unchanged
        };
    }

    let mut kept = Vec::with_capacity(max_degree);
    kept.extend_from_slice(&members[..at]);
    kept.push(newcomer);
    if kept.len() < max_degree {
        let from_newcomer = store.dist_calculator_from_id(newcomer);
        for (member, distance) in members[at..].iter().zip(&distances[at..]) {
            comparisons.record(1);
            let separation = from_newcomer.distance(*member).max(0.0);
            if alpha * separation > *distance {
                kept.push(*member);
                if kept.len() == max_degree {
                    break;
                }
            } else if separation == 0.0 {
                return Checked::Fallback;
            }
        }
    }
    Checked::Pruned(kept)
}

/// `insert.rs::insert_point`, with a timer and a counter around each phase.
#[allow(clippy::too_many_arguments)]
fn insert(
    graph: &mut PartitionGraph,
    store: &FlatFloatStorage,
    scratch: &mut SearchScratch,
    edges: &mut Vec<u32>,
    stable: &mut [bool],
    arm: Reprune,
    alpha: f32,
    search_list_size: usize,
    point: u32,
    entry_point: u32,
    tally: &mut Tally,
) {
    let max_degree = graph.max_degree() as usize;
    let searching = Comparisons::default();
    let pruning = Comparisons::default();
    let linking = Comparisons::default();

    let searched = Instant::now();
    let from_point = store.dist_calculator_from_id(point);
    let mut candidates = greedy_search(
        graph,
        &from_point,
        entry_point,
        search_list_size,
        scratch,
        &searching,
    )
    .unwrap()
    .visited;
    let pruned = Instant::now();
    tally.expansions += candidates.len() as u64;
    pruning.record(graph.neighbors(point).unwrap().len() as u64);
    candidates.extend(
        graph.neighbors(point).unwrap().iter().map(|neighbor| {
            OrderedNode::new(*neighbor, OrderedFloat(from_point.distance(*neighbor)))
        }),
    );
    tally.own_candidates += candidates.len() as u64;
    let selected = match arm {
        Reprune::Full => {
            robust_prune(store, point, candidates, alpha, max_degree, &pruning).unwrap()
        }
        Reprune::Incremental => {
            let (selected, filled) = select(store, point, candidates, alpha, max_degree, &pruning);
            stable[point as usize] = !filled;
            selected
        }
    };
    graph.set_neighbors(point, &selected).unwrap();
    let linked = Instant::now();
    tally.selected += selected.len() as u64;

    for neighbor in &selected {
        let neighbor = *neighbor;
        edges.clear();
        edges.extend_from_slice(graph.neighbors(neighbor).unwrap());
        if edges.contains(&point) {
            tally.back_present += 1;
            continue;
        }
        if edges.len() < max_degree {
            edges.push(point);
            graph.set_neighbors(neighbor, edges).unwrap();
            // Appended unchecked: the list stops being a prune's output.
            if arm == Reprune::Incremental {
                stable[neighbor as usize] = false;
            }
            tally.back_added += 1;
            continue;
        }
        tally.back_pruned += 1;
        if arm == Reprune::Incremental {
            if stable[neighbor as usize] {
                match reprune_incremental(
                    store, neighbor, edges, point, alpha, max_degree, &linking,
                ) {
                    Checked::Unchanged => {
                        tally.reprune_fast += 1;
                        tally.reprune_unchanged += 1;
                        continue;
                    }
                    Checked::Pruned(kept) => {
                        graph.set_neighbors(neighbor, &kept).unwrap();
                        tally.reprune_fast += 1;
                        continue;
                    }
                    Checked::Fallback => tally.reprune_fallback += 1,
                }
            } else {
                tally.reprune_unstable += 1;
            }
        }
        let from_neighbor = store.dist_calculator_from_id(neighbor);
        linking.record(edges.len() as u64 + 1);
        let contenders = edges
            .iter()
            .chain(std::iter::once(&point))
            .map(|id| OrderedNode::new(*id, OrderedFloat(from_neighbor.distance(*id))))
            .collect();
        let kept = match arm {
            Reprune::Full => {
                robust_prune(store, neighbor, contenders, alpha, max_degree, &linking).unwrap()
            }
            Reprune::Incremental => {
                let (kept, filled) =
                    select(store, neighbor, contenders, alpha, max_degree, &linking);
                stable[neighbor as usize] = !filled;
                kept
            }
        };
        graph.set_neighbors(neighbor, &kept).unwrap();
    }
    let done = Instant::now();

    tally.insertions += 1;
    tally.search_nanos += (pruned - searched).as_nanos() as u64;
    tally.own_nanos += (linked - pruned).as_nanos() as u64;
    tally.back_nanos += (done - linked).as_nanos() as u64;
    tally.search_distances += searching.get();
    tally.own_distances += pruning.get();
    tally.back_distances += linking.get();
}

/// `build.rs::build_partition`, one timed insertion at a time.
fn profiled_build(
    store: &FlatFloatStorage,
    params: &BuildParams,
    arm: Reprune,
    buckets: usize,
    origin: Instant,
    verbose: bool,
) -> Profiled {
    let num_vertices = u32::try_from(store.len()).unwrap();
    let mut rng = SmallRng::seed_from_u64(params.seed);
    let row_ids = (0..num_vertices)
        .map(|id| store.row_id(id))
        .collect::<Vec<_>>();
    let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids).unwrap();
    let clock = Instant::now();
    randomize(&mut graph, &mut rng);
    let randomize_nanos = clock.elapsed().as_nanos() as u64;
    let counted = Comparisons::default();
    let clock = Instant::now();
    let medoid = medoid(store, &counted).unwrap();
    let medoid_nanos = clock.elapsed().as_nanos() as u64;

    let mut scratch = SearchScratch::new(num_vertices as usize);
    let mut edges = Vec::with_capacity(params.max_degree as usize + 1);
    // Random lists are nobody's prune output.
    let mut stable = vec![false; num_vertices as usize];
    let mut order = (0..num_vertices).collect::<Vec<_>>();
    let mut passes = Vec::with_capacity(2);
    for alpha in [1.0, params.alpha] {
        order.shuffle(&mut rng);
        let start = origin.elapsed().as_secs_f64();
        let mut timeline = vec![Tally::default(); buckets];
        let mut current = 0;
        for (at, point) in order.iter().enumerate() {
            let bucket = at * buckets / order.len();
            if verbose && bucket != current {
                println!(
                    "  {arm:?} pass {} slice {current:>2}/{buckets} done at {:.1} s",
                    passes.len() + 1,
                    origin.elapsed().as_secs_f64()
                );
                std::io::stdout().flush().unwrap();
                current = bucket;
            }
            insert(
                &mut graph,
                store,
                &mut scratch,
                &mut edges,
                &mut stable,
                arm,
                alpha,
                params.search_list_size,
                *point,
                medoid,
                &mut timeline[bucket],
            );
        }
        let end = origin.elapsed().as_secs_f64();
        if verbose {
            println!(
                "  {arm:?} pass {} done at {end:.1} s ({:.1} s)",
                passes.len() + 1,
                end - start
            );
        }
        let mut total = Tally::default();
        for slice in &timeline {
            total.add(slice);
        }
        passes.push(Pass {
            alpha,
            start,
            end,
            total,
            timeline,
        });
    }
    Profiled {
        arm,
        graph,
        medoid,
        randomize_nanos,
        medoid_nanos,
        medoid_distances: counted.get(),
        passes,
    }
}

/// Vertices whose out-edges differ from the stored index's, for a graph built over
/// the whole partition; `None` for one built over a stride, or without the edges.
fn differing_from_stored(stored: &Stored, graph: &PartitionGraph, stride: usize) -> Option<usize> {
    let edges = stored.edges.as_ref().filter(|_| stride == 1)?;
    let width = stored.width;
    assert_eq!(graph.len() * width, edges.len());
    Some(
        (0..graph.len())
            .filter(|vertex| {
                let slots = &edges[vertex * width..][..width];
                let degree = slots
                    .iter()
                    .position(|slot| *slot == NO_NEIGHBOR)
                    .unwrap_or(width);
                &slots[..degree] != graph.neighbors(*vertex as u32).unwrap()
            })
            .count(),
    )
}

/// Vertices whose out-edges differ between two graphs over the same vertices.
fn differing(left: &PartitionGraph, right: &PartitionGraph) -> usize {
    assert_eq!(left.len(), right.len());
    (0..left.len() as u32)
        .filter(|vertex| left.neighbors(*vertex).unwrap() != right.neighbors(*vertex).unwrap())
        .count()
}

fn seconds(nanos: u64) -> f64 {
    nanos as f64 / 1e9
}

fn print_passes(profiled: &Profiled) {
    println!("  arm {:?}", profiled.arm);
    println!(
        "  {:<5} {:>6} {:>9} {:>7} {:>7} {:>7} {:>13} {:>13} {:>13} {:>7} {:>7} {:>7}",
        "pass",
        "alpha",
        "seconds",
        "search",
        "own",
        "back",
        "search dist",
        "own dist",
        "back dist",
        "ns/s",
        "ns/o",
        "ns/b"
    );
    let mut all = Tally::default();
    let rows = profiled
        .passes
        .iter()
        .enumerate()
        .map(|(at, pass)| {
            (
                format!("{}", at + 1),
                format!("{:.2}", pass.alpha),
                &pass.total,
            )
        })
        .collect::<Vec<_>>();
    for (_, _, total) in &rows {
        all.add(total);
    }
    for (name, alpha, tally) in rows
        .iter()
        .map(|(name, alpha, tally)| (name.as_str(), alpha.as_str(), *tally))
        .chain(std::iter::once(("all", "", &all)))
    {
        let spent = tally.search_nanos + tally.own_nanos + tally.back_nanos;
        let share = |nanos: u64| 100.0 * nanos as f64 / spent as f64;
        let per = |nanos: u64, count: u64| nanos as f64 / count.max(1) as f64;
        println!(
            "  {name:<5} {alpha:>6} {:>9.1} {:>6.1}% {:>6.1}% {:>6.1}% {:>13} {:>13} {:>13} {:>7.1} {:>7.1} {:>7.1}",
            seconds(spent),
            share(tally.search_nanos),
            share(tally.own_nanos),
            share(tally.back_nanos),
            tally.search_distances,
            tally.own_distances,
            tally.back_distances,
            per(tally.search_nanos, tally.search_distances),
            per(tally.own_nanos, tally.own_distances),
            per(tally.back_nanos, tally.back_distances),
        );
    }
    for (at, pass) in profiled.passes.iter().enumerate() {
        let tally = &pass.total;
        let per = |count: u64| count as f64 / tally.insertions.max(1) as f64;
        println!(
            "  pass {} per insertion: {:.1} expanded, {:.1} to the prune, {:.1} kept; back-edges \
             {:.2} present, {:.2} added, {:.3} re-pruned ({:.3} fast of which {:.3} unchanged, \
             {:.3} fallback, {:.3} unstable)",
            at + 1,
            per(tally.expansions),
            per(tally.own_candidates),
            per(tally.selected),
            per(tally.back_present),
            per(tally.back_added),
            per(tally.back_pruned),
            per(tally.reprune_fast),
            per(tally.reprune_unchanged),
            per(tally.reprune_fallback),
            per(tally.reprune_unstable),
        );
    }
}

#[tokio::main]
async fn main() {
    let origin = Instant::now();
    let index = env_string("INDEX");
    let out = env_string("OUT");
    let stride = env_usize("STRIDE", 1);
    let gate_stride = env_usize("GATE_STRIDE", 100);
    let gate_rounds = env_usize("GATE_ROUNDS", 2);
    let buckets = env_usize("BUCKETS", 20);
    assert!(stride >= 1 && gate_stride >= 1 && buckets >= 1);
    let arms = match std::env::var("REPRUNE").as_deref() {
        Err(_) | Ok("incremental") => vec![Reprune::Incremental],
        Ok("full") => vec![Reprune::Full],
        Ok("both") => vec![Reprune::Full, Reprune::Incremental],
        Ok("both-reversed") => vec![Reprune::Incremental, Reprune::Full],
        Ok("none") => Vec::new(),
        Ok(other) => {
            panic!("REPRUNE must be incremental, full, both, both-reversed or none, not {other}")
        }
    };
    assert!(
        !arms.is_empty() || (gate_stride == 1 && gate_rounds >= 1),
        "REPRUNE=none checks build_partition against the stored index alone, which takes \
         GATE_STRIDE=1 and at least one GATE_ROUNDS"
    );

    let head = git_head();
    println!("git {head}");
    println!("index {index}");
    let dataset = Dataset::open(&index).await.unwrap();
    let stored = load(&dataset, stride == 1 || gate_stride == 1).await;
    let params = BuildParams::maintenance(&stored.metadata);
    let loaded = origin.elapsed().as_secs_f64();
    println!(
        "loaded {} vectors of {} in {loaded:.1} s; R {}, L {}, alpha {}, seed {}; stored medoid {}; arms {arms:?}",
        stored.row_ids.len(),
        stored.metadata.dimension,
        params.max_degree,
        params.search_list_size,
        params.alpha,
        params.seed,
        stored.medoid
    );
    assert_eq!(params.seed, BuildParams::default().seed);

    // G1: every arm builds what `build_partition` builds, graph and medoid; the
    // incremental arm, which is a copy of it, also counts the same distances. Over
    // the whole partition `build_partition` must also build the stored graph.
    let gate_store = store_of(&stored, gate_stride);
    let mut gate_seconds = Vec::new();
    for round in 0..gate_rounds {
        let counted = Comparisons::default();
        let clock = Instant::now();
        let built = build_partition(&gate_store, &params, &counted).unwrap();
        let production = clock.elapsed().as_secs_f64();
        if let Some(apart) = differing_from_stored(&stored, &built.graph, gate_stride) {
            assert_eq!(
                apart, 0,
                "G1 round {round}: build_partition and the stored index differ in {apart} vertices"
            );
            assert_eq!(
                built.medoid, stored.medoid,
                "G1 round {round}: build_partition and the stored index differ in the medoid"
            );
            println!(
                "G1 round {round}: build_partition equals the stored index, graph and medoid, \
                 over {} distances",
                counted.get()
            );
        }
        let mut copies = Vec::new();
        for arm in &arms {
            let clock = Instant::now();
            let copied = profiled_build(&gate_store, &params, *arm, buckets, origin, false);
            let copy = clock.elapsed().as_secs_f64();
            assert_eq!(
                copied.medoid, built.medoid,
                "G1 {arm:?}: the medoids differ"
            );
            let apart = differing(&copied.graph, &built.graph);
            assert_eq!(
                apart, 0,
                "G1 {arm:?}: {apart} vertices have other out-edges"
            );
            if *arm == Reprune::Incremental {
                assert_eq!(
                    copied.distances(),
                    counted.get(),
                    "G1: the incremental copy counted other distances"
                );
            }
            println!(
                "G1 round {round} {arm:?}: {} vertices, identical graph and medoid; \
                 build_partition {production:.2} s over {} distances, copy {copy:.2} s over {} \
                 ({:+.1}%)",
                gate_store.len(),
                counted.get(),
                copied.distances(),
                100.0 * (copy / production - 1.0)
            );
            copies.push(json!({"arm": arm, "seconds": copy, "distances": copied.distances()}));
        }
        gate_seconds.push(json!({
            "production": production,
            "production_distances": counted.get(),
            "copies": copies,
        }));
    }
    drop(gate_store);

    let store = store_of(&stored, stride);
    let mut runs = Vec::new();
    let mut profiled_arms: Vec<Profiled> = Vec::new();
    let mut failed = false;
    for arm in &arms {
        println!(
            "profiling arm {arm:?} over {} vectors (stride {stride}) from {:.1} s",
            store.len(),
            origin.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().unwrap();
        let profiled = profiled_build(&store, &params, *arm, buckets, origin, true);
        println!(
            "setup: randomize {:.2} s, medoid {:.2} s over {} distances",
            seconds(profiled.randomize_nanos),
            seconds(profiled.medoid_nanos),
            profiled.medoid_distances
        );
        print_passes(&profiled);

        // G2: over the whole partition every arm must build the stored graph.
        let stored_match = differing_from_stored(&stored, &profiled.graph, stride).map(|apart| {
            let medoid_equal = profiled.medoid == stored.medoid;
            println!(
                "G2 {arm:?} against the stored index: {apart} vertices differ, medoid {}",
                if medoid_equal { "equal" } else { "DIFFERS" }
            );
            failed |= apart != 0 || !medoid_equal;
            json!({"differing_vertices": apart, "medoid_equal": medoid_equal})
        });
        runs.push(json!({
            "arm": arm,
            "randomize_nanos": profiled.randomize_nanos,
            "medoid_nanos": profiled.medoid_nanos,
            "medoid_distances": profiled.medoid_distances,
            "medoid": profiled.medoid,
            "passes": profiled.passes,
            "stored_match": stored_match,
        }));
        profiled_arms.push(profiled);
    }

    // The arms must agree with each other byte for byte, whatever the stored index says.
    let arms_differ = profiled_arms.split_first().map(|(first, rest)| {
        rest.iter()
            .map(|other| {
                let apart = differing(&first.graph, &other.graph)
                    + usize::from(first.medoid != other.medoid);
                println!(
                    "arms {:?} and {:?}: {apart} vertices (or the medoid) differ",
                    first.arm, other.arm
                );
                failed |= apart != 0;
                apart
            })
            .sum::<usize>()
    });

    let report = json!({
        "git": head,
        "index": index,
        "stored_rows": stored.row_ids.len(),
        "dimension": stored.metadata.dimension,
        "stride": stride,
        "rows": store.len(),
        "params": {
            "max_degree": params.max_degree,
            "search_list_size": params.search_list_size,
            "alpha": params.alpha,
            "seed": params.seed,
        },
        "loaded_seconds": loaded,
        "gate": {"stride": gate_stride, "rounds": gate_seconds},
        "runs": runs,
        "arms_differing": arms_differ,
        "finished_seconds": origin.elapsed().as_secs_f64(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("wrote {out}");
    if failed {
        eprintln!("FAILED: a graph differs from another arm or from the stored index");
        std::process::exit(1);
    }
}
