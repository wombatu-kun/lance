// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whether a walk steered by quantised distances arrives where an exact one does.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --profile release-no-lto --example coded_walk
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `QUERIES` (default 500), `ROWS_PER_PARTITION` (default `1000,8192,65536`),
//! `PROBE_PERCENT` (default 20) or `NPROBES` (one per granularity, overriding the
//! percentage), `BEAMS` (default `10,20,40,80,160,320`), `DEGREE` (default 64),
//! `RQ_BITS` (default `1,3,5`), `SQ_BITS` (default 8), `TARGET_RECALL` (default
//! 95), `ROWS_PER_FRAGMENT` (default 10000), `HYBRID` (default 1).
//!
//! The phase D gate (`examples/memory_gate.rs`) measured that a traversal reading
//! only the vertices it touches moves a tenth to a three-hundredth of the pages the
//! driver moves today - but only with quantised codes resident, and it assumed the
//! one thing it could not measure: that a walk steered by codes expands the same
//! vertices as one steered by floats. Every byte phase D saves rests on that
//! assumption, so it is measured here before any of it is built.
//!
//! Only the walk is quantised. The graph is the same graph in every arm, built with
//! exact distances - DiskANN keeps full vectors in memory while building and so do
//! we - and the codes are residuals against the partition's own IVF centroid, which
//! is the geometry a real segment has and the reason a one-bit code can work at
//! all. So the arms differ in exactly one thing: what the beam search compares.
//!
//! Four answers are scored per arm, because they cost different reads:
//!
//! - **walk**: the walk's own nearest `K`, in the order the codes put them. No
//!   vector is read at all, and no re-ranking happens.
//! - **rerank K**: the nearest `K` of each probed partition, re-scored with their
//!   real vectors. `nprobes * K` vertex reads, which is what the gate charged the
//!   coded arm for.
//! - **rerank L**: every candidate the walk kept, re-scored. `nprobes * beam`
//!   reads.
//! - **rerank E**: every vertex the walk expanded, re-scored. This is the set
//!   DiskANN actually answers from, and it is free in their layout: one page
//!   carries a vertex's vector next to its edges, so a hop that reads the edges
//!   has the vector in hand. Our columns are separate, so it is one more ranged
//!   read per expansion - and since a walk only stops when nothing in its list is
//!   unexpanded, this set *contains* the beam. It runs about a fifth larger at the
//!   narrow beams a working point uses, and the gap closes as the beam widens,
//!   because a list long enough to truncate nothing ends up holding everything the
//!   walk expanded.
//!
//! The `+h` arms take the other half of the same trade. They *walk* with the exact
//! distance of every vertex they expand, so the beam is re-sorted with a true value
//! wherever one has already been paid for, and the codes steer only the vertices
//! nothing has been read for yet. Everything else is held equal: same graph, same
//! codes, same beam, same entry point.
//!
//! Recall is compared at equal recall rather than at equal beam: a coded arm that
//! needs a wider beam to match is not cheaper for having read fewer bytes per
//! comparison, and the table prints the comparisons each arm spends to reach the
//! target so the two effects cannot be confused.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
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
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::RabitQuantizationStorage;
use lance_index::vector::bq::transform::RQTransformer;
use lance_index::vector::flat::storage::FlatFloatStorage;
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::sq::storage::ScalarQuantizationStorage;
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_index::vector::transform::Transformer;
use lance_index::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN, SQ_CODE_COLUMN};
use lance_io::scheduler::ScanScheduler;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::format::INDEX_FILE_NAME;
use lance_vamana::io::{open_file, read_partition, read_segment, scan_scheduler};
use lance_vamana::partition::{Partition, PartitionGraph};
use lance_vamana::search::{Comparisons, SearchResult, SearchScratch, flat_storage, greedy_search};
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

/// Exact nearest `K` positions of one query, by brute force over every row.
fn exact_top(store: &FlatFloatStorage, query: ArrayRef) -> HashSet<u64> {
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

/// One partition, held for as long as the sweep keeps walking it.
struct Probe {
    entry: PartitionEntry,
    partition: Partition,
    /// The exact distances: what the reference arm walks by, and what every coded
    /// arm's candidates are re-scored with.
    exact: FlatFloatStorage,
    /// Base-vector position of every local id, so a candidate can be checked
    /// against the brute-force answer.
    positions: Vec<u64>,
    /// The partition's IVF centroid, which the codes are residuals against.
    centroid: Vec<f32>,
}

async fn load_probes(
    scheduler: &Arc<ScanScheduler>,
    dir: &Path,
    file_sizes: &HashMap<String, u64>,
    manifest: &SegmentManifest,
    wanted: &HashSet<u32>,
    ids: &HashMap<u64, u64>,
) -> HashMap<u32, Probe> {
    let mut probes = HashMap::with_capacity(wanted.len());
    for partition_id in wanted {
        let entry = manifest.partition(*partition_id).unwrap().clone();
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
        let positions = partition
            .graph()
            .row_ids()
            .iter()
            .map(|address| ids[address])
            .collect();
        let centroid = manifest
            .ivf()
            .centroid(*partition_id as usize)
            .expect("a probed partition has a centroid");
        let centroid = centroid.as_primitive::<Float32Type>().values().to_vec();
        probes.insert(
            *partition_id,
            Probe {
                entry,
                partition,
                exact,
                positions,
                centroid,
            },
        );
    }
    probes
}

/// A partition's vectors as residuals against its centroid, with `|v - c|^2`.
///
/// Both are what `RQTransformer` expects, and the residual is the whole reason a
/// code of one bit a dimension can steer anything: it quantises the offset from a
/// centroid the query knows exactly, not the vector.
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

/// RaBitQ codes for one partition, resident, in Lance's own implementation.
///
/// Lance's rather than ours on purpose: what is being measured is whether a
/// quantised distance is good enough to walk by, and a hand-rolled estimator would
/// measure its own bugs instead. The layout it holds is not the layout we would
/// ship - it packs codes in blocks of 32 vectors, which a fixed-stride column
/// cannot - but layout does not enter a recall measurement.
fn rabit_store(probe: &Probe, num_bits: u8) -> RabitQuantizationStorage {
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

    // Everything the transform needed as input goes again, so that what stays
    // resident is codes and factors and nothing else - the bytes the arm is
    // claiming, rather than the bytes it happened to be built from.
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
    RabitQuantizationStorage::try_from_batch(coded, &quantizer.metadata(None), DISTANCE_TYPE, None)
        .unwrap()
}

/// Scalar codes for one partition: a byte a dimension, the generous control.
///
/// Four times the bytes of a one-bit RaBitQ code at `d = 128` and eight times at
/// `d = 960`, so it is not a candidate for what stays resident - it is here to
/// separate "codes cannot steer a walk" from "one bit cannot steer a walk".
/// Bounds come from the whole dataset rather than from the partition, because that
/// is what Lance's own `IVF_SQ` build does.
fn scalar_store(probe: &Probe, num_bits: u16, bounds: Range<f64>) -> ScalarQuantizationStorage {
    let dim = probe.partition.vectors().value_length() as usize;
    let quantizer = ScalarQuantizer::with_bounds(num_bits, dim, bounds.clone());
    let codes = quantizer
        .transform::<Float32Type>(probe.partition.vectors())
        .unwrap();
    let batch = RecordBatch::try_from_iter_with_nullable(vec![
        (
            ROW_ID,
            Arc::new(UInt64Array::from(
                probe.partition.graph().row_ids().to_vec(),
            )) as ArrayRef,
            false,
        ),
        (SQ_CODE_COLUMN, codes, false),
    ])
    .unwrap();
    ScalarQuantizationStorage::try_new(num_bits, DISTANCE_TYPE, bounds, [batch], None).unwrap()
}

/// What a vertex costs in a resident store, from the store's own schema.
///
/// Row ids are excluded: a resident code column is addressed by local id, and the
/// row id of an answer comes off the partition file with its vector.
fn resident_bytes<S: VectorStore>(store: &S) -> usize {
    store
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name() != ROW_ID)
        .map(|field| match field.data_type() {
            DataType::FixedSizeList(inner, width) => {
                *width as usize * inner.data_type().primitive_width().unwrap_or(0)
            }
            other => other.primitive_width().unwrap_or(0),
        })
        .sum()
}

/// The distance a store wants a query expressed as.
///
/// Not a detail: RaBitQ quantises the residual `v - c`, but its raw-query estimator
/// folds the centroid into each vertex's own factors, so what the calculator wants
/// is the *raw* query plus `|q - c|^2` - handing it the residual instead produces
/// distances that are wrong rather than approximate. Lance's own IVF path decides
/// the same thing in `use_query_residual` (`ivf/v2.rs`), and the calibration below
/// is what keeps this from being a matter of trust.
fn query_key(probe: &Probe, query: &ArrayRef, centroid_distance: bool) -> (ArrayRef, f32) {
    if !centroid_distance {
        return (query.clone(), 0.0);
    }
    let values = query.as_primitive::<Float32Type>().values();
    let norm = values
        .iter()
        .zip(&probe.centroid)
        .map(|(value, center)| (value - center) * (value - center))
        .sum();
    (query.clone(), norm)
}

/// How well one arm's distances stand in for the real ones, over a whole partition.
///
/// The walk table alone cannot tell a quantiser that loses a little accuracy from a
/// glue mistake that makes its distances meaningless, because both show up as lost
/// recall. This measures the codes directly: the median relative error against the
/// exact distance, and how much of a partition's true nearest `K` the approximate
/// order puts in its own nearest `K`.
fn calibrate<S: VectorStore>(
    probe: &Probe,
    store: &S,
    centroid_distance: bool,
    queries: &[ArrayRef],
) -> (f64, f64) {
    let mut errors = Vec::with_capacity(queries.len() * probe.partition.len());
    let mut agreement = 0.0;
    for query in queries {
        let (key, dist_q_c) = query_key(probe, query, centroid_distance);
        let approximate = store.dist_calculator(key, dist_q_c);
        let exact = probe.exact.dist_calculator(query.clone(), 0.0);
        let mut scored = (0..probe.partition.len() as u32)
            .map(|id| (approximate.distance(id), exact.distance(id), id))
            .collect::<Vec<_>>();
        for (approximate, exact, _) in &scored {
            if *exact > 0.0 {
                errors.push(((approximate - exact) / exact).abs() as f64);
            }
        }
        let take = K.min(scored.len());
        scored.sort_unstable_by(|left, right| left.1.total_cmp(&right.1));
        let truth = scored
            .iter()
            .take(take)
            .map(|(_, _, id)| *id)
            .collect::<HashSet<_>>();
        scored.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
        let found = scored
            .iter()
            .take(take)
            .filter(|(_, _, id)| truth.contains(id))
            .count();
        agreement += found as f64 / take as f64;
    }
    errors.sort_unstable_by(f64::total_cmp);
    (errors[errors.len() / 2], agreement / queries.len() as f64)
}

/// A seat in a hybrid walk's search list.
///
/// The same shape `greedy_search` keeps privately. This walk is a copy of it rather
/// than a parameter added to it because the question is whether an exact distance
/// at expansion changes where a walk goes, and that only means something if the
/// walk it is compared against stays untouched.
struct Seat {
    node: OrderedNode,
    expanded: bool,
}

/// A walk that spends one exact distance on every vertex it expands.
///
/// The correction happens *at* the expansion and never before it, and that is what
/// keeps the arm honest rather than flattering: a vertex whose vector is read is a
/// vertex whose page was fetched, so peeking at a true distance before deciding
/// whether to expand would charge DiskANN's price for a read DiskANN never makes.
/// A correction therefore cannot change which vertex is expanded now, only where
/// the list stands when the next one is chosen - and that is a bigger change than
/// it sounds. A code that flatters a vertex seats it too near the front; correcting
/// it at expansion sends it to the back, and the back of a full list is the bar a
/// new candidate has to beat to be admitted at all. So a hybrid walk keeps letting
/// candidates in that the same walk on codes alone would have turned away, and it
/// expands more vertices at the same beam width. That is why the arms are compared
/// at equal recall and not at equal beam.
///
/// With `approximate` and `exact` the same store every correction is a no-op and
/// this is `greedy_search` exactly, which is what [`verify_hybrid`] checks.
fn hybrid_search(
    graph: &PartitionGraph,
    approximate: &impl DistCalculator,
    exact: &impl DistCalculator,
    entry_point: u32,
    search_list_size: usize,
    comparisons: &Comparisons,
    corrections: &Comparisons,
) -> SearchResult {
    let mut seen = vec![false; graph.len()];
    seen[entry_point as usize] = true;
    comparisons.record(1);
    let mut list = Vec::with_capacity(search_list_size.min(graph.len()) + 1);
    list.push(Seat {
        node: OrderedNode::new(entry_point, OrderedFloat(approximate.distance(entry_point))),
        expanded: false,
    });
    let mut visited = Vec::new();

    while let Some(position) = list.iter().position(|seat| !seat.expanded) {
        list[position].expanded = true;
        let id = list[position].node.id;
        corrections.record(1);
        let corrected = OrderedFloat(exact.distance(id));
        if corrected != list[position].node.dist {
            let mut seat = list.remove(position);
            seat.node.dist = corrected;
            // No truncation test, unlike the insertion below: the removal already
            // freed the seat this is going back into, so a corrected vertex always
            // has somewhere to land however far back it belongs.
            let at = list.partition_point(|other| other.node.dist <= corrected);
            list.insert(at, seat);
            // The list is scanned by position, so a re-insertion that landed out of
            // order would not fail - it would quietly change which vertex is
            // expanded next. Its two neighbours are the whole of the invariant, and
            // checking them is constant time.
            assert!(
                list[..at]
                    .last()
                    .is_none_or(|before| before.node.dist <= corrected)
                    && list[at + 1..]
                        .first()
                        .is_none_or(|after| corrected <= after.node.dist),
                "a corrected vertex was re-seated out of order"
            );
        }
        visited.push(OrderedNode::new(id, corrected));

        for neighbor in graph.neighbors(id).unwrap() {
            if seen[*neighbor as usize] {
                continue;
            }
            seen[*neighbor as usize] = true;
            comparisons.record(1);
            let distance = OrderedFloat(approximate.distance(*neighbor));
            let at = list.partition_point(|seat| seat.node.dist <= distance);
            if at >= search_list_size {
                continue;
            }
            list.insert(
                at,
                Seat {
                    node: OrderedNode::new(*neighbor, distance),
                    expanded: false,
                },
            );
            list.truncate(search_list_size);
        }
    }

    SearchResult {
        candidates: list.into_iter().map(|seat| seat.node).collect(),
        visited,
    }
}

/// Check the hybrid walk against the walk it is a copy of.
///
/// Handed one store for both calculators, every correction is a no-op, so the run
/// has to come back from `greedy_search` bit for bit: same candidates with the same
/// distances, same vertices expanded in the same order, same comparison count. Any
/// difference is a difference in the copy - the loop, the truncation, the counting -
/// rather than a property of the hybrid, and without this the arm would be
/// measuring its own reimplementation of the thing it is compared against.
///
/// Run with the exact store and with a coded one, because the two differ in
/// something this walk does care about: how many distances tie.
fn verify_hybrid<S: VectorStore>(
    probe: &Probe,
    store: &S,
    scoring: Scoring,
    queries: &[ArrayRef],
    beam: usize,
) {
    for query in queries {
        let (key, dist_q_c) = query_key(probe, query, scoring.centroid_distance);
        let calculator = store.dist_calculator(key, dist_q_c);

        let expected_comparisons = Comparisons::default();
        let mut scratch = SearchScratch::new(probe.partition.len());
        let expected = greedy_search(
            probe.partition.graph(),
            &calculator,
            probe.entry.medoid,
            beam,
            &mut scratch,
            &expected_comparisons,
        )
        .unwrap();

        let comparisons = Comparisons::default();
        let corrections = Comparisons::default();
        let found = hybrid_search(
            probe.partition.graph(),
            &calculator,
            &calculator,
            probe.entry.medoid,
            beam,
            &comparisons,
            &corrections,
        );

        assert_eq!(
            comparisons.get(),
            expected_comparisons.get(),
            "a hybrid walk with nothing to correct spent other comparisons"
        );
        assert_eq!(
            corrections.get() as usize,
            found.visited.len(),
            "a hybrid walk must read exactly one vector per vertex it expands"
        );
        assert_eq!(
            found.visited, expected.visited,
            "a hybrid walk with nothing to correct expanded other vertices"
        );
        assert_eq!(
            found.candidates, expected.candidates,
            "a hybrid walk with nothing to correct kept other candidates"
        );
    }
}

/// One candidate a walk came back with, scored both ways.
struct Candidate {
    /// The distance the walk ranked it by, quantised in every arm but the first.
    approx: f32,
    exact: f32,
    position: u64,
    /// Whether it was among its own partition's nearest `K`, which is the cheapest
    /// of the re-ranking reads.
    shortlisted: bool,
    /// Whether the beam still held it when the walk stopped.
    in_beam: bool,
    /// Whether the walk followed its out-edges.
    expanded: bool,
}

/// Which vertices of a walk an answer may be read from, and so what it costs.
///
/// There is no "beam and expanded" between the last two, because a walk stops when
/// no seat in its list is unexpanded and every expanded vertex is recorded: the beam
/// it ends with is always a subset of what it expanded. That invariant is asserted
/// where the two sets are merged rather than stated here, because a column equal to
/// another column by construction is noise in a measurement.
#[derive(Clone, Copy)]
enum Answer {
    /// The walk's own nearest `K` per partition: `nprobes * K` vector reads.
    Shortlist,
    /// Every candidate the beam kept: `nprobes * beam` reads.
    Beam,
    /// Every vertex the walk expanded: free where one page carries a vertex's
    /// vector next to its edges, one extra ranged read per expansion here.
    Expanded,
}

impl Answer {
    fn admits(self, candidate: &Candidate) -> bool {
        match self {
            Self::Shortlist => candidate.shortlisted,
            Self::Beam => candidate.in_beam,
            Self::Expanded => candidate.expanded,
        }
    }
}

/// Hits among the nearest `K` by `key`, over the vertices `answer` admits.
fn hits<F>(candidates: &[Candidate], expected: &HashSet<u64>, answer: Answer, key: F) -> usize
where
    F: Fn(&Candidate) -> f32,
{
    let mut ranked = candidates
        .iter()
        .filter(|candidate| answer.admits(candidate))
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| key(left).total_cmp(&key(right)));
    ranked
        .iter()
        .take(K)
        .filter(|candidate| expected.contains(&candidate.position))
        .count()
}

/// How an arm scores what its walk sees.
#[derive(Clone, Copy)]
struct Scoring {
    /// Whether the store wants the raw query and `|q - c|^2` rather than the
    /// residual - see [`query_key`], where getting this wrong is not approximate.
    centroid_distance: bool,
    /// Whether a vertex's distance is recomputed exactly when it is expanded.
    hybrid: bool,
}

impl Scoring {
    /// The query as it came.
    const PLAIN: Self = Self {
        centroid_distance: false,
        hybrid: false,
    };
    /// The query as it came *and* `|q - c|^2`, which is what RaBitQ's estimator
    /// wants and what nothing else does.
    const RABIT: Self = Self {
        centroid_distance: true,
        hybrid: false,
    };

    fn hybrid(self) -> Self {
        Self {
            hybrid: true,
            ..self
        }
    }
}

#[derive(Default)]
struct Row {
    beam: usize,
    walk: f64,
    rerank_k: f64,
    rerank_all: f64,
    /// Recall answering from the expanded set, which is what DiskANN does - and
    /// what a hybrid walk has already paid the reads for.
    rerank_expanded: f64,
    comparisons: f64,
    visited: f64,
    rescored: f64,
    /// Of the vertices the exact walk expanded, the fraction this one expanded
    /// too. The diagnostic behind the recall: a coded walk that keeps recall by
    /// walking somewhere else entirely is a different claim from one that follows
    /// the same path.
    overlap: f64,
    micros: u128,
}

/// One way of forming an answer out of what a walk came back with.
///
/// `reads` takes whether the arm walked hybrid because that decides the floor: a
/// walk that has already read the vector of everything it expanded cannot answer
/// for less than those reads, whatever set it answers from.
struct Strategy {
    name: &'static str,
    recall: fn(&Row) -> f64,
    reads: fn(&Row, bool) -> f64,
}

/// One arm: what it costs resident, how good its distances are, and what it scored.
struct Arm {
    label: String,
    bytes: usize,
    /// Median relative error of its distances against the exact ones.
    error: f64,
    /// The share of a partition's true nearest `K` its own order keeps.
    agreement: f64,
    /// Whether its walk read the vector of every vertex it expanded. Those reads
    /// are already spent by the time the answer is formed, so they set the floor
    /// under what any answer this arm gives costs.
    hybrid: bool,
    rows: Vec<Row>,
}

/// Walk every query with one arm's distances, and score what came back.
#[allow(clippy::too_many_arguments)]
fn measure<S: VectorStore>(
    probes: &HashMap<u32, Probe>,
    stores: &HashMap<u32, S>,
    scoring: Scoring,
    queries: &[ArrayRef],
    plans: &[Vec<u32>],
    truth: &[HashSet<u64>],
    beam: usize,
    reference: Option<&[Vec<(u32, u32)>]>,
) -> (Row, Vec<Vec<(u32, u32)>>) {
    let mut row = Row {
        beam,
        ..Default::default()
    };
    let mut expansions = Vec::with_capacity(queries.len());
    let started = Instant::now();
    for (index, query) in queries.iter().enumerate() {
        let mut candidates = Vec::new();
        let mut expanded = Vec::new();
        for partition_id in &plans[index] {
            let probe = &probes[partition_id];
            let (key, dist_q_c) = query_key(probe, query, scoring.centroid_distance);
            let calculator = stores[partition_id].dist_calculator(key, dist_q_c);
            let exact = probe.exact.dist_calculator(query.clone(), 0.0);
            let comparisons = Comparisons::default();
            let walk = if scoring.hybrid {
                let corrections = Comparisons::default();
                let walk = hybrid_search(
                    probe.partition.graph(),
                    &calculator,
                    &exact,
                    probe.entry.medoid,
                    beam,
                    &comparisons,
                    &corrections,
                );
                assert_eq!(
                    corrections.get() as usize,
                    walk.visited.len(),
                    "the read count of a hybrid arm is its expansion count"
                );
                walk
            } else {
                let mut scratch = SearchScratch::new(probe.partition.len());
                greedy_search(
                    probe.partition.graph(),
                    &calculator,
                    probe.entry.medoid,
                    beam,
                    &mut scratch,
                    &comparisons,
                )
                .unwrap()
            };

            // The beam and the expanded set overlap heavily and are held as one set,
            // because a vertex in both is one vector read, not two.
            let mut seat_of = HashMap::with_capacity(walk.candidates.len() + walk.visited.len());
            for (rank, node) in walk.candidates.iter().enumerate() {
                seat_of.insert(node.id, candidates.len());
                candidates.push(Candidate {
                    approx: node.dist.0,
                    exact: exact.distance(node.id),
                    position: probe.positions[node.id as usize],
                    shortlisted: rank < K,
                    in_beam: true,
                    expanded: false,
                });
            }
            let mut in_both = 0;
            for node in &walk.visited {
                match seat_of.get(&node.id) {
                    Some(seat) => {
                        candidates[*seat].expanded = true;
                        in_both += 1;
                    }
                    None => candidates.push(Candidate {
                        approx: node.dist.0,
                        exact: exact.distance(node.id),
                        position: probe.positions[node.id as usize],
                        shortlisted: false,
                        in_beam: false,
                        expanded: true,
                    }),
                }
            }
            // A walk stops when nothing in its list is unexpanded, so every seat it
            // ends with was expanded. The read counts rest on it: answering from the
            // expanded set is a superset read, never a second one.
            assert_eq!(
                in_both,
                walk.candidates.len(),
                "a walk came back holding a candidate it never expanded"
            );
            row.comparisons += comparisons.get() as f64;
            row.visited += walk.visited.len() as f64;
            row.rescored += walk.candidates.len() as f64;
            expanded.extend(walk.visited.iter().map(|node| (*partition_id, node.id)));
        }
        expanded.sort_unstable();

        let expected = &truth[index];
        row.walk += hits(&candidates, expected, Answer::Beam, |candidate| {
            candidate.approx
        }) as f64;
        row.rerank_k += hits(&candidates, expected, Answer::Shortlist, |candidate| {
            candidate.exact
        }) as f64;
        row.rerank_all += hits(&candidates, expected, Answer::Beam, |candidate| {
            candidate.exact
        }) as f64;
        row.rerank_expanded += hits(&candidates, expected, Answer::Expanded, |candidate| {
            candidate.exact
        }) as f64;
        match reference {
            Some(reference) => {
                let same = reference[index]
                    .iter()
                    .filter(|vertex| expanded.binary_search(vertex).is_ok())
                    .count();
                row.overlap += same as f64 / reference[index].len().max(1) as f64;
            }
            // The arm that is the reference overlaps itself entirely, which is
            // worth printing rather than leaving as a zero that reads as a loss.
            None => row.overlap += 1.0,
        }
        expansions.push(expanded);
    }
    row.micros = started.elapsed().as_micros();
    (row, expansions)
}

/// What every granularity is measured at.
struct Grid {
    beams: Vec<usize>,
    rq_bits: Vec<u8>,
    sq_bits: u16,
    degree: u32,
    probe_percent: usize,
    target_recall: f64,
    rows_per_fragment: usize,
    /// Whether every coded arm is measured a second time walking with the exact
    /// distance of each vertex it expands.
    hybrid: bool,
    /// Scalar bounds over the whole dataset, which is where Lance's own `IVF_SQ`
    /// build takes them from. A per-partition range would flatter the arm.
    bounds: Range<f64>,
}

/// Everything, at one partition size.
///
/// The sweep is not decoration: a coarser partition is a longer residual, so the
/// code that steers the walk gets worse in exactly the direction phase D wants to
/// move - fewer, larger partitions. The two effects have to be read together.
async fn granularity(
    grid: &Grid,
    vectors: &FixedSizeListArray,
    queries: &[ArrayRef],
    truth: &[HashSet<u64>],
    rows_per_partition: usize,
    nprobes: Option<usize>,
) {
    let rows = vectors.len();
    let num_queries = queries.len();
    let partitions = rows.div_ceil(rows_per_partition).max(1) as u32;
    let nprobes = nprobes
        .unwrap_or_else(|| ((grid.probe_percent * partitions as usize).div_ceil(100)).max(1));
    let temp = tempfile::tempdir().unwrap();
    let uri = temp.path().to_str().unwrap();
    let mut dataset = write_dataset(uri, vectors.clone(), grid.rows_per_fragment).await;
    let started = Instant::now();
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_FIELD, partitions)
            .with_distance_type(DISTANCE_TYPE)
            .with_graph_params(BuildParams {
                max_degree: grid.degree,
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

    let ids = ids_by_address(&dataset).await;
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
    let started = Instant::now();
    let probes = load_probes(
        &scheduler,
        &segment_dir,
        &file_sizes,
        &manifest,
        &wanted,
        &ids,
    )
    .await;
    println!(
        "{} of {partitions} partitions probed by some query, read in {:.1}s",
        probes.len(),
        started.elapsed().as_secs_f64()
    );

    let started = Instant::now();
    let exact_stores = probes
        .iter()
        .map(|(id, probe)| (*id, probe.exact.clone()))
        .collect::<HashMap<_, _>>();
    let scalar_stores = probes
        .iter()
        .map(|(id, probe)| (*id, scalar_store(probe, grid.sq_bits, grid.bounds.clone())))
        .collect::<HashMap<_, _>>();
    let rabit_stores = grid
        .rq_bits
        .iter()
        .map(|bits| {
            (
                *bits,
                probes
                    .iter()
                    .map(|(id, probe)| (*id, rabit_store(probe, *bits)))
                    .collect::<HashMap<_, _>>(),
            )
        })
        .collect::<Vec<_>>();
    println!("codes built in {:.1}s", started.elapsed().as_secs_f64());

    // One partition, the queries that route to it first: enough to place the error
    // of every arm's distances before any of them is asked to walk.
    let sample = *plans[0].first().unwrap();
    let calibration_queries = &queries[..queries.len().min(20)];
    if grid.hybrid {
        let widest = grid.beams.iter().max().copied().unwrap();
        verify_hybrid(
            &probes[&sample],
            &exact_stores[&sample],
            Scoring::PLAIN,
            calibration_queries,
            widest,
        );
        for (_, stores) in &rabit_stores {
            verify_hybrid(
                &probes[&sample],
                &stores[&sample],
                Scoring::RABIT,
                calibration_queries,
                widest,
            );
        }
        println!(
            "hybrid walk reproduces greedy_search on {} queries at beam {widest}, \
             {} stores",
            calibration_queries.len(),
            1 + rabit_stores.len()
        );
    }
    let mut arms = Vec::with_capacity(2 + rabit_stores.len());
    for (label, bytes, calibration) in [
        (
            "exact".to_string(),
            resident_bytes(&exact_stores[&sample]),
            calibrate(
                &probes[&sample],
                &exact_stores[&sample],
                false,
                calibration_queries,
            ),
        ),
        (
            format!("sq{}", grid.sq_bits),
            resident_bytes(&scalar_stores[&sample]),
            calibrate(
                &probes[&sample],
                &scalar_stores[&sample],
                false,
                calibration_queries,
            ),
        ),
    ] {
        arms.push(Arm {
            label,
            bytes,
            error: calibration.0,
            agreement: calibration.1,
            hybrid: false,
            rows: Vec::new(),
        });
    }
    for (bits, stores) in &rabit_stores {
        let (error, agreement) = calibrate(
            &probes[&sample],
            &stores[&sample],
            true,
            calibration_queries,
        );
        arms.push(Arm {
            label: format!("rq{bits}"),
            bytes: resident_bytes(&stores[&sample]),
            error,
            agreement,
            hybrid: false,
            rows: Vec::new(),
        });
    }
    // A hybrid arm holds the same codes as its twin and therefore the same resident
    // bytes and the same calibration: it differs in what it does with them, not in
    // what it stores. Only the coded arms get one - `sq8` is here to separate "codes
    // cannot steer" from "one bit cannot steer", and at 128 bytes a vertex it is not
    // a candidate for what stays resident whatever it scores.
    if grid.hybrid {
        for index in 0..rabit_stores.len() {
            let twin = &arms[2 + index];
            arms.push(Arm {
                label: format!("{}+h", twin.label),
                bytes: twin.bytes,
                error: twin.error,
                agreement: twin.agreement,
                hybrid: true,
                rows: Vec::new(),
            });
        }
    }
    println!(
        "\n  over partition {sample} ({} vertices), {} queries:",
        probes[&sample].partition.len(),
        calibration_queries.len()
    );
    println!(
        "  {:<7} {:>9} {:>16} {:>16}",
        "arm", "B/vertex", "median |err|", "top-K agreement"
    );
    for arm in &arms {
        println!(
            "  {:<7} {:>9} {:>15.4} {:>16.4}",
            arm.label, arm.bytes, arm.error, arm.agreement
        );
    }

    let hybrid_base = 2 + rabit_stores.len();
    for beam in &grid.beams {
        let (exact, reference) = measure(
            &probes,
            &exact_stores,
            Scoring::PLAIN,
            queries,
            &plans,
            truth,
            *beam,
            None,
        );
        arms[0].rows.push(exact);
        let (scalar, _) = measure(
            &probes,
            &scalar_stores,
            Scoring::PLAIN,
            queries,
            &plans,
            truth,
            *beam,
            Some(&reference),
        );
        arms[1].rows.push(scalar);
        for (index, (_, stores)) in rabit_stores.iter().enumerate() {
            let (rabit, _) = measure(
                &probes,
                stores,
                Scoring::RABIT,
                queries,
                &plans,
                truth,
                *beam,
                Some(&reference),
            );
            arms[2 + index].rows.push(rabit);
            if grid.hybrid {
                let (hybrid, _) = measure(
                    &probes,
                    stores,
                    Scoring::RABIT.hybrid(),
                    queries,
                    &plans,
                    truth,
                    *beam,
                    Some(&reference),
                );
                arms[hybrid_base + index].rows.push(hybrid);
            }
        }
    }

    let scale = num_queries as f64;
    let recalls = scale * K as f64;
    println!(
        "\n  {:<7} {:>9} {:>6} {:>8} {:>9} {:>9} {:>9} {:>8} {:>8} {:>9} {:>8} {:>7}",
        "arm",
        "B/vertex",
        "beam",
        "walk",
        "rerank K",
        "rerank L",
        "rerank E",
        "cmp/q",
        "exp/q",
        "rescore/q",
        "overlap",
        "us/q"
    );
    for arm in &arms {
        for row in &arm.rows {
            println!(
                "  {:<7} {:>9} {:>6} {:>8.4} {:>9.4} {:>9.4} {:>9.4} {:>8.0} {:>8.1} {:>9.1} \
                 {:>8.3} {:>7.0}",
                arm.label,
                arm.bytes,
                row.beam,
                row.walk / recalls,
                row.rerank_k / recalls,
                row.rerank_all / recalls,
                row.rerank_expanded / recalls,
                row.comparisons / scale,
                row.visited / scale,
                row.rescored / scale,
                row.overlap / scale,
                row.micros as f64 / scale,
            );
        }
    }

    // The comparison that matters: not what an arm scores at a given beam, but what
    // it has to spend to reach one recall. An arm that needs a wider beam pays for
    // it in comparisons and in re-ranking reads, and both are printed here rather
    // than left for the reader to reconstruct from the table above.
    println!("\n  at recall {:.2}:", grid.target_recall);
    println!(
        "  {:<10} {:>9} {:>6} {:>9} {:>8} {:>8} {:>13} {:>15}",
        "arm/answer",
        "B/vertex",
        "beam",
        "recall",
        "cmp/q",
        "reads/q",
        "cmp vs exact",
        "reads vs exact"
    );
    // A hybrid arm has already read the vector of every vertex it expanded by the
    // time it answers, so answering from the narrower set costs it nothing less.
    let strategies = [
        Strategy {
            name: "L",
            recall: |row| row.rerank_all,
            reads: |row, hybrid| if hybrid { row.visited } else { row.rescored },
        },
        Strategy {
            name: "E",
            recall: |row| row.rerank_expanded,
            reads: |row, _| row.visited,
        },
    ];
    let baseline = arms[0]
        .rows
        .iter()
        .find(|row| row.rerank_all / recalls >= grid.target_recall);
    let exact_comparisons = baseline.map(|row| row.comparisons / scale);
    let exact_reads = baseline.map(|row| row.rescored / scale);
    for arm in &arms {
        for strategy in &strategies {
            let (recall, reads) = (strategy.recall, strategy.reads);
            let label = format!("{}/{}", arm.label, strategy.name);
            match arm
                .rows
                .iter()
                .find(|row| recall(row) / recalls >= grid.target_recall)
            {
                Some(row) => println!(
                    "  {:<10} {:>9} {:>6} {:>9.4} {:>8.0} {:>8.1} {:>13} {:>15}",
                    label,
                    arm.bytes,
                    row.beam,
                    recall(row) / recalls,
                    row.comparisons / scale,
                    reads(row, arm.hybrid) / scale,
                    match exact_comparisons {
                        Some(exact) => format!("{:.2}x", row.comparisons / scale / exact),
                        None => "-".to_string(),
                    },
                    match exact_reads {
                        Some(exact) => format!("{:.2}x", reads(row, arm.hybrid) / scale / exact),
                        None => "-".to_string(),
                    }
                ),
                None => println!(
                    "  {:<10} {:>9} {:>6} {:>9} {:>8} {:>8} {:>13} {:>15}",
                    label,
                    arm.bytes,
                    format!(">{}", grid.beams.last().copied().unwrap_or_default()),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
                ),
            }
        }
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
    let num_queries = env_usize("QUERIES", 500).min(total_queries);
    let sweep = env_list("ROWS_PER_PARTITION", "1000,8192,65536");
    let sq_bits = env_usize("SQ_BITS", 8) as u16;
    let grid = Grid {
        beams: env_list("BEAMS", "10,20,40,80,160,320"),
        rq_bits: env_list("RQ_BITS", "1,3,5")
            .into_iter()
            .map(|bits| bits as u8)
            .collect(),
        sq_bits,
        degree: env_usize("DEGREE", 64) as u32,
        probe_percent: env_usize("PROBE_PERCENT", 20),
        target_recall: env_usize("TARGET_RECALL", 95) as f64 / 100.0,
        rows_per_fragment: env_usize("ROWS_PER_FRAGMENT", 10_000),
        hybrid: env_usize("HYBRID", 1) != 0,
        bounds: 0.0..0.0,
    };

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
        "SIFT {rows} x {dim}, {num_queries} queries, k = {K}, R = {}, target recall {:.2}",
        grid.degree, grid.target_recall
    );

    let started = Instant::now();
    let store = FlatFloatStorage::new(vectors.clone(), DISTANCE_TYPE);
    let truth = queries
        .iter()
        .map(|query| exact_top(&store, query.clone()))
        .collect::<Vec<_>>();
    let bounds = ScalarQuantizer::new(sq_bits, dim)
        .update_bounds::<Float32Type>(&vectors)
        .unwrap();
    println!(
        "exact top-{K} by brute force in {:.1}s, scalar bounds {:.1}..{:.1}",
        started.elapsed().as_secs_f64(),
        bounds.start,
        bounds.end
    );
    let grid = Grid { bounds, ..grid };

    // The gate chose an `nprobes` per granularity and reported its traffic there, so
    // being able to name the same points is what lets the two tables multiply
    // instead of describing two different indexes.
    let probing = std::env::var("NPROBES")
        .ok()
        .map(|_| env_list("NPROBES", ""));
    for (index, rows_per_partition) in sweep.iter().enumerate() {
        let nprobes = probing.as_ref().map(|list| list[index.min(list.len() - 1)]);
        granularity(
            &grid,
            &vectors,
            &queries,
            &truth,
            *rows_per_partition,
            nprobes,
        )
        .await;
    }
}
