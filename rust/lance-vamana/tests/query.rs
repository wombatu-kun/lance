// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Does a query through our own driver find what Lance's brute force finds?
//!
//! The reference is Lance's own exhaustive k-NN with `use_index(false)`, not a
//! second implementation of ours - checking a graph against a scan written by
//! the same hand proves only that the hand is consistent.
//!
//! Every recall figure here comes with what it cost. A walk that reaches every
//! vertex in a partition has perfect recall and has answered nothing, so recall
//! on its own cannot tell a working index from a scan in a costume.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::transaction::{
    DataOverlayGroup, Operation, UpdateMode, UpdatedFragmentOffsets,
};
use lance::dataset::{ProjectionRequest, WriteDestination};
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_core::utils::address::RowAddress;
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::DistanceType;
use lance_table::format::DataFile;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, build_index_segment, build_segment, create_index,
};
use lance_vamana::format::{FORMAT_VERSION, IndexMetadata, RowIdMode};
use lance_vamana::io::{SegmentWriter, read_segment, scan_scheduler};
use lance_vamana::partition::Partition;
use lance_vamana::query::{SearchParams, VamanaIndex};
use roaring::RoaringBitmap;
use uuid::Uuid;

mod common;
use common::{
    DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, brute_force, random_vectors, recall,
    sample_partition,
};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 4;
const K: usize = 10;
const BEAM: usize = 30;
const QUERIES: usize = 40;

/// Partitions of ~2048 vertices, because a graph only stops being a scan once a
/// partition is much larger than what one walk can reach. A walk reaches roughly
/// `expansions * max_degree` vertices, so at R=16 and L=30 that is around 500 -
/// and a 512-vertex partition would be exhausted by a single query. Measured, not
/// assumed: the smaller fixture this file started with scored recall 1.0 while
/// touching 77% of the dataset.
fn measurement_fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 4,
        rows_per_fragment: 2048,
        ..Default::default()
    }
}

/// Enough rows and fragments to be a real dataset, small enough to build in a
/// second. Used by everything that is not measuring.
fn small_fixture() -> DatasetFixture {
    DatasetFixture {
        fragments: 2,
        rows_per_fragment: 512,
        ..Default::default()
    }
}

/// A narrower graph than the default, so the tests build in seconds. The working
/// point measured on SIFT is R=64; nothing here is a quality statement.
fn params() -> IndexParams {
    IndexParams::new(VECTOR_COLUMN, PARTITIONS).with_graph_params(BuildParams {
        max_degree: 16,
        search_list_size: 64,
        ..Default::default()
    })
}

async fn indexed_dataset(uri: &str, fixture: &DatasetFixture) -> Dataset {
    let mut dataset = fixture.write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    dataset
}

/// The distance from `query` to its true nearest neighbour, from Lance.
async fn brute_force_best_distance(dataset: &Dataset, query: &[f32]) -> f32 {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.nearest(VECTOR_COLUMN, &key, 1).unwrap();
    scanner.use_index(false);
    let batch = scanner.try_into_batch().await.unwrap();
    batch["_distance"].as_primitive::<Float32Type>().value(0)
}

struct Measured {
    recall: f64,
    comparisons: f64,
    partitions: f64,
}

/// The exhaustive answer for every query, computed once: a Lance scan per query
/// per configuration would dominate the runtime and measure nothing new.
async fn ground_truth(dataset: &Dataset, queries: &[Vec<f32>]) -> Vec<Vec<u64>> {
    let mut truth = Vec::with_capacity(queries.len());
    for query in queries {
        truth.push(brute_force(dataset, query, K).await);
    }
    truth
}

async fn measure(
    index: &VamanaIndex,
    queries: &[Vec<f32>],
    truth: &[Vec<u64>],
    search: &SearchParams,
) -> Measured {
    let mut total_recall = 0.0;
    let mut total_comparisons = 0u64;
    let mut total_partitions = 0usize;
    for (query, exact) in queries.iter().zip(truth) {
        let result = index.search(query, search).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(found.len(), search.k, "a query returned the wrong count");
        total_recall += recall(&found, exact);
        total_comparisons += result.comparisons;
        total_partitions += result.partitions_read;
    }
    Measured {
        recall: total_recall / queries.len() as f64,
        comparisons: total_comparisons as f64 / queries.len() as f64,
        partitions: total_partitions as f64 / queries.len() as f64,
    }
}

#[tokio::test]
async fn top_k_matches_lance_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = measurement_fixture();
    let dataset = indexed_dataset(uri, &fixture).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let measured = measure(&index, &queries, &truth, &search).await;
    let rows = fixture.indexed_rows() as f64;
    println!(
        "nprobes={} L={} -> recall@{K}={:.4}, {:.0} comparisons ({:.1}% of {rows} rows), \
         {:.1} partitions",
        search.nprobes,
        search.search_list_size,
        measured.recall,
        measured.comparisons,
        100.0 * measured.comparisons / rows,
        measured.partitions
    );

    // Bars pinned to the measured pair rather than set loosely around it. The
    // build is seeded and the queries are fixed, so both numbers are stable; a
    // bar of "recall >= 0.95, cost < a quarter of the dataset" would have let
    // cost regress by half while still reading as a specification.
    assert!(
        measured.recall >= 0.98,
        "recall@{K} was {:.4}, measured at 0.9925",
        measured.recall
    );
    assert!(
        (1150.0..1500.0).contains(&measured.comparisons),
        "a query cost {:.0} comparisons, measured at 1313 ({:.1}% of {rows} rows)",
        measured.comparisons,
        100.0 * measured.comparisons / rows
    );
}

/// Search parameters that describe no search at all. Each of these guards was
/// removable without any test noticing.
#[tokio::test]
async fn search_parameters_that_describe_nothing_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let query = random_vectors(1, 7)[0].clone();

    for (params, expected) in [
        (SearchParams::new(0), "k must be greater than zero"),
        (SearchParams::new(K).with_nprobes(0), "nprobes must be"),
        (
            SearchParams::new(K).with_search_list_size(K - 1),
            "smaller than k",
        ),
        (
            SearchParams::new(K).with_search_list_size(0),
            "smaller than k",
        ),
    ] {
        let error = index.search(&query, &params).await.unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

/// The partition file and the segment table are two files, and only one of them
/// is read to decide how a walk is laid out. A partition whose width disagrees
/// with the table would be searched with a query of the wrong length against a
/// store that takes its dimension from the array - silently wrong distances
/// rather than an error - so the reader checks the pair on the way back in.
///
/// Both halves of the pair, because they fail differently and the guard is one
/// `if`: a wider neighbour list shifts every vertex's slot, a wider vector
/// shifts every coordinate.
#[tokio::test]
async fn a_partition_disagreeing_with_its_segment_is_refused() {
    /// The same vertices and row ids, one neighbour slot wider.
    fn widen_the_graph(partition: Partition) -> Partition {
        let (graph, vectors) = partition.into_parts();
        let widened = lance_vamana::partition::PartitionGraph::try_new(
            graph.max_degree() + 1,
            graph.row_ids().to_vec(),
            (0..graph.len())
                .map(|vertex| graph.neighbors(vertex as u32).unwrap().to_vec())
                .collect(),
        )
        .unwrap();
        Partition::try_new(widened, vectors).unwrap()
    }

    /// The same graph over vectors of one more coordinate.
    fn widen_the_vectors(partition: Partition) -> Partition {
        let (graph, vectors) = partition.into_parts();
        let dimension = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<Float32Type>().values();
        let wider = (0..vectors.len())
            .flat_map(|row| {
                values[row * dimension..(row + 1) * dimension]
                    .iter()
                    .copied()
                    .chain(std::iter::once(0.0))
            })
            .collect::<Vec<_>>();
        Partition::try_new(
            graph,
            <FixedSizeListArray as lance_arrow::FixedSizeListArrayExt>::try_new_from_values(
                Float32Array::from(wider),
                dimension as i32 + 1,
            )
            .unwrap(),
        )
        .unwrap()
    }

    for (what, doctor) in [
        (
            "a slot wider",
            widen_the_graph as fn(Partition) -> Partition,
        ),
        (
            "a coordinate wider",
            widen_the_vectors as fn(Partition) -> Partition,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = small_fixture().write(uri).await;

        let covered = (0..dataset.get_fragments().len() as u32).collect::<Vec<_>>();
        let uuid = Uuid::new_v4();
        let store = dataset.object_store(None).await.unwrap();
        let segment_dir = dataset.indices_dir().join(uuid.to_string());
        let centroids =
            <FixedSizeListArray as lance_arrow::FixedSizeListArrayExt>::try_new_from_values(
                Float32Array::from(vec![0.5f32; VECTOR_DIM as usize]),
                VECTOR_DIM,
            )
            .unwrap();
        let mut writer = SegmentWriter::new(
            store.clone(),
            segment_dir.clone(),
            declaring(covered.clone()),
            lance_index::vector::ivf::storage::IvfModel::new(centroids, None),
        );
        let declared = sample_partition(16, 8, VECTOR_DIM as u32);
        writer.write_partition(0, 0, &declared).await.unwrap();
        let manifest = writer.finish().await.unwrap();

        // Written over the file the segment already holds, and before the
        // commit, so the file sizes Lance records are the real ones and the
        // reader reaches the check rather than a truncated footer.
        lance_vamana::io::write_partition(
            &store,
            &segment_dir.join(manifest.partitions()[0].file.as_str()),
            &doctor(declared),
        )
        .await
        .unwrap();

        let details = prost_types::Any {
            type_url: INDEX_DETAILS_TYPE_URL.to_string(),
            value: Vec::new(),
        };
        dataset
            .commit_existing_index_segments(
                INDEX_NAME,
                VECTOR_COLUMN,
                vec![IndexSegment::new(
                    uuid,
                    covered,
                    [dataset.schema().field(VECTOR_COLUMN).unwrap().id],
                    Arc::new(details),
                    FORMAT_VERSION as i32,
                    dataset.manifest.version,
                )],
            )
            .await
            .unwrap();

        // The segment itself is well formed, so `open` has nothing to object to.
        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        let error = index
            .search(
                &random_vectors(1, 7)[0],
                &SearchParams::new(K).with_search_list_size(BEAM),
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("its segment declares"),
            "a partition {what} than its segment was searched anyway: {error}"
        );
    }
}

/// A query nothing can be measured from must be refused, not answered.
///
/// Every distance against a non-finite query is NaN, every ordering on this path
/// goes through `total_cmp`, and a negative NaN sorts ahead of negative infinity
/// - so the walk would return `k` arbitrary rows carrying a NaN distance, and a
/// caller filtering on that distance would accept all of them. Under cosine a
/// zero-length query does the same thing by a different road: normalising it
/// divides by zero.
#[tokio::test]
async fn a_query_that_no_distance_can_be_measured_from_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let search = SearchParams::new(K).with_search_list_size(BEAM);

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut query = random_vectors(1, 7)[0].clone();
        query[3] = bad;
        let error = index.search(&query, &search).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no distance can be measured from"),
            "{bad}: {error}"
        );
    }

    // The zero query is only refused under cosine: under L2 it is an ordinary
    // point of the space, and answering it is correct.
    let zero = vec![0.0f32; VECTOR_DIM as usize];
    index.search(&zero, &search).await.unwrap();

    let cosine_dir = tempfile::tempdir().unwrap();
    let cosine_uri = cosine_dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(cosine_uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &params().with_distance_type(DistanceType::Cosine),
    )
    .await
    .unwrap();
    let cosine = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let error = cosine.search(&zero, &search).await.unwrap_err();
    assert!(error.to_string().contains("squared length of 0"), "{error}");

    // Both ends of the range reach a norm that cannot be divided by, from
    // components a component-wise check waves through. Underflow gives a norm of
    // exactly zero and then a NaN; overflow gives an infinite one, and dividing
    // by that yields a query of *zeroes*, which under cosine sits at distance
    // exactly 1.0 from every vertex in the index - `k` arbitrary rows with a
    // plausible distance and no error anywhere.
    for (query, what) in [
        (vec![1e-30f32; VECTOR_DIM as usize], "underflow"),
        (vec![1e20f32; VECTOR_DIM as usize], "overflow"),
    ] {
        assert!(
            query.iter().all(|value| value.is_finite()),
            "{what}: the components must be finite, or the guard above catches this instead"
        );
        let error = cosine.search(&query, &search).await.unwrap_err();
        assert!(
            error.to_string().contains("squared length of"),
            "{what}: {error}"
        );
    }
}

/// Cosine is stored differently from every other metric - the builder normalises
/// the vectors it writes - and it is routed differently too, by L2 over those
/// unit vectors, because the router panics on cosine. Neither detour is visible
/// from the outside, so the only way to know they compose is to ask Lance.
#[tokio::test]
async fn a_cosine_index_matches_lance_cosine_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = measurement_fixture();
    let mut dataset = fixture.write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &params().with_distance_type(DistanceType::Cosine),
    )
    .await
    .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.metadata().distance_type, DistanceType::Cosine);

    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let queries = random_vectors(QUERIES, 4242);
    let mut total = 0.0;
    let mut comparisons = 0u64;
    for query in &queries {
        let key = Float32Array::from(query.clone());
        let mut scanner = dataset.scan();
        scanner.nearest(VECTOR_COLUMN, &key, K).unwrap();
        scanner.distance_metric(DistanceType::Cosine);
        scanner.use_index(false);
        scanner.with_row_id();
        let exact = scanner.try_into_batch().await.unwrap()[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();

        let result = index.search(query, &search).await.unwrap();
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        total += recall(&found, &exact);
        comparisons += result.comparisons;

        // Ranking alone cannot tell cosine from L2 here: the builder stores unit
        // vectors, and for `‖u‖ = 1` the value `‖u − q‖²` is monotone in `u · q`,
        // so both metrics produce the same order and the same recall. Only the
        // distance *value* separates them - checked against the definition,
        // recomputed from the row the answer names.
        let taken = dataset
            .take_rows(
                &found,
                ProjectionRequest::from_columns(
                    [VECTOR_COLUMN, lance_core::ROW_ID],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        let row_ids = taken[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let vectors = taken[VECTOR_COLUMN].as_fixed_size_list();
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<Float32Type>().values();
        let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();

        for neighbor in &result.neighbors {
            let row = row_ids
                .iter()
                .position(|id| *id == neighbor.row_addr)
                .expect("the answer named a row the dataset does not have");
            let stored = &values[row * dim..(row + 1) * dim];
            let dot = stored
                .iter()
                .zip(query)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let norm = stored.iter().map(|value| value * value).sum::<f32>().sqrt();
            let expected = 1.0 - dot / (norm * query_norm);
            assert!(
                (neighbor.distance - expected).abs() < 1e-5,
                "row {} came back at distance {} but its cosine distance is {expected}",
                neighbor.row_addr,
                neighbor.distance
            );
        }
    }
    let recall = total / queries.len() as f64;
    let comparisons = comparisons as f64 / queries.len() as f64;
    println!("cosine -> recall@{K}={recall:.4}, {comparisons:.0} comparisons");

    assert!(recall >= 0.95, "cosine recall@{K} was {recall:.4}");
    // A cosine index is routed and stored differently from every other metric,
    // so it gets a cost bar of its own rather than sharing L2's. Recall alone
    // would pass just as happily for a driver that opened every partition.
    assert!(
        (1150.0..1500.0).contains(&comparisons),
        "a cosine query cost {comparisons:.0} comparisons, measured at 1281"
    );
}

/// `Comparisons` holds a `Cell`, so it is `!Sync`, and a reference to one alive
/// across an `.await` would make this future `!Send`. No ordinary test would
/// notice: `#[tokio::test]` defaults to a single-threaded runtime that never
/// asks. `tokio::spawn` does ask.
#[tokio::test(flavor = "multi_thread")]
async fn a_search_can_be_spawned_onto_another_thread() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = std::sync::Arc::new(VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap());

    let query = random_vectors(1, 8)[0].clone();
    let found = tokio::spawn(async move {
        index
            .search(&query, &SearchParams::new(K).with_search_list_size(BEAM))
            .await
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(found.neighbors.len(), K);
}

/// Routing is a trade, and both halves of it have to be visible. A driver that
/// quietly opened every partition would still pass a recall bar.
#[tokio::test]
async fn a_narrow_probe_costs_recall_and_buys_work() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &measurement_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let base = SearchParams::new(K).with_search_list_size(BEAM);
    let narrow = measure(&index, &queries, &truth, &base.clone().with_nprobes(1)).await;
    let wide = measure(
        &index,
        &queries,
        &truth,
        &base.with_nprobes(PARTITIONS as usize),
    )
    .await;
    println!(
        "nprobes=1  -> recall={:.4}, {:.0} comparisons, {:.1} partitions\n\
         nprobes={PARTITIONS} -> recall={:.4}, {:.0} comparisons, {:.1} partitions",
        narrow.recall,
        narrow.comparisons,
        narrow.partitions,
        wide.recall,
        wide.comparisons,
        wide.partitions
    );

    assert!(narrow.partitions < wide.partitions);
    assert!(
        narrow.recall < wide.recall,
        "one probe scored as well as every probe ({:.4} against {:.4}); \
         either routing is not happening or the fixture is too easy to route",
        narrow.recall,
        wide.recall
    );
    // Without a floor this test cannot tell routing from a coin toss. With four
    // partitions, an assignment that ignored the vectors would put a quarter of
    // each query's true neighbours in the one partition read, so a broken router
    // scores about 0.25 here. The measured value is 0.5725.
    assert!(
        narrow.recall >= 0.45,
        "one probe recovered {:.4}, which is what an assignment that ignored the \
         vectors would score; the router is not routing",
        narrow.recall
    );
    assert!(
        narrow.comparisons < wide.comparisons / 2.0,
        "a narrow probe must actually save work: {:.0} against {:.0}",
        narrow.comparisons,
        wide.comparisons
    );
}

/// How many rows the biggest partition of the committed index holds.
async fn largest_partition(dataset: &Dataset) -> u64 {
    let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    let store = dataset.object_store(None).await.unwrap();
    let dir = dataset.indices_dir().join(indices[0].uuid.to_string());
    read_segment(&scan_scheduler(&store), &dir, None)
        .await
        .unwrap()
        .partitions()
        .iter()
        .map(|entry| u64::from(entry.num_rows))
        .max()
        .expect("the index committed no partitions")
}

/// Routing measures the query against *every* centroid a segment holds, and
/// pays for it whether or not a probe lands there. So a query that walks one
/// four-vertex partition of a 256-partition index costs at least 256
/// comparisons - where an accounting that counted only the walk would report
/// about ten, and a finely partitioned index would look free at exactly the
/// point it stops being.
///
/// Routing is a constant per index, so it cancels out of any difference between
/// two queries. An absolute lower bound is the only thing that can see it.
#[tokio::test]
async fn routing_is_charged_for_every_centroid_not_every_probe() {
    const CENTROIDS: u32 = 256;
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams {
            num_partitions: CENTROIDS,
            ..params()
        },
    )
    .await
    .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    // The cheapest walk this driver can be asked for: one probe, one neighbour,
    // a search list of one.
    let result = index
        .search(
            &random_vectors(1, 7)[0],
            &SearchParams::new(1).with_nprobes(1),
        )
        .await
        .unwrap();
    println!(
        "{CENTROIDS} centroids, one probe, k=1 -> {} comparisons over {} partitions",
        result.comparisons, result.partitions_read
    );

    assert_eq!(result.partitions_read, 1);
    assert!(
        result.comparisons >= u64::from(CENTROIDS),
        "a query paid {} comparisons, but routing alone measures {CENTROIDS} centroids",
        result.comparisons
    );
    // And a ceiling, because a floor alone makes over-counting free: doubling
    // every charge would still clear it. The walk is charged once per vertex it
    // measures and `SearchScratch` keeps it from measuring one twice, so a walk
    // cannot cost more than its partition holds.
    //
    // Taken from the largest partition that was actually written rather than
    // from the average one: k-means does not divide evenly, and a ceiling built
    // on `rows / centroids` would fail the day a probe landed on a big cell -
    // on CI, without a line of this crate having changed.
    let partition_rows = largest_partition(&dataset).await;
    assert!(
        result.comparisons <= u64::from(CENTROIDS) + partition_rows,
        "a query paid {} comparisons, more than routing {CENTROIDS} plus everything \
         one small partition could hold",
        result.comparisons
    );
}

/// The row ids we return must fetch the vectors we claimed distances for. This
/// is what a mixed-up local id looks like from the outside, and it survives both
/// the commit and a recall bar.
#[tokio::test]
async fn every_answer_resolves_to_the_row_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    for query in random_vectors(8, 99) {
        let result = index.search(&query, &search).await.unwrap();
        let row_ids = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        assert_eq!(
            row_ids.len(),
            K,
            "an index that answered nothing would satisfy every check below"
        );
        assert_eq!(
            row_ids.iter().collect::<HashSet<_>>().len(),
            row_ids.len(),
            "the same row was returned twice, so partitions overlap or the merge is wrong"
        );

        let taken = dataset
            .take_rows(
                &row_ids,
                ProjectionRequest::from_columns(
                    [VECTOR_COLUMN, lance_core::ROW_ID],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        let fetched = taken[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let vectors = taken[VECTOR_COLUMN].as_fixed_size_list();
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<Float32Type>().values();

        for neighbor in &result.neighbors {
            // Joined on `_rowid`: `take_rows` neither preserves nor reports the
            // positions it dropped.
            let row = fetched
                .iter()
                .position(|id| *id == neighbor.row_addr)
                .expect("a returned row id is not in the dataset");
            let stored = &values[row * dim..(row + 1) * dim];
            let distance = stored
                .iter()
                .zip(&query)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            assert!(
                (distance - neighbor.distance).abs() < 1e-4,
                "row {} was reported at distance {} but is at {distance}",
                neighbor.row_addr,
                neighbor.distance
            );
        }
    }
}

/// A compaction that could not open the index leaves it naming fragments that no
/// longer exist, and every row address it stored for them is dead. It answers
/// for none of them, and says so.
///
/// The rows are not lost with them: a compaction moves them to fragments this
/// index does not cover, which is the same position as rows appended after the
/// build. What the index must not do is hand back the addresses they used to be
/// at - so the query here has to reach the partitions and come back empty,
/// rather than be short-circuited by an index that knows it covers nothing.
///
/// The compaction is real here, and asserted to be: deleting every row first
/// would drop the fragments outright and the test would pass without compacting
/// anything at all.
#[tokio::test]
async fn an_index_over_a_rewritten_fragment_answers_for_none_of_it() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;
    let built_over = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .covered_fragments()
        .clone();
    assert!(!built_over.is_empty(), "the index covered nothing to start");

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert!(
        metrics.fragments_removed > 0,
        "nothing was compacted, so this test proves nothing"
    );
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| !built_over.contains(fragment.id() as u32)),
        "the compaction left an indexed fragment behind, so this test proves less than it says"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert!(
        index.covered_fragments().is_empty(),
        "the index still claims {:?} after every fragment it read was rewritten",
        index.covered_fragments()
    );

    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let result = index
        .search(&random_vectors(1, 4242)[0], &search)
        .await
        .unwrap();
    assert!(
        result.partitions_read > 0,
        "no partition was read, so nothing was filtered and this proves nothing"
    );
    assert!(
        result.neighbors.is_empty(),
        "the index answered with {} rows from fragments the dataset no longer has",
        result.neighbors.len()
    );
}

/// Retention is an ordinary reason for a fragment to disappear: `DELETE WHERE
/// date < ...` over data laid out by time empties the early fragments, and Lance
/// drops a fragment from the manifest once its last row is deleted. The index
/// has to go on answering from the fragments that are left.
///
/// The same query is run before the delete and after it, and the assertions are
/// the two halves of one claim: rows of the doomed fragment are what this query
/// used to get back, and none of them comes back once the fragment is gone. The
/// first half is what keeps the second from passing vacuously - the vertices are
/// still in the partition files either way, so the filter is the only thing
/// standing between the walk and a dead address.
#[tokio::test]
async fn an_index_over_a_deleted_fragment_answers_from_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;

    let query = random_vectors(1, 77)[0].clone();
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let before = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .unwrap()
        .search(&query, &search)
        .await
        .unwrap();
    assert!(
        before
            .neighbors
            .iter()
            .any(|neighbor| RowAddress::from(neighbor.row_addr).fragment_id() == 1),
        "this query never reached fragment 1, so deleting it would prove nothing"
    );

    // Every row of fragment 1 and nothing else: an address is
    // `(fragment << 32) | offset`, so the fragment's rows are a contiguous run
    // above its first address, and the fixture has no fragment above it.
    let first_row_of_fragment = u64::from(RowAddress::new_from_parts(1, 0));
    dataset
        .delete(&format!("_rowid >= {first_row_of_fragment}"))
        .await
        .unwrap();
    assert_eq!(
        dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id())
            .collect::<Vec<_>>(),
        vec![0],
        "the delete was meant to take fragment 1 out of the manifest whole"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index.covered_fragments(),
        &RoaringBitmap::from_iter([0u32]),
        "the index should cover the fragment that is left, and only it"
    );

    let after = index.search(&query, &search).await.unwrap();
    assert_eq!(
        after.neighbors.len(),
        K,
        "the surviving fragment holds 512 rows, so a k of {K} is still answerable"
    );
    assert!(
        after
            .neighbors
            .iter()
            .all(|neighbor| RowAddress::from(neighbor.row_addr).fragment_id() == 0),
        "an answer came back at an address in the fragment the dataset dropped"
    );
}

/// Rewrite one fragment's vector column in place, exactly as `update_columns`
/// does: a new data file inside the *same* fragment, the old one tombstoned, and
/// every row address left where it was.
async fn rewrite_vector_column_in_place(dataset: &Dataset, uri: &str, fragment_id: u64) -> Dataset {
    let mut fragment = dataset.get_fragment(fragment_id as usize).unwrap();
    let mut scan = fragment.scan();
    scan.with_row_id();
    scan.project::<&str>(&[]).unwrap();
    let row_ids = scan.try_into_batch().await.unwrap()[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();

    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let update_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(lance_core::ROW_ID, DataType::UInt64, false),
        Field::new(
            VECTOR_COLUMN,
            DataType::FixedSizeList(item, VECTOR_DIM),
            true,
        ),
    ]));
    // Far from anything the fixture drew, so an index answering from the stored
    // copy and one answering from the new data cannot be confused.
    let fresh = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        row_ids
            .iter()
            .map(|_| Some(vec![Some(9999.0f32); VECTOR_DIM as usize]))
            .collect::<Vec<_>>(),
        VECTOR_DIM,
    );
    let update_batch = RecordBatch::try_new(
        update_schema.clone(),
        vec![Arc::new(UInt64Array::from(row_ids)), Arc::new(fresh)],
    )
    .unwrap();
    let right: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(update_batch)],
        update_schema,
    ));

    let updated = fragment
        .update_columns_with_offsets(right, lance_core::ROW_ID, lance_core::ROW_ID)
        .await
        .unwrap();
    let updated_fragment_id = updated.fragment.id;
    Dataset::commit(
        uri,
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![updated.fragment],
            new_fragments: vec![],
            fields_modified: updated.fields_modified,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(HashMap::from([(
                updated_fragment_id,
                updated.matched_offsets,
            )]))),
        },
        Some(dataset.version().version),
        None,
        None,
        Default::default(),
        true,
    )
    .await
    .unwrap()
}

/// The rewrite that no liveness check can see.
///
/// `update_columns` keeps the fragment id and every row address, so the fragment
/// is still live and the index's addresses still resolve - to rows whose vectors
/// have been replaced. The only signal Lance emits is pruning the fragment out of
/// the index's `fragment_bitmap`, which shrinks the coverage rather than the
/// dataset, and a guard that compares coverage against the *dataset* sees nothing
/// at all. Comparing it against what the segment was built from is what catches it.
#[tokio::test]
async fn an_index_over_a_rewritten_column_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let dataset = rewrite_vector_column_in_place(&dataset, uri, 0).await;
    assert!(
        dataset.get_fragments().iter().any(|f| f.id() == 0),
        "the rewritten fragment must still be live, or the liveness guard would \
         catch this and the test would prove nothing"
    );

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("an index holding vectors that were overwritten must not answer");
    assert!(
        error.to_string().contains("rewrote data under it"),
        "{error}"
    );
}

/// The two ways coverage narrows, one after the other: Lance takes a fragment
/// out of the bitmap when its column is rewritten in place, and the fragment
/// itself goes when its last row is deleted.
///
/// While it was live, the narrowed bitmap was the only sign that the vectors
/// stored for it were stale, and the index refused to open. Once the fragment is
/// gone there is nothing stale left to serve: no address in it resolves to
/// anything, so the index goes back to answering from the rest. This is what the
/// liveness half of that guard is for, and the only way to reach it.
#[tokio::test]
async fn a_rewritten_fragment_that_is_then_deleted_stops_being_a_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;

    let mut dataset = rewrite_vector_column_in_place(&dataset, uri, 0).await;
    VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("a live fragment whose column was rewritten must still be a refusal");

    let first_row_of_fragment = u64::from(RowAddress::new_from_parts(1, 0));
    dataset
        .delete(&format!("_rowid < {first_row_of_fragment}"))
        .await
        .unwrap();
    assert_eq!(
        dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id())
            .collect::<Vec<_>>(),
        vec![1],
        "the delete was meant to take the rewritten fragment out of the manifest whole"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(
        index.covered_fragments(),
        &RoaringBitmap::from_iter([1u32]),
        "the index should cover the fragment that was never touched, and only it"
    );
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let result = index
        .search(&random_vectors(1, 313)[0], &search)
        .await
        .unwrap();
    assert!(
        !result.neighbors.is_empty()
            && result
                .neighbors
                .iter()
                .all(|neighbor| RowAddress::from(neighbor.row_addr).fragment_id() == 1),
        "the index answered with {} rows, at least one of them in the fragment that is gone",
        result.neighbors.len()
    );
}

/// Replace one fragment's vectors with an overlay, the way Lance's own overlay
/// tests do: write a file holding the new values for the indexed field alone,
/// then commit `Operation::DataOverlay` naming the offsets it covers.
///
/// `committed_version` is stamped by the commit, not by this caller, so an
/// overlay is newer than every index built before it and older than every index
/// built after it - which is the whole basis of the version gate under test.
async fn commit_overlay(
    dataset: Dataset,
    fragment_id: u64,
    offsets: &[u32],
    name: &str,
) -> Dataset {
    let read_version = dataset.version().version;
    let field_id = dataset.schema().field(VECTOR_COLUMN).unwrap().id;
    let overlay_schema = dataset.schema().project_by_ids(&[field_id], true);

    // A constant vector, so an answer ranked on the pre-overlay values is
    // distinguishable from one ranked on these.
    let replacement = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        offsets
            .iter()
            .map(|_| Some(vec![Some(9.0f32); VECTOR_DIM as usize]))
            .collect::<Vec<_>>(),
        VECTOR_DIM,
    );

    let file = format!("{name}.lance");
    let store = dataset.object_store(None).await.unwrap();
    let mut writer = create_writer(
        ConcreteFileVersion::V2_1,
        store
            .create(&dataset.data_dir().join(file.as_str()))
            .await
            .unwrap(),
        overlay_schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_column(0, Arc::new(replacement)).await.unwrap();
    let summary = writer.finish().await.unwrap();

    let mut data_file = DataFile::new_unstarted(file, ConcreteFileVersion::V2_1);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(field_id, _)| *field_id as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, column_index)| *column_index as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id,
                overlays: vec![DataOverlayFile {
                    data_file,
                    coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter(
                        offsets.iter().copied(),
                    ))),
                    committed_version: 0,
                }],
            }],
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

/// The rewrite that leaves *every* coverage record intact.
///
/// `Operation::DataOverlay` touches fragments and never indices: the fragment
/// ids, the index's `fragment_bitmap` and the segment's own record of what it
/// read all come through unchanged, so both coverage guards pass while the
/// values at those addresses have been replaced. Ranking would run on the
/// pre-overlay vectors and `take_rows` would hand back the post-overlay ones.
///
/// Lance keeps its own indices usable here by masking the stale rows per query;
/// that path is `pub(crate)` and lives in the scanner this driver bypasses.
#[tokio::test]
async fn an_index_whose_vectors_an_overlay_replaced_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let dataset = commit_overlay(dataset, 0, &[0, 1, 2], "stale").await;
    let index = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(
        index[0].fragment_bitmap.as_ref().map(|b| b.len()),
        Some(small_fixture().fragments as u64),
        "the overlay must leave the coverage untouched, or an existing guard \
         would catch this and the test would prove nothing"
    );
    // The guard reads metadata, so the test would pass even if the overlay had
    // not landed in the data at all. Lance's own scan finding the replacement
    // vector at distance zero is what makes the refusal necessary rather than
    // merely triggered.
    let replacement = vec![9.0f32; VECTOR_DIM as usize];
    assert_eq!(
        brute_force_best_distance(&dataset, &replacement).await,
        0.0,
        "the overlay never reached the data, so an index answering from the \
         pre-overlay vectors would still be right and this test proves nothing"
    );

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("an index ranking by vectors an overlay replaced must not answer");
    assert!(
        error.to_string().contains("replaced by an overlay"),
        "{error}"
    );
}

/// A shallow clone inherits every index by reference, stamping a base id onto it
/// so the files resolve against the *source* dataset's root. This driver computes
/// the directory itself, from the clone's root, because the helpers that resolve
/// a base id are `pub(crate)` - so it would read a path that does not exist while
/// the recorded file sizes still described the real ones.
#[tokio::test]
async fn an_index_inherited_by_a_shallow_clone_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;

    let clone_dir = tempfile::tempdir().unwrap();
    let clone_uri = clone_dir.path().to_str().unwrap();
    let version = dataset.version().version;
    let clone = dataset
        .shallow_clone(clone_uri, version, None)
        .await
        .unwrap();

    let inherited = clone.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert!(
        inherited[0].base_id.is_some(),
        "the clone did not stamp a base id, so this test is not exercising one"
    );

    let error = VamanaIndex::open(&clone, INDEX_NAME)
        .await
        .expect_err("an index whose files live under another dataset's root must not open");
    assert!(error.to_string().contains("base path"), "{error}");
}

/// The other side of the version gate: an overlay committed *before* the index
/// was built is already in the vectors the segment holds, because the build read
/// the column through the ordinary scanner. Refusing here would make the index
/// unbuildable on any dataset that had ever been overlaid.
#[tokio::test]
async fn an_index_built_over_an_overlay_opens() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = small_fixture().write(uri).await;
    let mut dataset = commit_overlay(dataset, 0, &[0, 1, 2], "settled").await;

    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect("an overlay the build already read is not stale");
}

async fn live_row_ids(dataset: &Dataset) -> HashSet<u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project::<&str>(&[]).unwrap();
    scanner.try_into_batch().await.unwrap()[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect()
}

/// Deleting a row does not touch the index: the vertex, its edges and its vector
/// all stay in the partition file, and its address still decodes. The delete list
/// is the only thing standing between it and the answer.
///
/// Checked against Lance's own brute force over the *same* post-delete dataset,
/// so this is not just "no deleted row came back" - it is also "the live rows the
/// deleted ones used to displace came back instead".
#[tokio::test]
async fn deleted_rows_are_not_returned() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;

    dataset.delete("_rowid % 7 == 0").await.unwrap();
    let deleted_count = dataset.count_deleted_rows().await.unwrap();
    assert!(deleted_count > 0, "the fixture deleted nothing");
    let touched = dataset
        .get_fragments()
        .iter()
        .filter(|fragment| fragment.metadata().deletion_file.is_some())
        .count();
    assert!(touched > 1, "deletions must span several fragments");

    let live = live_row_ids(&dataset).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);

    let queries = random_vectors(QUERIES, 909);
    let mut total_recall = 0.0;
    for query in &queries {
        let result = index.search(query, &search).await.unwrap();
        for neighbor in &result.neighbors {
            assert!(
                live.contains(&neighbor.row_addr),
                "a deleted row was returned: {}",
                neighbor.row_addr
            );
        }
        assert_eq!(
            result.neighbors.len(),
            K,
            "the delete list cost the query rows it could have filled"
        );
        let found = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<Vec<_>>();
        total_recall += recall(&found, &brute_force(&dataset, query, K).await);
    }
    let recall = total_recall / queries.len() as f64;
    println!("recall@{K} over a dataset with {deleted_count} deleted rows = {recall:.4}");
    assert!(recall >= 0.95, "recall@{K} was {recall:.4}");
}

/// The stated boundary, pinned: the delete list is a snapshot taken at open.
///
/// Worth a test rather than only a doc line, because the two behaviours are
/// indistinguishable from the answer alone - a stale list returns rows that look
/// exactly like live ones until the caller tries to fetch them.
#[tokio::test]
async fn the_delete_list_is_a_snapshot_taken_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let query = random_vectors(1, 77).remove(0);
    let before = index.search(&query, &search).await.unwrap();

    // Delete exactly what that query just returned.
    let doomed = before
        .neighbors
        .iter()
        .map(|neighbor| neighbor.row_addr.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    dataset
        .delete(&format!("_rowid in ({doomed})"))
        .await
        .unwrap();

    let stale = index.search(&query, &search).await.unwrap();
    assert_eq!(
        stale.neighbors, before.neighbors,
        "an index opened before the delete must keep answering from its snapshot"
    );

    let reopened = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let fresh = reopened.search(&query, &search).await.unwrap();
    let gone = before
        .neighbors
        .iter()
        .map(|neighbor| neighbor.row_addr)
        .collect::<HashSet<_>>();
    assert!(
        fresh.neighbors.iter().all(|n| !gone.contains(&n.row_addr)),
        "reopening must pick up the deletions"
    );
}

/// The mirror image, and the reason the guard tests equality rather than subset.
///
/// Build a segment over `built_over`, then commit it under a description the
/// caller chooses. The coverage and the version Lance records come from here
/// rather than from the builder, which is the only way to make a segment and its
/// manifest entry disagree on purpose.
async fn commit_a_segment_described_as(
    dataset: &mut Dataset,
    built_over: &[u32],
    coverage: &[u32],
    version: i32,
) {
    let uuid = Uuid::new_v4();
    let segment_dir = dataset.indices_dir().join(uuid.to_string());
    build_segment(dataset, &params(), &segment_dir, built_over)
        .await
        .unwrap();

    let field_id = dataset.schema().field(VECTOR_COLUMN).unwrap().id;
    let details = prost_types::Any {
        type_url: INDEX_DETAILS_TYPE_URL.to_string(),
        value: Vec::new(),
    };
    let described = IndexSegment::new(
        uuid,
        coverage.to_vec(),
        [field_id],
        Arc::new(details),
        version,
        dataset.manifest.version,
    );
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![described])
        .await
        .unwrap();
}

/// Lance does not only shrink an index's coverage - `Transaction::
/// register_pure_rewrite_rows_update_frags_in_indices` adds fragments *back*
/// into the bitmap after a pure row rewrite, and it skips only the indices it
/// recognises as address-domain, which an out-of-tree type is not. Coverage that
/// grew is a claim to hold rows the segment never read, and a subset test would
/// wave it through.
#[tokio::test]
async fn an_index_credited_with_a_fragment_it_never_read_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;
    assert!(dataset.get_fragments().len() >= 2);

    commit_a_segment_described_as(&mut dataset, &[0], &[0, 1], FORMAT_VERSION as i32).await;

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("a segment credited with rows it never read must not answer");
    assert!(error.to_string().contains("credits it with 2"), "{error}");
}

/// Write a segment that declares exactly `metadata`, and describe it ready to
/// commit.
///
/// The one partition it holds matches the declaration in width and degree, and
/// the routing model matches its dimension, so the only thing wrong with the
/// segment is whatever the caller put in the metadata. That is what makes it the
/// way to test a refusal that no build can produce.
async fn hand_made_segment(dataset: &Dataset, metadata: IndexMetadata) -> IndexSegment {
    let uuid = Uuid::new_v4();
    let covered = metadata.fragments.clone();
    let centroids =
        <FixedSizeListArray as lance_arrow::FixedSizeListArrayExt>::try_new_from_values(
            Float32Array::from(vec![0.5f32; metadata.dimension as usize]),
            metadata.dimension as i32,
        )
        .unwrap();
    let partition = sample_partition(metadata.max_degree, 8, metadata.dimension);
    let mut writer = SegmentWriter::new(
        dataset.object_store(None).await.unwrap(),
        dataset.indices_dir().join(uuid.to_string()),
        metadata,
        lance_index::vector::ivf::storage::IvfModel::new(centroids, None),
    );
    writer.write_partition(0, 0, &partition).await.unwrap();
    writer.finish().await.unwrap();

    IndexSegment::new(
        uuid,
        covered,
        [dataset.schema().field(VECTOR_COLUMN).unwrap().id],
        Arc::new(prost_types::Any {
            type_url: INDEX_DETAILS_TYPE_URL.to_string(),
            value: Vec::new(),
        }),
        FORMAT_VERSION as i32,
        dataset.manifest.version,
    )
}

/// A segment claiming stable row ids has to be refused on its own account, not
/// only through the dataset's setting. The builder will not produce one, so this
/// half of the check has never run - and the two identifier spaces are not
/// distinguishable from a stored id, so getting it wrong is silent: the delete
/// list is built from deletion vectors, which are always addresses, and applying
/// it to logical ids would filter live rows and return deleted ones.
#[tokio::test]
async fn an_index_built_for_stable_row_ids_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;
    let covered = (0..dataset.get_fragments().len() as u32).collect::<Vec<_>>();

    let segment = hand_made_segment(
        &dataset,
        IndexMetadata {
            row_id_mode: RowIdMode::Stable,
            ..declaring(covered)
        },
    )
    .await;
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![segment])
        .await
        .unwrap();

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("a segment in the wrong identifier space must not answer");
    assert!(error.to_string().contains("Stable"), "{error}");
}

/// What a segment of this fixture ordinarily declares. Every refusal below is
/// one field of it changed.
fn declaring(fragments: Vec<u32>) -> IndexMetadata {
    IndexMetadata {
        format_version: FORMAT_VERSION,
        max_degree: 16,
        search_list_size: 32,
        alpha: 1.2,
        dimension: VECTOR_DIM as u32,
        distance_type: DistanceType::L2,
        row_id_mode: RowIdMode::Address,
        fragments,
    }
}

/// The metric is refused on open as well as on the build path. The two are
/// separate doors into the same crate - a segment can be written by an older
/// build, or by hand - and only the build one was ever tried.
#[tokio::test]
async fn an_index_declaring_an_unsupported_metric_is_refused_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;
    let covered = (0..dataset.get_fragments().len() as u32).collect::<Vec<_>>();

    let segment = hand_made_segment(
        &dataset,
        IndexMetadata {
            distance_type: DistanceType::Dot,
            ..declaring(covered)
        },
    )
    .await;
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![segment])
        .await
        .unwrap();

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("a segment built under a metric this crate cannot rank by must not answer");
    assert!(error.to_string().contains("dot distance"), "{error}");
}

/// A query mixes the answers of every segment, so the segments have to agree on
/// what an answer means. Degree and pruning slack may differ - a segment
/// appended later is allowed a different graph - but the metric may not, and
/// nothing downstream of the merge would notice two distance scales.
#[tokio::test]
async fn segments_that_disagree_about_their_vectors_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;

    // The metric, through a real build on both sides and different in more than
    // the field under test - a wider graph and a different pruning slack too -
    // because degree and alpha are *allowed* to differ between segments and the
    // check must not be reading those.
    let (left, _) = build_index_segment(&dataset, &params(), &[0])
        .await
        .unwrap();
    let (right, _) = build_index_segment(
        &dataset,
        &params()
            .with_distance_type(DistanceType::Cosine)
            .with_graph_params(BuildParams {
                max_degree: 24,
                search_list_size: 64,
                alpha: 1.4,
                ..Default::default()
            }),
        &[1],
    )
    .await
    .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![left, right])
        .await
        .unwrap();

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("segments measuring distance differently must not be merged");
    assert!(
        error.to_string().contains("disagree about the vectors"),
        "{error}"
    );

    // The other two fields of the same check, which no build can disagree on -
    // the width comes from the column and the identifier space from the dataset,
    // so a segment that differs in either has to be written by hand.
    for (what, doctored) in [
        (
            "the width",
            IndexMetadata {
                dimension: VECTOR_DIM as u32 + 1,
                ..declaring(vec![1])
            },
        ),
        (
            "the identifier space",
            IndexMetadata {
                row_id_mode: RowIdMode::Stable,
                ..declaring(vec![1])
            },
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = small_fixture().write(uri).await;
        let agreeing = hand_made_segment(&dataset, declaring(vec![0])).await;
        let disagreeing = hand_made_segment(&dataset, doctored).await;
        dataset
            .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![agreeing, disagreeing])
            .await
            .unwrap();

        let error = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap_err();
        assert!(
            error.to_string().contains("disagree about the vectors"),
            "segments disagreeing about {what} were merged instead: {error}"
        );
    }

    // And the pair the check must stay quiet about, so that it is testing the
    // fields it names rather than "the two segments are not identical".
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;
    let left = hand_made_segment(&dataset, declaring(vec![0])).await;
    let right = hand_made_segment(
        &dataset,
        IndexMetadata {
            max_degree: 24,
            alpha: 1.4,
            ..declaring(vec![1])
        },
    )
    .await;
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![left, right])
        .await
        .unwrap();
    VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect("a segment appended with a different graph is allowed to differ");
}

/// The format version lives in two places - the dataset manifest and the
/// segment's own metadata - and the manifest's copy is the one a reader meets
/// first. A segment written by a later build has to be turned away there, before
/// any of its files are opened and misread.
#[tokio::test]
async fn an_index_at_another_format_version_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = small_fixture().write(uri).await;

    commit_a_segment_described_as(&mut dataset, &[0, 1], &[0, 1], FORMAT_VERSION as i32 + 1).await;

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("a segment from a later build must not be read by this one");
    assert!(
        matches!(error, lance_core::Error::NotSupported { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("format version"), "{error}");
}

/// The same guard must stay quiet for everything that does not rewrite data.
///
/// Appending fragments leaves the committed coverage exactly as it was, so an
/// index that refused to open after an append would be useless - and the guard
/// would be testing the dataset's shape rather than its own coverage.
#[tokio::test]
async fn appending_rows_leaves_the_index_open() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let before = dataset.get_fragments().len();

    let dataset = small_fixture().append(uri).await;
    assert!(
        dataset.get_fragments().len() > before,
        "the append added no fragments, so this test proves nothing"
    );

    let index = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect("an append must not invalidate an index");
    assert_eq!(index.num_segments(), 1);
}

/// More than one segment is the normal state of an index that has been extended,
/// and it is the only case where local ids from different graphs meet.
#[tokio::test]
async fn an_index_of_several_segments_answers_from_all_of_them() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = measurement_fixture();
    let mut dataset = fixture.write(uri).await;

    let (left, _) = build_index_segment(&dataset, &params(), &[0, 1])
        .await
        .unwrap();
    let (right, _) = build_index_segment(&dataset, &params(), &[2, 3])
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(INDEX_NAME, VECTOR_COLUMN, vec![left, right])
        .await
        .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    assert_eq!(index.num_segments(), 2);

    let queries = random_vectors(QUERIES, 4242);
    let truth = ground_truth(&dataset, &queries).await;
    let search = SearchParams::new(K)
        .with_nprobes(PARTITIONS as usize)
        .with_search_list_size(BEAM);
    let measured = measure(&index, &queries, &truth, &search).await;
    println!(
        "two segments -> recall@{K}={:.4}, {:.0} comparisons, {:.1} partitions",
        measured.recall, measured.comparisons, measured.partitions
    );

    // Every row is in exactly one segment, so the merge must never hand back the
    // same row twice, and it must reach the half that lives in the other one.
    for query in queries.iter().take(8) {
        let result = index.search(query, &search).await.unwrap();
        let ids = result
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_addr)
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), K, "the merge returned a row twice");
    }
    assert!(
        measured.recall >= 0.95,
        "recall across two segments was {:.4}",
        measured.recall
    );
    // Two segments cost two routing tables and two walks, so the bar is roughly
    // twice the single-segment one rather than the same number.
    assert!(
        (2100.0..2600.0).contains(&measured.comparisons),
        "a two-segment query cost {:.0} comparisons, measured at 2315",
        measured.comparisons
    );
    // Exactly four probes in each of the two segments, all eight read. This is
    // also more partitions than a query keeps reads in flight for, so anything
    // the buffering dropped past its depth would show up as a shortfall here.
    assert_eq!(
        measured.partitions,
        f64::from(PARTITIONS) * 2.0,
        "a two-segment index must probe both segments in full"
    );
}

#[tokio::test]
async fn an_absent_index_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = small_fixture().write(uri).await;

    let error = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap_err();
    assert!(error.to_string().contains("no index named"), "{error}");
}

#[tokio::test]
async fn a_query_of_the_wrong_width_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let error = index
        .search(&[0.0; 3], &SearchParams::new(K))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("3 dimensions"), "{error}");

    let error = index
        .search(
            &random_vectors(1, 1)[0],
            &SearchParams::new(K).with_search_list_size(K - 1),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("smaller than k"), "{error}");
}

/// Probing past the end of the routing table asks for every partition it has,
/// not for an error and not for a panic.
#[tokio::test]
async fn probing_past_the_end_of_the_table_is_clamped() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = indexed_dataset(uri, &small_fixture()).await;
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let query = &random_vectors(1, 7)[0];
    let all = index
        .search(
            query,
            &SearchParams::new(K)
                .with_nprobes(PARTITIONS as usize)
                .with_search_list_size(BEAM),
        )
        .await
        .unwrap();
    let beyond = index
        .search(
            query,
            &SearchParams::new(K)
                .with_nprobes(PARTITIONS as usize * 10)
                .with_search_list_size(BEAM),
        )
        .await
        .unwrap();
    assert_eq!(all.neighbors, beyond.neighbors);
    assert_eq!(
        all.partitions_read, beyond.partitions_read,
        "asking for ten times the partitions read a different number of them"
    );
    assert_eq!(
        all.partitions_read, PARTITIONS as usize,
        "the small fixture should populate every partition, so both arms read them all"
    );
}

/// Every row of the dataset, with the address it lives at.
async fn rows_with_vectors(dataset: &Dataset) -> (Vec<u64>, FixedSizeListArray) {
    let mut scanner = dataset.scan();
    scanner.project(&[VECTOR_COLUMN]).unwrap().with_row_id();
    let batch = scanner.try_into_batch().await.unwrap();
    (
        batch[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec(),
        batch[VECTOR_COLUMN].as_fixed_size_list().clone(),
    )
}

/// The rows at `positions`, as a column of their own.
fn gather_rows(vectors: &FixedSizeListArray, positions: &[usize]) -> FixedSizeListArray {
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        positions
            .iter()
            .map(|row| {
                Some(
                    vectors
                        .value(*row)
                        .as_primitive::<Float32Type>()
                        .values()
                        .iter()
                        .map(|value| Some(*value))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
        VECTOR_DIM,
    )
}

/// An empty partition has no row in the segment table and no file of its own,
/// but routing can still name it. Stepping over it is the normal case, and it
/// must not cost the probe budget either.
///
/// The segment is written by hand rather than built. `create_index` can only be
/// made to leave a partition empty by handing k-means more centroids than there
/// are distinct vectors, and that fixture pays for it twice: the empty clusters
/// are split by an **OS-seeded** RNG, so the build is the one thing in this
/// suite that does not reproduce, and every partition becomes a bag of exact
/// duplicates, so no assertion can tell which *vertex* came back.
#[tokio::test]
async fn a_probed_partition_that_holds_nothing_is_skipped() {
    const CENTROIDS: u32 = 4;
    const POPULATED: [u32; 2] = [1, 3];
    const BEAM_OVER_PARTITION: usize = 64;

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture {
        fragments: 2,
        rows_per_fragment: 64,
        ..Default::default()
    }
    .write(uri)
    .await;
    let (row_ids, vectors) = rows_with_vectors(&dataset).await;

    // The two populated centroids differ from each other in one coordinate only,
    // so "which centroid is nearest" and "which side of 0.5 the first coordinate
    // falls on" are the same question - the split the segment is written to is
    // the split routing will make. The other two sit far outside a dataset drawn
    // from the unit cube, so nothing is ever assigned to them and no ordinary
    // query routes to them first.
    let mut centroid_values = vec![0.0f32; CENTROIDS as usize * VECTOR_DIM as usize];
    let dimension = VECTOR_DIM as usize;
    centroid_values[..dimension].fill(5.0);
    centroid_values[dimension..2 * dimension].fill(0.5);
    centroid_values[dimension] = 0.25;
    centroid_values[2 * dimension..3 * dimension].fill(-5.0);
    centroid_values[3 * dimension..].fill(0.5);
    centroid_values[3 * dimension] = 0.75;
    let centroids =
        <FixedSizeListArray as lance_arrow::FixedSizeListArrayExt>::try_new_from_values(
            Float32Array::from(centroid_values),
            VECTOR_DIM,
        )
        .unwrap();

    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for row in 0..vectors.len() {
        let first = vectors.value(row).as_primitive::<Float32Type>().value(0);
        let partition = if first < 0.5 {
            POPULATED[0]
        } else {
            POPULATED[1]
        };
        members.entry(partition).or_default().push(row);
    }
    for partition in POPULATED {
        assert!(
            members[&partition].len() >= K,
            "partition {partition} holds too few rows to answer a k of {K}"
        );
    }

    let covered = (0..dataset.get_fragments().len() as u32).collect::<Vec<_>>();
    let uuid = Uuid::new_v4();
    let store = dataset.object_store(None).await.unwrap();
    let segment_dir = dataset.indices_dir().join(uuid.to_string());
    let mut writer = SegmentWriter::new(
        store,
        segment_dir,
        IndexMetadata {
            format_version: FORMAT_VERSION,
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
            dimension: VECTOR_DIM as u32,
            distance_type: DistanceType::L2,
            row_id_mode: RowIdMode::Address,
            fragments: covered.clone(),
        },
        lance_index::vector::ivf::storage::IvfModel::new(centroids, None),
    );
    // Ascending, because the writer refuses a partition id below the last one it
    // wrote - the segment table is what a probe is looked up in.
    for partition in POPULATED {
        let positions = &members[&partition];
        let taken = gather_rows(&vectors, positions);
        let member_row_ids = positions
            .iter()
            .map(|row| row_ids[*row])
            .collect::<Vec<_>>();
        let store =
            lance_vamana::search::flat_storage(&member_row_ids, &taken, DistanceType::L2).unwrap();
        let built = lance_vamana::build::build_partition(
            &store,
            &BuildParams {
                max_degree: 16,
                search_list_size: BEAM_OVER_PARTITION,
                ..Default::default()
            },
            &lance_vamana::search::Comparisons::default(),
        )
        .unwrap();
        let graph = Partition::try_new(built.graph, taken).unwrap();
        writer
            .write_partition(partition, built.medoid, &graph)
            .await
            .unwrap();
    }
    writer.finish().await.unwrap();

    let details = prost_types::Any {
        type_url: INDEX_DETAILS_TYPE_URL.to_string(),
        value: Vec::new(),
    };
    dataset
        .commit_existing_index_segments(
            INDEX_NAME,
            VECTOR_COLUMN,
            vec![IndexSegment::new(
                uuid,
                covered,
                [dataset.schema().field(VECTOR_COLUMN).unwrap().id],
                Arc::new(details),
                FORMAT_VERSION as i32,
                dataset.manifest.version,
            )],
        )
        .await
        .unwrap();
    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    // Asking for every centroid reads the two that hold something. The index
    // still covers the whole dataset, so the answer is the exhaustive one.
    let query = random_vectors(1, 5)[0].clone();
    let result = index
        .search(
            &query,
            &SearchParams::new(K)
                .with_nprobes(CENTROIDS as usize)
                .with_search_list_size(BEAM_OVER_PARTITION),
        )
        .await
        .unwrap();
    assert_eq!(
        result.partitions_read,
        POPULATED.len(),
        "an empty partition was read, or a populated one was not"
    );
    let found = result
        .neighbors
        .iter()
        .map(|neighbor| neighbor.row_addr)
        .collect::<Vec<_>>();
    assert_eq!(
        recall(&found, &brute_force(&dataset, &query, K).await),
        1.0,
        "stepping over the empty partitions cost part of the answer"
    );

    // A centroid with nothing behind it must not spend the probe budget: this
    // query is nearest to one, and one probe still has to reach a partition
    // that holds rows.
    let onto_empty = vec![5.0f32; dimension];
    let result = index
        .search(
            &onto_empty,
            &SearchParams::new(K)
                .with_nprobes(1)
                .with_search_list_size(BEAM_OVER_PARTITION),
        )
        .await
        .unwrap();
    assert_eq!(result.partitions_read, 1);
    assert_eq!(
        result.neighbors.len(),
        K,
        "the probe was spent on a centroid that holds nothing"
    );

    // The populated ids are sparse, so a lookup that used a partition's
    // *position* in the table instead of its id would open somebody else's file
    // and answer from it. Each single probe is checked against the rows that
    // partition actually holds, which is what makes the swap visible.
    for (partition, first) in POPULATED.into_iter().zip([0.1f32, 0.9]) {
        let mut query = vec![0.5f32; dimension];
        query[0] = first;
        let result = index
            .search(
                &query,
                &SearchParams::new(K)
                    .with_nprobes(1)
                    .with_search_list_size(BEAM_OVER_PARTITION),
            )
            .await
            .unwrap();
        let held = members[&partition]
            .iter()
            .map(|row| row_ids[*row])
            .collect::<HashSet<_>>();
        assert_eq!(result.neighbors.len(), K);
        for neighbor in &result.neighbors {
            assert!(
                held.contains(&neighbor.row_addr),
                "a probe routed to partition {partition} answered with row {}, which lives \
                 somewhere else",
                neighbor.row_addr
            );
        }
    }
}

/// A stored vector that is not finite makes every distance measured against it
/// NaN, and a NaN goes wherever `total_cmp` puts it - which depends on its sign.
/// A negative one sorts ahead of every real answer: it survives the merge and
/// comes back as the nearest neighbour, at a distance a caller comparing against
/// a threshold accepts. A positive one sorts behind every real answer and is
/// dropped by the beam, so it is only ever returned by a query with fewer live
/// rows than it asked for. Which sign comes out of `(a - NaN)^2` is the
/// hardware's business, so the partition here is smaller than the beam and every
/// vertex is in the walk's candidates whichever way the sign falls.
///
/// The graph is built over finite vectors and only the payload written to the
/// file is poisoned, which is the shape corruption takes: an adjacency that
/// still walks, over values that are no longer numbers. Nothing on the read path
/// sweeps the column for it, deliberately - that is `rows * dimension` per
/// partition on the hot path of every query, more work than the walk it would be
/// protecting - so the walk's own candidates are what gets checked.
#[tokio::test]
async fn a_partition_holding_a_non_finite_vector_is_reported_as_corrupt() {
    const ROWS: usize = 16;
    const POISONED: usize = 5;
    const BEAM: usize = 32;

    for poison in [f32::NAN, -f32::NAN, f32::INFINITY] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = DatasetFixture {
            fragments: 1,
            rows_per_fragment: ROWS,
            ..Default::default()
        }
        .write(uri)
        .await;
        let (row_ids, vectors) = rows_with_vectors(&dataset).await;

        let graph_params = BuildParams {
            max_degree: 8,
            search_list_size: BEAM,
            ..Default::default()
        };
        let store =
            lance_vamana::search::flat_storage(&row_ids, &vectors, DistanceType::L2).unwrap();
        let built = lance_vamana::build::build_partition(
            &store,
            &graph_params,
            &lance_vamana::search::Comparisons::default(),
        )
        .unwrap();

        let mut values = vectors
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        values[POISONED * VECTOR_DIM as usize] = poison;
        let poisoned =
            <FixedSizeListArray as lance_arrow::FixedSizeListArrayExt>::try_new_from_values(
                Float32Array::from(values),
                VECTOR_DIM,
            )
            .unwrap();
        let partition = Partition::try_new(built.graph, poisoned).unwrap();

        let centroids =
            <FixedSizeListArray as lance_arrow::FixedSizeListArrayExt>::try_new_from_values(
                Float32Array::from(vec![0.5f32; VECTOR_DIM as usize]),
                VECTOR_DIM,
            )
            .unwrap();
        let covered = vec![0u32];
        let uuid = Uuid::new_v4();
        let mut writer = SegmentWriter::new(
            dataset.object_store(None).await.unwrap(),
            dataset.indices_dir().join(uuid.to_string()),
            IndexMetadata {
                format_version: FORMAT_VERSION,
                max_degree: graph_params.max_degree,
                search_list_size: graph_params.search_list_size,
                alpha: graph_params.alpha,
                dimension: VECTOR_DIM as u32,
                distance_type: DistanceType::L2,
                row_id_mode: RowIdMode::Address,
                fragments: covered.clone(),
            },
            lance_index::vector::ivf::storage::IvfModel::new(centroids, None),
        );
        writer
            .write_partition(0, built.medoid, &partition)
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let details = prost_types::Any {
            type_url: INDEX_DETAILS_TYPE_URL.to_string(),
            value: Vec::new(),
        };
        dataset
            .commit_existing_index_segments(
                INDEX_NAME,
                VECTOR_COLUMN,
                vec![IndexSegment::new(
                    uuid,
                    covered,
                    [dataset.schema().field(VECTOR_COLUMN).unwrap().id],
                    Arc::new(details),
                    FORMAT_VERSION as i32,
                    dataset.manifest.version,
                )],
            )
            .await
            .unwrap();

        let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
        let error = index
            .search(
                &random_vectors(1, 21)[0],
                &SearchParams::new(ROWS).with_search_list_size(BEAM),
            )
            .await
            .expect_err("a walk that measured a non-finite distance must not answer with it");
        assert!(error.to_string().contains("is not finite"), "{error}");
        assert!(
            error.to_string().contains(&row_ids[POISONED].to_string()),
            "the error should name the row whose vector is not a number: {error}"
        );
    }
}
