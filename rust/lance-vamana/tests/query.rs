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
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::transaction::{Operation, UpdateMode, UpdatedFragmentOffsets};
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, build_index_segment, build_segment, create_index,
};
use lance_vamana::format::FORMAT_VERSION;
use lance_vamana::query::{SearchParams, VamanaIndex};
use uuid::Uuid;

mod common;
use common::{DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, random_vectors, recall};

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

/// Lance's own exhaustive k-NN over the same column.
async fn brute_force(dataset: &Dataset, query: &[f32], k: usize) -> Vec<u64> {
    let key = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.nearest(VECTOR_COLUMN, &key, k).unwrap();
    scanner.use_index(false);
    scanner.with_row_id();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
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
            .map(|neighbor| neighbor.row_id)
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

        let found = index
            .search(query, &search)
            .await
            .unwrap()
            .neighbors
            .iter()
            .map(|neighbor| neighbor.row_id)
            .collect::<Vec<_>>();
        total += recall(&found, &exact);
    }
    let recall = total / queries.len() as f64;
    println!("cosine -> recall@{K}={recall:.4}");
    assert!(recall >= 0.95, "cosine recall@{K} was {recall:.4}");
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
    assert!(
        narrow.comparisons < wide.comparisons / 2.0,
        "a narrow probe must actually save work: {:.0} against {:.0}",
        narrow.comparisons,
        wide.comparisons
    );
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
            .map(|neighbor| neighbor.row_id)
            .collect::<Vec<_>>();
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
                .position(|id| *id == neighbor.row_id)
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
                neighbor.row_id,
                neighbor.distance
            );
        }
    }
}

/// A compaction that could not open the index leaves it naming fragments that no
/// longer exist, and every row address it stored for them is dead. Answering
/// from what remains would look like a real answer.
///
/// The compaction is real here, and asserted to be: deleting every row first
/// would drop the fragments outright and the test would pass without compacting
/// anything at all.
#[tokio::test]
async fn an_index_over_a_rewritten_fragment_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = indexed_dataset(uri, &small_fixture()).await;
    VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert!(
        metrics.fragments_removed > 0,
        "nothing was compacted, so this test proves nothing"
    );

    let error = VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect_err("an index over retired fragments must not answer queries");
    assert!(error.to_string().contains("no longer has"), "{error}");
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
                live.contains(&neighbor.row_id),
                "a deleted row was returned: {}",
                neighbor.row_id
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
            .map(|neighbor| neighbor.row_id)
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
        .map(|neighbor| neighbor.row_id.to_string())
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
        .map(|neighbor| neighbor.row_id)
        .collect::<HashSet<_>>();
    assert!(
        fresh.neighbors.iter().all(|n| !gone.contains(&n.row_id)),
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
            .map(|neighbor| neighbor.row_id)
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), K, "the merge returned a row twice");
    }
    assert!(
        measured.recall >= 0.95,
        "recall across two segments was {:.4}",
        measured.recall
    );
    assert!(
        measured.partitions > f64::from(PARTITIONS),
        "a two-segment index must probe both segments, got {:.1} partitions",
        measured.partitions
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

/// An empty partition has no row in the segment table and no file of its own,
/// but routing can still name it. Stepping over it is the normal case.
///
/// Forced rather than hoped for: 256 rows drawn from 8 distinct vectors cannot
/// fill 64 centroids, and the test says so if the fixture stops producing any.
#[tokio::test]
async fn a_probed_partition_that_holds_nothing_is_skipped() {
    const MANY: u32 = 64;

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture {
        fragments: 2,
        rows_per_fragment: 128,
        distinct_vectors: Some(8),
        ..Default::default()
    }
    .write(uri)
    .await;
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, MANY).with_graph_params(BuildParams {
            max_degree: 16,
            search_list_size: 64,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let index = VamanaIndex::open(&dataset, INDEX_NAME).await.unwrap();
    let result = index
        .search(
            &random_vectors(1, 5)[0],
            &SearchParams::new(K)
                .with_nprobes(MANY as usize)
                .with_search_list_size(BEAM),
        )
        .await
        .unwrap();

    assert!(
        result.partitions_read < MANY as usize,
        "every partition holds rows, so this test proves nothing about skipping"
    );
    assert_eq!(result.neighbors.len(), K);

    // The populated partition ids are sparse here, so a lookup that used a
    // partition's *position* in the table instead of its id would open somebody
    // else's file and never notice. With one probe the answer has to be exact:
    // the routed partition holds the query's nearest vector, and no other does.
    let single = SearchParams::new(1)
        .with_nprobes(1)
        .with_search_list_size(BEAM);
    for query in random_vectors(8, 31) {
        let found = index.search(&query, &single).await.unwrap();
        let exact = brute_force_best_distance(&dataset, &query).await;
        assert!(
            (found.neighbors[0].distance - exact).abs() < 1e-4,
            "one probe returned {} where the nearest vector is at {exact}",
            found.neighbors[0].distance
        );
    }
}
