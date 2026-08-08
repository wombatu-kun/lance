// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Does building an index over a real dataset produce something Lance keeps, and
//! does what it stores actually correspond to the rows it claims?
//!
//! The two questions are separate and both have to be asked. A segment can be
//! committed, survive a reopen and still name the wrong rows - the graph is built
//! over a `VectorStore`, and one of the two ways to build that store synthesises
//! row ids `0..n`. Nothing about the commit would notice.

use std::collections::{HashMap, HashSet};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use lance::index::DatasetIndexExt;
use lance_vamana::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, build_segment, create_index, live_fragments,
};
use lance_vamana::format::INDEX_FILE_NAME;
use lance_vamana::io::{open_file, read_partition, read_segment};
use lance_vamana::partition::Partition;
use lance_vamana::segment::SegmentManifest;
use object_store::path::Path;

mod common;
use common::{DatasetFixture, VECTOR_COLUMN};

const INDEX_NAME: &str = "vamana_idx";
const PARTITIONS: u32 = 8;

fn params() -> IndexParams {
    IndexParams::new(VECTOR_COLUMN, PARTITIONS)
}

/// Locate the committed segment and read every partition back off disk.
async fn read_committed(dataset: &Dataset) -> (SegmentManifest, HashMap<u32, Partition>) {
    let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(indices.len(), 1, "expected exactly one committed segment");
    let store = dataset.object_store(None).await.unwrap();
    let dir = dataset.indices_dir().join(indices[0].uuid.to_string());

    let manifest = read_segment(store.clone(), &dir).await.unwrap();
    let mut partitions = HashMap::new();
    for entry in manifest.partitions() {
        let reader = open_file(store.clone(), &dir.clone().join(entry.file.as_str()), None)
            .await
            .unwrap();
        partitions.insert(entry.partition_id, read_partition(&reader).await.unwrap());
    }
    (manifest, partitions)
}

/// Every `_rowid` in the dataset, in scan order.
async fn live_row_ids(dataset: &Dataset) -> Vec<u64> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project::<&str>(&[]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[lance_core::ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

fn vector_at(vectors: &FixedSizeListArray, row: usize) -> &[f32] {
    let dim = vectors.value_length() as usize;
    &vectors.values().as_primitive::<Float32Type>().values()[row * dim..(row + 1) * dim]
}

#[tokio::test]
async fn a_built_index_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture::default();
    let mut dataset = fixture.write(uri).await;

    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();

    // Reopened by URI, so nothing here leans on in-process state.
    let reopened = Dataset::open(uri).await.unwrap();
    let indices = reopened.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(indices.len(), 1);
    let index = &indices[0];
    assert_eq!(
        index.index_details.as_ref().unwrap().type_url,
        INDEX_DETAILS_TYPE_URL,
        "a resolvable type url would put the segment under a version ceiling we do not control"
    );
    assert_eq!(
        index.fragment_bitmap.as_ref().unwrap().len() as usize,
        fixture.fragments
    );

    // Lance fills `files` by listing the segment directory, so this is the proof
    // that our per-partition files are part of the index as far as Lance knows.
    let files = index
        .files
        .as_ref()
        .expect("the commit must record its files");
    let names = files
        .iter()
        .map(|f| f.path.as_str())
        .collect::<HashSet<_>>();
    assert!(names.contains(INDEX_FILE_NAME), "{names:?}");
    assert!(files.iter().all(|f| f.size_bytes > 0));

    let (manifest, _) = read_committed(&reopened).await;
    assert_eq!(files.len(), manifest.partitions().len() + 1);
}

/// The load-bearing test of this stage: what the index stores must be the rows it
/// says it stores, both the identifiers and the vectors.
#[tokio::test]
async fn the_index_stores_the_dataset_rows_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();

    let (_, partitions) = read_committed(&dataset).await;
    for (partition_id, partition) in &partitions {
        let row_ids = partition.graph().row_ids().to_vec();
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
        assert_eq!(
            taken.num_rows(),
            row_ids.len(),
            "partition {partition_id} names row ids the dataset does not have"
        );

        // Joined on `_rowid`, never on position: `take_rows` drops rows it cannot
        // find rather than erroring, so position would silently shift.
        let fetched = taken[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .zip(0..)
            .collect::<HashMap<u64, usize>>();
        let vectors = taken[VECTOR_COLUMN].as_fixed_size_list();

        for (local_id, row_id) in row_ids.iter().enumerate() {
            let row = *fetched
                .get(row_id)
                .unwrap_or_else(|| panic!("row {row_id} is missing from the take"));
            assert_eq!(
                partition.vector(local_id as u32),
                vector_at(vectors, row),
                "partition {partition_id} vertex {local_id} holds another row's vector"
            );
        }
    }
}

/// Partitioning is a partition: no row lost, no row counted twice.
#[tokio::test]
async fn every_indexed_row_lands_in_exactly_one_partition() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture::default();
    let mut dataset = fixture.write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();

    let (manifest, partitions) = read_committed(&dataset).await;
    let mut indexed = Vec::new();
    for partition in partitions.values() {
        indexed.extend_from_slice(partition.graph().row_ids());
    }
    let unique = indexed.iter().copied().collect::<HashSet<_>>();
    assert_eq!(
        unique.len(),
        indexed.len(),
        "a row was written into more than one partition"
    );
    assert_eq!(
        unique,
        live_row_ids(&dataset)
            .await
            .into_iter()
            .collect::<HashSet<_>>()
    );

    // The table has to agree with the files it points at.
    for entry in manifest.partitions() {
        let partition = &partitions[&entry.partition_id];
        assert_eq!(entry.num_rows as usize, partition.len());
        assert!((entry.medoid as usize) < partition.len());
    }
    assert!(
        manifest.partitions().len() > 1,
        "a single-partition fixture would make routing untestable"
    );
}

/// A dataset may hold rows without vectors; they are skipped, not guessed at.
#[tokio::test]
async fn rows_without_a_vector_are_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture {
        null_every: Some(4),
        ..Default::default()
    };
    let mut dataset = fixture.write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();

    let (_, partitions) = read_committed(&dataset).await;
    let indexed = partitions
        .values()
        .flat_map(|partition| partition.graph().row_ids().iter().copied())
        .collect::<HashSet<_>>();
    assert_eq!(indexed.len(), fixture.indexed_rows());
    assert!(
        indexed.len() < fixture.rows(),
        "the fixture indexed everything"
    );

    let with_vectors = dataset
        .take_rows(
            &indexed.iter().copied().collect::<Vec<_>>(),
            ProjectionRequest::from_columns([VECTOR_COLUMN], dataset.schema()),
        )
        .await
        .unwrap();
    assert_eq!(
        with_vectors[VECTOR_COLUMN]
            .as_fixed_size_list()
            .null_count(),
        0,
        "a row with no vector was indexed anyway"
    );
}

/// Stable row ids are a different identifier space, and stage C's delete list is
/// only valid in the address space. Refuse loudly rather than be wrong quietly.
#[tokio::test]
async fn a_stable_row_id_dataset_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture {
        stable_row_ids: true,
        ..Default::default()
    }
    .write(uri)
    .await;

    let error = create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("stable row ids"), "{error}");
    assert!(
        dataset
            .load_indices_by_name(INDEX_NAME)
            .await
            .unwrap()
            .is_empty(),
        "a refused build must not leave an index behind"
    );
}

/// One seed, one index. Lance's own k-means seeds itself from the OS, so this
/// only holds because the builder hands it a starting set of centroids.
#[tokio::test]
async fn the_same_seed_builds_the_same_index() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = DatasetFixture::default().write(uri).await;
    let store = dataset.object_store(None).await.unwrap();

    let mut built = Vec::new();
    for run in 0..2 {
        let segment_dir = Path::from_absolute_path(dir.path().join(format!("run_{run}"))).unwrap();
        let manifest = build_segment(&dataset, &params(), &segment_dir, &live_fragments(&dataset))
            .await
            .unwrap();
        let mut partitions = Vec::new();
        for entry in manifest.partitions() {
            let reader = open_file(
                store.clone(),
                &segment_dir.clone().join(entry.file.as_str()),
                None,
            )
            .await
            .unwrap();
            partitions.push(read_partition(&reader).await.unwrap());
        }
        built.push((manifest, partitions));
    }

    assert_eq!(built[0].0, built[1].0, "the routing model or table drifted");
    assert_eq!(built[0].1, built[1].1, "the graphs drifted");

    // Determinism alone would also hold for a builder that ignored the seed
    // entirely - which is exactly what handing k-means its own starting
    // centroids is there to prevent. A different seed has to build differently.
    let mut other = params();
    other.graph.seed += 1;
    let elsewhere = Path::from_absolute_path(dir.path().join("other_seed")).unwrap();
    let other_manifest = build_segment(&dataset, &other, &elsewhere, &live_fragments(&dataset))
        .await
        .unwrap();
    assert_ne!(
        other_manifest.ivf().centroids,
        built[0].0.ivf().centroids,
        "a different seed trained the same router, so the seed is not reaching k-means"
    );
}

#[tokio::test]
async fn more_partitions_than_rows_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture {
        fragments: 1,
        rows_per_fragment: 32,
        ..Default::default()
    };
    let mut dataset = fixture.write(uri).await;

    let error = create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, fixture.rows() as u32 + 1),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("fewer partitions"), "{error}");
}

/// Dot distance is refused rather than quietly building a worse graph: Lance's
/// `1 - dot` goes negative for unnormalised vectors, which inverts what the
/// pruning slack does.
#[tokio::test]
async fn a_dot_distance_index_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture::default().write(uri).await;

    let error = create_index(
        &mut dataset,
        INDEX_NAME,
        &params().with_distance_type(lance_linalg::distance::DistanceType::Dot),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("dot distance"), "{error}");
}

#[tokio::test]
async fn an_unknown_column_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture::default().write(uri).await;

    let error = create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new("nope", PARTITIONS),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("'nope'"), "{error}");
}

/// The object store is the one the dataset hands out, so a segment written by the
/// builder is readable by anything holding the dataset - including a driver that
/// only ever sees the committed manifest.
#[tokio::test]
async fn a_segment_is_readable_through_the_datasets_own_store() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture::default().write(uri).await;
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();

    let (manifest, _) = read_committed(&dataset).await;
    assert_eq!(manifest.metadata().max_degree, params().graph.max_degree);
    assert_eq!(manifest.metadata().dimension, common::VECTOR_DIM as u32);
    assert_eq!(
        manifest.ivf().num_partitions(),
        PARTITIONS as usize,
        "the router must describe every partition, populated or not"
    );
}

/// Committing a Vamana index changes what Lance itself can do with the dataset.
///
/// The scanner picks a vector index by field id alone, with no type check, so it
/// selects our segment and then cannot read it as one of its own; and
/// `optimize_indices` classifies an index as a vector index by the presence of
/// `index.idx`, so one unreadable index fails the loop over *every* index.
///
/// Neither is a defect in this crate - both follow from there being no way to
/// register an external vector index type - but both are invisible to any test
/// that reaches for the exhaustive path with `use_index(false)`, which is every
/// other test here. Pinned so that an upstream change is noticed rather than
/// discovered, and so the README cannot drift away from the behaviour.
#[tokio::test]
async fn a_committed_index_shadows_lances_own_vector_paths() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture::default();
    let mut dataset = fixture.write(uri).await;

    let query = Float32Array::from(vec![0.5f32; common::VECTOR_DIM as usize]);
    let nearest = |dataset: &Dataset, use_index: bool| {
        let mut scanner = dataset.scan();
        scanner.nearest(VECTOR_COLUMN, &query, 5).unwrap();
        scanner.use_index(use_index);
        async move { scanner.try_into_batch().await }
    };

    assert_eq!(nearest(&dataset, true).await.unwrap().num_rows(), 5);
    create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();

    let shadowed = nearest(&dataset, true).await.unwrap_err();
    assert!(
        shadowed.to_string().contains("Index Metadata not found"),
        "Lance found a way to read our index: {shadowed}"
    );
    assert_eq!(
        nearest(&dataset, false).await.unwrap().num_rows(),
        5,
        "the exhaustive path must stay open, it is the documented escape hatch"
    );

    let error = dataset
        .optimize_indices(&Default::default())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Index Metadata not found"));
    let error = dataset.index_statistics(INDEX_NAME).await.unwrap_err();
    assert!(error.to_string().contains("Index Metadata not found"));

    // Everything that does not go looking for a vector index is unaffected.
    let mut scanner = dataset.scan();
    scanner.project(&[VECTOR_COLUMN]).unwrap();
    assert_eq!(
        scanner.try_into_batch().await.unwrap().num_rows(),
        fixture.rows()
    );
}
