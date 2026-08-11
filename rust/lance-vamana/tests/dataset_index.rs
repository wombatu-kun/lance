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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StructArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::{ProjectionRequest, WriteParams};
use lance::index::DatasetIndexExt;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{
    INDEX_DETAILS_TYPE_URL, IndexParams, MAX_KMEANS_SAMPLE_RATE, build_segment, create_index,
    live_fragments,
};
use lance_vamana::format::INDEX_FILE_NAME;
use lance_vamana::io::{open_file, read_partition, read_segment, scan_scheduler};
use lance_vamana::partition::Partition;
use lance_vamana::query::VamanaIndex;
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

    let scheduler = scan_scheduler(&store);
    let manifest = read_segment(&scheduler, &dir, None).await.unwrap();
    let mut partitions = HashMap::new();
    for entry in manifest.partitions() {
        let reader = open_file(
            &scheduler,
            &dir.clone().join(entry.file.as_str()),
            None,
            None,
        )
        .await
        .unwrap();
        partitions.insert(
            entry.partition_id,
            read_partition(&reader, entry.num_rows).await.unwrap(),
        );
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

/// Summed squared distance from one vertex to every vertex of its partition -
/// the quantity `build::medoid` minimises, recomputed the long way.
fn summed_distance(partition: &Partition, local_id: usize) -> f32 {
    let from = partition
        .vector(local_id as u32)
        .expect("a vertex of this partition");
    (0..partition.len())
        .map(|other| {
            let to = partition
                .vector(other as u32)
                .expect("a vertex of this partition");
            from.iter()
                .zip(to)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f32>()
        })
        .sum()
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
    assert!(
        !manifest.partitions().is_empty(),
        "with no partitions the file count below is 1 == 0 + 1, which proves nothing"
    );
    assert_eq!(files.len(), manifest.partitions().len() + 1);

    // A query takes each partition's size from here rather than probing storage
    // for it, which holds only while Lance records every file it committed under
    // the name the segment table uses.
    let sizes = files
        .iter()
        .map(|file| (file.path.as_str(), file.size_bytes))
        .collect::<HashMap<_, _>>();
    for entry in manifest.partitions() {
        assert!(
            sizes.get(entry.file.as_str()).is_some_and(|size| *size > 0),
            "the manifest records no size for {}, so opening it would cost a probe",
            entry.file
        );
    }
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
    assert!(
        !partitions.is_empty(),
        "an index of no partitions would satisfy every loop below without \
         checking a single row"
    );
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
                partition.vector(local_id as u32).unwrap(),
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
        // Recomputed, not merely bounded: `medoid < len` follows from the line
        // above plus what `try_new` already refuses, so it cannot fail, and the
        // entry point a real build chose was checked nowhere at all.
        let central = (0..partition.len())
            .min_by(|left, right| {
                summed_distance(partition, *left).total_cmp(&summed_distance(partition, *right))
            })
            .expect("a listed partition has vertices");
        assert_eq!(
            entry.medoid as usize, central,
            "partition {} starts its walks somewhere other than its most central vertex",
            entry.partition_id
        );
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
    let stats = create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    assert_eq!(
        stats.vectors,
        fixture.indexed_rows(),
        "the build reported rows it never indexed"
    );

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

/// What a build cost is half of what a graph index is, and it used to be counted
/// and thrown away. A number nobody can read cannot be traded against the query
/// cost, which is the only comparison that means anything.
#[tokio::test]
async fn a_build_reports_what_it_cost() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let fixture = DatasetFixture::default();
    let mut dataset = fixture.write(uri).await;

    let stats = create_index(&mut dataset, INDEX_NAME, &params())
        .await
        .unwrap();
    let (manifest, _) = read_committed(&dataset).await;
    println!(
        "{} vectors in {} partitions -> {} build comparisons ({:.1} per vector)",
        stats.vectors,
        stats.partitions,
        stats.comparisons,
        stats.comparisons as f64 / stats.vectors as f64
    );

    assert_eq!(stats.vectors, fixture.rows());
    // Both sides are incremented in the same loop, so this can only catch a
    // partition written without a table row or the reverse - which is what the
    // count is for. The independent check is against the directory: Lance lists
    // it at commit time, and `files` is what it found.
    assert_eq!(
        stats.partitions,
        manifest.partitions().len(),
        "the build counted partitions the segment does not list"
    );
    let files = dataset.load_indices_by_name(INDEX_NAME).await.unwrap()[0]
        .files
        .as_ref()
        .expect("the commit records its files")
        .len();
    assert_eq!(
        stats.partitions + 1,
        files,
        "the build reports {} partitions but the segment directory holds {files} files \
         including index.idx",
        stats.partitions
    );
    // Pinned to the measured value, not merely to "greater than zero": the whole
    // point is to notice a build that got three times more expensive.
    assert!(
        (1_550_000..2_050_000).contains(&stats.comparisons),
        "a build cost {} comparisons, measured at 1798593 (1171 per vector)",
        stats.comparisons
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
        let (manifest, _) =
            build_segment(&dataset, &params(), &segment_dir, &live_fragments(&dataset))
                .await
                .unwrap();
        let mut partitions = Vec::new();
        for entry in manifest.partitions() {
            let reader = open_file(
                &scan_scheduler(&store),
                &segment_dir.clone().join(entry.file.as_str()),
                None,
                None,
            )
            .await
            .unwrap();
            partitions.push(read_partition(&reader, entry.num_rows).await.unwrap());
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
    let (other_manifest, _) =
        build_segment(&dataset, &other, &elsewhere, &live_fragments(&dataset))
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

    // Three failures matching one string could all be some fourth thing going
    // wrong. Dropping the index and watching every one of them recover is what
    // makes the Vamana segment the cause rather than a coincidence.
    dataset.drop_index(INDEX_NAME).await.unwrap();
    assert_eq!(nearest(&dataset, true).await.unwrap().num_rows(), 5);
    dataset.optimize_indices(&Default::default()).await.unwrap();

    // Everything that does not go looking for a vector index is unaffected.
    let mut scanner = dataset.scan();
    scanner.project(&[VECTOR_COLUMN]).unwrap();
    assert_eq!(
        scanner.try_into_batch().await.unwrap().num_rows(),
        fixture.rows()
    );
}

/// A dataset whose vectors are given cell by cell, so a test can put a null
/// coordinate or a NaN exactly where it wants one. `DatasetFixture` draws its
/// vectors at random, which is the right shape for measuring and the wrong one
/// for pinning a rejection.
async fn dataset_of_vectors(
    uri: &str,
    rows: Vec<Option<Vec<Option<f32>>>>,
    rows_per_fragment: usize,
) -> Dataset {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        VECTOR_COLUMN,
        DataType::FixedSizeList(item, common::VECTOR_DIM),
        true,
    )]));
    let vectors =
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(rows, common::VECTOR_DIM);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            max_rows_per_group: rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// Distinct vectors, one row per entry, as a starting point for a test that then
/// damages exactly one cell.
fn plain_vectors(rows: usize) -> Vec<Option<Vec<Option<f32>>>> {
    (0..rows)
        .map(|row| {
            Some(
                (0..common::VECTOR_DIM as usize)
                    .map(|axis| Some((row * common::VECTOR_DIM as usize + axis) as f32))
                    .collect(),
            )
        })
        .collect()
}

/// A null vector has nothing to index and is skipped; a null *coordinate* is a
/// row whose position is partly unknown, and the byte under it is not a zero the
/// index may believe. Under L2 `Partition::try_new` catches it, but a cosine
/// build normalises first and `normalize_fsl` rebuilds the child through
/// `from_iter_values`, dropping the item-level null mask on the way - so the same
/// column was an error under one metric and silently indexed under the other.
#[tokio::test]
async fn a_null_inside_a_vector_is_refused_under_every_metric() {
    let mut rows = plain_vectors(40);
    rows[7].as_mut().unwrap()[3] = None;

    for distance_type in [DistanceType::L2, DistanceType::Cosine] {
        let dir = tempfile::tempdir().unwrap();
        let mut dataset = dataset_of_vectors(dir.path().to_str().unwrap(), rows.clone(), 20).await;
        let error = create_index(
            &mut dataset,
            INDEX_NAME,
            &IndexParams::new(VECTOR_COLUMN, 2).with_distance_type(distance_type),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("nulls inside its vectors"),
            "{distance_type}: {error}"
        );
    }
}

/// A vector that is not finite cannot be assigned to a partition, and the row
/// has to be nameable: the position the assignment loop knows is into the array
/// left after null vectors were dropped, which no caller can match against
/// anything. Row 25 of a 20-row-per-fragment dataset is offset 5 of fragment 1,
/// so its row id and its position are different numbers on purpose.
#[tokio::test]
async fn a_vector_that_cannot_be_assigned_is_named_by_its_row_id() {
    let mut rows = plain_vectors(40);
    rows[10] = None;
    rows[25].as_mut().unwrap()[0] = Some(f32::NAN);

    let dir = tempfile::tempdir().unwrap();
    let mut dataset = dataset_of_vectors(dir.path().to_str().unwrap(), rows, 20).await;
    let error = create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_COLUMN, 2),
    )
    .await
    .unwrap_err();

    let row_id = (1u64 << 32) | 5;
    assert!(
        error.to_string().contains(&row_id.to_string()),
        "the error names a position rather than a row id: {error}"
    );
}

/// Both k-means parameters are public, both have no default of their own once a
/// caller starts setting them, and zero means something different and equally
/// silent in each: no training set at all, which panics inside `rand` when the
/// centroids are sampled, and no iterations at all, which leaves the router
/// routing by the k rows the initialisation happened to draw.
#[tokio::test]
async fn a_zero_kmeans_parameter_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture::default().write(uri).await;
    for (params, named) in [
        (params().with_kmeans_sample_rate(0), "kmeans_sample_rate"),
        (params().with_kmeans_max_iters(0), "kmeans_max_iters"),
        // And the other end of the sample rate, which is not a matter of taste:
        // above it Lance trains on the front of the training set rather than on
        // a sample of it.
        (
            params().with_kmeans_sample_rate(MAX_KMEANS_SAMPLE_RATE + 1),
            "kmeans_sample_rate",
        ),
    ] {
        let error = create_index(&mut dataset, INDEX_NAME, &params)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(named), "{error}");
    }
}

/// `Schema::field` resolves a dotted path, so a nested leaf passes the column
/// check and then fails inside the build with "column does not exist" - the
/// scanner projects a nested leaf as its top-level parent, under the parent's
/// name. Refused where the reason can be given.
#[tokio::test]
async fn a_nested_vector_column_is_refused_with_its_reason() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();

    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let vector = Field::new(
        "vec",
        DataType::FixedSizeList(item, common::VECTOR_DIM),
        true,
    );
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "emb",
        DataType::Struct(vec![vector.clone()].into()),
        true,
    )]));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        plain_vectors(8),
        common::VECTOR_DIM,
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StructArray::from(vec![(
            Arc::new(vector),
            Arc::new(vectors) as ArrayRef,
        )]))],
    )
    .unwrap();
    let mut dataset = Dataset::write(RecordBatchIterator::new(vec![Ok(batch)], schema), uri, None)
        .await
        .unwrap();

    let error = create_index(&mut dataset, INDEX_NAME, &IndexParams::new("emb.vec", 2))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("is nested"), "{error}");
}

/// A fragment named twice would have its rows read twice and indexed twice, and
/// the coverage bitmap would collapse the duplicate on the way into the manifest
/// - so the segment would look ordinary while holding two copies of everything
/// that fragment carries.
#[tokio::test]
async fn a_fragment_named_twice_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = DatasetFixture::default().write(uri).await;
    let segment_dir = dataset.indices_dir().join("doubled");

    let error = build_segment(&dataset, &params(), &segment_dir, &[0, 1, 0])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("distinct"), "{error}");
}

/// The column type is asked of the schema, not of the data. `read_vectors`
/// checks the same thing on the array it decoded, which is the last line of
/// defence and much too late to be the first: a `FixedSizeList<Float64>` column
/// of five million rows would be read into memory in full and only then refused.
/// The message is what says which of the two spoke.
#[tokio::test]
async fn a_column_of_the_wrong_type_is_refused_before_it_is_read() {
    const ROWS: usize = 8;

    for (name, column, values) in [
        (
            "float64 vectors",
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float64, true)),
                    common::VECTOR_DIM,
                ),
                true,
            ),
            Arc::new(arrow_array::FixedSizeListArray::from_iter_primitive::<
                arrow_array::types::Float64Type,
                _,
                _,
            >(
                (0..ROWS)
                    .map(|row| {
                        Some(
                            (0..common::VECTOR_DIM)
                                .map(move |axis| Some(f64::from(row as i32 + axis)))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                common::VECTOR_DIM,
            )) as ArrayRef,
        ),
        (
            "not a vector at all",
            Field::new("vec", DataType::Int32, true),
            Arc::new(arrow_array::Int32Array::from(
                (0..ROWS as i32).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![column]));
        let batch = RecordBatch::try_new(schema.clone(), vec![values]).unwrap();
        let mut dataset =
            Dataset::write(RecordBatchIterator::new(vec![Ok(batch)], schema), uri, None)
                .await
                .unwrap();

        let error = create_index(&mut dataset, INDEX_NAME, &IndexParams::new("vec", 2))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Vamana indexes FixedSizeList<Float32>"),
            "{name}: refused by the reader rather than by the schema check: {error}"
        );
    }
}

/// Copy a checked-in dataset fixture into a temporary directory.
///
/// Lance's own `copy_test_data_to_tmp` is `pub(crate)`, so this repeats it. The
/// only place this crate reaches out of its own directory: a standalone copy
/// would have to bring `test_data/v0.8.14` along or drop the test with it.
fn copy_fixture(name: &str) -> tempfile::TempDir {
    fn copy_dir(source: &std::path::Path, target: &std::path::Path) {
        std::fs::create_dir_all(target).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let target = target.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).unwrap();
            }
        }
    }

    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name);
    let target = tempfile::tempdir().unwrap();
    copy_dir(&source, target.path());
    target
}

/// The fourth Lance path a Vamana index collides with, and the only one that
/// breaks *writing* rather than reading.
///
/// When the manifest a commit starts from was written before Lance 0.8.15, whose
/// fragment bitmaps could be wrong, `migrate_indices` recalculates the coverage
/// of every index - by *opening* it, with `?` and no fallback. Lance cannot open
/// this format, so the commit fails, and it fails only after the whole graph has
/// been built. This crate refuses up front and names the remedy, which is one
/// commit by any current build: the check is on the manifest, not on the data.
#[tokio::test]
async fn a_dataset_older_than_lances_bitmap_fix_is_refused_before_the_build() {
    let fixture = copy_fixture("v0.8.14/corrupt_index");
    let uri = fixture.path().to_str().unwrap();
    let mut dataset = Dataset::open(uri).await.unwrap();
    assert!(
        dataset
            .manifest()
            .writer_version
            .as_ref()
            .and_then(|writer| writer.lance_lib_version())
            .is_some_and(|parsed| (parsed.major, parsed.minor, parsed.patch) < (0, 8, 15)),
        "the fixture is no longer older than the bitmap fix, so this proves nothing"
    );

    let params = IndexParams::new("vector", 4);
    let error = create_index(&mut dataset, INDEX_NAME, &params)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("lance 0.8.14"), "{error}");

    // A commit that changes nothing still rewrites the manifest with a current
    // writer version, which is all the recalculation is gated on.
    dataset.delete("false").await.unwrap();
    create_index(&mut dataset, INDEX_NAME, &params)
        .await
        .unwrap();
    VamanaIndex::open(&dataset, INDEX_NAME)
        .await
        .expect("the index built after the manifest was refreshed must open");
}

/// A build is a long stretch of arithmetic with no await anywhere inside it. Run
/// on the caller's runtime it holds a worker for that whole stretch, and on a
/// single-threaded runtime - which is what `#[tokio::test]` gives, and what an
/// embedded caller may well hand this crate - every other task on it stops for
/// the duration, including the io loop the scan scheduler runs every read
/// through.
///
/// Measured rather than argued: a ticker task asks for 5ms of sleep at a time
/// and records the longest it was ever kept waiting. With the arithmetic on the
/// CPU pool the longest wait is a few milliseconds; with it inline the longest
/// wait is however long one partition takes to build, which on this fixture is
/// most of the build.
#[tokio::test(flavor = "current_thread")]
async fn a_build_leaves_the_calling_runtime_free() {
    const TICK: Duration = Duration::from_millis(5);
    // Far above the tick and far below a partition of a thousand vertices.
    const LONGEST_TOLERATED_GAP: Duration = Duration::from_millis(200);

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = DatasetFixture {
        fragments: 1,
        rows_per_fragment: 1024,
        ..Default::default()
    }
    .write(uri)
    .await;

    let longest_gap_ms = Arc::new(AtomicU64::new(0));
    let building = Arc::new(AtomicBool::new(true));
    let ticker = tokio::spawn({
        let longest_gap_ms = longest_gap_ms.clone();
        let building = building.clone();
        async move {
            let mut last = Instant::now();
            while building.load(Ordering::Relaxed) {
                tokio::time::sleep(TICK).await;
                let now = Instant::now();
                longest_gap_ms.fetch_max(
                    now.duration_since(last).as_millis() as u64,
                    Ordering::Relaxed,
                );
                last = now;
            }
        }
    });

    // One partition, so the whole graph is one uninterrupted stretch of work.
    let params = IndexParams::new(VECTOR_COLUMN, 1).with_graph_params(BuildParams {
        max_degree: 16,
        search_list_size: 64,
        ..Default::default()
    });
    let started = Instant::now();
    create_index(&mut dataset, INDEX_NAME, &params)
        .await
        .unwrap();
    let build = started.elapsed();
    building.store(false, Ordering::Relaxed);
    ticker.await.unwrap();

    assert!(
        build > 4 * LONGEST_TOLERATED_GAP,
        "the build took {build:?}, which is too little to tell a blocked runtime from a free one"
    );
    let longest_gap = Duration::from_millis(longest_gap_ms.load(Ordering::Relaxed));
    assert!(
        longest_gap < LONGEST_TOLERATED_GAP,
        "a task asking for {TICK:?} of sleep waited {longest_gap:?} during a {build:?} build, so \
         the build is holding the runtime it was called on"
    );
}
