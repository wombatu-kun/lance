// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! S0 spike for an out-of-tree Vamana vector index.
//!
//! These are integration tests on purpose. A target under `tests/` links against
//! the `lance` crate's public API only, so "this module compiles" is itself the
//! experiment: every item it touches is provably reachable from a crate that
//! merely depends on `lance`.
//!
//! Throwaway: this is executable documentation of the S0 answers, not a feature.

use std::sync::Arc;

use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator, UInt32Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::{DatasetIndexExt, IndexSegment};
use lance_arrow::FixedSizeListArrayExt;
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_index::INDEX_FILE_NAME;
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use uuid::Uuid;

const DIM: i32 = 8;

/// A dataset with a vector column, split across `num_frags` fragments.
async fn write_vector_dataset(uri: &str, num_frags: usize, rows_per_frag: usize) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM,
            ),
            false,
        ),
    ]));

    let total = num_frags * rows_per_frag;
    let ids = Int32Array::from_iter_values(0..total as i32);
    let values =
        Float32Array::from_iter_values((0..total * DIM as usize).map(|i| (i % 97) as f32 / 97.0));
    let vectors = arrow_array::FixedSizeListArray::try_new_from_values(values, DIM).unwrap();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)]).unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_frag,
            max_rows_per_group: rows_per_frag,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// Write a Lance file of our own shape under `<indices_dir>/<uuid>/<file_name>`.
///
/// This is the shape S1 wants for `index.idx`: one row per partition. Nothing
/// here goes through any Lance index builder.
async fn write_handwritten_index_file(
    dataset: &Dataset,
    uuid: Uuid,
    file_name: &str,
    num_partitions: usize,
) -> u64 {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("__partition_id", DataType::UInt32, false),
        ArrowField::new("__medoid", DataType::UInt32, false),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..num_partitions as u32)),
            Arc::new(UInt32Array::from_iter_values(
                (0..num_partitions as u32).map(|p| p * 7),
            )),
        ],
    )
    .unwrap();

    let object_store = dataset.object_store(None).await.unwrap();
    let path = dataset.indices_dir().join(uuid.to_string()).join(file_name);
    let writer = object_store.create(&path).await.unwrap();

    let lance_schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap();
    let mut file_writer = create_writer(
        ConcreteFileVersion::V2_1,
        writer,
        lance_schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    file_writer.add_schema_metadata("vamana:format_version", "1");
    file_writer.write_batch(&batch).await.unwrap();
    let summary = file_writer.finish().await.unwrap();
    summary.size_bytes
}

fn vector_details() -> Arc<prost_types::Any> {
    Arc::new(prost_types::Any::from_msg(&lance_index::pb::VectorIndexDetails::default()).unwrap())
}

/// Commit one hand-written segment covering every fragment of the dataset.
async fn commit_spike_segment(
    dataset: &mut Dataset,
    index_name: &str,
    uuid: Uuid,
    details: Arc<prost_types::Any>,
) {
    commit_spike_segment_versioned(dataset, index_name, uuid, details, 1).await
}

async fn commit_spike_segment_versioned(
    dataset: &mut Dataset,
    index_name: &str,
    uuid: Uuid,
    details: Arc<prost_types::Any>,
    index_version: i32,
) {
    let all_fragments = fragment_ids(dataset);
    commit_spike_segments_versioned(
        dataset,
        index_name,
        vec![(uuid, all_fragments)],
        details,
        index_version,
    )
    .await
    .unwrap()
}

fn fragment_ids(dataset: &Dataset) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|f| f.id() as u32)
        .collect()
}

/// Commit an explicit set of `(uuid, fragment ids)` segments under one index name.
///
/// Returns the error instead of unwrapping: the rejection cases are the point.
async fn commit_spike_segments(
    dataset: &mut Dataset,
    index_name: &str,
    segments: Vec<(Uuid, Vec<u32>)>,
) -> lance::Result<()> {
    commit_spike_segments_versioned(dataset, index_name, segments, vector_details(), 1).await
}

async fn commit_spike_segments_versioned(
    dataset: &mut Dataset,
    index_name: &str,
    segments: Vec<(Uuid, Vec<u32>)>,
    details: Arc<prost_types::Any>,
    index_version: i32,
) -> lance::Result<()> {
    let field_id = dataset.schema().field("vec").unwrap().id;
    let dataset_version = dataset.manifest.version;
    let segments = segments
        .into_iter()
        .map(|(uuid, frags)| {
            IndexSegment::new(
                uuid,
                frags,
                [field_id],
                details.clone(),
                index_version,
                dataset_version,
            )
        })
        .collect::<Vec<_>>();
    dataset
        .commit_existing_index_segments(index_name, "vec", segments)
        .await
}

/// Append more rows, producing at least one new fragment.
async fn append_vector_rows(uri: &str, rows: usize) {
    let existing = Dataset::open(uri).await.unwrap();
    let schema = Arc::new(arrow_schema::Schema::from(existing.schema()));
    let ids = Int32Array::from_iter_values(10_000..(10_000 + rows as i32));
    let values =
        Float32Array::from_iter_values((0..rows * DIM as usize).map(|i| (i % 89) as f32 / 89.0));
    let vectors = arrow_array::FixedSizeListArray::try_new_from_values(values, DIM).unwrap();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: rows,
            max_rows_per_group: rows,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
}

/// Committed segment uuids for `index_name`, read back through a fresh open.
async fn committed_uuids(uri: &str, index_name: &str) -> Vec<Uuid> {
    let reopened = Dataset::open(uri).await.unwrap();
    reopened
        .load_indices_by_name(index_name)
        .await
        .unwrap()
        .iter()
        .map(|idx| idx.uuid)
        .collect()
}

/// Q0.1 - can a hand-written index directory be committed and survive a reopen?
#[tokio::test]
async fn q0_1_commit_handwritten_index_directory() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 3, 8).await;

    let field_id = dataset.schema().field("vec").unwrap().id;
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|f| f.id() as u32)
        .collect::<Vec<_>>();
    assert_eq!(fragment_ids.len(), 3, "fixture must be multi-fragment");

    let uuid = Uuid::new_v4();
    let written_bytes = write_handwritten_index_file(&dataset, uuid, INDEX_FILE_NAME, 4).await;
    assert!(written_bytes > 0);

    let details = prost_types::Any::from_msg(&lance_index::pb::VectorIndexDetails::default())
        .map(Arc::new)
        .unwrap();
    let segment = IndexSegment::new(
        uuid,
        fragment_ids.iter().copied(),
        [field_id],
        details,
        1,
        dataset.manifest.version,
    );

    dataset
        .commit_existing_index_segments("vamana_spike", "vec", vec![segment])
        .await
        .unwrap();

    // Reopen from the URI: nothing may depend on in-process state.
    let reopened = Dataset::open(uri).await.unwrap();
    let committed = reopened.load_indices_by_name("vamana_spike").await.unwrap();
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].uuid, uuid);
    assert_eq!(committed[0].fields, vec![field_id]);

    let files = committed[0]
        .files
        .as_ref()
        .expect("commit must record the files it listed");
    assert_eq!(files.len(), 1, "one file was written, one must be listed");
    assert_eq!(files[0].path, INDEX_FILE_NAME);
    assert_eq!(files[0].size_bytes, written_bytes);
}

/// Q0.1 - are extra files in the segment directory picked up by the listing?
///
/// S1 wants one file per partition next to `index.idx`. The commit path fills
/// `IndexMetadata::files` by listing the directory, so this must hold without
/// telling Lance anything about the extra files.
#[tokio::test]
async fn q0_1_extra_partition_files_are_listed() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 2, 8).await;

    let uuid = Uuid::new_v4();
    let mut expected = vec![(INDEX_FILE_NAME.to_string(), 0u64)];
    expected[0].1 = write_handwritten_index_file(&dataset, uuid, INDEX_FILE_NAME, 3).await;
    for partition in 0..3usize {
        let name = format!("part_{partition:05}.idx");
        let size = write_handwritten_index_file(&dataset, uuid, &name, 1).await;
        expected.push((name, size));
    }
    expected.sort();

    commit_spike_segment(
        &mut dataset,
        "vamana_spike_multifile",
        uuid,
        vector_details(),
    )
    .await;

    let reopened = Dataset::open(uri).await.unwrap();
    let committed = reopened
        .load_indices_by_name("vamana_spike_multifile")
        .await
        .unwrap();
    let mut listed = committed[0]
        .files
        .as_ref()
        .unwrap()
        .iter()
        .map(|f| (f.path.clone(), f.size_bytes))
        .collect::<Vec<_>>();
    listed.sort();
    assert_eq!(listed, expected);
}

/// Q0.1 - which `index_details` survive `retain_supported_indices`?
///
/// The version filter reads a max supported version out of the details. An
/// unknown `type_url` cannot be resolved, and the fallback keeps the index; a
/// `VectorIndexDetails` resolves to the vector maximum. Both must round-trip, or
/// an out-of-tree index would vanish on reopen with no error anywhere.
#[tokio::test]
async fn q0_1_index_details_variants_survive_reopen() {
    for (case, details) in [
        ("vector_details", vector_details()),
        (
            "unknown_type_url",
            Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.vamana.VamanaIndexDetails".to_string(),
                value: vec![8, 1],
            }),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = write_vector_dataset(uri, 2, 8).await;

        let uuid = Uuid::new_v4();
        write_handwritten_index_file(&dataset, uuid, INDEX_FILE_NAME, 2).await;
        commit_spike_segment(&mut dataset, "vamana_spike_details", uuid, details).await;

        let reopened = Dataset::open(uri).await.unwrap();
        let committed = reopened
            .load_indices_by_name("vamana_spike_details")
            .await
            .unwrap();
        assert_eq!(
            committed.len(),
            1,
            "case {case}: index was dropped on reopen"
        );
        assert_eq!(committed[0].uuid, uuid, "case {case}");
    }
}

/// Q0.1 - the version filter really can swallow an index, and it does so silently.
///
/// This is the mutation proof for the test above: without it, "the index
/// survived" would be an assertion that cannot fail. It also pins the asymmetry
/// that decides which `index_details` an out-of-tree index should write - a
/// resolvable `type_url` subjects it to a version ceiling it does not control,
/// an unresolvable one does not.
#[tokio::test]
async fn q0_1_future_index_version_is_dropped_silently() {
    for (case, details, expect_survives) in [
        ("vector_details", vector_details(), false),
        (
            "unknown_type_url",
            Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.vamana.VamanaIndexDetails".to_string(),
                value: vec![8, 1],
            }),
            true,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = write_vector_dataset(uri, 2, 8).await;

        let uuid = Uuid::new_v4();
        write_handwritten_index_file(&dataset, uuid, INDEX_FILE_NAME, 2).await;
        commit_spike_segment_versioned(&mut dataset, "vamana_spike_future", uuid, details, 999)
            .await;

        let reopened = Dataset::open(uri).await.unwrap();
        let committed = reopened
            .load_indices_by_name("vamana_spike_future")
            .await
            .unwrap();
        assert_eq!(
            !committed.is_empty(),
            expect_survives,
            "case {case}: unexpected survival of index_version 999"
        );
    }
}

/// Q0.2 - recommitting the same fragment coverage replaces the segment.
///
/// The plan assumed this would be rejected because the commit path checks for
/// disjoint coverage. It is not: that check runs only across the segments of a
/// single call, and an existing segment fully covered by the incoming set is
/// removed rather than refused.
#[tokio::test]
async fn q0_2_same_coverage_replaces_the_segment() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 2, 8).await;
    let all = fragment_ids(&dataset);

    let first = Uuid::new_v4();
    write_handwritten_index_file(&dataset, first, INDEX_FILE_NAME, 2).await;
    commit_spike_segments(&mut dataset, "vamana_replace", vec![(first, all.clone())])
        .await
        .unwrap();
    assert_eq!(committed_uuids(uri, "vamana_replace").await, vec![first]);

    let second = Uuid::new_v4();
    write_handwritten_index_file(&dataset, second, INDEX_FILE_NAME, 2).await;
    commit_spike_segments(&mut dataset, "vamana_replace", vec![(second, all)])
        .await
        .unwrap();
    assert_eq!(
        committed_uuids(uri, "vamana_replace").await,
        vec![second],
        "the old segment must be gone, not kept alongside"
    );
}

/// Q0.2 - a disjoint segment survives while its sibling is rewritten.
///
/// This is the commit shape consolidation (S5) needs: rewrite one segment, say
/// nothing about the others, and they are left exactly as they were.
#[tokio::test]
async fn q0_2_disjoint_segment_survives_sibling_rewrite() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 4, 8).await;
    let all = fragment_ids(&dataset);
    assert_eq!(all.len(), 4);
    let (left, right) = (all[..2].to_vec(), all[2..].to_vec());

    let seg_left = Uuid::new_v4();
    let seg_right = Uuid::new_v4();
    write_handwritten_index_file(&dataset, seg_left, INDEX_FILE_NAME, 2).await;
    write_handwritten_index_file(&dataset, seg_right, INDEX_FILE_NAME, 2).await;
    commit_spike_segments(
        &mut dataset,
        "vamana_sibling",
        vec![(seg_left, left.clone()), (seg_right, right)],
    )
    .await
    .unwrap();
    let mut before = committed_uuids(uri, "vamana_sibling").await;
    before.sort();
    let mut expected = vec![seg_left, seg_right];
    expected.sort();
    assert_eq!(before, expected);

    // Rewrite only the left segment.
    let seg_left_v2 = Uuid::new_v4();
    write_handwritten_index_file(&dataset, seg_left_v2, INDEX_FILE_NAME, 2).await;
    commit_spike_segments(&mut dataset, "vamana_sibling", vec![(seg_left_v2, left)])
        .await
        .unwrap();

    let mut after = committed_uuids(uri, "vamana_sibling").await;
    after.sort();
    let mut want = vec![seg_left_v2, seg_right];
    want.sort();
    assert_eq!(
        after, want,
        "the untouched segment must survive a commit that never mentions it"
    );
}

/// Q0.2 - covering only part of an existing segment is rejected.
///
/// This is the constraint that forces segments to be fragment-aligned. An IVF
/// partition cuts across fragments, so "rewrite one partition" can never be
/// expressed as a commit: the resulting segment would cover a slice of every
/// sibling's fragments and orphan the rest.
#[tokio::test]
async fn q0_2_partial_coverage_of_existing_segment_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 2, 8).await;
    let all = fragment_ids(&dataset);

    let whole = Uuid::new_v4();
    write_handwritten_index_file(&dataset, whole, INDEX_FILE_NAME, 2).await;
    commit_spike_segments(&mut dataset, "vamana_partial", vec![(whole, all.clone())])
        .await
        .unwrap();

    let half = Uuid::new_v4();
    write_handwritten_index_file(&dataset, half, INDEX_FILE_NAME, 1).await;
    let err = commit_spike_segments(&mut dataset, "vamana_partial", vec![(half, vec![all[0]])])
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("orphan fragments"),
        "unexpected error: {err}"
    );
    assert_eq!(
        committed_uuids(uri, "vamana_partial").await,
        vec![whole],
        "a rejected commit must leave the index untouched"
    );
}

/// Q0.2 - overlapping coverage *within one call* is the check the plan mistook
/// for a check against the manifest. Pinning it keeps the two rules apart.
#[tokio::test]
async fn q0_2_overlap_within_one_commit_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 2, 8).await;
    let all = fragment_ids(&dataset);

    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    write_handwritten_index_file(&dataset, a, INDEX_FILE_NAME, 1).await;
    write_handwritten_index_file(&dataset, b, INDEX_FILE_NAME, 1).await;
    let err = commit_spike_segments(
        &mut dataset,
        "vamana_overlap",
        vec![(a, all.clone()), (b, vec![all[0]])],
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("overlapping fragment coverage"),
        "unexpected error: {err}"
    );
}

/// Q0.2 - a segment covering only newly appended fragments can be added without
/// touching the existing ones. This is the append path (S6 mode (a)).
#[tokio::test]
async fn q0_2_new_fragments_get_their_own_segment() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 2, 8).await;
    let base_fragments = fragment_ids(&dataset);

    let base = Uuid::new_v4();
    write_handwritten_index_file(&dataset, base, INDEX_FILE_NAME, 2).await;
    commit_spike_segments(
        &mut dataset,
        "vamana_append",
        vec![(base, base_fragments.clone())],
    )
    .await
    .unwrap();

    append_vector_rows(uri, 8).await;
    let mut dataset = Dataset::open(uri).await.unwrap();
    let fresh = fragment_ids(&dataset)
        .into_iter()
        .filter(|id| !base_fragments.contains(id))
        .collect::<Vec<_>>();
    assert!(!fresh.is_empty(), "append must produce new fragments");

    let delta = Uuid::new_v4();
    write_handwritten_index_file(&dataset, delta, INDEX_FILE_NAME, 1).await;
    commit_spike_segments(&mut dataset, "vamana_append", vec![(delta, fresh)])
        .await
        .unwrap();

    let mut after = committed_uuids(uri, "vamana_append").await;
    after.sort();
    let mut want = vec![base, delta];
    want.sort();
    assert_eq!(
        after, want,
        "the base segment must survive a delta commit that never mentions it"
    );
}

/// Q0.4 - what does `compact_files` do to an index it cannot open?
///
/// Compaction rewrites row addresses, so every index has to be remapped. The
/// remap dispatch opens each index first, and an out-of-tree format cannot be
/// opened by the built-in reader. The plan predicted the index would be dropped
/// from the manifest. It is not: it is **stranded**. The manifest entry and its
/// files survive untouched while its fragment coverage now names only fragments
/// that no longer exist, so it indexes nothing.
///
/// A built-in index on the same dataset is the control arm - without it we could
/// not tell "our index was mistreated" from "compaction does this to everyone".
#[tokio::test]
async fn q0_4_compaction_strands_an_unreadable_index() {
    let _ = env_logger::builder().is_test(true).try_init();

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = write_vector_dataset(uri, 4, 8).await;

    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("builtin_control".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();

    let uuid = Uuid::new_v4();
    write_handwritten_index_file(&dataset, uuid, INDEX_FILE_NAME, 2).await;
    commit_spike_segment(&mut dataset, "vamana_compact", uuid, vector_details()).await;
    let fragments_before = fragment_ids(&dataset);
    assert_eq!(fragments_before.len(), 4);

    let mut dataset = Dataset::open(uri).await.unwrap();
    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    // Guard: without this the assertions below could pass vacuously because
    // compaction never rewrote anything.
    assert!(
        metrics.fragments_removed > 0,
        "compaction must actually rewrite fragments, got {metrics:?}"
    );

    let after = Dataset::open(uri).await.unwrap();
    let live: roaring::RoaringBitmap = fragment_ids(&after).into_iter().collect();
    assert!(
        live.is_disjoint(&fragments_before.iter().copied().collect()),
        "compaction must have produced brand new fragment ids"
    );
    let indices = after.load_indices().await.unwrap();
    let effective = |name: &str| -> Vec<u32> {
        indices
            .iter()
            .find(|idx| idx.name == name)
            .unwrap_or_else(|| panic!("index '{name}' vanished from the manifest"))
            .effective_fragment_bitmap(&live)
            .map(|b| b.iter().collect())
            .unwrap_or_default()
    };

    assert_eq!(
        effective("builtin_control"),
        live.iter().collect::<Vec<_>>(),
        "control: a built-in index follows compaction onto the new fragments"
    );
    assert!(
        effective("vamana_compact").is_empty(),
        "an unreadable index is stranded, covering no live fragment"
    );
}
