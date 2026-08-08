// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Does the partition file actually give one vertex per ranged read?
//!
//! The whole layout exists to make `__neighbors` and `__vector` addressable at
//! `base + local_id * stride`. That is a property of the *encoding* Lance chose,
//! not of the schema we wrote, so it has to be measured rather than assumed -
//! and the measurement itself has to be shown capable of failing, which is what
//! the mini-block arm is for.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{Fields, Schema as ArrowSchema};
use lance_core::utils::io_stats::IoStatsRecorder;
use lance_encoding::constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::ObjectStore;
use lance_vamana::format::{NEIGHBORS_COLUMN, ROW_ID_COLUMN, VECTOR_COLUMN, partition_schema};
use lance_vamana::io::{SEGMENT_FILE_VERSION, open_file, read_partition, read_rows};
use lance_vamana::partition::Partition;
use object_store::path::Path;

mod common;
use common::sample_partition;

const VERTICES: usize = 4096;

/// Both widths sit below the 256-byte threshold at which Lance would choose
/// full-zip unprompted, so both columns depend on the explicit hint. They differ
/// from each other so that a test measuring the wrong column cannot pass.
const MAX_DEGREE: u32 = 32;
const DIMENSION: u32 = 24;

const NEIGHBOR_STRIDE: f64 = (MAX_DEGREE * 4) as f64;
const VECTOR_STRIDE: f64 = (DIMENSION * 4) as f64;

/// Counts what the scheduler actually submitted to storage, after coalescing.
#[derive(Debug, Default)]
struct ByteCounter {
    bytes: AtomicU64,
}

impl ByteCounter {
    fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

impl IoStatsRecorder for ByteCounter {
    fn record_request(&self, ranges: &[Range<u64>]) {
        let total: u64 = ranges.iter().map(|range| range.end - range.start).sum();
        self.bytes.fetch_add(total, Ordering::Relaxed);
    }
}

fn local_store_and_path(dir: &tempfile::TempDir, name: &str) -> (Arc<ObjectStore>, Path) {
    let store = Arc::new(ObjectStore::local());
    let path = Path::from_absolute_path(dir.path().join(name)).unwrap();
    (store, path)
}

/// The same partition written through a schema that does *not* ask for full-zip.
///
/// Below 64 values a column is under 256 bytes, so Lance's own heuristic picks
/// mini-block and the addressing is gone. This is the arm that proves the byte
/// measurements below can fail. It writes literally the same arrays, stripped of
/// the encoding hints, so nothing but the hint differs between the two arms.
async fn write_without_encoding_hint(store: &ObjectStore, path: &Path, partition: &Partition) {
    let hinted = partition.to_batch().unwrap();
    let fields = hinted
        .schema()
        .fields()
        .iter()
        .map(|field| Arc::new(field.as_ref().clone().with_metadata(HashMap::new())))
        .collect::<Fields>();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    assert!(
        arrow_schema
            .fields()
            .iter()
            .all(|field| field.metadata().is_empty()),
        "the control arm must carry no encoding hint"
    );
    let batch = RecordBatch::try_new(arrow_schema.clone(), hinted.columns().to_vec()).unwrap();

    let schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap();
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store.create(path).await.unwrap(),
        schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();
}

/// Bytes charged for reading `vertices`, measured on a reader of its own so the
/// fixed cost of opening the file is charged to every arm identically.
async fn bytes_to_read(
    store: Arc<ObjectStore>,
    path: &Path,
    columns: &[&str],
    vertices: Range<usize>,
) -> u64 {
    let reader = open_file(store, path, Some(columns)).await.unwrap();
    let counter = Arc::new(ByteCounter::default());
    let reader = reader.with_io_stats(counter.clone());
    read_rows(&reader, vertices).await.unwrap();
    counter.bytes()
}

/// What one vertex costs, from two angles.
///
/// `marginal` differences two reads of different lengths, so the file's fixed
/// overhead cancels and what is left is the stride. `single` is the whole cost
/// of fetching one vertex, which is where read amplification shows up: a
/// chunked encoding has a *lower* marginal cost than an addressable one - the
/// neighbours already came along for the ride - while charging far more for the
/// first vertex. Reporting only the marginal number would read backwards.
struct VertexCost {
    marginal: f64,
    single: u64,
}

async fn vertex_cost(store: Arc<ObjectStore>, path: &Path, columns: &[&str]) -> VertexCost {
    let one = bytes_to_read(store.clone(), path, columns, 0..1).await;
    let five = bytes_to_read(store, path, columns, 0..5).await;
    VertexCost {
        marginal: (five as f64 - one as f64) / 4.0,
        single: one,
    }
}

/// The fixture both measurement tests share: identical data, one arm hinted and
/// one not.
async fn addressable_and_chunked(dir: &tempfile::TempDir) -> (Arc<ObjectStore>, Path, Path) {
    let partition = sample_partition(MAX_DEGREE, VERTICES, DIMENSION);
    let (store, addressable) = local_store_and_path(dir, "fullzip.idx");
    lance_vamana::io::write_partition(&store, &addressable, &partition)
        .await
        .unwrap();
    let (_, chunked) = local_store_and_path(dir, "miniblock.idx");
    write_without_encoding_hint(&store, &chunked, &partition).await;
    (store, addressable, chunked)
}

#[tokio::test]
async fn partition_round_trips_through_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "part_00000.idx");
    let partition = sample_partition(64, 512, DIMENSION);

    let size = lance_vamana::io::write_partition(&store, &path, &partition)
        .await
        .unwrap();
    assert!(size > 0);

    let reader = open_file(store, &path, None).await.unwrap();
    assert_eq!(read_partition(&reader).await.unwrap(), partition);
}

/// A partition file must be an ordinary Lance file, not our own format wearing
/// a Lance extension.
#[tokio::test]
async fn partition_file_opens_with_the_stock_reader() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "part_00000.idx");
    let partition = sample_partition(64, 128, DIMENSION);
    lance_vamana::io::write_partition(&store, &path, &partition)
        .await
        .unwrap();

    let scheduler = lance_io::scheduler::ScanScheduler::new(
        store.clone(),
        lance_io::scheduler::SchedulerConfig::max_bandwidth(&store),
    );
    let file = scheduler
        .open_file(&path, &lance_io::utils::CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = FileReader::try_open(
        file,
        None,
        Arc::<lance_encoding::decoder::DecoderPlugins>::default(),
        &lance_core::cache::LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(reader.metadata().num_rows, partition.len() as u64);
    let names = reader
        .schema()
        .fields
        .iter()
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec![ROW_ID_COLUMN, NEIGHBORS_COLUMN, VECTOR_COLUMN]);
}

/// The measurement, and the proof that it can fail.
#[tokio::test]
async fn a_vertex_costs_one_stride_only_because_full_zip_was_requested() {
    let dir = tempfile::tempdir().unwrap();
    let (store, addressable, chunked) = addressable_and_chunked(&dir).await;

    let addressable = vertex_cost(store.clone(), &addressable, &[NEIGHBORS_COLUMN]).await;
    let chunked = vertex_cost(store, &chunked, &[NEIGHBORS_COLUMN]).await;
    println!(
        "max_degree={MAX_DEGREE} stride={NEIGHBOR_STRIDE}\n  full-zip:   marginal={} B, one vertex={} B\n  mini-block: marginal={} B, one vertex={} B",
        addressable.marginal, addressable.single, chunked.marginal, chunked.single
    );

    assert_eq!(
        addressable.marginal, NEIGHBOR_STRIDE,
        "a full-zip vertex must cost exactly its stride, got {} for a {MAX_DEGREE}-wide \
         neighbour list",
        addressable.marginal
    );
    assert!(
        chunked.single > 4 * addressable.single,
        "the mini-block arm must show read amplification, or this test proves nothing: \
         one vertex cost {} B chunked against {} B addressable",
        chunked.single,
        addressable.single
    );
    assert!(
        chunked.marginal < NEIGHBOR_STRIDE,
        "a chunked encoding has no per-vertex stride to find; got {}",
        chunked.marginal
    );
}

/// The vector column is the one the disk-resident traversal reads on every hop,
/// so it has to be addressable on exactly the same terms as the edges.
#[tokio::test]
async fn a_vector_costs_one_stride_too() {
    let dir = tempfile::tempdir().unwrap();
    let (store, addressable, chunked) = addressable_and_chunked(&dir).await;

    let addressable = vertex_cost(store.clone(), &addressable, &[VECTOR_COLUMN]).await;
    let chunked = vertex_cost(store, &chunked, &[VECTOR_COLUMN]).await;
    println!(
        "dimension={DIMENSION} stride={VECTOR_STRIDE}\n  full-zip:   marginal={} B, one vertex={} B\n  mini-block: marginal={} B, one vertex={} B",
        addressable.marginal, addressable.single, chunked.marginal, chunked.single
    );

    assert_eq!(
        addressable.marginal, VECTOR_STRIDE,
        "a full-zip vector must cost exactly its stride, got {} for {DIMENSION} dimensions",
        addressable.marginal
    );
    assert!(
        chunked.single > 4 * addressable.single,
        "the mini-block arm must show read amplification, or this test proves nothing: \
         one vector cost {} B chunked against {} B addressable",
        chunked.single,
        addressable.single
    );
}

/// Edges and vectors are separate columns, and a reader pays only for the ones
/// it projects. This is what makes a graph walk able to skip `__row_id`, and a
/// consolidation able to rewrite edges without touching vectors - neither is
/// exercised yet, and both stop being possible if this ever regresses.
#[tokio::test]
async fn a_projection_pays_only_for_the_columns_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path, _) = addressable_and_chunked(&dir).await;

    let neighbors = vertex_cost(store.clone(), &path, &[NEIGHBORS_COLUMN]).await;
    let vector = vertex_cost(store.clone(), &path, &[VECTOR_COLUMN]).await;
    let both = vertex_cost(store, &path, &[NEIGHBORS_COLUMN, VECTOR_COLUMN]).await;

    assert_eq!(neighbors.marginal, NEIGHBOR_STRIDE);
    assert_eq!(vector.marginal, VECTOR_STRIDE);
    assert_eq!(
        both.marginal,
        NEIGHBOR_STRIDE + VECTOR_STRIDE,
        "reading both columns must cost both strides and nothing else"
    );
    assert!(
        both.single > neighbors.single,
        "projecting one column must fetch fewer bytes than projecting two: \
         {} B for the edges alone against {} B for both",
        neighbors.single,
        both.single
    );
}

/// At the natural degree the heuristic would have chosen full-zip anyway; the
/// explicit hint must not change that.
#[tokio::test]
async fn the_hint_is_harmless_above_the_heuristic_threshold() {
    const WIDE_DEGREE: u32 = 64;
    const WIDE_DIMENSION: u32 = 96;

    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "wide.idx");
    lance_vamana::io::write_partition(
        &store,
        &path,
        &sample_partition(WIDE_DEGREE, VERTICES, WIDE_DIMENSION),
    )
    .await
    .unwrap();

    assert_eq!(
        vertex_cost(store.clone(), &path, &[NEIGHBORS_COLUMN])
            .await
            .marginal,
        f64::from(WIDE_DEGREE * 4)
    );
    assert_eq!(
        vertex_cost(store, &path, &[VECTOR_COLUMN]).await.marginal,
        f64::from(WIDE_DIMENSION * 4)
    );
}

/// Several partitions with several vertices each: a one-row-per-partition
/// fixture would hide an off-by-one in the vertex range arithmetic.
#[tokio::test]
async fn vertices_are_addressed_independently_across_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(ObjectStore::local());
    let mut written = HashMap::new();

    for partition_id in 0..3usize {
        let path = Path::from_absolute_path(dir.path().join(format!("part_{partition_id:05}.idx")))
            .unwrap();
        let partition = sample_partition(64, 40 + partition_id * 7, DIMENSION);
        lance_vamana::io::write_partition(&store, &path, &partition)
            .await
            .unwrap();
        written.insert(partition_id, (path, partition));
    }

    for (partition_id, (path, partition)) in &written {
        let reader = open_file(store.clone(), path, None).await.unwrap();
        assert_eq!(
            &read_partition(&reader).await.unwrap(),
            partition,
            "partition {partition_id} did not round trip"
        );

        // A slice out of the middle must line up with the same vertices in memory.
        let middle = 7..19;
        let batch = read_rows(&reader, middle.clone()).await.unwrap();
        let row_ids = batch[ROW_ID_COLUMN]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(row_ids, partition.graph().row_ids()[middle].to_vec());
    }
}

/// Guard against the schema drifting away from what the layout needs.
#[tokio::test]
async fn the_written_schema_keeps_the_encoding_hint() {
    let schema = partition_schema(64, 96).unwrap();
    let hinted = schema
        .fields()
        .iter()
        .filter(|field| {
            field.metadata().get(STRUCTURAL_ENCODING_META_KEY)
                == Some(&STRUCTURAL_ENCODING_FULLZIP.to_string())
        })
        .map(|field| field.name().as_str())
        .collect::<HashSet<_>>();
    assert_eq!(hinted, HashSet::from([NEIGHBORS_COLUMN, VECTOR_COLUMN]));
}
