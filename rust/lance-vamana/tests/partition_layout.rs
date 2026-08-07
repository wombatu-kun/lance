// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Does the partition file actually give one vertex per ranged read?
//!
//! The whole layout exists to make `__neighbors` addressable at
//! `base + local_id * max_degree * 4`. That is a property of the *encoding*
//! Lance chose, not of the schema we wrote, so it has to be measured rather
//! than assumed - and the measurement itself has to be shown capable of
//! failing, which is what the mini-block arm is for.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_core::utils::io_stats::IoStatsRecorder;
use lance_encoding::constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::ObjectStore;
use lance_vamana::format::{NEIGHBORS_COLUMN, NO_NEIGHBOR, ROW_ID_COLUMN, partition_schema};
use lance_vamana::io::{SEGMENT_FILE_VERSION, open_file, read_partition, read_rows};
use lance_vamana::partition::PartitionGraph;
use object_store::path::Path;

mod common;
use common::sample_graph;

const VERTICES: usize = 4096;

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

/// The same data written through a schema that does *not* ask for full-zip.
///
/// Below 64 neighbours a value is under 256 bytes, so Lance's own heuristic
/// picks mini-block and the addressing is gone. This is the arm that proves the
/// byte measurement below can fail.
async fn write_without_encoding_hint(
    store: &ObjectStore,
    path: &Path,
    graph: &PartitionGraph,
) -> u64 {
    let width = graph.max_degree() as i32;
    let item = Arc::new(Field::new("item", DataType::UInt32, false));
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ROW_ID_COLUMN, DataType::UInt64, false),
        Field::new(
            NEIGHBORS_COLUMN,
            DataType::FixedSizeList(item.clone(), width),
            false,
        ),
    ]));
    assert!(
        arrow_schema
            .field_with_name(NEIGHBORS_COLUMN)
            .unwrap()
            .metadata()
            .get(STRUCTURAL_ENCODING_META_KEY)
            .is_none(),
        "the control arm must carry no encoding hint"
    );

    let mut flat = Vec::with_capacity(graph.len() * width as usize);
    for local_id in 0..graph.len() as u32 {
        let neighbors = graph.neighbors(local_id);
        flat.extend_from_slice(neighbors);
        flat.resize(flat.len() + width as usize - neighbors.len(), NO_NEIGHBOR);
    }
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(graph.row_ids().to_vec())),
            Arc::new(
                FixedSizeListArray::try_new(item, width, Arc::new(UInt32Array::from(flat)), None)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();

    let schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap();
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store.create(path).await.unwrap(),
        schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap().size_bytes
}

/// Bytes charged for reading `vertices`, measured on a reader of its own so the
/// fixed cost of opening the file is charged to every arm identically.
async fn bytes_to_read(store: Arc<ObjectStore>, path: &Path, vertices: Range<usize>) -> u64 {
    let reader = open_file(store, path, Some(&[NEIGHBORS_COLUMN]))
        .await
        .unwrap();
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

async fn vertex_cost(store: Arc<ObjectStore>, path: &Path) -> VertexCost {
    let one = bytes_to_read(store.clone(), path, 0..1).await;
    let five = bytes_to_read(store, path, 0..5).await;
    VertexCost {
        marginal: (five as f64 - one as f64) / 4.0,
        single: one,
    }
}

#[tokio::test]
async fn partition_round_trips_through_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "part_00000.idx");
    let graph = sample_graph(64, 512);

    let size = lance_vamana::io::write_partition(&store, &path, &graph)
        .await
        .unwrap();
    assert!(size > 0);

    let reader = open_file(store, &path, None).await.unwrap();
    assert_eq!(read_partition(&reader).await.unwrap(), graph);
}

/// A partition file must be an ordinary Lance file, not our own format wearing
/// a Lance extension.
#[tokio::test]
async fn partition_file_opens_with_the_stock_reader() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "part_00000.idx");
    let graph = sample_graph(64, 128);
    lance_vamana::io::write_partition(&store, &path, &graph)
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

    assert_eq!(reader.metadata().num_rows, graph.len() as u64);
    let names = reader
        .schema()
        .fields
        .iter()
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec![ROW_ID_COLUMN, NEIGHBORS_COLUMN]);
}

/// The measurement, and the proof that it can fail.
///
/// Both arms hold the same graph at the same degree; they differ only in
/// whether the schema asked for full-zip. `max_degree = 32` is chosen because it
/// is *below* the 256-byte threshold at which Lance would pick full-zip on its
/// own, so without the explicit hint the heuristic goes to mini-block.
#[tokio::test]
async fn a_vertex_costs_one_stride_only_because_full_zip_was_requested() {
    const MAX_DEGREE: u32 = 32;
    let stride = f64::from(MAX_DEGREE * 4);

    let dir = tempfile::tempdir().unwrap();
    let graph = sample_graph(MAX_DEGREE, VERTICES);

    let (store, addressable) = local_store_and_path(&dir, "fullzip.idx");
    lance_vamana::io::write_partition(&store, &addressable, &graph)
        .await
        .unwrap();
    let (_, chunked) = local_store_and_path(&dir, "miniblock.idx");
    write_without_encoding_hint(&store, &chunked, &graph).await;

    let addressable = vertex_cost(store.clone(), &addressable).await;
    let chunked = vertex_cost(store, &chunked).await;
    println!(
        "max_degree={MAX_DEGREE} stride={stride}\n  full-zip:   marginal={} B, one vertex={} B\n  mini-block: marginal={} B, one vertex={} B",
        addressable.marginal, addressable.single, chunked.marginal, chunked.single
    );

    assert_eq!(
        addressable.marginal, stride,
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
        chunked.marginal < stride,
        "a chunked encoding has no per-vertex stride to find; got {}",
        chunked.marginal
    );
}

/// At the natural degree the heuristic would have chosen full-zip anyway; the
/// explicit hint must not change that.
#[tokio::test]
async fn the_hint_is_harmless_above_the_heuristic_threshold() {
    const MAX_DEGREE: u32 = 64;
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "wide.idx");
    lance_vamana::io::write_partition(&store, &path, &sample_graph(MAX_DEGREE, VERTICES))
        .await
        .unwrap();

    assert_eq!(
        vertex_cost(store, &path).await.marginal,
        f64::from(MAX_DEGREE * 4)
    );
}

/// Several partitions with several vertices each: a one-row-per-partition
/// fixture would hide an off-by-one in the vertex range arithmetic.
#[tokio::test]
async fn vertices_are_addressed_independently_across_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(ObjectStore::local());
    let mut written = HashMap::new();

    for partition in 0..3usize {
        let path =
            Path::from_absolute_path(dir.path().join(format!("part_{partition:05}.idx"))).unwrap();
        let graph = sample_graph(64, 40 + partition * 7);
        lance_vamana::io::write_partition(&store, &path, &graph)
            .await
            .unwrap();
        written.insert(partition, (path, graph));
    }

    for (partition, (path, graph)) in &written {
        let reader = open_file(store.clone(), path, None).await.unwrap();
        assert_eq!(
            &read_partition(&reader).await.unwrap(),
            graph,
            "partition {partition} did not round trip"
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
        assert_eq!(row_ids, graph.row_ids()[middle].to_vec());
    }
}

/// Guard against the schema drifting away from what the layout needs.
#[tokio::test]
async fn the_written_schema_keeps_the_encoding_hint() {
    let schema = partition_schema(64).unwrap();
    let neighbors = schema.field_with_name(NEIGHBORS_COLUMN).unwrap();
    assert_eq!(
        neighbors.metadata().get(STRUCTURAL_ENCODING_META_KEY),
        Some(&STRUCTURAL_ENCODING_FULLZIP.to_string())
    );
}
