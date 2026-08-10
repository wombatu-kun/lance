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

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt64Array};
use arrow_schema::{Fields, Schema as ArrowSchema};
use lance_core::utils::io_stats::IoStatsRecorder;
use lance_encoding::constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::ObjectStore;
use lance_vamana::format::{NEIGHBORS_COLUMN, ROW_ID_COLUMN, VECTOR_COLUMN, partition_schema};
use lance_vamana::io::{
    SEGMENT_FILE_VERSION, open_file, read_partition, read_rows, scan_scheduler,
};
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
    let reader = open_file(&scan_scheduler(&store), path, Some(columns), None)
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

async fn vertex_cost(store: Arc<ObjectStore>, path: &Path, columns: &[&str]) -> VertexCost {
    vertex_cost_at(store, path, columns, 0).await
}

async fn vertex_cost_at(
    store: Arc<ObjectStore>,
    path: &Path,
    columns: &[&str],
    start: usize,
) -> VertexCost {
    let one = bytes_to_read(store.clone(), path, columns, start..start + 1).await;
    let five = bytes_to_read(store, path, columns, start..start + 5).await;
    VertexCost {
        marginal: (five as f64 - one as f64) / 4.0,
        single: one,
    }
}

/// The same partition written through a nullable schema, optionally with one
/// vertex's neighbour list actually set to null.
async fn write_nullable(store: &ObjectStore, path: &Path, partition: &Partition, hole: bool) {
    let hinted = partition.to_batch().unwrap();
    let neighbors = hinted[NEIGHBORS_COLUMN].as_fixed_size_list();
    let width = neighbors.value_length() as usize;
    let slots = neighbors
        .values()
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    let rows = (0..partition.len()).map(|vertex| {
        if hole && vertex == partition.len() / 2 {
            None
        } else {
            Some(
                slots[vertex * width..(vertex + 1) * width]
                    .iter()
                    .map(|neighbor| Some(*neighbor))
                    .collect::<Vec<_>>(),
            )
        }
    });
    let neighbors = FixedSizeListArray::from_iter_primitive::<UInt32Type, _, _>(
        rows.collect::<Vec<_>>(),
        width as i32,
    );

    let fields = hinted
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let field = field.as_ref().clone().with_nullable(true);
            if field.name() == NEIGHBORS_COLUMN {
                Arc::new(field.with_data_type(neighbors.data_type().clone()))
            } else {
                Arc::new(field)
            }
        })
        .collect::<Fields>();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            hinted.column(0).clone(),
            Arc::new(neighbors),
            hinted.column(2).clone(),
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
    writer.finish().await.unwrap();
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

    let reader = open_file(&scan_scheduler(&store), &path, None, None)
        .await
        .unwrap();
    assert_eq!(
        read_partition(&reader, partition.len() as u32)
            .await
            .unwrap(),
        partition
    );

    // The row count lives in `index.idx` and the rows live here, so a reader
    // that took the file's word for it would believe a damaged footer.
    let error = read_partition(&reader, partition.len() as u32 + 1)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("segment table lists"), "{error}");

    // Zero is not a partition. The format gives an empty partition no file and
    // no row in the table, so nothing on the read path should be able to ask for
    // one - and a caller who does gets an error rather than a partition
    // conjured out of the file's schema.
    let error = read_partition(&reader, 0).await.unwrap_err();
    assert!(error.to_string().contains("segment table lists"), "{error}");
    let error = read_rows(&reader, 0..0).await.unwrap_err();
    assert!(error.to_string().contains("selects nothing"), "{error}");
}

/// The size a caller declares is the size the reader uses. That is what lets a
/// query open a partition without first asking storage how big it is, and it is
/// only worth passing if a wrong one is refused rather than quietly re-probed -
/// otherwise the parameter would be decoration and the probe would still happen.
#[tokio::test]
async fn a_declared_file_size_is_the_one_used() {
    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "part_00000.idx");
    let partition = sample_partition(64, 128, DIMENSION);
    let size = lance_vamana::io::write_partition(&store, &path, &partition)
        .await
        .unwrap();

    let scheduler = scan_scheduler(&store);
    let reader = open_file(&scheduler, &path, None, Some(size))
        .await
        .unwrap();
    assert_eq!(reader.metadata().num_rows, partition.len() as u64);

    assert!(
        open_file(&scheduler, &path, None, Some(size / 2))
            .await
            .is_err(),
        "the declared size was ignored, so nothing was saved by declaring it"
    );
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
    // `both > neighbors` follows from the marginal costs above and so cannot
    // fail. What the single-vertex numbers can still show is the *size* of the
    // gap: adding the vector column must cost at least one vector and at most
    // what fetching that column on its own costs, or the projection is reading
    // something other than the column it was asked for.
    assert!(
        (neighbors.single + u64::from(DIMENSION * 4)..=neighbors.single + vector.single)
            .contains(&both.single),
        "one vertex costs {} B for edges and {} B for the vector, but {} B for both",
        neighbors.single,
        vector.single,
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
        // Row ids that name their own partition. `sample_partition` is a pure
        // function of its arguments, so without this every partition holds the
        // same row ids and the same vectors at the same offsets - and the middle
        // slice below would read the same bytes out of any of the three files.
        let (graph, vectors) = sample_partition(64, 40 + partition_id * 7, DIMENSION).into_parts();
        let adjacency = (0..graph.len())
            .map(|vertex| graph.neighbors(vertex as u32).unwrap().to_vec())
            .collect::<Vec<_>>();
        let row_ids = graph
            .row_ids()
            .iter()
            .map(|row_id| row_id + ((partition_id as u64) << 32))
            .collect::<Vec<_>>();
        let partition = lance_vamana::partition::Partition::try_new(
            lance_vamana::partition::PartitionGraph::try_new(
                graph.max_degree(),
                row_ids,
                adjacency,
            )
            .unwrap(),
            vectors,
        )
        .unwrap();
        lance_vamana::io::write_partition(&store, &path, &partition)
            .await
            .unwrap();
        written.insert(partition_id, (path, partition));
    }

    for (partition_id, (path, partition)) in &written {
        let reader = open_file(&scan_scheduler(&store), path, None, None)
            .await
            .unwrap();
        assert_eq!(
            &read_partition(&reader, partition.len() as u32)
                .await
                .unwrap(),
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

/// The stride has to hold everywhere in the file, not only at its head.
///
/// Every other measurement here reads vertices 0..5, and Lance flushes a column
/// into a new page once it has buffered enough of it - so a stride exact at the
/// start and drifting across a page boundary would be invisible. This file is
/// sized past that threshold and measured at both ends.
#[tokio::test]
async fn the_stride_holds_past_a_page_boundary() {
    // 8 MiB per column is the default flush threshold, so 128 B per vertex needs
    // well over 65k vertices to reach a second page.
    const MANY: usize = 200_000;

    let dir = tempfile::tempdir().unwrap();
    let (store, path) = local_store_and_path(&dir, "many.idx");
    lance_vamana::io::write_partition(
        &store,
        &path,
        &sample_partition(MAX_DEGREE, MANY, DIMENSION),
    )
    .await
    .unwrap();

    let head = vertex_cost_at(store.clone(), &path, &[NEIGHBORS_COLUMN], 0).await;
    let tail = vertex_cost_at(store.clone(), &path, &[NEIGHBORS_COLUMN], MANY - 8).await;
    println!(
        "neighbours: head marginal={} tail marginal={}",
        head.marginal, tail.marginal
    );
    assert_eq!(head.marginal, NEIGHBOR_STRIDE);
    assert_eq!(
        tail.marginal, NEIGHBOR_STRIDE,
        "the stride drifted down the file"
    );

    let tail = vertex_cost_at(store, &path, &[VECTOR_COLUMN], MANY - 8).await;
    assert_eq!(
        tail.marginal, VECTOR_STRIDE,
        "the stride drifted down the file"
    );
}

/// The layout's other precondition, measured rather than assumed - and it turns
/// out to be narrower than "the column must not be nullable".
///
/// A nullable *flag* costs nothing: Lance downgrades a validity bitmap with no
/// nulls in it back to the no-null case, so the stride survives. What costs is an
/// actual null, which adds a control word to every value in the column. The
/// non-nullable field is therefore not the thing that buys the stride; it is what
/// makes a null impossible to write in the first place, and Arrow enforces it.
#[tokio::test]
async fn a_null_costs_the_stride_but_a_nullable_flag_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let partition = sample_partition(MAX_DEGREE, VERTICES, DIMENSION);

    let (store, permissive) = local_store_and_path(&dir, "nullable.idx");
    write_nullable(&store, &permissive, &partition, false).await;
    let permissive = vertex_cost(store.clone(), &permissive, &[NEIGHBORS_COLUMN]).await;

    let (_, holed) = local_store_and_path(&dir, "with_a_null.idx");
    write_nullable(&store, &holed, &partition, true).await;
    let holed = vertex_cost(store, &holed, &[NEIGHBORS_COLUMN]).await;

    println!(
        "nullable flag only: marginal={}\nwith one null:      marginal={}",
        permissive.marginal, holed.marginal
    );
    assert_eq!(
        permissive.marginal, NEIGHBOR_STRIDE,
        "a validity bitmap with no nulls in it should have been dropped"
    );
    assert!(
        holed.marginal > NEIGHBOR_STRIDE,
        "one null must widen every value; got {} against a stride of {NEIGHBOR_STRIDE}",
        holed.marginal
    );
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
