// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What storage is asked for when a traversal wants one vertex.
//!
//! ```text
//! cd rust/lance-vamana
//! SIFT_DIR=~/datasets/sift cargo run --release --example lazy_read_probe
//! ```
//!
//! Environment: `SIFT_DIR` (required), `VECTORS` (default 100000, `0` for all),
//! `PARTITIONS` (default 10), `DEGREE` (default 64), `SCATTER` (default 256),
//! `SEED` (default 42).
//!
//! The gate for phase D turns on one number the plan assumes rather than
//! measures: what a reader asks storage for when it wants a single vertex
//! instead of a whole partition. Everything else - how much of a partition a
//! walk touches, how many queries one load amortises over - is arithmetic on top
//! of it.
//!
//! Both `__neighbors` and `__vector` are written full-zip so that a vertex is one
//! addressable stride, but full-zip is a property of a *column*, and a Lance file
//! stores each column in a region of its own. So a vertex may be two ranged reads
//! far apart rather than one contiguous `8 + 4R + 4d` bytes, and the coalescing
//! window differs per column because the strides do. Which of those it is decides
//! whether a lazy traversal is worth writing.
//!
//! **Bytes and iops, not wall clock.** The numbers come off the scheduler's own
//! [`ScanStats`], which counts what the reader asked for - exactly the model input
//! the gate needs. Elapsed time is printed beside them and must not be read as
//! storage latency: these files were written by this process moments earlier, so
//! every read is served from the page cache, and dropping it needs root. Turning
//! bytes and iops into time is the gate's job, over a stated range of device
//! parameters.
//!
//! The row to read first is `break-even`: how much of a partition a walk may
//! touch before reading it vertex by vertex has asked storage for more bytes than
//! reading it whole would have.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_io::scheduler::ScanScheduler;
use lance_linalg::distance::DistanceType;
use lance_vamana::build::BuildParams;
use lance_vamana::builder::{IndexParams, create_index};
use lance_vamana::format::{INDEX_FILE_NAME, NEIGHBORS_COLUMN, ROW_ID_COLUMN, VECTOR_COLUMN};
use lance_vamana::io::{open_file, read_partition, read_rows, read_segment, scan_scheduler};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::index::sample;

#[path = "common/mod.rs"]
mod common;
use common::{env_usize, read_fvecs};

const VECTOR_FIELD: &str = "vector";
const INDEX_NAME: &str = "vamana_idx";
const DISTANCE_TYPE: DistanceType = DistanceType::L2;

/// What one read asked storage for.
struct Cost {
    bytes: u64,
    iops: u64,
    requests: u64,
    micros: u128,
}

impl Cost {
    fn per_row(&self, rows: usize) -> f64 {
        self.bytes as f64 / rows as f64
    }
}

/// Measure one read against the scheduler's counters.
async fn cost<F, T>(scheduler: &Arc<ScanScheduler>, read: F) -> Cost
where
    F: std::future::Future<Output = T>,
{
    let before = scheduler.stats();
    let started = Instant::now();
    let value = read.await;
    let micros = started.elapsed().as_micros();
    let after = scheduler.stats();
    drop(value);
    Cost {
        bytes: after.bytes_read - before.bytes_read,
        iops: after.iops - before.iops,
        requests: after.requests - before.requests,
        micros,
    }
}

/// Read a set of single rows as one request, the way a lazy traversal would.
///
/// [`ReadBatchParams::Ranges`] rather than a call per row: the scheduler
/// coalesces adjacent ranges in one pass and does not sort, so the ranges have to
/// arrive ascending. What that coalescing is worth is the difference between this
/// probe and the one that issues a call per row.
async fn read_scattered(reader: &FileReader, rows: &[usize]) -> usize {
    let ranges = rows
        .iter()
        .map(|row| *row as u64..*row as u64 + 1)
        .collect::<Vec<Range<u64>>>();
    let batches = reader
        .read_stream(
            ReadBatchParams::Ranges(ranges.into()),
            u32::MAX,
            1,
            FilterExpression::no_filter(),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    batches.iter().map(RecordBatch::num_rows).sum()
}

async fn write_dataset(uri: &str, vectors: FixedSizeListArray) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        VECTOR_FIELD,
        vectors.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams::default()),
    )
    .await
    .unwrap()
}

#[tokio::main]
async fn main() {
    let dir =
        std::env::var("SIFT_DIR").expect("set SIFT_DIR to the directory holding sift_*.fvecs");
    let (base, dim, total) = read_fvecs(&format!("{dir}/sift_base.fvecs"));
    let requested = env_usize("VECTORS", 100_000);
    let rows = if requested == 0 {
        total
    } else {
        requested.min(total)
    };
    let partitions = env_usize("PARTITIONS", 10) as u32;
    let degree = env_usize("DEGREE", 64) as u32;
    let scatter = env_usize("SCATTER", 256);
    let seed = env_usize("SEED", 42) as u64;

    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from(base[..rows * dim].to_vec()),
        dim as i32,
    )
    .unwrap();

    let temp = tempfile::tempdir().unwrap();
    let uri = temp.path().to_str().unwrap();
    let mut dataset = write_dataset(uri, vectors).await;
    let started = Instant::now();
    create_index(
        &mut dataset,
        INDEX_NAME,
        &IndexParams::new(VECTOR_FIELD, partitions)
            .with_distance_type(DISTANCE_TYPE)
            .with_graph_params(BuildParams {
                max_degree: degree,
                ..Default::default()
            }),
    )
    .await
    .unwrap();
    println!(
        "SIFT {rows} x {dim}, {partitions} partitions, R = {degree}: indexed in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let index = dataset
        .load_indices_by_name(INDEX_NAME)
        .await
        .unwrap()
        .into_iter()
        .next()
        .expect("the index was just created");
    let segment_dir = dataset.indices_dir().join(index.uuid.to_string());
    let file_sizes = index
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
    // The largest partition, because the question is what a lazy read saves and
    // the saving is proportional to what reading whole costs.
    let entry = manifest
        .partitions()
        .iter()
        .max_by_key(|entry| entry.num_rows)
        .expect("a segment over rows has partitions")
        .clone();
    let path = segment_dir.clone().join(entry.file.as_str());
    let size = file_sizes.get(&entry.file).copied();
    let num_rows = entry.num_rows as usize;
    let stride = 8 + 4 * degree as usize + 4 * dim;
    println!(
        "probing partition {} of {} vertices: {} bytes on disk, {stride} bytes of payload a vertex",
        entry.partition_id,
        num_rows,
        size.unwrap_or_default(),
    );

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut scattered = sample(&mut rng, num_rows, scatter.min(num_rows)).into_vec();
    scattered.sort_unstable();
    let contiguous = 16.min(num_rows);
    let middle = num_rows / 2;

    let all_columns: Option<&[&str]> = None;
    let walk_columns = Some([NEIGHBORS_COLUMN, VECTOR_COLUMN].as_slice());
    let edge_columns = Some([NEIGHBORS_COLUMN].as_slice());
    let id_columns = Some([ROW_ID_COLUMN].as_slice());

    println!(
        "\n{:<34} {:>7} {:>12} {:>6} {:>9} {:>10} {:>11}",
        "read", "rows", "bytes", "iops", "requests", "us (warm)", "bytes/row"
    );

    let open = cost(&scheduler, async {
        open_file(&scheduler, &path, all_columns, size)
            .await
            .unwrap()
    })
    .await;
    report("open the file (every column)", 1, &open);

    let reader = open_file(&scheduler, &path, all_columns, size)
        .await
        .unwrap();
    let whole = cost(&scheduler, read_partition(&reader, entry.num_rows)).await;
    report("whole partition, every column", num_rows, &whole);

    let one_all = cost(&scheduler, read_rows(&reader, middle..middle + 1)).await;
    report("one vertex, every column", 1, &one_all);

    let contiguous_all = cost(&scheduler, read_rows(&reader, middle..middle + contiguous)).await;
    report(
        &format!("{contiguous} adjacent vertices, every column"),
        contiguous,
        &contiguous_all,
    );

    let coalesced_all = cost(&scheduler, read_scattered(&reader, &scattered)).await;
    report(
        &format!("{} scattered vertices, one request", scattered.len()),
        scattered.len(),
        &coalesced_all,
    );

    let separate_all = cost(&scheduler, async {
        for row in &scattered {
            read_rows(&reader, *row..*row + 1).await.unwrap();
        }
    })
    .await;
    report(
        &format!("{} scattered vertices, a call each", scattered.len()),
        scattered.len(),
        &separate_all,
    );

    // Projected: what a traversal that keeps codes in RAM would ask for, and
    // what one that keeps nothing would.
    let walk_open = open_file(&scheduler, &path, walk_columns, size)
        .await
        .unwrap();
    let walk_whole = cost(&scheduler, read_rows(&walk_open, 0..num_rows)).await;
    report("whole partition, edges + vectors", num_rows, &walk_whole);

    let walk_one = cost(&scheduler, read_rows(&walk_open, middle..middle + 1)).await;
    report("one vertex, edges + vectors", 1, &walk_one);

    let walk_scattered = cost(&scheduler, read_scattered(&walk_open, &scattered)).await;
    report(
        &format!("{} scattered, edges + vectors", scattered.len()),
        scattered.len(),
        &walk_scattered,
    );

    let edge_open = open_file(&scheduler, &path, edge_columns, size)
        .await
        .unwrap();
    let edge_whole = cost(&scheduler, read_rows(&edge_open, 0..num_rows)).await;
    report("whole partition, edges only", num_rows, &edge_whole);

    let edge_one = cost(&scheduler, read_rows(&edge_open, middle..middle + 1)).await;
    report("one vertex, edges only", 1, &edge_one);

    let edge_scattered = cost(&scheduler, read_scattered(&edge_open, &scattered)).await;
    report(
        &format!("{} scattered, edges only", scattered.len()),
        scattered.len(),
        &edge_scattered,
    );

    let id_open = open_file(&scheduler, &path, id_columns, size)
        .await
        .unwrap();
    let id_whole = cost(&scheduler, read_rows(&id_open, 0..num_rows)).await;
    report("whole partition, row ids only", num_rows, &id_whole);

    println!("\nwhat it means");
    let payload = (stride * num_rows) as f64;
    println!(
        "  a vertex of payload {stride} B costs {:.0} B of request on its own, {:.1}x its payload",
        one_all.per_row(1),
        one_all.per_row(1) / stride as f64,
    );
    println!(
        "  reading {contiguous} adjacent vertices costs {:.0} B a vertex, so a request spans about \
         {:.1} vertices",
        contiguous_all.per_row(contiguous),
        one_all.bytes as f64 / contiguous_all.per_row(contiguous),
    );
    println!(
        "  one coalesced request for {} scattered vertices: {} iops against {} for a call each",
        scattered.len(),
        coalesced_all.iops,
        separate_all.iops,
    );
    println!(
        "  the whole partition is {} B of request for {:.0} B of payload ({:.2}x)",
        whole.bytes,
        payload,
        whole.bytes as f64 / payload,
    );
    for (label, per_vertex, whole_bytes) in [
        (
            "every column",
            coalesced_all.per_row(scattered.len()),
            whole.bytes,
        ),
        (
            "edges + vectors",
            walk_scattered.per_row(scattered.len()),
            walk_whole.bytes,
        ),
        (
            "edges only",
            edge_scattered.per_row(scattered.len()),
            edge_whole.bytes,
        ),
    ] {
        let vertices = whole_bytes as f64 / per_vertex;
        println!(
            "  break-even ({label}): {vertices:.0} of {num_rows} vertices, {:.1}% of the partition",
            100.0 * vertices / num_rows as f64,
        );
    }
}

fn report(label: &str, rows: usize, cost: &Cost) {
    println!(
        "{label:<34} {rows:>7} {:>12} {:>6} {:>9} {:>10} {:>11.0}",
        cost.bytes,
        cost.iops,
        cost.requests,
        cost.micros,
        cost.per_row(rows),
    );
}
