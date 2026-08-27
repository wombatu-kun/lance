// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! TEMPORARY measurement harness -- delete after the measurement.
//!
//! Question: is the peak live heap of a scan governed by `io_buffer_size`?
//!
//! Modes:
//!   write <uri>   generate a dataset
//!   scan  <uri>   scan it with a deliberately slow consumer and report peaks

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arrow_array::types::Float32Type;
use futures::StreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use tracing::Event;
use tracing::field::{Field, Visit};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

// ---------------------------------------------------------------- peak heap

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn on_alloc(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

#[inline]
fn on_free(size: usize) {
    LIVE.fetch_sub(size, Ordering::Relaxed);
}

struct PeakAlloc;

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            on_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        on_free(layout.size());
        System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            on_alloc(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                on_alloc(new_size - layout.size());
            } else {
                on_free(layout.size() - new_size);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

// ------------------------------------------------- scheduler state observer

static SCHED_EVENTS: AtomicU64 = AtomicU64::new(0);
static MAX_RESERVED: AtomicU64 = AtomicU64::new(0);
static IO_BUF_SEEN: AtomicU64 = AtomicU64::new(0);

struct SchedVisitor;

impl Visit for SchedVisitor {
    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "bytes_reserved" && value > 0 {
            MAX_RESERVED.fetch_max(value as u64, Ordering::Relaxed);
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "bytes_reserved" => {
                MAX_RESERVED.fetch_max(value, Ordering::Relaxed);
            }
            "io_buffer_size_bytes" => {
                IO_BUF_SEEN.store(value, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

struct SchedLayer;

impl<S: tracing::Subscriber> Layer<S> for SchedLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != "lance_io::scheduler::state" {
            return;
        }
        SCHED_EVENTS.fetch_add(1, Ordering::Relaxed);
        event.record(&mut SchedVisitor);
    }
}

// ------------------------------------------------------------------ helpers

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn vm_hwm_bytes() -> u64 {
    let status = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// --------------------------------------------------------------------- main

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(SchedLayer)
        .with(EnvFilter::new("lance_io::scheduler::state=trace"))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let uri = args.get(2).cloned().unwrap_or_default();
    if uri.is_empty() {
        eprintln!("usage: io_buffer_peak <write|scan> <uri>");
        std::process::exit(2);
    }

    match mode.as_str() {
        "write" => do_write(&uri).await,
        "scan" => do_scan(&uri).await,
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }
}

async fn do_write(uri: &str) {
    let dim = env_usize("HARNESS_DIM", 1024) as u32;
    let rows_per_batch = env_usize("HARNESS_ROWS_PER_BATCH", 8192) as u64;
    let batches = env_usize("HARNESS_BATCHES", 256) as u32;
    let rows_per_file = env_usize("HARNESS_ROWS_PER_FILE", 262_144);

    let total_rows = rows_per_batch * batches as u64;
    let bytes = total_rows * dim as u64 * 4;
    println!(
        "writing {} rows x {} dims = {:.2} GiB to {}",
        total_rows,
        dim,
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        uri
    );

    let reader = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(dim)))
        .into_reader_rows(RowCount::from(rows_per_batch), BatchCount::from(batches));

    let params = WriteParams {
        mode: WriteMode::Overwrite,
        max_rows_per_file: rows_per_file,
        ..Default::default()
    };

    let start = Instant::now();
    let ds = Dataset::write(reader, uri, Some(params))
        .await
        .expect("write failed");
    println!(
        "wrote {} rows in {} fragments in {:.1}s",
        ds.count_rows(None).await.unwrap(),
        ds.get_fragments().len(),
        start.elapsed().as_secs_f64()
    );
}

async fn do_scan(uri: &str) {
    let sleep_ms = env_usize("HARNESS_SLEEP_MS", 20) as u64;
    let batch_size = env_usize("HARNESS_BATCH_SIZE", 8192);
    let max_batches = env_usize("HARNESS_MAX_BATCHES", 0);

    let open_start = Instant::now();
    let ds = DatasetBuilder::from_uri(uri)
        .load()
        .await
        .expect("open failed");
    let open_secs = open_start.elapsed().as_secs_f64();
    let peak_open = PEAK.load(Ordering::Relaxed) as u64;
    let live_open = LIVE.load(Ordering::Relaxed) as u64;

    let mut scanner = ds.scan();
    scanner.batch_size(batch_size);
    if let Ok(v) = std::env::var("HARNESS_BATCH_READAHEAD") {
        scanner.batch_readahead(v.parse().unwrap());
    }
    if let Ok(v) = std::env::var("HARNESS_FRAG_READAHEAD") {
        scanner.fragment_readahead(v.parse().unwrap());
    }

    let scan_start = Instant::now();
    let mut stream = scanner.try_into_stream().await.expect("stream failed");
    let mut rows: u64 = 0;
    let mut batches: u64 = 0;
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("batch failed");
        rows += batch.num_rows() as u64;
        batches += 1;
        if max_batches > 0 && batches as usize >= max_batches {
            break;
        }
        if sleep_ms > 0 {
            tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
        }
    }
    let scan_secs = scan_start.elapsed().as_secs_f64();

    let peak_total = PEAK.load(Ordering::Relaxed) as u64;
    println!(
        "RESULT io_buffer_env={} io_buffer_seen_mib={:.1} sched_events={} \
max_bytes_reserved_mib={:.1} peak_open_mib={:.1} live_open_mib={:.1} \
peak_total_mib={:.1} vm_hwm_mib={:.1} rows={} batches={} open_s={:.2} scan_s={:.2} \
sleep_ms={} batch_size={}",
        std::env::var("LANCE_DEFAULT_IO_BUFFER_SIZE").unwrap_or_else(|_| "unset".into()),
        mib(IO_BUF_SEEN.load(Ordering::Relaxed)),
        SCHED_EVENTS.load(Ordering::Relaxed),
        mib(MAX_RESERVED.load(Ordering::Relaxed)),
        mib(peak_open),
        mib(live_open),
        mib(peak_total),
        mib(vm_hwm_bytes()),
        rows,
        batches,
        open_secs,
        scan_secs,
        sleep_ms,
        batch_size,
    );
}
