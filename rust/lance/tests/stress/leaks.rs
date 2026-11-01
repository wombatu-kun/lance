// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a step may leave behind: tokio tasks, temporary files, descriptors,
//! threads and memory.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_BYTES: AtomicIsize = AtomicIsize::new(0);

/// Counts the bytes the program holds. Unlike the resident set, the count does
/// not depend on what the allocator keeps for later.
struct CountingAllocator;

fn count_up(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes as isize, Ordering::Relaxed) + bytes as isize;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn count_down(bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes as isize, Ordering::Relaxed);
}

// SAFETY: every call goes to `System` with the arguments it came with; the
// counters do not touch the memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's guarantees are passed on.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            count_up(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's guarantees are passed on.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            count_up(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's guarantees are passed on.
        unsafe { System.dealloc(pointer, layout) };
        count_down(layout.size());
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's guarantees are passed on.
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            count_down(layout.size());
            count_up(new_size);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub fn live_heap_bytes() -> i64 {
    LIVE_BYTES.load(Ordering::Relaxed) as i64
}

/// Starts a new measure of the peak, and returns where it starts from.
pub fn start_heap_peak() -> i64 {
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(live, Ordering::Relaxed);
    live as i64
}

pub fn heap_peak_bytes() -> i64 {
    PEAK_BYTES.load(Ordering::Relaxed) as i64
}

/// Watches the temporary directory for the time of a step: spill files come
/// and go between any two checks of the step.
pub struct TmpWatch {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<u64>,
}

impl TmpWatch {
    pub fn start(tmp: &Path) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::spawn({
            let stop = stop.clone();
            let tmp = tmp.to_path_buf();
            move || {
                let mut peak = 0;
                while !stop.load(Ordering::Relaxed) {
                    let (mut files, mut dirs) = (Vec::new(), 0);
                    walk(&tmp, &mut files, &mut dirs);
                    let bytes: u64 = files
                        .iter()
                        .filter_map(|file| fs::metadata(file).ok())
                        .map(|meta| meta.len())
                        .sum();
                    peak = peak.max(bytes);
                    std::thread::sleep(Duration::from_millis(10));
                }
                peak
            }
        });
        Self { stop, thread }
    }

    /// The most bytes the directory held since the start.
    pub fn stop(self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.join().unwrap_or(0)
    }
}

pub struct Snapshot {
    /// Async tasks only: tokio exposes no stable count of blocking-pool work.
    pub tasks: usize,
    pub tmp_files: Vec<PathBuf>,
    pub tmp_dirs: usize,
    pub fds: usize,
    /// Recorded, not checked: Lance's thread pools start lazily.
    threads: usize,
    rss_kib: u64,
    hwm_kib: u64,
}

impl Snapshot {
    /// Must run on the runtime whose tasks it counts.
    pub fn take(tmp: &Path) -> Self {
        let mut tmp_files = Vec::new();
        let mut tmp_dirs = 0;
        walk(tmp, &mut tmp_files, &mut tmp_dirs);
        let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
        Self {
            tasks: tokio::runtime::Handle::current()
                .metrics()
                .num_alive_tasks(),
            tmp_files,
            tmp_dirs,
            fds: count_entries("/proc/self/fd"),
            threads: count_entries("/proc/self/task"),
            rss_kib: status_kib(&status, "VmRSS:"),
            hwm_kib: status_kib(&status, "VmHWM:"),
        }
    }

    pub fn line(&self) -> String {
        format!(
            "tasks={} tmp_files={} tmp_dirs={} fds={} threads={} rss_kib={} hwm_kib={}",
            self.tasks,
            self.tmp_files.len(),
            self.tmp_dirs,
            self.fds,
            self.threads,
            self.rss_kib,
            self.hwm_kib
        )
    }
}

/// Waits until no more than `baseline_tasks` tasks are alive and the
/// temporary directory is empty, or until `timeout` passes. Lance often
/// drops the last reference to a resource in a background task, so checking
/// right after an operation would see leaks that are not there.
pub async fn quiesce(baseline_tasks: usize, tmp: &Path, timeout: Duration) -> (Snapshot, Duration) {
    let started = Instant::now();
    loop {
        let snapshot = Snapshot::take(tmp);
        let quiet = snapshot.tasks <= baseline_tasks
            && snapshot.tmp_files.is_empty()
            && snapshot.tmp_dirs == 0;
        if quiet || started.elapsed() >= timeout {
            return (snapshot, started.elapsed());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Every file below a directory, whatever a listing through a store would show.
pub fn files_below(dir: &Path) -> Vec<PathBuf> {
    let (mut files, mut dirs) = (Vec::new(), 0);
    walk(dir, &mut files, &mut dirs);
    files
}

fn walk(dir: &Path, files: &mut Vec<PathBuf>, dirs: &mut usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if path.is_dir() {
            *dirs += 1;
            walk(&path, files, dirs);
        } else {
            files.push(path);
        }
    }
}

fn count_entries(dir: &str) -> usize {
    fs::read_dir(dir).map_or(0, |entries| entries.count())
}

fn status_kib(status: &str, key: &str) -> u64 {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}
