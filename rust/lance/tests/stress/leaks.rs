// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a step may leave behind: tokio tasks, temporary files, descriptors,
//! threads and memory.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct Snapshot {
    /// Async tasks only: tokio exposes no stable count of blocking-pool work.
    pub tasks: usize,
    pub tmp_files: Vec<PathBuf>,
    tmp_dirs: usize,
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
/// temporary directory holds no file, or until `timeout` passes. Lance often
/// drops the last reference to a resource in a background task, so checking
/// right after an operation would see leaks that are not there.
pub async fn quiesce(baseline_tasks: usize, tmp: &Path, timeout: Duration) -> (Snapshot, Duration) {
    let started = Instant::now();
    loop {
        let snapshot = Snapshot::take(tmp);
        let quiet = snapshot.tasks <= baseline_tasks && snapshot.tmp_files.is_empty();
        if quiet || started.elapsed() >= timeout {
            return (snapshot, started.elapsed());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
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
