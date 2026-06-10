// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

/// A sink that records I/O requests as they are submitted to storage.
///
/// This lives in `lance-core` so that the encoding layer (`lance-encoding`) and
/// the I/O layer (`lance-io`) can both refer to it without depending on one
/// another.  It lets a caller attach a lightweight counter to a file reader and
/// measure the exact bytes/IOPS performed for a bounded scope (e.g. a single
/// query); see `lance_io::scheduler::IoStats` for the concrete implementation.
pub trait IoStatsRecorder: std::fmt::Debug + Send + Sync {
    /// Record one completed request, given the byte ranges as actually
    /// submitted to storage (i.e. after any coalescing/splitting), so the
    /// counts reflect physical I/O.
    fn record_request(&self, ranges: &[Range<u64>]);
}
