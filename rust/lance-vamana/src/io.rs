// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading and writing a segment: `index.idx` plus one file per partition.
//!
//! Everything here goes through the published `lance-file` / `lance-io` crates,
//! so every file we write is an ordinary Lance file that Lance's own reader can
//! open. Nothing in this module needs the `lance` crate.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::{FixedSizeListArray, RecordBatch};
use arrow_select::concat::concat_batches;
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::{create_writer, reader_projection_from_column_names};
use lance_file::writer::FileWriterOptions;
use lance_index::pb;
use lance_index::vector::ivf::storage::IvfModel;
use lance_io::ReadBatchParams;
use lance_io::local::to_local_path;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{FileScheduler, IoStats, ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use prost::Message;

use crate::cache::FileKey;
use crate::codes::encode;
use crate::format::{
    INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, IndexMetadata, VECTOR_COLUMN,
    index_schema, partition_file_name,
};
use crate::partition::{Partition, row_ids_from_batch};
use crate::raw::{self, VectorLayout};
use crate::segment::{PartitionEntry, SegmentManifest};

/// The file format every file in a segment is written in.
///
/// Pinned rather than inferred: the constant-stride layout the whole design
/// rests on is a property of a specific structural encoding, so the writer must
/// not drift onto another version silently.
pub const SEGMENT_FILE_VERSION: ConcreteFileVersion = ConcreteFileVersion::V2_1;

/// Write one partition and return the size of the file in bytes.
pub async fn write_partition(
    store: &ObjectStore,
    path: &Path,
    partition: &Partition,
    codes: Option<&FixedSizeListArray>,
) -> Result<u64> {
    // The format says an empty partition gets no row in `index.idx` and no file.
    // `SegmentWriter` enforces that; this function is public and delegated to, so
    // it has to enforce it too rather than write a file nothing can point at.
    if partition.is_empty() {
        return Err(Error::invalid_input(
            "Vamana will not write a file for an empty partition".to_string(),
        ));
    }
    let batch = partition.to_batch(codes)?;
    let schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref())?;
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store.create(path).await?,
        schema,
        FileWriterOptions::default(),
    )?;
    writer.write_batch(&batch).await?;
    Ok(writer.finish().await?.size_bytes)
}

/// The one scheduler an index reads through.
///
/// One per index open, never one per file, which is what Lance's own vector
/// index does. A scheduler spawns a background task, so one per partition read
/// would be one background task per partition read.
///
/// It is *not* what keeps the working set bounded, despite declaring a byte
/// budget of `32 MiB * io_parallelism`. Every file here is opened at base
/// priority 0, and a task's priority is `(base << 64) | top_level_row`, so the
/// first page of every partition of every segment has priority exactly 0.
/// `can_deliver_without_warning` admits a task unconditionally when its priority
/// is at or below the minimum in flight, which zero always is, so the
/// byte-budget branch is never reached and `bytes_avail` simply goes negative
/// with a `log::debug!`. What actually bounds a query's working set is
/// `PARTITIONS_IN_FLIGHT`.
pub fn scan_scheduler(store: &Arc<ObjectStore>) -> Arc<ScanScheduler> {
    ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(store))
}

/// One file of a segment, opened once and projected as often as wanted.
///
/// A projection is fixed when a reader is built, and a lazy walk reads three
/// different sets of columns out of one partition: the codes it steers by, then
/// the edges of every vertex it expands, then the vectors of what it ended up
/// with. Opening the file once per projection would re-read the footer for each,
/// and the footer is a round trip - which is the currency the lazy path is
/// spending to save bytes, so it must not spend three where one will do.
pub struct PartitionFile {
    path: Path,
    file: FileScheduler,
    /// The unprojected reader: where the file metadata every projection is built
    /// from comes from, and the answer when a caller wants every column.
    reader: FileReader,
    /// Where the vectors are, worked out from the footer the first time a
    /// re-score asks and shared by every clone of this file afterwards.
    /// `None` inside the cell means this file is not addressable and the
    /// decoder has to do the reading.
    vectors: Arc<OnceLock<Option<VectorLayout>>>,
    /// How this file re-scores when its store is one whose objects are local
    /// files and a caller asked for that. See [`Self::with_local_reads`].
    local: Option<LocalReads>,
}

/// A partition file open for reading without going through the scheduler.
///
/// The scheduler's job on a local file is to bound how many reads are in flight
/// and to move each one onto the blocking pool. For a re-score neither buys
/// anything: it wants twenty ranges of a few kilobytes that the page cache
/// almost always holds, and it pays a queue, a task and a wakeup for each of
/// them. Reading them here costs one `pread` each and no hop at all.
///
/// What it costs instead is honesty about where the read happens: a `pread` of a
/// page the kernel does not have blocks the thread it runs on, which is a tokio
/// worker. Warm that is about a microsecond; cold it is about a hundred. The
/// path that would make this unconditionally safe is running the whole resident
/// query off the runtime, which is a change of its own.
#[derive(Clone)]
struct LocalReads {
    /// The scheduler's own coalescing parameters, kept so that reading by hand
    /// moves the same bytes in the same number of reads.
    block_size: u64,
    max_iop_size: u64,
    /// Where these reads are added to the index's running totals. The scheduler
    /// keeps its own and never sees these, so an index that did not keep them
    /// here would report less than it read.
    totals: Arc<IoStats>,
    /// Opened on the first re-score and shared by every clone of this file.
    /// Lazily, for two reasons: the whole-partition modes never re-score and
    /// would hold a descriptor for nothing, and opening one is a blocking call
    /// that has no business on a runtime worker.
    file: Arc<OnceLock<std::fs::File>>,
}

/// Fill `buf` from `offset`, whatever the platform calls it.
#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(not(unix))]
fn read_at(_file: &std::fs::File, _buf: &mut [u8], _offset: u64) -> std::io::Result<()> {
    // Unreachable: `with_local_reads` binds no file off Unix.
    Err(std::io::ErrorKind::Unsupported.into())
}

impl PartitionFile {
    /// Open `path`, reading its footer once.
    ///
    /// `size_bytes` skips the size probe when the caller already knows the
    /// answer - Lance records the size of every file of a committed index in the
    /// dataset manifest, so at query time it always does.
    /// `stats` records everything read through this file, on top of the
    /// scheduler's own totals; see [`Self::open_with`].
    pub async fn open(
        scheduler: &Arc<ScanScheduler>,
        path: &Path,
        size_bytes: Option<u64>,
        stats: Option<&IoStats>,
    ) -> Result<Self> {
        Self::open_with(scheduler, path, size_bytes, None, stats).await
    }

    /// Open `path`, taking its layout from `cache` if a query has read it before.
    ///
    /// The footer is a round trip before a single vertex can be fetched, and a
    /// partition file is immutable - maintenance writes a new segment under a
    /// new uuid rather than editing one - so a query that probes the same
    /// partition as an earlier query is re-reading a byte-for-byte identical
    /// answer.
    pub async fn open_cached(
        scheduler: &Arc<ScanScheduler>,
        path: &Path,
        size_bytes: Option<u64>,
        cache: &LanceCache,
        stats: Option<&IoStats>,
    ) -> Result<Self> {
        Self::open_with(scheduler, path, size_bytes, Some(cache), stats).await
    }

    /// `stats` is a secondary sink the scheduler records into beside its own
    /// running totals, which is how a caller measures the bytes of a *scope*
    /// rather than of an index: the handle is attached before the footer is read
    /// and is carried by every clone of it, so one sink covers the footer, the
    /// resident codes and every hop a walk takes. [`Self::project_into`] is how
    /// the other half of a query gets counted apart from that.
    async fn open_with(
        scheduler: &Arc<ScanScheduler>,
        path: &Path,
        size_bytes: Option<u64>,
        cache: Option<&LanceCache>,
        stats: Option<&IoStats>,
    ) -> Result<Self> {
        let size = size_bytes.map_or_else(CachedFileSize::unknown, CachedFileSize::new);
        let file = scheduler.open_file(path, &size).await?;
        let file = match stats {
            Some(stats) => file.with_io_stats(stats.recorder()),
            None => file,
        };
        // Only the cached arm goes near a key, because the uncached one is what
        // every build and maintenance pass takes, and those open a file once
        // each: hashing a path and weighing the metadata would be pure overhead
        // there.
        let metadata = match cache {
            Some(cache) => {
                cache
                    .get_or_insert_with_key(FileKey { path }, || {
                        FileReader::read_all_metadata(&file)
                    })
                    .await?
            }
            None => Arc::new(FileReader::read_all_metadata(&file).await?),
        };
        // The version is pinned on the way out and therefore has to be checked
        // on the way in. It is not a formality: a projection is computed against
        // the structural grammar of [`SEGMENT_FILE_VERSION`], and a file written
        // under another one lays its columns out differently - the read would
        // succeed and return the wrong bytes rather than fail.
        if metadata.version() != SEGMENT_FILE_VERSION {
            return Err(Error::corrupt_file_named(
                path.filename().unwrap_or(INDEX_FILE_NAME),
                format!(
                    "Vamana segment file is a Lance {} file, and this crate writes and reads {}",
                    metadata.version(),
                    SEGMENT_FILE_VERSION
                ),
            ));
        }
        let options = FileReaderOptions::default();
        let reader = FileReader::try_open_with_file_metadata(
            Arc::new(
                LanceEncodingsIo::new(file.clone()).with_read_chunk_size(options.read_chunk_size),
            ),
            path.clone(),
            None,
            Arc::<DecoderPlugins>::default(),
            metadata,
            &LanceCache::no_cache(),
            options,
        )
        .await?;
        Ok(Self {
            path: path.clone(),
            file,
            reader,
            vectors: Arc::new(OnceLock::new()),
            local: None,
        })
    }

    /// A reader that decodes `columns` and nothing else.
    ///
    /// Built from the metadata [`Self::open`] already read rather than from the
    /// path: a projection changes what is decoded, not what the file says about
    /// itself, and `try_open` would go back to storage for the footer to be told
    /// so.
    pub async fn project(&self, columns: &[&str]) -> Result<FileReader> {
        self.project_with(columns, None).await
    }

    async fn project_with(&self, columns: &[&str], stats: Option<&IoStats>) -> Result<FileReader> {
        let options = FileReaderOptions::default();
        let projection = reader_projection_from_column_names(
            SEGMENT_FILE_VERSION,
            self.reader.schema(),
            columns,
        )?;
        let file = match stats {
            Some(stats) => self.file.with_io_stats(stats.recorder()),
            None => self.file.clone(),
        };
        FileReader::try_open_with_file_metadata(
            Arc::new(LanceEncodingsIo::new(file).with_read_chunk_size(options.read_chunk_size)),
            self.path.clone(),
            Some(projection),
            Arc::<DecoderPlugins>::default(),
            self.reader.metadata().clone(),
            &LanceCache::no_cache(),
            options,
        )
        .await
    }

    /// [`Self::project`], with the reads it performs recorded into `stats`
    /// instead of into whatever this file was opened with.
    ///
    /// Replacing the sink rather than adding a second one is what makes the two
    /// halves of a query separable: `FileScheduler::with_io_stats` sets the
    /// handle rather than appending to a list, so a projection built here is
    /// counted here and nowhere else. The scheduler's global totals still see
    /// everything either way.
    pub async fn project_into(&self, columns: &[&str], stats: &IoStats) -> Result<FileReader> {
        self.project_with(columns, Some(stats)).await
    }

    /// A clone of this file whose reads are recorded into `stats` rather than
    /// into wherever this one records.
    ///
    /// The whole file, not one projection of it: [`Self::project_into`] rebinds
    /// a single reader, which is what separates the two halves of a query, while
    /// this rebinds the handle every projection is built from, which is what
    /// lets one open file be shared by queries that must not be counted
    /// together. Cheap in both directions - each of the two handles clones a few
    /// `Arc`s and reuses the footer, so nothing is re-read and nothing is
    /// rebuilt.
    pub fn with_io_stats(&self, stats: &IoStats) -> Self {
        Self {
            path: self.path.clone(),
            file: self.file.with_io_stats(stats.recorder()),
            reader: self.reader.with_io_stats(stats.recorder()),
            vectors: self.vectors.clone(),
            local: self.local.clone(),
        }
    }

    /// The reader every projection of this file is built from.
    #[cfg(test)]
    pub(crate) fn reader(&self) -> &FileReader {
        &self.reader
    }

    /// Open this file a second time for reading directly, if `store` is local
    /// storage and the platform has positional reads.
    ///
    /// A builder step rather than part of opening, because only a query wants
    /// it: a build or a maintenance pass reads whole columns once, where the
    /// scheduler's queueing is what it is for. Failure to open is not an error -
    /// the file is already open through the scheduler, and that is the path this
    /// one is an optimisation of.
    ///
    /// `totals` is the caller's running count of everything read off the
    /// scheduler, which it has to add to whatever the scheduler reports.
    ///
    /// `has_direct_local_paths` and not `is_local`, because that is the
    /// predicate Lance's own reader dispatch turns on: a local store rooted
    /// below `/` addresses its objects relative to that root, and
    /// `to_local_path` would name an absolute path somewhere else entirely. The
    /// failure would be silent rather than loud - a descriptor on another inode,
    /// read at this file's offsets. `file+uring` is left out for the opposite
    /// reason: a caller who configured io_uring asked for the scheduler, and
    /// substituting synchronous reads would undo what they chose.
    pub(crate) fn with_local_reads(mut self, store: &ObjectStore, totals: &Arc<IoStats>) -> Self {
        if cfg!(unix) && store.has_direct_local_paths() && !store.prefers_lite_scheduler() {
            self.local = Some(LocalReads {
                block_size: store.block_size() as u64,
                max_iop_size: store.max_iop_size(),
                totals: totals.clone(),
                file: Arc::new(OnceLock::new()),
            });
        }
        self
    }

    /// The descriptor a re-score reads through, opened the first time one asks.
    ///
    /// `None` when this file has none to open or opening it failed, both of
    /// which mean the scheduler reads the same bytes instead. The open runs off
    /// the worker: it is a blocking syscall, and Lance's own local reader takes
    /// the same care with the same call.
    async fn local_reads(&self) -> Option<(&LocalReads, &std::fs::File)> {
        let local = self.local.as_ref()?;
        if local.file.get().is_none() {
            let path = to_local_path(&self.path);
            if let Ok(Ok(opened)) =
                tokio::task::spawn_blocking(move || std::fs::File::open(path)).await
            {
                // The loser of a race drops its descriptor here rather than
                // publishing a second one.
                let _ = local.file.set(opened);
            }
        }
        Some((local, local.file.get()?))
    }

    /// Whether this file will read its vectors off the disk itself. Says
    /// nothing about whether it has opened the descriptor yet - it does that on
    /// the first re-score.
    #[cfg(test)]
    pub(crate) fn reads_locally(&self) -> bool {
        self.local.is_some()
    }

    /// The vectors of `rows`, fetched by byte offset instead of decoded.
    ///
    /// `Ok(None)` when this file's vector column is not addressable, or when it
    /// is not the width the segment declares - both mean the caller has to read
    /// the ordinary way, and the second of them is left to the decoder on
    /// purpose, so that a mismatched width is reported by the check that has
    /// always reported it.
    ///
    /// The bytes still go through the scheduler, so they are coalesced, counted
    /// and throttled exactly as the decoder's would be: what is skipped is the
    /// decoder, not the reading. `rows` must ascend, which the scheduler
    /// requires and a candidate list already satisfies.
    pub(crate) async fn read_vectors(
        &self,
        rows: &[u32],
        dimension: u32,
        stats: &IoStats,
    ) -> Result<Option<FixedSizeListArray>> {
        let Some(layout) = self
            .vectors
            .get_or_init(|| VectorLayout::of(&self.reader, VECTOR_COLUMN))
        else {
            return Ok(None);
        };
        if layout.items() != u64::from(dimension) || layout.stride() != u64::from(dimension) * 4 {
            return Ok(None);
        }
        let wanted = layout.ranges(rows)?;
        let Some((local, _)) = self.local_reads().await else {
            let chunks = self
                .file
                .with_io_stats(stats.recorder())
                .submit_request(wanted, 0)
                .await?;
            return raw::vectors(chunks.iter().map(|chunk| chunk.as_ref()), dimension).map(Some);
        };

        let reads = raw::coalesced(&wanted, local.block_size, local.max_iop_size);

        // One hop for the whole batch. The reads are blocking syscalls and there
        // can be one per candidate, which without a re-score budget is the whole
        // search list - leaving that on a runtime worker would hold it through
        // every one of them with nowhere to yield.
        let descriptor = local.file.clone();
        let batch = reads.clone();
        let blocks = tokio::task::spawn_blocking(move || {
            let Some(file) = descriptor.get() else {
                return Err(std::io::Error::from(std::io::ErrorKind::NotFound));
            };
            batch
                .iter()
                .map(|read| {
                    let mut block = vec![0u8; (read.end - read.start) as usize];
                    read_at(file, &mut block, read.start).map(|()| block)
                })
                .collect::<std::io::Result<Vec<_>>>()
        })
        .await
        .map_err(|source| Error::io(format!("Vamana re-score read was cancelled: {source}")))?
        .map_err(|source| {
            Error::io(format!(
                "Vamana could not read {reads:?} of {}: {source}",
                self.path
            ))
        })?;

        // Counted here because nothing else counts them at all: these bytes never
        // reach the scheduler, so the query's own sink and the index's running
        // total both have to be told by hand. After the reads and not before, so
        // that a read that failed is not charged to an index for the rest of its
        // life - the scheduler charges first and would have, but a number that
        // survives its own failure is worse than one that matches it.
        stats.record_request(&reads);
        local.totals.record_request(&reads);

        let slices = raw::slices(&wanted, &reads, &blocks)?;
        raw::vectors(slices.iter().map(|slice| slice.as_ref()), dimension).map(Some)
    }

    /// The reader over every column.
    pub fn whole(self) -> FileReader {
        self.reader
    }
}

/// How many partition files an index keeps open at once.
///
/// A descriptor is a real resource and an index can hold thousands of
/// partitions, so this is a cap and not a count. What it caps is entries, and an
/// entry can cost two descriptors rather than one: the reader's, and - once a
/// re-score has run against it on local storage - the one it `pread`s through.
/// Nor is the pool the only holder. A query keeps a handle per probe until its
/// re-score is done, so `nprobes` times the queries in flight are open whatever
/// this says, and evicting an entry a query still holds releases nothing until
/// that query finishes.
///
/// The number wants to be at least the *distinct partitions* the queries in
/// flight probe between them, which is `nprobes` times their number, not
/// `PARTITIONS_IN_FLIGHT` times it. Below that the queries evict each other's
/// handles and every probe misses, which costs what every probe cost before this
/// existed. Sixty-four covers a server answering a dozen queries over a handful
/// of partitions each; an index partitioned finely enough to probe dozens at a
/// time wants more, and a caller cannot say so yet.
pub const OPEN_FILES: usize = 64;

/// The partition files an index has open, shared by every query that probes
/// them.
///
/// Opening one is not a read and so is not bounded by anything a read is bounded
/// by: `ScanScheduler::open_file` on local storage is
/// `spawn_blocking(File::open)`, a hop through the blocking pool and a real
/// `open(2)`, and a query paid it once for every partition it probed. The footer
/// that comes with it has been shared through the cache since the cache existed;
/// the descriptor never was.
///
/// It also pins what the handle holds, which is more than a descriptor: the
/// footer the reader was built from is an `Arc<CachedFileMetadata>` shared with
/// the cache, so the cache can evict its entry and reclaim nothing. A few
/// kilobytes a file at the partition sizes this crate is written for, tens of
/// megabytes for sixty-four files of a million 960-wide rows.
///
/// A handle is stored with the sink of whichever query opened it still bound on,
/// and that sink is never used again: every handout rebinds
/// ([`PartitionFile::with_io_stats`] replaces the recorder rather than adding
/// one), and the raw handle never leaves this type. Storing it that way rather
/// than sink-free keeps the accounting exactly where it was - the query that
/// opens a file pays for its footer, the queries that share it pay for nothing -
/// so a query's two phases still add up to what the scheduler counted for it
/// even when a handle is evicted and opened again mid-run.
pub(crate) struct OpenFiles {
    cap: usize,
    inner: Mutex<Opened>,
}

#[derive(Default)]
struct Opened {
    files: HashMap<Path, Handle>,
    /// Stamped onto a handle whenever it is looked up, so the smallest stamp is
    /// the least recently used.
    tick: u64,
}

struct Handle {
    file: Arc<PartitionFile>,
    used: u64,
}

impl std::fmt::Debug for OpenFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let held = self.inner.lock().map(|open| open.files.len()).ok();
        f.debug_struct("OpenFiles")
            .field("cap", &self.cap)
            .field("held", &held)
            .finish()
    }
}

impl OpenFiles {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: Mutex::new(Opened::default()),
        }
    }

    /// The open file for `path` with `stats` bound onto it, if this index still
    /// holds one.
    pub(crate) fn get(&self, path: &Path, stats: &IoStats) -> Option<PartitionFile> {
        let mut open = self.inner.lock().ok()?;
        open.tick += 1;
        let tick = open.tick;
        let handle = open.files.get_mut(path)?;
        handle.used = tick;
        Some(handle.file.with_io_stats(stats))
    }

    /// Hold `file` for the queries after this one, and return the handle they
    /// will all share, with `stats` bound onto it.
    ///
    /// Keeps what is already held under `path` when there is one rather than
    /// replacing it: two queries that miss on the same partition at the same
    /// moment both open the file, and the loser's handle is dropped here so that
    /// both of them read through one descriptor rather than two.
    pub(crate) fn put(&self, path: &Path, file: PartitionFile, stats: &IoStats) -> PartitionFile {
        // Both of these hold descriptors, and both are dropped after the guard
        // goes out of scope: closing a file on a slow mount inside the lock
        // would stall every other query's lookup behind it.
        let mut loser = None;
        let mut evicted = Vec::new();
        let shared = {
            let Ok(mut open) = self.inner.lock() else {
                return file;
            };
            open.tick += 1;
            let tick = open.tick;
            match open.files.get_mut(path) {
                Some(handle) => {
                    handle.used = tick;
                    let shared = handle.file.with_io_stats(stats);
                    loser = Some(file);
                    shared
                }
                None => {
                    let held = Arc::new(file);
                    let shared = held.with_io_stats(stats);
                    open.files.insert(
                        path.clone(),
                        Handle {
                            file: held,
                            used: tick,
                        },
                    );
                    // After the insert, so the handle just taken is the newest
                    // and cannot be the one evicted.
                    while open.files.len() > self.cap {
                        let Some(stalest) = open
                            .files
                            .iter()
                            .min_by_key(|(_, handle)| handle.used)
                            .map(|(path, _)| path.clone())
                        else {
                            break;
                        };
                        evicted.extend(open.files.remove(&stalest));
                    }
                    shared
                }
            }
        };
        drop(loser);
        drop(evicted);
        shared
    }
}

/// Open a file of a segment for reading.
///
/// `columns` narrows what is fetched; pass `None` to read every column. Reach
/// for [`PartitionFile`] instead when the same file is to be read under more
/// than one projection.
pub async fn open_file(
    scheduler: &Arc<ScanScheduler>,
    path: &Path,
    columns: Option<&[&str]>,
    size_bytes: Option<u64>,
) -> Result<FileReader> {
    let file = PartitionFile::open(scheduler, path, size_bytes, None).await?;
    match columns {
        Some(columns) => file.project(columns).await,
        None => Ok(file.whole()),
    }
}

/// Read a contiguous run of rows.
///
/// `Range` rather than the whole file because the layout is built for it: the
/// reason `__neighbors` has a fixed stride is that reading one vertex fetches
/// `max_degree * 4` bytes and nothing else. A lazy walk reaches for
/// [`read_scattered`] instead, which is the same read for a set of rows that are
/// not adjacent.
pub async fn read_rows(reader: &FileReader, rows: Range<usize>) -> Result<RecordBatch> {
    if rows.is_empty() {
        return Err(Error::invalid_input(format!(
            "row range {}..{} selects nothing",
            rows.start, rows.end
        )));
    }
    let batches = reader
        .read_stream(
            ReadBatchParams::Range(rows.clone()),
            u32::MAX,
            1,
            FilterExpression::no_filter(),
        )
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    // The schema comes from the data, never from the reader: `FileReader::schema`
    // reports the whole file even when the reader is projected onto one column.
    let schema = batches
        .first()
        .ok_or_else(|| {
            Error::corrupt_file_named(
                "segment",
                format!("row range {}..{} returned no data", rows.start, rows.end),
            )
        })?
        .schema();
    Ok(concat_batches(&schema, batches.iter())?)
}

/// Read a scattered set of single rows as one request.
///
/// [`ReadBatchParams::Ranges`] and not a call per row: the scheduler coalesces
/// adjacent ranges in one pass, which measured half the iops and half the bytes
/// of issuing them separately, and it is what turns one hop of a lazy walk into
/// one round trip instead of `beam_width` of them.
///
/// `rows` must be strictly ascending, because that coalescing pass does not
/// sort - and the returned batch is in the order given, so a caller reading a
/// row back by position depends on it too. Both are internal contracts of the
/// lazy walk rather than anything a file can violate, hence the plain check.
pub async fn read_scattered(reader: &FileReader, rows: &[u32]) -> Result<RecordBatch> {
    if rows.is_empty() {
        return Err(Error::invalid_input(
            "Vamana was asked to read no rows at all".to_string(),
        ));
    }
    if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::internal(
            "Vamana scattered reads must arrive strictly ascending; the scheduler coalesces \
             ranges without sorting them"
                .to_string(),
        ));
    }
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
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let schema = batches
        .first()
        .ok_or_else(|| {
            Error::corrupt_file_named(
                "partition",
                format!(
                    "Vamana read of {} scattered rows returned no data",
                    rows.len()
                ),
            )
        })?
        .schema();
    let batch = concat_batches(&schema, batches.iter())?;
    if batch.num_rows() != rows.len() {
        return Err(Error::corrupt_file_named(
            "partition",
            format!(
                "Vamana asked for {} scattered rows and got {}",
                rows.len(),
                batch.num_rows()
            ),
        ));
    }
    Ok(batch)
}

/// Check a partition file against what `index.idx` says it holds.
///
/// `expected_rows` comes from the segment table, which is a different file from
/// the one being read. Requiring the two to agree is what keeps a damaged footer
/// from being believed, and it is also the only ceiling on the read that
/// follows: without it the row count written in the footer is what decides how
/// much memory to allocate.
fn check_row_count(reader: &FileReader, expected_rows: u32) -> Result<()> {
    if reader.metadata().num_rows != expected_rows as u64 {
        return Err(Error::corrupt_file_named(
            "partition",
            format!(
                "Vamana partition file holds {} rows but the segment table lists {expected_rows}",
                reader.metadata().num_rows
            ),
        ));
    }
    Ok(())
}

/// Read a whole partition's file, checked against what the segment table says.
///
/// The batch rather than the [`Partition`] because a partition file holds more
/// than a partition: a query wants the codes out of the same read, and codes are
/// deliberately not a field of `Partition`.
///
/// No empty-partition branch: an empty partition is written no file and given no
/// row in the segment table, so `expected_rows` is never zero on any path that
/// reaches here, and a caller who passes zero anyway gets the empty-range error
/// from [`read_rows`] rather than a partition invented from a schema.
pub async fn read_partition_batch(reader: &FileReader, expected_rows: u32) -> Result<RecordBatch> {
    check_row_count(reader, expected_rows)?;
    read_rows(reader, 0..expected_rows as usize).await
}

/// Read a whole partition back into memory.
pub async fn read_partition(reader: &FileReader, expected_rows: u32) -> Result<Partition> {
    Partition::try_from_batch(&read_partition_batch(reader, expected_rows).await?)
}

/// Refuse a partition whose shape disagrees with the segment that lists it.
///
/// The writer checks both against the segment on the way out; a reader has to
/// check them on the way back in, and every reader has to, which is why this is
/// not inlined at one of them. A partition whose width disagrees with the
/// manifest would be searched with a query of the wrong length against
/// `flat_storage`, which takes its dimension from the array - silently wrong
/// distances rather than an error - and consolidation would rewrite it into a
/// segment that declares the other number.
pub fn check_partition_shape(
    partition: &Partition,
    entry: &PartitionEntry,
    max_degree: u32,
    dimension: u32,
) -> Result<()> {
    if partition.graph().max_degree() != max_degree || partition.dimension() != dimension {
        return Err(Error::corrupt_file_named(
            entry.file.as_str(),
            format!(
                "Vamana partition {} holds degree {} and dimension {} but its segment declares \
                 degree {max_degree} and dimension {dimension}",
                entry.partition_id,
                partition.graph().max_degree(),
                partition.dimension(),
            ),
        ));
    }
    Ok(())
}

/// Read the row id of every vertex, and nothing else.
///
/// The saving is the point. Deciding whether a partition holds any deleted row
/// costs eight bytes a vertex this way, against `4 * (max_degree + dimension)`
/// for the whole partition - 776 bytes a vertex at the crate's own working
/// point. Consolidation asks that question of every partition of a segment and
/// reads the rest of only the ones that answer yes.
///
/// `reader` should have been opened projected onto that one column; the saving
/// is in the projection, not here. Reading it off an unprojected reader is
/// correct and merely pointless.
pub async fn read_row_ids(reader: &FileReader, expected_rows: u32) -> Result<Vec<u64>> {
    check_row_count(reader, expected_rows)?;
    row_ids_from_batch(&read_rows(reader, 0..expected_rows as usize).await?)
}

/// How many partitions a pass that writes a segment prepares at once.
///
/// Every such pass - build, consolidation, insertion, merge - reads and rebuilds
/// partitions concurrently up to this many, then hands the results to
/// [`SegmentWriter`] one at a time in ascending id order, because
/// [`SegmentWriter::write_partition`] accepts them in no other order. So the
/// arithmetic overlaps and the writing does not, which is the whole of what this
/// bounds. The same number is what Lance's own index builder gives the
/// equivalent stage (`lance/src/index/vector/builder.rs`), and a round of graph
/// maintenance is processor-bound by measurement, so the bound that matters is
/// the pool's width rather than the store's.
///
/// It costs memory: a pass holds this many partitions' vectors and edges at
/// once, and a partition being rebuilt holds both what was read and what came
/// out of it. That is not the guarantee the query path's `PARTITIONS_IN_FLIGHT`
/// makes - that one is a ceiling a caller can quote, this one is a throughput
/// knob on a batch operation - but it is set by the same lever, `num_partitions`.
///
/// Never zero, which matters because `buffered(0)` admits nothing and waits
/// forever rather than failing: the count falls back to one core on a machine
/// with fewer cores than Lance reserves for io, and the environment variable that
/// overrides it refuses a value below one.
pub(crate) fn partitions_in_flight() -> usize {
    get_num_compute_intensive_cpus()
}

/// Writes a segment directory one partition at a time.
///
/// Partitions are written and dropped as they arrive rather than assembled in
/// memory: a segment is as large as the dataset it indexes, while everything
/// this writer retains is the one table row per partition that `index.idx` ends
/// up holding.
///
/// A segment only exists once [`Self::finish`] has written `index.idx`; a run
/// that fails partway leaves partition files behind, and the caller must discard
/// the directory rather than reuse it.
pub struct SegmentWriter {
    store: Arc<ObjectStore>,
    dir: Path,
    metadata: IndexMetadata,
    ivf: IvfModel,
    partitions: Vec<PartitionEntry>,
}

impl SegmentWriter {
    pub fn new(store: Arc<ObjectStore>, dir: Path, metadata: IndexMetadata, ivf: IvfModel) -> Self {
        Self {
            store,
            dir,
            metadata,
            ivf,
            partitions: Vec::new(),
        }
    }

    /// Write one partition and return the size of its file in bytes.
    ///
    /// Partition ids must arrive in ascending order, and `partition` must not be
    /// empty: an empty partition gets no file and no row in `index.idx`, so
    /// calling this for one would write a file nothing points at.
    pub async fn write_partition(
        &mut self,
        partition_id: u32,
        medoid: u32,
        partition: &Partition,
    ) -> Result<u64> {
        if partition.is_empty() {
            return Err(Error::invalid_input(format!(
                "Vamana partition {partition_id} is empty; empty partitions are not written"
            )));
        }
        if partition.graph().max_degree() != self.metadata.max_degree {
            return Err(Error::invalid_input(format!(
                "Vamana partition {partition_id} has max_degree {} but the segment declares {}",
                partition.graph().max_degree(),
                self.metadata.max_degree
            )));
        }
        if partition.dimension() != self.metadata.dimension {
            return Err(Error::invalid_input(format!(
                "Vamana partition {partition_id} has dimension {} but the segment declares {}",
                partition.dimension(),
                self.metadata.dimension
            )));
        }
        self.check_entry(partition_id, medoid, partition.len() as u32)?;

        // Encoded here rather than by the caller, so that no pass that produces a
        // partition can forget to, and none of them has to keep a code in step
        // with a vertex it moved. The centroid comes off this segment's own
        // routing model, which is the one the partition was assigned by.
        let codes = self
            .metadata
            .codes
            .as_ref()
            .map(|params| {
                let centroid = self.ivf.centroid(partition_id as usize).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Vamana partition {partition_id} has no centroid in a routing model of {}",
                        self.ivf.num_partitions()
                    ))
                })?;
                encode(
                    params,
                    self.metadata.distance_type,
                    partition.vectors(),
                    &centroid,
                )
            })
            .transpose()?;

        let file = partition_file_name(partition_id);
        let path = self.dir.clone().join(file.as_str());
        let size = write_partition(&self.store, &path, partition, codes.as_ref()).await?;
        self.partitions.push(PartitionEntry {
            partition_id,
            medoid,
            num_rows: partition.len() as u32,
            file,
        });
        Ok(size)
    }

    /// Take a partition of `from` into this segment without decoding it.
    ///
    /// A partition with nothing deleted in it survives consolidation byte for
    /// byte, and the only reason it has to be touched at all is that it has to
    /// end up in *this* directory: [`PartitionEntry::file`] is a plain name
    /// inside the segment, never a path, so a partition of the new segment
    /// cannot point at a file of the old one. On a blob store the copy is
    /// server-side and no byte crosses the network.
    ///
    /// The source's own metadata is required rather than trusted, because the
    /// copied bytes are a graph of a given width over vectors of a given
    /// dimension and this segment declares both. Nothing downstream reads the
    /// file's schema back against `partition_schema`, so a copy from a segment
    /// built with another degree would leave `index.idx` describing a file it
    /// does not describe.
    ///
    /// No size comes back, unlike [`Self::write_partition`]: on a blob store the
    /// answer would be a `HEAD` the copy itself does not need, and Lance fills
    /// the size of every file of a committed index by listing the directory.
    pub async fn copy_partition(
        &mut self,
        from_dir: &Path,
        from: &SegmentManifest,
        partition_id: u32,
    ) -> Result<()> {
        let entry = from.partition(partition_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "Vamana was asked to copy partition {partition_id}, which the source segment does \
                 not list"
            ))
        })?;
        for (what, source, mine) in [
            (
                "max_degree",
                from.metadata().max_degree,
                self.metadata.max_degree,
            ),
            (
                "dimension",
                from.metadata().dimension,
                self.metadata.dimension,
            ),
        ] {
            if source != mine {
                return Err(Error::invalid_input(format!(
                    "Vamana cannot copy partition {partition_id} from a segment declaring {what} \
                     {source} into one declaring {mine}"
                )));
            }
        }
        // Codes are bytes quantised under one rotation, and nothing downstream
        // reads a rotation back off a partition file: copied into a segment
        // declaring another one, they would be decoded into distances that are
        // meaningless rather than approximate. Equality of the whole parameters
        // is the check because the rotation is inside them, and it is what makes
        // "one rotation per index" enforced rather than merely inherited.
        if from.metadata().codes != self.metadata.codes {
            return Err(Error::invalid_input(format!(
                "Vamana cannot copy partition {partition_id} between segments whose codes \
                 disagree; the rotation a code was built under is not recoverable from it"
            )));
        }
        self.check_entry(partition_id, entry.medoid, entry.num_rows)?;

        let file = partition_file_name(partition_id);
        let from_path = from_dir.clone().join(entry.file.as_str());
        let to_path = self.dir.clone().join(file.as_str());
        // A copy onto itself is not a no-op. `std::fs::copy`, which the local
        // store uses, truncates the destination before reading the source and
        // then reports `Ok(0)` - measured: a 35-byte file comes back at 0 bytes
        // with no error. Reachable only through this public API, consolidation
        // always writing a segment of its own, and it destroys a partition.
        if from_path == to_path {
            return Err(Error::invalid_input(format!(
                "Vamana was asked to copy partition {partition_id} onto itself at {to_path}"
            )));
        }
        self.store.copy(&from_path, &to_path).await?;
        self.partitions.push(PartitionEntry {
            partition_id,
            medoid: entry.medoid,
            num_rows: entry.num_rows,
            file,
        });
        Ok(())
    }

    /// What both ways into the table have to agree on before a row is added.
    fn check_entry(&self, partition_id: u32, medoid: u32, num_rows: u32) -> Result<()> {
        if medoid >= num_rows {
            return Err(Error::invalid_input(format!(
                "Vamana partition {partition_id} has medoid {medoid} but holds only {num_rows} \
                 vertices"
            )));
        }
        if let Some(last) = self.partitions.last()
            && last.partition_id >= partition_id
        {
            return Err(Error::invalid_input(format!(
                "Vamana partition {partition_id} was written after partition {}; partitions must \
                 arrive in ascending order",
                last.partition_id
            )));
        }
        Ok(())
    }

    /// Write `index.idx` and return the segment as it was committed to disk.
    pub async fn finish(self) -> Result<SegmentManifest> {
        let manifest = SegmentManifest::try_new(self.metadata, self.ivf, self.partitions)?;
        let batch = manifest.to_batch()?;
        let schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref())?;
        let mut writer = create_writer(
            SEGMENT_FILE_VERSION,
            self.store
                .create(&self.dir.clone().join(INDEX_FILE_NAME))
                .await?,
            schema,
            FileWriterOptions::default(),
        )?;

        writer.add_schema_metadata(INDEX_METADATA_KEY, manifest.metadata().to_json()?);
        let ivf_position = writer
            .add_global_buffer(pb::Ivf::try_from(manifest.ivf())?.encode_to_vec().into())
            .await?;
        writer.add_schema_metadata(IVF_POSITION_KEY, ivf_position.to_string());
        if batch.num_rows() > 0 {
            writer.write_batch(&batch).await?;
        }
        writer.finish().await?;
        Ok(manifest)
    }
}

/// Read a segment's `index.idx`.
///
/// One read of one small file: the partition table and the routing model are
/// everything a query needs before it knows which partitions to open.
pub async fn read_segment(
    scheduler: &Arc<ScanScheduler>,
    dir: &Path,
    size_bytes: Option<u64>,
) -> Result<SegmentManifest> {
    let reader = open_file(
        scheduler,
        &dir.clone().join(INDEX_FILE_NAME),
        None,
        size_bytes,
    )
    .await?;
    let schema_metadata = &reader.schema().metadata;

    let metadata =
        IndexMetadata::from_json(schema_metadata.get(INDEX_METADATA_KEY).ok_or_else(|| {
            Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!("Vamana segment has no {INDEX_METADATA_KEY} in its schema metadata"),
            )
        })?)?;

    let ivf_position = schema_metadata
        .get(IVF_POSITION_KEY)
        .ok_or_else(|| {
            Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!("Vamana segment has no {IVF_POSITION_KEY} in its schema metadata"),
            )
        })?
        .parse::<u32>()
        .map_err(|e| {
            Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!("Vamana segment has an unreadable {IVF_POSITION_KEY}: {e}"),
            )
        })?;
    // Global buffer indices are one-based - buffer 0 is the file's own schema
    // descriptor - so a stored 0 is corruption, not a model.
    if ivf_position == 0 {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!("Vamana segment stores {IVF_POSITION_KEY} = 0, which is the file descriptor"),
        ));
    }
    let proto = pb::Ivf::decode(reader.read_global_buffer(ivf_position).await?)?;
    validate_ivf_model(&proto)?;
    let ivf = IvfModel::try_from(proto)?;

    let num_rows = reader.metadata().num_rows as usize;
    let batch = if num_rows == 0 {
        RecordBatch::new_empty(Arc::new(index_schema()))
    } else {
        read_rows(&reader, 0..num_rows).await?
    };
    // Everything the constructor refuses it refuses as `invalid_input`, because
    // a *writer* goes through the same constructor and there the caller is the
    // one who got it wrong. Reaching it from here means the same values arrived
    // out of a file, and the repository's rule sorts errors by where the bad
    // value came from rather than by what was wrong with it.
    SegmentManifest::try_from_batch(metadata, ivf, &batch).map_err(|error| match error {
        Error::InvalidInput { source, .. } => {
            Error::corrupt_file_named(INDEX_FILE_NAME, source.to_string())
        }
        other => other,
    })
}

/// Reject an IVF buffer that [`IvfModel::try_from`] would crash on.
///
/// Every case here is a process abort taken on bytes read off disk. `try_from`
/// is written for models Lance produced itself, so it asserts, divides and
/// unwraps on fields its own writer always fills - which a buffer arriving from
/// anywhere else need not.
fn validate_ivf_model(proto: &pb::Ivf) -> Result<()> {
    // Asserted rather than checked, so a mismatch aborts instead of reporting.
    if !proto.offsets.is_empty() && proto.offsets.len() != proto.lengths.len() {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "Vamana segment carries an IVF model with {} offsets and {} lengths",
                proto.offsets.len(),
                proto.lengths.len()
            ),
        ));
    }
    // The v1 centroid layout is a flat buffer whose width is recovered by
    // dividing by the number of partitions - taken from `lengths`, which the v1
    // writer always filled and nothing enforces.
    if proto.centroids_tensor.is_none() && !proto.centroids.is_empty() && proto.lengths.is_empty() {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "Vamana segment carries {} legacy centroid values but no partition lengths to \
                 recover their width from",
                proto.centroids.len()
            ),
        ));
    }

    let Some(tensor) = proto.centroids_tensor.as_ref() else {
        return Ok(());
    };
    let data_type = pb::tensor::DataType::try_from(tensor.data_type).map_err(|_| {
        Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "Vamana segment carries IVF centroids of unknown data type {}",
                tensor.data_type
            ),
        )
    })?;
    // Not a crash but a failure deferred: centroids of another width open
    // cleanly and then fail per query, because routing dispatches on the pair
    // of centroid and query types and this crate only ever builds an f32 query.
    if data_type != pb::tensor::DataType::Float32 {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!("Vamana segment carries {data_type:?} IVF centroids, expected Float32"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Float32Array;
    use arrow_schema::{DataType, Field};

    use crate::partition::PartitionGraph;

    const DIMENSION: i32 = 3;

    fn sample_partition() -> Partition {
        let graph =
            PartitionGraph::try_new(2, vec![100, 200, 300], vec![vec![1], vec![2], vec![0]])
                .unwrap();
        let values = (0..graph.len() as i32 * DIMENSION)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            DIMENSION,
            Arc::new(Float32Array::from(values)),
            None,
        )
        .unwrap();
        Partition::try_new(graph, vectors).unwrap()
    }

    /// One partition file on disk, and the scheduler to read it through.
    async fn written(dir: &tempfile::TempDir, name: &str) -> (Arc<ScanScheduler>, Path) {
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join(name)).unwrap();
        write_partition(&store, &path, &sample_partition(), None)
            .await
            .unwrap();
        (scan_scheduler(&store), path)
    }

    async fn open(scheduler: &Arc<ScanScheduler>, path: &Path) -> PartitionFile {
        PartitionFile::open(scheduler, path, None, None)
            .await
            .unwrap()
    }

    /// The handle held under `path`, by identity rather than by value: what the
    /// pool is for is that everyone gets the *same* open file.
    fn held(files: &OpenFiles, path: &Path) -> Option<Arc<PartitionFile>> {
        let open = files.inner.lock().unwrap();
        open.files.get(path).map(|handle| handle.file.clone())
    }

    #[tokio::test]
    async fn a_held_file_is_handed_out_rather_than_opened_again() {
        let dir = tempfile::tempdir().unwrap();
        let (scheduler, path) = written(&dir, "part_00000.idx").await;
        let files = OpenFiles::new(4);
        let stats = IoStats::new();

        assert!(
            files.get(&path, &stats).is_none(),
            "an empty pool handed out a file it was never given"
        );
        files.put(&path, open(&scheduler, &path).await, &stats);

        let first = held(&files, &path).unwrap();
        assert!(files.get(&path, &stats).is_some());
        let after = held(&files, &path).unwrap();
        assert!(
            Arc::ptr_eq(&first, &after),
            "a lookup replaced the handle instead of handing it out"
        );
    }

    /// Two queries that miss on the same partition at the same moment both open
    /// the file. Only one of the two descriptors may survive, and it has to be
    /// the one already published, or the loser's callers read through a handle
    /// nobody else can reach.
    #[tokio::test]
    async fn a_second_open_of_one_path_keeps_the_first_handle() {
        let dir = tempfile::tempdir().unwrap();
        let (scheduler, path) = written(&dir, "part_00000.idx").await;
        let files = OpenFiles::new(4);
        let stats = IoStats::new();

        files.put(&path, open(&scheduler, &path).await, &stats);
        let first = held(&files, &path).unwrap();
        files.put(&path, open(&scheduler, &path).await, &stats);
        let after = held(&files, &path).unwrap();

        assert!(
            Arc::ptr_eq(&first, &after),
            "the second open displaced the handle the first one published"
        );
        assert_eq!(files.inner.lock().unwrap().files.len(), 1);
    }

    #[tokio::test]
    async fn the_least_recently_used_handle_goes_when_the_cap_is_reached() {
        let dir = tempfile::tempdir().unwrap();
        let (scheduler, first) = written(&dir, "part_00000.idx").await;
        let (_, second) = written(&dir, "part_00001.idx").await;
        let (_, third) = written(&dir, "part_00002.idx").await;
        let files = OpenFiles::new(2);
        let stats = IoStats::new();

        files.put(&first, open(&scheduler, &first).await, &stats);
        files.put(&second, open(&scheduler, &second).await, &stats);
        assert_eq!(
            files.inner.lock().unwrap().files.len(),
            2,
            "the pool evicted before it was over its cap"
        );

        // Make the first the freshly used one, so eviction by insertion order
        // and eviction by use pick different handles.
        assert!(files.get(&first, &stats).is_some());
        files.put(&third, open(&scheduler, &third).await, &stats);

        let open_now = files.inner.lock().unwrap();
        assert_eq!(open_now.files.len(), 2, "the pool grew past its cap");
        assert!(
            open_now.files.contains_key(&first),
            "the handle used most recently was evicted"
        );
        assert!(
            open_now.files.contains_key(&third),
            "the handle just taken was evicted"
        );
        assert!(
            !open_now.files.contains_key(&second),
            "the stalest handle survived"
        );
    }
}
