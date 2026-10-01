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

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field};
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
    INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, IndexMetadata, ROW_ID_COLUMN,
    VECTOR_COLUMN, VectorSource, index_schema, partition_file_name, partition_schema,
};
use crate::partition::{Partition, graph_from_batch, row_ids_from_batch};
use crate::query::RescoreReads;
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
    vector_source: VectorSource,
) -> Result<u64> {
    // The format says an empty partition gets no row in `index.idx` and no file.
    // `SegmentWriter` enforces that; this function is public and delegated to, so
    // it has to enforce it too rather than write a file nothing can point at.
    if partition.is_empty() {
        return Err(Error::invalid_input(
            "Vamana will not write a file for an empty partition".to_string(),
        ));
    }
    write_batch(store, path, &partition.to_batch(codes, vector_source)?).await
}

/// The column of `stored` that `field` names, as `field` declares it: a
/// fixed-size list of that width and item type with no null at either level,
/// under the non-nullable item field [`partition_schema`] declares.
///
/// Lance hands every fixed-size list back with a nullable item field, whatever
/// it was written with, so the values are the ones written and the type is
/// not the declared one, which `RecordBatch::try_new` would refuse.
fn carried(stored: &RecordBatch, field: &Field, file: &str) -> Result<ArrayRef> {
    let name = field.name();
    let corrupt = |reason: String| {
        Error::corrupt_file_named(file, format!("Vamana partition column {name} {reason}"))
    };
    let DataType::FixedSizeList(item, width) = field.data_type() else {
        return Err(Error::internal(format!(
            "Vamana partition column {name} is declared {}, not a fixed-size list",
            field.data_type()
        )));
    };
    let column = stored
        .column_by_name(name)
        .ok_or_else(|| corrupt("is missing".to_string()))?;
    let list = column
        .as_fixed_size_list_opt()
        .ok_or_else(|| corrupt(format!("is {}, not a fixed-size list", column.data_type())))?;
    if list.value_length() != *width || list.value_type() != *item.data_type() {
        return Err(corrupt(format!(
            "holds {} x {} a row where the segment declares {width} x {}",
            list.value_length(),
            list.value_type(),
            item.data_type()
        )));
    }
    let length = *width as usize;
    let values = list
        .values()
        .slice(list.offset() * length, list.len() * length);
    if list.null_count() != 0 || values.null_count() != 0 {
        return Err(corrupt("holds nulls".to_string()));
    }
    Ok(Arc::new(FixedSizeListArray::try_new(
        item.clone(),
        *width,
        values,
        None,
    )?))
}

/// Write `batch` as a partition file and return the size of the file in bytes.
async fn write_batch(store: &ObjectStore, path: &Path, batch: &RecordBatch) -> Result<u64> {
    let schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref())?;
    let mut writer = create_writer(
        SEGMENT_FILE_VERSION,
        store.create(path).await?,
        schema,
        FileWriterOptions::default(),
    )?;
    writer.write_batch(batch).await?;
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

/// A file - a partition file, or one of the dataset's data files - open for
/// reading without going through the scheduler.
///
/// The scheduler's job on a local file is to bound how many reads are in flight
/// and to move each one onto the blocking pool. For a re-score neither buys
/// anything: it wants twenty ranges of a few kilobytes that the page cache
/// almost always holds, and it pays a queue, a task and a wakeup for each of
/// them.
///
/// So it reads each of them itself, on the thread that asked, whenever the page
/// cache can hand the bytes over - which is what `RWF_NOWAIT` asks of the
/// kernel, a read at a time. The flag is a promise not to wait for the data,
/// not a promise never to block: a read the page cache misses can still start
/// its own readahead, and read the file's block map to do it, before it is
/// refused. A refusal does not stop the reads after it from being tried.
///
/// A read the kernel refuses goes to the blocking pool, because a `pread` of a
/// page the kernel does not have blocks the thread it runs on until the device
/// answers - about a hundred microseconds on an NVMe drive, milliseconds on a
/// disk or a network volume - and the thread that asked is a tokio worker.
/// Every such read of one batch goes in one trip, and so does a whole batch
/// larger than [`IN_PLACE_BYTES`]. Other Unixes have no such flag, and every
/// batch there takes the trip; off Unix a file never reads locally at all.
///
/// The trip is not free even for a page the cache holds, and sparing it is the
/// whole reason for reading in place: it wakes a pool thread and then a runtime
/// worker, where a read in place wakes nobody, and the pool thread has slept
/// through the search whenever that read nothing through the scheduler.
/// Measured on 23 September 2026 with resident edges, a budget of twenty and a
/// warm local NVMe: with one query in flight a re-score fell from 83-130 us to
/// 30-45 us from `d = 128` to `d = 960`, and it stopped growing with the length
/// of the walk before it, as it had while every batch took the trip. At twelve
/// queries in flight it fell to 0.39-0.64 of what it was.
#[derive(Clone)]
pub(crate) struct LocalReads {
    /// The scheduler's own coalescing parameters, kept so that reading by hand
    /// moves the same bytes in the same number of reads.
    block_size: u64,
    max_iop_size: u64,
    /// The most one batch reads in place: [`IN_PLACE_BYTES`] on Linux and zero
    /// elsewhere, where nothing can be asked without waiting and every batch
    /// goes to the blocking pool untried. A field so that a test can move it.
    in_place_bytes: u64,
    /// Where these reads are counted: see [`DirectReads`].
    direct: Arc<DirectReads>,
    /// Opened on the first re-score and shared by every clone of this file.
    /// Lazily, for two reasons: the whole-partition modes never re-score and
    /// would hold a descriptor for nothing, and opening one is a blocking call
    /// that has no business on a runtime worker.
    file: Arc<OnceLock<std::fs::File>>,
}

impl LocalReads {
    /// Reads of this crate's own for the files of `store`, if it is local
    /// storage and the platform has positional reads; `None` otherwise.
    ///
    /// `has_direct_local_paths` and not `is_local`, because that is the
    /// predicate Lance's own reader dispatch turns on: a local store rooted
    /// below `/` addresses its objects relative to that root, and
    /// `to_local_path` would name an absolute path somewhere else entirely. The
    /// failure would be silent rather than loud - a descriptor on another inode,
    /// read at this file's offsets. `file+uring` is left out for the opposite
    /// reason: a caller who configured io_uring asked for the scheduler, and
    /// substituting synchronous reads would undo what they chose.
    pub(crate) fn for_store(store: &ObjectStore, direct: &Arc<DirectReads>) -> Option<Self> {
        (cfg!(unix) && store.has_direct_local_paths() && !store.prefers_lite_scheduler()).then(
            || Self {
                block_size: store.block_size() as u64,
                max_iop_size: store.max_iop_size(),
                in_place_bytes: if cfg!(target_os = "linux") {
                    IN_PLACE_BYTES
                } else {
                    0
                },
                direct: direct.clone(),
                file: Arc::new(OnceLock::new()),
            },
        )
    }

    /// The descriptor of the file at `path` a re-score reads through, opened
    /// the first time one asks.
    ///
    /// `None` when opening it failed, which means the scheduler reads the same
    /// bytes instead. The open runs off the worker: it is a blocking syscall,
    /// and Lance's own local reader takes the same care with the same call.
    async fn descriptor(&self, path: &Path) -> Option<&std::fs::File> {
        if self.file.get().is_none() {
            let path = to_local_path(path);
            if let Ok(Ok(opened)) =
                tokio::task::spawn_blocking(move || std::fs::File::open(path)).await
            {
                // The loser of a race drops its descriptor here rather than
                // publishing a second one.
                let _ = self.file.set(opened);
            }
        }
        self.file.get()
    }
}

/// One file as a read by offset sees it, whichever kind it is: where it is, the
/// scheduler it is open through, and reads of this crate's own when it has
/// them.
pub(crate) struct OffsetFile<'a> {
    pub(crate) path: &'a Path,
    pub(crate) file: &'a FileScheduler,
    pub(crate) local: Option<&'a LocalReads>,
}

impl OffsetFile<'_> {
    /// The vectors of `rows` out of this file, laid out as `layout` says,
    /// fetched by byte offset instead of decoded.
    ///
    /// The bytes are coalesced and counted exactly as the scheduler's would be,
    /// whoever reads them: the scheduler, unless the file has reads of its own
    /// and could open its descriptor, and that descriptor when it could - see
    /// [`LocalReads`] for which thread does that, and `in_place` for what is
    /// tried on the calling thread first. What is skipped is the decoder.
    /// `rows` must ascend, which the scheduler requires and a candidate list
    /// already satisfies.
    pub(crate) async fn read_vectors(
        &self,
        layout: &VectorLayout,
        rows: &[u32],
        dimension: u32,
        stats: &IoStats,
        in_place: impl Fn(&std::fs::File, &mut [u8], u64) -> bool,
    ) -> Result<FixedSizeListArray> {
        let wanted = layout.ranges(rows)?;
        let local = match self.local {
            Some(local) => local.descriptor(self.path).await.map(|file| (local, file)),
            None => None,
        };
        let Some((local, file)) = local else {
            let chunks = self
                .file
                .with_io_stats(stats.recorder())
                .submit_request(wanted, 0)
                .await?;
            return raw::vectors(chunks.iter().map(|chunk| chunk.as_ref()), dimension);
        };

        let reads = raw::coalesced(&wanted, local.block_size, local.max_iop_size);
        let planned = reads.iter().map(|read| read.end - read.start).sum::<u64>();

        // Every read the page cache serves is done here and now; the rest are
        // handed off below, and so is the whole of a batch over the limit.
        let mut blocks = vec![Vec::new(); reads.len()];
        let pending = if planned > local.in_place_bytes {
            (0..reads.len()).collect::<Vec<_>>()
        } else {
            let mut pending = Vec::new();
            for (index, (read, block)) in reads.iter().zip(&mut blocks).enumerate() {
                block.resize((read.end - read.start) as usize, 0);
                if !in_place(file, block, read.start) {
                    pending.push(index);
                }
            }
            pending
        };

        // One trip for all of them. Each is a blocking syscall, and there can be
        // one per candidate, which without a re-score budget is the whole search
        // list - leaving that on a runtime worker would hold it through every one
        // of them with nowhere to yield.
        let handed = pending
            .iter()
            .map(|&index| reads[index].clone())
            .collect::<Vec<_>>();
        if !handed.is_empty() {
            let descriptor = local.file.clone();
            let batch = handed.clone();
            let fetched = tokio::task::spawn_blocking(move || {
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
            for (&index, block) in pending.iter().zip(fetched) {
                blocks[index] = block;
            }
        }

        // Counted here because nothing else counts them at all: these bytes never
        // reach the scheduler, so the query's own sink and the index's running
        // total both have to be told by hand, and told the same reads whichever
        // thread made them. After the reads and not before, so that a read that
        // failed is not charged to an index for the rest of its life - the
        // scheduler charges first and would have, but a number that survives its
        // own failure is worse than one that matches it. The trip is counted
        // only when there was one: an empty request still counts as a request.
        stats.record_request(&reads);
        local.direct.totals.record_request(&reads);
        if !handed.is_empty() {
            local.direct.handed_off.record_request(&handed);
        }

        let slices = raw::slices(&wanted, &reads, &blocks)?;
        raw::vectors(slices.iter().map(|slice| slice.as_ref()), dimension)
    }
}

/// What an index read off its own descriptors rather than through its
/// scheduler.
///
/// `totals` is every such read, whichever thread made it. The scheduler keeps
/// its own counts and never sees these, so an index that did not keep them here
/// would report less than it read. `handed_off` is the part of `totals` that
/// went to the blocking pool - a split of them, never an addition to them - and
/// a request there is one trip to the pool.
#[derive(Debug)]
pub(crate) struct DirectReads {
    pub(crate) totals: IoStats,
    pub(crate) handed_off: IoStats,
}

impl Default for DirectReads {
    fn default() -> Self {
        Self {
            totals: IoStats::new(),
            handed_off: IoStats::new(),
        }
    }
}

impl DirectReads {
    /// The reads counted here, split by the thread that made them.
    ///
    /// The two counters are read one after the other, so while re-scores are
    /// in flight the split can be off by their batches; between passes it is
    /// exact.
    pub(crate) fn split(&self) -> RescoreReads {
        let totals = self.totals.snapshot();
        let handed_off = self.handed_off.snapshot();
        RescoreReads {
            in_place: totals.iops.saturating_sub(handed_off.iops),
            handed_off: handed_off.iops,
            trips: handed_off.requests,
            // Not a read off a descriptor; the index adds it from where it
            // counts it.
            through_lance: 0,
        }
    }
}

/// The most one file's share of a re-score reads on the thread that asked for
/// it: a partition's batch, or, re-scoring from the dataset, one data file's.
///
/// A query re-scoring several partitions is held to it once for each file, one
/// after another on the worker polling them. A batch the page cache holds costs a copy
/// a byte whichever thread copies it, so the only question is which. For a
/// re-score with a budget of twenty - at most twenty vectors in one partition's
/// batch, 10 to 77 kB from `d = 128` to `d = 960` - it is the thread that asked:
/// the copy takes microseconds, and the trip to the blocking pool, which wakes
/// two threads to do it, cost such a batch 53-85 us more than reading it in
/// place did with one query in flight (see [`LocalReads`]). Without a budget a
/// probe re-scores its whole search list and several probes re-score at once:
/// on the blocking pool they copy in parallel, where in place they would copy
/// one after another on one worker, which then answers nothing else until they
/// are done.
/// A megabyte is over thirteen twenty-vector batches at `d = 960`, or 273
/// vectors, short of a whole search list of a few hundred; at `d = 128` it is
/// 2048 vectors, so there even an unbudgeted re-score is read in place.
const IN_PLACE_BYTES: u64 = 1 << 20;

/// Fill `buf` from `offset`, whatever the platform calls it.
///
/// Blocking, so it runs on the blocking pool: it is what a read that
/// [`read_now`] could not serve falls back to, and what reads every batch over
/// [`IN_PLACE_BYTES`].
#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(not(unix))]
fn read_at(_file: &std::fs::File, _buf: &mut [u8], _offset: u64) -> std::io::Result<()> {
    // Unreachable: `with_local_reads` binds no file off Unix.
    Err(std::io::ErrorKind::Unsupported.into())
}

/// Fill `buf` from `offset` if the page cache can do it without waiting for
/// the data, and say whether it did.
///
/// `false` hands the read to [`read_at`], which reads it again from the start
/// into a buffer of its own: anything already copied into `buf` is thrown away
/// with it, so a partial copy never has to be trusted.
#[cfg(target_os = "linux")]
pub(crate) fn read_now(file: &std::fs::File, buf: &mut [u8], offset: u64) -> bool {
    let wanted = buf.len();
    let outcome = rustix::io::preadv2(
        file,
        &mut [std::io::IoSliceMut::new(buf)],
        offset,
        rustix::io::ReadWriteFlags::NOWAIT,
    );
    filled(outcome, wanted)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn read_now(_file: &std::fs::File, _buf: &mut [u8], _offset: u64) -> bool {
    false
}

/// Whether a `preadv2` that came back with `outcome` served all `wanted`
/// bytes.
///
/// Only a full count is served. A short one is a range the page cache holds only
/// part of, or one that runs past the end of the file, which the blocking read
/// then reports; and `0` is also how kernels 5.9 and 5.10 answered a miss
/// instead of with `EAGAIN`. Every error is "not served" as well rather than an
/// error of its own - a filesystem that refuses the flag, a kernel older than
/// 4.14, a sandbox that refuses the call, a signal, a disk fault - because the
/// blocking read that follows asks the same question again and reports whatever
/// is really wrong exactly as it always has.
#[cfg(target_os = "linux")]
fn filled(outcome: rustix::io::Result<usize>, wanted: usize) -> bool {
    matches!(outcome, Ok(read) if read == wanted)
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
    /// storage and the platform has positional reads - see
    /// [`LocalReads::for_store`].
    ///
    /// A builder step rather than part of opening, because only a query wants
    /// it: a build or a maintenance pass reads whole columns once, where the
    /// scheduler's queueing is what it is for. Failure to open is not an error -
    /// the file is already open through the scheduler, and that is the path this
    /// one is an optimisation of.
    ///
    /// `direct` is where the caller counts everything this file reads without
    /// its scheduler, and which of it went to the blocking pool; the caller has
    /// to add the first to whatever the scheduler reports.
    pub(crate) fn with_local_reads(
        mut self,
        store: &ObjectStore,
        direct: &Arc<DirectReads>,
    ) -> Self {
        self.local = LocalReads::for_store(store, direct);
        self
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
    /// always reported it. See [`OffsetFile::read_vectors`] for the rest.
    pub(crate) async fn read_vectors(
        &self,
        rows: &[u32],
        dimension: u32,
        stats: &IoStats,
    ) -> Result<Option<FixedSizeListArray>> {
        self.read_vectors_with(rows, dimension, stats, read_now)
            .await
    }

    /// [`Self::read_vectors`], trying each read with `in_place` on the calling
    /// thread before anything goes to the blocking pool - unless the batch is
    /// over the in-place limit, when none is tried.
    ///
    /// Only a test passes anything but [`read_now`]: which reads the page cache
    /// can serve is the kernel's business, and a test that wants a particular
    /// mix of them has to decide it itself.
    async fn read_vectors_with(
        &self,
        rows: &[u32],
        dimension: u32,
        stats: &IoStats,
        in_place: impl Fn(&std::fs::File, &mut [u8], u64) -> bool,
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
        OffsetFile {
            path: &self.path,
            file: &self.file,
            local: self.local.as_ref(),
        }
        .read_vectors(layout, rows, dimension, stats, in_place)
        .await
        .map(Some)
    }

    /// The reader over every column.
    pub fn whole(self) -> FileReader {
        self.reader
    }
}

/// How many partition files an index keeps open at once - and, in a pool of
/// their own, how many of the dataset's data files a re-score from the dataset
/// keeps open, so an index can hold twice this many entries.
///
/// A descriptor is a real resource and an index can hold thousands of
/// partitions, so this is a cap and not a count. What it caps is entries, and an
/// entry can cost two descriptors rather than one: the reader's, and - once a
/// re-score has run against it on local storage - the one it reads through.
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

/// The files an index has open - its partition files in one pool, the data
/// files a re-score from the dataset reads in another - shared by every query
/// that reads them.
///
/// Opening one is not a read and so is not bounded by anything a read is bounded
/// by: `ScanScheduler::open_file` on local storage is
/// `spawn_blocking(File::open)`, a hop through the blocking pool and a real
/// `open(2)`, and a query paid it once for every partition it probed. The footer
/// that comes with it has been shared through the cache since the cache existed;
/// the descriptor never was.
///
/// It also pins what the handle holds, which is more than a descriptor. For a
/// partition file, the footer the reader was built from is an
/// `Arc<CachedFileMetadata>` shared with the cache, so the cache can evict its
/// entry and reclaim nothing: a few kilobytes a file at the partition sizes
/// this crate is written for. A data file's handle holds the layout of its one
/// column and nothing of its footer - [`crate::data_file`] reads no more of it.
///
/// A handle is stored with the sink of whichever query opened it still bound on,
/// and that sink is never used again: every handout rebinds
/// ([`PartitionFile::with_io_stats`] replaces the recorder rather than adding
/// one), and the raw handle leaves this type only through [`OpenFiles::held`]
/// and [`OpenFiles::hold`], for reads that bind a sink of their own. Storing it
/// that way rather than sink-free keeps the accounting exactly where it was -
/// the query that opens a file pays for its footer, the queries that share it
/// pay for nothing - so a query's two phases still add up to what the scheduler
/// counted for it even when a handle is evicted and opened again mid-run.
pub(crate) struct OpenFiles<F = PartitionFile> {
    cap: usize,
    inner: Mutex<Opened<F>>,
}

struct Opened<F> {
    files: HashMap<Path, Handle<F>>,
    /// Stamped onto a handle whenever it is looked up, so the smallest stamp is
    /// the least recently used.
    tick: u64,
}

impl<F> Default for Opened<F> {
    fn default() -> Self {
        Self {
            files: HashMap::new(),
            tick: 0,
        }
    }
}

struct Handle<F> {
    file: Arc<F>,
    used: u64,
}

impl<F> std::fmt::Debug for OpenFiles<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let held = self.inner.lock().map(|open| open.files.len()).ok();
        f.debug_struct("OpenFiles")
            .field("cap", &self.cap)
            .field("held", &held)
            .finish()
    }
}

impl<F> OpenFiles<F> {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: Mutex::new(Opened::default()),
        }
    }

    /// The open file for `path` as it is held - bound to the counters of the
    /// query that opened it - if this index still holds one.
    ///
    /// Only for reads that bind counters of their own, which
    /// [`OffsetFile::read_vectors`] does - and a data file is asked for nothing
    /// else. [`OpenFiles::get`]'s rebinding clones a reader and a scheduler on
    /// every call, and such a read never looks at what they were bound to.
    pub(crate) fn held(&self, path: &Path) -> Option<Arc<F>> {
        let mut open = self.inner.lock().ok()?;
        open.tick += 1;
        let tick = open.tick;
        let handle = open.files.get_mut(path)?;
        handle.used = tick;
        Some(handle.file.clone())
    }

    /// [`OpenFiles::put`] without the rebinding, for the reads [`Self::held`]
    /// is for.
    ///
    /// Keeps what is already held under `path` when there is one rather than
    /// replacing it: two queries that miss on the same partition at the same
    /// moment both open the file, and the loser's handle is dropped here so that
    /// both of them read through one descriptor rather than two.
    pub(crate) fn hold(&self, path: &Path, file: F) -> Arc<F> {
        // Both of these hold descriptors, and both are dropped after the guard
        // goes out of scope: closing a file on a slow mount inside the lock
        // would stall every other query's lookup behind it.
        let mut loser = None;
        let mut evicted = Vec::new();
        let shared = {
            let Ok(mut open) = self.inner.lock() else {
                return Arc::new(file);
            };
            open.tick += 1;
            let tick = open.tick;
            match open.files.get_mut(path) {
                Some(handle) => {
                    handle.used = tick;
                    loser = Some(file);
                    handle.file.clone()
                }
                None => {
                    let held = Arc::new(file);
                    let shared = held.clone();
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

impl OpenFiles {
    /// The open file for `path` with `stats` bound onto it, if this index still
    /// holds one.
    pub(crate) fn get(&self, path: &Path, stats: &IoStats) -> Option<PartitionFile> {
        self.held(path).map(|file| file.with_io_stats(stats))
    }

    /// Hold `file` for the queries after this one, and return the handle they
    /// will all share, with `stats` bound onto it.
    pub(crate) fn put(&self, path: &Path, file: PartitionFile, stats: &IoStats) -> PartitionFile {
        self.hold(path, file).with_io_stats(stats)
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

/// Read a whole partition back into memory, vectors and all.
///
/// For a segment that keeps its vectors. A partition of one that leaves them to
/// the dataset has no [`VECTOR_COLUMN`] to read, which this reports as a missing
/// column: the vectors of such a partition are the dataset's to give.
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
        let size = write_partition(
            &self.store,
            &path,
            partition,
            codes.as_ref(),
            self.metadata.vector_source,
        )
        .await?;
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
        let entry = self.carried_entry(from, partition_id)?;
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

    /// Write partition `partition_id` of `from` into this segment as `stored`,
    /// the batch its file holds, with every vertex at `row_ids` instead of the
    /// address it was stored under.
    ///
    /// For a partition whose rows a deferred compaction moved and nothing else
    /// touched: the graph, the codes and any vectors it keeps are what they
    /// were, so they are carried over as read rather than built into a graph
    /// and coded
    /// again - a code is a function of a vector and the partition's centroid,
    /// and neither moved with the row. That is also what spares a segment
    /// without vectors from reading every one of them out of the dataset.
    ///
    /// Checked as [`Self::copy_partition`] checks, and then for what a copy
    /// never has to: that `stored` is the shape both segments declare, and
    /// that the centroid its codes were taken against is this segment's.
    pub(crate) async fn write_readdressed(
        &mut self,
        from: &SegmentManifest,
        partition_id: u32,
        stored: &RecordBatch,
        row_ids: Vec<u64>,
    ) -> Result<u64> {
        let entry = self.carried_entry(from, partition_id)?;
        let file = entry.file.as_str();
        if stored.num_rows() != entry.num_rows as usize {
            return Err(Error::corrupt_file_named(
                file,
                format!(
                    "Vamana partition {partition_id} lists {} vertices and its file holds {}",
                    entry.num_rows,
                    stored.num_rows()
                ),
            ));
        }
        if row_ids.len() != stored.num_rows() {
            return Err(Error::internal(format!(
                "Vamana was given {} addresses for the {} vertices of partition {partition_id}",
                row_ids.len(),
                stored.num_rows()
            )));
        }
        // `IvfModel::centroid` indexes without a check, so a partition past the
        // end of either model is asked about first.
        let centroid = |ivf: &IvfModel| {
            ((partition_id as usize) < ivf.num_partitions())
                .then(|| ivf.centroid(partition_id as usize))
                .flatten()
        };
        if centroid(from.ivf()) != centroid(&self.ivf) {
            return Err(Error::invalid_input(format!(
                "Vamana cannot carry partition {partition_id} between segments that do not route \
                 it by one centroid; its rows were filed, and any codes taken, against another"
            )));
        }
        // The edges as the graph they are, which a copy of the bytes would
        // never look at and reading the partition to rewrite it always did.
        graph_from_batch(stored)?;

        let stride = self
            .metadata
            .codes
            .as_ref()
            .map(|codes| codes.stride(self.metadata.dimension))
            .transpose()?;
        let schema = Arc::new(partition_schema(
            self.metadata.max_degree,
            self.metadata.dimension,
            stride,
            self.metadata.vector_source,
        )?);
        let row_ids: ArrayRef = Arc::new(UInt64Array::from(row_ids));
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                if field.name() == ROW_ID_COLUMN {
                    Ok(row_ids.clone())
                } else {
                    carried(stored, field, file)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = RecordBatch::try_new(schema, columns)?;

        let written = partition_file_name(partition_id);
        let size = write_batch(
            &self.store,
            &self.dir.clone().join(written.as_str()),
            &batch,
        )
        .await?;
        self.partitions.push(PartitionEntry {
            partition_id,
            medoid: entry.medoid,
            num_rows: entry.num_rows,
            file: written,
        });
        Ok(size)
    }

    /// The entry of partition `partition_id` in `from`, if its bytes can be
    /// carried into this segment as they are, graph, vectors and codes alike.
    fn carried_entry<'a>(
        &self,
        from: &'a SegmentManifest,
        partition_id: u32,
    ) -> Result<&'a PartitionEntry> {
        let entry = from.partition(partition_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "Vamana was asked to carry partition {partition_id} over from a segment that does \
                 not list it"
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
                    "Vamana cannot carry partition {partition_id} from a segment declaring {what} \
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
                "Vamana cannot carry partition {partition_id} between segments whose codes \
                 disagree; the rotation a code was built under is not recoverable from it"
            )));
        }
        if from.metadata().vector_source != self.metadata.vector_source {
            return Err(Error::invalid_input(format!(
                "Vamana cannot carry partition {partition_id} from a segment whose vectors are in \
                 the {} into one whose vectors are in the {}; the file either holds them or does \
                 not",
                from.metadata().vector_source,
                self.metadata.vector_source
            )));
        }
        self.check_entry(partition_id, entry.medoid, entry.num_rows)?;
        Ok(entry)
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

    use std::cell::RefCell;

    use arrow_array::{Array, Float32Array};
    use arrow_schema::{DataType, Field};
    use futures::FutureExt;
    use lance_arrow::FixedSizeListArrayExt;

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
        write_partition(
            &store,
            &path,
            &sample_partition(),
            None,
            VectorSource::Index,
        )
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

    /// Every value of `vectors`, in order.
    fn values(vectors: &FixedSizeListArray) -> Vec<f32> {
        vectors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    /// What `sample_partition` holds, row after row.
    fn every_value() -> Vec<f32> {
        (0..3 * DIMENSION).map(|value| value as f32).collect()
    }

    /// `sample_partition` opened for local reads, split so finely that two of
    /// its reads cut a vector in half.
    ///
    /// The three vectors are adjacent, so they coalesce into one 36-byte run,
    /// and a run longer than `max_iop_size` is cut into equal pieces that know
    /// nothing about where a vector ends: at 8 that is five reads of 7, 7, 7, 7
    /// and 8 bytes, the second and the fourth across a boundary. A read handed
    /// off from the middle of a vector has to land back in the middle of it.
    ///
    /// The in-place limit is checked as [`PartitionFile::with_local_reads`]
    /// set it and then set to [`IN_PLACE_BYTES`] everywhere, so that the reads
    /// a test hands its own page cache are tried on every platform.
    #[cfg(unix)]
    async fn split_local(dir: &tempfile::TempDir) -> (PartitionFile, Arc<DirectReads>) {
        let (scheduler, path) = written(dir, "part_00000.idx").await;
        let direct = Arc::new(DirectReads::default());
        let mut file = open(&scheduler, &path)
            .await
            .with_local_reads(&ObjectStore::local(), &direct);
        let local = file
            .local
            .as_mut()
            .expect("a file on local storage did not take the local path");
        let expected = if cfg!(target_os = "linux") {
            IN_PLACE_BYTES
        } else {
            0
        };
        assert_eq!(
            local.in_place_bytes, expected,
            "a file bound for local reads was given the wrong in-place limit"
        );
        local.in_place_bytes = IN_PLACE_BYTES;
        local.max_iop_size = 8;
        (file, direct)
    }

    /// The reads a fake page cache was asked for, checked to be the split the
    /// tests are written against before anything is claimed about them.
    fn assert_split(asked: &[Range<u64>]) {
        const STRIDE: u64 = DIMENSION as u64 * 4;
        assert_eq!(
            asked.len(),
            5,
            "the batch was not split into five reads: {asked:?}"
        );
        let start = asked[0].start;
        for (read, boundary) in [(1, start + STRIDE), (3, start + 2 * STRIDE)] {
            assert!(
                asked[read].start < boundary && boundary < asked[read].end,
                "read {read}, {:?}, does not cut the vector boundary at {boundary}, so the \
                 fixture no longer tests a vector handed off in halves",
                asked[read]
            );
        }
    }

    /// A byte that makes a vector read as about minus three times ten to the
    /// minus sixteen, which no row of `sample_partition` holds.
    const POISON: u8 = 0xA5;

    /// A page cache that holds every read but the ones `refused` names, by the
    /// order they are asked for in, and writes down every read it is asked for.
    ///
    /// A refused read has its buffer filled with [`POISON`] before it is turned
    /// down, so anything that trusts a refused buffer answers with numbers no
    /// row holds.
    fn cache_refusing<'a>(
        refused: &'a [usize],
        asked: &'a RefCell<Vec<Range<u64>>>,
    ) -> impl Fn(&std::fs::File, &mut [u8], u64) -> bool + 'a {
        move |file, buf, offset| {
            let call = {
                let mut asked = asked.borrow_mut();
                asked.push(offset..offset + buf.len() as u64);
                asked.len() - 1
            };
            if refused.contains(&call) {
                buf.fill(POISON);
                return false;
            }
            read_at(file, buf, offset).is_ok()
        }
    }

    /// Only a read that filled its whole buffer was served.
    ///
    /// Everything else is handed to the blocking read, errors included, so the
    /// error a caller sees is still the one that read reports. A short count is
    /// a range the page cache holds only part of or one past the end of the
    /// file, and a zero is also how two kernels answered a miss.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_read_is_served_only_when_it_filled_the_buffer() {
        use rustix::io::Errno;

        assert!(filled(Ok(12), 12));
        for short in [11, 1, 0] {
            assert!(
                !filled(Ok(short), 12),
                "{short} of 12 bytes was taken as served"
            );
        }
        for errno in [
            Errno::AGAIN,
            Errno::OPNOTSUPP,
            Errno::INVAL,
            Errno::NOSYS,
            Errno::PERM,
            Errno::INTR,
            Errno::IO,
        ] {
            assert!(!filled(Err(errno), 12), "{errno:?} was taken as served");
        }
    }

    /// Refused reads go to the blocking pool in one trip and come back where
    /// they were asked for.
    ///
    /// Two refused out of five, both cut across a vector, so a merge that
    /// appended rather than placed, placed in reverse, or trusted a refused
    /// buffer gives values out of order or out of the fixture. The counts are
    /// pinned as well: the query and the index are charged every read exactly
    /// once whichever thread made it, the two refused reads are one trip, and
    /// the split the index reports adds them up to that.
    #[cfg(unix)]
    #[tokio::test]
    async fn refused_reads_go_in_one_trip_and_land_where_they_were_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let (file, direct) = split_local(&dir).await;
        let stats = IoStats::new();
        let asked = RefCell::new(Vec::new());

        let read = file
            .read_vectors_with(
                &[0, 1, 2],
                DIMENSION as u32,
                &stats,
                cache_refusing(&[1, 3], &asked),
            )
            .await
            .unwrap()
            .expect("a file this crate wrote must be addressable");

        assert_split(&asked.borrow());
        assert_eq!(values(&read), every_value());
        for (who, counted) in [
            ("the query", stats.snapshot()),
            ("the index", direct.totals.snapshot()),
        ] {
            assert_eq!(
                (counted.bytes_read, counted.iops, counted.requests),
                (36, 5, 1),
                "{who} was charged {counted:?} for one request of five reads"
            );
        }
        let handed = direct.handed_off.snapshot();
        assert_eq!(
            (handed.bytes_read, handed.iops, handed.requests),
            (14, 2, 1),
            "two refused reads of seven bytes went to the pool as {handed:?}"
        );
        assert_eq!(
            direct.split(),
            RescoreReads {
                in_place: 3,
                handed_off: 2,
                trips: 1,
                through_lance: 0,
            }
        );
    }

    /// A batch the page cache holds never leaves the thread that asked.
    ///
    /// Not a counter's word for it: the read is polled once, outside any
    /// runtime, and has to finish in that one poll. A batch that went to the
    /// blocking pool - even an empty one - cannot even be handed over there,
    /// let alone come back within the poll. The first read of the file happens
    /// inside the runtime, because it opens the descriptor, and that is a trip
    /// of its own.
    #[cfg(unix)]
    #[test]
    fn a_batch_the_page_cache_holds_never_leaves_the_thread() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let (file, direct) = runtime.block_on(split_local(&dir));
        let stats = IoStats::new();
        let asked = RefCell::new(Vec::new());
        runtime
            .block_on(file.read_vectors_with(
                &[0, 1, 2],
                DIMENSION as u32,
                &stats,
                cache_refusing(&[], &asked),
            ))
            .unwrap()
            .unwrap();
        assert_split(&asked.borrow());

        let asked = RefCell::new(Vec::new());
        let read = file
            .read_vectors_with(
                &[0, 1, 2],
                DIMENSION as u32,
                &stats,
                cache_refusing(&[], &asked),
            )
            .now_or_never()
            .expect("a batch the page cache holds waited on something")
            .unwrap()
            .unwrap();

        assert_split(&asked.borrow());
        assert_eq!(values(&read), every_value());
        let handed = direct.handed_off.snapshot();
        assert_eq!(
            (handed.bytes_read, handed.iops, handed.requests),
            (0, 0, 0),
            "a batch served in place was counted as handed off: {handed:?}"
        );
        assert_eq!(
            direct.split(),
            RescoreReads {
                in_place: 10,
                handed_off: 0,
                trips: 0,
                through_lance: 0,
            }
        );
    }

    /// A batch over the limit goes to the blocking pool whole, with not one read
    /// tried in place; a batch exactly at the limit is read in place.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_batch_over_the_limit_is_handed_off_whole() {
        let dir = tempfile::tempdir().unwrap();
        let (mut file, direct) = split_local(&dir).await;

        file.local.as_mut().unwrap().in_place_bytes = 35;
        let asked = RefCell::new(Vec::new());
        let read = file
            .read_vectors_with(
                &[0, 1, 2],
                DIMENSION as u32,
                &IoStats::new(),
                cache_refusing(&[], &asked),
            )
            .await
            .unwrap()
            .unwrap();
        assert!(
            asked.borrow().is_empty(),
            "a 36-byte batch over a limit of 35 was tried in place: {:?}",
            asked.borrow()
        );
        assert_eq!(values(&read), every_value());
        let handed = direct.handed_off.snapshot();
        assert_eq!(
            (handed.bytes_read, handed.iops, handed.requests),
            (36, 5, 1),
            "a batch over the limit went to the pool as {handed:?}"
        );
        assert_eq!(
            direct.split(),
            RescoreReads {
                in_place: 0,
                handed_off: 5,
                trips: 1,
                through_lance: 0,
            }
        );

        file.local.as_mut().unwrap().in_place_bytes = 36;
        let asked = RefCell::new(Vec::new());
        file.read_vectors_with(
            &[0, 1, 2],
            DIMENSION as u32,
            &IoStats::new(),
            cache_refusing(&[], &asked),
        )
        .await
        .unwrap()
        .unwrap();
        assert_split(&asked.borrow());
        assert_eq!(
            direct.split(),
            RescoreReads {
                in_place: 5,
                handed_off: 5,
                trips: 1,
                through_lance: 0,
            },
            "a batch exactly at the limit was handed off"
        );
    }

    /// Whether an `RWF_NOWAIT` read of the first byte of the file at `path` is
    /// served.
    ///
    /// Asked of the kernel directly and never through the code under test, so
    /// that code which stopped reading in place cannot talk a test into
    /// skipping the assertions that would catch it. The callers ask it of a
    /// file they have just written, which the page cache holds: anything but a
    /// served read means a filesystem, kernel or sandbox that refuses the flag.
    #[cfg(target_os = "linux")]
    fn serves_without_waiting(path: &std::path::Path) -> bool {
        let file = std::fs::File::open(path).unwrap();
        let mut byte = [0u8; 1];
        matches!(
            rustix::io::preadv2(
                &file,
                &mut [std::io::IoSliceMut::new(&mut byte)],
                0,
                rustix::io::ReadWriteFlags::NOWAIT,
            ),
            Ok(1)
        )
    }

    /// Whether this machine refuses a read of a page the page cache has just
    /// dropped, asked of a file written for the purpose beside `dir`'s others.
    ///
    /// A sibling rather than the file a test reads, because asking starts the
    /// readahead that would put the page back. Some machines never refuse: a
    /// device that completes a read before the call returns, or a filesystem
    /// whose pages are the file.
    #[cfg(target_os = "linux")]
    fn refuses_a_dropped_page(dir: &std::path::Path) -> bool {
        let path = dir.join("dropped");
        std::fs::write(&path, [7u8; 4096]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        file.sync_all().unwrap();
        rustix::fs::fadvise(&file, 0, None, rustix::fs::Advice::DontNeed).unwrap();
        let mut page = [0u8; 4096];
        !matches!(
            rustix::io::preadv2(
                &file,
                &mut [std::io::IoSliceMut::new(&mut page)],
                0,
                rustix::io::ReadWriteFlags::NOWAIT,
            ),
            Ok(4096)
        )
    }

    /// On a file the page cache holds, the real read serves everything in
    /// place: the premise of reading in place at all, checked on whatever
    /// filesystem the test directory is on.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_warm_file_is_read_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let (scheduler, path) = written(&dir, "part_00000.idx").await;
        let local_path = dir.path().join("part_00000.idx");
        if !serves_without_waiting(&local_path) {
            eprintln!(
                "skipped: the filesystem under {} cannot be asked to read without waiting",
                dir.path().display()
            );
            return;
        }
        let direct = Arc::new(DirectReads::default());
        let file = open(&scheduler, &path)
            .await
            .with_local_reads(&ObjectStore::local(), &direct);

        let read = file
            .read_vectors(&[0, 1, 2], DIMENSION as u32, &IoStats::new())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(values(&read), every_value());
        let totals = direct.totals.snapshot();
        assert!(
            totals.iops > 0,
            "nothing was read at all, so nothing was read in place"
        );
        assert_eq!(
            direct.split(),
            RescoreReads {
                in_place: totals.iops,
                handed_off: 0,
                trips: 0,
                through_lance: 0,
            },
            "a warm file was handed off"
        );
    }

    /// On a file the page cache has dropped, the real read is refused in place
    /// and handed off rather than waited for, and still comes back right.
    ///
    /// Not skipped on a filesystem that refuses the flag outright: there every
    /// read is handed off too, which is exactly what this asserts.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_cold_file_is_handed_off_rather_than_waited_for() {
        let dir = tempfile::tempdir().unwrap();
        let (scheduler, path) = written(&dir, "part_00000.idx").await;
        if !refuses_a_dropped_page(dir.path()) {
            eprintln!(
                "skipped: this machine serves a page the page cache has dropped without \
                 waiting, so nothing under {} can be refused",
                dir.path().display()
            );
            return;
        }
        let direct = Arc::new(DirectReads::default());
        let file = open(&scheduler, &path)
            .await
            .with_local_reads(&ObjectStore::local(), &direct);
        // Written back first, because only a clean page can be dropped.
        let handle = std::fs::File::open(dir.path().join("part_00000.idx")).unwrap();
        handle.sync_all().unwrap();
        rustix::fs::fadvise(&handle, 0, None, rustix::fs::Advice::DontNeed).unwrap();

        let read = file
            .read_vectors(&[0, 1, 2], DIMENSION as u32, &IoStats::new())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(values(&read), every_value());
        let totals = direct.totals.snapshot();
        assert_eq!(
            direct.split(),
            RescoreReads {
                in_place: 0,
                handed_off: totals.iops,
                trips: 1,
                through_lance: 0,
            },
            "a file out of the page cache was read in place"
        );
    }

    /// A read the file cannot satisfy fails as it always has, naming the file,
    /// and charges nobody - rather than being taken for a miss that is then
    /// answered out of a buffer nobody filled.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_read_past_the_end_is_an_error_naming_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let (file, direct) = split_local(&dir).await;
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("part_00000.idx"))
            .unwrap()
            .set_len(0)
            .unwrap();
        let stats = IoStats::new();

        let error = file
            .read_vectors(&[0, 1, 2], DIMENSION as u32, &stats)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::IO { .. }), "{error}");
        let message = error.to_string();
        assert!(
            message.contains("could not read") && message.contains("part_00000.idx"),
            "{message}"
        );
        for (who, counted) in [
            ("the query", stats.snapshot()),
            ("the index", direct.totals.snapshot()),
            ("the hand-off", direct.handed_off.snapshot()),
        ] {
            assert_eq!(
                (counted.iops, counted.requests),
                (0, 0),
                "{who} was charged {counted:?} for a read that failed"
            );
        }
    }

    fn readdressed_metadata() -> IndexMetadata {
        IndexMetadata {
            format_version: crate::format::FORMAT_VERSION,
            max_degree: 2,
            search_list_size: 10,
            alpha: 1.2,
            dimension: DIMENSION as u32,
            distance_type: lance_linalg::distance::DistanceType::L2,
            row_id_mode: crate::format::RowIdMode::Address,
            fragments: vec![0],
            codes: None,
            vector_source: VectorSource::Index,
        }
    }

    /// A routing model of two partitions, the first centroid at `first`.
    fn readdressed_ivf(first: f32) -> IvfModel {
        IvfModel::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(vec![first, first, first, 9.0, 9.0, 9.0]),
                DIMENSION,
            )
            .unwrap(),
            None,
        )
    }

    /// A segment holding `sample_partition` as partitions 0 and 1, and the batch
    /// the first one's file holds.
    async fn stored_segment(dir: &tempfile::TempDir) -> (SegmentManifest, RecordBatch) {
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("from")).unwrap();
        let mut writer = SegmentWriter::new(
            store.clone(),
            path.clone(),
            readdressed_metadata(),
            readdressed_ivf(0.0),
        );
        for partition_id in [0, 1] {
            writer
                .write_partition(partition_id, 1, &sample_partition())
                .await
                .unwrap();
        }
        let manifest = writer.finish().await.unwrap();
        let file = path.join(manifest.partitions()[0].file.as_str());
        let reader = open_file(&scan_scheduler(&store), &file, None, None)
            .await
            .unwrap();
        let stored = read_partition_batch(&reader, 3).await.unwrap();
        (manifest, stored)
    }

    fn readdressing_writer(dir: &tempfile::TempDir, ivf: IvfModel) -> SegmentWriter {
        SegmentWriter::new(
            Arc::new(ObjectStore::local()),
            Path::from_absolute_path(dir.path().join("to")).unwrap(),
            readdressed_metadata(),
            ivf,
        )
    }

    /// A readdressed partition is the stored one, column for column and in the
    /// same schema, at the addresses it was given, under the entry it had.
    #[tokio::test]
    async fn a_readdressed_partition_is_the_stored_one_at_its_new_addresses() {
        let dir = tempfile::tempdir().unwrap();
        let (from, stored) = stored_segment(&dir).await;
        let mut writer = readdressing_writer(&dir, readdressed_ivf(0.0));
        writer
            .write_readdressed(&from, 0, &stored, vec![7, 8, 9])
            .await
            .unwrap();
        let manifest = writer.finish().await.unwrap();
        assert_eq!(manifest.partitions(), &from.partitions()[..1]);

        let store = Arc::new(ObjectStore::local());
        let file = Path::from_absolute_path(dir.path().join("to"))
            .unwrap()
            .join(manifest.partitions()[0].file.as_str());
        let reader = open_file(&scan_scheduler(&store), &file, None, None)
            .await
            .unwrap();
        let written = read_partition_batch(&reader, 3).await.unwrap();
        assert_eq!(written.schema(), stored.schema());
        assert_eq!(row_ids_from_batch(&written).unwrap(), vec![7, 8, 9]);
        for column in 1..stored.num_columns() {
            assert_eq!(
                written.column(column),
                stored.column(column),
                "column {column}"
            );
        }
    }

    /// What reading gave no one the chance to refuse is refused on the way out:
    /// a count of addresses that is not the count of vertices, a centroid other
    /// than the one the codes were taken against, and a column missing, of
    /// another width than declared, or holding a null.
    #[tokio::test]
    async fn readdressing_refuses_what_it_cannot_carry() {
        let dir = tempfile::tempdir().unwrap();
        let (from, stored) = stored_segment(&dir).await;
        let refusal = |stored: RecordBatch, row_ids: Vec<u64>, ivf: IvfModel| {
            let mut writer = readdressing_writer(&dir, ivf);
            let from = &from;
            async move {
                writer
                    .write_readdressed(from, 0, &stored, row_ids)
                    .await
                    .unwrap_err()
            }
        };
        let replaced = |name: &str, column: ArrayRef| {
            RecordBatch::try_from_iter(stored.schema().fields().iter().enumerate().map(
                |(at, field)| {
                    let kept = stored.column(at).clone();
                    (
                        field.name().clone(),
                        if field.name() == name {
                            column.clone()
                        } else {
                            kept
                        },
                    )
                },
            ))
            .unwrap()
        };

        let error = refusal(stored.clone(), vec![7, 8], readdressed_ivf(0.0)).await;
        assert!(matches!(error, Error::Internal { .. }), "{error}");
        assert!(error.to_string().contains("2 addresses"), "{error}");

        let error = refusal(stored.clone(), vec![7, 8, 9], readdressed_ivf(1.0)).await;
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("one centroid"), "{error}");

        let without = stored.project(&[0, 1]).unwrap();
        let error = refusal(without, vec![7, 8, 9], readdressed_ivf(0.0)).await;
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("__vector is missing"), "{error}");

        let narrower = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 6]), 2)
                .unwrap(),
        );
        let error = refusal(
            replaced(VECTOR_COLUMN, narrower),
            vec![7, 8, 9],
            readdressed_ivf(0.0),
        )
        .await;
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("holds 2 x Float32"), "{error}");

        let holed = Arc::new(FixedSizeListArray::from_iter_primitive::<
            arrow_array::types::UInt32Type,
            _,
            _,
        >(
            vec![
                Some(vec![Some(1), Some(2)]),
                None,
                Some(vec![Some(0), Some(1)]),
            ],
            2,
        ));
        let error = refusal(
            replaced(crate::format::NEIGHBORS_COLUMN, holed),
            vec![7, 8, 9],
            readdressed_ivf(0.0),
        )
        .await;
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("holds nulls"), "{error}");

        let astray = Arc::new(FixedSizeListArray::from_iter_primitive::<
            arrow_array::types::UInt32Type,
            _,
            _,
        >(
            vec![
                Some(vec![Some(5), Some(crate::format::NO_NEIGHBOR)]),
                Some(vec![Some(2), Some(crate::format::NO_NEIGHBOR)]),
                Some(vec![Some(0), Some(crate::format::NO_NEIGHBOR)]),
            ],
            2,
        ));
        let error = refusal(
            replaced(crate::format::NEIGHBORS_COLUMN, astray),
            vec![7, 8, 9],
            readdressed_ivf(0.0),
        )
        .await;
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("local id 5"), "{error}");

        // A partition this segment's router has no centroid for, which asking
        // the router about would have panicked on rather than refused.
        let one_centroid = IvfModel::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 3]), DIMENSION)
                .unwrap(),
            None,
        );
        let error = readdressing_writer(&dir, one_centroid)
            .write_readdressed(&from, 1, &stored, vec![7, 8, 9])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("one centroid"), "{error}");
    }
}
