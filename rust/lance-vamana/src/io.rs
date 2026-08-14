// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading and writing a segment: `index.idx` plus one file per partition.
//!
//! Everything here goes through the published `lance-file` / `lance-io` crates,
//! so every file we write is an ordinary Lance file that Lance's own reader can
//! open. Nothing in this module needs the `lance` crate.

use std::ops::Range;
use std::sync::Arc;

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
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use prost::Message;

use crate::cache::FileKey;
use crate::codes::encode;
use crate::format::{
    INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, IndexMetadata, index_schema,
    partition_file_name,
};
use crate::partition::{Partition, row_ids_from_batch};
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
}

impl PartitionFile {
    /// Open `path`, reading its footer once.
    ///
    /// `size_bytes` skips the size probe when the caller already knows the
    /// answer - Lance records the size of every file of a committed index in the
    /// dataset manifest, so at query time it always does.
    pub async fn open(
        scheduler: &Arc<ScanScheduler>,
        path: &Path,
        size_bytes: Option<u64>,
    ) -> Result<Self> {
        Self::open_with(scheduler, path, size_bytes, None).await
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
    ) -> Result<Self> {
        Self::open_with(scheduler, path, size_bytes, Some(cache)).await
    }

    async fn open_with(
        scheduler: &Arc<ScanScheduler>,
        path: &Path,
        size_bytes: Option<u64>,
        cache: Option<&LanceCache>,
    ) -> Result<Self> {
        let size = size_bytes.map_or_else(CachedFileSize::unknown, CachedFileSize::new);
        let file = scheduler.open_file(path, &size).await?;
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
        })
    }

    /// A reader that decodes `columns` and nothing else.
    ///
    /// Built from the metadata [`Self::open`] already read rather than from the
    /// path: a projection changes what is decoded, not what the file says about
    /// itself, and `try_open` would go back to storage for the footer to be told
    /// so.
    pub async fn project(&self, columns: &[&str]) -> Result<FileReader> {
        let options = FileReaderOptions::default();
        let projection = reader_projection_from_column_names(
            SEGMENT_FILE_VERSION,
            self.reader.schema(),
            columns,
        )?;
        FileReader::try_open_with_file_metadata(
            Arc::new(
                LanceEncodingsIo::new(self.file.clone())
                    .with_read_chunk_size(options.read_chunk_size),
            ),
            self.path.clone(),
            Some(projection),
            Arc::<DecoderPlugins>::default(),
            self.reader.metadata().clone(),
            &LanceCache::no_cache(),
            options,
        )
        .await
    }

    /// The reader over every column.
    pub fn whole(self) -> FileReader {
        self.reader
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
    let file = PartitionFile::open(scheduler, path, size_bytes).await?;
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
