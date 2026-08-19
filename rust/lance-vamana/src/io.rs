// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading and writing a segment: `index.idx` plus one file per partition.
//!
//! Everything here goes through the published `lance-file` / `lance-io` crates,
//! so every file we write is an ordinary Lance file that Lance's own reader can
//! open. Nothing in this module needs the `lance` crate.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_select::concat::concat_batches;
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
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
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use prost::Message;

use crate::format::{
    INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, IndexMetadata, index_schema,
    partition_file_name,
};
use crate::partition::Partition;
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
) -> Result<u64> {
    // The format says an empty partition gets no row in `index.idx` and no file.
    // `SegmentWriter` enforces that; this function is public and delegated to, so
    // it has to enforce it too rather than write a file nothing can point at.
    if partition.is_empty() {
        return Err(Error::invalid_input(
            "Vamana will not write a file for an empty partition".to_string(),
        ));
    }
    let batch = partition.to_batch()?;
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

/// Open a file of a segment for reading.
///
/// `columns` narrows what is fetched; pass `None` to read every column.
/// `size_bytes` skips the size probe when the caller already knows the answer -
/// Lance records the size of every file of a committed index in the dataset
/// manifest, so at query time it always does.
pub async fn open_file(
    scheduler: &Arc<ScanScheduler>,
    path: &Path,
    columns: Option<&[&str]>,
    size_bytes: Option<u64>,
) -> Result<FileReader> {
    let options = FileReaderOptions::default();
    let size = size_bytes.map_or_else(CachedFileSize::unknown, CachedFileSize::new);
    let file = scheduler.open_file(path, &size).await?;
    let reader = FileReader::try_open(
        file.clone(),
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        options.clone(),
    )
    .await?;
    // The version is pinned on the way out and therefore has to be checked on
    // the way in. It is not a formality: the projection below is computed
    // against the structural grammar of [`SEGMENT_FILE_VERSION`], and a file
    // written under another one lays its columns out differently - the read
    // would succeed and return the wrong bytes rather than fail.
    if reader.metadata().version() != SEGMENT_FILE_VERSION {
        return Err(Error::corrupt_file_named(
            path.filename().unwrap_or(INDEX_FILE_NAME),
            format!(
                "Vamana segment file is a Lance {} file, and this crate writes and reads {}",
                reader.metadata().version(),
                SEGMENT_FILE_VERSION
            ),
        ));
    }

    let Some(columns) = columns else {
        return Ok(reader);
    };
    // Reopened from the metadata the first open already read, not from the path.
    // A projection changes what is decoded, not what the file says about itself,
    // and `try_open` would go back to storage for the footer to be told so.
    let projection =
        reader_projection_from_column_names(SEGMENT_FILE_VERSION, reader.schema(), columns)?;
    FileReader::try_open_with_file_metadata(
        Arc::new(LanceEncodingsIo::new(file).with_read_chunk_size(options.read_chunk_size)),
        path.clone(),
        Some(projection),
        Arc::<DecoderPlugins>::default(),
        reader.metadata().clone(),
        &LanceCache::no_cache(),
        options,
    )
    .await
}

/// Read a contiguous run of rows.
///
/// `Range` rather than the whole file because the layout is built for it: the
/// reason `__neighbors` has a fixed stride is that reading one vertex must fetch
/// `max_degree * 4` bytes and nothing else. Nothing does that yet - both callers
/// read a partition whole - so today the range is always `0..num_rows`. It stays
/// a range because the lazy traversal that will use it is the point of the
/// layout, and a whole-file signature would quietly give that up.
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

/// Read a whole partition back into memory.
///
/// `expected_rows` comes from the segment table in `index.idx`, which is a
/// different file from the one being read. Requiring the two to agree is what
/// keeps a damaged footer from being believed, and it is also the only ceiling
/// on this read: without it the row count written in the footer is what decides
/// how much memory to allocate.
pub async fn read_partition(reader: &FileReader, expected_rows: u32) -> Result<Partition> {
    if reader.metadata().num_rows != expected_rows as u64 {
        return Err(Error::corrupt_file_named(
            "partition",
            format!(
                "Vamana partition file holds {} rows but the segment table lists {expected_rows}",
                reader.metadata().num_rows
            ),
        ));
    }
    // No empty-partition branch: an empty partition is written no file and given
    // no row in the segment table, so `expected_rows` is never zero on any path
    // that reaches here, and a caller who passes zero anyway gets the empty-range
    // error from `read_rows` rather than a partition invented from a schema.
    Partition::try_from_batch(&read_rows(reader, 0..expected_rows as usize).await?)
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
        if medoid as usize >= partition.len() {
            return Err(Error::invalid_input(format!(
                "Vamana partition {partition_id} has medoid {medoid} but holds only {} vertices",
                partition.len()
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

        let file = partition_file_name(partition_id);
        let path = self.dir.clone().join(file.as_str());
        let size = write_partition(&self.store, &path, partition).await?;
        self.partitions.push(PartitionEntry {
            partition_id,
            medoid,
            num_rows: partition.len() as u32,
            file,
        });
        Ok(size)
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
