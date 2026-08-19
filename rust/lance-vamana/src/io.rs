// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading and writing a segment: `index.idx` plus one file per partition.
//!
//! Everything here goes through the published `lance-file` / `lance-io` crates,
//! so every file we write is an ordinary Lance file that Lance's own reader can
//! open. Nothing in this module needs the `lance` crate.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
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
    INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, IndexMetadata, NEIGHBORS_COLUMN,
    VECTOR_COLUMN, index_schema, partition_file_name,
};
use crate::partition::{Partition, PartitionGraph};
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

/// Open a file of a segment for reading.
///
/// `columns` narrows what is fetched; pass `None` to read every column.
pub async fn open_file(
    store: Arc<ObjectStore>,
    path: &Path,
    columns: Option<&[&str]>,
) -> Result<FileReader> {
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
    let file = scheduler
        .open_file(path, &CachedFileSize::unknown())
        .await?;
    let reader = FileReader::try_open(
        file.clone(),
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await?;

    let Some(columns) = columns else {
        return Ok(reader);
    };
    let projection =
        reader_projection_from_column_names(SEGMENT_FILE_VERSION, reader.schema(), columns)?;
    FileReader::try_open(
        file,
        Some(projection),
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
}

/// Read a contiguous run of rows.
///
/// `Range` rather than the whole file on purpose: this is the call a graph
/// traversal makes, and the reason `__neighbors` has a fixed stride is that
/// such a read must fetch `max_degree * 4` bytes per vertex and nothing else.
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
pub async fn read_partition(reader: &FileReader) -> Result<Partition> {
    let num_rows = reader.metadata().num_rows as usize;
    if num_rows == 0 {
        // An IVF partition may legitimately hold no vectors, and then there is no
        // batch to take a schema from - so both widths come from the file itself.
        let graph = PartitionGraph::try_new(max_degree(reader)?, Vec::new(), Vec::new())?;
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            list_width(reader, VECTOR_COLUMN)?,
            Arc::new(Float32Array::from(Vec::<f32>::new())),
            None,
        )?;
        return Partition::try_new(graph, vectors);
    }
    Partition::try_from_batch(&read_rows(reader, 0..num_rows).await?)
}

/// The `max_degree` a partition file was written with.
pub fn max_degree(reader: &FileReader) -> Result<u32> {
    positive_width(reader, NEIGHBORS_COLUMN)
}

/// The vector dimension a partition file was written with.
pub fn dimension(reader: &FileReader) -> Result<u32> {
    positive_width(reader, VECTOR_COLUMN)
}

fn positive_width(reader: &FileReader, column: &str) -> Result<u32> {
    let width = list_width(reader, column)?;
    u32::try_from(width).map_err(|_| {
        Error::corrupt_file_named(
            column,
            format!("Vamana {column} column has a negative width {width}"),
        )
    })
}

fn list_width(reader: &FileReader, column: &str) -> Result<i32> {
    let schema: ArrowSchema = reader.schema().as_ref().into();
    let field = schema.field_with_name(column)?;
    let DataType::FixedSizeList(_, width) = field.data_type() else {
        return Err(Error::corrupt_file_named(
            column,
            format!(
                "Vamana {column} column has type {}, expected a fixed size list",
                field.data_type()
            ),
        ));
    };
    Ok(*width)
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
pub async fn read_segment(store: Arc<ObjectStore>, dir: &Path) -> Result<SegmentManifest> {
    let reader = open_file(store, &dir.clone().join(INDEX_FILE_NAME), None).await?;
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
    let ivf = IvfModel::try_from(pb::Ivf::decode(
        reader.read_global_buffer(ivf_position).await?,
    )?)?;

    let num_rows = reader.metadata().num_rows as usize;
    let batch = if num_rows == 0 {
        RecordBatch::new_empty(Arc::new(index_schema()))
    } else {
        read_rows(&reader, 0..num_rows).await?
    };
    SegmentManifest::try_from_batch(metadata, ivf, &batch)
}
