// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading and writing partition files.
//!
//! Everything here goes through the published `lance-file` / `lance-io` crates,
//! so a partition file is an ordinary Lance file that Lance's own reader can
//! open. Nothing in this module needs the `lance` crate.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::{create_writer, reader_projection_from_column_names};
use lance_file::writer::FileWriterOptions;
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

use crate::format::NEIGHBORS_COLUMN;
use crate::partition::PartitionGraph;

/// The file format partitions are written in.
///
/// Pinned rather than inferred: the constant-stride layout the whole design
/// rests on is a property of a specific structural encoding, so the writer must
/// not drift onto another version silently.
pub const PARTITION_FILE_VERSION: ConcreteFileVersion = ConcreteFileVersion::V2_1;

/// Write one partition and return the size of the file in bytes.
pub async fn write_partition(
    store: &ObjectStore,
    path: &Path,
    graph: &PartitionGraph,
) -> Result<u64> {
    let batch = graph.to_batch()?;
    let schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref())?;
    let mut writer = create_writer(
        PARTITION_FILE_VERSION,
        store.create(path).await?,
        schema,
        FileWriterOptions::default(),
    )?;
    writer.write_batch(&batch).await?;
    Ok(writer.finish().await?.size_bytes)
}

/// Open a partition file for reading.
///
/// `projection` narrows what is fetched; pass `None` to read every column.
pub async fn open_partition(
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
        reader_projection_from_column_names(PARTITION_FILE_VERSION, reader.schema(), columns)?;
    FileReader::try_open(
        file,
        Some(projection),
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
}

/// Read a contiguous run of vertices.
///
/// `Range` rather than the whole file on purpose: this is the call a graph
/// traversal makes, and the reason `__neighbors` has a fixed stride is that
/// this read must fetch `max_degree * 4` bytes per vertex and nothing else.
pub async fn read_vertices(reader: &FileReader, vertices: Range<usize>) -> Result<RecordBatch> {
    if vertices.is_empty() {
        return Err(Error::invalid_input(format!(
            "vertex range {}..{} selects nothing",
            vertices.start, vertices.end
        )));
    }
    let batches = reader
        .read_stream(
            ReadBatchParams::Range(vertices.clone()),
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
                "partition",
                format!(
                    "vertex range {}..{} returned no data",
                    vertices.start, vertices.end
                ),
            )
        })?
        .schema();
    Ok(concat_batches(&schema, batches.iter())?)
}

/// Read a whole partition back into memory.
pub async fn read_partition(reader: &FileReader) -> Result<PartitionGraph> {
    let num_rows = reader.metadata().num_rows as usize;
    if num_rows == 0 {
        // An IVF partition may legitimately hold no vectors, and then there is
        // no batch to take a schema from - so the width comes from the file.
        return PartitionGraph::try_new(max_degree(reader)?, Vec::new(), Vec::new());
    }
    PartitionGraph::try_from_batch(&read_vertices(reader, 0..num_rows).await?)
}

/// The `max_degree` a partition file was written with.
pub fn max_degree(reader: &FileReader) -> Result<u32> {
    let schema: ArrowSchema = reader.schema().as_ref().into();
    let field = schema.field_with_name(NEIGHBORS_COLUMN)?;
    let DataType::FixedSizeList(_, width) = field.data_type() else {
        return Err(Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            format!(
                "Vamana neighbours column has type {}, expected a fixed size list",
                field.data_type()
            ),
        ));
    };
    u32::try_from(*width).map_err(|_| {
        Error::corrupt_file_named(
            NEIGHBORS_COLUMN,
            format!("Vamana neighbours column has a negative width {width}"),
        )
    })
}
