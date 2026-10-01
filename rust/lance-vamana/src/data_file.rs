// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One of the dataset's own data files, open to read the vectors of one column
//! by offset, and where in it those vectors are.
//!
//! Where they are is in the footer, and Lance reads a footer for a caller only
//! whole: [`FileReader::read_all_metadata`](lance_file::reader::FileReader::read_all_metadata)
//! decodes the metadata of every column, which on a wide table is nearly all of
//! what it reads and all of what it holds, and the narrower read Lance's own
//! take makes is private to it. So the one column is read here: the fixed
//! footer at the end of the file, the column's entry in the table of where each
//! column's metadata is, and that metadata. All three come out of one read of
//! the file's tail when they fall inside it, which on a table of a few columns
//! they do, and out of at most two more small reads when they do not.
//!
//! Every file version the offset read accepts - 2.1 on - lays the footer and
//! the table out alike and describes a page with the same protobuf, and every
//! check Lance makes on what this reads - the footer, the column's entry in the
//! table, the column's metadata - is made here too. The other columns' entries
//! and the global buffers are neither read nor checked: nothing in them says
//! where this column's values are. Anything this does not recognise gives no
//! layout rather than an error, and the rows are then read through Lance, which
//! reads anything and reports what is really wrong. What is an error here is a
//! read that fails.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::FixedSizeListArray;
use lance_core::Result;
use lance_core::cache::{CacheKey, CacheKeySchema, Context, DeepSizeOf, KeyBuilder};
use lance_encoding::format::{pb, pb21};
use lance_file::format::{MAGIC, pbfile};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{FileScheduler, IoStats};
use object_store::path::Path;
use prost::{Message, Name};

use crate::io::{DirectReads, LocalReads, OffsetFile, read_now};
use crate::raw::{PageView, VectorLayout};

/// Where one of the dataset's own data files keeps the indexed vectors.
///
/// Found through the dataset manifest, which records both halves: the file's
/// format version and which of its physical columns holds which field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DataColumn {
    pub(crate) version: ConcreteFileVersion,
    /// The physical column, under `version`'s grammar.
    pub(crate) index: u32,
    /// The width the dataset schema declares for the field:
    /// `FixedSizeList<Float32, items>`.
    pub(crate) items: u64,
}

/// The bytes at the end of every Lance file from 2.0 on that say where the
/// rest of its metadata is.
const FOOTER_LEN: usize = 40;

/// One entry of the table of where each column's metadata is: its position
/// and its length, eight bytes each.
const TABLE_ENTRY_LEN: u64 = 16;

/// What Lance aligns every page buffer to, and refuses a page on the way in
/// for not being aligned to. Lance's own constant is private to its file
/// reader, so this repeats it.
const PAGE_BUFFER_ALIGNMENT: u64 = 64;

/// The part of a data file's footer this reads.
struct Footer {
    column_meta_start: u64,
    column_meta_offsets_start: u64,
    global_buff_offsets_start: u64,
    num_columns: u32,
    version: ConcreteFileVersion,
}

impl Footer {
    /// The footer in the last [`FOOTER_LEN`] bytes of `tail`, if they are one:
    /// Lance's magic at the very end and a version Lance knows.
    fn parse(tail: &[u8]) -> Option<Self> {
        let footer = tail.get(tail.len().checked_sub(FOOTER_LEN)?..)?;
        if footer.get(36..)? != MAGIC {
            return None;
        }
        let u64_at = |at: usize| Some(u64::from_le_bytes(footer.get(at..at + 8)?.try_into().ok()?));
        let u32_at = |at: usize| Some(u32::from_le_bytes(footer.get(at..at + 4)?.try_into().ok()?));
        let u16_at = |at: usize| Some(u16::from_le_bytes(footer.get(at..at + 2)?.try_into().ok()?));
        Some(Self {
            column_meta_start: u64_at(0)?,
            column_meta_offsets_start: u64_at(8)?,
            global_buff_offsets_start: u64_at(16)?,
            num_columns: u32_at(28)?,
            version: ConcreteFileVersion::from_footer_numbers(u16_at(32)?, u16_at(34)?).ok()?,
        })
    }
}

/// Where the vectors of `column` of the data file behind `file` are, for a
/// column of `rows` rows - [`read_column_metadata`], then [`layout_of`] - or
/// `None` when no offset read can serve them.
pub(crate) async fn read_layout(
    file: &FileScheduler,
    column: DataColumn,
    rows: u64,
) -> Result<Option<VectorLayout>> {
    Ok(read_column_metadata(file, column)
        .await?
        .and_then(|metadata| layout_of(&metadata, column.items, rows)))
}

/// The metadata of `column` of the data file behind `file`, read out of its
/// footer without the metadata of any other column, or `None` when the footer
/// is not what the manifest said the file would be.
///
/// Its version above all: a footer naming another version than the manifest
/// records is left alone, because the column was located under the manifest's
/// grammar, and a grammar that numbered columns differently would have the
/// same number name another column, read at offsets as vectors. Lance reads
/// such a file - it projects by the manifest's version and decodes by the
/// footer's - so its rows are left to Lance's take rather than refused.
async fn read_column_metadata(
    file: &FileScheduler,
    column: DataColumn,
) -> Result<Option<pbfile::ColumnMetadata>> {
    let size = file.reader().size().await? as u64;
    let tail_start = size.saturating_sub(file.reader().block_size() as u64);
    let tail = file.submit_single(tail_start..size, 0).await?;
    let Some(footer) = Footer::parse(&tail) else {
        return Ok(None);
    };
    if footer.version != column.version {
        return Ok(None);
    }
    // Where Lance's own narrow read looks for the table, held to the length it
    // holds it to - an entry a column, back to back - and inside the file.
    let table = footer.column_meta_offsets_start..footer.global_buff_offsets_start;
    let table_len = TABLE_ENTRY_LEN.checked_mul(u64::from(footer.num_columns));
    if column.index >= footer.num_columns
        || table.end.checked_sub(table.start) != table_len
        || table.end > size - FOOTER_LEN as u64
    {
        return Ok(None);
    }
    let entry_start = table.start + TABLE_ENTRY_LEN * u64::from(column.index);
    let entry = read_within(
        file,
        &tail,
        tail_start,
        entry_start..entry_start + TABLE_ENTRY_LEN,
    )
    .await?;
    let half = |at: usize| Some(u64::from_le_bytes(entry.get(at..at + 8)?.try_into().ok()?));
    let (Some(position), Some(length)) = (half(0), half(8)) else {
        return Ok(None);
    };
    let Some(end) = position.checked_add(length) else {
        return Ok(None);
    };
    // Between the start of the column metadata and the table, as Lance holds
    // every entry to.
    if position < footer.column_meta_start || end > footer.column_meta_offsets_start {
        return Ok(None);
    }
    let bytes = read_within(file, &tail, tail_start, position..end).await?;
    Ok(pbfile::ColumnMetadata::decode(bytes.as_ref()).ok())
}

/// The bytes of `range`, out of `tail` - the file from `tail_start` to its end,
/// already read - when they lie inside it, or read on their own when not.
async fn read_within<'a>(
    file: &FileScheduler,
    tail: &'a [u8],
    tail_start: u64,
    range: Range<u64>,
) -> Result<Cow<'a, [u8]>> {
    if range.start >= tail_start {
        let start = (range.start - tail_start) as usize;
        let end = (range.end - tail_start) as usize;
        if let Some(within) = tail.get(start..end) {
            return Ok(Cow::Borrowed(within));
        }
    }
    Ok(Cow::Owned(file.submit_single(range, 0).await?.to_vec()))
}

/// Where the vectors of a column described by `metadata` are, for a column of
/// `rows` rows of `items`-wide vectors, or `None` when no offset read can serve
/// them - which [`VectorLayout::of_pages`] decides, after every check Lance
/// makes on the metadata itself when it reads a column.
fn layout_of(metadata: &pbfile::ColumnMetadata, items: u64, rows: u64) -> Option<VectorLayout> {
    decoded::<pb::ColumnEncoding>(metadata.encoding.as_ref()?)?;
    if metadata.buffer_offsets.len() != metadata.buffer_sizes.len() {
        return None;
    }
    let mut layouts = Vec::with_capacity(metadata.pages.len());
    let mut buffers = Vec::with_capacity(metadata.pages.len());
    for page in &metadata.pages {
        layouts.push(decoded::<pb21::PageLayout>(page.encoding.as_ref()?)?);
        if page.buffer_offsets.len() != page.buffer_sizes.len()
            || page
                .buffer_offsets
                .iter()
                .any(|offset| offset % PAGE_BUFFER_ALIGNMENT != 0)
        {
            return None;
        }
        buffers.push(
            page.buffer_offsets
                .iter()
                .copied()
                .zip(page.buffer_sizes.iter().copied())
                .collect::<Vec<_>>(),
        );
    }
    let pages =
        metadata
            .pages
            .iter()
            .zip(&layouts)
            .zip(&buffers)
            .map(|((page, layout), buffers)| PageView {
                num_rows: page.length,
                layout,
                buffers,
            });
    VectorLayout::of_pages(pages, items, rows)
}

/// The message `encoding` carries in place, if it carries one of type `M`:
/// what Lance's reader accepts there, and nothing it refuses - an encoding kept
/// in a buffer elsewhere in the file, or a message of another type.
fn decoded<M: Message + Name + Default>(encoding: &pbfile::Encoding) -> Option<M> {
    let Some(pbfile::encoding::Location::Direct(direct)) = &encoding.location else {
        return None;
    };
    prost_types::Any::decode(direct.encoding.as_ref())
        .ok()?
        .to_msg::<M>()
        .ok()
}

/// What a data file's footer says about one of its columns, as a cache holds
/// it: where the vectors are, or `None` when no offset read can serve them.
#[derive(Debug)]
pub(crate) struct DataLayout(pub(crate) Option<Arc<VectorLayout>>);

impl DeepSizeOf for DataLayout {
    /// The `Arc`'s own allocation - its two counts and the layout - and what
    /// the layout holds beyond it.
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.as_ref().map_or(0, |layout| {
            2 * std::mem::size_of::<usize>()
                + std::mem::size_of::<VectorLayout>()
                + layout.deep_size_of_children(context)
        })
    }
}

/// One column of one data file, as its layout is cached.
///
/// The column and not only the file: two indices of one dataset can re-score
/// from two columns of the same file and share a cache, and the one that asked
/// second must not be handed the other's layout - a column of the same shape
/// passes every check the offset read makes, and answers with the wrong
/// vectors. The version, the width and the rows with it, because the same index
/// names a different column under another version's grammar, and the width and
/// the rows are what the layout was checked against: a verdict reached for one
/// count of rows is not one for another.
#[derive(Debug)]
pub(crate) struct DataLayoutKey<'a> {
    pub(crate) path: &'a Path,
    pub(crate) column: DataColumn,
    pub(crate) rows: u64,
}

impl CacheKey for DataLayoutKey<'_> {
    type ValueType = DataLayout;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "{}#{}/{}/{}/{}",
            self.path, self.column.index, self.column.items, self.column.version, self.rows
        ))
    }

    fn type_name() -> &'static str {
        "VamanaDataLayout"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance-vamana.data-layout", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.path.as_ref());
        builder.write_u32(self.column.index);
        builder.write_u64(self.column.items);
        builder.write_str(&self.column.version.to_string());
        builder.write_u64(self.rows);
    }
}

/// One of the dataset's data files, open to read the vectors of one column by
/// offset: its scheduler, where that column's vectors are, and reads of this
/// crate's own when the index was given them.
pub(crate) struct DataFile {
    path: Path,
    file: FileScheduler,
    layout: Arc<VectorLayout>,
    local: Option<LocalReads>,
}

impl DataFile {
    pub(crate) fn new(path: Path, file: FileScheduler, layout: Arc<VectorLayout>) -> Self {
        Self {
            path,
            file,
            layout,
            local: None,
        }
    }

    /// Read this file's vectors off a descriptor of its own where `store` is
    /// local storage, as
    /// [`PartitionFile::with_local_reads`](crate::io::PartitionFile::with_local_reads)
    /// does a partition file's.
    pub(crate) fn with_local_reads(
        mut self,
        store: &ObjectStore,
        direct: &Arc<DirectReads>,
    ) -> Self {
        self.local = LocalReads::for_store(store, direct);
        self
    }

    /// The vectors of `rows`, which must ascend, counted into `stats`.
    pub(crate) async fn read_vectors(
        &self,
        rows: &[u32],
        dimension: u32,
        stats: &IoStats,
    ) -> Result<FixedSizeListArray> {
        // What the layout was checked against when it was read: the width the
        // dataset schema declares, which is the one every re-score reads at.
        debug_assert_eq!(
            (self.layout.items(), self.layout.stride()),
            (u64::from(dimension), u64::from(dimension) * 4),
            "a data file laid out for another width was read at {dimension}"
        );
        OffsetFile {
            path: &self.path,
            file: &self.file,
            local: self.local.as_ref(),
        }
        .read_vectors(&self.layout, rows, dimension, stats, read_now)
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use arrow_array::types::Float32Type;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, RecordBatchIterator};
    use lance::Dataset;
    use lance::dataset::WriteParams;
    use lance_core::cache::LanceCache;
    use lance_file::reader::{FileReader, FileReaderOptions};
    use lance_file::version::LanceFileVersion;
    use lance_file::versions::reader_projection_from_field_ids;
    use lance_io::utils::CachedFileSize;

    use crate::io::scan_scheduler;

    /// Rows of every column below: enough for the widest to span two pages of
    /// Lance's 8 MiB, written in batches of [`SLICE`] - a page is flushed at a
    /// batch, so one batch of them all would be one page however large.
    const ROWS: usize = 700;
    const SLICE: usize = 50;

    /// `ROWS` vectors of `width`, every value distinct, and none at `null_at`.
    fn vectors(width: i32, null_at: Option<usize>) -> ArrayRef {
        Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                (0..ROWS).map(|row| {
                    (Some(row) != null_at).then(|| {
                        (0..width as usize)
                            .map(|lane| Some((row * width as usize + lane) as f32))
                            .collect::<Vec<_>>()
                    })
                }),
                width,
            ),
        )
    }

    /// A dataset of one data file, written in `version`, of every kind of
    /// column the offset read meets: narrow vectors Lance lays out in
    /// mini-blocks, wide ones it lays out full-zip, wider ones spanning two
    /// pages, a column holding a null, and one that is not vectors at all.
    async fn every_kind(dir: &tempfile::TempDir, version: LanceFileVersion) -> Dataset {
        let batch = RecordBatch::try_from_iter(vec![
            ("narrow", vectors(8, None)),
            ("vec", vectors(64, None)),
            ("wide", vectors(4096, None)),
            ("nulls", vectors(64, Some(3))),
            (
                "id",
                Arc::new(Int64Array::from_iter_values(0..ROWS as i64)) as ArrayRef,
            ),
        ])
        .unwrap();
        let schema = batch.schema();
        let slices = (0..ROWS)
            .step_by(SLICE)
            .map(|offset| Ok(batch.slice(offset, SLICE.min(ROWS - offset))))
            .collect::<Vec<_>>();
        Dataset::write(
            RecordBatchIterator::new(slices, schema),
            dir.path().to_str().unwrap(),
            Some(WriteParams {
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    /// The dataset's one data file, open through a scheduler of its own.
    async fn data_file(dataset: &Dataset) -> FileScheduler {
        let store = dataset.object_store(None).await.unwrap();
        let fragments = dataset.get_fragments();
        let file = &fragments[0].metadata().files[0];
        scan_scheduler(&store)
            .open_file(
                &dataset.data_dir().join(file.path.as_str()),
                &CachedFileSize::unknown(),
            )
            .await
            .unwrap()
    }

    /// The physical column of field `name`, found as a re-score finds it:
    /// through the manifest, under the file's version.
    fn column_of(dataset: &Dataset, name: &str, version: ConcreteFileVersion) -> u32 {
        let field_id = dataset.schema().field(name).unwrap().id;
        let fragments = dataset.get_fragments();
        let file = &fragments[0].metadata().files[0];
        let columns = file
            .fields
            .iter()
            .zip(file.column_indices.iter())
            .filter_map(|(&field, &column)| {
                Some((u32::try_from(field).ok()?, u32::try_from(column).ok()?))
            })
            .collect::<BTreeMap<_, _>>();
        let projection = reader_projection_from_field_ids(
            version,
            &dataset.schema().project_by_ids(&[field_id], true),
            &columns,
        )
        .unwrap();
        let [index] = projection.column_indices[..] else {
            panic!("{name} is {:?}", projection.column_indices);
        };
        index
    }

    /// The oracle: what is read of one column is what reading the whole footer
    /// reads of it, column for column, and the layout made of it is the one
    /// made of the whole footer - for vectors the offset read serves and for
    /// those it does not.
    async fn one_column_of_the_footer_is_the_whole_footers(version: LanceFileVersion) {
        let dir = tempfile::tempdir().unwrap();
        let dataset = every_kind(&dir, version).await;
        let file = data_file(&dataset).await;
        let concrete = version.resolve();

        let whole = FileReader::read_all_metadata(&file).await.unwrap();
        assert_eq!(whole.version, concrete);
        for (index, expected) in whole.column_metadatas.iter().enumerate() {
            let column = DataColumn {
                version: concrete,
                index: index as u32,
                items: 0,
            };
            let read = read_column_metadata(&file, column).await.unwrap();
            assert_eq!(read.as_ref(), Some(expected), "{version:?}: column {index}");
        }
        let past_the_last = DataColumn {
            version: concrete,
            index: whole.column_metadatas.len() as u32,
            items: 64,
        };
        assert!(
            read_column_metadata(&file, past_the_last)
                .await
                .unwrap()
                .is_none()
        );

        let reader = FileReader::try_open(
            file.clone(),
            None,
            Arc::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        for (name, width, by_offset) in [
            ("narrow", 8, false),
            ("vec", 64, true),
            ("wide", 4096, true),
            ("nulls", 64, false),
        ] {
            let index = column_of(&dataset, name, concrete);
            let expected = VectorLayout::of_column(&reader, index, width);
            assert_eq!(
                expected.is_some(),
                by_offset,
                "{version:?}: {name} is not the fixture it was meant to be"
            );
            let column = DataColumn {
                version: concrete,
                index,
                items: width,
            };
            let read = read_layout(&file, column, ROWS as u64).await.unwrap();
            assert_eq!(read, expected, "{version:?}: {name}");
        }
        let wide = DataColumn {
            version: concrete,
            index: column_of(&dataset, "wide", concrete),
            items: 4096,
        };
        let pages = read_layout(&file, wide, ROWS as u64)
            .await
            .unwrap()
            .map_or(0, |layout| layout.num_pages());
        assert!(
            pages > 1,
            "{version:?}: the wide column fits in {pages} page, so no page boundary is tested"
        );
    }

    #[tokio::test]
    async fn one_column_of_a_2_1_footer_is_the_whole_footers() {
        one_column_of_the_footer_is_the_whole_footers(LanceFileVersion::V2_1).await;
    }

    #[tokio::test]
    async fn one_column_of_a_2_2_footer_is_the_whole_footers() {
        one_column_of_the_footer_is_the_whole_footers(LanceFileVersion::V2_2).await;
    }

    #[tokio::test]
    async fn one_column_of_a_2_3_footer_is_the_whole_footers() {
        one_column_of_the_footer_is_the_whole_footers(LanceFileVersion::V2_3).await;
    }

    /// A data file whose footer names another version than its manifest
    /// records is not read by offset, but left to Lance, which reads it by the
    /// footer's grammar - while the same file recorded as what it is, is.
    #[tokio::test]
    async fn a_data_file_its_manifest_misdates_is_not_read_by_offset() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = every_kind(&dir, LanceFileVersion::V2_2).await;
        let file = data_file(&dataset).await;
        for (version, by_offset) in [
            (ConcreteFileVersion::V2_2, true),
            (ConcreteFileVersion::V2_1, false),
            (ConcreteFileVersion::V2_3, false),
        ] {
            let column = DataColumn {
                version,
                index: column_of(&dataset, "vec", ConcreteFileVersion::V2_2),
                items: 64,
            };
            let read = read_layout(&file, column, ROWS as u64).await.unwrap();
            assert_eq!(read.is_some(), by_offset, "recorded as {version:?}");
        }
    }

    /// A Lance file with its magic gone is not taken for one, although every
    /// other byte of its footer still says where its columns are - and read as
    /// it was written, it gives the layout it always did.
    #[tokio::test]
    async fn a_lance_file_without_its_magic_gives_no_layout() {
        let dir = tempfile::tempdir().unwrap();
        let batch = RecordBatch::try_from_iter(vec![("vec", vectors(64, None))]).unwrap();
        let schema = batch.schema();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir.path().join("dataset").to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        let version = dataset.get_fragments()[0].metadata().files[0]
            .file_version()
            .unwrap();
        let column = DataColumn {
            version,
            index: column_of(&dataset, "vec", version),
            items: 64,
        };
        let written = data_file(&dataset).await;
        assert!(
            read_layout(&written, column, ROWS as u64)
                .await
                .unwrap()
                .is_some()
        );

        let store = dataset.object_store(None).await.unwrap();
        let size = written.reader().size().await.unwrap() as u64;
        let mut bytes = written.submit_single(0..size, 0).await.unwrap().to_vec();
        let at = bytes.len() - MAGIC.len();
        bytes[at..].copy_from_slice(b"LANX");
        let path = Path::from_absolute_path(dir.path().join("unmagicked.lance")).unwrap();
        store.put(&path, &bytes).await.unwrap();
        let unmagicked = scan_scheduler(&store)
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        assert!(
            read_column_metadata(&unmagicked, column)
                .await
                .unwrap()
                .is_none()
        );
    }

    /// Bytes that are not a Lance file give no layout rather than an error:
    /// it is Lance's to say what is wrong with them, when it is asked to read
    /// the rows.
    #[tokio::test]
    async fn what_is_not_a_lance_file_gives_no_layout() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let scheduler = scan_scheduler(&store);
        let column = DataColumn {
            version: ConcreteFileVersion::V2_2,
            index: 0,
            items: 64,
        };
        for (name, content) in [
            ("short", b"not a lance file".to_vec()),
            ("zeros", vec![0u8; 4096]),
        ] {
            let path = Path::from_absolute_path(dir.path().join(name)).unwrap();
            store.put(&path, &content).await.unwrap();
            let file = scheduler
                .open_file(&path, &CachedFileSize::unknown())
                .await
                .unwrap();
            assert!(
                read_column_metadata(&file, column).await.unwrap().is_none(),
                "{name}"
            );
        }
    }
}
