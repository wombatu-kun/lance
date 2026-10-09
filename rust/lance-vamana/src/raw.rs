// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Where a partition's vectors sit in its file - or the dataset's, in one of its
//! data files - so that re-scoring can fetch them without building a decoder
//! for twenty rows.
//!
//! A re-score reads `budget` whole vectors out of one column and computes a
//! distance against each. Going through `FileReader::read_stream` to do it
//! builds a projected reader, a cache of capacity zero, a decode engine, a
//! channel, a scheduler and a task per batch, and the reading itself is twenty
//! ranges of `dimension * 4` bytes - a few kilobytes that the page cache almost
//! always already has. The apparatus costs more than the read.
//!
//! It is avoidable because the column is not really encoded. `__vector` is a
//! `FixedSizeList<Float32, dimension>` that is non-nullable at both levels and
//! carries Lance's `fullzip` hint ([`crate::format::partition_schema`]), and a
//! full-zip page of a fixed-width block stores its values flat: Lance's own
//! scheduler locates row `r` at `data_buf_position + r * bytes_per_value`
//! (`FullZipScheduler::schedule_ranges_simple`), with no validity bitmap, no
//! repetition index and no control words, because there is no repetition or
//! definition to record. What this module does is read that one arithmetic out
//! of the footer - the one Lance already parsed, for a partition file, and one
//! column's part of it read by [`crate::data_file`], for a data file - and then
//! do it itself. A dataset's data
//! file carries no hint: Lance lays a fixed-width value of 256 bytes or more
//! out full-zip on its own ([`crate::format::MIN_DATASET_VECTOR_DIMENSION`]),
//! and whether it did is left to the page checks alone.
//!
//! Every assumption in that paragraph is checked rather than trusted, and a
//! file that fails any of them gets no layout at all - the caller falls back to
//! the decoder, or for a data file to Lance's take, either of which can read
//! anything. That is deliberate: this is an optimisation of a path that already
//! works, so it must never be the reason a file stops being readable.

use std::borrow::Cow;
use std::ops::Range;

use arrow_array::{FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_encoding::decoder::PageEncoding;
use lance_encoding::format::pb21;
use lance_file::reader::FileReader;
use lance_file::versions::reader_projection_from_column_names;

use crate::io::SEGMENT_FILE_VERSION;

/// The byte layout of one fixed-width column of a partition file, or of a data
/// file.
///
/// Built once per file and shared by every query that re-scores out of it.
#[derive(Debug, PartialEq)]
pub(crate) struct VectorLayout {
    /// Bytes one value occupies, which for this column is `dimension * 4`.
    stride: u64,
    /// Values one vector holds, as the column's own type declares it. Checked
    /// against what the segment says the dimension is, because the stride alone
    /// cannot tell a narrower vector of wider values from a wider one.
    items: u64,
    /// In file order, which for one column is also row order.
    pages: Vec<PageSpan>,
}

/// One page of the column: the rows it holds and where its values start.
///
/// A column is not one buffer. The writer flushes a page every 8 MiB, so a
/// million rows of a 960-wide vector is some four hundred of them, each with a
/// position of its own that only the footer knows - the buffers are aligned, so
/// there is no arithmetic that gets from one page to the next.
#[derive(Debug, PartialEq)]
struct PageSpan {
    first_row: u64,
    num_rows: u64,
    position: u64,
}

/// Held in a cache when it describes one of the dataset's data files, which
/// weighs it by this.
impl DeepSizeOf for VectorLayout {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.pages.capacity() * std::mem::size_of::<PageSpan>()
    }
}

/// One page of a column as [`VectorLayout::of_pages`] checks it, whichever
/// footer it was read out of.
#[derive(Debug)]
pub(crate) struct PageView<'a> {
    pub(crate) num_rows: u64,
    pub(crate) layout: &'a pb21::PageLayout,
    /// Where each of the page's buffers is in the file, and how long it is.
    pub(crate) buffers: &'a [(u64, u64)],
}

impl VectorLayout {
    /// The layout of `column`, or `None` if it is not addressable by offset.
    ///
    /// `None` is not an error: it means this file has to be read the ordinary
    /// way. Every check below is a reason to say so - another structural
    /// encoding, a compressed value buffer, nullability, a page whose bytes do
    /// not divide into whole values, pages out of file order.
    ///
    /// They overlap on purpose, and a file this crate cannot write trips several
    /// at once. Reading at a computed offset is the one thing here that fails by
    /// returning the wrong answer rather than by returning an error, so each
    /// check states a different part of the assumption: the descriptors say
    /// there are no control words, the arithmetic says the declared value width
    /// is exactly its values, and the buffer says the page holds exactly its
    /// rows. Any one of them passing while the others fail would mean Lance has
    /// changed something this module needs to be told about.
    pub(crate) fn of(reader: &FileReader, column: &str) -> Option<Self> {
        let projection =
            reader_projection_from_column_names(SEGMENT_FILE_VERSION, reader.schema(), &[column])
                .ok()?;
        // One physical column: a fixed-size list of a primitive bottoms out at
        // its item leaf and nothing else is projected with it. More than one
        // index means the schema is not the shape this module was written for.
        let [index] = projection.column_indices[..] else {
            return None;
        };
        // The declared type, and not only the byte width. Everything below
        // reasons in bytes, and bytes cannot tell `FixedSizeList<Float32, d>`
        // from `FixedSizeList<Float64, d/2>`: both are `4d` bytes a row, both
        // are flat, and reading the second as the first returns numbers rather
        // than an error. On the decoder path `vectors_of` was the check that
        // caught it, and nothing else on the lazy path is in a position to -
        // it never builds a `Partition`.
        let DataType::FixedSizeList(item, width) = reader.schema().field(column)?.data_type()
        else {
            return None;
        };
        if item.data_type() != &DataType::Float32 {
            return None;
        }
        Self::of_column(reader, index, u64::try_from(width).ok()?)
    }

    /// The layout of physical column `index` of the file `reader` read the
    /// footer of, whose values the caller has already established are
    /// `FixedSizeList<Float32, items>`.
    pub(crate) fn of_column(reader: &FileReader, index: u32, items: u64) -> Option<Self> {
        let info = reader.metadata().column_infos.get(index as usize)?;
        let pages = info
            .page_infos
            .iter()
            .map(|page| match &page.encoding {
                PageEncoding::Structural(layout) => Some(PageView {
                    num_rows: page.num_rows,
                    layout,
                    buffers: &page.buffer_offsets_and_sizes,
                }),
                PageEncoding::Legacy(_) => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Self::of_pages(pages, items, reader.num_rows())
    }

    /// The layout of a column of `rows` rows made of `pages`, whose values the
    /// caller has already established are `FixedSizeList<Float32, items>`.
    ///
    /// Split from [`Self::of_column`] for the dataset's own data files, whose
    /// pages are read out of the footer a column at a time rather than all of
    /// them at once by a [`FileReader`]: what the pages have to look like is
    /// the same whichever way they were read, and so is where that is checked.
    pub(crate) fn of_pages<'a>(
        pages: impl IntoIterator<Item = PageView<'a>>,
        items: u64,
        rows: u64,
    ) -> Option<Self> {
        let pages_in = pages.into_iter();
        let mut pages = Vec::with_capacity(pages_in.size_hint().0);
        let mut stride: Option<u64> = None;
        let mut first_row = 0u64;
        let mut written_to = 0u64;
        for page in pages_in {
            let Some(pb21::page_layout::Layout::FullZipLayout(zip)) = page.layout.layout.as_ref()
            else {
                return None;
            };
            // Control words ride in front of every value when there is
            // repetition or definition to record, and they would shift every
            // offset past the first.
            if zip.bits_rep != 0 || zip.bits_def != 0 {
                return None;
            }
            let Some(pb21::full_zip_layout::Details::BitsPerValue(bits)) = zip.details else {
                return None;
            };
            if bits == 0 || bits % 8 != 0 {
                return None;
            }
            let bytes = u64::from(bits / 8);
            // Compression is what makes a value's position depend on the values
            // before it, so the only shape this module can read is the one that
            // applies none: a fixed-size list of flat items, with no validity
            // buffer riding along and no codec over the bytes. Lance describes
            // it in two layers, the list and then the items, and both have to be
            // checked - the outer one alone would accept a dictionary of
            // vectors.
            let Some(pb21::compressive_encoding::Compression::FixedSizeList(list)) = zip
                .value_compression
                .as_ref()
                .and_then(|encoding| encoding.compression.as_ref())
            else {
                return None;
            };
            if list.has_validity {
                return None;
            }
            let Some(pb21::compressive_encoding::Compression::Flat(flat)) = list
                .values
                .as_ref()
                .and_then(|encoding| encoding.compression.as_ref())
            else {
                return None;
            };
            if flat.data.is_some()
                || flat.bits_per_value != 32
                || list.items_per_value != items
                || list.items_per_value.checked_mul(flat.bits_per_value)? != u64::from(bits)
            {
                return None;
            }
            if *stride.get_or_insert(bytes) != bytes {
                return None;
            }

            let [(position, size)] = page.buffers[..] else {
                return None;
            };
            // The arithmetic, checked against the file rather than assumed: the
            // page's own buffer is exactly its rows at that stride. Anything
            // that shortened it - a codec, a bitmap, a packed tail - fails here
            // whatever the descriptors above claimed.
            if size != page.num_rows.checked_mul(bytes)? {
                return None;
            }
            if u64::from(zip.num_items) != page.num_rows {
                return None;
            }
            // Ranges are handed to the scheduler in ascending order, and rows
            // ascend, so positions have to ascend with them.
            if position < written_to {
                return None;
            }
            written_to = position.checked_add(size)?;

            pages.push(PageSpan {
                first_row,
                num_rows: page.num_rows,
                position,
            });
            first_row = first_row.checked_add(page.num_rows)?;
        }

        if first_row != rows {
            return None;
        }
        Some(Self {
            stride: stride?,
            items,
            pages,
        })
    }

    /// Bytes one vector occupies.
    pub(crate) fn stride(&self) -> u64 {
        self.stride
    }

    /// Values one vector holds, as the column declares it.
    pub(crate) fn items(&self) -> u64 {
        self.items
    }

    /// Pages the column spans, for a test to say its fixture spans several.
    #[cfg(test)]
    pub(crate) fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Where each of `rows` is, in the order given.
    ///
    /// `rows` must ascend, which is what the scheduler requires of the ranges
    /// and what a candidate list already is.
    pub(crate) fn ranges(&self, rows: &[u32]) -> Result<Vec<Range<u64>>> {
        // The same contract `read_scattered` enforced before this path existed,
        // and enforced here for the same reason: out of order, the failure
        // surfaces two layers down as a coalescing or a sortedness complaint,
        // and the caller is left guessing which of its lists was unsorted.
        if let Some(pair) = rows.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(Error::internal(format!(
                "Vamana re-scores strictly ascending vertices; got {} then {}",
                pair[0], pair[1]
            )));
        }
        let mut ranges = Vec::with_capacity(rows.len());
        let mut page = 0;
        for &row in rows {
            let row = u64::from(row);
            // One cursor for the whole ascending list rather than a search per
            // row: a re-score reads a handful of rows out of a file with
            // hundreds of pages, and both of them only ever move forward.
            while page < self.pages.len() && row >= self.pages[page].end() {
                page += 1;
            }
            let Some(span) = self.pages.get(page).filter(|span| row >= span.first_row) else {
                return Err(Error::internal(format!(
                    "Vamana re-scored row {row}, which is not a row of a file holding {} rows",
                    self.pages.last().map_or(0, PageSpan::end)
                )));
            };
            let start = span.position + (row - span.first_row) * self.stride;
            ranges.push(start..start + self.stride);
        }
        Ok(ranges)
    }
}

impl PageSpan {
    fn end(&self) -> u64 {
        self.first_row + self.num_rows
    }
}

/// The reads the scheduler would have made for `wanted`.
///
/// Mirrors `FileScheduler::submit_request`: ranges within `block_size` of each
/// other are one read, and a read longer than `max_iop_size` is split into equal
/// pieces. Reading by hand means counting by hand, and a count that did not
/// merge what the scheduler merges would make every byte column of a run
/// incomparable with the runs taken before - which is the only reason this is
/// here, since for scattered candidates it almost never merges anything.
pub(crate) fn coalesced(
    wanted: &[Range<u64>],
    block_size: u64,
    max_iop_size: u64,
) -> Vec<Range<u64>> {
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(wanted.len());
    for range in wanted {
        match merged.last_mut() {
            Some(last) if range.start <= last.end + block_size => {
                last.end = last.end.max(range.end);
            }
            _ => merged.push(range.clone()),
        }
    }
    let mut reads = Vec::with_capacity(merged.len());
    for range in merged {
        if range.is_empty() || max_iop_size == 0 {
            reads.push(range);
            continue;
        }
        let pieces = (range.end - range.start).div_ceil(max_iop_size);
        let each = (range.end - range.start) / pieces;
        for piece in 0..pieces {
            let start = range.start + piece * each;
            let end = if piece == pieces - 1 {
                range.end
            } else {
                start + each
            };
            reads.push(start..end);
        }
    }
    reads
}

/// Where each of `wanted` sits inside the blocks that were read for it.
///
/// Returns one slice per wanted range, in the order asked for, which is what
/// [`vectors`] consumes. Borrowed when the range sits inside one read, which is
/// every range a re-score asks for in practice, and stitched when the split
/// above cut one in half - the scheduler does the same, and a candidate list
/// dense enough to merge past `max_iop_size` would otherwise fail here while
/// succeeding there. A wanted range that starts in no read at all is an error
/// rather than a silent short read: it would mean the coalescing above and the
/// reading below disagree, and the answer would be built out of the wrong bytes.
pub(crate) fn slices<'a>(
    wanted: &[Range<u64>],
    reads: &[Range<u64>],
    blocks: &'a [Vec<u8>],
) -> Result<Vec<Cow<'a, [u8]>>> {
    let mut slices = Vec::with_capacity(wanted.len());
    let mut read = 0;
    for range in wanted {
        while read < reads.len() && reads[read].end <= range.start {
            read += 1;
        }
        let starts_here = reads
            .get(read)
            .is_some_and(|read| read.start <= range.start && range.start < read.end);
        if !starts_here {
            return Err(Error::internal(format!(
                "Vamana read {reads:?} and then wanted {range:?}, which none of them holds"
            )));
        }
        let offset = (range.start - reads[read].start) as usize;
        let wanted_len = (range.end - range.start) as usize;
        if range.end <= reads[read].end {
            slices.push(Cow::Borrowed(&blocks[read][offset..offset + wanted_len]));
            continue;
        }
        // The split does not know where a vector ends, so a merged run longer
        // than one read can be cut in the middle of one. The scheduler puts such
        // a value back together out of the pieces it read; reading by hand means
        // doing that too, or a candidate list dense enough to merge past
        // `max_iop_size` would fail where the scheduler succeeds.
        let mut stitched = Vec::with_capacity(wanted_len);
        stitched.extend_from_slice(&blocks[read][offset..]);
        let mut piece = read;
        while stitched.len() < wanted_len {
            piece += 1;
            let Some(block) = blocks.get(piece) else {
                return Err(Error::internal(format!(
                    "Vamana read {reads:?} and then wanted {range:?}, which runs past all of them"
                )));
            };
            let take = (wanted_len - stitched.len()).min(block.len());
            stitched.extend_from_slice(&block[..take]);
        }
        slices.push(Cow::Owned(stitched));
    }
    Ok(slices)
}

/// A run of whole vectors, in the order they were requested, as the array the
/// decoder would have handed back.
///
/// Each chunk is one vector: `dimension * 4` bytes of little-endian `f32`,
/// which is what Lance writes. The copy is what buys the equality - arrow wants
/// its values aligned and contiguous, and a coalesced read is neither - and it
/// is 10 to 77 kB at a re-score budget of twenty.
pub(crate) fn vectors<'a>(
    chunks: impl ExactSizeIterator<Item = &'a [u8]>,
    dimension: u32,
) -> Result<FixedSizeListArray> {
    let stride = u64::from(dimension) * 4;
    let mut values = Vec::with_capacity(chunks.len() * dimension as usize);
    for chunk in chunks {
        if chunk.len() as u64 != stride {
            return Err(Error::internal(format!(
                "Vamana asked for a {stride}-byte vector and got {} bytes",
                chunk.len()
            )));
        }
        values.extend(
            chunk
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value)),
        );
    }
    Ok(FixedSizeListArray::try_new_from_values(
        Float32Array::from(values),
        dimension as i32,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::Float64Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use arrow_array::{Array, ArrayRef, RecordBatch};
    use arrow_schema::{Fields, Schema as ArrowSchema};
    use lance_file::versions::create_writer;
    use lance_file::writer::FileWriterOptions;
    use lance_io::object_store::ObjectStore;
    use lance_io::scheduler::IoStats;
    use object_store::path::Path;

    use crate::format::{VECTOR_COLUMN, VectorSource};
    use crate::io::{DirectReads, PartitionFile, read_scattered, scan_scheduler, write_partition};
    use crate::partition::{Partition, PartitionGraph, vectors_of};

    /// Wide enough that Lance's 8 MiB page flush lands inside a fixture a test
    /// can afford: at this width a page holds 512 rows.
    const WIDE: u32 = 4096;
    const NARROW: u32 = 3;
    /// An even width, so the same bytes can also be written as half as many
    /// `f64`s.
    const PAIRED: u32 = 4;

    /// Every value distinct, and distinct along both axes: `row * WIDE + lane`
    /// separates a vector read one row over from one read one lane over, which a
    /// ramp along either axis alone would not. Stays exact in `f32` while the
    /// product is under two to the twenty-fourth.
    fn partition(rows: usize, dimension: u32) -> Partition {
        let graph = PartitionGraph::try_new(
            2,
            (0..rows as u64).map(|row| row * 7 + 3).collect(),
            (0..rows)
                .map(|row| vec![((row + 1) % rows) as u32])
                .collect(),
        )
        .unwrap();
        let values = (0..rows)
            .flat_map(|row| (0..dimension).map(move |lane| (row as u32 * dimension + lane) as f32))
            .collect::<Vec<_>>();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension as i32)
                .unwrap();
        Partition::try_new(graph, vectors).unwrap()
    }

    async fn opened(dir: &tempfile::TempDir, rows: usize, dimension: u32) -> PartitionFile {
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("part_00000.idx")).unwrap();
        write_partition(
            &store,
            &path,
            &partition(rows, dimension),
            None,
            VectorSource::Index,
        )
        .await
        .unwrap();
        PartitionFile::open(&scan_scheduler(&store), &path, None, None)
            .await
            .unwrap()
    }

    fn expected(row: u32, dimension: u32) -> Vec<f32> {
        (0..dimension)
            .map(|lane| (row * dimension + lane) as f32)
            .collect()
    }

    fn vector_at(vectors: &FixedSizeListArray, position: usize) -> Vec<f32> {
        vectors
            .value(position)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[tokio::test]
    async fn a_partition_file_is_addressable() {
        let dir = tempfile::tempdir().unwrap();
        let file = opened(&dir, 8, NARROW).await;
        let layout = VectorLayout::of(file.reader(), VECTOR_COLUMN).expect(
            "the vector column of a file this crate wrote is not addressable, so every \
             re-score below falls back and proves nothing",
        );
        assert_eq!(layout.stride(), u64::from(NARROW) * 4);
        assert_eq!(layout.pages.len(), 1);
        assert_eq!(
            layout.ranges(&[0, 7]).unwrap(),
            vec![
                layout.pages[0].position..layout.pages[0].position + layout.stride(),
                layout.pages[0].position + 7 * layout.stride()
                    ..layout.pages[0].position + 8 * layout.stride(),
            ]
        );
    }

    /// The same partition, written a slice at a time so the column crosses a
    /// page boundary.
    ///
    /// One `write_batch` is one page however large it is - the index files this
    /// crate writes today are a single page of several gigabytes - so the only
    /// way to reach the second page is to hand the writer the rows in pieces.
    /// Lance flushes a column once it has buffered 8 MiB of it, which at this
    /// width is 512 rows.
    async fn write_in_slices(
        store: &ObjectStore,
        path: &Path,
        partition: &Partition,
        slice: usize,
    ) {
        let batch = partition.to_batch(None, VectorSource::Index).unwrap();
        let schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap();
        let mut writer = create_writer(
            SEGMENT_FILE_VERSION,
            store.create(path).await.unwrap(),
            schema,
            FileWriterOptions::default(),
        )
        .unwrap();
        let mut offset = 0;
        while offset < batch.num_rows() {
            let rows = slice.min(batch.num_rows() - offset);
            writer
                .write_batch(&batch.slice(offset, rows))
                .await
                .unwrap();
            offset += rows;
        }
        writer.finish().await.unwrap();
    }

    /// The arithmetic that a single-page fixture cannot test at all.
    ///
    /// The buffers are aligned and a page starts wherever the writer put it, so
    /// nothing about the first page predicts the second. The test says the
    /// fixture spans more than one page before it asserts anything else - a
    /// single page would make every claim below vacuously true, which is exactly
    /// what the first draft of this module did.
    #[tokio::test]
    async fn the_layout_follows_the_column_across_its_pages() {
        let dir = tempfile::tempdir().unwrap();
        let rows = 700;
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("sliced.idx")).unwrap();
        write_in_slices(&store, &path, &partition(rows, WIDE), 50).await;
        let file = PartitionFile::open(&scan_scheduler(&store), &path, None, None)
            .await
            .unwrap();
        let layout = VectorLayout::of(file.reader(), VECTOR_COLUMN).unwrap();
        assert!(
            layout.pages.len() > 1,
            "the fixture fits in one page, so it tests no page arithmetic"
        );
        assert_eq!(
            layout.pages.iter().map(|page| page.num_rows).sum::<u64>(),
            rows as u64
        );

        // Both sides of every boundary, which is where an offset that ignored
        // the page would first be wrong.
        let edges = layout
            .pages
            .iter()
            .flat_map(|page| [page.first_row as u32, page.end() as u32 - 1])
            .collect::<Vec<_>>();
        let stats = IoStats::new();
        let read = file
            .read_vectors(&edges, WIDE, &stats)
            .await
            .unwrap()
            .expect("a file this crate wrote must be addressable");
        for (position, &row) in edges.iter().enumerate() {
            assert_eq!(
                vector_at(&read, position),
                expected(row, WIDE),
                "row {row}, the {position}th asked for, came back as another row"
            );
        }
    }

    /// The oracle: raw reading has to hand back what the decoder hands back,
    /// value for value, across a page boundary as well as inside a page.
    #[tokio::test]
    async fn a_raw_read_equals_what_the_decoder_returns() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("sliced.idx")).unwrap();
        write_in_slices(&store, &path, &partition(700, WIDE), 50).await;
        let file = PartitionFile::open(&scan_scheduler(&store), &path, None, None)
            .await
            .unwrap();
        assert!(
            VectorLayout::of(file.reader(), VECTOR_COLUMN)
                .unwrap()
                .pages
                .len()
                > 1,
            "the oracle runs on one page, so it never compares a page boundary"
        );
        let wanted = [0, 1, 511, 512, 513, 600, 699];
        let stats = IoStats::new();

        let raw = file
            .read_vectors(&wanted, WIDE, &stats)
            .await
            .unwrap()
            .unwrap();
        let reader = file.project(&[VECTOR_COLUMN]).await.unwrap();
        let batch = read_scattered(&reader, &wanted).await.unwrap();
        let decoded = vectors_of(&batch, WIDE).unwrap();

        assert_eq!(raw.len(), decoded.len());
        for position in 0..raw.len() {
            assert_eq!(
                vector_at(&raw, position),
                vector_at(&decoded, position),
                "the {position}th vector differs between the raw read and the decoder"
            );
        }
    }

    /// The reading is the same reading, and the asking is not.
    ///
    /// `bytes_read` and `iops` have to match to the byte and to the read: the
    /// ranges handed to the scheduler are the ones its own full-zip scheduler
    /// would have built, so it coalesces, splits and counts them identically.
    /// That is what keeps a run comparable with the runs taken before this
    /// existed - those two columns are how one is checked against another.
    ///
    /// `requests` counts calls to the scheduler rather than reads. A re-score
    /// knows all of its rows at once and asks once; the decoder asked once per
    /// page it had scheduled until Lance began handing a scheduling step's reads
    /// over as one request (#9473), and now asks once here too. So the raw read
    /// is pinned at one request and at no more than the decoder's: the day it
    /// asks twice, something started asking twice.
    #[tokio::test]
    async fn a_raw_read_costs_what_the_decoder_costs() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("sliced.idx")).unwrap();
        write_in_slices(&store, &path, &partition(700, WIDE), 50).await;
        let scheduler = scan_scheduler(&store);
        let wanted = [0, 1, 511, 512, 513, 600, 699];

        let raw_stats = IoStats::new();
        PartitionFile::open(&scheduler, &path, None, None)
            .await
            .unwrap()
            .read_vectors(&wanted, WIDE, &raw_stats)
            .await
            .unwrap()
            .unwrap();

        // A file of its own, so the footer is charged to neither and the two
        // numbers are the re-score alone.
        let decoded_stats = IoStats::new();
        let file = PartitionFile::open(&scheduler, &path, None, None)
            .await
            .unwrap();
        let reader = file
            .project_into(&[VECTOR_COLUMN], &decoded_stats)
            .await
            .unwrap();
        read_scattered(&reader, &wanted).await.unwrap();

        let raw = raw_stats.snapshot();
        let decoded = decoded_stats.snapshot();
        assert_eq!(
            (raw.bytes_read, raw.iops),
            (decoded.bytes_read, decoded.iops),
            "raw read {raw:?} where the decoder read {decoded:?}"
        );
        assert_eq!(
            raw.requests, 1,
            "a re-score knows all of its rows at once, and asked the scheduler {} times",
            raw.requests
        );
        assert!(
            raw.requests <= decoded.requests,
            "raw asked the scheduler {} times where the decoder asked {}",
            raw.requests,
            decoded.requests
        );
        assert_eq!(
            raw.bytes_read,
            wanted.len() as u64 * u64::from(WIDE) * 4,
            "the re-score read something other than its candidates"
        );
    }

    /// The two rules [`coalesced`] copies, each exercised on its own.
    ///
    /// A re-score's own ranges never reach the second one - a vector is
    /// kilobytes and `max_iop_size` is megabytes - so it is pinned here rather
    /// than left to a fixture that would have to be enormous to reach it.
    #[test]
    fn coalescing_follows_the_schedulers_two_rules() {
        // Within a block of each other: one read, ending at the farther end.
        assert_eq!(coalesced(&[0..10, 12..20], 4, 100), vec![0..20]);
        // Exactly a block apart still merges, which is where the scheduler's
        // comparison sits.
        assert_eq!(coalesced(&[0..10, 14..20], 4, 100), vec![0..20]);
        // One byte further and it does not.
        assert_eq!(coalesced(&[0..10, 15..20], 4, 100), vec![0..10, 15..20]);
        // Longer than one read: equal pieces, with the remainder on the last.
        let long = 0..10u64;
        assert_eq!(
            coalesced(std::slice::from_ref(&long), 0, 4),
            vec![0..3, 3..6, 6..10]
        );
        // A range that ends inside one already taken must not shorten it. A
        // re-score never asks for nested ranges; the scheduler's rule allows
        // them, and copying the rule means copying this.
        assert_eq!(coalesced(&[0..10, 2..4], 0, 100), vec![0..10]);
        // Both at once: the two adjacent ranges merge into eight bytes, which
        // three-byte reads cover in three pieces rather than four, because the
        // last piece takes the remainder.
        assert_eq!(
            coalesced(&[0..4, 4..8, 100..104], 0, 3),
            vec![0..2, 2..4, 4..8, 100..102, 102..104]
        );
    }

    /// Every wanted range comes back out of the reads that covered it.
    ///
    /// The case worth writing down is a range that starts exactly where the
    /// previous read ended, which is what a split produces and what an
    /// off-by-one in the cursor turns into a refusal.
    #[test]
    fn slices_come_out_of_the_read_that_covers_them() {
        let blocks = vec![vec![0u8, 1, 2, 3], vec![4, 5, 6, 7]];
        let reads = vec![0..4, 4..8];

        assert_eq!(
            slices(&[0..2, 4..6, 6..8], &reads, &blocks).unwrap(),
            vec![
                Cow::Borrowed(&[0u8, 1][..]),
                Cow::Borrowed(&[4, 5][..]),
                Cow::Borrowed(&[6, 7][..])
            ]
        );
        let past = 8..10u64;
        let missed = slices(std::slice::from_ref(&past), &reads, &blocks)
            .unwrap_err()
            .to_string();
        assert!(missed.contains("which none of them holds"), "{missed}");
    }

    /// A vector the split cut in half is handed back whole.
    ///
    /// `coalesced` splits a merged run at `max_iop_size` and knows nothing about
    /// where a vector ends, so a dense enough candidate list gets one cut down
    /// the middle. The scheduler stitches such a value back together out of the
    /// pieces it read; this is the same thing, and without it the local path
    /// would refuse a list the scheduled path answers.
    ///
    /// Composed rather than asserted on `slices` alone, because the two halves
    /// have to agree about where the cut is.
    #[test]
    fn a_vector_cut_by_the_split_is_put_back_together() {
        let wanted = vec![0..4, 4..8, 8..12];
        let reads = coalesced(&wanted, 0, 6);
        assert_eq!(
            reads,
            vec![0..6, 6..12],
            "the fixture was not cut in the middle of a value, so it tests nothing"
        );
        let blocks = vec![vec![0u8, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]];
        assert_eq!(
            slices(&wanted, &reads, &blocks).unwrap(),
            vec![
                Cow::Borrowed(&[0u8, 1, 2, 3][..]),
                Cow::Owned(vec![4u8, 5, 6, 7]),
                Cow::Borrowed(&[8u8, 9, 10, 11][..])
            ]
        );
    }

    /// Reading off the disk and reading through the scheduler are the same read.
    ///
    /// Same values, and the same three counters - which is the point of
    /// [`coalesced`]: the hand-written reads have to be the reads the scheduler
    /// would have made, or a run taken this way cannot be held against one taken
    /// the other way.
    ///
    /// The row list is deliberately part adjacent and part scattered, and the
    /// test checks that the adjacent part really did merge before it claims
    /// anything: a list that merged nothing would compare two paths that both
    /// did the trivial thing.
    #[tokio::test]
    async fn a_local_read_equals_the_scheduled_one() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("sliced.idx")).unwrap();
        write_in_slices(&store, &path, &partition(700, WIDE), 50).await;
        let scheduler = scan_scheduler(&store);
        let wanted = [0, 1, 2, 511, 512, 513, 699];

        let scheduled_stats = IoStats::new();
        let scheduled = PartitionFile::open(&scheduler, &path, None, None)
            .await
            .unwrap()
            .read_vectors(&wanted, WIDE, &scheduled_stats)
            .await
            .unwrap()
            .unwrap();

        let local_stats = IoStats::new();
        let local = PartitionFile::open(&scheduler, &path, None, None)
            .await
            .unwrap()
            .with_local_reads(&store, &Arc::new(DirectReads::default()))
            .read_vectors(&wanted, WIDE, &local_stats)
            .await
            .unwrap()
            .unwrap();

        let scheduled_cost = scheduled_stats.snapshot();
        let local_cost = local_stats.snapshot();
        assert!(
            scheduled_cost.iops < wanted.len() as u64,
            "no two of {wanted:?} were coalesced, so this compares nothing about coalescing"
        );
        assert_eq!(
            (local_cost.bytes_read, local_cost.iops, local_cost.requests),
            (
                scheduled_cost.bytes_read,
                scheduled_cost.iops,
                scheduled_cost.requests
            ),
            "reading locally cost {local_cost:?} where the scheduler cost {scheduled_cost:?}"
        );
        for position in 0..wanted.len() {
            assert_eq!(
                vector_at(&local, position),
                vector_at(&scheduled, position),
                "the {position}th vector differs between a local read and a scheduled one"
            );
        }
    }

    /// A remote store keeps the scheduler, whatever the path happens to name on
    /// this machine.
    #[tokio::test]
    async fn a_file_is_only_read_locally_when_the_store_is_local() {
        let dir = tempfile::tempdir().unwrap();
        let file = opened(&dir, 8, NARROW).await;
        assert!(
            !file.reads_locally(),
            "a file opened without asking for local reads took the local path anyway"
        );
        let store = Arc::new(ObjectStore::memory());
        assert!(
            !opened(&dir, 8, NARROW)
                .await
                .with_local_reads(&store, &Arc::new(DirectReads::default()))
                .reads_locally(),
            "a file on a memory store took the local path"
        );
        let local = Arc::new(ObjectStore::local());
        assert!(
            opened(&dir, 8, NARROW)
                .await
                .with_local_reads(&local, &Arc::new(DirectReads::default()))
                .reads_locally(),
            "a file on local storage did not take the local path, so every local \
             read below is a scheduled one"
        );
    }

    /// Out of order is refused where it happens, not two layers down.
    ///
    /// The scheduler rejects unsorted ranges and the coalescing quietly swallows
    /// them, so without this the diagnosis a caller gets is either about ranges
    /// it never built or about a byte span it never asked for.
    #[tokio::test]
    async fn descending_vertices_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let file = opened(&dir, 8, NARROW).await;
        let layout = VectorLayout::of(file.reader(), VECTOR_COLUMN).unwrap();
        for out_of_order in [vec![3u32, 1], vec![1, 5, 5], vec![0, 7, 2]] {
            let error = layout.ranges(&out_of_order).unwrap_err().to_string();
            assert!(
                error.contains("strictly ascending"),
                "{out_of_order:?} was accepted or misdiagnosed: {error}"
            );
        }
    }

    #[tokio::test]
    async fn a_vertex_outside_the_partition_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let file = opened(&dir, 8, NARROW).await;
        let layout = VectorLayout::of(file.reader(), VECTOR_COLUMN).unwrap();
        let error = layout.ranges(&[8]).unwrap_err().to_string();
        assert!(
            error.contains("not a row of a file holding 8 rows"),
            "{error}"
        );
    }

    /// The same partition written with the encoding hints stripped, which is how
    /// Lance stores a column small enough to prefer mini-block. It is the arm
    /// that proves the checks above can fail.
    async fn write_without_hint(store: &ObjectStore, path: &Path, partition: &Partition) {
        let hinted = partition.to_batch(None, VectorSource::Index).unwrap();
        let fields = hinted
            .schema()
            .fields()
            .iter()
            .map(|field| Arc::new(field.as_ref().clone().with_metadata(HashMap::new())))
            .collect::<Fields>();
        let schema = Arc::new(ArrowSchema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), hinted.columns().to_vec()).unwrap();
        let mut writer = create_writer(
            SEGMENT_FILE_VERSION,
            store.create(path).await.unwrap(),
            lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// The same partition with one vector missing, written through a nullable
    /// schema.
    ///
    /// One null puts a control word in front of every value of the column, so
    /// the values stop being `stride` apart. Lance drops a validity bitmap that
    /// holds no nulls, so the hole has to be real for this to be the case it is
    /// meant to be.
    async fn write_with_a_hole(store: &ObjectStore, path: &Path, partition: &Partition) {
        let hinted = partition.to_batch(None, VectorSource::Index).unwrap();
        let vectors = hinted[VECTOR_COLUMN].as_fixed_size_list();
        let width = vectors.value_length() as usize;
        let slots = vectors
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let holed = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..partition.len())
                .map(|vertex| {
                    (vertex != partition.len() / 2).then(|| {
                        slots[vertex * width..(vertex + 1) * width]
                            .iter()
                            .map(|value| Some(*value))
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>(),
            width as i32,
        );
        let fields = hinted
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let field = field.as_ref().clone().with_nullable(true);
                if field.name() == VECTOR_COLUMN {
                    Arc::new(field.with_data_type(holed.data_type().clone()))
                } else {
                    Arc::new(field)
                }
            })
            .collect::<Fields>();
        let schema = Arc::new(ArrowSchema::new(fields));
        let columns = hinted
            .columns()
            .iter()
            .enumerate()
            .map(|(index, column)| {
                if hinted.schema().field(index).name() == VECTOR_COLUMN {
                    Arc::new(holed.clone()) as ArrayRef
                } else {
                    column.clone()
                }
            })
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let mut writer = create_writer(
            SEGMENT_FILE_VERSION,
            store.create(path).await.unwrap(),
            lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// The same partition whose vectors are written as half as many `f64`s.
    ///
    /// Byte for byte it is the same column: `PAIRED` floats and `PAIRED / 2`
    /// doubles are both `PAIRED * 4` bytes a row, both flat, both full-zip.
    /// Nothing about the layout descriptors distinguishes them.
    async fn write_as_doubles(store: &ObjectStore, path: &Path, partition: &Partition) {
        let hinted = partition.to_batch(None, VectorSource::Index).unwrap();
        let slots = hinted[VECTOR_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f64::from(pair[0]) + f64::from(pair[1]))
            .collect::<Vec<_>>();
        let doubles =
            FixedSizeListArray::try_new_from_values(Float64Array::from(slots), PAIRED as i32 / 2)
                .unwrap();
        let fields = hinted
            .schema()
            .fields()
            .iter()
            .map(|field| {
                if field.name() == VECTOR_COLUMN {
                    Arc::new(
                        field
                            .as_ref()
                            .clone()
                            .with_data_type(doubles.data_type().clone()),
                    )
                } else {
                    Arc::new(field.as_ref().clone())
                }
            })
            .collect::<Fields>();
        let schema = Arc::new(ArrowSchema::new(fields));
        let columns = hinted
            .columns()
            .iter()
            .enumerate()
            .map(|(index, column)| {
                if hinted.schema().field(index).name() == VECTOR_COLUMN {
                    Arc::new(doubles.clone()) as ArrayRef
                } else {
                    column.clone()
                }
            })
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let mut writer = create_writer(
            SEGMENT_FILE_VERSION,
            store.create(path).await.unwrap(),
            lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();
    }

    /// A column of the right width and the wrong type is not addressable.
    ///
    /// Everything this module reasons about is bytes, and bytes cannot tell
    /// `FixedSizeList<Float32, d>` from `FixedSizeList<Float64, d / 2>`. Reading
    /// the second as the first returns plausible numbers and no error, which is
    /// the one failure this module must not have: on the lazy path nothing else
    /// looks at the column's type, because nothing there builds a `Partition`.
    #[tokio::test]
    async fn a_column_of_the_right_width_and_the_wrong_type_gets_no_layout() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());

        // The control: at this width and this type there is a layout, so what
        // the arm below proves is the type check and not the width.
        let float = Path::from_absolute_path(dir.path().join("f32.idx")).unwrap();
        write_partition(
            &store,
            &float,
            &partition(8, PAIRED),
            None,
            VectorSource::Index,
        )
        .await
        .unwrap();
        let float = PartitionFile::open(&scan_scheduler(&store), &float, None, None)
            .await
            .unwrap();
        assert_eq!(
            VectorLayout::of(float.reader(), VECTOR_COLUMN)
                .expect("a float column of this width must be addressable")
                .stride(),
            u64::from(PAIRED) * 4
        );

        let double = Path::from_absolute_path(dir.path().join("f64.idx")).unwrap();
        write_as_doubles(&store, &double, &partition(8, PAIRED)).await;
        let double = PartitionFile::open(&scan_scheduler(&store), &double, None, None)
            .await
            .unwrap();
        assert!(
            VectorLayout::of(double.reader(), VECTOR_COLUMN).is_none(),
            "a column of doubles was taken for one of floats, so a re-score would \
             have returned numbers read out of the wrong halves of them"
        );
        let stats = IoStats::new();
        assert!(
            double
                .read_vectors(&[0, 7], PAIRED, &stats)
                .await
                .unwrap()
                .is_none(),
            "a re-score would have read a column of doubles by offset"
        );
    }

    /// A column with a hole in it is not addressable, and the checks that say so
    /// are the only thing between a re-score and silently reading every vector
    /// past the hole at the wrong offset.
    ///
    /// The file trips several of them at once, which is why removing any one
    /// leaves this passing: Lance describes the holed column as `bits_def: 1`,
    /// `has_validity: true`, a declared value width of 104 bits where three
    /// floats are 96, and a buffer of 112 bytes where eight values of 13 are
    /// 104. The arithmetic one is the last to give way, and it is the one that
    /// would still hold if a future encoding described a control word some other
    /// way.
    #[tokio::test]
    async fn a_column_with_a_null_gets_no_layout() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("holed.idx")).unwrap();
        write_with_a_hole(&store, &path, &partition(8, NARROW)).await;
        let file = PartitionFile::open(&scan_scheduler(&store), &path, None, None)
            .await
            .unwrap();
        assert!(
            VectorLayout::of(file.reader(), VECTOR_COLUMN).is_none(),
            "a column carrying a validity bitmap was taken for a flat one"
        );
    }

    /// A file whose vector column is not laid out the way this module needs is
    /// not read by it.
    #[tokio::test]
    async fn a_column_that_is_not_full_zip_gets_no_layout() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectStore::local());
        let path = Path::from_absolute_path(dir.path().join("mini.idx")).unwrap();
        write_without_hint(&store, &path, &partition(8, NARROW)).await;
        let file = PartitionFile::open(&scan_scheduler(&store), &path, None, None)
            .await
            .unwrap();
        assert!(
            VectorLayout::of(file.reader(), VECTOR_COLUMN).is_none(),
            "a mini-block column was taken for an addressable one"
        );
        let stats = IoStats::new();
        assert!(
            file.read_vectors(&[0, 7], NARROW, &stats)
                .await
                .unwrap()
                .is_none(),
            "a re-score would have read a mini-block column by offset"
        );
    }
}
