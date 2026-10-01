// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Re-scoring out of the dataset's own data files.
//!
//! A lazy walk steers by codes, and the one thing it wants full vectors for is
//! the exact distance of each candidate it ends with, read in one batch after
//! the walk. The dataset holds those vectors already. A data file Lance wrote in
//! the full-zip layout keeps row `r` of a fixed-width column at one computable
//! offset - the arithmetic [`crate::raw::VectorLayout`] reads out of a partition
//! file - so the batch can come from there, by the same positional reads.
//!
//! Not every data file is laid out that way. Lance stores values narrower than
//! 256 bytes as mini-blocks; a Lance 2.0 file has another grammar; a compressed
//! column, or one holding nulls, carries more than its values; an overlay
//! replaces a column's values from a file of its own; a file under another base
//! path cannot be resolved from here; a count of rows the manifest does not
//! vouch for leaves nothing to hold the column's pages to. Those rows are read
//! through Lance's take ([`Dataset::take_builder`], handed the row addresses as
//! the row ids they are
//! in a dataset without stable row ids, the only kind this crate indexes)
//! instead, which reads every one of them correctly and costs milliseconds
//! where the offset read costs microseconds, and [`DatasetVectors::lance_rows`]
//! counts them so that a measurement can check the fast path served it. Lance
//! reads them through its own store, so no count of this crate's bytes
//! includes them.
//!
//! Lance's take returns live rows only, and an insertion walks through deleted
//! vertices as well. A deleted row's bytes stay in its data file, so the offset
//! read still reaches them, and where it cannot, a take from the fragment as it
//! would be without its deletion file does. See [`DatasetVectors::fetch_deleted`].

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use futures::{StreamExt, TryStreamExt, stream};
use lance::Dataset;
use lance::dataset::fragment::FileFragment;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR, Result};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::reader_projection_from_field_ids;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{IoStats, ScanScheduler};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;
use lance_table::format::Fragment;
use object_store::path::Path;
use roaring::RoaringBitmap;

use crate::data_file::{DataColumn, DataFile, DataLayout, DataLayoutKey, read_layout};
use crate::io::{DirectReads, OPEN_FILES, OpenFiles};
use crate::query::descends_from;

/// How [`DatasetVectors::fetch`] opens a data file: on the terms the index
/// opens a query's partition files - held when the index has a cache, read in
/// place unless it was opened for a pass that rewrites partitions - so that
/// one is counted the way the other is.
pub(crate) struct FileAccess<'a> {
    pub(crate) scheduler: &'a Arc<ScanScheduler>,
    /// Where the layout of a data file's column is kept, and whether its file
    /// is held between reads: `None` for an index given no cache, which opens
    /// a file the offset read serves and reads its layout for every read
    /// rather than holding either - the rule its partition files follow. A file
    /// found not to be served is remembered all the same (`unreadable`).
    pub(crate) cache: Option<&'a LanceCache>,
    /// Whether a held data file reads off a descriptor of its own, as a held
    /// partition file does, rather than through the scheduler. Off for a pass
    /// that rewrites partitions, which reads thousands of scattered rows at a
    /// time, which the scheduler reads several at once where one blocking
    /// thread would read them in turn.
    pub(crate) local_reads: bool,
    pub(crate) store: &'a ObjectStore,
    pub(crate) direct: &'a Arc<DirectReads>,
}

/// How a read reaches, through Lance, the rows of a fragment no offset reaches.
#[derive(Debug, Clone, Copy)]
enum ThroughLance {
    /// Lance's take, which returns live rows only.
    Take,
    /// Lance's take from the fragment without its deletion file, which
    /// returns deleted rows as their files hold them.
    Physical,
}

/// Where the vectors of one covered fragment are read from.
#[derive(Debug)]
enum Source {
    /// A data file the offset read may serve. Whether it can is up to the
    /// column's pages, which are checked the first time a read asks, and a
    /// column that fails that check is read through Lance instead.
    File(FileSource),
    /// Through Lance, because nothing in the manifest leaves the offset read
    /// possible.
    Lance,
}

/// The data file of a covered fragment that the offset read may serve.
#[derive(Debug)]
struct FileSource {
    path: Path,
    /// As the manifest records it, and as an open that had to ask the store
    /// found it out when the manifest does not.
    size: CachedFileSize,
    column: DataColumn,
    /// The fragment's rows as the manifest records them, which the column's
    /// pages must hold between them.
    rows: u64,
    /// Set once the file's footer has shown that no offset read can serve the
    /// column, after which the file is never opened again and its rows go
    /// straight to Lance - for an index given no cache as well, which
    /// otherwise opens a file and reads its layout for every read. What it
    /// keeps is where a read goes, not anything a read would return: the rows
    /// themselves are read through Lance every time. Never set by a read that
    /// failed, which the next read tries again.
    unreadable: AtomicBool,
}

/// The dataset's own copy of the indexed vectors, as of the version an index
/// was opened on.
#[derive(Debug)]
pub(crate) struct DatasetVectors {
    /// Pinned: the addresses a query re-scores are this version's, and so are
    /// the data files below, which a compaction and a cleanup can remove from
    /// under a later one.
    dataset: Arc<Dataset>,
    field_id: i32,
    dimension: u32,
    /// Cosine is stored as L2 over unit vectors, so what is read here goes
    /// through the normalisation the build put the partition's copy through -
    /// the same function, row by row, which is what makes the two copies equal
    /// to the last bit.
    normalize: bool,
    /// By fragment id, for the fragments the index covers.
    fragments: HashMap<u32, Source>,
    /// Why nothing can be read from the dataset at all, when that is so. Kept
    /// rather than raised: an index that re-scores out of its own partitions
    /// reads the column only for a query that asks it to, which is refused, and
    /// must not fail to open over a column it does not read. One that keeps no
    /// copy of its own asks on open.
    unavailable: Option<String>,
    /// The data files this index has open, apart from its partition files so
    /// that one kind does not evict the other.
    files: OpenFiles<DataFile>,
    /// Rows read through Lance's take since the index was opened.
    lance_rows: AtomicU64,
}

impl DatasetVectors {
    /// Where every fragment in `covered` keeps the field the index row names in
    /// `fields`.
    ///
    /// Never fails. Whatever the manifest leaves unreadable by offset is marked
    /// to be read through Lance, and a field that is gone or has changed type is
    /// reported by [`Self::unavailable`] and by every read.
    pub(crate) fn of(
        dataset: &Dataset,
        fields: &[i32],
        dimension: u32,
        distance_type: DistanceType,
        covered: &RoaringBitmap,
    ) -> Self {
        let field_id = fields.first().copied().unwrap_or(-1);
        let mut vectors = Self {
            dataset: Arc::new(dataset.clone()),
            field_id,
            dimension,
            normalize: distance_type == DistanceType::Cosine,
            fragments: HashMap::new(),
            unavailable: None,
            files: OpenFiles::new(OPEN_FILES),
            lance_rows: AtomicU64::new(0),
        };
        if fields.len() != 1 {
            vectors.unavailable = Some(format!(
                "the index row names fields {fields:?} rather than the one vector column"
            ));
            return vectors;
        }
        let schema = dataset.schema();
        let Some(field) = schema.field_by_id(field_id) else {
            vectors.unavailable = Some(format!(
                "the indexed field {field_id} is no longer in the dataset schema"
            ));
            return vectors;
        };
        match field.data_type() {
            DataType::FixedSizeList(item, width)
                if item.data_type() == &DataType::Float32
                    && u32::try_from(width).ok() == Some(dimension) => {}
            other => {
                vectors.unavailable = Some(format!(
                    "column '{}' is {other}, not the FixedSizeList<Float32, {dimension}> the \
                     index was built over",
                    field.name
                ));
                return vectors;
            }
        }
        let projected = schema.project_by_ids(&[field_id], true);
        for fragment in dataset.get_fragments() {
            let fragment_id = fragment.id() as u32;
            if covered.contains(fragment_id) {
                let source = source(
                    dataset,
                    fragment.metadata(),
                    &projected,
                    field_id,
                    dimension,
                );
                vectors.fragments.insert(fragment_id, source);
            }
        }
        vectors
    }

    /// The vectors at `rows` - addresses the dataset holds them under now, of
    /// fragments the index covers - in the order asked for.
    ///
    /// Rows are read in one read a fragment, the fragments' reads at once, in
    /// ascending order within each, because that is the order a data file is
    /// read in, and a row asked for twice is read once. When that order is
    /// already the order asked for - one fragment, ascending, which is how a
    /// partition's candidates come off a dataset of one data file - what was
    /// read is the answer as it stands.
    pub(crate) async fn fetch(
        &self,
        rows: &[u64],
        stats: &IoStats,
        access: &FileAccess<'_>,
    ) -> Result<FixedSizeListArray> {
        self.read(rows, stats, access, ThroughLance::Take).await
    }

    /// [`Self::fetch`] for rows Lance's take does not return, which is deleted
    /// ones, at the addresses they were stored under.
    ///
    /// Deleting a row leaves its bytes in the data file, where an offset still
    /// finds them. Every take Lance offers - by address, or by offset within a
    /// fragment - applies the fragment's deletion vector first, so a fragment
    /// no offset reaches is read by a take from the fragment as it would be
    /// without its deletion file, which has no deletion vector to apply.
    pub(crate) async fn fetch_deleted(
        &self,
        rows: &[u64],
        stats: &IoStats,
        access: &FileAccess<'_>,
    ) -> Result<FixedSizeListArray> {
        self.read(rows, stats, access, ThroughLance::Physical).await
    }

    async fn read(
        &self,
        rows: &[u64],
        stats: &IoStats,
        access: &FileAccess<'_>,
        through: ThroughLance,
    ) -> Result<FixedSizeListArray> {
        if let Some(reason) = &self.unavailable {
            return Err(Error::invalid_input(format!(
                "Vamana cannot read vectors from the dataset: {reason}"
            )));
        }
        let plan = plan(rows);
        let mut read = read_runs(
            &plan,
            access.store.io_parallelism(),
            |fragment_id, offsets| self.read_run(fragment_id, offsets, stats, access, through),
        )
        .await?;

        let vectors = match read.pop() {
            Some(vectors) if plan.in_order => vectors,
            last => {
                read.extend(last);
                let flats = read
                    .iter()
                    .map(|vectors| {
                        vectors
                            .values()
                            .as_primitive::<Float32Type>()
                            .values()
                            .as_ref()
                    })
                    .collect::<Vec<_>>();
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(gather(&plan, &flats, self.dimension as usize)),
                    self.dimension as i32,
                )?
            }
        };
        // Its buffer is this call's alone either way - built just above, or
        // decoded for it - so this normalises in place, the arm the build took
        // over the whole column.
        Ok(if self.normalize {
            normalize_fsl_owned(vectors)?
        } else {
            vectors
        })
    }

    /// The vectors at `offsets` of `fragment_id`, in that order: by offset
    /// where the fragment's data file allows it, through Lance otherwise.
    async fn read_run(
        &self,
        fragment_id: u32,
        offsets: &[u32],
        stats: &IoStats,
        access: &FileAccess<'_>,
        through: ThroughLance,
    ) -> Result<FixedSizeListArray> {
        let by_offset = match self.fragments.get(&fragment_id) {
            Some(Source::File(source)) => match self.open(source, stats, access).await? {
                Some(file) => Some(file.read_vectors(offsets, self.dimension, stats).await?),
                None => None,
            },
            Some(Source::Lance) | None => None,
        };
        let vectors = match (by_offset, through) {
            (Some(vectors), _) => vectors,
            (None, ThroughLance::Take) => self.take(fragment_id, offsets).await?,
            (None, ThroughLance::Physical) => self.physical(fragment_id, offsets).await?,
        };
        checked(vectors, fragment_id, offsets, self.dimension)
    }

    /// How many of the covered fragments the manifest alone sends through
    /// Lance: a data file older than 2.1, another base path, an overlay on the
    /// indexed field, a count of rows the manifest does not vouch for. A column
    /// whose pages turn out not to allow an offset read is found out only when
    /// its file is first opened, and is not counted here.
    pub(crate) fn fragments_through_lance(&self) -> usize {
        self.fragments
            .values()
            .filter(|source| matches!(source, Source::Lance))
            .count()
    }

    /// Why no vector can be read from the dataset, or `None` when they can.
    pub(crate) fn unavailable(&self) -> Option<&str> {
        self.unavailable.as_deref()
    }

    /// Rows read through Lance - its take of live rows, or of deleted ones -
    /// since the index was opened: the ones the offset read could not serve.
    pub(crate) fn lance_rows(&self) -> u64 {
        self.lance_rows.load(Ordering::Relaxed)
    }

    /// The data file of `source`, open to read its column by offset, or `None`
    /// when no offset read can serve the column - found out from the file's
    /// footer the first time and remembered after, whether or not the index
    /// was given a cache.
    ///
    /// Held, by an index given a cache, rather than rebound, since all it is
    /// asked for is [`DataFile::read_vectors`], which counts into the `stats`
    /// it is handed whatever the file was opened with; the layout of its column
    /// is kept in that cache, the footer's other columns nowhere. Opened for
    /// this read alone, its layout read again, by an index that was not.
    async fn open(
        &self,
        source: &FileSource,
        stats: &IoStats,
        access: &FileAccess<'_>,
    ) -> Result<Option<Arc<DataFile>>> {
        if source.unreadable.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if access.cache.is_some()
            && let Some(open) = self.files.held(&source.path)
        {
            return Ok(Some(open));
        }
        let file = access
            .scheduler
            .open_file(&source.path, &source.size)
            .await?
            .with_io_stats(stats.recorder());
        let layout = match access.cache {
            Some(cache) => {
                let key = DataLayoutKey {
                    path: &source.path,
                    column: source.column,
                    rows: source.rows,
                };
                let read = cache
                    .get_or_insert_with_key(key, || async {
                        let layout = read_layout(&file, source.column, source.rows).await?;
                        Ok(DataLayout(layout.map(Arc::new)))
                    })
                    .await?;
                read.0.clone()
            }
            None => read_layout(&file, source.column, source.rows)
                .await?
                .map(Arc::new),
        };
        let Some(layout) = layout else {
            source.unreadable.store(true, Ordering::Relaxed);
            return Ok(None);
        };
        let opened = DataFile::new(source.path.clone(), file, layout);
        Ok(Some(match access.cache {
            Some(_) if access.local_reads => self.files.hold(
                &source.path,
                opened.with_local_reads(access.store, access.direct),
            ),
            Some(_) => self.files.hold(&source.path, opened),
            None => Arc::new(opened),
        }))
    }

    /// The vectors at `offsets` of `fragment_id`, in that order, read through
    /// Lance.
    ///
    /// Matched to what was asked by the address each row comes back with rather
    /// than by position. Lance returns them in the order asked and fails
    /// outright when one of them is deleted, so today the match is the
    /// identity; its documentation promises neither, and a change to either
    /// would come out here as an error rather than as a neighbour's vector.
    async fn take(&self, fragment_id: u32, offsets: &[u32]) -> Result<FixedSizeListArray> {
        let addresses = offsets
            .iter()
            .map(|&offset| u64::from(RowAddress::new_from_parts(fragment_id, offset)))
            .collect::<Vec<_>>();
        let projection = self.dataset.schema().project_by_ids(&[self.field_id], true);
        let batch = self
            .dataset
            .take_builder(&addresses, projection)?
            .with_row_address(true)
            .execute()
            .await?;

        let returned = batch
            .column_by_name(ROW_ADDR)
            .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| {
                Error::internal(format!("Lance's take returned no {ROW_ADDR} column"))
            })?;
        let vectors = batch.column(0).as_fixed_size_list_opt().ok_or_else(|| {
            Error::internal(format!(
                "Lance's take returned {} for the vector column",
                batch.column(0).data_type()
            ))
        })?;
        let at = returned
            .values()
            .iter()
            .enumerate()
            .map(|(index, &address)| (address, index as u32))
            .collect::<HashMap<_, _>>();
        let indices = addresses
            .iter()
            .map(|address| {
                at.get(address).copied().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "the dataset has no row {} in fragment {fragment_id} to read a vector from",
                        RowAddress::from(*address).row_offset()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let taken =
            arrow_select::take::take(vectors, &arrow_array::UInt32Array::from(indices), None)?;
        self.lance_rows
            .fetch_add(addresses.len() as u64, Ordering::Relaxed);
        Ok(taken.as_fixed_size_list().clone())
    }

    /// The vectors at `offsets` of `fragment_id`, deleted rows included, read
    /// through Lance: every row as the fragment's files hold it, overlays
    /// applied, which is what the build read.
    ///
    /// All of them in one take, from the fragment's own metadata without its
    /// deletion file - as Lance's own physical read
    /// ([`FileFragment::read_physical_slice`]) views it - which reads by
    /// position with no deletion vector to map the positions through or drop
    /// rows by.
    /// Only an insertion comes here, and only for the deleted rows of a
    /// fragment no offset reaches; `offsets` ascend, as that take requires.
    async fn physical(&self, fragment_id: u32, offsets: &[u32]) -> Result<FixedSizeListArray> {
        let fragment = self
            .dataset
            .get_fragment(fragment_id as usize)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "the dataset has no fragment {fragment_id} to read deleted rows from"
                ))
            })?;
        let projection = self.dataset.schema().project_by_ids(&[self.field_id], true);
        let name = projection
            .fields
            .first()
            .map(|field| field.name.clone())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "the indexed field {} is no longer in the dataset schema",
                    self.field_id
                ))
            })?;
        let mut undeleted = fragment.metadata().clone();
        undeleted.deletion_file = None;
        let batch = FileFragment::new(self.dataset.clone(), undeleted)
            .take(offsets, &projection)
            .await?;
        let column = batch.column_by_name(&name).ok_or_else(|| {
            Error::internal(format!(
                "Lance's take of deleted rows returned no column '{name}'"
            ))
        })?;
        let vectors = column.as_fixed_size_list_opt().ok_or_else(|| {
            Error::internal(format!(
                "Lance's take of deleted rows returned {} for the vector column",
                column.data_type()
            ))
        })?;
        self.lance_rows
            .fetch_add(offsets.len() as u64, Ordering::Relaxed);
        Ok(vectors.clone())
    }
}

/// How to read `field_id` of `fragment`: by offset wherever the manifest leaves
/// that possible, through Lance everywhere else.
fn source(
    dataset: &Dataset,
    fragment: &Fragment,
    projected: &Schema,
    field_id: i32,
    dimension: u32,
) -> Source {
    // An overlay replaces the values of the fields it names from a file of its
    // own, so the base data file no longer holds the row's vector. The index saw
    // every overlay it is opened over - `VamanaIndex::open` refuses one that came
    // later - so the overlaid values are the ones to re-score by.
    let overlaid = fragment.overlays.iter().any(|overlay| {
        overlay.data_file.fields.iter().any(|&overlaid| {
            overlaid == field_id
                || descends_from(dataset.schema(), overlaid, field_id)
                || descends_from(dataset.schema(), field_id, overlaid)
        })
    });
    if overlaid {
        return Source::Lance;
    }
    let Some(file) = fragment
        .files
        .iter()
        .find(|file| file.fields.contains(&field_id))
    else {
        return Source::Lance;
    };
    // `Dataset::data_file_dir` resolves a base id, and it is `pub(crate)`.
    if file.base_id.is_some() {
        return Source::Lance;
    }
    let Ok(version) = file.file_version() else {
        return Source::Lance;
    };
    if !matches!(
        version,
        ConcreteFileVersion::V2_1 | ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3
    ) {
        return Source::Lance;
    }
    let columns = file
        .fields
        .iter()
        .zip(file.column_indices.iter())
        .filter_map(|(&field, &column)| {
            Some((u32::try_from(field).ok()?, u32::try_from(column).ok()?))
        })
        .collect::<BTreeMap<_, _>>();
    let Ok(projection) = reader_projection_from_field_ids(version, projected, &columns) else {
        return Source::Lance;
    };
    let [index] = projection.column_indices[..] else {
        return Source::Lance;
    };
    // What the column's pages are held to hold between them. Lance trusts the
    // manifest's count of a fragment's rows only where it knows which writer
    // recorded it, since early writers could record a wrong one; so does this.
    let (Some(rows), Some(_)) = (
        fragment.physical_rows,
        dataset.manifest().writer_version.as_ref(),
    ) else {
        return Source::Lance;
    };
    // Joined the way Lance joins it for a file with no base id.
    let path = dataset.data_dir().join(file.path.as_str());
    Source::File(FileSource {
        path,
        size: file.file_size_bytes.clone(),
        column: DataColumn {
            version,
            index,
            items: u64::from(dimension),
        },
        rows: rows as u64,
        unreadable: AtomicBool::new(false),
    })
}

/// Which rows [`DatasetVectors::fetch`] reads, and where each asked-for row is
/// among them.
#[derive(Debug, PartialEq)]
struct Plan {
    /// One per fragment asked about, ascending by fragment id, each with the
    /// distinct offsets to read from it, ascending.
    runs: Vec<(u32, Vec<u32>)>,
    /// For each row in the order asked: which run holds it, and where among
    /// that run's offsets.
    slots: Vec<(usize, usize)>,
    /// Whether the rows were asked for in exactly the order they are read: one
    /// fragment, strictly ascending, so that what is read is the answer.
    in_order: bool,
}

fn plan(rows: &[u64]) -> Plan {
    let mut keyed = rows
        .iter()
        .enumerate()
        .map(|(position, &row)| {
            let address = RowAddress::from(row);
            (address.fragment_id(), address.row_offset(), position)
        })
        .collect::<Vec<_>>();
    keyed.sort_unstable();
    let mut runs: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut slots = vec![(0, 0); rows.len()];
    for (fragment_id, offset, position) in keyed {
        if runs.last().is_none_or(|(last, _)| *last != fragment_id) {
            runs.push((fragment_id, Vec::new()));
        }
        let run = runs.len() - 1;
        let offsets = &mut runs[run].1;
        if offsets.last() != Some(&offset) {
            offsets.push(offset);
        }
        slots[position] = (run, offsets.len() - 1);
    }
    let in_order = runs.len() == 1
        && slots
            .iter()
            .enumerate()
            .all(|(position, &(_, at))| at == position)
        && runs[0].1.len() == rows.len();
    Plan {
        runs,
        slots,
        in_order,
    }
}

/// The rows of `plan` in the order they were asked for, `width` values each,
/// out of `flats`: the values read for each of its runs, back to back.
fn gather(plan: &Plan, flats: &[&[f32]], width: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(plan.slots.len() * width);
    for &(run, at) in &plan.slots {
        values.extend_from_slice(&flats[run][at * width..(at + 1) * width]);
    }
    values
}

/// Every run of `plan` read by `read_run`, up to `parallelism` of them at once,
/// in the plan's order - the order [`gather`] takes them in - or the error of
/// the first run, in that order, that failed.
///
/// A run is one fragment's share of the rows, read from its own file, and on
/// an object store every one of them is a round trip; read one after another,
/// each would wait for the ones before it for nothing. A read the page cache
/// serves without waiting is made in place, on the task polling it, so those
/// still come one after another: what runs at once is the reads that wait.
///
/// Bounded here and nowhere else. A bound shared by the reads of an index
/// would couple them: a caller that stops polling one search - racing it
/// against a timer, say - would leave it holding its share while the next
/// search waited on it for good. So an index given no cache, which opens a
/// file for every read, can have `parallelism` of them open for every read in
/// flight; one with a cache holds its files in its pool ([`OPEN_FILES`]) and
/// opens only what that has no room for. And a run that fails does not stop
/// the runs beside it that have already read: what they read is counted with
/// the rest, though the read fails.
async fn read_runs<'a, F, Fut>(
    plan: &'a Plan,
    parallelism: usize,
    read_run: F,
) -> Result<Vec<FixedSizeListArray>>
where
    F: Fn(u32, &'a [u32]) -> Fut,
    Fut: Future<Output = Result<FixedSizeListArray>>,
{
    // One fragment, which is how a partition's candidates come off a dataset
    // of one data file: nothing to read at once, and nothing to pay for.
    if let [(fragment_id, offsets)] = &plan.runs[..] {
        return Ok(vec![read_run(*fragment_id, offsets).await?]);
    }
    // Collected first rather than mapped lazily: a closure kept inside the
    // stream would have to be callable for every lifetime of the run it is
    // handed, which the queries that await this in a `Send` future cannot
    // promise the compiler. Nothing starts before it is polled either way.
    let reads = plan
        .runs
        .iter()
        .map(|(fragment_id, offsets)| read_run(*fragment_id, offsets))
        .collect::<Vec<_>>();
    stream::iter(reads)
        .buffered(parallelism.max(1))
        .try_collect()
        .await
}

/// `vectors`, read for `offsets` of `fragment_id`, if it holds exactly one
/// non-null `dimension`-wide vector of `Float32` for each of them - which is
/// what makes reading its values as one flat slice of `f32` safe.
fn checked(
    vectors: FixedSizeListArray,
    fragment_id: u32,
    offsets: &[u32],
    dimension: u32,
) -> Result<FixedSizeListArray> {
    if vectors.len() != offsets.len()
        || vectors.value_type() != DataType::Float32
        || u32::try_from(vectors.value_length()).ok() != Some(dimension)
    {
        return Err(Error::internal(format!(
            "Vamana read {} vectors of {} x {} for {} rows of fragment {fragment_id}, which are \
             {dimension} Float32 wide",
            vectors.len(),
            vectors.value_length(),
            vectors.value_type(),
            offsets.len()
        )));
    }
    if vectors.null_count() != 0 || vectors.values().null_count() != 0 {
        let row = (0..vectors.len())
            .find(|&at| vectors.is_null(at) || vectors.value(at).null_count() != 0)
            .map_or(offsets[0], |at| offsets[at]);
        return Err(Error::invalid_input(format!(
            "row {row} of fragment {fragment_id} has no {dimension}-wide vector in the dataset to \
             read"
        )));
    }
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator};
    use arrow_schema::{Field, Schema as ArrowSchema};
    use lance::dataset::WriteParams;
    use lance_file::reader::FileReader;
    use lance_file::version::LanceFileVersion;
    use tokio::sync::Notify;

    use crate::io::scan_scheduler;

    fn address(fragment_id: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(fragment_id, offset))
    }

    #[test]
    fn rows_asked_in_the_order_they_are_read_are_the_answer_as_read() {
        let plan = plan(&[address(3, 1), address(3, 4), address(3, 9)]);
        assert_eq!(
            plan,
            Plan {
                runs: vec![(3, vec![1, 4, 9])],
                slots: vec![(0, 0), (0, 1), (0, 2)],
                in_order: true,
            }
        );
    }

    #[test]
    fn rows_out_of_order_twice_or_in_two_fragments_are_gathered() {
        // Descending within one fragment.
        let descending = plan(&[address(0, 9), address(0, 4)]);
        assert_eq!(descending.runs, vec![(0, vec![4, 9])]);
        assert_eq!(descending.slots, vec![(0, 1), (0, 0)]);
        assert!(!descending.in_order);

        // One row twice: read once, answered twice.
        let twice = plan(&[address(0, 4), address(0, 4)]);
        assert_eq!(twice.runs, vec![(0, vec![4])]);
        assert_eq!(twice.slots, vec![(0, 0), (0, 0)]);
        assert!(!twice.in_order);

        // Two fragments, the later one asked first; ascending otherwise, which
        // is still not the order they are read in.
        let two = plan(&[address(7, 2), address(1, 5), address(7, 1)]);
        assert_eq!(two.runs, vec![(1, vec![5]), (7, vec![1, 2])]);
        assert_eq!(two.slots, vec![(1, 1), (0, 0), (1, 0)]);
        assert!(!two.in_order);
        let ascending = plan(&[address(1, 5), address(7, 1)]);
        assert!(
            !ascending.in_order,
            "two runs are two reads, not one answer"
        );

        // Every row lands where it was asked for, from the run that read it.
        let flats: [&[f32]; 2] = [&[10.0, 11.0], &[70.0, 71.0, 72.0, 73.0]];
        assert_eq!(
            gather(&two, &flats, 2),
            vec![72.0, 73.0, 10.0, 11.0, 70.0, 71.0]
        );
        let flat: [&[f32]; 1] = [&[4.0, 4.5]];
        assert_eq!(gather(&twice, &flat, 2), vec![4.0, 4.5, 4.0, 4.5]);
    }

    #[test]
    fn nothing_asked_for_is_nothing_read() {
        let plan = plan(&[]);
        assert!(plan.runs.is_empty() && plan.slots.is_empty() && !plan.in_order);
    }

    /// What a synthetic read of a run returns: one value a row, the fragment's
    /// id, so that a result names the run it came from.
    fn read_of(fragment_id: u32, offsets: &[u32]) -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![fragment_id as f32; offsets.len()]),
            1,
        )
        .unwrap()
    }

    fn fragments_of(read: &[FixedSizeListArray]) -> Vec<u32> {
        read.iter()
            .map(|vectors| vectors.values().as_primitive::<Float32Type>().value(0) as u32)
            .collect()
    }

    /// The runs of one read are in flight at once: the first run here finishes
    /// only after the second has, which read one after another never happens.
    /// And they come back in the plan's order, not the order they finished in.
    #[tokio::test]
    async fn the_runs_of_a_read_are_read_at_once_and_come_back_in_order() {
        let plan = plan(&[address(7, 2), address(1, 5), address(7, 1)]);
        let second_done = Notify::new();
        let read = tokio::time::timeout(
            Duration::from_secs(5),
            read_runs(&plan, 4, |fragment_id, offsets| {
                let second_done = &second_done;
                async move {
                    if fragment_id == 1 {
                        second_done.notified().await;
                    }
                    let read = read_of(fragment_id, offsets);
                    if fragment_id == 7 {
                        second_done.notify_one();
                    }
                    Ok(read)
                }
            }),
        )
        .await
        .expect("the first run waited for a second that never started")
        .unwrap();
        assert_eq!(fragments_of(&read), vec![1, 7]);
        assert_eq!(read[1].len(), 2);
    }

    /// Of two runs that fail, the error reported is the first one's in the
    /// plan, as it was when the runs were read one after another - whichever
    /// failed first.
    #[tokio::test]
    async fn the_error_of_the_first_failing_run_in_the_plan_is_the_one_reported() {
        let plan = plan(&[address(1, 5), address(7, 1)]);
        let second_failed = Notify::new();
        let error = tokio::time::timeout(
            Duration::from_secs(5),
            read_runs(&plan, 4, |fragment_id, _| {
                let second_failed = &second_failed;
                async move {
                    if fragment_id == 1 {
                        second_failed.notified().await;
                    } else {
                        second_failed.notify_one();
                    }
                    Err::<FixedSizeListArray, _>(Error::invalid_input(format!(
                        "fragment {fragment_id} failed"
                    )))
                }
            }),
        )
        .await
        .expect("the first run waited for a second that never started")
        .unwrap_err();
        assert!(error.to_string().contains("fragment 1 failed"), "{error}");
    }

    /// However many runs a read has, no more are in flight than it may take
    /// at once - and that many are.
    #[tokio::test]
    async fn no_more_runs_are_in_flight_than_a_read_may_take_at_once() {
        let plan = plan(
            &(0..6)
                .map(|fragment_id| address(fragment_id, 0))
                .collect::<Vec<_>>(),
        );
        let in_flight = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let read = read_runs(&plan, 2, |fragment_id, offsets| {
            let (in_flight, most) = (&in_flight, &most);
            async move {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                tokio::task::yield_now().await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(read_of(fragment_id, offsets))
            }
        })
        .await
        .unwrap();
        assert_eq!(fragments_of(&read), (0..6).collect::<Vec<_>>());
        assert_eq!(most.load(Ordering::SeqCst), 2);
    }

    /// Rows of one fragment asked for out of the order they are read in come
    /// back in the order asked: what was read is handed back as it stands only
    /// when the two orders are one. No caller in the crate asks that way today -
    /// a probe's candidates arrive ascending by local id - which is why this
    /// asks directly.
    #[tokio::test]
    async fn rows_of_one_fragment_asked_out_of_order_come_back_in_that_order() {
        const WIDTH: i32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(item, WIDTH),
            true,
        )]));
        // Every value of row r is r, so a value names the row it was read from.
        let values = Float32Array::from_iter_values(
            (0..16).flat_map(|row| std::iter::repeat_n(row as f32, WIDTH as usize)),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                FixedSizeListArray::try_new_from_values(values, WIDTH).unwrap(),
            )],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir.path().to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &RoaringBitmap::from_iter([0]),
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };

        let read = vectors
            .fetch(
                &[address(0, 9), address(0, 2), address(0, 5)],
                &IoStats::new(),
                &access,
            )
            .await
            .unwrap();

        let rows = (0..read.len())
            .map(|at| read.value(at).as_primitive::<Float32Type>().value(0))
            .collect::<Vec<_>>();
        assert_eq!(rows, vec![9.0, 2.0, 5.0]);
        assert_eq!(
            vectors.lance_rows(),
            0,
            "the rows went through Lance, so the offset read was never asked"
        );
    }

    /// Deleted rows of a fragment no offset reaches are read through Lance in
    /// one read however many runs they fall into: every other row here, so a
    /// read a run would be a read a row. Eight reads of the store leave room
    /// for what Lance reads to open the fragment, against the thirty-two a
    /// read a run makes.
    async fn deleted_rows_are_read_through_lance_in_one_read(version: LanceFileVersion) {
        const WIDTH: i32 = 64;
        const ROWS: u32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(item, WIDTH),
            true,
        )]));
        // Every value of row r is r, so a value names the row it was read from.
        let values = Float32Array::from_iter_values(
            (0..ROWS).flat_map(|row| std::iter::repeat_n(row as f32, WIDTH as usize)),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                FixedSizeListArray::try_new_from_values(values, WIDTH).unwrap(),
            )],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            Some(WriteParams {
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("_rowid % 2 = 1").await.unwrap();

        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &RoaringBitmap::from_iter([0]),
        );
        assert_eq!(
            vectors.fragments_through_lance(),
            1,
            "{version:?}: the fragment is read by offset, so this tests nothing"
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };

        let deleted = (1..ROWS)
            .step_by(2)
            .map(|offset| address(0, offset))
            .collect::<Vec<_>>();
        // The dataset's own store, which Lance reads the fragment through: it
        // counts a local read too, where a wrapper around it would not.
        let _ = store.io_stats_incremental();
        let read = vectors
            .fetch_deleted(&deleted, &IoStats::new(), &access)
            .await
            .unwrap();
        let reads = store.io_stats_incremental().read_iops;

        let rows = (0..read.len())
            .map(|at| read.value(at).as_primitive::<Float32Type>().value(0))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            (1..ROWS)
                .step_by(2)
                .map(|row| row as f32)
                .collect::<Vec<_>>(),
            "{version:?}"
        );
        assert_eq!(vectors.lance_rows(), deleted.len() as u64, "{version:?}");
        assert!(
            reads <= 8,
            "{version:?}: {reads} reads of the store for {} deleted rows",
            deleted.len()
        );
    }

    /// A dataset of one fragment holding `columns`, in Lance's default file
    /// format.
    async fn one_fragment(dir: &tempfile::TempDir, columns: Vec<(&str, ArrayRef)>) -> Dataset {
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let schema = batch.schema();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir.path().to_str().unwrap(),
            None,
        )
        .await
        .unwrap()
    }

    /// `rows` vectors of `width`, every value of row r being r.
    fn numbered(rows: u32, width: i32) -> ArrayRef {
        Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from_iter_values(
                    (0..rows).flat_map(|row| std::iter::repeat_n(row as f32, width as usize)),
                ),
                width,
            )
            .unwrap(),
        )
    }

    /// Found out once: the first read learns from a data file's footer that no
    /// offset reaches its column - narrow vectors, laid out in mini-blocks - and
    /// no read after it opens the file again, though the index keeps no cache.
    /// Its rows go straight to Lance, which reads them every time.
    #[tokio::test]
    async fn a_data_file_no_offset_reaches_is_found_out_once() {
        const WIDTH: i32 = 8;
        let dir = tempfile::tempdir().unwrap();
        let dataset = one_fragment(&dir, vec![("vec", numbered(1024, WIDTH))]).await;
        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &RoaringBitmap::from_iter([0]),
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };

        let first = IoStats::new();
        vectors
            .fetch(&[address(0, 3), address(0, 700)], &first, &access)
            .await
            .unwrap();
        let second = IoStats::new();
        let read = vectors
            .fetch(&[address(0, 5)], &second, &access)
            .await
            .unwrap();
        assert_eq!(read.value(0).as_primitive::<Float32Type>().value(0), 5.0);
        assert!(
            first.snapshot().requests > 0,
            "the first read never looked at the footer, so it found nothing out"
        );
        assert_eq!(
            second.snapshot().requests,
            0,
            "the second read opened the file again: {:?}",
            second.snapshot()
        );
        assert_eq!(vectors.lance_rows(), 3);
    }

    /// A data file's footer is read for the one column the index re-scores by:
    /// before three hundred others, the first read of one row costs the tail of
    /// the file, that column's metadata and the row, where the whole footer is
    /// every column's metadata.
    #[tokio::test]
    async fn a_wide_data_file_costs_the_footer_of_one_column() {
        const WIDTH: i32 = 64;
        const ROWS: u32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let names = (0..300)
            .map(|column| format!("c{column}"))
            .collect::<Vec<_>>();
        let mut columns = vec![("vec", numbered(ROWS, WIDTH))];
        columns.extend(names.iter().map(|name| {
            (
                name.as_str(),
                Arc::new(arrow_array::Int32Array::from_iter_values(0..ROWS as i32)) as ArrayRef,
            )
        }));
        let dataset = one_fragment(&dir, columns).await;
        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &RoaringBitmap::from_iter([0]),
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };

        let stats = IoStats::new();
        let read = vectors
            .fetch(&[address(0, 5)], &stats, &access)
            .await
            .unwrap();
        assert_eq!(read.value(0).as_primitive::<Float32Type>().value(0), 5.0);
        assert_eq!(vectors.lance_rows(), 0, "the row was not read by offset");
        let bytes = stats.snapshot().bytes_read;
        let bound = 2 * store.block_size() as u64 + (WIDTH * 4) as u64;
        assert!(
            bytes <= bound,
            "one row cost {bytes} bytes, over the {bound} of a tail, one column and the row"
        );
        // The premise: the whole footer is more than that, or the bound would
        // hold whatever was read of it.
        let fragments = dataset.get_fragments();
        let path = dataset
            .data_dir()
            .join(fragments[0].metadata().files[0].path.as_str());
        let whole = IoStats::new();
        let file = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap()
            .with_io_stats(whole.recorder());
        FileReader::read_all_metadata(&file).await.unwrap();
        assert!(
            whole.snapshot().bytes_read > bound,
            "the whole footer is {} bytes, within the bound, so this tests nothing",
            whole.snapshot().bytes_read
        );
    }

    /// The half of the rule for an index given no cache that still holds: a
    /// data file the offset read serves is opened and laid out for every read,
    /// as a partition file is, so a second read costs what the first did.
    #[tokio::test]
    async fn without_a_cache_a_data_file_is_laid_out_for_every_read() {
        const WIDTH: i32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let dataset = one_fragment(&dir, vec![("vec", numbered(64, WIDTH))]).await;
        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &RoaringBitmap::from_iter([0]),
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };
        let mut reads = Vec::new();
        for _ in 0..2 {
            let stats = IoStats::new();
            vectors
                .fetch(&[address(0, 5)], &stats, &access)
                .await
                .unwrap();
            let read = stats.snapshot();
            reads.push((read.requests, read.bytes_read));
        }
        assert_eq!(reads[1], reads[0]);
        assert!(
            reads[0].0 > 1,
            "one request read the row and no footer: {:?}",
            reads[0]
        );
        assert_eq!(vectors.lance_rows(), 0);
    }

    /// A read that fails decides nothing: the data file is tried again by the
    /// next read, which reads it by offset once it is back - with a cache and
    /// without one.
    #[tokio::test]
    async fn a_data_file_that_could_not_be_read_is_tried_again() {
        const WIDTH: i32 = 64;
        for held in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let dataset = one_fragment(&dir, vec![("vec", numbered(64, WIDTH))]).await;
            let vectors = DatasetVectors::of(
                &dataset,
                &[dataset.schema().field("vec").unwrap().id],
                WIDTH as u32,
                DistanceType::L2,
                &RoaringBitmap::from_iter([0]),
            );
            let store = dataset.object_store(None).await.unwrap();
            let scheduler = scan_scheduler(&store);
            let direct = Arc::new(DirectReads::default());
            let cache = LanceCache::with_capacity(1 << 20);
            let access = FileAccess {
                scheduler: &scheduler,
                cache: held.then_some(&cache),
                local_reads: false,
                store: &store,
                direct: &direct,
            };
            let fragments = dataset.get_fragments();
            let name = fragments[0].metadata().files[0].path.clone();
            let data = dir.path().join("data").join(&name);
            let away = dir.path().join(&name);

            std::fs::rename(&data, &away).unwrap();
            let failed = vectors
                .fetch(&[address(0, 5)], &IoStats::new(), &access)
                .await;
            std::fs::rename(&away, &data).unwrap();
            assert!(failed.is_err(), "held {held}: a missing file was read");

            let read = vectors
                .fetch(&[address(0, 5)], &IoStats::new(), &access)
                .await
                .unwrap();
            assert_eq!(read.value(0).as_primitive::<Float32Type>().value(0), 5.0);
            assert_eq!(
                vectors.lance_rows(),
                0,
                "held {held}: the file that failed once was sent to Lance for good"
            );
        }
    }

    /// A layout a cache holds answers only for the count of rows it was
    /// checked against: two indexes sharing a cache, whose manifests disagree
    /// about how many rows a data file holds, each have their own verdict.
    #[tokio::test]
    async fn a_cached_layout_answers_only_for_the_rows_it_was_checked_against() {
        const WIDTH: i32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let dataset = one_fragment(&dir, vec![("vec", numbered(64, WIDTH))]).await;
        let field = dataset.schema().field("vec").unwrap().id;
        let covered = RoaringBitmap::from_iter([0]);
        let counted =
            DatasetVectors::of(&dataset, &[field], WIDTH as u32, DistanceType::L2, &covered);
        let mut miscounted =
            DatasetVectors::of(&dataset, &[field], WIDTH as u32, DistanceType::L2, &covered);
        let Some(Source::File(source)) = miscounted.fragments.get_mut(&0) else {
            panic!("the fragment is not one the offset read may serve");
        };
        source.rows += 1;
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let cache = LanceCache::with_capacity(1 << 20);
        let access = FileAccess {
            scheduler: &scheduler,
            cache: Some(&cache),
            local_reads: false,
            store: &store,
            direct: &direct,
        };

        for vectors in [&counted, &miscounted] {
            let read = vectors
                .fetch(&[address(0, 5)], &IoStats::new(), &access)
                .await
                .unwrap();
            assert_eq!(read.value(0).as_primitive::<Float32Type>().value(0), 5.0);
        }
        assert_eq!(counted.lance_rows(), 0);
        assert_eq!(
            miscounted.lance_rows(),
            1,
            "a layout checked against one count of rows was taken for another"
        );
    }

    /// A read left unpolled holds nothing another read of the index waits for.
    /// Eight reads over sixty-five fragments are polled once each - far enough
    /// to have as many fragments in flight as a read takes at once - and then
    /// left alone, which a caller racing a search against a timer may do; a
    /// ninth read still finishes.
    #[tokio::test]
    async fn a_read_left_unpolled_holds_up_no_other_read() {
        const WIDTH: i32 = 64;
        const FRAGMENTS: u32 = 65;
        let dir = tempfile::tempdir().unwrap();
        let batch = RecordBatch::try_from_iter(vec![("vec", numbered(FRAGMENTS, WIDTH))]).unwrap();
        let schema = batch.schema();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir.path().to_str().unwrap(),
            Some(WriteParams {
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), FRAGMENTS as usize);
        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &(0..FRAGMENTS).collect(),
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };
        let every_fragment = (0..FRAGMENTS)
            .map(|fragment_id| address(fragment_id, 0))
            .collect::<Vec<_>>();
        let stats = IoStats::new();

        let mut left = (0..8)
            .map(|_| Box::pin(vectors.fetch(&every_fragment, &stats, &access)))
            .collect::<Vec<_>>();
        for read in &mut left {
            assert!(futures::poll!(read.as_mut()).is_pending());
        }
        let read = tokio::time::timeout(
            Duration::from_secs(5),
            vectors.fetch(&[address(7, 0)], &IoStats::new(), &access),
        )
        .await
        .expect("a read waited on reads nobody polls")
        .unwrap();
        assert_eq!(read.value(0).as_primitive::<Float32Type>().value(0), 7.0);
        drop(left);
    }

    /// Deleted rows that run on without a gap are read by one range rather
    /// than one position at a time - the other way Lance's take goes - and come
    /// back as the rows they are.
    #[tokio::test]
    async fn a_run_of_deleted_rows_is_read_as_the_rows_it_is() {
        const WIDTH: i32 = 64;
        let dir = tempfile::tempdir().unwrap();
        let batch = RecordBatch::try_from_iter(vec![("vec", numbered(64, WIDTH))]).unwrap();
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir.path().to_str().unwrap(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .delete("_rowid >= 10 AND _rowid < 20")
            .await
            .unwrap();
        let vectors = DatasetVectors::of(
            &dataset,
            &[dataset.schema().field("vec").unwrap().id],
            WIDTH as u32,
            DistanceType::L2,
            &RoaringBitmap::from_iter([0]),
        );
        let store = dataset.object_store(None).await.unwrap();
        let scheduler = scan_scheduler(&store);
        let direct = Arc::new(DirectReads::default());
        let access = FileAccess {
            scheduler: &scheduler,
            cache: None,
            local_reads: false,
            store: &store,
            direct: &direct,
        };
        let read = vectors
            .fetch_deleted(
                &(10..20)
                    .map(|offset| address(0, offset))
                    .collect::<Vec<_>>(),
                &IoStats::new(),
                &access,
            )
            .await
            .unwrap();
        let rows = (0..read.len())
            .map(|at| read.value(at).as_primitive::<Float32Type>().value(0))
            .collect::<Vec<_>>();
        assert_eq!(rows, (10..20).map(|row| row as f32).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn deleted_rows_of_a_lance_2_0_file_are_read_in_one_read() {
        deleted_rows_are_read_through_lance_in_one_read(LanceFileVersion::V2_0).await;
    }

    #[tokio::test]
    async fn deleted_rows_of_a_legacy_file_are_read_in_one_read() {
        deleted_rows_are_read_through_lance_in_one_read(LanceFileVersion::Legacy).await;
    }
}
