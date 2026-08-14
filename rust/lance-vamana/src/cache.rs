// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a query would otherwise read again about a partition it has probed
//! before.
//!
//! A lazy walk reads two things whose contents cannot change: the layout of a
//! partition file, and the codes and row ids of the vertices in it. A segment is
//! written once and never edited - every maintenance pass writes a new segment
//! under a new uuid and commits it in place of the old one - so a partition file
//! is immutable for as long as anything can name it, and both of those reads are
//! the same read on every query that probes it.
//!
//! Which makes the cache the last multiplier of the lazy read, and the largest
//! one left. Of the 18.2 MB a query reads at 65536 rows a partition,
//! **17.5 MB is the code column** (`examples/lazy_walk.rs`), re-read by every
//! query that probes the partition; the edges and the vectors the walk actually
//! fetches are the remaining 0.7 MB. Holding the codes across queries is
//! therefore worth another factor of twenty-five, and it is what makes the
//! lazy read's own number - a tenth of reading whole - a *lower* bound rather
//! than the result.
//!
//! An index that was given no cache goes down neither path: it reads, and it
//! does not build a key to decide to. That is not the same thing as a cache of
//! capacity zero, which is what it looks like from the outside -
//! [`LanceCache::no_cache`] admits an entry and reclaims it when it next runs
//! its housekeeping, so a partition read a moment ago is served out of a cache
//! holding nothing at all. Rare, invisible, and enough to make a measurement of
//! the uncached arm quietly wrong.
//!
//! Two things this deliberately does not do:
//!
//! - **It does not decide what the budget is.** [`LanceCache`] arrives from the
//!   caller, so an index can be given a cache of its own, or share one with
//!   every other index in a process, or be handed one that spills to disk. What
//!   the working set costs is a property of the data: at three bits and
//!   `d = 128` a vertex is 68 bytes on disk and about 116 in memory, so a
//!   million rows is 65 MiB read and 110 MiB held.
//! - **It does not cache the walk.** Edges and vectors are fetched by the walk
//!   itself, a few hundred rows of a partition per query, and which few hundred
//!   depends on the query. Caching those would be caching a query's answer, not
//!   a partition's contents.

use std::borrow::Cow;
use std::sync::Arc;

use lance_core::cache::{CacheKey, CacheKeySchema, Context, DeepSizeOf, KeyBuilder, LanceCache};
use lance_core::{Error, Result};
use lance_file::reader::CachedFileMetadata;
use lance_index::vector::bq::storage::RabitQuantizationStorage;
use object_store::path::Path;
use uuid::Uuid;

use crate::codes::{self, CODE_COLUMN};
use crate::format::{IndexMetadata, ROW_ID_COLUMN};
use crate::io::{PartitionFile, read_partition_batch};
use crate::partition::row_ids_from_batch;
use crate::segment::PartitionEntry;

/// The part of a partition a lazy walk holds in memory for the whole walk.
///
/// The two together rather than one entry each, because they are read by one
/// projection and are useless apart: the codes are what the walk steers by and
/// the row ids are what its answer is made of, and a walk that had one without
/// the other would have to read the partition to get the other.
///
/// Sized by [`DeepSizeOf`] rather than by the bytes it was read from, which is
/// the whole point of that trait here: the codes are stored as one contiguous
/// stride a vertex and read back into the column layout Lance's estimator wants,
/// so the resident form is about 1.7 times the on-disk one at three bits. A
/// budget in on-disk bytes would quietly hold two thirds of what it was asked to.
pub(crate) struct Resident {
    pub(crate) row_ids: Vec<u64>,
    pub(crate) codes: RabitQuantizationStorage,
}

impl DeepSizeOf for Resident {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.row_ids.deep_size_of_children(context) + self.codes.deep_size_of_children(context)
    }
}

/// One partition's resident part, keyed by the segment it belongs to.
///
/// Both halves of the key are load-bearing. Without the partition id every
/// partition of a segment would answer with the first one's row ids; without the
/// segment uuid every segment's partition 3 would, which is the ordinary state
/// of an index that has been appended to. Neither mistake is visible as an
/// error - the codes of the wrong partition steer a walk to plausible local ids,
/// and the row ids beside them turn those into plausible row addresses.
#[derive(Debug)]
pub(crate) struct ResidentKey {
    pub(crate) segment: Uuid,
    pub(crate) partition_id: u32,
}

impl CacheKey for ResidentKey {
    type ValueType = Resident;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{}/{}", self.segment, self.partition_id))
    }

    fn type_name() -> &'static str {
        "VamanaResident"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance-vamana.resident", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.segment.as_bytes());
        builder.write_u32(self.partition_id);
    }
}

/// One partition file's layout, keyed by its path.
///
/// The path and not the segment uuid: a segment holds a file per partition, and
/// the path is what Lance's own file-metadata cache is keyed by, so the two
/// agree about what identifies a file.
#[derive(Debug)]
pub(crate) struct FileKey<'a> {
    pub(crate) path: &'a Path,
}

impl CacheKey for FileKey<'_> {
    type ValueType = CachedFileMetadata;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.path.as_ref())
    }

    fn type_name() -> &'static str {
        "VamanaFileMetadata"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance-vamana.file-metadata", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.path.as_ref());
    }
}

/// Read the row ids and the codes of a partition, or take the ones already read.
///
/// The one read a lazy walk makes that is proportional to the partition rather
/// than to what the walk touches, which is what makes it the one worth keeping.
///
/// A cache miss under concurrency loads once and not once per caller: two
/// queries probing the same partition at the same moment share the read rather
/// than duplicating it, which matters exactly here, because the thing being
/// shared is the largest read either of them makes.
///
/// `None` reads, every time, and does not go near a key. It is not the same as
/// a cache of capacity zero, which admits an entry before reclaiming it and so
/// serves the occasional hit out of what is supposed to be nothing.
pub(crate) async fn resident(
    cache: Option<&LanceCache>,
    segment: Uuid,
    entry: &PartitionEntry,
    file: &PartitionFile,
    metadata: &IndexMetadata,
) -> Result<Arc<Resident>> {
    let params = metadata.codes.as_ref().ok_or_else(|| {
        Error::internal("a Vamana lazy walk was scheduled for a segment without codes".to_string())
    })?;
    let read = || async {
        // One projection over two columns rather than two reads: the row ids are
        // three per cent of what the codes weigh, and the walk needs them for
        // its answer anyway. Reading them lazily instead would be the worse
        // trade by far - `__row_id` is the one compressed column of a partition
        // file, so a single scattered row of it drags a two-kilobyte mini-block.
        let reader = file.project(&[ROW_ID_COLUMN, CODE_COLUMN]).await?;
        let batch = read_partition_batch(&reader, entry.num_rows).await?;
        let row_ids = row_ids_from_batch(&batch)?;
        let codes = codes::storage(
            params,
            metadata.distance_type,
            metadata.dimension,
            &row_ids,
            &codes::column(&batch)?,
        )?;
        Ok(Resident { row_ids, codes })
    };
    match cache {
        Some(cache) => {
            let key = ResidentKey {
                segment,
                partition_id: entry.partition_id,
            };
            cache.get_or_insert_with_key(key, read).await
        }
        None => Ok(Arc::new(read().await?)),
    }
}
