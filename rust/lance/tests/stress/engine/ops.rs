// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The operations a seed runs, what the dataset must hold after them, and how one
//! runs on a handle.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use lance::Dataset;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use lance::dataset::{MergeInsertBuilder, WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance::index::{DatasetIndexExt, IndexBuildProgress};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_linalg::distance::MetricType;

use crate::incidents::StressHooks;

use super::checks::int64_column;
use super::data::{IVF_PARTITIONS, Scale, rows};
use super::rows::ClassRows;

pub const POOL_VARIABLE: &str = "LANCE_MEM_POOL_SIZE";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexKind {
    BTree,
    /// On the join key of a merge insert, which then takes its indexed path.
    BTreeId,
    /// On a column that changes, which an update in place takes fragments away from.
    BTreeStamp,
    IvfFlat,
    Inverted,
}

impl IndexKind {
    pub const ALL: [Self; 5] = [
        Self::BTree,
        Self::BTreeId,
        Self::BTreeStamp,
        Self::IvfFlat,
        Self::Inverted,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::BTree => "idx_category",
            Self::BTreeId => "idx_id",
            Self::BTreeStamp => "idx_gen",
            Self::IvfFlat => "idx_vec",
            Self::Inverted => "idx_text",
        }
    }

    pub fn column(self) -> &'static str {
        match self {
            Self::BTree => "category",
            Self::BTreeId => "id",
            Self::BTreeStamp => "gen",
            Self::IvfFlat => "vec",
            Self::Inverted => "text",
        }
    }

    /// The stages the build reports to its progress hooks.
    pub fn stages(self) -> &'static [&'static str] {
        match self {
            Self::BTree | Self::BTreeId | Self::BTreeStamp => &["load_data"],
            Self::IvfFlat => &[
                "train_ivf",
                "train_quantizer",
                "shuffle",
                "merge_partitions",
            ],
            Self::Inverted => &[
                "load_data",
                "tokenize_docs",
                "copy_partitions",
                "write_metadata",
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Append {
        first_id: i64,
        rows: i64,
    },
    Delete {
        lo: i64,
        hi: i64,
    },
    CreateIndex(IndexKind),
    /// Updates the rows with these ids that exist and inserts the others.
    /// Columns are functions of the id, so a lost update stays invisible,
    /// while a lost or doubled row does not.
    MergeInsert {
        first_id: i64,
        rows: i64,
    },
    Compact,
    Scan,
    /// Removes the versions before the last `retain` and the files only they use.
    Cleanup {
        retain: usize,
        delete_unverified: bool,
    },
    /// Indexes the fragments the indices do not cover yet.
    OptimizeIndices,
}

/// How an operation moves the latest version when it succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitShape {
    Exactly1,
    /// An index optimization may find nothing to do.
    UpTo1,
    /// A compaction commits a reservation of fragment ids before its rewrite,
    /// and may find nothing to do.
    UpTo2,
    Never,
}

impl CommitShape {
    pub fn max_delta(self) -> u64 {
        match self {
            Self::Exactly1 | Self::UpTo1 => 1,
            Self::UpTo2 => 2,
            Self::Never => 0,
        }
    }
}

impl Op {
    pub fn shape(&self) -> CommitShape {
        match self {
            Self::OptimizeIndices => CommitShape::UpTo1,
            Self::Compact => CommitShape::UpTo2,
            Self::Scan | Self::Cleanup { .. } => CommitShape::Never,
            _ => CommitShape::Exactly1,
        }
    }

    /// The operation without its arguments, for coverage counts.
    pub fn kind(&self) -> String {
        match self {
            Self::CreateIndex(kind) => format!("create_index_{}", kind.name()),
            _ => self.focus_name().to_string(),
        }
    }

    /// The operation as `LANCE_STRESS_ONLY` names it: one name for every index.
    pub fn focus_name(&self) -> &'static str {
        match self {
            Self::Append { .. } => "append",
            Self::Delete { .. } => "delete",
            Self::CreateIndex(_) => "create_index",
            Self::MergeInsert { .. } => "merge_insert",
            Self::Compact => "compact",
            Self::Scan => "scan",
            Self::Cleanup { .. } => "cleanup",
            Self::OptimizeIndices => "optimize_indices",
        }
    }
}

/// The memory limit on a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// `memory.max` at what the cgroup holds plus this many KiB. Going over it
    /// ends with the OOM killer.
    Cgroup { budget_kib: u64 },
    /// `LANCE_MEM_POOL_SIZE`: going over it means spilling to disk or a clean
    /// error. Sorts for BTree training and the indexed path of a merge insert
    /// honor it.
    Pool { kib: u64 },
    /// `InvertedIndexParams::memory_limit_mb`: going over it means more
    /// partitions in the index.
    FtsMemory { mib: u64 },
    /// `Scanner::io_buffer_size`: the read budget of a scan.
    IoBuffer { kib: u64 },
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cgroup { budget_kib } => write!(f, "cgroup:{budget_kib}"),
            Self::Pool { kib } => write!(f, "pool:{kib}"),
            Self::FtsMemory { mib } => write!(f, "fts_memory:{mib}"),
            Self::IoBuffer { kib } => write!(f, "io_buffer:{kib}"),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Append { first_id, rows } => {
                write!(f, "append:ids={first_id}..{}", first_id + rows)
            }
            Self::Delete { lo, hi } => write!(f, "delete:ids={lo}..{hi}"),
            Self::CreateIndex(kind) => write!(f, "create_index:{}", kind.name()),
            Self::MergeInsert { first_id, rows } => {
                write!(f, "merge_insert:ids={first_id}..{}", first_id + rows)
            }
            Self::Compact => f.write_str("compact"),
            Self::Scan => f.write_str("scan"),
            Self::Cleanup {
                retain,
                delete_unverified,
            } => write!(
                f,
                "cleanup:retain={retain},delete_unverified={delete_unverified}"
            ),
            Self::OptimizeIndices => f.write_str("optimize_indices"),
        }
    }
}

/// What the dataset must hold: the live row ids and the indices.
#[derive(Clone, Debug)]
pub struct Model {
    pub live: BTreeSet<i64>,
    pub next_id: i64,
    pub indices: BTreeSet<IndexKind>,
    /// An index misses fragments that rows were written to since it was built,
    /// which gives an index optimization work.
    pub unindexed: bool,
    /// `gen` of every row, in a scenario that has the column, once its checks
    /// have confirmed it.
    pub stamps: Option<ClassRows>,
    /// Every value of `gen` a version has held. An index that went stale can
    /// still answer one that no row holds any more.
    pub stamps_ever: Option<BTreeSet<i64>>,
}

impl Model {
    pub fn new(rows: i64) -> Self {
        Self {
            live: (0..rows).collect(),
            next_id: rows,
            indices: BTreeSet::new(),
            unindexed: false,
            stamps: None,
            stamps_ever: None,
        }
    }

    pub fn apply(&mut self, op: &Op) {
        match *op {
            Op::Append { first_id, rows } => {
                self.live.extend(first_id..first_id + rows);
                self.next_id = first_id + rows;
                self.unindexed = !self.indices.is_empty();
            }
            Op::Delete { lo, hi } => self.live.retain(|id| !(lo..hi).contains(id)),
            Op::CreateIndex(kind) => {
                self.indices.insert(kind);
                // The other indices may still miss fragments.
                if self.indices.len() == 1 {
                    self.unindexed = false;
                }
            }
            Op::MergeInsert { first_id, rows } => {
                self.live.extend(first_id..first_id + rows);
                self.next_id = self.next_id.max(first_id + rows);
                self.unindexed = !self.indices.is_empty();
            }
            Op::OptimizeIndices => self.unindexed = false,
            Op::Compact | Op::Scan | Op::Cleanup { .. } => {}
        }
    }
}

/// What a finished operation reports beyond success.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpOutput {
    pub fragments_removed: usize,
    /// The rows a scan returned, and the wrapping sum of their ids.
    pub scanned: Option<(usize, i64)>,
    /// The files a cleanup failed to delete, and went on without.
    pub failed_deletes: u64,
}

pub enum Outcome {
    Ok(OpOutput),
    Failed {
        message: String,
        status_unknown: bool,
    },
    Cancelled,
    /// The process died during the operation; a later process tells.
    Killed,
    Panicked(String),
}

impl Outcome {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ok(_) => "ok",
            Self::Failed { .. } => "err",
            Self::Cancelled => "cancelled",
            Self::Killed => "killed",
            Self::Panicked(_) => "panic",
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Self::Failed { message, .. } | Self::Panicked(message) => message,
            Self::Ok(_) | Self::Cancelled | Self::Killed => "",
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }
}

/// The operation of a step on the handle that meets the incidents.
pub fn op_future<'a>(
    ds: &'a mut Dataset,
    hooks: &Arc<StressHooks>,
    data_seed: u64,
    scale: Scale,
    op: Op,
    limit: Option<Limit>,
) -> BoxFuture<'a, lance::Result<OpOutput>> {
    let hooks = hooks.clone();
    match op {
        Op::Append {
            first_id,
            rows: count,
        } => Box::pin(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: scale.rows_per_file,
                progress: hooks,
                ..Default::default()
            };
            let source = rows(data_seed, first_id..first_id + count, scale);
            Box::pin(ds.append(source, Some(params))).await?;
            Ok(OpOutput::default())
        }),
        Op::Delete { lo, hi } => Box::pin(async move {
            let predicate = format!("id >= {lo} AND id < {hi}");
            Box::pin(ds.delete(&predicate)).await?;
            Ok(OpOutput::default())
        }),
        Op::MergeInsert {
            first_id,
            rows: count,
        } => Box::pin(async move {
            let source: Box<dyn RecordBatchReader + Send> =
                Box::new(rows(data_seed, first_id..first_id + count, scale));
            let job = MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])?
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()?;
            let (merged, _) = Box::pin(job.execute_reader(source)).await?;
            *ds = merged.as_ref().clone();
            Ok(OpOutput::default())
        }),
        Op::Compact => Box::pin(async move {
            let metrics = Box::pin(compact_files(ds, CompactionOptions::default(), None)).await?;
            Ok(OpOutput {
                fragments_removed: metrics.fragments_removed,
                ..Default::default()
            })
        }),
        Op::Cleanup {
            retain,
            delete_unverified,
        } => Box::pin(async move {
            let policy = CleanupPolicyBuilder::default()
                .retain_n_versions(ds, retain)
                .await?
                .delete_unverified(delete_unverified)
                .error_if_tagged_old_versions(false)
                .build();
            let removed = Box::pin(ds.cleanup_with_policy(policy)).await?;
            Ok(OpOutput {
                failed_deletes: removed.failed_deletes,
                ..Default::default()
            })
        }),
        Op::OptimizeIndices => Box::pin(async move {
            Box::pin(ds.optimize_indices(&OptimizeOptions::new().progress(hooks))).await?;
            Ok(OpOutput::default())
        }),
        Op::Scan => Box::pin(async move {
            let mut scan = ds.scan();
            if let Some(Limit::IoBuffer { kib }) = limit {
                scan.io_buffer_size(kib << 10);
            }
            let mut stream = scan.try_into_stream().await?;
            let mut count = 0;
            let mut sum = 0i64;
            while let Some(batch) = stream.try_next().await? {
                count += batch.num_rows();
                for id in int64_column(&batch, "id") {
                    sum = sum.wrapping_add(id);
                }
            }
            Ok(OpOutput {
                scanned: Some((count, sum)),
                ..Default::default()
            })
        }),
        Op::CreateIndex(kind) => Box::pin(async move {
            let columns = [kind.column()];
            let progress: Arc<dyn IndexBuildProgress> = hooks;
            let name = kind.name().to_string();
            match kind {
                IndexKind::BTree | IndexKind::BTreeId | IndexKind::BTreeStamp => {
                    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
                    ds.create_index_builder(&columns, IndexType::BTree, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
                IndexKind::IvfFlat => {
                    let params = VectorIndexParams::ivf_flat(IVF_PARTITIONS, MetricType::L2);
                    ds.create_index_builder(&columns, IndexType::IvfFlat, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
                IndexKind::Inverted => {
                    let params = match limit {
                        Some(Limit::FtsMemory { mib }) => {
                            InvertedIndexParams::default().memory_limit_mb(mib)
                        }
                        _ => InvertedIndexParams::default(),
                    };
                    ds.create_index_builder(&columns, IndexType::Inverted, &params)
                        .name(name)
                        .replace(true)
                        .progress(progress)
                        .await
                }
            }
            .map(|_| OpOutput::default())
        }),
    }
}
