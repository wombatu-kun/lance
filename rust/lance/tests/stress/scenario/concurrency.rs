// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The `concurrency` scenario: writers and readers at once on one dataset.
//!
//! A step is a round: workers run their operations in parallel on handles of
//! their own, readers check snapshots meanwhile, and once it is quiet every
//! version the round made is opened and checked.
//!
//! A row belongs to the class `id % (workers + 2)`, whatever fragment it lies
//! in, and a worker only changes the rows of its own class. So what a worker's
//! rows hold depends on its own operations alone, and every snapshot must show
//! them as of some prefix of those operations, while the operations still fight
//! over the same fragments. Two more classes are shared. All workers upsert, update
//! and delete the same keys of the first, so an operation must apply to all of them
//! or to none. All workers add 1 to the whole of the second, a value computed from
//! what was read, so that a lost increment shows exactly.
//!
//! Which rebases happen is not left to timing alone. Some rounds force a pair of
//! `PAIRS`, the others committing over what the first worker landed, and round 1
//! commits writes prepared before a cleanup took every version they were based
//! on, see `HOLES`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{self, Write as _};
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::{BoxFuture, try_join_all};
use lance::Dataset;
use lance::Error;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::index::DatasetIndexRemapperOptions;
use lance::dataset::optimize::{
    CompactionOptions, RewriteResult, commit_compaction, plan_compaction,
};
use lance::dataset::scanner::RowAddrTreeMap;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::write::delete::DeleteBuilder;
use lance::dataset::write::merge_insert::{
    MergeInsertWriteMode, MergeStats, WhenMatched, WhenNotMatched,
};
use lance::dataset::{
    CommitBuilder, MergeInsertBuilder, MergeInsertJob, UncommittedMergeInsert, UpdateBuilder,
    WriteMode, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance::session::Session;
use lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::object_store::ObjectStoreParams;
use lance_table::format::IndexMetadata;
use tokio::sync::{Barrier, watch};
use tokio::task::JoinHandle;

use super::{
    Answers, Failure, IndexKind, Journal, Op, OpOutput, PlanRng, Run, Scale, StoreChoice,
    VECTOR_DIM, category, check_index, fail, int64_column, one_line, op_future, pick_from, text,
    vector,
};
use crate::harness::{ENV_SERIAL, SelfTest};
use crate::incidents::{Area, Gate, Incident, IncidentState, Method, StressHooks, StressWrapper};

/// A hold point can eat the default 30 s, and an attempt past the limit is cut
/// off even if its commit has landed.
const RETRY_TIMEOUT: Duration = Duration::from_secs(120);
/// The order of a plan is a wish: a wait that cannot end costs this much and
/// counts as `gate_timeout`, while a real hang stays the watchdog's. Ranks rule
/// out a circle of holds, but a shared `Session` can close one: its metadata cache
/// loads a file once, so a hold inside that load stalls whoever reads that file,
/// perhaps the very operation the hold waits for.
const GATE_LIMIT: Duration = Duration::from_secs(10);
/// How many times a worker tries an operation among the others. After that it
/// waits for them to finish and tries alone, when it must succeed.
const PARALLEL_ATTEMPTS: usize = 3;
const MIN_CLASS_ROWS: usize = 20;
const CONTENDED_EXISTING_KEYS: RangeInclusive<u64> = 8..=16;
const CONTENDED_NEW_KEYS: RangeInclusive<u64> = 4..=12;
const CONTENDED_KINDS: [&str; 3] = ["contended_upsert", "contended_update", "contended_delete"];
/// An operation's number is `round * NUMBERS_PER_ROUND + rank + 1`.
const NUMBERS_PER_ROUND: i64 = 100;

/// The rows of one class: each id with the `gen` it holds.
pub(super) type ClassRows = BTreeMap<i64, i64>;

pub(super) fn id_run(first_id: i64, rows: i64, stride: i64) -> impl Iterator<Item = i64> {
    (0..rows).map(move |slot| first_id + slot * stride)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundOp {
    Append {
        first_id: i64,
        rows: i64,
    },
    Delete {
        lo: i64,
        hi: i64,
    },
    /// Updates the rows with these ids that exist and inserts the others.
    MergeInsert {
        first_id: i64,
        rows: i64,
    },
    /// Rewrites `gen` in place for the rows with these ids that exist.
    MergeInsertColumns {
        first_id: i64,
        rows: i64,
    },
    Update {
        lo: i64,
        hi: i64,
    },
    /// An upsert of the round's keys, the same ones for every worker.
    ContendedUpsert,
    /// Sets `gen` on those of the round's keys that exist.
    ContendedUpdate,
    /// Deletes the round's keys.
    ContendedDelete,
    /// `gen = gen + 1` for the whole class of counters.
    CounterUpdate,
    CreateIndex(IndexKind),
    OptimizeIndices,
    Compact,
    Cleanup {
        retain: usize,
    },
    /// A reservation of fragment ids: a commit that changes no rows, so that a
    /// cleanup keeping one version takes every version before it.
    Reserve,
}

impl RoundOp {
    fn kind(&self) -> &'static str {
        match self {
            Self::Append { .. } => "append",
            Self::Delete { .. } => "delete",
            Self::MergeInsert { .. } => "merge_insert",
            Self::MergeInsertColumns { .. } => "merge_insert_columns",
            Self::Update { .. } => "update",
            Self::ContendedUpsert => "contended_upsert",
            Self::ContendedUpdate => "contended_update",
            Self::ContendedDelete => "contended_delete",
            Self::CounterUpdate => "counter_update",
            Self::CreateIndex(_) => "create_index",
            Self::OptimizeIndices => "optimize_indices",
            Self::Compact => "compact",
            Self::Cleanup { .. } => "cleanup",
            Self::Reserve => "reserve",
        }
    }

    /// What a race may make an operation return. Which pairs conflict follows the
    /// matrix in `rust/lance/src/io/commit/conflict_matrix/cases.rs`: an append meets
    /// no conflict in what this scenario runs, an index only one with another index
    /// or a compaction, and a compaction anything but an append. Lance retries the
    /// row operations itself and gives up with `TooMuchWriteContention`. So can any
    /// commit that keeps losing the race for a manifest until its time is up.
    fn is_legal_error(&self, error: &Error, round: &Round, rank: usize) -> bool {
        let gave_up = matches!(error, Error::TooMuchWriteContention { .. });
        let conflict = matches!(error, Error::RetryableCommitConflict { .. });
        let mut others = round.ops.iter().filter(|other| other.rank != rank);
        match self {
            Self::Cleanup { .. } => false,
            Self::Append { .. }
            | Self::Reserve
            | Self::Delete { .. }
            | Self::MergeInsert { .. }
            | Self::MergeInsertColumns { .. }
            | Self::Update { .. }
            | Self::ContendedUpsert
            | Self::ContendedUpdate
            | Self::ContendedDelete
            | Self::CounterUpdate => gave_up,
            Self::CreateIndex(_) | Self::OptimizeIndices => {
                gave_up
                    || (conflict
                        && others.any(|other| {
                            matches!(
                                other.op,
                                Self::CreateIndex(_) | Self::OptimizeIndices | Self::Compact
                            )
                        }))
            }
            Self::Compact => {
                gave_up
                    || (conflict
                        && others.any(|other| {
                            !matches!(other.op, Self::Append { .. } | Self::Cleanup { .. })
                        }))
            }
        }
    }

    /// Whether it acts on the round's keys.
    fn is_contended(&self) -> bool {
        matches!(
            self,
            Self::ContendedUpsert | Self::ContendedUpdate | Self::ContendedDelete
        )
    }

    /// Whether it changes rows rather than how they are stored or found.
    fn is_row_op(&self) -> bool {
        matches!(
            self,
            Self::Append { .. }
                | Self::Delete { .. }
                | Self::MergeInsert { .. }
                | Self::MergeInsertColumns { .. }
                | Self::Update { .. }
                | Self::ContendedUpsert
                | Self::ContendedUpdate
                | Self::ContendedDelete
                | Self::CounterUpdate
        )
    }

    /// How many versions a finished try can add.
    fn commits(&self, output: Option<&OpOutput>) -> (u64, u64) {
        match (self, output) {
            (Self::Cleanup { .. }, _) => (0, 0),
            // A reservation of fragment ids, then the rewrite. A failed rewrite
            // leaves the reservation behind.
            (Self::Compact, None) => (0, 1),
            (Self::Compact, Some(output)) if output.fragments_removed > 0 => (2, 2),
            (Self::Compact, Some(_)) => (0, 2),
            // Nothing to optimize, or no key left to act on.
            (Self::OptimizeIndices | Self::ContendedUpdate | Self::ContendedDelete, Some(_)) => {
                (0, 1)
            }
            (_, None) => (0, 0),
            (_, Some(_)) => (1, 1),
        }
    }

    /// Every operation commits at most this many versions when it succeeds.
    fn max_commits(&self) -> usize {
        match self {
            Self::Cleanup { .. } => 0,
            Self::Compact => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for RoundOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Append { first_id, rows } => write!(f, "append:first={first_id},rows={rows}"),
            Self::Delete { lo, hi } => write!(f, "delete:ids={lo}..{hi}"),
            Self::MergeInsert { first_id, rows } => {
                write!(f, "merge_insert:first={first_id},rows={rows}")
            }
            Self::MergeInsertColumns { first_id, rows } => {
                write!(f, "merge_insert_columns:first={first_id},rows={rows}")
            }
            Self::Update { lo, hi } => write!(f, "update:ids={lo}..{hi}"),
            Self::ContendedUpsert | Self::ContendedUpdate | Self::ContendedDelete => {
                f.write_str(self.kind())
            }
            Self::CounterUpdate => f.write_str("counter_update"),
            Self::CreateIndex(kind) => write!(f, "create_index:{}", kind.name()),
            Self::OptimizeIndices => f.write_str("optimize_indices"),
            Self::Compact => f.write_str("compact"),
            Self::Cleanup { retain } => write!(f, "cleanup:retain={retain}"),
            Self::Reserve => f.write_str("reserve"),
        }
    }
}

/// The operation stops at a named point until an operation of a lower rank is
/// over. A worker's own operations rise in rank too, so nothing waits in a circle.
/// With `arrival` it waits instead until an operation of a higher rank gets to its
/// own point, where that one waits for this one to end: still one way.
#[derive(Clone, Debug)]
struct Hold {
    point: Incident,
    until: usize,
    arrival: bool,
}

#[derive(Clone, Debug)]
struct PlannedOp {
    /// Its place in one order of the round that keeps each worker's own order.
    rank: usize,
    worker: usize,
    /// What it writes to `gen`, which tells it from every other operation.
    number: i64,
    /// Whether the worker's handle catches up with the latest version first.
    catch_up: bool,
    op: RoundOp,
    hold: Option<Hold>,
    /// It begins only once the operation of this rank is over. The wait has no
    /// limit: what it must see is that operation's result.
    start_after: Option<usize>,
    /// Its first try is prepared on the version the round starts from before
    /// anyone writes, and committed only once it may begin.
    staged: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReaderKind {
    /// Opens the latest version for every read.
    FreshOpen,
    /// Keeps one handle and moves it to the latest version for every read.
    CheckoutLatest,
    /// Keeps a handle on the version the round starts from.
    Pinned,
    /// Keeps a handle on the oldest version there is, which a cleanup can take.
    OldSnapshot,
}

impl ReaderKind {
    fn name(self) -> &'static str {
        match self {
            Self::FreshOpen => "fresh_open",
            Self::CheckoutLatest => "checkout_latest",
            Self::Pinned => "pinned",
            Self::OldSnapshot => "old_snapshot",
        }
    }
}

#[derive(Clone, Debug)]
struct PlannedReader {
    kind: ReaderKind,
    hold: Option<Hold>,
    /// It begins reading only once the operation of this rank is over.
    start_after: Option<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct Round {
    theme: Option<&'static str>,
    keys: Vec<i64>,
    ops: Vec<PlannedOp>,
    readers: Vec<PlannedReader>,
    /// In a round of `HOLES`, what the first worker hides from the others.
    hole: Option<&'static str>,
}

impl Round {
    fn ops_of(&self, worker: usize) -> Vec<PlannedOp> {
        self.ops
            .iter()
            .filter(|op| op.worker == worker)
            .cloned()
            .collect()
    }

    fn has_cleanup(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op.op, RoundOp::Cleanup { .. }))
    }
}

/// What a seed decides about its rounds.
#[derive(Clone, Debug)]
pub(super) struct RoundsPlan {
    workers: usize,
    pub(super) stable_row_ids: bool,
    /// One `Session` for the whole round, so that its caches are shared, or one
    /// per handle, as separate processes would have.
    shared_session: bool,
    /// `id` is declared the primary key, which promises a conflict between two
    /// inserts of one new key. Only such a seed races for new keys: without a
    /// declared key a duplicate is legal, as `conflict_resolver.rs` only promises
    /// the conflict for one, see #5633.
    declared_key: bool,
    pub(super) rounds: Vec<Round>,
}

/// The rows the plan expects, class by class.
#[derive(Clone, Debug)]
struct RowsModel {
    workers: usize,
    classes: Vec<ClassRows>,
    next_ids: Vec<i64>,
}

impl RowsModel {
    fn new(initial_rows: i64, workers: usize) -> Self {
        let stride = workers as i64 + 2;
        let mut classes = vec![ClassRows::new(); workers + 2];
        for id in 0..initial_rows {
            classes[(id % stride) as usize].insert(id, 0);
        }
        let next_ids = (0..stride)
            .map(|class| {
                let offset = (class - initial_rows % stride).rem_euclid(stride);
                initial_rows + offset
            })
            .collect();
        Self {
            workers,
            classes,
            next_ids,
        }
    }

    fn stride(&self) -> i64 {
        self.workers as i64 + 2
    }

    fn contended(&self) -> usize {
        self.workers
    }

    fn counters(&self) -> usize {
        self.workers + 1
    }

    fn apply(&mut self, worker: usize, op: &RoundOp, number: i64, keys: &[i64]) {
        let stride = self.stride();
        match *op {
            RoundOp::Append { first_id, rows } => {
                self.classes[worker].extend(id_run(first_id, rows, stride).map(|id| (id, number)));
                self.next_ids[worker] = first_id + rows * stride;
            }
            RoundOp::Delete { lo, hi } => {
                self.classes[worker].retain(|id, _| !(lo..hi).contains(id));
            }
            RoundOp::MergeInsert { first_id, rows } => {
                self.classes[worker].extend(id_run(first_id, rows, stride).map(|id| (id, number)));
                let next = first_id + rows * stride;
                self.next_ids[worker] = self.next_ids[worker].max(next);
            }
            RoundOp::MergeInsertColumns { first_id, rows } => {
                for id in id_run(first_id, rows, stride) {
                    if let Some(stamp) = self.classes[worker].get_mut(&id) {
                        *stamp = number;
                    }
                }
            }
            RoundOp::Update { lo, hi } => {
                for (_, stamp) in self.classes[worker].range_mut(lo..hi) {
                    *stamp = number;
                }
            }
            RoundOp::ContendedUpsert => {
                let class = self.contended();
                self.classes[class].extend(keys.iter().map(|key| (*key, number)));
                if let Some(last) = keys.iter().max() {
                    self.next_ids[class] = self.next_ids[class].max(last + stride);
                }
            }
            RoundOp::ContendedUpdate => {
                let class = self.contended();
                for key in keys {
                    if let Some(stamp) = self.classes[class].get_mut(key) {
                        *stamp = number;
                    }
                }
            }
            RoundOp::ContendedDelete => {
                let class = self.contended();
                for key in keys {
                    self.classes[class].remove(key);
                }
            }
            RoundOp::CounterUpdate => {
                let class = self.counters();
                for stamp in self.classes[class].values_mut() {
                    *stamp += 1;
                }
            }
            RoundOp::CreateIndex(_)
            | RoundOp::OptimizeIndices
            | RoundOp::Compact
            | RoundOp::Cleanup { .. }
            | RoundOp::Reserve => {}
        }
    }

    fn all_rows(&self) -> ClassRows {
        self.classes
            .iter()
            .flatten()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    fn next_id(&self) -> i64 {
        self.classes
            .iter()
            .flat_map(|class| class.keys().next_back())
            .max()
            .map_or(0, |last| last + 1)
    }
}

/// The keys a round's workers fight over. They all exist when the round starts or
/// none does, so that a snapshot tells what happened to them.
fn plan_keys(
    rng: &mut PlanRng,
    rows: &mut RowsModel,
    queues: &mut [Vec<RoundOp>],
    declared_key: bool,
    only_new_keys: bool,
) -> Vec<i64> {
    let has =
        |queues: &[Vec<RoundOp>], kind: RoundOp| queues.iter().flatten().any(|op| *op == kind);
    let deletes = has(queues, RoundOp::ContendedDelete);
    // Two upserts after a delete race for a key that is new again, which only a
    // seed with a declared key may race for: see `RoundsPlan::declared_key`.
    if deletes && !declared_key {
        let mut upserts = queues
            .iter_mut()
            .flatten()
            .filter(|op| **op == RoundOp::ContendedUpsert);
        upserts.next();
        for op in upserts {
            *op = RoundOp::ContendedUpdate;
        }
    }
    if !queues.iter().flatten().any(RoundOp::is_contended) {
        return Vec::new();
    }
    let class = rows.contended();
    let stride = rows.stride();
    let only_upserts = !deletes && !has(queues, RoundOp::ContendedUpdate);
    let new_only = declared_key && only_upserts && (only_new_keys || rng.chance(0.5));
    let mut keys = Vec::new();
    if !new_only {
        let ids: Vec<i64> = rows.classes[class].keys().copied().collect();
        let existing = rng.range(CONTENDED_EXISTING_KEYS).min(ids.len() as u64) as usize;
        let start = rng.range(0..=(ids.len() - existing) as u64) as usize;
        keys.extend_from_slice(&ids[start..start + existing]);
    }
    if new_only || (declared_key && only_upserts && rng.chance(0.5)) {
        let fresh = rng.range(CONTENDED_NEW_KEYS) as i64;
        keys.extend(id_run(rows.next_ids[class], fresh, stride));
    }
    if keys.is_empty() {
        // Deletes have used up the class.
        for op in queues.iter_mut().flatten() {
            if op.is_contended() {
                *op = RoundOp::CounterUpdate;
            }
        }
        return keys;
    }
    if deletes {
        // Whether they exist after this round depends on who commits last, so no
        // later round picks them.
        rows.apply(class, &RoundOp::ContendedDelete, 0, &keys);
    } else if has(queues, RoundOp::ContendedUpsert) {
        rows.apply(class, &RoundOp::ContendedUpsert, 0, &keys);
    }
    keys
}

/// A contiguous run of a class's rows, as a range of ids.
fn pick_run(rng: &mut PlanRng, class: &ClassRows, len: usize) -> (i64, i64) {
    let ids: Vec<i64> = class.keys().copied().collect();
    let start = rng.range(0..=(ids.len() - len) as u64) as usize;
    (ids[start], ids[start + len - 1] + 1)
}

const OP_WEIGHTS: [(&str, u64); 13] = [
    ("append", 15),
    ("delete", 15),
    ("merge_insert", 15),
    ("merge_insert_columns", 14),
    ("update", 15),
    ("contended_upsert", 10),
    ("contended_update", 6),
    ("contended_delete", 4),
    ("counter_update", 10),
    ("create_index", 10),
    ("optimize_indices", 4),
    ("compact", 10),
    ("cleanup", 3),
];

/// What every worker of a themed round does: a pair the weights rarely bring
/// together is then certain to meet.
const THEMES: [&str; 11] = [
    "append",
    "delete",
    "merge_insert",
    "merge_insert_columns",
    "update",
    "contended_upsert",
    "contended_mixed",
    "counter_update",
    "create_index",
    // An index on `gen` being built while `gen` is rewritten in place.
    "stamp_index",
    "compact",
];

/// Pairs that have to meet: the first commits on a snapshot from before the
/// second landed, so that its rebase has to deal with it. A fast operation rarely
/// sees a slow one land under it by chance: 16 seeds brought one delete over a
/// compaction. So a round of a seed gets one for sure, the others committing over
/// the first of its workers. Seeds take the pairs in turn, so that with seven
/// rounds or more any `PAIRS.len() / PAIR_ROUNDS` seeds in a row meet them all.
const PAIRS: [(&str, &str); 24] = [
    ("delete", "compact"),
    ("update", "compact"),
    ("merge_insert", "compact"),
    ("merge_insert_columns", "compact"),
    ("contended_upsert", "compact"),
    ("counter_update", "compact"),
    ("create_index", "compact"),
    ("optimize_indices", "compact"),
    ("stamp_index", "merge_insert_columns"),
    ("optimize_indices", "merge_insert_columns"),
    ("create_index", "update"),
    ("create_index", "delete"),
    ("compact", "delete"),
    ("compact", "update"),
    ("compact", "merge_insert_columns"),
    ("compact", "create_index"),
    ("merge_insert_columns", "delete"),
    ("merge_insert_columns", "update"),
    ("update", "merge_insert_columns"),
    // Files a writer has yet to commit, which a cleanup must leave alone.
    ("append", "cleanup"),
    ("merge_insert", "cleanup"),
    ("update", "cleanup"),
    ("delete", "cleanup"),
    ("create_index", "cleanup"),
];
const PAIR_ROUNDS: usize = 3;

/// The round with a hole in the history. In the first, every fragment holds rows
/// of every class, so a write to the whole class of the first worker touches all
/// that the others can, and no deletion file or index exists yet whose removal
/// could fail them before their commit.
const HOLE_ROUND: usize = 1;

/// Pairs over a hole in the history. The first is prepared without a commit on
/// the version the round starts from. Then the first worker makes the second
/// kind of write, a commit that changes no rows, and a cleanup that keeps only
/// that last version, so that every version in between is gone. Only then is the
/// first committed. It must be refused, or must change nothing that went missing.
/// Committing only after the cleanup keeps the check of conflicts after it too:
/// a commit checked before a cleanup and written after it can take the number of
/// a version the cleanup removed, <https://github.com/lance-format/lance/issues/3718>.
/// Seeds take the pairs in turn.
const HOLES: [(&str, &str); 8] = [
    ("delete", "hole_rows"),
    ("merge_insert", "hole_rows"),
    ("merge_insert_columns", "hole_rows"),
    ("compact", "hole_rows"),
    ("delete", "hole_compact"),
    ("merge_insert", "hole_compact"),
    ("contended_upsert", "hole_keys"),
    ("stamp_index", "hole_stamp"),
];

/// The operation a name of `PAIRS` or `HOLES` runs, as `LANCE_STRESS_ONLY` names it.
fn focus_name(name: &'static str) -> &'static str {
    match name {
        "stamp_index" => "create_index",
        "hole_rows" => "update",
        "hole_compact" => "compact",
        "hole_keys" => "contended_upsert",
        "hole_stamp" => "merge_insert_columns",
        other => other,
    }
}

/// Two operations a round makes meet: in `PAIRS` the one in flight and the one
/// that lands under it, in `HOLES` the one in flight and the hidden write.
type Pair = (&'static str, &'static str);

/// Whether a forced pair runs an operation of the focus. The first worker of a
/// round over a hole cleans up as well.
fn in_focus(pair: Pair, hole: bool, focus: &BTreeSet<&'static str>) -> bool {
    focus.contains(focus_name(pair.0))
        || focus.contains(focus_name(pair.1))
        || (hole && focus.contains("cleanup"))
}

/// What a focus can keep of this scenario: the operations of its forced pairs.
pub(super) fn focusable_ops() -> Vec<&'static str> {
    let mut ops: BTreeSet<&'static str> = PAIRS
        .iter()
        .chain(&HOLES)
        .flat_map(|&(first, second)| [focus_name(first), focus_name(second)])
        .collect();
    ops.insert("cleanup");
    ops.into_iter().collect()
}

/// The pairs a seed forces: one per pair round, which seeds take in turn from
/// `PAIRS`, and the one over the hole from `HOLES`, which moves one further every
/// `HOLES.len()` seeds: with as many holes as turns of the pairs, a hole would
/// otherwise meet the same pairs on every seed. A focus keeps the pairs that run
/// one of its operations; a round left without one is a random round.
fn forced_pairs(
    seed: u64,
    rounds: usize,
    focus: &BTreeSet<&'static str>,
) -> (BTreeMap<usize, Pair>, Option<Pair>) {
    let kept = |pairs: &[Pair], hole: bool| -> Vec<Pair> {
        pairs
            .iter()
            .copied()
            .filter(|pair| focus.is_empty() || in_focus(*pair, hole, focus))
            .collect()
    };
    let pairs = kept(&PAIRS, false);
    let holes = kept(&HOLES, true);
    let mut pair_rounds = BTreeMap::new();
    if !pairs.is_empty() {
        // From the third round on: a cleanup needs some versions behind it.
        let turn = (seed % pairs.len() as u64) as usize * PAIR_ROUNDS;
        for (slot, n) in (3..=rounds).step_by(2).take(PAIR_ROUNDS).enumerate() {
            pair_rounds.insert(n, pairs[(turn + slot) % pairs.len()]);
        }
    }
    let hole = (!holes.is_empty()).then(|| {
        let len = holes.len() as u64;
        holes[((seed + seed / len) % len) as usize]
    });
    (pair_rounds, hole)
}

/// What decides the operations of a round.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RoundKind {
    Random,
    /// One round of a seed with a declared key, where only the key filter can stop
    /// the duplicate. Another
    /// key the racers share makes them conflict on that row instead. When
    /// `indexed`, the first worker builds the index on the key before the others
    /// start, so that every racer takes the indexed path of merge insert.
    OnlyNewKeys {
        indexed: bool,
    },
    Pair {
        in_flight: &'static str,
        landed: &'static str,
    },
    Hole {
        in_flight: &'static str,
        hidden: &'static str,
    },
}

/// The point right before an operation's last commit.
fn commit_point(op: &RoundOp) -> Incident {
    Incident::HoldCall {
        method: Method::Put,
        area: Area::Manifests,
        // After its reservation of fragment ids.
        nth: if matches!(op, RoundOp::Compact) { 2 } else { 1 },
        after: false,
    }
}

fn random_kind(rng: &mut PlanRng) -> &'static str {
    let total: u64 = OP_WEIGHTS.iter().map(|(_, weight)| weight).sum();
    let mut roll = rng.range(0..=total - 1);
    for (kind, weight) in OP_WEIGHTS {
        if roll < weight {
            return kind;
        }
        roll -= weight;
    }
    unreachable!("the roll is below the total of the weights")
}

/// The points an operation passes on this store. Hold points are the places
/// cancellations use: a local store writes everything but manifests around the
/// object store, so only the other stores see those calls.
fn hold_points(op: &RoundOp, store: StoreChoice, scale: &Scale) -> Vec<Incident> {
    let call = |method, area, nth, after| Incident::HoldCall {
        method,
        area,
        nth,
        after,
    };
    let mut points = Vec::new();
    if !matches!(op, RoundOp::Cleanup { .. }) {
        for after in [false, true] {
            points.push(call(Method::Put, Area::Manifests, 1, after));
        }
    }
    match op {
        RoundOp::Append { rows, .. } => {
            let fragments = (*rows as u64).div_ceil(scale.rows_per_file as u64);
            for nth in 1..=fragments {
                for at_begin in [true, false] {
                    points.push(Incident::HoldFragment { nth, at_begin });
                }
            }
        }
        RoundOp::CreateIndex(kind) => {
            for &stage in kind.stages() {
                for at_start in [true, false] {
                    points.push(Incident::HoldStage { stage, at_start });
                }
            }
        }
        // Its second manifest: between the reservation of fragment ids and the
        // rewrite.
        RoundOp::Compact => {
            for after in [false, true] {
                points.push(call(Method::Put, Area::Manifests, 2, after));
            }
        }
        // Between its look at the manifests and its first deletion. The first
        // listing of manifests only serves to work out the retention.
        RoundOp::Cleanup { .. } => points.push(call(Method::List, Area::Transactions, 1, false)),
        _ => {}
    }
    if store != StoreChoice::Local {
        let written = match op {
            RoundOp::Delete { .. } | RoundOp::ContendedDelete => Some(Area::Deletions),
            RoundOp::CreateIndex(_) | RoundOp::OptimizeIndices => Some(Area::Indices),
            RoundOp::Cleanup { .. } => None,
            _ => Some(Area::Data),
        };
        let transaction = (!matches!(op, RoundOp::Cleanup { .. })).then_some(Area::Transactions);
        for area in [written, transaction].into_iter().flatten() {
            for after in [false, true] {
                points.push(call(Method::Put, area, 1, after));
            }
        }
        // A read done before a writer's commit: its result goes stale then.
        if !matches!(
            op,
            RoundOp::Append { .. } | RoundOp::Cleanup { .. } | RoundOp::OptimizeIndices
        ) {
            points.push(call(Method::Get, Area::Data, 1, false));
        }
    }
    points
}

#[allow(clippy::too_many_arguments)]
pub(super) fn plan_rounds(
    rng: &mut PlanRng,
    seed: u64,
    store: StoreChoice,
    scale: &Scale,
    initial_rows: i64,
    rounds: usize,
    with_holds: bool,
    focus: &BTreeSet<&'static str>,
) -> RoundsPlan {
    let workers = rng.range(2..=4) as usize;
    let stable_row_ids = rng.chance(0.5);
    let shared_session = rng.chance(0.5);
    // A quarter of the seeds leave the key undeclared, where a duplicate is legal.
    let declared_key = rng.range(0..=3) != 3;
    let mut rows = RowsModel::new(initial_rows, workers);
    let mut indices = BTreeSet::new();
    let (pair_rounds, hole) = forced_pairs(seed, rounds, focus);
    // An insert of new keys over the hole is what its pair is for, and without a
    // declared key the round upserts existing keys instead.
    let declared_key = declared_key || hole.is_some_and(|(_, hidden)| hidden == "hole_keys");
    let free_rounds: Vec<usize> = (1..=rounds)
        .filter(|n| !pair_rounds.contains_key(n) && (hole.is_none() || *n != HOLE_ROUND))
        .collect();
    let new_keys_round =
        (declared_key && !free_rounds.is_empty()).then(|| *pick_from(rng, &free_rounds));
    let indexed_new_keys = declared_key && rng.chance(0.5);
    let planned = (1..=rounds)
        .map(|n| {
            let kind = match (pair_rounds.get(&n), hole) {
                (Some(&(in_flight, landed)), _) => RoundKind::Pair { in_flight, landed },
                (None, Some((in_flight, hidden))) if n == HOLE_ROUND => {
                    RoundKind::Hole { in_flight, hidden }
                }
                _ if new_keys_round == Some(n) => RoundKind::OnlyNewKeys {
                    indexed: indexed_new_keys,
                },
                _ => RoundKind::Random,
            };
            // Something for it to optimize: an index, and rows that it lacks.
            let index_ahead = matches!(pair_rounds.get(&(n + 1)), Some(("optimize_indices", _)));
            plan_round(
                rng,
                n,
                store,
                scale,
                &mut rows,
                &mut indices,
                declared_key,
                kind,
                index_ahead,
                with_holds,
            )
        })
        .collect();
    RoundsPlan {
        workers,
        stable_row_ids,
        shared_session,
        declared_key,
        rounds: planned,
    }
}

/// One operation, on the rows its worker holds at this point of the plan.
#[allow(clippy::too_many_arguments)]
fn plan_op(
    rng: &mut PlanRng,
    kind: &str,
    worker: usize,
    n: usize,
    scale: &Scale,
    rows: &RowsModel,
    indices: &mut BTreeSet<IndexKind>,
    index: IndexKind,
) -> RoundOp {
    let stride = rows.stride();
    let class = &rows.classes[worker];
    let next = rows.next_ids[worker];
    let added = |rng: &mut PlanRng| rng.range(scale.added_rows.0..=scale.added_rows.1) as i64;
    match kind {
        "delete" if class.len() > MIN_CLASS_ROWS + 1 => {
            let most = (class.len() / 5).min(class.len() - MIN_CLASS_ROWS).max(1);
            let len = rng.range(1..=most as u64) as usize;
            let (lo, hi) = pick_run(rng, class, len);
            RoundOp::Delete { lo, hi }
        }
        "merge_insert" => {
            let count = added(rng) / 4 + 1;
            // Half of them among rows that may share a fragment with other
            // classes, half at the tail of its own appends.
            let ids: Vec<i64> = class.keys().copied().collect();
            let first_id = if rng.chance(0.5) && !ids.is_empty() {
                *pick_from(rng, &ids)
            } else {
                (next - (count / 2) * stride).max(worker as i64)
            };
            RoundOp::MergeInsert {
                first_id,
                rows: count,
            }
        }
        "merge_insert_columns" => {
            let count = added(rng) / 4 + 1;
            let ids: Vec<i64> = class.keys().copied().collect();
            let start = rng.range(0..=ids.len() as u64 - 1) as usize;
            RoundOp::MergeInsertColumns {
                first_id: ids[start],
                rows: count,
            }
        }
        "update" => {
            let most = (class.len() / 3).max(1);
            let len = rng.range(1..=most as u64) as usize;
            let (lo, hi) = pick_run(rng, class, len);
            RoundOp::Update { lo, hi }
        }
        "contended_upsert" => RoundOp::ContendedUpsert,
        "contended_update" => RoundOp::ContendedUpdate,
        "contended_delete" => RoundOp::ContendedDelete,
        "counter_update" => RoundOp::CounterUpdate,
        "create_index" => {
            indices.insert(index);
            RoundOp::CreateIndex(index)
        }
        "stamp_index" => {
            indices.insert(IndexKind::BTreeStamp);
            RoundOp::CreateIndex(IndexKind::BTreeStamp)
        }
        "optimize_indices" if !indices.is_empty() => RoundOp::OptimizeIndices,
        "compact" => RoundOp::Compact,
        // Its retention is set below, once the round's commits are known.
        // The first rounds leave too few versions for it to matter.
        "cleanup" if n >= 3 => RoundOp::Cleanup { retain: 0 },
        _ => RoundOp::Append {
            first_id: next,
            rows: added(rng),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_round(
    rng: &mut PlanRng,
    n: usize,
    store: StoreChoice,
    scale: &Scale,
    rows: &mut RowsModel,
    indices: &mut BTreeSet<IndexKind>,
    declared_key: bool,
    kind: RoundKind,
    index_ahead: bool,
    with_holds: bool,
) -> Round {
    let workers = rows.workers;
    let hole = matches!(kind, RoundKind::Hole { .. });
    let theme = match kind {
        RoundKind::OnlyNewKeys { .. } => Some("contended_upsert"),
        RoundKind::Pair { .. } => Some("pair"),
        RoundKind::Hole { .. } => Some("hole"),
        RoundKind::Random => rng.chance(0.3).then(|| *pick_from(rng, &THEMES)),
    };
    let index_kinds = IndexKind::ALL;
    // A themed round races for one name, whichever index a plain round picks.
    let theme_index = *pick_from(rng, &index_kinds);
    let mut queues: Vec<Vec<RoundOp>> = vec![Vec::new(); workers];
    for (worker, queue) in queues.iter_mut().enumerate() {
        // The first worker's writes over a hole come below.
        if hole && worker == 0 {
            continue;
        }
        let count = match kind {
            RoundKind::Pair { .. } | RoundKind::Hole { .. } => 1,
            _ if theme.is_some() => rng.range(1..=2),
            _ => rng.range(1..=3),
        };
        for _ in 0..count {
            let op_kind = match (kind, theme) {
                (RoundKind::Pair { landed, .. }, _) if worker == 0 => landed,
                (RoundKind::Pair { in_flight, .. } | RoundKind::Hole { in_flight, .. }, _) => {
                    in_flight
                }
                (_, Some("stamp_index")) if worker == 0 => "stamp_index",
                (_, Some("stamp_index")) => "merge_insert_columns",
                (_, Some("contended_mixed")) => *pick_from(rng, &CONTENDED_KINDS),
                (_, Some(theme)) => theme,
                (_, None) => random_kind(rng),
            };
            let index = match theme {
                Some(_) => theme_index,
                None => *pick_from(rng, &index_kinds),
            };
            let op = plan_op(rng, op_kind, worker, n, scale, rows, indices, index);
            rows.apply(worker, &op, 0, &[]);
            queue.push(op);
        }
    }
    let indexed_keys = kind == RoundKind::OnlyNewKeys { indexed: true };
    if indexed_keys {
        let op = plan_op(
            rng,
            "create_index",
            0,
            n,
            scale,
            rows,
            indices,
            IndexKind::BTreeId,
        );
        queues[0].insert(0, op);
    }
    // Rows that no compaction has seen yet, whatever the rounds before compacted:
    // a compaction that finds nothing to do commits nothing to meet.
    if let RoundKind::Pair {
        landed: "compact", ..
    } = kind
    {
        let op = plan_op(rng, "append", 0, n, scale, rows, indices, theme_index);
        rows.apply(0, &op, 0, &[]);
        queues[0].insert(0, op);
    }
    if let RoundKind::Hole { hidden, .. } = kind {
        let class = &rows.classes[0];
        let write = match hidden {
            // Every fragment holds rows of the class in the first round.
            "hole_rows" => match (class.keys().next(), class.keys().next_back()) {
                (Some(lo), Some(last)) => RoundOp::Update {
                    lo: *lo,
                    hi: last + 1,
                },
                _ => plan_op(rng, "append", 0, n, scale, rows, indices, theme_index),
            },
            "hole_compact" => RoundOp::Compact,
            "hole_keys" => RoundOp::ContendedUpsert,
            "hole_stamp" => plan_op(
                rng,
                "merge_insert_columns",
                0,
                n,
                scale,
                rows,
                indices,
                theme_index,
            ),
            _ => unreachable!("a hidden write of `HOLES`"),
        };
        for op in [write, RoundOp::Reserve, RoundOp::Cleanup { retain: 1 }] {
            rows.apply(0, &op, 0, &[]);
            queues[0].push(op);
        }
    }
    if index_ahead {
        let index = *pick_from(rng, &index_kinds);
        for op_kind in ["create_index", "append"] {
            let op = plan_op(rng, op_kind, 0, n, scale, rows, indices, index);
            rows.apply(0, &op, 0, &[]);
            queues[0].push(op);
        }
    }
    // Enough to keep every version of this round and the one it starts from, so a
    // write held across a cleanup never takes the number of a version it removed
    // (#3718). Only the round with the hole removes a version inside a writer's
    // window of conflicts, and it plans no holds: see `HOLES`.
    let retain = queues
        .iter()
        .flatten()
        .map(|op| {
            op.max_commits() + usize::from(matches!(op, RoundOp::Compact)) * PARALLEL_ATTEMPTS
        })
        .sum::<usize>()
        + 2;
    let keys = plan_keys(
        rng,
        rows,
        &mut queues,
        declared_key,
        matches!(
            kind,
            RoundKind::OnlyNewKeys { .. }
                | RoundKind::Hole {
                    hidden: "hole_keys",
                    ..
                }
        ),
    );
    // On a pair or a hole the others wait for the first worker's last operation.
    let first_last = queues[0].len().saturating_sub(1);
    // Any merge of the queues keeps each worker's own order.
    let mut cursors = vec![0; workers];
    let mut ops: Vec<PlannedOp> = Vec::new();
    loop {
        let waiting: Vec<usize> = (0..workers)
            .filter(|worker| cursors[*worker] < queues[*worker].len())
            .collect();
        if waiting.is_empty() {
            break;
        }
        // What a pair's others commit over comes first: only a lower rank can be
        // waited for.
        let worker = match kind {
            RoundKind::Pair { .. } | RoundKind::Hole { .. } => waiting[0],
            _ if indexed_keys && ops.is_empty() => 0,
            _ => *pick_from(rng, &waiting),
        };
        let mut op = queues[worker][cursors[worker]];
        cursors[worker] += 1;
        if let RoundOp::Cleanup { .. } = op
            && !hole
        {
            op = RoundOp::Cleanup { retain };
        }
        let rank = ops.len();
        let others: Vec<usize> = ops
            .iter()
            .filter(|other| other.worker != worker)
            .map(|other| other.rank)
            .collect();
        let mut hold = None;
        let roll = rng.chance(0.6);
        // Without a catch up, the handle stands on the version the round starts from
        // for as long as the other takes.
        let mut catch_up = rng.chance(0.7);
        // The racers start once the index on the key is there, and see it.
        let mut start_after = (indexed_keys && worker != 0 && cursors[worker] == 1).then_some(0);
        catch_up |= start_after.is_some();
        let mut staged = false;
        if matches!(kind, RoundKind::Pair { .. }) {
            if worker != 0 {
                catch_up = false;
                hold = with_holds.then(|| Hold {
                    point: commit_point(&op),
                    until: first_last,
                    arrival: false,
                });
            } else if rank == first_last && matches!(op, RoundOp::Cleanup { .. }) {
                // The others write their files before their commit point. Started
                // once the first of them is there, the cleanup runs while files
                // nobody has committed yet exist, and must leave them alone. Lance
                // leaves new data and deletion files out of its listing by their
                // time, and keeps new index files as files in progress.
                hold = with_holds.then(|| Hold {
                    point: Incident::HoldCall {
                        method: Method::List,
                        area: Area::Manifests,
                        nth: 1,
                        after: false,
                    },
                    until: first_last + 1,
                    arrival: true,
                });
            }
        } else if hole {
            // Prepared on the first version before anyone writes, committed once
            // the cleanup is over. No hold: see `HOLES`.
            catch_up = worker == 0;
            if worker != 0 {
                start_after = Some(first_last);
                staged = true;
            }
        } else if with_holds && roll && !others.is_empty() {
            let points = hold_points(&op, store, scale);
            if !points.is_empty() {
                hold = Some(Hold {
                    point: pick_from(rng, &points).clone(),
                    until: *pick_from(rng, &others),
                    arrival: false,
                });
            }
        }
        ops.push(PlannedOp {
            rank,
            worker,
            number: n as i64 * NUMBERS_PER_ROUND + rank as i64 + 1,
            catch_up,
            op,
            hold,
            start_after,
            staged,
        });
    }
    let cleanup_rank = ops
        .iter()
        .find(|op| matches!(op.op, RoundOp::Cleanup { .. }))
        .map(|op| op.rank);
    // A cleanup in a pair waits for one of the others, who may wait for a read
    // held until the cleanup is over: a file's metadata loads once per session.
    let cleanup_waits = matches!(
        kind,
        RoundKind::Pair {
            landed: "cleanup",
            ..
        }
    );
    let readers = (0..rng.range(1..=2))
        .map(|_| {
            // Files go while that cleanup runs, so these read what it leaves.
            if hole {
                let kind = if rng.chance(0.5) {
                    ReaderKind::FreshOpen
                } else {
                    ReaderKind::CheckoutLatest
                };
                return PlannedReader {
                    kind,
                    hold: None,
                    start_after: cleanup_rank,
                };
            }
            let kind = match (cleanup_rank.is_some() && rng.chance(0.7), rng.range(0..=9)) {
                (true, _) => ReaderKind::OldSnapshot,
                (_, 0..=3) => ReaderKind::FreshOpen,
                (_, 4..=7) => ReaderKind::CheckoutLatest,
                (_, 8) => ReaderKind::Pinned,
                _ => ReaderKind::OldSnapshot,
            };
            // Stopped in the middle of its read until the cleanup is over.
            let hold = match (kind, cleanup_rank) {
                (ReaderKind::OldSnapshot, Some(until))
                    if with_holds && store != StoreChoice::Local && !cleanup_waits =>
                {
                    Some(Hold {
                        point: Incident::HoldCall {
                            method: Method::Get,
                            area: Area::Data,
                            nth: 1,
                            after: false,
                        },
                        until,
                        arrival: false,
                    })
                }
                _ => None,
            };
            PlannedReader {
                kind,
                hold,
                start_after: None,
            }
        })
        .collect();
    Round {
        theme,
        keys,
        ops,
        readers,
        hole: match kind {
            RoundKind::Hole { hidden, .. } => Some(hidden),
            _ => None,
        },
    }
}

impl RoundsPlan {
    pub(super) fn render(&self) -> String {
        let mut text = format!(
            "workers={} stable_row_ids={} shared_session={} declared_key={}\n",
            self.workers, self.stable_row_ids, self.shared_session, self.declared_key,
        );
        for (index, round) in self.rounds.iter().enumerate() {
            writeln!(
                text,
                "round={} theme={} keys={:?}",
                index + 1,
                round.theme.unwrap_or("none"),
                round.keys
            )
            .unwrap();
            for op in &round.ops {
                let hold = op.hold.as_ref().map_or_else(
                    || "none".to_string(),
                    |hold| {
                        let arrival = if hold.arrival { ":arrival" } else { "" };
                        format!("{}:until={}{arrival}", hold.point, hold.until)
                    },
                );
                let start_after = op
                    .start_after
                    .map_or_else(|| "none".to_string(), |rank| rank.to_string());
                writeln!(
                    text,
                    "  rank={} worker={} number={} catch_up={} op={} hold={hold} start_after={start_after} staged={}",
                    op.rank, op.worker, op.number, op.catch_up, op.op, op.staged
                )
                .unwrap();
            }
            for reader in &round.readers {
                let hold = reader.hold.as_ref().map_or_else(
                    || "none".to_string(),
                    |hold| {
                        let arrival = if hold.arrival { ":arrival" } else { "" };
                        format!("{}:until={}{arrival}", hold.point, hold.until)
                    },
                );
                let start_after = reader
                    .start_after
                    .map_or_else(|| "none".to_string(), |rank| rank.to_string());
                writeln!(
                    text,
                    "  reader={} hold={hold} start_after={start_after}",
                    reader.kind.name()
                )
                .unwrap();
            }
        }
        text
    }

    /// One round of two whole-class updates, whose effect is known: the defect a
    /// self-test injects must stand out, and a random plan need not contain what
    /// that defect acts on.
    pub(super) fn for_self_test(&mut self, test: SelfTest, initial_rows: i64) {
        let update = |worker: usize, rank: usize| PlannedOp {
            rank,
            worker,
            number: NUMBERS_PER_ROUND + rank as i64 + 1,
            catch_up: true,
            op: RoundOp::Update {
                lo: 0,
                hi: initial_rows,
            },
            hold: None,
            start_after: None,
            staged: false,
        };
        let mut ops = vec![update(0, 0), update(1, 1)];
        // It waits for its own end, which a gate without a limit never sees.
        if test == SelfTest::GateHang {
            ops[1].hold = Some(Hold {
                point: Incident::HoldCall {
                    method: Method::Put,
                    area: Area::Manifests,
                    nth: 1,
                    after: false,
                },
                until: 1,
                arrival: false,
            });
        }
        let kind = match test {
            SelfTest::SnapshotChanged => ReaderKind::Pinned,
            _ => ReaderKind::FreshOpen,
        };
        self.stable_row_ids |= test == SelfTest::Lineage;
        self.rounds = vec![Round {
            theme: Some("self_test"),
            keys: Vec::new(),
            ops,
            readers: vec![PlannedReader {
                kind,
                hold: None,
                start_after: None,
            }],
            hole: None,
        }];
    }

    pub(super) fn start(&self, initial_rows: i64, reads: Journal) -> RoundsState {
        let rows = RowsModel::new(initial_rows, self.workers);
        RoundsState {
            stamps: rows.all_rows(),
            rows,
            stamps_ever: BTreeSet::new(),
            digests: BTreeMap::new(),
            lineage: LineageTrack::default(),
            reads: Arc::new(reads),
        }
    }

    fn schema(&self) -> SchemaRef {
        schema(self.declared_key)
    }

    pub(super) fn initial_source(
        &self,
        data_seed: u64,
        initial_rows: i64,
        scale: Scale,
    ) -> impl RecordBatchReader + Send + 'static {
        let ids: Vec<i64> = (0..initial_rows).collect();
        source_of(full_batch(&self.schema(), data_seed, &scale, &ids, 0))
    }
}

/// The columns of the rows, with `gen`, which tells the operations apart.
pub(super) fn schema(declared_key: bool) -> SchemaRef {
    let mut id = Field::new("id", DataType::Int64, false);
    if declared_key {
        id = id.with_metadata(HashMap::from([(
            LANCE_UNENFORCED_PRIMARY_KEY.to_string(),
            "true".to_string(),
        )]));
    }
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![
        id,
        Field::new("category", DataType::Int32, false),
        Field::new("vec", DataType::FixedSizeList(item, VECTOR_DIM), false),
        Field::new("text", DataType::Utf8, false),
        Field::new("gen", DataType::Int64, false),
    ]))
}

pub(super) fn source_of(batch: RecordBatch) -> impl RecordBatchReader + Send + 'static {
    let schema = batch.schema();
    RecordBatchIterator::new(vec![Ok(batch)], schema)
}

/// Full rows, every column a function of the id but `gen`.
pub(super) fn full_batch(
    schema: &SchemaRef,
    data_seed: u64,
    scale: &Scale,
    ids: &[i64],
    stamp: i64,
) -> RecordBatch {
    let vectors = Float32Array::from_iter_values(ids.iter().flat_map(|id| vector(data_seed, *id)));
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ids.to_vec())),
        Arc::new(Int32Array::from_iter_values(
            ids.iter().map(|id| category(*id)),
        )),
        Arc::new(
            FixedSizeListArray::try_new(item, VECTOR_DIM, Arc::new(vectors), None)
                .expect("build the vectors"),
        ),
        Arc::new(StringArray::from_iter_values(
            ids.iter().map(|id| text(*id, scale)),
        )),
        Arc::new(Int64Array::from(vec![stamp; ids.len()])),
    ];
    RecordBatch::try_new(schema.clone(), columns).expect("build a batch")
}

/// What `concurrency` knows between its rounds.
pub(super) struct RoundsState {
    rows: RowsModel,
    /// `gen` of every row, once a round's checks have confirmed it.
    stamps: ClassRows,
    /// Every value of `gen` a version has held. An index that went stale can
    /// still answer one that no row holds any more.
    stamps_ever: BTreeSet<i64>,
    /// What each version held when the checks first read it.
    digests: BTreeMap<u64, u64>,
    lineage: LineageTrack,
    reads: Arc<Journal>,
}

impl RoundsState {
    pub(super) fn stamps(&self) -> &ClassRows {
        &self.stamps
    }

    pub(super) fn stamps_ever(&self) -> &BTreeSet<i64> {
        &self.stamps_ever
    }
}

/// Whether any segment of an index covers a fragment of this version. An index
/// that covers none leaves query planning, whatever its name.
pub(super) fn covers_a_fragment(ds: &Dataset, indices: &[IndexMetadata], name: &str) -> bool {
    let live: roaring::RoaringBitmap = ds.fragments().iter().map(|f| f.id as u32).collect();
    indices
        .iter()
        .filter(|index| index.name == name)
        .filter_map(|index| index.fragment_bitmap.as_ref())
        .any(|bitmap| !(bitmap & &live).is_empty())
}

/// What a rows's lineage columns hold. A fragment without version metadata leaves
/// the versions empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lineage {
    row_id: u64,
    created: Option<u64>,
    updated: Option<u64>,
}

/// The rows of one version.
struct Image {
    version: u64,
    rows: ClassRows,
    lineage: Option<BTreeMap<i64, Lineage>>,
}

impl Image {
    /// Equal rows give equal digests, in whatever order they were read.
    fn digest(&self) -> u64 {
        self.rows
            .iter()
            .flat_map(|(id, stamp)| id.to_le_bytes().into_iter().chain(stamp.to_le_bytes()))
            .fold(0xCBF2_9CE4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
            })
    }
}

/// How rows are read and what they must hold whatever the version.
#[derive(Clone, Copy)]
struct Oracle {
    scale: Scale,
    data_seed: u64,
    stable_row_ids: bool,
}

impl Oracle {
    async fn scan(&self, ds: &Dataset, full: bool) -> lance::Result<RecordBatch> {
        let mut columns = vec!["id", "gen"];
        if full {
            columns.extend(["category", "vec", "text"]);
            if self.stable_row_ids {
                columns.extend([ROW_ID, ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION]);
            }
        }
        let mut scan = ds.scan();
        scan.project(&columns)?;
        scan.try_into_batch().await
    }

    /// Reads every row of a handle's version once and checks what is a function
    /// of the id.
    async fn image(
        &self,
        ds: &Dataset,
        n: usize,
        what: &str,
        full: bool,
    ) -> Result<Result<Image, Error>, Failure> {
        let batch = match self.scan(ds, full).await {
            Ok(batch) => batch,
            Err(error) => return Ok(Err(error)),
        };
        let version = ds.version().version;
        let ids = int64_column(&batch, "id");
        let stamps = int64_column(&batch, "gen");
        let rows: ClassRows = ids.iter().copied().zip(stamps).collect();
        if rows.len() != ids.len() {
            let mut seen = BTreeSet::new();
            let twice: Vec<i64> = ids
                .iter()
                .copied()
                .filter(|id| !seen.insert(*id))
                .take(5)
                .collect();
            return Err(fail(
                "duplicate_key",
                n,
                format_args!(
                    "{what}: version {version} holds {} rows but {} distinct ids; twice: {twice:?}",
                    ids.len(),
                    rows.len()
                ),
            ));
        }
        if full {
            self.check_columns(&batch, &ids, n, what, version)?;
        }
        let lineage = (full && self.stable_row_ids).then(|| lineage_of(&batch, &ids));
        let image = Image {
            version,
            rows,
            lineage,
        };
        check_lineage_of_a_version(&image, n, what)?;
        Ok(Ok(image))
    }

    fn check_columns(
        &self,
        batch: &RecordBatch,
        ids: &[i64],
        n: usize,
        what: &str,
        version: u64,
    ) -> Result<(), Failure> {
        let categories = batch
            .column_by_name("category")
            .expect("a category column")
            .as_primitive::<Int32Type>();
        let vectors = batch
            .column_by_name("vec")
            .expect("a vec column")
            .as_fixed_size_list();
        let texts = batch
            .column_by_name("text")
            .expect("a text column")
            .as_string::<i32>();
        for (row, id) in ids.iter().enumerate() {
            let wrong = if categories.value(row) != category(*id) {
                Some("category")
            } else if texts.value(row) != text(*id, &self.scale) {
                Some("text")
            } else if vectors
                .value(row)
                .as_primitive::<Float32Type>()
                .values()
                .as_ref()
                != vector(self.data_seed, *id).as_slice()
            {
                Some("vec")
            } else {
                None
            };
            if let Some(column) = wrong {
                return Err(fail(
                    "content",
                    n,
                    format_args!("{what}: version {version}: row {id} holds a wrong {column}"),
                ));
            }
        }
        Ok(())
    }

    /// The indices this very version names must answer what its rows hold. A reader
    /// asks about a few values of `gen`, to stay busy reading rather than querying.
    async fn check_indices(
        &self,
        ds: &Dataset,
        n: usize,
        image: &Image,
        few_stamps: bool,
    ) -> Result<(), Failure> {
        let indices = ds
            .load_indices()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let names: BTreeSet<&str> = indices.iter().map(|index| index.name.as_str()).collect();
        let live: BTreeSet<i64> = image.rows.keys().copied().collect();
        let answers = Answers {
            live: &live,
            next_id: live.last().map_or(0, |last| last + 1),
            stamps: Some(&image.rows),
            stamps_ever: None,
            few_stamps,
            scale: self.scale,
            data_seed: self.data_seed,
            check_plan: false,
            salt: image.version as usize,
        };
        for kind in IndexKind::ALL {
            if names.contains(kind.name()) {
                Box::pin(check_index(ds, n, kind, &answers)).await?;
            }
        }
        Ok(())
    }
}

fn lineage_of(batch: &RecordBatch, ids: &[i64]) -> BTreeMap<i64, Lineage> {
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("no {name} column"))
            .as_primitive::<UInt64Type>()
            .clone()
    };
    let (row_ids, created, updated) = (
        column(ROW_ID),
        column(ROW_CREATED_AT_VERSION),
        column(ROW_LAST_UPDATED_AT_VERSION),
    );
    let optional = |array: &arrow_array::UInt64Array, row: usize| {
        (!array.is_null(row)).then(|| array.value(row))
    };
    ids.iter()
        .enumerate()
        .map(|(row, id)| {
            (
                *id,
                Lineage {
                    row_id: row_ids.value(row),
                    created: optional(&created, row),
                    updated: optional(&updated, row),
                },
            )
        })
        .collect()
}

/// `docs/src/format/table/row_id_lineage.md`: a row id is unique, and no row
/// claims a version later than the one it is read from.
fn check_lineage_of_a_version(image: &Image, n: usize, what: &str) -> Result<(), Failure> {
    let Some(lineage) = &image.lineage else {
        return Ok(());
    };
    let mut owners = BTreeMap::new();
    for (id, row) in lineage {
        if let Some(other) = owners.insert(row.row_id, *id) {
            return Err(fail(
                "lineage",
                n,
                format_args!(
                    "{what}: version {}: rows {other} and {id} both have the row id {}",
                    image.version, row.row_id
                ),
            ));
        }
        for stamp in [row.created, row.updated].into_iter().flatten() {
            if stamp > image.version {
                return Err(fail(
                    "lineage",
                    n,
                    format_args!(
                        "{what}: row {id} claims version {stamp} in version {}: {row:?}",
                        image.version
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// What a row's lineage must do from one version to the next: its row id and the
/// version that created it stay while it lives, and the version it was last
/// updated at is the one where `gen` changed. When a cleanup took the versions in
/// between, that is any of them.
#[derive(Default)]
struct LineageTrack {
    last: BTreeMap<i64, (i64, Lineage)>,
    owners: BTreeMap<u64, i64>,
    /// The version `last` is of.
    seen: u64,
}

impl LineageTrack {
    /// Makes the track remember a row id that the row never had.
    fn tamper(&mut self) {
        if let Some((_, row)) = self.last.values_mut().next() {
            row.row_id ^= 1 << 40;
        }
    }

    fn start(&mut self, image: &Image) {
        let Some(lineage) = &image.lineage else {
            return;
        };
        self.last = lineage
            .iter()
            .map(|(id, row)| (*id, (image.rows[id], *row)))
            .collect();
        self.owners
            .extend(lineage.iter().map(|(id, row)| (row.row_id, *id)));
        self.seen = image.version;
    }

    fn advance(&mut self, image: &Image, n: usize) -> Result<(), Failure> {
        let Some(lineage) = &image.lineage else {
            return Ok(());
        };
        let version = image.version;
        let seen = self.seen;
        let since_seen = |at: Option<u64>| at.is_some_and(|at| seen < at && at <= version);
        let wrong = |detail: String| {
            Err(fail(
                "lineage",
                n,
                format_args!("version {version}: {detail}"),
            ))
        };
        for (id, row) in lineage {
            let stamp = image.rows[id];
            if let Some(owner) = self.owners.insert(row.row_id, *id)
                && owner != *id
            {
                return wrong(format!(
                    "row {id} took the row id {} of row {owner}",
                    row.row_id
                ));
            }
            match self.last.get(id) {
                Some((last_stamp, last)) => {
                    if row.row_id != last.row_id {
                        return wrong(format!(
                            "row {id} changed its row id from {last:?} to {row:?}"
                        ));
                    }
                    if row.created != last.created {
                        return wrong(format!(
                            "row {id} changed its creation from {last:?} to {row:?}"
                        ));
                    }
                    if row.updated.is_none() {
                        continue;
                    }
                    if stamp != *last_stamp && !since_seen(row.updated) {
                        return wrong(format!(
                            "gen of row {id} went from {last_stamp} to {stamp}, but it says {row:?}"
                        ));
                    }
                    if stamp == *last_stamp && row.updated != last.updated {
                        return wrong(format!(
                            "gen of row {id} stayed {stamp}, but it went from {last:?} to {row:?}"
                        ));
                    }
                }
                None => {
                    if row.created.is_some_and(|_| !since_seen(row.created))
                        || row.updated.is_some_and(|_| !since_seen(row.updated))
                    {
                        return wrong(format!("the new row {id} says {row:?}"));
                    }
                }
            }
        }
        self.last = lineage
            .iter()
            .map(|(id, row)| (*id, (image.rows[id], *row)))
            .collect();
        self.seen = version;
        Ok(())
    }
}

/// What the shared upserts of a round have done, as a snapshot shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Contended {
    Untouched,
    /// Every key exists and holds what this upsert or update wrote.
    Applied(i64),
    /// No key exists, after a delete.
    Deleted,
}

/// Where a snapshot stands in the round.
struct Placement {
    /// Per worker: how many of its operations the snapshot may include. Those
    /// that leave its rows as they are make this a range.
    prefixes: Vec<RangeInclusive<usize>>,
    /// The same, every place the rows match: an append and a delete of exactly
    /// those rows can bring a state back with others in between.
    places: Vec<Vec<usize>>,
    contended: Contended,
    increments: i64,
}

/// What each worker's rows hold after each of its operations, and what the
/// shared classes may hold.
struct RoundModel {
    workers: usize,
    stride: i64,
    states: Vec<Vec<ClassRows>>,
    keys: BTreeSet<i64>,
    contended_before: ClassRows,
    upserts: BTreeSet<i64>,
    updates: BTreeSet<i64>,
    deletes: BTreeSet<i64>,
    counters_before: ClassRows,
    counter_ops: i64,
}

fn differences(expected: &ClassRows, actual: &ClassRows) -> (usize, String) {
    let mut different = Vec::new();
    for (id, stamp) in expected {
        if actual.get(id) != Some(stamp) {
            different.push(format!("{id}:{stamp}->{:?}", actual.get(id)));
        }
    }
    for (id, stamp) in actual {
        if !expected.contains_key(id) {
            different.push(format!("{id}:none->{stamp}"));
        }
    }
    let count = different.len();
    different.truncate(6);
    (count, different.join(","))
}

impl RoundModel {
    /// An operation's report of how many rows it touched, in what it returns to
    /// the caller. A retry must not leave the count of an earlier try behind.
    fn check_counts(
        &self,
        op: &PlannedOp,
        position: usize,
        counts: &RowCounts,
        n: usize,
    ) -> Result<(), Failure> {
        let before = &self.states[op.worker][position];
        let keys = self.keys.len() as u64;
        let known = |first_id: i64, rows: i64| {
            id_run(first_id, rows, self.stride)
                .filter(|id| before.contains_key(id))
                .count() as u64
        };
        let right = match op.op {
            RoundOp::Delete { lo, hi } => counts.deleted == before.range(lo..hi).count() as u64,
            RoundOp::Update { lo, hi } => counts.updated == before.range(lo..hi).count() as u64,
            RoundOp::MergeInsert { first_id, rows } => {
                counts.updated == known(first_id, rows)
                    && counts.inserted == rows as u64 - known(first_id, rows)
            }
            RoundOp::MergeInsertColumns { first_id, rows } => {
                counts.updated == known(first_id, rows) && counts.inserted == 0
            }
            RoundOp::ContendedUpsert => counts.updated + counts.inserted == keys,
            RoundOp::ContendedUpdate => counts.updated == 0 || counts.updated == keys,
            RoundOp::ContendedDelete => counts.deleted == 0 || counts.deleted == keys,
            RoundOp::CounterUpdate => counts.updated == self.counters_before.len() as u64,
            _ => true,
        };
        if right {
            return Ok(());
        }
        Err(fail(
            "content",
            n,
            format_args!(
                "{} of worker {} reports {counts:?}, with {keys} keys and {} of its rows before it",
                op.op,
                op.worker,
                before.len()
            ),
        ))
    }

    fn new(rows: &RowsModel, round: &Round) -> Self {
        let mut states = Vec::with_capacity(rows.workers);
        for worker in 0..rows.workers {
            let mut model = rows.clone();
            let mut sequence = vec![model.classes[worker].clone()];
            for op in round.ops_of(worker) {
                if op.op.is_row_op() {
                    model.apply(worker, &op.op, op.number, &[]);
                }
                sequence.push(model.classes[worker].clone());
            }
            states.push(sequence);
        }
        let numbers = |kind: RoundOp| -> BTreeSet<i64> {
            round
                .ops
                .iter()
                .filter(|op| op.op == kind)
                .map(|op| op.number)
                .collect()
        };
        Self {
            workers: rows.workers,
            stride: rows.stride(),
            states,
            keys: round.keys.iter().copied().collect(),
            contended_before: rows.classes[rows.contended()].clone(),
            upserts: numbers(RoundOp::ContendedUpsert),
            updates: numbers(RoundOp::ContendedUpdate),
            deletes: numbers(RoundOp::ContendedDelete),
            counters_before: rows.classes[rows.counters()].clone(),
            counter_ops: round
                .ops
                .iter()
                .filter(|op| op.op == RoundOp::CounterUpdate)
                .count() as i64,
        }
    }

    /// Finds where a snapshot stands, or fails if it stands nowhere: a state no
    /// order of the round's operations can produce.
    fn place(&self, image: &Image, n: usize, what: &str) -> Result<Placement, Failure> {
        let version = image.version;
        let torn = |detail: String| {
            fail(
                "torn_snapshot",
                n,
                format_args!("{what}: version {version}: {detail}"),
            )
        };
        let mut classes = vec![ClassRows::new(); self.workers + 2];
        for (id, stamp) in &image.rows {
            classes[(id % self.stride) as usize].insert(*id, *stamp);
        }
        let mut prefixes = Vec::with_capacity(self.workers);
        let mut all_places = Vec::with_capacity(self.workers);
        for (worker, states) in self.states.iter().enumerate() {
            let places: Vec<usize> = (0..states.len())
                .filter(|place| states[*place] == classes[worker])
                .collect();
            let (Some(first), Some(last)) = (places.first(), places.last()) else {
                let nearest = states
                    .iter()
                    .enumerate()
                    .map(|(place, state)| (differences(state, &classes[worker]), place))
                    .min_by_key(|((count, _), place)| (*count, *place))
                    .expect("a worker has a state before its first operation");
                return Err(torn(format!(
                    "the rows of worker {worker} match none of its {} prefixes; the nearest is {}, {} rows off: {}",
                    states.len(),
                    nearest.1,
                    nearest.0.0,
                    nearest.0.1
                )));
            };
            prefixes.push(*first..=*last);
            all_places.push(places);
        }
        let contended = self.place_contended(&classes[self.workers], &torn)?;
        let increments = self.place_counters(&classes[self.workers + 1], n, what, version)?;
        Ok(Placement {
            prefixes,
            places: all_places,
            contended,
            increments,
        })
    }

    fn place_contended(
        &self,
        rows: &ClassRows,
        torn: &impl Fn(String) -> Failure,
    ) -> Result<Contended, Failure> {
        let outside = |class: &ClassRows| -> ClassRows {
            class
                .iter()
                .filter(|(id, _)| !self.keys.contains(id))
                .map(|(id, stamp)| (*id, *stamp))
                .collect()
        };
        let (count, detail) = differences(&outside(&self.contended_before), &outside(rows));
        if count > 0 {
            return Err(torn(format!(
                "{count} rows of the shared class outside the round's keys changed: {detail}"
            )));
        }
        if self
            .keys
            .iter()
            .all(|key| rows.get(key) == self.contended_before.get(key))
        {
            return Ok(Contended::Untouched);
        }
        let values: BTreeSet<Option<i64>> =
            self.keys.iter().map(|key| rows.get(key).copied()).collect();
        match values.into_iter().collect::<Vec<_>>().as_slice() {
            [Some(number)] if self.upserts.contains(number) || self.updates.contains(number) => {
                Ok(Contended::Applied(*number))
            }
            [None] if !self.deletes.is_empty() => Ok(Contended::Deleted),
            many => Err(torn(format!(
                "the shared keys hold {many:?}, from the upserts {:?}, updates {:?} and deletes {:?}",
                self.upserts, self.updates, self.deletes
            ))),
        }
    }

    fn place_counters(
        &self,
        rows: &ClassRows,
        n: usize,
        what: &str,
        version: u64,
    ) -> Result<i64, Failure> {
        let lost = |detail: String| {
            fail(
                "lost_write",
                n,
                format_args!("{what}: version {version}: {detail}"),
            )
        };
        if rows.len() != self.counters_before.len() {
            return Err(lost(format!(
                "{} counters where there were {}",
                rows.len(),
                self.counters_before.len()
            )));
        }
        let mut increments = BTreeSet::new();
        for (id, stamp) in rows {
            let Some(before) = self.counters_before.get(id) else {
                return Err(lost(format!("a counter {id} that never existed")));
            };
            increments.insert(stamp - before);
        }
        match increments.into_iter().collect::<Vec<_>>().as_slice() {
            [] => Ok(0),
            [one] if (0..=self.counter_ops).contains(one) => Ok(*one),
            [one] => Err(lost(format!(
                "counters moved by {one}, with {} increments planned",
                self.counter_ops
            ))),
            many => Err(fail(
                "torn_snapshot",
                n,
                format_args!("{what}: version {version}: counters moved by {many:?} at once"),
            )),
        }
    }

    /// A version is no older than what finished before it was opened, and holds
    /// nothing that had not started by the end of its read.
    fn check_bounds(
        &self,
        version: u64,
        placed: &Placement,
        lower: &ClockView,
        upper: &ClockView,
        n: usize,
        what: &str,
    ) -> Result<(), Failure> {
        let stale = |detail: String| Err(fail("stale_read", n, format_args!("{what}: {detail}")));
        let early =
            |detail: String| Err(fail("torn_snapshot", n, format_args!("{what}: {detail}")));
        if version < lower.version {
            return stale(format!(
                "version {version}, though an operation had returned with version {}",
                lower.version
            ));
        }
        for worker in 0..self.workers {
            let prefix = &placed.prefixes[worker];
            if lower.ops[worker] > *prefix.end() {
                return stale(format!(
                    "worker {worker} had finished {} operations, but the rows are as of {prefix:?}",
                    lower.ops[worker]
                ));
            }
            if *prefix.start() > upper.ops[worker] {
                return early(format!(
                    "worker {worker} had started {} operations, but the rows are as of {prefix:?}",
                    upper.ops[worker]
                ));
            }
        }
        match placed.contended {
            Contended::Untouched if !lower.contended.is_empty() => {
                return stale(format!(
                    "the operations {:?} had finished, but the keys are as they were",
                    lower.contended
                ));
            }
            Contended::Applied(number) if !upper.contended.contains(&number) => {
                return early(format!(
                    "the keys hold operation {number}, but only {:?} had started",
                    upper.contended
                ));
            }
            Contended::Deleted if self.deletes.is_disjoint(&upper.contended) => {
                return early(format!(
                    "the keys are gone, but only {:?} had started",
                    upper.contended
                ));
            }
            _ => {}
        }
        if placed.increments < lower.increments {
            return stale(format!(
                "{} increments had finished, but the counters moved by {}",
                lower.increments, placed.increments
            ));
        }
        if placed.increments > upper.increments {
            return early(format!(
                "{} increments had started, but the counters moved by {}",
                upper.increments, placed.increments
            ));
        }
        Ok(())
    }
}

/// When things happened in a round.
struct Clocks {
    tick: Arc<AtomicU64>,
    started: Vec<AtomicUsize>,
    completed: Vec<AtomicUsize>,
    contended_started: Mutex<BTreeSet<i64>>,
    contended_completed: Mutex<BTreeSet<i64>>,
    increments_started: AtomicI64,
    increments_completed: AtomicI64,
    /// The newest version any operation has returned with.
    newest_returned: AtomicU64,
    spans: Vec<Span>,
}

/// An operation on the logical clock: zero means not yet.
#[derive(Default)]
struct Span {
    begin: AtomicU64,
    commit: AtomicU64,
    /// The version of that commit.
    version: AtomicU64,
    end: AtomicU64,
    /// The worker stopped trying among the others. What it did alone later met
    /// nobody, and the others' commits in between found it idle.
    gave_up: AtomicU64,
}

/// A view of how far the operations have come.
struct ClockView {
    ops: Vec<usize>,
    contended: BTreeSet<i64>,
    increments: i64,
    version: u64,
}

impl Clocks {
    fn new(workers: usize, ops: usize) -> Self {
        Self {
            tick: Arc::new(AtomicU64::new(0)),
            started: (0..workers).map(|_| AtomicUsize::new(0)).collect(),
            completed: (0..workers).map(|_| AtomicUsize::new(0)).collect(),
            contended_started: Mutex::new(BTreeSet::new()),
            contended_completed: Mutex::new(BTreeSet::new()),
            increments_started: AtomicI64::new(0),
            increments_completed: AtomicI64::new(0),
            newest_returned: AtomicU64::new(0),
            spans: (0..ops).map(|_| Span::default()).collect(),
        }
    }

    fn now(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn begin(&self, op: &PlannedOp, position: usize) {
        self.started[op.worker].fetch_max(position + 1, Ordering::SeqCst);
        let span = &self.spans[op.rank];
        if span.begin.load(Ordering::SeqCst) != 0 {
            return;
        }
        if op.op.is_contended() {
            self.contended_started.lock().unwrap().insert(op.number);
        }
        if op.op == RoundOp::CounterUpdate {
            self.increments_started.fetch_add(1, Ordering::SeqCst);
        }
        span.begin.store(self.now(), Ordering::SeqCst);
    }

    fn complete(&self, op: &PlannedOp, position: usize, commit: u64, version: u64) {
        if op.op.is_contended() {
            self.contended_completed.lock().unwrap().insert(op.number);
        }
        if op.op == RoundOp::CounterUpdate {
            self.increments_completed.fetch_add(1, Ordering::SeqCst);
        }
        self.newest_returned.fetch_max(version, Ordering::SeqCst);
        self.completed[op.worker].fetch_max(position + 1, Ordering::SeqCst);
        let span = &self.spans[op.rank];
        // What the worker's state last saw land may be an earlier operation's.
        if commit > span.begin.load(Ordering::SeqCst) {
            span.commit.store(commit, Ordering::SeqCst);
            span.version.store(version, Ordering::SeqCst);
        }
        span.end.store(self.now(), Ordering::SeqCst);
    }

    fn give_up(&self, op: &PlannedOp) {
        self.spans[op.rank]
            .gave_up
            .store(self.now(), Ordering::SeqCst);
    }

    fn started(&self) -> ClockView {
        ClockView {
            ops: self
                .started
                .iter()
                .map(|ops| ops.load(Ordering::SeqCst))
                .collect(),
            contended: self.contended_started.lock().unwrap().clone(),
            increments: self.increments_started.load(Ordering::SeqCst),
            version: self.newest_returned.load(Ordering::SeqCst),
        }
    }

    fn completed(&self) -> ClockView {
        ClockView {
            ops: self
                .completed
                .iter()
                .map(|ops| ops.load(Ordering::SeqCst))
                .collect(),
            contended: self.contended_completed.lock().unwrap().clone(),
            increments: self.increments_completed.load(Ordering::SeqCst),
            version: self.newest_returned.load(Ordering::SeqCst),
        }
    }
}

/// Everything the tasks of a round share.
struct RoundContext {
    n: usize,
    uri: String,
    oracle: Oracle,
    /// The seed rows are written from, which a self-test can tell from the
    /// oracle's.
    data_seed: u64,
    scale: Scale,
    schema: SchemaRef,
    workers: usize,
    stride: i64,
    round: Round,
    model: RoundModel,
    clocks: Clocks,
    gates: Vec<Gate>,
    open: Vec<watch::Sender<bool>>,
    /// Per rank, that the operation got to its hold point; `arrivals` wait for it.
    arrived: Vec<Arc<watch::Sender<bool>>>,
    arrivals: Vec<Gate>,
    gate_timeouts: Arc<AtomicU64>,
    start: Barrier,
    writers_done: watch::Sender<bool>,
    journal: Arc<Journal>,
    reads: Arc<Journal>,
    state: Arc<IncidentState>,
    session: Option<Arc<Session>>,
    before: u64,
    /// The oldest version there was when the round began.
    oldest: u64,
    digests: BTreeMap<u64, u64>,
    min_commits: AtomicU64,
    max_commits: AtomicU64,
    /// The newest version any cleanup of this round was allowed to remove from.
    cleanup_boundary: AtomicU64,
    /// No row held a key of the round when it began.
    only_new_keys: bool,
    /// The index on the key was there when the round began.
    indexed_before: bool,
    /// A reader found that a file of the oldest version is gone.
    oldest_gone: AtomicBool,
    /// The version and digest of what each worker's handle showed.
    handle_reads: Mutex<Vec<(u64, u64)>>,
    /// The fragments each version's transaction touched, by version.
    touched: Mutex<BTreeMap<u64, BTreeSet<u64>>>,
    /// What the shared upserts left, as the history check found it.
    history_contended: Mutex<ClassRows>,
    serial: bool,
    self_test: Option<SelfTest>,
}

impl RoundContext {
    async fn open_handle(
        &self,
        wrapper: Option<Arc<StressWrapper>>,
        version: Option<u64>,
    ) -> lance::Result<Dataset> {
        let mut builder = DatasetBuilder::from_uri(&self.uri);
        if let Some(wrapper) = wrapper {
            builder = builder.with_store_params(ObjectStoreParams {
                object_store_wrapper: Some(wrapper),
                ..Default::default()
            });
        }
        if let Some(session) = &self.session {
            builder = builder.with_session(session.clone());
        }
        if let Some(version) = version {
            builder = builder.with_version(version);
        }
        Box::pin(builder.load()).await
    }

    async fn worker_handle(&self, worker: usize) -> Result<WorkerHandle, Failure> {
        let state = Arc::new(self.state.sibling(self.clocks.tick.clone()));
        state.set_phase(format!("{}.worker{worker}", self.n));
        let wrapper = Arc::new(StressWrapper::new(state.clone()));
        let ds = self
            .open_handle(Some(wrapper), None)
            .await
            .map_err(|error| fail("unreadable", self.n, error))?;
        Ok(WorkerHandle {
            hooks: Arc::new(StressHooks::new(state.clone())),
            state,
            ds,
        })
    }

    /// A commit that the plan does not know of.
    fn note_an_extra_commit(&self) {
        self.min_commits.fetch_add(1, Ordering::SeqCst);
        self.max_commits.fetch_add(1, Ordering::SeqCst);
    }

    fn open_gate(&self, rank: usize) {
        self.open[rank].send_replace(true);
    }

    fn observe(&self, observation: impl fmt::Display) {
        self.journal
            .line(format_args!("step={} observation={observation}", self.n));
    }

    /// The cleanup took two versions after the first at least: the write that the
    /// staged commits never saw is among them.
    fn hole_made(&self) -> bool {
        self.cleanup_boundary.load(Ordering::SeqCst) > self.before + 1
    }

    /// What became of a staged commit over the hole: `refused` is the fix of
    /// <https://github.com/lance-format/lance/issues/10603>, and for an insert of
    /// declared keys of <https://github.com/lance-format/lance/issues/10659>;
    /// `landed` only right for what changes no fragment and inserts no declared
    /// key, an index build withdrawing the coverage the hidden rewrite made stale
    /// (<https://github.com/lance-format/lance/issues/10660>); `masked` a missing
    /// file that came first, and `conflict` any other conflict.
    fn observe_hole(&self, op: &PlannedOp, result: &lance::Result<Attempted>) {
        let hidden = self.round.hole.unwrap_or("none");
        let outcome = match result {
            Ok(_) => "landed",
            Err(error) if is_hole_refusal(error) => "refused",
            Err(Error::RetryableCommitConflict { .. }) => "conflict",
            Err(error) if is_gone(error) => "masked",
            Err(_) => "error",
        };
        self.observe(format_args!("hole_{outcome}:{}:{hidden}", op.op.kind()));
    }
}

struct WorkerHandle {
    ds: Dataset,
    state: Arc<IncidentState>,
    hooks: Arc<StressHooks>,
}

struct WorkerEnd {
    worker: usize,
    handle: WorkerHandle,
    /// The first of its operations that has not gone through.
    next: usize,
}

fn touched_fragments(operation: &Operation) -> BTreeSet<u64> {
    match operation {
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            ..
        } => updated_fragments
            .iter()
            .map(|fragment| fragment.id)
            .chain(deleted_fragment_ids.iter().copied())
            .collect(),
        Operation::Update {
            updated_fragments,
            removed_fragment_ids,
            ..
        } => updated_fragments
            .iter()
            .map(|fragment| fragment.id)
            .chain(removed_fragment_ids.iter().copied())
            .collect(),
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .flat_map(|group| group.old_fragments.iter().map(|fragment| fragment.id))
            .collect(),
        _ => BTreeSet::new(),
    }
}

/// How many rows an operation says it touched.
#[derive(Clone, Copy, Debug, Default)]
struct RowCounts {
    inserted: u64,
    updated: u64,
    deleted: u64,
}

/// What a try of an operation reports.
#[derive(Default)]
pub(super) struct Attempted {
    output: OpOutput,
    counts: Option<RowCounts>,
    /// How many tries Lance itself made inside, where it says.
    tries_inside: u32,
}

/// The rows of a worker's class in a range of ids.
pub(super) fn class_range(stride: i64, worker: usize, lo: i64, hi: i64) -> String {
    format!("id % {stride} = {worker} AND id >= {lo} AND id < {hi}")
}

/// One try of an operation.
fn attempt<'a>(
    context: &'a RoundContext,
    handle: &'a mut WorkerHandle,
    op: &PlannedOp,
) -> BoxFuture<'a, lance::Result<Attempted>> {
    let number = op.number;
    let stride = context.stride;
    let worker = op.worker;
    let hooks = handle.hooks.clone();
    let ds = &mut handle.ds;
    let full_source = move |ids: &[i64]| -> Box<dyn RecordBatchReader + Send> {
        Box::new(source_of(full_batch(
            &context.schema,
            context.data_seed,
            &context.scale,
            ids,
            number,
        )))
    };
    let keys_predicate = move || {
        let keys: Vec<String> = context.round.keys.iter().map(i64::to_string).collect();
        format!("id IN ({})", keys.join(","))
    };
    match op.op {
        RoundOp::Append { first_id, rows } => Box::pin(async move {
            let params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: context.scale.rows_per_file,
                progress: hooks,
                ..Default::default()
            };
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            Box::pin(ds.append(full_source(&ids), Some(params))).await?;
            Ok(Attempted::default())
        }),
        RoundOp::Delete { lo, hi } => {
            Box::pin(async move { delete(ds, class_range(stride, worker, lo, hi)).await })
        }
        RoundOp::ContendedDelete => Box::pin(async move { delete(ds, keys_predicate()).await }),
        RoundOp::Update { lo, hi } => Box::pin(async move {
            let predicate = class_range(stride, worker, lo, hi);
            if context.self_test == Some(SelfTest::TornSnapshot) {
                let middle = (lo + hi) / 2;
                let first_half = format!("{predicate} AND id < {middle}");
                let first = update(ds, &first_half, &number.to_string()).await?;
                context.note_an_extra_commit();
                let second_half = format!("{predicate} AND id >= {middle}");
                let mut second = update(ds, &second_half, &number.to_string()).await?;
                if let (Some(first), Some(second)) = (first.counts, second.counts.as_mut()) {
                    second.updated += first.updated;
                }
                return Ok(second);
            }
            update(ds, &predicate, &number.to_string()).await
        }),
        RoundOp::ContendedUpdate => {
            Box::pin(async move { update(ds, &keys_predicate(), &number.to_string()).await })
        }
        RoundOp::CounterUpdate => Box::pin(async move {
            let predicate = format!("id % {stride} = {}", context.workers + 1);
            update(ds, &predicate, "gen + 1").await
        }),
        RoundOp::MergeInsert { first_id, rows } => Box::pin(async move {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            upsert(ds, full_source(&ids)).await
        }),
        RoundOp::ContendedUpsert => {
            Box::pin(async move { upsert(ds, full_source(&context.round.keys)).await })
        }
        RoundOp::MergeInsertColumns { first_id, rows } => Box::pin(async move {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            let source = stamp_source(&context.schema, ids, number);
            let (merged, stats) = Box::pin(stamp_job(ds)?.execute_reader(source)).await?;
            *ds = merged.as_ref().clone();
            Ok(merge_report(&stats))
        }),
        RoundOp::CreateIndex(kind) => {
            let build = op_future(
                ds,
                &hooks,
                context.data_seed,
                context.scale,
                Op::CreateIndex(kind),
                None,
            );
            Box::pin(async move {
                Ok(Attempted {
                    output: build.await?,
                    ..Default::default()
                })
            })
        }
        RoundOp::OptimizeIndices => Box::pin(async move {
            Box::pin(ds.optimize_indices(&OptimizeOptions::new().progress(hooks))).await?;
            Ok(Attempted::default())
        }),
        RoundOp::Compact => {
            let compaction = op_future(
                ds,
                &hooks,
                context.data_seed,
                context.scale,
                Op::Compact,
                None,
            );
            Box::pin(async move {
                Ok(Attempted {
                    output: compaction.await?,
                    ..Default::default()
                })
            })
        }
        RoundOp::Reserve => Box::pin(async move {
            let transaction = Transaction::new(
                ds.version().version,
                Operation::ReserveFragments { num_fragments: 1 },
                None,
            );
            *ds = Box::pin(CommitBuilder::new(Arc::new(ds.clone())).execute(transaction)).await?;
            Ok(Attempted::default())
        }),
        RoundOp::Cleanup { retain } => Box::pin(async move {
            let policy = CleanupPolicyBuilder::default()
                .retain_n_versions(ds, retain)
                .await?
                .error_if_tagged_old_versions(false)
                .build();
            if let Some(boundary) = policy.before_version {
                context
                    .cleanup_boundary
                    .fetch_max(boundary, Ordering::SeqCst);
            }
            let removed = Box::pin(ds.cleanup_with_policy(policy)).await?;
            context.journal.line(format_args!(
                "step={} cleanup old_versions={} data_files={} index_files={} transaction_files={} deletion_files={}",
                context.n,
                removed.old_versions,
                removed.data_files_removed,
                removed.index_files_removed,
                removed.transaction_files_removed,
                removed.deletion_files_removed
            ));
            Ok(Attempted::default())
        }),
    }
}

pub(super) async fn delete(ds: &mut Dataset, predicate: String) -> lance::Result<Attempted> {
    let deleted = DeleteBuilder::new(Arc::new(ds.clone()), predicate)
        .retry_timeout(RETRY_TIMEOUT)
        .execute()
        .await?;
    *ds = deleted.new_dataset.as_ref().clone();
    Ok(Attempted {
        counts: Some(RowCounts {
            deleted: deleted.num_deleted_rows,
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub(super) async fn update(
    ds: &mut Dataset,
    predicate: &str,
    value: &str,
) -> lance::Result<Attempted> {
    let job = UpdateBuilder::new(Arc::new(ds.clone()))
        .update_where(predicate)?
        .set("gen", value)?
        .retry_timeout(RETRY_TIMEOUT)
        .build()?;
    let updated = Box::pin(job.execute()).await?;
    *ds = updated.new_dataset.as_ref().clone();
    Ok(Attempted {
        counts: Some(RowCounts {
            updated: updated.rows_updated,
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn upsert_job(ds: &Dataset) -> lance::Result<MergeInsertJob> {
    MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])?
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .retry_timeout(RETRY_TIMEOUT)
        .try_build()
}

pub(super) async fn upsert(
    ds: &mut Dataset,
    source: Box<dyn RecordBatchReader + Send>,
) -> lance::Result<Attempted> {
    let (merged, stats) = Box::pin(upsert_job(ds)?.execute_reader(source)).await?;
    *ds = merged.as_ref().clone();
    Ok(merge_report(&stats))
}

/// Rewrites `gen` in place for the rows that exist, and inserts none.
pub(super) fn stamp_job(ds: &Dataset) -> lance::Result<MergeInsertJob> {
    MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])?
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .retry_timeout(RETRY_TIMEOUT)
        .try_build()
}

pub(super) fn stamp_source(
    schema: &SchemaRef,
    ids: Vec<i64>,
    number: i64,
) -> Box<dyn RecordBatchReader + Send> {
    let columns = Arc::new(Schema::new(vec![
        schema.field(0).clone(),
        Field::new("gen", DataType::Int64, false),
    ]));
    let stamps = vec![number; ids.len()];
    let batch = RecordBatch::try_new(
        columns,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(stamps)),
        ],
    )
    .expect("build a batch");
    Box::new(source_of(batch))
}

fn merge_report(stats: &MergeStats) -> Attempted {
    Attempted {
        counts: Some(RowCounts {
            inserted: stats.num_inserted_rows,
            updated: stats.num_updated_rows,
            deleted: stats.num_deleted_rows,
        }),
        tries_inside: stats.num_attempts,
        ..Default::default()
    }
}

/// A first try prepared without a commit.
enum Staged {
    Transaction {
        transaction: Box<Transaction>,
        affected_rows: Option<RowAddrTreeMap>,
        counts: Option<RowCounts>,
    },
    Compaction(Vec<RewriteResult>),
}

/// Prepares the first try of an operation on the version its handle stands on,
/// as a distributed writer does. `None` if there is nothing to commit.
async fn stage(
    context: &RoundContext,
    handle: &WorkerHandle,
    op: &PlannedOp,
) -> lance::Result<Option<Staged>> {
    let stride = context.stride;
    let ds = &handle.ds;
    let full_source = |ids: &[i64]| -> Box<dyn RecordBatchReader + Send> {
        Box::new(source_of(full_batch(
            &context.schema,
            context.data_seed,
            &context.scale,
            ids,
            op.number,
        )))
    };
    let merged = |uncommitted: UncommittedMergeInsert| Staged::Transaction {
        transaction: Box::new(uncommitted.transaction),
        affected_rows: uncommitted.affected_rows,
        counts: merge_report(&uncommitted.stats).counts,
    };
    let staged = match op.op {
        RoundOp::Delete { lo, hi } => {
            let predicate = class_range(stride, op.worker, lo, hi);
            let deleted = DeleteBuilder::new(Arc::new(ds.clone()), predicate)
                .execute_uncommitted()
                .await?;
            Staged::Transaction {
                transaction: Box::new(deleted.transaction),
                affected_rows: deleted.affected_rows,
                counts: Some(RowCounts {
                    deleted: deleted.num_deleted_rows,
                    ..Default::default()
                }),
            }
        }
        RoundOp::MergeInsert { first_id, rows } => {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            merged(Box::pin(upsert_job(ds)?.execute_uncommitted(full_source(&ids))).await?)
        }
        RoundOp::ContendedUpsert => {
            let source = full_source(&context.round.keys);
            merged(Box::pin(upsert_job(ds)?.execute_uncommitted(source)).await?)
        }
        RoundOp::MergeInsertColumns { first_id, rows } => {
            let ids: Vec<i64> = id_run(first_id, rows, stride).collect();
            let source = stamp_source(&context.schema, ids, op.number);
            merged(Box::pin(stamp_job(ds)?.execute_uncommitted(source)).await?)
        }
        RoundOp::Compact => {
            let options = CompactionOptions::default();
            let plan = plan_compaction(ds, &options).await?;
            let mut rewritten = Vec::new();
            for task in plan.compaction_tasks() {
                rewritten.push(Box::pin(task.execute(ds)).await?);
            }
            if rewritten.is_empty() {
                return Ok(None);
            }
            Staged::Compaction(rewritten)
        }
        RoundOp::CreateIndex(
            kind @ (IndexKind::BTree | IndexKind::BTreeId | IndexKind::BTreeStamp),
        ) => {
            let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
            let mut building = ds.clone();
            let index = Box::pin(
                building
                    .create_index_builder(&[kind.column()], IndexType::BTree, &params)
                    .name(kind.name().to_string())
                    .replace(true)
                    .execute_uncommitted(),
            )
            .await?;
            let removed_indices = ds
                .load_indices()
                .await?
                .iter()
                .filter(|old| old.name == kind.name())
                .cloned()
                .collect();
            let operation = Operation::CreateIndex {
                new_indices: vec![index],
                removed_indices,
            };
            Staged::Transaction {
                transaction: Box::new(Transaction::new(ds.version().version, operation, None)),
                affected_rows: None,
                counts: None,
            }
        }
        _ => unreachable!("only the operations of `HOLES` are staged, not {}", op.op),
    };
    Ok(Some(staged))
}

/// Commits a staged try from the handle, which still stands on the version the
/// try was prepared on.
async fn commit_staged(handle: &mut WorkerHandle, staged: Staged) -> lance::Result<Attempted> {
    match staged {
        Staged::Transaction {
            transaction,
            affected_rows,
            counts,
        } => {
            let mut commit = CommitBuilder::new(Arc::new(handle.ds.clone()));
            if let Some(affected_rows) = affected_rows {
                commit = commit.with_affected_rows(affected_rows);
            }
            handle.ds = Box::pin(commit.execute(*transaction)).await?;
            Ok(Attempted {
                counts,
                ..Default::default()
            })
        }
        Staged::Compaction(rewritten) => {
            let metrics = Box::pin(commit_compaction(
                &mut handle.ds,
                rewritten,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &CompactionOptions::default(),
            ))
            .await?;
            Ok(Attempted {
                output: OpOutput {
                    fragments_removed: metrics.fragments_removed,
                    ..Default::default()
                },
                ..Default::default()
            })
        }
    }
}

/// Runs an operation, with the tries it gets, and reads back what it wrote.
/// Returns false if it did not get through and has to wait for the others.
/// Alone it must get through.
async fn run_op(
    context: &RoundContext,
    handle: &mut WorkerHandle,
    op: &PlannedOp,
    position: usize,
    alone: bool,
    mut staged: Option<Staged>,
) -> Result<bool, Failure> {
    let n = context.n;
    let worker = op.worker;
    let rank = op.rank;
    if context.serial && !alone && rank > 0 {
        wait_for_rank(context, rank - 1).await;
    }
    handle.state.set_phase(format!("{n}.w{worker}.r{rank}"));
    // A staged operation with nothing staged goes the usual way.
    if op.catch_up || alone || (op.staged && staged.is_none()) {
        Box::pin(handle.ds.checkout_latest())
            .await
            .map_err(|error| fail("unreadable", n, error))?;
    }
    context.clocks.begin(op, position);
    let lower = context.clocks.completed();
    let tries = if alone { 1 } else { PARALLEL_ATTEMPTS };
    for attempt_number in 1..=tries {
        // One at a time, what an arrival waits for only starts after it ends.
        let hold = op
            .hold
            .as_ref()
            .filter(|hold| attempt_number == 1 && !alone && !(context.serial && hold.arrival));
        if let Some(hold) = hold {
            let gate = if hold.arrival {
                &context.arrivals[hold.until]
            } else {
                &context.gates[hold.until]
            };
            handle.state.arm_hold(
                hold.point.clone(),
                gate.clone().announcing(context.arrived[rank].clone()),
            );
        }
        let version_before = handle.ds.version().version;
        let staged_try = staged.take();
        let is_staged = staged_try.is_some();
        let result = match staged_try {
            Some(staged) => {
                // Worker 0 gave up before its cleanup, or the cleanup took too
                // little: the round would test nothing it is meant to.
                if !context.hole_made() {
                    return Err(fail(
                        "harness",
                        n,
                        format_args!(
                            "{} of worker {worker} was staged over a hole the round did not make",
                            op.op
                        ),
                    ));
                }
                let result = commit_staged(handle, staged).await;
                context.observe_hole(op, &result);
                result
            }
            None => attempt(context, handle, op).await,
        };
        let fired = handle.state.disarm();
        let waited = handle.state.take_gate_waits();
        // An operation that found nothing to do never got to the point, whichever
        // of the racers that is, so it says nothing about the point. A cleanup
        // commits nothing, but always lists the versions an arrival holds it at.
        let committed = handle.ds.version().version != version_before;
        if let Some(hold) = hold.filter(|hold| fired > 0 || committed || hold.arrival) {
            context.journal.line(format_args!(
                "step={n} hold=hold:{}:{} worker={worker} rank={rank} until={} fired={fired} waited={waited}",
                op.op.kind(),
                hold.point.kind(),
                hold.until
            ));
        }
        for _ in 0..handle.state.take_lost_races() {
            context.observe(format_args!("lost_manifest_race:{}", op.op.kind()));
        }
        let (min, max) = op
            .op
            .commits(result.as_ref().ok().map(|attempted| &attempted.output));
        context.min_commits.fetch_add(min, Ordering::SeqCst);
        context.max_commits.fetch_add(max, Ordering::SeqCst);
        match result {
            Ok(attempted) => {
                context.clocks.complete(
                    op,
                    position,
                    handle.state.last_commit(),
                    handle.ds.version().version,
                );
                context.open_gate(rank);
                context.journal.line(format_args!(
                    "step={n} worker={worker} rank={rank} op_end op={} attempt={attempt_number} alone={alone} outcome=ok version={}",
                    op.op,
                    handle.ds.version().version
                ));
                if attempted.tries_inside > 1 {
                    context.observe(format_args!("internal_retry:{}", op.op.kind()));
                }
                if let Some(counts) = &attempted.counts {
                    context.model.check_counts(op, position, counts, n)?;
                }
                if context.self_test == Some(SelfTest::DuplicateKey) && rank == 0 {
                    let id = worker as i64;
                    let source = source_of(full_batch(
                        &context.schema,
                        context.data_seed,
                        &context.scale,
                        &[id],
                        op.number,
                    ));
                    Box::pin(handle.ds.append(source, None))
                        .await
                        .map_err(|error| fail("harness", n, error))?;
                    context.note_an_extra_commit();
                }
                check_own_read(
                    context,
                    handle,
                    op,
                    position,
                    version_before,
                    &lower,
                    attempted.counts.as_ref(),
                )
                .await?;
                return Ok(true);
            }
            Err(error) => {
                let message = one_line(&error.to_string());
                // Refusing a commit over the hole is the point of the hole. Any other
                // error of a staged commit is judged as any other operation's.
                let legal = op.op.is_legal_error(&error, &context.round, rank)
                    || (is_staged && is_hole_refusal(&error));
                context.journal.line(format_args!(
                    "step={n} worker={worker} rank={rank} op_end op={} attempt={attempt_number} alone={alone} outcome={} detail={message}",
                    op.op,
                    if legal { "legal_error" } else { "error" }
                ));
                if alone {
                    return Err(fail(
                        "retry_failed",
                        n,
                        format_args!("alone, {} of worker {worker} ended with: {message}", op.op),
                    ));
                }
                if !legal {
                    return Err(fail(
                        "error_without_fault",
                        n,
                        format_args!(
                            "{} of worker {worker} failed with nothing to explain it: {message}",
                            op.op
                        ),
                    ));
                }
                context.observe(format_args!("legal_error:{}", op.op.kind()));
                // Others that wait for its end wait for its first try, unless they
                // come one at a time.
                if !context.serial {
                    context.open_gate(rank);
                }
                Box::pin(handle.ds.checkout_latest())
                    .await
                    .map_err(|error| fail("unreadable", n, error))?;
            }
        }
    }
    context.clocks.give_up(op);
    context.open_gate(rank);
    Ok(false)
}

/// What a handle shows right after its own operation: its own write, and if the
/// operation committed, everything that finished before it began. The history
/// check then compares what the handle showed with the version it stood on.
async fn check_own_read(
    context: &RoundContext,
    handle: &WorkerHandle,
    op: &PlannedOp,
    position: usize,
    version_before: u64,
    lower: &ClockView,
    counts: Option<&RowCounts>,
) -> Result<(), Failure> {
    let n = context.n;
    let worker = op.worker;
    let what = format!("worker {worker} after {}", op.op);
    let image = context
        .oracle
        .image(&handle.ds, n, &what, false)
        .await?
        .map_err(|error| fail("unreadable", n, error))?;
    context
        .handle_reads
        .lock()
        .unwrap()
        .push((image.version, image.digest()));
    let upper = context.clocks.started();
    let placed = context.model.place(&image, n, &what)?;
    if !placed.prefixes[worker].contains(&(position + 1)) {
        return Err(fail(
            "torn_snapshot",
            n,
            format_args!(
                "{what}: its handle shows its rows as of {:?}, after operation {}",
                placed.prefixes[worker],
                position + 1
            ),
        ));
    }
    if handle.ds.version().version == version_before {
        return Ok(());
    }
    // An update or a delete that found no key has no effect to show, and an upsert
    // may have landed on top of its empty commit.
    let own_effect = match op.op {
        RoundOp::ContendedUpsert => Some(Contended::Applied(op.number)),
        RoundOp::ContendedUpdate if counts.is_some_and(|counts| counts.updated > 0) => {
            Some(Contended::Applied(op.number))
        }
        RoundOp::ContendedDelete if counts.is_some_and(|counts| counts.deleted > 0) => {
            Some(Contended::Deleted)
        }
        _ => None,
    };
    if let Some(expected) = own_effect
        && placed.contended != expected
    {
        return Err(fail(
            "lost_write",
            n,
            format_args!(
                "{what}: the commit of operation {} shows {:?}, not {expected:?}",
                op.number, placed.contended
            ),
        ));
    }
    if op.op == RoundOp::CounterUpdate && placed.increments == 0 {
        return Err(fail(
            "lost_write",
            n,
            format_args!("{what}: the commit does not show the increment"),
        ));
    }
    context
        .model
        .check_bounds(image.version, &placed, lower, &upper, n, &what)
}

async fn run_worker(
    context: Arc<RoundContext>,
    worker: usize,
    mut handle: WorkerHandle,
) -> Result<WorkerEnd, Failure> {
    let ops = context.round.ops_of(worker);
    // Prepared on the version the round starts from, before anyone writes.
    let staged = match ops.first() {
        Some(op) if op.staged => Some(stage(&context, &handle, op).await),
        _ => None,
    };
    context.start.wait().await;
    let mut staged = match staged {
        Some(Ok(staged)) => {
            if staged.is_none() {
                let hidden = context.round.hole.unwrap_or("none");
                context.observe(format_args!("hole_nothing:{}:{hidden}", ops[0].op.kind()));
            }
            staged
        }
        Some(Err(error)) => {
            return Err(fail(
                "error_without_fault",
                context.n,
                format_args!("staging {} of worker {worker}: {error}", ops[0].op),
            ));
        }
        None => None,
    };
    for (position, op) in ops.iter().enumerate() {
        if let Some(rank) = op.start_after {
            wait_for_rank(&context, rank).await;
        }
        if !run_op(&context, &mut handle, op, position, false, staged.take()).await? {
            for later in &ops[position + 1..] {
                context.open_gate(later.rank);
            }
            return Ok(WorkerEnd {
                worker,
                handle,
                next: position,
            });
        }
    }
    Ok(WorkerEnd {
        worker,
        next: ops.len(),
        handle,
    })
}

/// Waits, with no limit, until the operation of this rank is over. A hang here is
/// the watchdog's to find, as in the serial mode.
async fn wait_for_rank(context: &RoundContext, rank: usize) {
    let mut opened = context.open[rank].subscribe();
    let _ = opened.wait_for(|opened| *opened).await;
}

/// One read of the latest version. Returns its version and digest.
async fn read_latest(
    context: &RoundContext,
    ds: &Dataset,
    lower: &ClockView,
    what: &str,
    previous: Option<u64>,
) -> Result<(u64, u64), Failure> {
    let n = context.n;
    let image = context
        .oracle
        .image(ds, n, what, true)
        .await?
        .map_err(|error| fail("unreadable", n, format_args!("{what}: {error}")))?;
    if let Some(previous) = previous
        && image.version < previous
    {
        return Err(fail(
            "stale_read",
            n,
            format_args!(
                "{what}: the latest version went back from {previous} to {}",
                image.version
            ),
        ));
    }
    let upper = context.clocks.started();
    let placed = context.model.place(&image, n, what)?;
    context
        .model
        .check_bounds(image.version, &placed, lower, &upper, n, what)?;
    Box::pin(context.oracle.check_indices(ds, n, &image, true)).await?;
    let digest = image.digest();
    context.reads.line(format_args!(
        "step={n} what={} version={} digest={digest:016x} rows={} prefixes={:?} contended={:?} increments={}",
        what.replace(' ', "_"),
        image.version,
        image.rows.len(),
        placed.prefixes,
        placed.contended,
        placed.increments
    ));
    Ok((image.version, digest))
}

/// A manifest a cleanup took is `DatasetNotFound`, a file it took `NotFound`.
/// Lance's refusal of a commit over versions that a cleanup removed, in the words
/// of `check_missing_versions`.
fn is_hole_refusal(error: &Error) -> bool {
    matches!(error, Error::RetryableCommitConflict { .. })
        && error.to_string().contains("were cleaned up")
}

fn is_gone(error: &Error) -> bool {
    error.is_not_found() || matches!(error, Error::DatasetNotFound { .. })
}

async fn run_reader(
    context: Arc<RoundContext>,
    index: usize,
    plan: PlannedReader,
) -> Result<Vec<(u64, u64)>, Failure> {
    let n = context.n;
    let what = format!("reader {index} {}", plan.kind.name());
    let state = plan.hold.as_ref().map(|_| {
        let state = Arc::new(context.state.sibling(context.clocks.tick.clone()));
        state.set_phase(format!("{n}.reader{index}"));
        state
    });
    let wrapper = state
        .as_ref()
        .map(|state| Arc::new(StressWrapper::new(state.clone())));
    let unreadable = |error: Error| fail("unreadable", n, format_args!("{what}: {error}"));
    let pin_to = match plan.kind {
        ReaderKind::FreshOpen | ReaderKind::CheckoutLatest => None,
        ReaderKind::Pinned => Some(context.before),
        ReaderKind::OldSnapshot => Some(context.oldest),
    };
    let handle = match plan.kind {
        ReaderKind::FreshOpen => None,
        _ => Some(
            context
                .open_handle(wrapper.clone(), pin_to)
                .await
                .map_err(unreadable)?,
        ),
    };
    let mut writers_done = context.writers_done.subscribe();
    context.start.wait().await;
    // Armed only now: opening the handle must not wait for a round not yet begun.
    if let (Some(state), Some(hold)) = (&state, &plan.hold) {
        state.arm_hold(hold.point.clone(), context.gates[hold.until].clone());
    }
    if let Some(rank) = plan.start_after {
        wait_for_rank(&context, rank).await;
    }
    let mut reads = Vec::new();
    match (plan.kind, handle) {
        (ReaderKind::FreshOpen, _) => loop {
            let last = *writers_done.borrow();
            let lower = context.clocks.completed();
            let stale = last && context.self_test == Some(SelfTest::StaleRead);
            let ds = context
                .open_handle(wrapper.clone(), stale.then_some(context.before))
                .await
                .map_err(unreadable)?;
            let previous = reads.last().map(|(version, _)| *version);
            reads.push(read_latest(&context, &ds, &lower, &what, previous).await?);
            if last {
                break;
            }
        },
        (ReaderKind::CheckoutLatest, Some(mut ds)) => loop {
            let last = *writers_done.borrow();
            let lower = context.clocks.completed();
            Box::pin(ds.checkout_latest()).await.map_err(unreadable)?;
            let previous = reads.last().map(|(version, _)| *version);
            reads.push(read_latest(&context, &ds, &lower, &what, previous).await?);
            if last {
                break;
            }
        },
        (_, Some(ds)) => {
            let expected = pin_to.and_then(|version| context.digests.get(&version).copied());
            for pass in 0..2 {
                if pass == 1 {
                    let _ = writers_done.wait_for(|done| *done).await;
                }
                let moved = if pass == 1 && context.self_test == Some(SelfTest::SnapshotChanged) {
                    Some(context.open_handle(None, None).await.map_err(unreadable)?)
                } else {
                    None
                };
                match context
                    .oracle
                    .image(moved.as_ref().unwrap_or(&ds), n, &what, true)
                    .await?
                {
                    Ok(image) => {
                        let digest = image.digest();
                        context.reads.line(format_args!(
                            "step={n} what={} pass={pass} version={} digest={digest:016x} rows={}",
                            what.replace(' ', "_"),
                            image.version,
                            image.rows.len()
                        ));
                        if let Some(expected) = expected
                            && expected != digest
                        {
                            return Err(fail(
                                "snapshot_changed",
                                n,
                                format_args!(
                                    "{what}: version {} read as {digest:016x}, and before as {expected:016x}",
                                    image.version
                                ),
                            ));
                        }
                        // The version a round starts from stays, indices included.
                        if plan.kind == ReaderKind::Pinned {
                            let pinned = moved.as_ref().unwrap_or(&ds);
                            Box::pin(context.oracle.check_indices(pinned, n, &image, true)).await?;
                        }
                        reads.push((image.version, digest));
                    }
                    // Without a cleanup in the round, the version it could see at the
                    // start must stay; `check_history` holds a cleanup to its boundary.
                    Err(error)
                        if plan.kind == ReaderKind::OldSnapshot
                            && context.round.has_cleanup()
                            && is_gone(&error) =>
                    {
                        context.oldest_gone.store(true, Ordering::SeqCst);
                        context.observe("old_snapshot_gone");
                        context.reads.line(format_args!(
                            "step={n} what={} pass={pass} gone detail={}",
                            what.replace(' ', "_"),
                            one_line(&error.to_string())
                        ));
                    }
                    Err(error) => return Err(unreadable(error)),
                }
            }
        }
        _ => unreachable!("every kind but a fresh open has a handle"),
    }
    if let (Some(state), Some(hold)) = (&state, &plan.hold) {
        let fired = state.disarm();
        let waited = state.take_gate_waits();
        context.journal.line(format_args!(
            "step={n} hold=hold:reader_{}:{} reader={index} until={} fired={fired} waited={waited}",
            plan.kind.name(),
            hold.point.kind(),
            hold.until
        ));
    }
    Ok(reads)
}

async fn joined<T>(task: JoinHandle<Result<T, Failure>>, n: usize) -> Result<T, Failure> {
    task.await.unwrap_or_else(|error| {
        Err(fail(
            "panic",
            n,
            format_args!("a task of the round: {error}"),
        ))
    })
}

impl Run {
    pub(super) async fn run_rounds(&mut self) -> Result<(), Failure> {
        let plan = self.plan.rounds.clone().expect("a plan with rounds");
        for (index, round) in plan.rounds.iter().enumerate() {
            Box::pin(self.run_round(index + 1, &plan, round)).await?;
        }
        Ok(())
    }

    fn oracle(&self, plan: &RoundsPlan) -> Oracle {
        Oracle {
            scale: self.plan.scale,
            data_seed: self.oracle_seed,
            stable_row_ids: plan.stable_row_ids,
        }
    }

    async fn round_context(
        &self,
        n: usize,
        plan: &RoundsPlan,
        round: &Round,
        before: u64,
    ) -> Result<RoundContext, Failure> {
        let state = self.rounds.as_ref().expect("the state of the rounds");
        let verifier = self.open_verifier(n).await?;
        let versions = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let oldest = versions.first().map_or(before, |version| version.version);
        let gate_timeouts = Arc::new(AtomicU64::new(0));
        let gate_limit = (self.self_test != Some(SelfTest::GateHang)).then_some(GATE_LIMIT);
        let (open, gates) = round
            .ops
            .iter()
            .map(|_| Gate::new(gate_limit, gate_timeouts.clone()))
            .unzip();
        let (arrived, arrivals) = round
            .ops
            .iter()
            .map(|_| {
                let (arrived, arrival) = Gate::new(gate_limit, gate_timeouts.clone());
                (Arc::new(arrived), arrival)
            })
            .unzip();
        let (writers_done, _) = watch::channel(false);
        Ok(RoundContext {
            n,
            uri: self.uri.clone(),
            oracle: self.oracle(plan),
            data_seed: self.plan.data_seed,
            scale: self.plan.scale,
            schema: plan.schema(),
            workers: plan.workers,
            stride: state.rows.stride(),
            round: round.clone(),
            model: RoundModel::new(&state.rows, round),
            clocks: Clocks::new(plan.workers, round.ops.len()),
            gates,
            open,
            arrived,
            arrivals,
            gate_timeouts,
            start: Barrier::new(plan.workers + round.readers.len()),
            writers_done,
            journal: self.journal.clone(),
            reads: state.reads.clone(),
            state: self.state.clone(),
            session: plan.shared_session.then(|| Arc::new(Session::default())),
            before,
            oldest,
            digests: state.digests.clone(),
            min_commits: AtomicU64::new(0),
            max_commits: AtomicU64::new(0),
            cleanup_boundary: AtomicU64::new(0),
            only_new_keys: round
                .keys
                .iter()
                .all(|key| !state.rows.classes[state.rows.contended()].contains_key(key)),
            indexed_before: self.model.indices.contains(&IndexKind::BTreeId),
            oldest_gone: AtomicBool::new(false),
            handle_reads: Mutex::new(Vec::new()),
            touched: Mutex::new(BTreeMap::new()),
            history_contended: Mutex::new(ClassRows::new()),
            serial: std::env::var(ENV_SERIAL).is_ok_and(|value| value == "1"),
            self_test: self.self_test,
        })
    }

    async fn run_round(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        round: &Round,
    ) -> Result<(), Failure> {
        self.state.set_phase(n.to_string());
        let before = self.latest_version(n).await?;
        if n == 1 {
            self.remember_first_version(n, plan, before).await?;
        }
        self.journal.line(format_args!(
            "step={n} start op=round theme={} ops={} readers={} before={before}",
            round.theme.unwrap_or("none"),
            round.ops.len(),
            round.readers.len()
        ));
        let context = Arc::new(self.round_context(n, plan, round, before).await?);
        let mut worker_tasks = Vec::with_capacity(plan.workers);
        for worker in 0..plan.workers {
            let handle = context.worker_handle(worker).await?;
            worker_tasks.push(tokio::spawn(run_worker(context.clone(), worker, handle)));
        }
        let reader_tasks: Vec<_> = round
            .readers
            .iter()
            .enumerate()
            .map(|(index, reader)| tokio::spawn(run_reader(context.clone(), index, reader.clone())))
            .collect();
        let writers = async {
            let ends = try_join_all(worker_tasks.into_iter().map(|task| joined(task, n))).await?;
            for mut end in ends {
                let ops = context.round.ops_of(end.worker);
                for (position, op) in ops.iter().enumerate().skip(end.next) {
                    run_op(&context, &mut end.handle, op, position, true, None).await?;
                }
            }
            context.writers_done.send_replace(true);
            Ok::<_, Failure>(())
        };
        let readers = try_join_all(reader_tasks.into_iter().map(|task| joined(task, n)));
        let ((), reads) = tokio::try_join!(writers, readers)?;
        // Every task is over. Its session goes now, along with what it caches.
        let mut context = Arc::into_inner(context).expect("the tasks of the round are over");
        context.session = None;
        Box::pin(self.conclude_round(n, plan, round, &context, before, reads)).await
    }

    /// The version a seed starts from is the only one no round has read.
    async fn remember_first_version(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        version: u64,
    ) -> Result<(), Failure> {
        let oracle = self.oracle(plan);
        let ds = self.open_verifier(n).await?;
        let image = oracle
            .image(&ds, n, "the first version", true)
            .await?
            .map_err(|error| fail("unreadable", n, error))?;
        let state = self.rounds.as_mut().expect("the state of the rounds");
        state.digests.insert(version, image.digest());
        state.lineage.start(&image);
        Ok(())
    }

    async fn conclude_round(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        round: &Round,
        context: &RoundContext,
        before: u64,
        reads: Vec<Vec<(u64, u64)>>,
    ) -> Result<(), Failure> {
        self.quiet_down(n).await;
        match self.self_test {
            Some(SelfTest::LostWrite) => {
                let mut old = self
                    .open_verifier(n)
                    .await?
                    .checkout_version(before)
                    .await
                    .map_err(|error| fail("harness", n, error))?;
                old.restore()
                    .await
                    .map_err(|error| fail("harness", n, error))?;
                context.note_an_extra_commit();
            }
            Some(SelfTest::Lineage) => {
                self.rounds
                    .as_mut()
                    .expect("the state of the rounds")
                    .lineage
                    .tamper();
            }
            _ => {}
        }
        let after = self.latest_version(n).await?;
        let (min, max) = (
            context.min_commits.load(Ordering::SeqCst),
            context.max_commits.load(Ordering::SeqCst),
        );
        if after < before + min || after > before + max {
            return Err(fail(
                "version_jump",
                n,
                format_args!(
                    "the round moved the version from {before} to {after}, by {min} to {max} it should"
                ),
            ));
        }
        self.journal.line(format_args!(
            "step={n} verdict incident=none fired=0 outcome=ok committed={} before={before} after={after}",
            after > before
        ));
        let timeouts = context.gate_timeouts.load(Ordering::SeqCst);
        for _ in 0..timeouts {
            self.observe(n, "gate_timeout");
        }
        Box::pin(self.check_history(n, plan, context, before, after, &reads)).await?;
        self.observe_overlaps(n, context);
        if plan.declared_key {
            self.observe_key_race(n, context);
        }

        let state = self.rounds.as_mut().expect("the state of the rounds");
        let contended_after = context.history_contended.lock().unwrap().clone();
        for op in &round.ops {
            state.rows.apply(op.worker, &op.op, op.number, &round.keys);
        }
        // The last of the shared upserts is the one that committed last.
        let class = state.rows.contended();
        state.rows.classes[class] = contended_after;
        state.stamps = state.rows.all_rows();
        self.model.live = state.stamps.keys().copied().collect();
        self.model.next_id = state.rows.next_id();
        for op in &round.ops {
            if let RoundOp::CreateIndex(kind) = op.op {
                self.model.indices.insert(kind);
            }
        }
        Box::pin(self.check_content(n)).await?;
        self.let_go_of_the_session(n, None).await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)?;
        self.journal.line(format_args!("step={n} done"));
        Ok(())
    }

    /// Opens every version the round made, and what a cleanup must have left of
    /// the older ones. Each is a snapshot like any other, and together they must
    /// tell one story: nothing goes back, and each commit does one thing.
    async fn check_history(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        context: &RoundContext,
        before: u64,
        after: u64,
        reads: &[Vec<(u64, u64)>],
    ) -> Result<(), Failure> {
        let oracle = self.oracle(plan);
        let verifier = self.open_verifier(n).await?;
        let present: BTreeSet<u64> = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?
            .iter()
            .map(|version| version.version)
            .collect();
        let boundary = context.cleanup_boundary.load(Ordering::SeqCst);
        if context.oldest_gone.load(Ordering::SeqCst) && context.oldest >= boundary {
            return Err(fail(
                "missing_files",
                n,
                format_args!(
                    "a reader found files of version {} gone, though the cleanup could only take versions before {boundary}",
                    context.oldest
                ),
            ));
        }
        if boundary != 0
            && let Some(oldest) = present.first()
            && *oldest < boundary
        {
            return Err(fail(
                "cleanup",
                n,
                format_args!(
                    "the cleanup returned Ok, but version {oldest} is still there, below its boundary {boundary}"
                ),
            ));
        }
        // Whatever was there at the start stays, but for what a cleanup may take.
        // The round with the hole takes the very version it starts from.
        let hole = context.round.hole.is_some() && boundary > before;
        let must_stay = if hole {
            boundary
        } else {
            context.oldest.max(boundary).min(before)
        };
        if hole {
            // Values only the versions taken held, which an index may still answer.
            let state = self.rounds.as_mut().expect("the state of the rounds");
            let held: Vec<i64> = state.stamps.values().copied().collect();
            state.stamps_ever.extend(held);
        }
        let gaps: Vec<u64> = (must_stay..=after)
            .filter(|version| !present.contains(version))
            .collect();
        if !gaps.is_empty() {
            return Err(fail(
                "version_jump",
                n,
                format_args!(
                    "versions {gaps:?} are gone, though only versions before {must_stay} could go, and {before} to {after} are this round's"
                ),
            ));
        }
        let first_kept = if boundary == 0 { before } else { must_stay };
        let model = &context.model;
        let mut floors = vec![0usize; model.workers];
        let mut contended_values: BTreeSet<i64> = BTreeSet::new();
        let mut contended_now = Contended::Untouched;
        let mut keys_went = false;
        let mut increments = 0;
        let mut last_placement = None;
        let mut last_rows = None;
        for version in first_kept..=after {
            let what = format!("the history of round {n}");
            let ds = DatasetBuilder::from_uri(&self.uri)
                .with_version(version)
                .load()
                .await
                .map_err(|error| {
                    fail("unreadable", n, format_args!("version {version}: {error}"))
                })?;
            let image = oracle.image(&ds, n, &what, true).await?.map_err(|error| {
                fail("unreadable", n, format_args!("version {version}: {error}"))
            })?;
            let digest = image.digest();
            let state = self.rounds.as_mut().expect("the state of the rounds");
            if let Some(known) = state.digests.insert(version, digest)
                && known != digest
            {
                return Err(fail(
                    "snapshot_changed",
                    n,
                    format_args!(
                        "version {version} read as {digest:016x}, and before as {known:016x}"
                    ),
                ));
            }
            state.stamps_ever.extend(image.rows.values().copied());
            if version < before || (version == before && boundary != 0) {
                Box::pin(oracle.check_indices(&ds, n, &image, false)).await?;
            }
            if version <= before {
                if version == before {
                    last_placement = Some(model.place(&image, n, &what)?);
                    last_rows = Some(image.rows);
                }
                continue;
            }
            let transaction = ds.read_transaction().await.map_err(|error| {
                fail("unreadable", n, format_args!("version {version}: {error}"))
            })?;
            if let Some(transaction) = transaction {
                context
                    .touched
                    .lock()
                    .unwrap()
                    .insert(version, touched_fragments(&transaction.operation));
            }
            let placed = model.place(&image, n, &what)?;
            let mut changed = 0;
            for (worker, floor) in floors.iter_mut().enumerate() {
                let Some(place) = placed.places[worker].iter().find(|place| *place >= floor) else {
                    return Err(fail(
                        "lost_write",
                        n,
                        format_args!(
                            "version {version}: the rows of worker {worker} went back to {:?}, from {floor}",
                            placed.prefixes[worker]
                        ),
                    ));
                };
                let states = &model.states[worker];
                let steps = (*floor..*place)
                    .filter(|place| states[*place] != states[*place + 1])
                    .count();
                if steps > 1 {
                    return Err(fail(
                        "torn_snapshot",
                        n,
                        format_args!(
                            "version {version}: one commit did {steps} operations of worker {worker}"
                        ),
                    ));
                }
                changed += steps;
                *floor = *place;
            }
            let lost = |detail: String| {
                Err(fail(
                    "lost_write",
                    n,
                    format_args!("version {version}: {detail}"),
                ))
            };
            match (contended_now, placed.contended) {
                (from, Contended::Untouched) if from != Contended::Untouched => {
                    return lost(format!(
                        "the keys went back to what they were at the start, from {from:?}"
                    ));
                }
                (Contended::Deleted, Contended::Applied(number))
                    if !model.upserts.contains(&number) =>
                {
                    return lost(format!("update {number} brought deleted keys back"));
                }
                (from, Contended::Applied(number)) if from != Contended::Applied(number) => {
                    if !contended_values.insert(number) {
                        return lost(format!(
                            "the keys went back to operation {number}, from {from:?}"
                        ));
                    }
                    changed += 1;
                }
                (from, Contended::Deleted) if from != Contended::Deleted => {
                    keys_went = true;
                    changed += 1;
                }
                _ => {}
            }
            contended_now = placed.contended;
            let moved = placed.increments - increments;
            if !(0..=1).contains(&moved) {
                return Err(fail(
                    "lost_write",
                    n,
                    format_args!(
                        "version {version}: the counters moved by {moved}, from {increments}"
                    ),
                ));
            }
            changed += moved as usize;
            increments = placed.increments;
            if changed > 1 {
                return Err(fail(
                    "torn_snapshot",
                    n,
                    format_args!("version {version}: one commit changed {changed} classes of rows"),
                ));
            }
            // After the rows, whose failure says more than what follows from it.
            self.rounds
                .as_mut()
                .expect("the state of the rounds")
                .lineage
                .advance(&image, n)?;
            Box::pin(oracle.check_indices(&ds, n, &image, false)).await?;
            last_placement = Some(placed);
            last_rows = Some(image.rows);
        }
        let state = self.rounds.as_mut().expect("the state of the rounds");
        let handle_reads = context.handle_reads.lock().unwrap();
        for (version, digest) in reads.iter().flatten().chain(handle_reads.iter()) {
            match state.digests.get(version) {
                Some(known) if known != digest => {
                    return Err(fail(
                        "snapshot_changed",
                        n,
                        format_args!(
                            "a reader or a worker read version {version} as {digest:016x}, and the history as {known:016x}"
                        ),
                    ));
                }
                // The first worker read its own hidden writes before the cleanup.
                None if hole && *version < boundary => {}
                None => {
                    return Err(fail(
                        "version_jump",
                        n,
                        format_args!(
                            "a reader or a worker read a version {version} that the history never saw"
                        ),
                    ));
                }
                Some(_) => {}
            }
        }
        // Where the round ended. Every operation has returned Ok by now.
        let placed = last_placement.expect("the versions read include the first of the round");
        for worker in 0..model.workers {
            let finished = model.states[worker].len() - 1;
            if !placed.prefixes[worker].contains(&finished) {
                return Err(fail(
                    "lost_write",
                    n,
                    format_args!(
                        "version {after}: the rows of worker {worker} are as of {:?}, after its {finished} operations",
                        placed.prefixes[worker]
                    ),
                ));
            }
        }
        if placed.increments != model.counter_ops {
            return Err(fail(
                "lost_write",
                n,
                format_args!(
                    "version {after}: the counters moved by {}, after {} increments",
                    placed.increments, model.counter_ops
                ),
            ));
        }
        let acted_on_keys =
            !(model.upserts.is_empty() && model.updates.is_empty() && model.deletes.is_empty());
        if acted_on_keys && placed.contended == Contended::Untouched {
            return Err(fail(
                "lost_write",
                n,
                format_args!(
                    "version {after}: the keys are as they were, after every operation on them"
                ),
            ));
        }
        // An upsert always writes. An update does if the keys are there, which only
        // a delete changes. `retain` keeps every version of the round.
        let writers: BTreeSet<i64> = if model.deletes.is_empty() {
            model.upserts.union(&model.updates).copied().collect()
        } else {
            model.upserts.clone()
        };
        let lost: Vec<&i64> = writers.difference(&contended_values).collect();
        if !lost.is_empty() {
            return Err(fail(
                "lost_write",
                n,
                format_args!(
                    "the operations {lost:?} returned Ok, but no version ever showed them"
                ),
            ));
        }
        if !model.deletes.is_empty() && !keys_went {
            return Err(fail(
                "lost_write",
                n,
                format_args!(
                    "the deletes {:?} returned Ok, but no version ever showed the keys gone",
                    model.deletes
                ),
            ));
        }
        let rows = last_rows.expect("a placed version has rows");
        let stride = model.stride;
        let class = model.workers as i64;
        *context.history_contended.lock().unwrap() = rows
            .into_iter()
            .filter(|(id, _)| id % stride == class)
            .collect();
        Ok(())
    }

    /// Counts the pairs that really met: `overlap:<X>:<Y>` when the manifest of Y
    /// landed between the start of X and the manifest of X, the window X has to
    /// conflict in. A transaction's `read_version` cannot tell, as a rebase of a
    /// delete or an update rewrites it. Disjoint fragments pass the rebase without
    /// a look, so `shared_fragment:<X>:<Y>` says that both also landed on a fragment
    /// in common. A pair that retried away from one shows in neither, but in
    /// `legal_error` and `internal_retry`.
    fn observe_overlaps(&self, n: usize, context: &RoundContext) {
        let spans = &context.clocks.spans;
        let tick = |value: &AtomicU64| value.load(Ordering::SeqCst);
        let touched = context.touched.lock().unwrap();
        let fragments = |op: &PlannedOp| touched.get(&tick(&spans[op.rank].version));
        if context.round.theme == Some("pair") {
            // The first worker's last operation: an append may come before it.
            let landed = context
                .round
                .ops
                .iter()
                .rfind(|op| op.worker == 0)
                .expect("a pair has an operation of the first worker");
            // A cleanup commits nothing, and has to be over before the others do.
            let landed_at = match landed.op {
                RoundOp::Cleanup { .. } => tick(&spans[landed.rank].end),
                _ => tick(&spans[landed.rank].commit),
            };
            // Once a round, as its others meet or miss together: forced by the holds
            // only, and a pair forced twice that never meets fails the run.
            let forced: BTreeSet<String> = context
                .round
                .ops
                .iter()
                .filter(|op| op.worker != 0 && op.hold.is_some())
                .map(|x| format!("pair_planned:{}:{}", x.op.kind(), landed.op.kind()))
                .collect();
            for planned in forced {
                self.observe(n, &planned);
            }
            for x in context.round.ops.iter().filter(|op| op.worker != 0) {
                let commit = tick(&spans[x.rank].commit);
                // Neither of them found anything to do.
                if commit == 0 || landed_at == 0 {
                    continue;
                }
                let met = if landed_at < commit { "met" } else { "missed" };
                self.observe(
                    n,
                    &format!("pair_{met}:{}:{}", x.op.kind(), landed.op.kind()),
                );
            }
        }
        for x in &context.round.ops {
            let span = &spans[x.rank];
            let begin = tick(&span.begin);
            // A cleanup has no commit of its own, but has a window. An operation
            // that found nothing to do has neither.
            let end = match (
                tick(&span.gave_up),
                tick(&span.commit),
                matches!(x.op, RoundOp::Cleanup { .. }),
            ) {
                (0, 0, true) => tick(&span.end),
                (0, 0, false) => continue,
                (0, commit, _) => commit,
                (gave_up, _, _) => gave_up,
            };
            for y in &context.round.ops {
                if y.worker == x.worker {
                    continue;
                }
                let commit = tick(&spans[y.rank].commit);
                if commit != 0 && begin < commit && commit < end {
                    self.observe(n, &format!("overlap:{}:{}", x.op.kind(), y.op.kind()));
                    if let (Some(ours), Some(theirs)) = (fragments(x), fragments(y))
                        && !ours.is_disjoint(theirs)
                    {
                        self.observe(
                            n,
                            &format!("shared_fragment:{}:{}", x.op.kind(), y.op.kind()),
                        );
                    }
                }
            }
        }
    }

    /// Upserts of the same keys by two workers raced when one committed while the
    /// other was under way. Over only new keys nothing but the filter of keys stops
    /// the duplicate, and with the index on the key there before the racers began,
    /// every racer took the indexed path of merge insert. Over a hole the refusal
    /// stops it before any filter is compared, so such a round counts no race.
    fn observe_key_race(&self, n: usize, context: &RoundContext) {
        if context.round.hole.is_some() {
            return;
        }
        let spans = &context.clocks.spans;
        let tick = |value: &AtomicU64| value.load(Ordering::SeqCst);
        let upserts: Vec<&PlannedOp> = context
            .round
            .ops
            .iter()
            .filter(|op| op.op == RoundOp::ContendedUpsert)
            .collect();
        let raced = upserts.iter().any(|x| {
            let span = &spans[x.rank];
            let begin = tick(&span.begin);
            // As in `observe_overlaps`: one that gave up idled from then on.
            let end = match (tick(&span.gave_up), tick(&span.commit)) {
                (0, commit) => commit,
                (gave_up, _) => gave_up,
            };
            upserts.iter().any(|y| {
                let commit = tick(&spans[y.rank].commit);
                y.worker != x.worker && commit != 0 && begin < commit && commit < end
            })
        });
        if !raced {
            return;
        }
        self.observe(n, "new_key_race");
        if !context.only_new_keys {
            return;
        }
        self.observe(n, "only_new_keys_race");
        // Built first in this round, and committed before any racer began.
        let built = tick(&spans[0].commit);
        let built_first = context.round.ops[0].op == RoundOp::CreateIndex(IndexKind::BTreeId)
            && built != 0
            && upserts.iter().all(|op| built < tick(&spans[op.rank].begin));
        if context.indexed_before || built_first {
            self.observe(n, "only_new_keys_race_indexed");
        }
    }
}

#[test]
fn selection_forces_only_the_pairs_of_the_focus() {
    let everything = BTreeSet::new();
    let mut holes_of_a_gate = BTreeSet::new();
    let mut met: BTreeSet<(Pair, Pair)> = BTreeSet::new();
    for seed in 1..=64 {
        // Without a focus, seeds take every pair in turn, as they always have.
        let (pair_rounds, hole) = forced_pairs(seed, 8, &everything);
        let turn = (seed % PAIRS.len() as u64) as usize * PAIR_ROUNDS;
        let expected: BTreeMap<usize, (&str, &str)> = [3, 5, 7]
            .into_iter()
            .enumerate()
            .map(|(slot, n)| (n, PAIRS[(turn + slot) % PAIRS.len()]))
            .collect();
        assert_eq!(pair_rounds, expected);
        let hole = hole.expect("a seed without a focus has a hole");
        if seed <= 16 {
            holes_of_a_gate.insert(hole);
        }
        met.insert((hole, expected[&3]));
    }
    assert_eq!(
        holes_of_a_gate.len(),
        HOLES.len(),
        "16 seeds meet every hole"
    );
    // Every hole meets every turn of the pairs, not the same pairs on every seed.
    assert_eq!(met.len(), HOLES.len() * PAIRS.len() / PAIR_ROUNDS);
    for op in focusable_ops() {
        let focus = BTreeSet::from([op]);
        let mut forced = 0;
        for seed in 1..=48 {
            let (pair_rounds, hole) = forced_pairs(seed, 8, &focus);
            for pair in pair_rounds.values() {
                assert!(in_focus(*pair, false, &focus), "{pair:?} for {op}");
            }
            if let Some(pair) = hole {
                assert!(
                    in_focus(pair, true, &focus),
                    "{pair:?} over a hole for {op}"
                );
            }
            forced += pair_rounds.len() + usize::from(hole.is_some());
        }
        assert!(forced > 0, "a focus on {op} forced no pair");
    }
    let (pair_rounds, hole) = forced_pairs(1, 8, &BTreeSet::from(["append"]));
    assert_eq!(pair_rounds.values().next(), Some(&("append", "cleanup")));
    assert_eq!(hole, None, "no pair over a hole appends");
}
