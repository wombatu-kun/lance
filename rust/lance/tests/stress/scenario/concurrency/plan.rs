// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The plan of the rounds: what each worker runs, which pairs a round forces, the
//! hole of round 1, and which errors Lance may legally return.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use lance::Error;

use super::{
    CATCH_UP, CONTENDED_EXISTING_KEYS, CONTENDED_KINDS, CONTENDED_NEW_KEYS, HOLD, MIN_CLASS_ROWS,
    NUMBERS_PER_ROUND, PARALLEL_ATTEMPTS, WORKERS,
};
use crate::engine::data::Scale;
use crate::engine::ops::{IndexKind, OpOutput};
use crate::engine::rows::{ClassRows, id_run};
use crate::engine::steps::pick_from;
use crate::engine::{PlanRng, StoreChoice};
use crate::incidents::{Area, Incident, Method};
use crate::knobs::Knobs;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundOp {
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
    pub fn kind(&self) -> &'static str {
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
    pub fn is_legal_error(&self, error: &Error, round: &Round, rank: usize) -> bool {
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
    pub fn is_contended(&self) -> bool {
        matches!(
            self,
            Self::ContendedUpsert | Self::ContendedUpdate | Self::ContendedDelete
        )
    }

    /// Whether it changes rows rather than how they are stored or found.
    pub fn is_row_op(&self) -> bool {
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
    pub fn commits(&self, output: Option<&OpOutput>) -> (u64, u64) {
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
pub struct Hold {
    pub point: Incident,
    pub until: usize,
    pub arrival: bool,
}

#[derive(Clone, Debug)]
pub struct PlannedOp {
    /// Its place in one order of the round that keeps each worker's own order.
    pub rank: usize,
    pub worker: usize,
    /// What it writes to `gen`, which tells it from every other operation.
    pub number: i64,
    /// Whether the worker's handle catches up with the latest version first.
    pub catch_up: bool,
    pub op: RoundOp,
    pub hold: Option<Hold>,
    /// It begins only once the operation of this rank is over. The wait has no
    /// limit: what it must see is that operation's result.
    pub start_after: Option<usize>,
    /// Its first try is prepared on the version the round starts from before
    /// anyone writes, and committed only once it may begin.
    pub staged: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderKind {
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
    pub fn name(self) -> &'static str {
        match self {
            Self::FreshOpen => "fresh_open",
            Self::CheckoutLatest => "checkout_latest",
            Self::Pinned => "pinned",
            Self::OldSnapshot => "old_snapshot",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlannedReader {
    pub kind: ReaderKind,
    pub hold: Option<Hold>,
    /// It begins reading only once the operation of this rank is over.
    pub start_after: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Round {
    pub theme: Option<&'static str>,
    pub keys: Vec<i64>,
    pub ops: Vec<PlannedOp>,
    pub readers: Vec<PlannedReader>,
    /// In a round of `HOLES`, what the first worker hides from the others.
    pub hole: Option<&'static str>,
}

impl Round {
    pub fn ops_of(&self, worker: usize) -> Vec<PlannedOp> {
        self.ops
            .iter()
            .filter(|op| op.worker == worker)
            .cloned()
            .collect()
    }

    pub fn has_cleanup(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op.op, RoundOp::Cleanup { .. }))
    }
}

/// What a seed decides about its rounds.
#[derive(Clone, Debug)]
pub struct RoundsPlan {
    pub workers: usize,
    pub stable_row_ids: bool,
    /// One `Session` for the whole round, so that its caches are shared, or one
    /// per handle, as separate processes would have.
    pub shared_session: bool,
    /// `id` is declared the primary key, which promises a conflict between two
    /// inserts of one new key. Only such a seed races for new keys: without a
    /// declared key a duplicate is legal, as `conflict_resolver.rs` only promises
    /// the conflict for one, see #5633.
    pub declared_key: bool,
    pub rounds: Vec<Round>,
}

/// The rows the plan expects, class by class.
#[derive(Clone, Debug)]
pub struct RowsModel {
    pub workers: usize,
    pub classes: Vec<ClassRows>,
    next_ids: Vec<i64>,
}

impl RowsModel {
    pub fn new(initial_rows: i64, workers: usize) -> Self {
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

    pub fn stride(&self) -> i64 {
        self.workers as i64 + 2
    }

    pub fn contended(&self) -> usize {
        self.workers
    }

    pub fn counters(&self) -> usize {
        self.workers + 1
    }

    pub fn apply(&mut self, worker: usize, op: &RoundOp, number: i64, keys: &[i64]) {
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

    pub fn all_rows(&self) -> ClassRows {
        self.classes
            .iter()
            .flatten()
            .map(|(k, v)| (*k, *v))
            .collect()
    }

    pub fn next_id(&self) -> i64 {
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
pub const HOLES: [(&str, &str); 8] = [
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
pub fn focusable_ops() -> Vec<&'static str> {
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
pub fn plan_rounds(
    rng: &mut PlanRng,
    seed: u64,
    store: StoreChoice,
    scale: &Scale,
    initial_rows: i64,
    rounds: usize,
    with_holds: bool,
    focus: &BTreeSet<&'static str>,
    knobs: &Knobs,
) -> RoundsPlan {
    let workers = rng.range(WORKERS.get(knobs)) as usize;
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
                knobs,
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
    knobs: &Knobs,
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
        let roll = rng.chance(HOLD.get(knobs));
        // Without a catch up, the handle stands on the version the round starts from
        // for as long as the other takes.
        let mut catch_up = rng.chance(CATCH_UP.get(knobs));
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
