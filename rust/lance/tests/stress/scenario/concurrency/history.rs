// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The checks of what a round left: every version it made, the rows of each class,
//! the lineage of the rows, and the indices.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
use arrow_array::{Array, RecordBatch};
use lance::Dataset;
use lance::Error;
use lance::index::DatasetIndexExt;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};

use super::plan::{PlannedOp, Round, RoundOp, RowsModel};
use super::workers::ClockView;
use crate::engine::checks::{Answers, check_index, int64_column};
use crate::engine::data::{Scale, category, text, vector};
use crate::engine::ops::IndexKind;
use crate::engine::rows::{ClassRows, RowCounts, id_run};
use crate::engine::{Failure, fail};

/// What a rows's lineage columns hold. A fragment without version metadata leaves
/// the versions empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lineage {
    row_id: u64,
    created: Option<u64>,
    updated: Option<u64>,
}

/// The rows of one version.
pub struct Image {
    pub version: u64,
    pub rows: ClassRows,
    lineage: Option<BTreeMap<i64, Lineage>>,
}

impl Image {
    /// Equal rows give equal digests, in whatever order they were read.
    pub fn digest(&self) -> u64 {
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
pub struct Oracle {
    pub scale: Scale,
    pub data_seed: u64,
    pub stable_row_ids: bool,
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
    pub async fn image(
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
    pub async fn check_indices(
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
pub struct LineageTrack {
    last: BTreeMap<i64, (i64, Lineage)>,
    owners: BTreeMap<u64, i64>,
    /// The version `last` is of.
    seen: u64,
}

impl LineageTrack {
    /// Makes the track remember a row id that the row never had.
    pub fn tamper(&mut self) {
        if let Some((_, row)) = self.last.values_mut().next() {
            row.row_id ^= 1 << 40;
        }
    }

    pub fn start(&mut self, image: &Image) {
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

    pub fn advance(&mut self, image: &Image, n: usize) -> Result<(), Failure> {
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
pub enum Contended {
    Untouched,
    /// Every key exists and holds what this upsert or update wrote.
    Applied(i64),
    /// No key exists, after a delete.
    Deleted,
}

/// Where a snapshot stands in the round.
pub struct Placement {
    /// Per worker: how many of its operations the snapshot may include. Those
    /// that leave its rows as they are make this a range.
    pub prefixes: Vec<RangeInclusive<usize>>,
    /// The same, every place the rows match: an append and a delete of exactly
    /// those rows can bring a state back with others in between.
    pub places: Vec<Vec<usize>>,
    pub contended: Contended,
    pub increments: i64,
}

/// What each worker's rows hold after each of its operations, and what the
/// shared classes may hold.
pub struct RoundModel {
    pub workers: usize,
    pub stride: i64,
    pub states: Vec<Vec<ClassRows>>,
    keys: BTreeSet<i64>,
    contended_before: ClassRows,
    pub upserts: BTreeSet<i64>,
    pub updates: BTreeSet<i64>,
    pub deletes: BTreeSet<i64>,
    counters_before: ClassRows,
    pub counter_ops: i64,
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
    pub fn check_counts(
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

    pub fn new(rows: &RowsModel, round: &Round) -> Self {
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
    pub fn place(&self, image: &Image, n: usize, what: &str) -> Result<Placement, Failure> {
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
    pub fn check_bounds(
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
