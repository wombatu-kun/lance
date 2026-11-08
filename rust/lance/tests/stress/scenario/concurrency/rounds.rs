// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The rounds as a seed runs them, and the checks once a round is quiet.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::try_join_all;
use lance::dataset::builder::DatasetBuilder;
use lance::session::Session;
use tokio::sync::{Barrier, watch};

use super::history::{Contended, Oracle, RoundModel};
use super::plan::{PlannedOp, Round, RoundOp, RoundsPlan};
use super::workers::{
    Clocks, RoundContext, joined, run_op, run_reader, run_worker, touched_fragments,
};
use super::{GATE_HANG, GATE_LIMIT, LINEAGE, LOST_WRITE, RoundsState};
use crate::engine::ops::IndexKind;
use crate::engine::rows::ClassRows;
use crate::engine::run::Run;
use crate::engine::{Failure, fail};
use crate::harness::ENV_SERIAL;
use crate::incidents::Gate;

impl Run {
    pub async fn run_rounds(
        &mut self,
        plan: &RoundsPlan,
        state: &mut RoundsState,
    ) -> Result<(), Failure> {
        for (index, round) in plan.rounds.iter().enumerate() {
            Box::pin(self.run_round(index + 1, plan, round, state)).await?;
        }
        Ok(())
    }

    fn oracle(&self, plan: &RoundsPlan) -> Oracle {
        Oracle {
            scale: self.header.scale,
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
        state: &RoundsState,
    ) -> Result<RoundContext, Failure> {
        let verifier = self.open_verifier(n).await?;
        let versions = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let oldest = versions.first().map_or(before, |version| version.version);
        let gate_timeouts = Arc::new(AtomicU64::new(0));
        let gate_limit = (self.self_test != Some(&GATE_HANG)).then(|| GATE_LIMIT.get(&self.knobs));
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
            data_seed: self.header.data_seed,
            scale: self.header.scale,
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
        state: &mut RoundsState,
    ) -> Result<(), Failure> {
        self.state.set_phase(n.to_string());
        let before = self.latest_version(n).await?;
        if n == 1 {
            self.remember_first_version(n, plan, before, state).await?;
        }
        self.journal.line(format_args!(
            "step={n} start op=round theme={} ops={} readers={} before={before}",
            round.theme.unwrap_or("none"),
            round.ops.len(),
            round.readers.len()
        ));
        let context = Arc::new(self.round_context(n, plan, round, before, state).await?);
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
        Box::pin(self.conclude_round(n, plan, round, &context, before, reads, state)).await
    }

    /// The version a seed starts from is the only one no round has read.
    async fn remember_first_version(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        version: u64,
        state: &mut RoundsState,
    ) -> Result<(), Failure> {
        let oracle = self.oracle(plan);
        let ds = self.open_verifier(n).await?;
        let image = oracle
            .image(&ds, n, "the first version", true)
            .await?
            .map_err(|error| fail("unreadable", n, error))?;
        state.digests.insert(version, image.digest());
        state.lineage.start(&image);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn conclude_round(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        round: &Round,
        context: &RoundContext,
        before: u64,
        reads: Vec<Vec<(u64, u64)>>,
        state: &mut RoundsState,
    ) -> Result<(), Failure> {
        self.quiet_down(n).await;
        if self.self_test == Some(&LOST_WRITE) {
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
        } else if self.self_test == Some(&LINEAGE) {
            state.lineage.tamper();
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
        Box::pin(self.check_history(n, plan, context, before, after, &reads, state)).await?;
        self.observe_overlaps(n, context);
        if plan.declared_key {
            self.observe_key_race(n, context);
        }

        let contended_after = context.history_contended.lock().unwrap().clone();
        for op in &round.ops {
            state.rows.apply(op.worker, &op.op, op.number, &round.keys);
        }
        // The last of the shared upserts is the one that committed last.
        let class = state.rows.contended();
        state.rows.classes[class] = contended_after;
        let stamps = state.rows.all_rows();
        self.model.live = stamps.keys().copied().collect();
        self.model.stamps = Some(stamps);
        self.model.next_id = state.rows.next_id();
        for op in &round.ops {
            if let RoundOp::CreateIndex(kind) = op.op {
                self.model.indices.insert(kind);
            }
        }
        Box::pin(self.check_content(n)).await?;
        self.let_go_of_the_session(n, None, false).await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)?;
        self.journal.line(format_args!("step={n} done"));
        Ok(())
    }

    /// Opens every version the round made, and what a cleanup must have left of
    /// the older ones. Each is a snapshot like any other, and together they must
    /// tell one story: nothing goes back, and each commit does one thing.
    #[allow(clippy::too_many_arguments)]
    async fn check_history(
        &mut self,
        n: usize,
        plan: &RoundsPlan,
        context: &RoundContext,
        before: u64,
        after: u64,
        reads: &[Vec<(u64, u64)>],
        state: &mut RoundsState,
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
            let held: Vec<i64> = self.stamps().values().copied().collect();
            self.stamps_ever().extend(held);
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
            self.stamps_ever().extend(image.rows.values().copied());
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
            state.lineage.advance(&image, n)?;
            Box::pin(oracle.check_indices(&ds, n, &image, false)).await?;
            last_placement = Some(placed);
            last_rows = Some(image.rows);
        }
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

    fn stamps(&self) -> &ClassRows {
        self.model
            .stamps
            .as_ref()
            .expect("the rounds keep `gen` in the model")
    }

    fn stamps_ever(&mut self) -> &mut BTreeSet<i64> {
        self.model
            .stamps_ever
            .as_mut()
            .expect("the rounds keep `gen` in the model")
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
            let forced: BTreeSet<&str> = context
                .round
                .ops
                .iter()
                .filter(|op| op.worker != 0 && op.hold.is_some())
                .map(|x| x.op.kind())
                .collect();
            for planned in forced {
                self.observe(n, &format!("pair_planned:{planned}:{}", landed.op.kind()));
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
