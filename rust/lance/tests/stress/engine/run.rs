// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The state of a seed in its process, and the checks every scenario shares.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_io::object_store::ObjectStoreParams;
use object_store::path::Path as ObjectPath;
use object_store::{PutOptions, PutPayload};
use roaring::RoaringBitmap;

use crate::harness::CALLS_LOG;
use crate::incidents::{IncidentState, StressHooks, StressWrapper};
use crate::leaks::{Snapshot, files_below, quiesce};
use crate::limits::OwnCgroup;
use crate::scenario::{Scenario, SelfTest, nothing_more};

use super::checks::{
    Answers, check_index, compare_ids, covers_a_fragment, file_sizes, file_times, files_by_version,
    ids_digest, int64_column, listed_versions, paths,
};
use super::data::{category, text, vector};
use super::ops::{Model, Op};
use super::steps::CleanupFiles;
use super::{Body, ChildSetting, ContentRules, Failure, Header, Journal, StoreChoice, fail};

/// An unreferenced file that the final cleanup must remove, planted after it.
pub const ORPHAN: SelfTest = SelfTest {
    name: "orphan",
    expected: &["orphans"],
    reason: None,
    verify: nothing_more,
};

pub struct Run {
    pub scenario: &'static dyn Scenario,
    pub header: Header,
    pub content: ContentRules,
    pub generation: u64,
    /// The first step this process has to run; `resume` moves it.
    pub next_step: usize,
    /// The seed the checks recompute vectors from; a self-test changes it.
    pub oracle_seed: u64,
    pub uri: String,
    /// The object path of the dataset root.
    pub root: String,
    pub tmp: PathBuf,
    pub journal: Arc<Journal>,
    /// The logs of the body's own, by name.
    journals: Vec<(&'static str, Arc<Journal>)>,
    pub state: Arc<IncidentState>,
    pub hooks: Arc<StressHooks>,
    pub wrapper: Arc<StressWrapper>,
    /// Where this process lives, under a scenario with memory limits.
    pub cgroup: Option<OwnCgroup>,
    /// The handle every operation runs on; reads that check it go through a
    /// fresh handle without the wrapper.
    pub ds: Dataset,
    /// The latest commit is one `ds` never saw: its operation committed but
    /// did not return Ok.
    pub handle_stale: bool,
    pub model: Model,
    /// A cleanup was allowed to remove every version before this one, so such a
    /// version may be gone, or listed without some of its files when a cleanup
    /// stopped half way.
    pub expired_below: u64,
    /// What each version held when it was the latest, as `ids_digest` puts it.
    pub digests: BTreeMap<u64, String>,
    /// The expired versions already reported without some of their files.
    pub broken_reported: BTreeSet<u64>,
    /// What the cleanup about to run must remove and must leave.
    pub cleanup_files: Option<CleanupFiles>,
    /// The fragments of a stale handle that an index optimization starts on:
    /// those are what it covers.
    pub read_fragments: Option<RoaringBitmap>,
    pub baseline_tasks: usize,
    baseline_fds: usize,
    pub quiet: Duration,
    pub self_test: Option<&'static SelfTest>,
}

impl Run {
    pub async fn setup(
        setting: ChildSetting,
        header: Header,
        body: &dyn Body,
        dir: &Path,
        journal: Arc<Journal>,
    ) -> Result<Self, Failure> {
        let ChildSetting {
            scenario,
            quiet,
            generation,
            self_test,
            started,
        } = setting;
        let (uri, root) = match header.store {
            StoreChoice::Local | StoreChoice::FileObjectStore => {
                let path = dir
                    .join("dataset")
                    .to_str()
                    .expect("a UTF-8 artifacts path")
                    .to_string();
                let root = path.trim_start_matches('/').to_string();
                match header.store {
                    StoreChoice::Local => (path, root),
                    _ => (format!("file-object-store://{path}"), root),
                }
            }
            StoreChoice::SharedMemory => (
                format!("shared-memory://stress-{}/ds", std::process::id()),
                "ds".to_string(),
            ),
        };
        let state = Arc::new(IncidentState::new(
            root.clone(),
            &dir.join(CALLS_LOG),
            started,
        ));
        let hooks = Arc::new(StressHooks::new(state.clone()));
        for check in scenario.checks() {
            check
                .start(dir, started, generation, state.clone())
                .map_err(|error| fail("harness", 0, error))?;
        }
        journal.line(format_args!(
            "setup store={:?} uri={uri} rows={} generation={generation}",
            header.store, header.initial_rows
        ));
        let wrapper = Arc::new(StressWrapper::new(state.clone()));
        let store_params = ObjectStoreParams {
            object_store_wrapper: Some(wrapper.clone()),
            ..Default::default()
        };
        let journals = body
            .journals()
            .iter()
            .map(|name| (*name, Arc::new(Journal::open(&dir.join(name), started))))
            .collect();
        let ds = if generation == 1 {
            let params = WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: header.scale.rows_per_file,
                store_params: Some(store_params),
                progress: hooks.clone(),
                enable_stable_row_ids: body.stable_row_ids(),
                ..Default::default()
            };
            let initial = body.initial_source(&header);
            Box::pin(Dataset::write(initial, uri.as_str(), Some(params))).await
        } else {
            Box::pin(
                DatasetBuilder::from_uri(&uri)
                    .with_store_params(store_params)
                    .load(),
            )
            .await
        }
        // A dead process must not take the dataset with it.
        .map_err(|error| {
            let class = if generation == 1 {
                "setup"
            } else {
                "unreadable"
            };
            fail(class, 0, error)
        })?;
        let mut run = Self {
            scenario,
            oracle_seed: header.data_seed,
            model: body.initial_model(&header),
            header,
            content: body.content(),
            generation,
            next_step: 0,
            uri,
            root,
            tmp: dir.join("tmp"),
            journal,
            journals,
            state,
            hooks,
            wrapper,
            cgroup: OwnCgroup::from_env().map_err(|error| fail("harness", 0, error))?,
            ds,
            handle_stale: false,
            expired_below: 0,
            digests: BTreeMap::new(),
            broken_reported: BTreeSet::new(),
            cleanup_files: None,
            read_fragments: None,
            baseline_tasks: 0,
            baseline_fds: 0,
            quiet,
            self_test,
        };
        if generation == 1 {
            Box::pin(run.measure_baseline(0)).await?;
        }
        Ok(run)
    }

    /// What the runtime and the process keep between operations, measured
    /// after a first read, which starts the pools that stay.
    pub async fn measure_baseline(&mut self, n: usize) -> Result<(), Failure> {
        Box::pin(self.check_content(n)).await?;
        let (snapshot, waited) = quiesce(0, &self.tmp, self.quiet).await;
        self.baseline_tasks = snapshot.tasks;
        self.baseline_fds = snapshot.fds;
        self.journal.line(format_args!(
            "setup baseline waited_ms={} {}",
            waited.as_millis(),
            snapshot.line()
        ));
        self.check_leaks(n, &snapshot)
    }

    /// A local object store writes `name` by way of a file `name#digits`, and
    /// its listings hide those. So a death inside a write can leave a file that
    /// neither a listing nor a cleanup through the store ever sees: only the
    /// file system tells.
    /// Files on the disk that the store does not list, such as a staged upload;
    /// how many there are.
    pub fn observe_hidden_files(
        &self,
        n: usize,
        observation: &str,
        listed: &BTreeMap<String, i64>,
    ) -> usize {
        let dataset = self.seed_dir().join("dataset");
        let hidden: Vec<PathBuf> = files_below(&dataset)
            .into_iter()
            .filter(|file| {
                file.strip_prefix(&dataset)
                    .ok()
                    .and_then(Path::to_str)
                    .is_some_and(|path| !listed.contains_key(path))
            })
            .collect();
        if hidden.is_empty() {
            return 0;
        }
        let bytes: u64 = hidden
            .iter()
            .filter_map(|file| fs::metadata(file).ok())
            .map(|meta| meta.len())
            .sum();
        self.journal.line(format_args!(
            "step={n} observation={observation} files={} bytes={bytes} detail={hidden:?}",
            hidden.len()
        ));
        hidden.len()
    }

    pub fn seed_dir(&self) -> &Path {
        self.tmp.parent().expect("the seed's directory")
    }

    /// A log of the body's own, which the setup opened.
    pub fn journal_of(&self, name: &str) -> Arc<Journal> {
        self.journals
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, journal)| journal.clone())
            .unwrap_or_else(|| panic!("the body names no log {name}"))
    }

    /// Replaces the operations' handle with a fresh one, in a session of its
    /// own. An operation that reads through an index leaves it in the cache of
    /// its session, and with it the task of the index store's I/O scheduler
    /// (`ScanScheduler`) until the session goes. Only what outlives the handle
    /// is a leak. A scenario that keeps one session does so for its caches to see
    /// everything that follows, but for an index optimization, which reads the
    /// segments it merges through it.
    pub async fn let_go_of_the_session(
        &mut self,
        n: usize,
        op: Option<&Op>,
        keeps_session: bool,
    ) -> Result<(), Failure> {
        if keeps_session && !matches!(op, Some(Op::OptimizeIndices)) {
            return Ok(());
        }
        self.ds = Box::pin(
            DatasetBuilder::from_uri(&self.uri)
                .with_store_params(ObjectStoreParams {
                    object_store_wrapper: Some(self.wrapper.clone()),
                    ..Default::default()
                })
                .load(),
        )
        .await
        .map_err(|error| fail("unreadable", n, error))?;
        self.handle_stale = false;
        Ok(())
    }

    pub async fn catch_up(&mut self, n: usize) -> Result<(), Failure> {
        Box::pin(self.ds.checkout_latest())
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        self.handle_stale = false;
        Ok(())
    }

    pub fn observe(&self, n: usize, observation: &str) {
        self.journal
            .line(format_args!("step={n} observation={observation}"));
    }

    /// An operation that returned Ok must leave its handle on its own commit.
    pub fn check_handle(
        &self,
        n: usize,
        op: &Op,
        handle: u64,
        version: u64,
    ) -> Result<(), Failure> {
        if handle != version {
            return Err(fail(
                "stale_handle",
                n,
                format_args!(
                    "{op} returned Ok, but its handle is at version {handle}, not {version}"
                ),
            ));
        }
        Ok(())
    }

    pub async fn open_verifier(&self, n: usize) -> Result<Dataset, Failure> {
        Box::pin(Dataset::open(&self.uri))
            .await
            .map_err(|error| fail("unreadable", n, error))
    }

    /// The version a fresh open lands on.
    pub async fn opened_version(&self, n: usize) -> Result<u64, Failure> {
        Ok(self.open_verifier(n).await?.version().version)
    }

    /// The latest version, as a fresh handle without the wrapper sees it, once
    /// nothing is landing any more: three ways of asking must agree.
    pub async fn latest_version(&self, n: usize) -> Result<u64, Failure> {
        let verifier = self.open_verifier(n).await?;
        let opened = verifier.version().version;
        let latest = verifier
            .latest_version_id()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let listed = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?
            .iter()
            .map(|version| version.version)
            .max()
            .unwrap_or_default();
        if opened != latest || latest != listed {
            return Err(fail(
                "version_jump",
                n,
                format_args!(
                    "a fresh open lands on version {opened}, latest_version_id says {latest}, and the newest listed manifest is {listed}"
                ),
            ));
        }
        Ok(latest)
    }

    pub async fn check_content(&mut self, n: usize) -> Result<(), Failure> {
        let unreadable = |error: lance::Error| fail("unreadable", n, error);
        let ds = self.open_verifier(n).await?;
        Box::pin(ds.validate())
            .await
            .map_err(|error| fail("validate", n, error))?;
        self.check_rows(&ds, n).await?;
        let latest = ds.version().version;
        if self.content.versions_stay_listed {
            let listed = listed_versions(&ds, n).await?;
            let gone: Vec<u64> = (self.expired_below.max(1)..=latest)
                .filter(|version| !listed.contains(version))
                .collect();
            if !gone.is_empty() {
                return Err(fail(
                    "version_jump",
                    n,
                    format_args!(
                        "versions {gone:?} are gone, though cleanups could only remove the versions before {}",
                        self.expired_below
                    ),
                ));
            }
        }
        let all = file_times(ds.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let tracked = files_by_version(ds.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let mut missing = Vec::new();
        let mut broken = BTreeSet::new();
        for (path, versions) in tracked.iter().filter(|(path, _)| !all.contains_key(*path)) {
            // A cleanup deletes the files of the versions it expires before
            // their manifests, so one that stopped half way leaves such a version
            // listed without some of them.
            match versions.last() {
                Some(newest) if *newest >= self.expired_below => {
                    missing.push(format!("{path} of versions {versions:?}"));
                }
                _ => broken.extend(versions),
            }
        }
        if !missing.is_empty() {
            return Err(fail(
                "missing_files",
                n,
                format_args!("manifests reference files that are not there: {missing:?}"),
            ));
        }
        let newly_broken: Vec<String> = broken
            .difference(&self.broken_reported)
            .map(u64::to_string)
            .collect();
        if !newly_broken.is_empty() {
            self.journal.line(format_args!(
                "step={n} observation=expired_version_broken versions={}",
                newly_broken.join(",")
            ));
            self.broken_reported.extend(broken);
        }
        let indices = ds.load_indices().await.map_err(unreadable)?;
        let names: BTreeSet<String> = indices.iter().map(|index| index.name.clone()).collect();
        let expected: BTreeSet<String> = self
            .model
            .indices
            .iter()
            .map(|kind| kind.name().to_string())
            .collect();
        if names != expected {
            return Err(fail(
                "content",
                n,
                format_args!("the indices are {names:?} where the model has {expected:?}"),
            ));
        }
        for kind in &self.model.indices {
            let answers = Answers {
                live: &self.model.live,
                next_id: self.model.next_id,
                stamps: self.model.stamps.as_ref(),
                stamps_ever: self.model.stamps_ever.as_ref(),
                few_stamps: false,
                scale: self.header.scale,
                data_seed: self.oracle_seed,
                check_plan: self.content.indices_cover_the_model
                    || covers_a_fragment(&ds, &indices, kind.name()),
                salt: n,
            };
            Box::pin(check_index(&ds, n, *kind, &answers)).await?;
        }
        if self.content.digests {
            // The rows of the latest version are the model's by now.
            let digest = ids_digest(self.model.live.iter().copied());
            match self.digests.get(&latest) {
                Some(known) if *known != digest => {
                    return Err(fail(
                        "snapshot_changed",
                        n,
                        format_args!(
                            "version {latest} reads as {digest}, and read as {known} before"
                        ),
                    ));
                }
                Some(_) => {}
                None => {
                    self.journal
                        .line(format_args!("step={n} digest_of={latest} ids={digest}"));
                    self.digests.insert(latest, digest);
                }
            }
        }
        Ok(())
    }

    /// Only a failed deletion of its manifest leaves an expired version behind.
    pub async fn check_expired_gone(
        &self,
        n: usize,
        boundary: u64,
        may_stay: u64,
    ) -> Result<(), Failure> {
        let left: Vec<u64> = listed_versions(&self.open_verifier(n).await?, n)
            .await?
            .into_iter()
            .filter(|version| *version < boundary)
            .collect();
        if left.len() as u64 > may_stay {
            return Err(fail(
                "cleanup",
                n,
                format_args!(
                    "the cleanup returned Ok, but versions {left:?} are still there, below its boundary {boundary}, where its failed deletions allow {may_stay}"
                ),
            ));
        }
        Ok(())
    }

    /// Every version the cleanups keep opens, reads as it did when it was the
    /// latest, and has the files of its indices, which `check_content` sees
    /// only by listing their directories.
    pub async fn check_kept_versions(&self, n: usize) -> Result<(), Failure> {
        let verifier = self.open_verifier(n).await?;
        let all = file_times(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let kept = listed_versions(&verifier, n)
            .await?
            .into_iter()
            .filter(|version| *version >= self.expired_below);
        for version in kept {
            let unreadable = |error: lance::Error| {
                fail("unreadable", n, format_args!("version {version}: {error}"))
            };
            let ds = verifier
                .checkout_version(version)
                .await
                .map_err(unreadable)?;
            let mut scan = ds.scan();
            scan.project(&["id"]).map_err(unreadable)?;
            let batch = scan.try_into_batch().await.map_err(unreadable)?;
            let digest = ids_digest(int64_column(&batch, "id"));
            if let Some(known) = self.digests.get(&version)
                && *known != digest
            {
                return Err(fail(
                    "snapshot_changed",
                    n,
                    format_args!(
                        "version {version} reads as {digest}, and read as {known} when it was the latest"
                    ),
                ));
            }
            for index in ds.load_indices().await.map_err(unreadable)?.iter() {
                let Some(files) = &index.files else {
                    continue;
                };
                let missing: Vec<String> = files
                    .iter()
                    .map(|file| format!("_indices/{}/{}", index.uuid, file.path))
                    .filter(|path| !all.contains_key(path))
                    .collect();
                if !missing.is_empty() {
                    return Err(fail(
                        "missing_files",
                        n,
                        format_args!(
                            "version {version} keeps the index {} without its files {missing:?}",
                            index.name
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Every row must be there once, with every column right.
    async fn check_rows(&self, ds: &Dataset, n: usize) -> Result<(), Failure> {
        let unreadable = |error: lance::Error| fail("unreadable", n, error);
        let batch = ds.scan().try_into_batch().await.map_err(unreadable)?;
        let ids = int64_column(&batch, "id");
        compare_ids(n, "a full scan", &ids, &self.model.live)?;
        let count = ds.count_rows(None).await.map_err(unreadable)?;
        if count != self.model.live.len() {
            return Err(fail(
                "content",
                n,
                format_args!(
                    "count_rows says {count} where the model has {}",
                    self.model.live.len()
                ),
            ));
        }
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
            } else if texts.value(row) != text(*id, &self.header.scale) {
                Some("text")
            } else if vectors
                .value(row)
                .as_primitive::<Float32Type>()
                .values()
                .as_ref()
                != vector(self.oracle_seed, *id).as_slice()
            {
                Some("vec")
            } else {
                None
            };
            if let Some(column) = wrong {
                return Err(fail(
                    "content",
                    n,
                    format_args!("row {id} holds a wrong {column}"),
                ));
            }
        }
        Ok(())
    }

    pub async fn quiet_down(&self, n: usize) -> Snapshot {
        let (snapshot, waited) = quiesce(self.baseline_tasks, &self.tmp, self.quiet).await;
        self.journal.line(format_args!(
            "step={n} quiet waited_ms={} {}",
            waited.as_millis(),
            snapshot.line()
        ));
        snapshot
    }

    pub fn check_leaks(&self, n: usize, snapshot: &Snapshot) -> Result<(), Failure> {
        if snapshot.tasks > self.baseline_tasks {
            return Err(fail(
                "task_leak",
                n,
                format_args!(
                    "{} tokio tasks are alive {}s after the step, {} before the first one",
                    snapshot.tasks,
                    self.quiet.as_secs(),
                    self.baseline_tasks
                ),
            ));
        }
        if !snapshot.tmp_files.is_empty() || snapshot.tmp_dirs > 0 {
            return Err(fail(
                "tmp_leak",
                n,
                format_args!(
                    "temporary files left behind: {:?}, and {} directories",
                    snapshot.tmp_files, snapshot.tmp_dirs
                ),
            ));
        }
        if snapshot.fds > self.baseline_fds {
            return Err(fail(
                "fd_leak",
                n,
                format_args!(
                    "{} open descriptors after the step, {} before the first one",
                    snapshot.fds, self.baseline_fds
                ),
            ));
        }
        Ok(())
    }

    /// Cleans up everything but the latest version, then checks that no
    /// unreferenced file is left and the dataset still reads right. Its step is
    /// `n`, the one after the plan.
    pub async fn finish(&mut self, n: usize) -> Result<(), Failure> {
        self.state.set_phase("final");
        self.journal.line(format_args!("step={n} final"));
        self.catch_up(n).await?;
        // Every version but the latest may go now.
        self.expired_below = self.ds.version().version;
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&self.ds, 1)
            .await
            .map_err(|error| fail("cleanup", n, error))?
            .delete_unverified(true)
            .error_if_tagged_old_versions(false)
            .build();
        let removed = self
            .ds
            .cleanup_with_policy(policy)
            .await
            .map_err(|error| fail("cleanup", n, error))?;
        self.journal.line(format_args!(
            "step={n} cleanup old_versions={} data_files={} index_files={} transaction_files={} deletion_files={} failed_deletes={}",
            removed.old_versions,
            removed.data_files_removed,
            removed.index_files_removed,
            removed.transaction_files_removed,
            removed.deletion_files_removed,
            removed.failed_deletes
        ));
        // It runs without an incident or a limit, so nothing explains a failure.
        if removed.failed_deletes > 0 {
            return Err(fail(
                "cleanup",
                n,
                format_args!(
                    "the final cleanup failed {} deletions",
                    removed.failed_deletes
                ),
            ));
        }
        self.check_expired_gone(n, self.expired_below, 0).await?;
        if self.self_test == Some(&ORPHAN) {
            let store = self
                .ds
                .object_store(None)
                .await
                .map_err(|error| fail("unreadable", n, error))?;
            let path = ObjectPath::from(format!("{}/data/self-test-orphan.lance", self.root));
            store
                .inner
                .put_opts(
                    &path,
                    PutPayload::from_static(b"orphan"),
                    PutOptions::default(),
                )
                .await
                .expect("plant an orphan");
        }
        self.check_orphans(n).await?;
        Box::pin(self.check_content(n)).await?;
        // The queries of an index need not touch every file of it.
        Box::pin(self.check_kept_versions(n)).await?;
        let snapshot = self.quiet_down(n).await;
        self.check_leaks(n, &snapshot)
    }

    /// After a cleanup that may delete unverified files, with nothing in flight,
    /// no file is left that no version references.
    pub async fn check_orphans(&self, n: usize) -> Result<(), Failure> {
        let verifier = self.open_verifier(n).await?;
        let listed = file_sizes(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let hidden = self.observe_hidden_files(n, "hidden_after_cleanup", &listed);
        if self.content.hidden_files_fail && hidden > 0 {
            return Err(fail(
                "orphans",
                n,
                format_args!(
                    "{hidden} files the store does not list, see observation=hidden_after_cleanup"
                ),
            ));
        }
        // As cleanup takes it: the earliest commit time of the versions it keeps,
        // which a clock that stepped back makes other than that of the oldest one.
        let cutoff = verifier
            .versions()
            .await
            .map_err(|error| fail("unreadable", n, error))?
            .iter()
            .map(|version| version.timestamp.timestamp_micros())
            .min()
            .unwrap_or(i64::MAX);
        let all = file_times(verifier.all_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let tracked = paths(verifier.tracked_files().await)
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        let mut orphans = Vec::new();
        for (path, modified) in &all {
            // A version hint is rewritten in place rather than referenced.
            if tracked.contains(path) || path.ends_with("latest_version_hint.json") {
                continue;
            }
            match path.split('/').next() {
                // Cleanup lists these only up to that time, to the nanosecond
                // (`listing_unmodified_since` in `cleanup.rs`), and these times
                // are in microseconds.
                Some(area @ ("_transactions" | "_deletions")) if *modified >= cutoff => {
                    self.observe(n, &format!("orphan{area}"));
                }
                _ => orphans.push(path.clone()),
            }
        }
        if !orphans.is_empty() {
            return Err(fail(
                "orphans",
                n,
                format_args!("unreferenced files after the cleanup: {orphans:?}"),
            ));
        }
        // A file in a committed index's directory counts as tracked whether
        // the index lists it or not.
        let indices = verifier
            .load_indices()
            .await
            .map_err(|error| fail("unreadable", n, error))?;
        for index in indices.iter() {
            let Some(files) = &index.files else {
                continue;
            };
            let directory = format!("_indices/{}/", index.uuid);
            let listed: BTreeSet<String> = files
                .iter()
                .map(|file| format!("{directory}{}", file.path))
                .collect();
            for path in all
                .keys()
                .filter(|path| path.starts_with(&directory) && !listed.contains(*path))
            {
                self.journal.line(format_args!(
                    "step={n} observation=index_dir_extra path={path}"
                ));
            }
        }
        Ok(())
    }
}
