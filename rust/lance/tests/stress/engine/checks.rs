// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What the checks of a seed read: listings of files and versions, and the answers of
//! the indices.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, TimestampMicrosecondType};
use arrow_array::{Float32Array, RecordBatch};
use futures::{Stream, TryStreamExt};
use lance::Dataset;
use lance_index::scalar::FullTextSearchQuery;
use lance_table::format::IndexMetadata;

use super::data::{
    ALL_PARTITIONS, CATEGORIES, NEIGHBORS, Scale, VECTOR_DIM, category, squared_l2, tokens, vector,
};
use super::ops::IndexKind;
use super::{Failure, fail, mix};

pub fn int64_column(batch: &RecordBatch, name: &str) -> Vec<i64> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no {name} column"))
        .as_primitive::<Int64Type>()
        .values()
        .to_vec()
}

fn describe_difference(expected: &BTreeSet<i64>, actual: &BTreeSet<i64>) -> String {
    let missing: Vec<i64> = expected.difference(actual).take(5).copied().collect();
    let unexpected: Vec<i64> = actual.difference(expected).take(5).copied().collect();
    format!(
        "{} rows where the model has {}; missing {missing:?}, unexpected {unexpected:?} (the first five of each)",
        actual.len(),
        expected.len()
    )
}

/// Checks a query's rows against the model, and that none comes back twice.
pub fn compare_ids(
    n: usize,
    what: &str,
    found: &[i64],
    expected: &BTreeSet<i64>,
) -> Result<(), Failure> {
    let distinct: BTreeSet<i64> = found.iter().copied().collect();
    if distinct.len() != found.len() {
        return Err(fail(
            "content",
            n,
            format_args!(
                "{what}: {} rows but {} distinct ids",
                found.len(),
                distinct.len()
            ),
        ));
    }
    if &distinct != expected {
        return Err(fail(
            "content",
            n,
            format_args!("{what}: {}", describe_difference(expected, &distinct)),
        ));
    }
    Ok(())
}

pub async fn paths<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeSet<String>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    Ok(batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("path")
                .expect("a path column")
                .as_string::<i32>()
                .iter()
                .flatten()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect())
}

/// The files `Dataset::tracked_files` lists, with the versions that reference
/// each of them.
pub async fn files_by_version<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeMap<String, BTreeSet<u64>>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut files: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    for batch in &batches {
        let paths = batch
            .column_by_name("path")
            .expect("a path column")
            .as_string::<i32>();
        for (row, version) in int64_column(batch, "version").into_iter().enumerate() {
            files
                .entry(paths.value(row).to_string())
                .or_default()
                .insert(version as u64);
        }
    }
    Ok(files)
}

/// The versions whose manifests a listing finds, without reading them, as
/// `retain_n_versions` does.
pub async fn listed_versions(ds: &Dataset, n: usize) -> Result<BTreeSet<u64>, Failure> {
    Ok(ds
        .version_refs()
        .await
        .map_err(|error| fail("unreadable", n, error))?
        .iter()
        .map(|version| version.version)
        .collect())
}

/// The rows of a version by their ids: how many, and a sum that does not
/// depend on their order.
pub fn ids_digest(ids: impl IntoIterator<Item = i64>) -> String {
    let (count, sum) = ids.into_iter().fold((0usize, 0u64), |(count, sum), id| {
        (count + 1, sum.wrapping_add(mix(id as u64)))
    });
    format!("{count}:{sum:016x}")
}

/// The files `Dataset::all_files` lists, with their modification times in
/// microseconds since the epoch.
pub async fn file_times<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeMap<String, i64>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut times = BTreeMap::new();
    for batch in &batches {
        let paths = batch
            .column_by_name("path")
            .expect("a path column")
            .as_string::<i32>();
        let modified = batch
            .column_by_name("last_modified")
            .expect("a last_modified column")
            .as_primitive::<TimestampMicrosecondType>();
        for row in 0..batch.num_rows() {
            times.insert(paths.value(row).to_string(), modified.value(row));
        }
    }
    Ok(times)
}

/// The files `Dataset::all_files` lists, with their sizes in bytes.
pub async fn file_sizes<E: fmt::Display>(
    stream: impl Stream<Item = Result<RecordBatch, E>>,
) -> Result<BTreeMap<String, i64>, E> {
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut sizes = BTreeMap::new();
    for batch in &batches {
        let paths = batch
            .column_by_name("path")
            .expect("a path column")
            .as_string::<i32>();
        let bytes = batch
            .column_by_name("size_bytes")
            .expect("a size_bytes column")
            .as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            sizes.insert(paths.value(row).to_string(), bytes.value(row));
        }
    }
    Ok(sizes)
}

/// What the queries of an index must answer: the rows of one version.
pub struct Answers<'a> {
    pub live: &'a BTreeSet<i64>,
    pub next_id: i64,
    /// The value of `gen` per row, in a scenario that has the column.
    pub stamps: Option<&'a BTreeMap<i64, i64>>,
    /// Values of `gen` to ask about, besides those the rows hold.
    pub stamps_ever: Option<&'a BTreeSet<i64>>,
    /// Whether to ask about a few of the values that the rows hold, not all.
    pub few_stamps: bool,
    pub scale: Scale,
    pub data_seed: u64,
    pub check_plan: bool,
    /// Varies the queries between checks.
    pub salt: usize,
}

/// Whether any segment of an index covers a fragment of this version. An index
/// that covers none leaves query planning, whatever its name.
pub fn covers_a_fragment(ds: &Dataset, indices: &[IndexMetadata], name: &str) -> bool {
    let live: roaring::RoaringBitmap = ds.fragments().iter().map(|f| f.id as u32).collect();
    indices
        .iter()
        .filter(|index| index.name == name)
        .filter_map(|index| index.fragment_bitmap.as_ref())
        .any(|bitmap| !(bitmap & &live).is_empty())
}

/// Queries an index and compares the answers with what the version holds. The
/// first query of each index can check its plan, since a query that falls back
/// to a scan would answer right whatever the index holds.
pub async fn check_index(
    ds: &Dataset,
    n: usize,
    kind: IndexKind,
    answers: &Answers<'_>,
) -> Result<(), Failure> {
    let unreadable = |error: lance::Error| fail("unreadable", n, error);
    match kind {
        IndexKind::BTreeStamp => {
            let stamps = answers
                .stamps
                .expect("an index on gen only exists in a scenario that has the column");
            let distinct: BTreeSet<i64> = stamps.values().copied().collect();
            let held: Vec<i64> = distinct.iter().copied().collect();
            let mut values: Vec<i64> = if answers.few_stamps && !held.is_empty() {
                vec![
                    held[0],
                    held[answers.salt % held.len()],
                    held[held.len() - 1],
                ]
            } else {
                held
            };
            // A value that no row holds now, which an index that went stale can
            // still answer, and one that no row ever held.
            values.extend(
                answers
                    .stamps_ever
                    .into_iter()
                    .flatten()
                    .filter(|value| !distinct.contains(value)),
            );
            values.push(-1);
            for (position, value) in values.into_iter().enumerate() {
                let filter = format!("gen = {value}");
                let mut scan = ds.scan();
                scan.filter(&filter).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if position == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("@idx_gen(BTree)") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("{filter} does not use idx_gen: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = stamps
                    .iter()
                    .filter(|(_, stamp)| **stamp == value)
                    .map(|(id, _)| *id)
                    .collect();
                compare_ids(n, &filter, &int64_column(&batch, "id"), &expected)?;
            }
        }
        IndexKind::BTreeId => {
            let last = answers.next_id.max(1);
            for slot in 0..3 {
                let lo = last * slot / 3;
                let hi = last * (slot + 1) / 3;
                let filter = format!("id >= {lo} AND id < {hi}");
                let mut scan = ds.scan();
                scan.filter(&filter).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if slot == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("@idx_id(BTree)") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("{filter} does not use idx_id: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = answers.live.range(lo..hi).copied().collect();
                compare_ids(n, &filter, &int64_column(&batch, "id"), &expected)?;
            }
        }
        IndexKind::BTree => {
            for value in 0..CATEGORIES {
                let filter = format!("category = {value}");
                let mut scan = ds.scan();
                scan.filter(&filter).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if value == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("@idx_category(BTree)") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("{filter} does not use idx_category: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = answers
                    .live
                    .iter()
                    .copied()
                    .filter(|id| category(*id) == value)
                    .collect();
                compare_ids(n, &filter, &int64_column(&batch, "id"), &expected)?;
            }
        }
        IndexKind::IvfFlat => {
            let mut queries: Vec<(String, [f32; VECTOR_DIM as usize])> = (0..3)
                .map(|slot| {
                    let id = (answers.salt * 3 + slot) as i64;
                    (
                        format!("random query {slot}"),
                        vector(!answers.data_seed, id),
                    )
                })
                .collect();
            // A deleted row's own vector: only an index that ignores
            // deletions would answer with that row.
            if let Some(deleted) = (0..answers.next_id).find(|id| !answers.live.contains(id)) {
                queries.push((
                    format!("deleted row {deleted}"),
                    vector(answers.data_seed, deleted),
                ));
            }
            for (position, (label, query)) in queries.iter().enumerate() {
                let mut scan = ds.scan();
                scan.nearest("vec", &Float32Array::from(query.to_vec()), NEIGHBORS)
                    .map_err(unreadable)?;
                // Every partition: IVF_FLAT is then exact.
                scan.minimum_nprobes(ALL_PARTITIONS);
                scan.project(&["id"]).map_err(unreadable)?;
                if position == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan.contains("ANNSubIndex: name=idx_vec") {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("a nearest neighbor query does not use idx_vec: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                check_neighbors(n, label, query, &int64_column(&batch, "id"), answers)?;
            }
        }
        IndexKind::Inverted => {
            for slot in 0..4 {
                let token = (answers.salt as u64 * 4 + slot) % answers.scale.vocabulary;
                let term = format!("tok{token:04}");
                let query = FullTextSearchQuery::new(term.clone())
                    .with_column("text".to_string())
                    .map_err(unreadable)?
                    .limit(None);
                let mut scan = ds.scan();
                scan.full_text_search(query).map_err(unreadable)?;
                scan.project(&["id"]).map_err(unreadable)?;
                if slot == 0 && answers.check_plan {
                    let plan = scan.explain_plan(true).await.map_err(unreadable)?;
                    if !plan
                        .lines()
                        .any(|line| line.trim_start().starts_with("MatchQuery: column=text"))
                    {
                        return Err(fail(
                            "index_not_used",
                            n,
                            format_args!("a search for {term} does not use idx_text: {plan}"),
                        ));
                    }
                }
                let batch = scan.try_into_batch().await.map_err(unreadable)?;
                let expected: BTreeSet<i64> = answers
                    .live
                    .iter()
                    .copied()
                    .filter(|id| tokens(*id, &answers.scale).any(|found| found == token))
                    .collect();
                compare_ids(
                    n,
                    &format!("a search for {term}"),
                    &int64_column(&batch, "id"),
                    &expected,
                )?;
            }
        }
    }
    Ok(())
}

/// The answer must be the brute-force nearest live rows, up to ties within
/// float rounding at the cut.
pub fn check_neighbors(
    n: usize,
    label: &str,
    query: &[f32; VECTOR_DIM as usize],
    found: &[i64],
    answers: &Answers<'_>,
) -> Result<(), Failure> {
    let mut ranked: Vec<(f32, i64)> = answers
        .live
        .iter()
        .map(|id| (squared_l2(query, &vector(answers.data_seed, *id)), *id))
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let k = NEIGHBORS.min(ranked.len());
    let cut = ranked[k - 1].0;
    let tolerance = cut * 1e-4 + 1e-6;
    let required: BTreeSet<i64> = ranked
        .iter()
        .take_while(|(distance, _)| *distance < cut - tolerance)
        .map(|(_, id)| *id)
        .collect();
    let allowed: BTreeSet<i64> = ranked
        .iter()
        .take_while(|(distance, _)| *distance <= cut + tolerance)
        .map(|(_, id)| *id)
        .collect();
    let distinct: BTreeSet<i64> = found.iter().copied().collect();
    if found.len() != k
        || distinct.len() != k
        || !required.is_subset(&distinct)
        || !distinct.is_subset(&allowed)
    {
        let expected: Vec<i64> = ranked.iter().take(k).map(|(_, id)| *id).collect();
        return Err(fail(
            "content",
            n,
            format_args!(
                "the nearest neighbors of {label} are {found:?} where brute force finds {expected:?}"
            ),
        ));
    }
    Ok(())
}

pub fn is_resources_exhausted(message: &str) -> bool {
    message.contains("Resources exhausted")
}

/// Whether an error says that a commit lost to another one.
pub fn is_commit_conflict(message: &str) -> bool {
    [
        "Commit conflict for version",
        "Retryable commit conflict for version",
        "Incompatible transaction",
    ]
    .iter()
    .any(|conflict| message.contains(conflict))
}
