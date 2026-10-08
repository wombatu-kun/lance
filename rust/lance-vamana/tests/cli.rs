// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The command line, exercised as a command line.
//!
//! Every case runs the built binary rather than a dispatcher, because the half
//! a unit test cannot see is the argument grammar, the exit code and stdout.
//! Cargo hands the binary's path over in `CARGO_BIN_EXE_vamana`.
#![cfg(feature = "cli")]

mod common;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance_arrow::FixedSizeListArrayExt;
use lance_file::version::LanceFileVersion;
use lance_vamana::query::{SearchParams, VamanaIndex, WalkMode, WalkStart};
use serde_json::Value;
use tempfile::TempDir;

use common::{DatasetFixture, VECTOR_COLUMN, VECTOR_DIM, random_vectors, random_vectors_of};

const ROWS: usize = 600;
const QUERIES: usize = 20;
const K: usize = 5;
/// 600 rows at 250 apiece is 2.4, so only `div_ceil` gives this number.
const PARTITIONS: usize = 3;
const ROWS_PER_PARTITION: &str = "250";
/// Five batches over `ROWS`, so the id column has to keep counting across them.
const BATCH_ROWS: &str = "128";
const APPENDED: usize = 100;
const DELETED: usize = 50;

/// A dataset ingested from a file of vectors, with the truth to score it by.
struct Fixture {
    _dir: TempDir,
    dataset: String,
    queries: PathBuf,
    truth: PathBuf,
    base: Vec<Vec<f32>>,
    query_vectors: Vec<Vec<f32>>,
}

impl Fixture {
    fn ingest() -> Self {
        let dir = TempDir::new().unwrap();
        let base = random_vectors(ROWS, 7);
        let query_vectors = random_vectors(QUERIES, 11);

        let base_path = dir.path().join("base.fvecs");
        let query_path = dir.path().join("query.fvecs");
        let truth_path = dir.path().join("truth.ivecs");
        write_fvecs(&base_path, &base);
        write_fvecs(&query_path, &query_vectors);
        let live = base
            .iter()
            .cloned()
            .zip(0..)
            .collect::<Vec<(Vec<f32>, u32)>>();
        write_ivecs(&truth_path, &nearest_positions(&live, &query_vectors, K));

        let dataset = dir.path().join("data.lance").to_str().unwrap().to_string();
        let out = run(&[
            "ingest",
            "--fvecs",
            base_path.to_str().unwrap(),
            "--dataset",
            &dataset,
            "--batch-rows",
            BATCH_ROWS,
        ]);
        assert!(
            out.contains(&format!("wrote {ROWS} rows of {VECTOR_DIM} dimensions")),
            "{out}"
        );

        Self {
            _dir: dir,
            dataset,
            queries: query_path,
            truth: truth_path,
            base,
            query_vectors,
        }
    }

    fn built() -> Self {
        let fixture = Self::ingest();
        fixture.build(&["--codes", "rabitq", "--code-bits", "3"]);
        fixture
    }

    fn build(&self, extra: &[&str]) -> String {
        let mut args = vec![
            "build",
            "--dataset",
            &self.dataset,
            "--index-name",
            "idx",
            "--rows-per-partition",
            ROWS_PER_PARTITION,
        ];
        args.extend_from_slice(extra);
        run(&args)
    }

    /// Every partition probed, so a shortfall is the graph's and not the
    /// router's.
    fn search(&self, extra: &[&str]) -> String {
        let probes = PARTITIONS.to_string();
        let k = K.to_string();
        let mut args = vec![
            "search",
            "--dataset",
            &self.dataset,
            "--index-name",
            "idx",
            "--fvecs",
            self.queries.to_str().unwrap(),
            "-k",
            &k,
            "--nprobes",
            &probes,
            "-L",
            "32",
        ];
        args.extend_from_slice(extra);
        run(&args)
    }

    fn info(&self) -> Value {
        serde_json::from_str(&run(&[
            "info",
            "--dataset",
            &self.dataset,
            "--index-name",
            "idx",
            "--json",
        ]))
        .unwrap()
    }
}

fn vamana(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vamana"))
        .args(args)
        .output()
        .expect("the binary Cargo built for this test")
}

/// Run and require success, returning stdout.
fn run(args: &[&str]) -> String {
    let output = vamana(args);
    assert!(
        output.status.success(),
        "vamana {}\nstatus {:?}\nstderr {}",
        args.join(" "),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// Run and require failure, returning stderr.
fn refused(args: &[&str]) -> String {
    let output = vamana(args);
    assert!(
        !output.status.success(),
        "vamana {} was expected to fail, but printed {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn write_fvecs(path: &Path, rows: &[Vec<f32>]) {
    let mut raw = Vec::with_capacity(rows.len() * (4 + rows[0].len() * 4));
    for row in rows {
        raw.extend_from_slice(&(row.len() as i32).to_le_bytes());
        row.iter()
            .for_each(|value| raw.extend_from_slice(&value.to_le_bytes()));
    }
    std::fs::write(path, raw).unwrap();
}

fn write_ivecs(path: &Path, rows: &[Vec<u32>]) {
    let mut raw = Vec::with_capacity(rows.len() * (4 + rows[0].len() * 4));
    for row in rows {
        raw.extend_from_slice(&(row.len() as i32).to_le_bytes());
        row.iter()
            .for_each(|value| raw.extend_from_slice(&value.to_le_bytes()));
    }
    std::fs::write(path, raw).unwrap();
}

/// The ids of each query's `k` nearest, by brute force over the rows given.
///
/// Takes `(vector, id)` rather than positions so that it can be recomputed over
/// what survives a delete and an append, which is not the base file any more.
fn nearest_positions(live: &[(Vec<f32>, u32)], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut scored = live
                .iter()
                .map(|(vector, id)| {
                    let distance = vector
                        .iter()
                        .zip(query)
                        .map(|(left, right)| (left - right).powi(2))
                        .sum::<f32>();
                    (distance, *id)
                })
                .collect::<Vec<_>>();
            scored.sort_by(|left, right| left.0.total_cmp(&right.0));
            scored.into_iter().take(k).map(|(_, id)| id).collect()
        })
        .collect()
}

/// Through the Lance API and not the command line: appending is the dataset's
/// business, and the index commands exist to catch up with it.
async fn append(uri: &str, vectors: &[Vec<f32>], ids: impl Iterator<Item = u64>) {
    let values = vectors.iter().flatten().copied().collect::<Vec<_>>();
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), VECTOR_DIM).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter_values(ids)),
            Arc::new(vectors) as Arc<dyn Array>,
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
}

/// A dataset whose id column is whatever the case under test needs it to be.
async fn dataset_with_ids(dir: &TempDir, name: &str, ids: Vec<Option<u64>>) -> String {
    let vectors = random_vectors(ids.len(), 3);
    let values = vectors.iter().flatten().copied().collect::<Vec<_>>();
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), VECTOR_DIM).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::UInt64, true),
        Field::new("vector", vectors.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from_iter(ids)),
            Arc::new(vectors) as Arc<dyn Array>,
        ],
    )
    .unwrap();
    let uri = dir.path().join(name).to_str().unwrap().to_string();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri.as_str(),
        None,
    )
    .await
    .unwrap();
    uri
}

async fn column_values(uri: &str, column: &str) -> Vec<u64> {
    let dataset = Dataset::open(uri).await.unwrap();
    let mut scanner = dataset.scan();
    scanner.project(&[column]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[column].as_primitive::<UInt64Type>().values().to_vec()
}

fn recall_of(report: &str) -> f64 {
    report
        .split("recall ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("no recall in {report}"))
}

#[tokio::test]
async fn the_whole_lifecycle_runs_from_a_file_of_vectors() {
    let fixture = Fixture::ingest();

    // The id column is what `--truth` scores by, and it is written a batch at a
    // time: a counter that restarted per batch would still ingest cleanly.
    assert_eq!(
        column_values(&fixture.dataset, "id").await,
        (0..ROWS as u64).collect::<Vec<_>>()
    );

    let built = fixture.build(&["--codes", "rabitq", "--code-bits", "3"]);
    assert!(
        built.contains(&format!(
            "indexed {ROWS} vectors into {PARTITIONS} partitions"
        )),
        "{built}"
    );

    let info = fixture.info();
    assert_eq!(info["segments"], 1);
    assert_eq!(info["first_segment"]["dimension"], VECTOR_DIM);
    assert_eq!(info["first_segment"]["codes"]["num_bits"], 3);

    // `exact` because the RaBitQ rotation is drawn afresh per build, which makes
    // any coded arm's recall a different number every run.
    let searched = fixture.search(&[
        "--mode",
        "exact",
        "--truth",
        fixture.truth.to_str().unwrap(),
    ]);
    assert!(recall_of(&searched) >= 0.99, "{searched}");

    append(
        &fixture.dataset,
        &random_vectors(APPENDED, 13),
        ROWS as u64..(ROWS + APPENDED) as u64,
    )
    .await;
    let inserted = run(&[
        "insert",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
    ]);
    assert!(
        inserted.contains(&format!("indexed {APPENDED} vectors")),
        "{inserted}"
    );
    assert_eq!(fixture.info()["segments"], 2, "a delta is its own segment");

    let merged = run(&[
        "merge",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
    ]);
    assert!(merged.contains("folded 2 segments"), "{merged}");
    // The delta already covered these rows, so a merge folds them rather than
    // inserting them: `vectors_inserted` is zero here and would be the wrong
    // number to pin.
    assert!(
        merged.contains(&format!("{APPENDED} vertices folded")),
        "{merged}"
    );
    assert_eq!(fixture.info()["segments"], 1, "a merge leaves one segment");

    let mut dataset = Dataset::open(&fixture.dataset).await.unwrap();
    dataset.delete(&format!("id < {DELETED}")).await.unwrap();
    let consolidated = run(&[
        "consolidate",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
    ]);
    assert!(consolidated.contains("1 rewritten"), "{consolidated}");
    assert!(
        consolidated.contains(&format!("{DELETED} vertices removed")),
        "the deleted rows are the ones that must leave the graphs: {consolidated}"
    );

    // Scored against what the dataset holds now, which is neither the base file
    // nor the ground truth shipped with it.
    let live = fixture
        .base
        .iter()
        .cloned()
        .zip(0u32..)
        .skip(DELETED)
        .chain(random_vectors(APPENDED, 13).into_iter().zip(ROWS as u32..))
        .collect::<Vec<_>>();
    let truth = fixture._dir.path().join("truth-after.ivecs");
    write_ivecs(&truth, &nearest_positions(&live, &fixture.query_vectors, K));
    let searched = fixture.search(&["--mode", "exact", "--truth", truth.to_str().unwrap()]);
    assert!(
        recall_of(&searched) >= 0.99,
        "maintenance must leave the index answering: {searched}"
    );
}

/// A lazy walk with a stop margin rides along: the one mode that stops by it.
/// A margin that never reached `SearchParams` would still answer, and quite
/// possibly with the same rows, so what pins it is the report - the setting,
/// the distances a query measured, which move with the margin, and the text
/// line.
#[tokio::test]
async fn a_search_answers_what_the_library_answers() {
    let fixture = Fixture::built();
    let dataset = Dataset::open(&fixture.dataset).await.unwrap();
    let index = VamanaIndex::open(&dataset, "idx").await.unwrap();
    let flat = SearchParams::new(K)
        .with_nprobes(PARTITIONS)
        .with_search_list_size(32)
        .with_mode(WalkMode::Flat)
        .with_rescore_budget(12);
    let ruled = flat
        .clone()
        .with_mode(WalkMode::Lazy)
        .with_stop_margin(0.25);
    let from_medoid = ruled.clone().with_start(WalkStart::Medoid);

    let mut costs = Vec::new();
    for (flags, params, margin) in [
        (
            vec!["--mode", "flat", "--rescore-budget", "12", "--json"],
            flat,
            Value::Null,
        ),
        (
            vec![
                "--mode",
                "lazy",
                "--rescore-budget",
                "12",
                "--stop-margin",
                "0.25",
                "--json",
            ],
            ruled,
            Value::from(0.25),
        ),
        (
            vec![
                "--mode",
                "lazy",
                "--start",
                "medoid",
                "--rescore-budget",
                "12",
                "--stop-margin",
                "0.25",
                "--json",
            ],
            from_medoid,
            Value::from(0.25),
        ),
    ] {
        let reported: Value = serde_json::from_str(&fixture.search(&flags)).unwrap();
        assert_eq!(reported["settings"]["stop_margin"], margin, "{flags:?}");
        let answers = reported["answers"].as_array().unwrap();
        assert_eq!(answers.len(), QUERIES, "every query must be answered");

        let mut comparisons = 0u64;
        for (query, answers) in fixture.query_vectors.iter().zip(answers) {
            let expected = index.search(query, &params).await.unwrap();
            comparisons += expected.comparisons;
            let expected = expected
                .neighbors
                .iter()
                .map(|neighbor| neighbor.row_addr)
                .collect::<Vec<_>>();
            let reported = answers
                .as_array()
                .unwrap()
                .iter()
                .map(|answer| answer["row_addr"].as_u64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(reported.len(), K, "a query must answer with k neighbours");
            assert_eq!(
                reported, expected,
                "the command line must answer what the call it wraps answers, given {flags:?}"
            );
        }
        assert_eq!(
            reported["per_query"]["comparisons"].as_f64().unwrap(),
            comparisons as f64 / QUERIES as f64,
            "the command line measured other distances than the call it wraps, given {flags:?}"
        );
        costs.push(comparisons);
    }
    assert_ne!(
        costs[1], costs[2],
        "a walk from the medoid costs what the default start costs here, so the two cannot be \
         told apart"
    );

    let table = fixture.search(&[
        "--mode",
        "lazy",
        "--rescore-budget",
        "12",
        "--stop-margin",
        "0.25",
    ]);
    assert!(
        table.contains(", stop margin 0.25"),
        "the table does not say the margin it walked by: {table}"
    );
}

/// Every argument must reach the index, not just the ones with a default.
#[tokio::test]
async fn a_build_records_the_parameters_it_was_given() {
    let fixture = Fixture::ingest();
    fixture.build(&[
        "--metric",
        "cosine",
        "-R",
        "32",
        "--search-list-size",
        "40",
        "--alpha",
        "1.0",
        "--codes",
        "rabitq",
        "--code-bits",
        "5",
    ]);

    let info = fixture.info();
    let segment = &info["first_segment"];
    assert_eq!(segment["distance_type"], "cosine");
    assert_eq!(segment["max_degree"], 32);
    assert_eq!(segment["search_list_size"], 40);
    assert_eq!(segment["alpha"], 1.0);
    assert_eq!(segment["codes"]["num_bits"], 5);
    assert!(segment["codes"]["rotation_signs"].is_array(), "{segment}");

    // Unnamed, the codes are the library's: eight-bit scalar ones, whose bounds
    // say which kind they are where a width of eight alone would not.
    fixture.build(&[]);
    let info = fixture.info();
    let codes = &info["first_segment"]["codes"];
    assert_eq!(codes["num_bits"], 8);
    assert!(codes["bounds"].is_object(), "{codes}");
    for scalar in [
        &["--codes", "scalar"][..],
        &["--codes", "scalar", "--code-bits", "8"],
    ] {
        fixture.build(scalar);
        let info = fixture.info();
        let codes = &info["first_segment"]["codes"];
        assert_eq!(codes["num_bits"], 8, "{scalar:?}");
        assert!(codes["bounds"].is_object(), "{scalar:?}: {codes}");
    }
    fixture.build(&["--codes", "none"]);
    assert!(fixture.info()["first_segment"]["codes"].is_null());

    let error = refused(&[
        "build",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--rows-per-partition",
        ROWS_PER_PARTITION,
        "--codes",
        "scalar",
        "--code-bits",
        "4",
    ]);
    assert!(
        error.contains("scalar codes are 8 bits a dimension; 4 was asked for"),
        "{error}"
    );
}

/// `--vectors dataset` builds an index that keeps none, which `info` says and a
/// lazy walk answers from. What it refuses, it refuses with the library's own
/// words.
#[tokio::test]
async fn an_index_may_leave_its_vectors_to_the_dataset() {
    let dir = TempDir::new().unwrap();
    let ingested = |dimension: i32| {
        let base = dir.path().join(format!("base_{dimension}.fvecs"));
        write_fvecs(&base, &random_vectors_of(ROWS, dimension, 7));
        let dataset = dir.path().join(format!("data_{dimension}.lance"));
        let dataset = dataset.to_str().unwrap().to_string();
        run(&[
            "ingest",
            "--fvecs",
            base.to_str().unwrap(),
            "--dataset",
            &dataset,
        ]);
        dataset
    };
    let build = |dataset: &str, extra: &[&str]| {
        let mut args = vec![
            "build",
            "--dataset",
            dataset,
            "--index-name",
            "idx",
            "--partitions",
            "2",
            "--vectors",
            "dataset",
        ];
        args.extend_from_slice(extra);
        vamana(&args)
    };

    let wide = ingested(64);
    let built = build(&wide, &["--codes", "rabitq", "--code-bits", "3"]);
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&built.stderr).contains("warning"),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let info: Value = serde_json::from_str(&run(&[
        "info",
        "--dataset",
        &wide,
        "--index-name",
        "idx",
        "--json",
    ]))
    .unwrap();
    assert_eq!(info["first_segment"]["vector_source"], "dataset");
    let plain = run(&["info", "--dataset", &wide, "--index-name", "idx"]);
    assert!(plain.contains("vectors        dataset"), "{plain}");
    let query = vec!["0.5"; 64].join(",");
    let answered: Value = serde_json::from_str(&run(&[
        "search",
        "--dataset",
        &wide,
        "--index-name",
        "idx",
        "--vector",
        &query,
        "--mode",
        "lazy",
        "-k",
        "5",
        "--json",
    ]))
    .unwrap();
    assert_eq!(answered["answers"][0].as_array().unwrap().len(), 5);
    let error = refused(&[
        "search",
        "--dataset",
        &wide,
        "--index-name",
        "idx",
        "--vector",
        &query,
        "--mode",
        "exact",
    ]);
    assert!(
        error.contains("leaves its vectors to the dataset"),
        "{error}"
    );

    let error = String::from_utf8(build(&wide, &["--codes", "none"]).stderr).unwrap();
    assert!(error.contains("to the dataset without codes"), "{error}");
    let narrow = ingested(VECTOR_DIM);
    let error =
        String::from_utf8(build(&narrow, &["--codes", "rabitq", "--code-bits", "3"]).stderr)
            .unwrap();
    assert!(
        error.contains("they have 16 dimensions and at least 64 are needed"),
        "{error}"
    );
}

/// A build that leaves its vectors to files no offset reaches - Lance 2.0 ones
/// here - goes ahead and says so on stderr, since this binary installs no
/// logger for the library's own warning to reach.
#[tokio::test]
async fn a_build_over_files_no_offset_reaches_warns() {
    let dir = TempDir::new().unwrap();
    let dataset = dir.path().join("data_2_0.lance");
    let dataset = dataset.to_str().unwrap();
    DatasetFixture {
        rows_per_fragment: 64,
        dimension: 64,
        storage_version: Some(LanceFileVersion::V2_0),
        ..Default::default()
    }
    .write(dataset)
    .await;
    let built = vamana(&[
        "build",
        "--dataset",
        dataset,
        "--index-name",
        "idx",
        "--column",
        VECTOR_COLUMN,
        "--partitions",
        "2",
        "--vectors",
        "dataset",
        "--codes",
        "rabitq",
        "--code-bits",
        "3",
    ]);
    let warned = String::from_utf8_lossy(&built.stderr);
    assert!(built.status.success(), "{warned}");
    assert!(
        warned.contains("warning: 3 fragments keep their vectors where no offset reaches"),
        "{warned}"
    );
}

/// The one mode no other case reaches.
///
/// `--prefetch-ahead` rides along because the report is the only place a caller
/// can see it: a look-ahead leaves no trace in any counter, so a flag that never
/// reached `SearchParams` would look exactly like one that did.
#[tokio::test]
async fn a_coded_walk_answers_too() {
    let fixture = Fixture::built();
    let reported = fixture.search(&["--mode", "coded", "--prefetch-ahead", "7", "--json"]);
    let reported: Value = serde_json::from_str(&reported).unwrap();
    assert_eq!(reported["settings"]["mode"], "coded");
    assert_eq!(reported["settings"]["prefetch_ahead"], 7);
    assert_eq!(reported["answers"].as_array().unwrap().len(), QUERIES);
    assert_eq!(reported["answers"][0].as_array().unwrap().len(), K);

    // Unnamed, the mode is the one the library walks by default.
    let reported: Value = serde_json::from_str(&fixture.search(&["--json"])).unwrap();
    assert_eq!(reported["settings"]["mode"], "lazy");
}

#[tokio::test]
async fn the_columns_of_a_neighbour_can_be_fetched_with_it() {
    let fixture = Fixture::built();
    // Not base vector zero: its rank, its row address and its id would all be
    // zero, and a take that ignored the row would still print `id=0`.
    let query = fixture.base[7]
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let out = run(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &query,
        "-k",
        "3",
        "--nprobes",
        &PARTITIONS.to_string(),
        "--mode",
        "exact",
        "--take",
        "id",
    ]);

    let ids = out
        .lines()
        .filter_map(|line| line.split("id=").nth(1))
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 3, "{out}");
    assert_eq!(ids[0], "7", "the query is its own nearest neighbour: {out}");
    assert_eq!(
        ids.iter().collect::<HashSet<_>>().len(),
        3,
        "three neighbours must carry three different rows: {out}"
    );
}

/// A component of a real embedding is as likely to be negative as not, and clap
/// reads a leading `-` as a flag unless the argument says otherwise.
#[tokio::test]
async fn a_query_may_be_negative() {
    let fixture = Fixture::built();
    let query = fixture.base[1]
        .iter()
        .map(|value| (-value).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let out = run(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &query,
        "-k",
        "1",
        "--nprobes",
        &PARTITIONS.to_string(),
    ]);
    assert!(out.contains("1 query at k = 1"), "{out}");
}

/// Zero queries would divide every per-query figure by zero.
#[tokio::test]
async fn a_run_of_no_queries_is_refused() {
    let fixture = Fixture::built();
    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--fvecs",
        fixture.queries.to_str().unwrap(),
        "--limit",
        "0",
    ]);
    assert!(error.contains("no queries to answer"), "{error}");
}

#[tokio::test]
async fn what_the_index_cannot_do_is_refused() {
    let fixture = Fixture::ingest();
    fixture.build(&["--codes", "none"]);
    let zeroes = vec!["0"; VECTOR_DIM as usize].join(",");

    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &zeroes,
        "--mode",
        "lazy",
    ]);
    assert!(error.contains("built without codes"), "{error}");
    // Asked to start at entry points as well, it is still the codes that are
    // missing first: entry points are chosen among by code.
    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &zeroes,
        "--mode",
        "lazy",
        "--start",
        "entry",
    ]);
    assert!(error.contains("built without codes"), "{error}");

    // `-k 1` so that the budget cannot be refused for being smaller than `k`
    // instead of for the mode it was given to.
    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &zeroes,
        "-k",
        "1",
        "--rescore-budget",
        "8",
        "--mode",
        "exact",
    ]);
    assert!(error.contains("which cannot spend it"), "{error}");

    // A margin is refused rather than ignored by a walk that does not stop by
    // it, and an exact one is such a walk.
    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &zeroes,
        "--stop-margin",
        "0.1",
        "--mode",
        "exact",
    ]);
    assert!(error.contains("does not stop by it"), "{error}");

    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        "1,2,3",
    ]);
    assert!(
        error.contains(&format!("{VECTOR_DIM}-dimensional")),
        "{error}"
    );

    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "absent",
        "--vector",
        &zeroes,
    ]);
    assert!(error.contains("no index named 'absent'"), "{error}");

    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--vector",
        &zeroes,
        "--mode",
        "exact",
        "--take",
        "nope",
    ]);
    assert!(error.contains("no column named nope"), "{error}");
}

/// Recall is scored by a column of base-file positions, and every way that
/// column can fail to be one is a wrong number rather than an error.
#[tokio::test]
async fn recall_needs_a_column_of_positions() {
    let dir = TempDir::new().unwrap();
    let queries = random_vectors(2, 11);
    let query_path = dir.path().join("query.fvecs");
    let truth_path = dir.path().join("truth.ivecs");
    write_fvecs(&query_path, &queries);
    write_ivecs(&truth_path, &vec![vec![0u32, 1, 2]; queries.len()]);

    let cases: [(&str, Vec<Option<u64>>, &str); 3] = [
        ("nulls", vec![None; 40], "is null for"),
        ("repeats", vec![Some(1); 40], "more than once"),
        (
            "too-wide",
            vec![Some(u64::from(u32::MAX) + 1); 40],
            "too large to be a base-file position",
        ),
    ];
    for (name, ids, expected) in cases {
        let uri = dataset_with_ids(&dir, name, ids).await;
        run(&[
            "build",
            "--dataset",
            &uri,
            "--index-name",
            "idx",
            "--partitions",
            "2",
        ]);
        let error = refused(&[
            "search",
            "--dataset",
            &uri,
            "--index-name",
            "idx",
            "--fvecs",
            query_path.to_str().unwrap(),
            "-k",
            "3",
            "--nprobes",
            "2",
            "--truth",
            truth_path.to_str().unwrap(),
        ]);
        assert!(error.contains(expected), "{name}: {error}");
    }
}

#[test]
fn the_parser_refuses_what_it_can_refuse_alone() {
    let error = refused(&[
        "build",
        "--dataset",
        "unused",
        "--index-name",
        "idx",
        "--partitions",
        "4",
        "--rows-per-partition",
        "150",
    ]);
    assert!(error.contains("cannot be used with"), "{error}");

    let error = refused(&["build", "--dataset", "unused", "--index-name", "idx"]);
    assert!(error.contains("required"), "{error}");

    let error = refused(&[
        "build",
        "--dataset",
        "unused",
        "--index-name",
        "idx",
        "--partitions",
        "4",
        "--metric",
        "dot",
    ]);
    assert!(error.contains("l2") && error.contains("cosine"), "{error}");

    // A width with no kind named is refused, so that `--code-bits 3`, which
    // once meant RaBitQ, cannot quietly build codes of another kind now that
    // the default is scalar.
    let build = |codes: &[&str]| {
        let mut args = vec![
            "build",
            "--dataset",
            "unused",
            "--index-name",
            "idx",
            "--partitions",
            "4",
        ];
        args.extend_from_slice(codes);
        refused(&args)
    };
    let error = build(&["--code-bits", "3"]);
    assert!(error.contains("--codes"), "{error}");
    let error = build(&["--codes", "none", "--code-bits", "3"]);
    assert!(error.contains("--codes none writes none"), "{error}");
    let error = build(&["--codes", "rabitq"]);
    assert!(
        error.contains("--codes rabitq needs --code-bits"),
        "{error}"
    );
    let error = build(&["--codes", "sq8"]);
    assert!(
        error.contains("none") && error.contains("scalar") && error.contains("rabitq"),
        "{error}"
    );

    let error = refused(&[
        "search",
        "--dataset",
        "unused",
        "--index-name",
        "idx",
        "--vector",
        "1,2",
        "--limit",
        "10",
    ]);
    assert!(error.contains("cannot be used with"), "{error}");

    // A ground truth's nth row is the nth query of a file, so pairing it with
    // an ad-hoc vector would score one query against another's answer.
    let error = refused(&[
        "search",
        "--dataset",
        "unused",
        "--index-name",
        "idx",
        "--vector",
        "1,2",
        "--truth",
        "truth.ivecs",
    ]);
    assert!(error.contains("cannot be used with"), "{error}");
}

/// A build stores entry points whenever there are codes and says what training
/// them cost, `info` names the parameters they were trained under,
/// `--entry-points` asks for a count in every partition or for none, and
/// `search --start` says where a walk starts. What cannot be had is refused:
/// a seed for entry points that are not stored, a count without codes, and a
/// walk asked by name to start at entry points on an index storing none.
#[tokio::test]
async fn entry_points_are_stored_named_and_started_from_as_asked() {
    let fixture = Fixture::ingest();
    let entry_points = |fixture: &Fixture| {
        run(&["info", "--dataset", &fixture.dataset, "--index-name", "idx"])
            .lines()
            .find(|line| line.trim_start().starts_with("entry points"))
            .map(|line| line.trim().to_string())
            .expect("info names the entry points")
    };

    let built = fixture.build(&[]);
    assert!(built.contains("entry points trained in"), "{built}");
    assert_eq!(
        entry_points(&fixture),
        "entry points   16 up to 23170 rows, else 64; 256 per entry; seed 42"
    );
    for start in ["medoid", "entry", "prefer-entry"] {
        let answered: Value =
            serde_json::from_str(&fixture.search(&["--start", start, "--json"])).unwrap();
        assert_eq!(answered["settings"]["start"], start, "{answered}");
        let printed = fixture.search(&["--start", start]);
        assert!(printed.contains(&format!(", start {start},")), "{printed}");
    }

    fixture.build(&[
        "--entry-points",
        "8",
        "--entry-seed",
        "3",
        "--entry-sample-rate",
        "64",
    ]);
    assert_eq!(
        entry_points(&fixture),
        "entry points   8; 64 per entry; seed 3"
    );

    fixture.build(&["--entry-points", "none"]);
    assert_eq!(entry_points(&fixture), "entry points   none");
    let probes = PARTITIONS.to_string();
    let error = refused(&[
        "search",
        "--dataset",
        &fixture.dataset,
        "--index-name",
        "idx",
        "--fvecs",
        fixture.queries.to_str().unwrap(),
        "--nprobes",
        &probes,
        "--start",
        "entry",
    ]);
    assert!(error.contains("stores no entry points"), "{error}");

    let build = |extra: &[&str]| {
        let mut args = vec![
            "build",
            "--dataset",
            &fixture.dataset,
            "--index-name",
            "idx",
            "--rows-per-partition",
            ROWS_PER_PARTITION,
        ];
        args.extend_from_slice(extra);
        refused(&args)
    };
    let error = build(&["--entry-points", "none", "--entry-seed", "3"]);
    assert!(error.contains("--entry-points none stores none"), "{error}");
    let error = build(&["--codes", "none", "--entry-points", "8"]);
    assert!(error.contains("without codes"), "{error}");
    let error = build(&["--codes", "none", "--entry-seed", "3"]);
    assert!(
        error.contains("an index built with --codes none stores none"),
        "{error}"
    );
}
