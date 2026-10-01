// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whether the Vamana indexes of two datasets hold the same segment, value for
//! value.
//!
//! The gate of a stand rebuild: an index rebuilt at a new format version must
//! hold what the old one held. The files are read through Lance's file reader
//! and the segment metadata is compared as JSON, so neither side passes this
//! build's format-version check, which the old side would fail. Values rather
//! than bytes, because the Lance versions that wrote the two sides may encode
//! the same values differently.
//!
//! Compared: the segment metadata without `format_version` and `vector_source`,
//! the IVF model, the partition table, and every partition's `__row_id`,
//! `__neighbors`, `__code` and `__vector`, bit for bit. With
//! `--right-without-vectors` the right index must leave its vectors to the
//! dataset: its partitions hold no `__vector` and everything else is equal.
//!
//! Usage: `compare_segments <left dataset> <right dataset> [--right-without-vectors]`,
//! `INDEX_NAME` (default `vamana_idx`). Prints one line per item and exits 1
//! when anything differs.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float32Type, UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType;
use lance::Dataset;
use lance_file::reader::FileReader;
use lance_index::pb;
use lance_io::scheduler::ScanScheduler;
use lance_vamana::codes::CODE_COLUMN;
use lance_vamana::format::{
    FILE_COLUMN, INDEX_FILE_NAME, INDEX_METADATA_KEY, IVF_POSITION_KEY, MEDOID_COLUMN,
    NEIGHBORS_COLUMN, NUM_ROWS_COLUMN, PARTITION_ID_COLUMN, ROW_ID_COLUMN, VECTOR_COLUMN,
};
use lance_vamana::io::{open_file, read_rows, scan_scheduler};
use lance_vamana::query::committed_segments;
use object_store::path::Path;
use prost::Message;
use serde_json::Value;

/// Rows read at a time: 63 MB of a GIST partition's vectors on each side, and
/// few enough under test that a fixture spans several reads.
const ROWS_PER_READ: usize = if cfg!(test) { 64 } else { 16_384 };

struct Segment {
    scheduler: Arc<ScanScheduler>,
    dir: Path,
    sizes: HashMap<String, u64>,
}

impl Segment {
    async fn open(uri: &str, index_name: &str) -> Self {
        let dataset = Dataset::open(uri).await.unwrap();
        let committed = committed_segments(&dataset, index_name).await.unwrap();
        let [segment] = committed.as_slice() else {
            panic!(
                "{uri}: expected one committed segment of {index_name}, found {}",
                committed.len()
            );
        };
        Self {
            scheduler: scan_scheduler(&dataset.object_store(None).await.unwrap()),
            dir: dataset.indices_dir().join(segment.uuid.to_string()),
            sizes: segment
                .files
                .iter()
                .flatten()
                .map(|file| (file.path.clone(), file.size_bytes))
                .collect(),
        }
    }

    async fn read(&self, file: &str, column: Option<&str>) -> FileReader {
        let columns = column.map(|column| [column]);
        open_file(
            &self.scheduler,
            &self.dir.clone().join(file),
            columns.as_ref().map(|columns| columns.as_slice()),
            self.sizes.get(file).copied(),
        )
        .await
        .unwrap()
    }
}

#[derive(Default)]
struct Report {
    lines: Vec<String>,
    differing: Vec<String>,
}

impl Report {
    fn record(&mut self, item: &str, difference: Option<String>) {
        match difference {
            None => self.lines.push(format!("{item}: equal")),
            Some(difference) => {
                self.lines.push(format!("{item}: DIFFERS, {difference}"));
                self.differing.push(item.to_string());
            }
        }
    }
}

/// The values of a column without nulls as one flat slice, and how many make a row.
fn flat<T: ArrowPrimitiveType>(column: &dyn Array) -> (&[T::Native], usize) {
    assert_eq!(column.null_count(), 0, "a segment column holds a null");
    match column.data_type() {
        DataType::FixedSizeList(_, width) => {
            let list = column.as_fixed_size_list();
            let width = *width as usize;
            let values = list.values();
            assert_eq!(values.null_count(), 0, "a segment list holds a null item");
            let values = values.as_primitive::<T>().values();
            (
                &values[list.offset() * width..][..list.len() * width],
                width,
            )
        }
        _ => (column.as_primitive::<T>().values(), 1),
    }
}

/// Rows at which two columns differ: how many and the first. `bits` maps a value
/// to what is compared, so a float is compared as stored rather than by `==`.
fn differing_rows<T: Copy, B: PartialEq>(
    left: &[T],
    right: &[T],
    width: usize,
    bits: impl Fn(T) -> B,
) -> (usize, Option<usize>) {
    let mut count = 0;
    let mut first = None;
    for (row, (left, right)) in left
        .chunks_exact(width)
        .zip(right.chunks_exact(width))
        .enumerate()
    {
        if !left.iter().zip(right).all(|(l, r)| bits(*l) == bits(*r)) {
            count += 1;
            first.get_or_insert(row);
        }
    }
    (count, first)
}

fn column_difference(name: &str, left: &ArrayRef, right: &ArrayRef) -> (usize, Option<usize>) {
    fn compare<T: ArrowPrimitiveType, B: PartialEq>(
        left: &ArrayRef,
        right: &ArrayRef,
        bits: impl Fn(T::Native) -> B,
    ) -> Option<(usize, Option<usize>)> {
        let (left, left_width) = flat::<T>(left.as_ref());
        let (right, right_width) = flat::<T>(right.as_ref());
        (left_width == right_width && left.len() == right.len())
            .then(|| differing_rows(left, right, left_width, bits))
    }
    let difference = match name {
        ROW_ID_COLUMN => compare::<UInt64Type, _>(left, right, |v| v),
        NEIGHBORS_COLUMN => compare::<UInt32Type, _>(left, right, |v| v),
        CODE_COLUMN => compare::<UInt8Type, _>(left, right, |v| v),
        VECTOR_COLUMN => compare::<Float32Type, _>(left, right, f32::to_bits),
        _ => unreachable!("{name} is not a partition column"),
    };
    difference.unwrap_or_else(|| panic!("{name}: the two sides have different widths"))
}

async fn compare_partition(
    left: &Segment,
    right: &Segment,
    file: &str,
    num_rows: usize,
    right_without_vectors: bool,
    report: &mut Report,
) {
    let has = |reader: &FileReader, column: &str| reader.schema().field(column).is_some();
    let left_whole = left.read(file, None).await;
    let right_whole = right.read(file, None).await;
    for column in [ROW_ID_COLUMN, NEIGHBORS_COLUMN, CODE_COLUMN, VECTOR_COLUMN] {
        let item = format!("{file} {column}");
        let expect_right = !(right_without_vectors && column == VECTOR_COLUMN);
        match (has(&left_whole, column), has(&right_whole, column)) {
            (false, false) => {
                report.lines.push(format!("{item}: absent on both sides"));
                continue;
            }
            (true, false) if !expect_right => {
                report
                    .lines
                    .push(format!("{item}: absent on the right, as expected"));
                continue;
            }
            (true, true) if expect_right => {}
            (on_left, on_right) => {
                report.record(
                    &item,
                    Some(format!(
                        "present on the left {on_left}, on the right {on_right}"
                    )),
                );
                continue;
            }
        }
        let left_reader = left.read(file, Some(column)).await;
        let right_reader = right.read(file, Some(column)).await;
        let (mut differing, mut first) = (0, None);
        for start in (0..num_rows).step_by(ROWS_PER_READ) {
            let rows = start..(start + ROWS_PER_READ).min(num_rows);
            let left_batch = read_rows(&left_reader, rows.clone()).await.unwrap();
            let right_batch = read_rows(&right_reader, rows).await.unwrap();
            let (count, at) = column_difference(column, &left_batch[column], &right_batch[column]);
            differing += count;
            if first.is_none() {
                first = at.map(|row| start + row);
            }
        }
        report.record(
            &item,
            first.map(|first| format!("{differing} of {num_rows} rows, the first at {first}")),
        );
    }
}

/// The segment metadata without the two fields a format bump may change, and
/// those two: the format version and where the vectors are.
fn segment_metadata(reader: &FileReader) -> (Value, Value, Value) {
    let json = &reader.schema().metadata[INDEX_METADATA_KEY];
    let mut value = serde_json::from_str::<Value>(json).unwrap();
    let fields = value.as_object_mut().unwrap();
    let version = fields.remove("format_version").unwrap();
    // Format 5 predates the field and kept every vector in the index.
    let source = fields
        .remove("vector_source")
        .unwrap_or_else(|| Value::from("index"));
    (version, source, value)
}

/// The columns of the partition table on which two segments disagree.
fn table_difference(left: &RecordBatch, right: &RecordBatch) -> Vec<&'static str> {
    [
        PARTITION_ID_COLUMN,
        MEDOID_COLUMN,
        NUM_ROWS_COLUMN,
        FILE_COLUMN,
    ]
    .into_iter()
    .filter(|column| left[*column].to_data() != right[*column].to_data())
    .collect()
}

async fn ivf_model(reader: &FileReader) -> pb::Ivf {
    let position = reader.schema().metadata[IVF_POSITION_KEY]
        .parse::<u32>()
        .unwrap();
    pb::Ivf::decode(reader.read_global_buffer(position).await.unwrap()).unwrap()
}

/// Which fields of two IVF models differ, and by how much; `None` when none do.
fn ivf_difference(left: &pb::Ivf, right: &pb::Ivf) -> Option<String> {
    if left == right {
        return None;
    }
    let shape = |ivf: &pb::Ivf| {
        ivf.centroids_tensor
            .as_ref()
            .map(|tensor| (tensor.data_type, tensor.shape.clone()))
    };
    let values = |ivf: &pb::Ivf| -> Vec<f32> {
        match &ivf.centroids_tensor {
            Some(tensor) => tensor
                .data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
            None => ivf.centroids.clone(),
        }
    };
    let (left_values, right_values) = (values(left), values(right));
    let mut fields = Vec::new();
    if shape(left) != shape(right) || left_values.len() != right_values.len() {
        fields.push(format!(
            "centroid shapes {:?} and {:?}",
            shape(left),
            shape(right)
        ));
    } else {
        let (differing, largest) = left_values
            .iter()
            .zip(&right_values)
            .filter(|(l, r)| l.to_bits() != r.to_bits())
            .fold((0, 0f32), |(count, largest), (l, r)| {
                (count + 1, largest.max((l - r).abs()))
            });
        if differing > 0 {
            fields.push(format!(
                "{differing} of {} centroid values, by at most {largest:e}",
                left_values.len()
            ));
        }
    }
    if left.offsets != right.offsets {
        fields.push("partition offsets".to_string());
    }
    if left.lengths != right.lengths {
        fields.push("partition lengths".to_string());
    }
    if left.loss != right.loss {
        let show = |loss: Option<f64>| loss.map_or_else(|| "none".to_string(), |v| v.to_string());
        fields.push(format!(
            "k-means loss {} against {}",
            show(left.loss),
            show(right.loss)
        ));
    }
    if fields.is_empty() {
        fields.push("the decoded protobufs differ".to_string());
    }
    Some(fields.join("; "))
}

async fn compare(
    left_uri: &str,
    right_uri: &str,
    index_name: &str,
    right_without_vectors: bool,
) -> Report {
    let mut report = Report::default();
    let left = Segment::open(left_uri, index_name).await;
    let right = Segment::open(right_uri, index_name).await;
    let left_index = left.read(INDEX_FILE_NAME, None).await;
    let right_index = right.read(INDEX_FILE_NAME, None).await;

    let (left_version, left_source, left_metadata) = segment_metadata(&left_index);
    let (right_version, right_source, right_metadata) = segment_metadata(&right_index);
    report.lines.push(format!(
        "format versions: left {left_version}, right {right_version}; vectors: left in \
         {left_source}, right in {right_source}"
    ));
    let expected_source = if right_without_vectors {
        "dataset"
    } else {
        "index"
    };
    report.record(
        "vector sources",
        (left_source != "index" || right_source != expected_source)
            .then(|| format!("expected the left in index and the right in {expected_source}")),
    );
    report.record(
        "metadata",
        (left_metadata != right_metadata)
            .then(|| format!("left {left_metadata}, right {right_metadata}")),
    );

    let (left_ivf, right_ivf) = (ivf_model(&left_index).await, ivf_model(&right_index).await);
    report.record("ivf model", ivf_difference(&left_ivf, &right_ivf));

    let rows = |reader: &FileReader| reader.metadata().num_rows as usize;
    let (left_rows, right_rows) = (rows(&left_index), rows(&right_index));
    if left_rows != right_rows || left_rows == 0 {
        report.record(
            "partition table",
            Some(format!(
                "{left_rows} partitions on the left, {right_rows} on the right"
            )),
        );
        return report;
    }
    let left_table = read_rows(&left_index, 0..left_rows).await.unwrap();
    let right_table = read_rows(&right_index, 0..right_rows).await.unwrap();
    let table_differs = table_difference(&left_table, &right_table);
    report.record(
        &format!("partition table ({left_rows} partitions)"),
        (!table_differs.is_empty()).then(|| format!("columns {table_differs:?}")),
    );
    if !table_differs.is_empty() {
        return report;
    }

    let files = left_table[FILE_COLUMN].as_string::<i32>();
    let counts = left_table[NUM_ROWS_COLUMN].as_primitive::<UInt32Type>();
    for (file, num_rows) in files.iter().zip(counts.values().iter()) {
        let file = file.unwrap();
        compare_partition(
            &left,
            &right,
            file,
            *num_rows as usize,
            right_without_vectors,
            &mut report,
        )
        .await;
    }
    report
}

#[tokio::main]
async fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let right_without_vectors = args.iter().any(|arg| arg == "--right-without-vectors");
    let uris = args
        .iter()
        .filter(|arg| !arg.starts_with("--"))
        .collect::<Vec<_>>();
    let [left, right] = uris.as_slice() else {
        panic!("usage: compare_segments <left dataset> <right dataset> [--right-without-vectors]");
    };
    let index_name = std::env::var("INDEX_NAME").unwrap_or_else(|_| "vamana_idx".to_string());
    println!("left  {left}\nright {right}\nindex {index_name}");
    let report = compare(left, right, &index_name, right_without_vectors).await;
    for line in &report.lines {
        println!("{line}");
    }
    if report.differing.is_empty() {
        println!("EQUAL");
    } else {
        println!("DIFFERENT: {}", report.differing.join(", "));
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{
        FixedSizeListArray, Float32Array, RecordBatchIterator, StringArray, UInt32Array,
        UInt64Array,
    };
    use arrow_schema::{Field, Schema};
    use lance::dataset::WriteParams;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_vamana::build::BuildParams;
    use lance_vamana::builder::{IndexParams, create_index};
    use lance_vamana::codes::CodeSpec;
    use lance_vamana::format::{VectorSource, index_schema};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    const INDEX: &str = "vamana_idx";
    const ROWS: usize = 300;
    const DIMENSION: i32 = 64;
    const GRAPH: BuildParams = BuildParams {
        max_degree: 12,
        search_list_size: 20,
        alpha: 1.2,
        seed: 42,
    };

    /// A dataset and the index built over it; every field but the one a test
    /// changes stays at its default.
    struct Build {
        values: Vec<f32>,
        rows_per_file: usize,
        graph: BuildParams,
        source: VectorSource,
    }

    impl Default for Build {
        fn default() -> Self {
            let mut rng = SmallRng::seed_from_u64(7);
            Self {
                values: (0..ROWS * DIMENSION as usize)
                    .map(|_| rng.random::<f32>())
                    .collect(),
                rows_per_file: ROWS / 3,
                graph: GRAPH,
                source: VectorSource::Index,
            }
        }
    }

    async fn indexed(dir: &tempfile::TempDir, name: &str, build: Build) -> String {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIMENSION,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from_iter_values(0..ROWS as u64)),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(build.values),
                        DIMENSION,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let uri = dir.path().join(name).to_str().unwrap().to_string();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            Some(WriteParams {
                max_rows_per_file: build.rows_per_file,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        create_index(
            &mut dataset,
            INDEX,
            &IndexParams::new("vector", 1)
                .with_codes(CodeSpec::Scalar { num_bits: 8 })
                .with_vector_source(build.source)
                .with_graph_params(build.graph),
        )
        .await
        .unwrap();
        uri
    }

    #[test]
    fn a_float_is_compared_as_stored() {
        let list = |values: Vec<f32>| -> ArrayRef {
            Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(values), 2).unwrap(),
            )
        };
        let left = list(vec![1.0, f32::NAN, 2.0, 0.0, 3.0, 4.0]);
        let right = list(vec![1.0, f32::NAN, 2.0, -0.0, 3.0, 5.0]);
        assert_eq!(
            column_difference(VECTOR_COLUMN, &left, &right),
            (2, Some(1))
        );
        assert_eq!(column_difference(VECTOR_COLUMN, &left, &left), (0, None));
    }

    #[test]
    fn an_ivf_difference_names_the_fields_that_differ() {
        let model = |values: &[f32], loss: f64| pb::Ivf {
            centroids_tensor: Some(pb::Tensor {
                data_type: pb::tensor::DataType::Float32 as i32,
                shape: vec![1, values.len() as u32],
                data: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            }),
            loss: Some(loss),
            ..Default::default()
        };
        let base = model(&[0.5, 1.0], 3.0);
        assert_eq!(ivf_difference(&base, &base), None);
        assert_eq!(
            ivf_difference(&base, &model(&[0.5, 1.0], 3.5)).as_deref(),
            Some("k-means loss 3 against 3.5")
        );
        assert_eq!(
            ivf_difference(&base, &model(&[0.5, 1.25], 3.0)).as_deref(),
            Some("1 of 2 centroid values, by at most 2.5e-1")
        );
        assert_eq!(
            ivf_difference(&base, &model(&[0.5, 1.0, 2.0], 3.0)).as_deref(),
            Some("centroid shapes Some((2, [1, 2])) and Some((2, [1, 3]))")
        );
        let offsets = pb::Ivf {
            offsets: vec![0],
            ..base.clone()
        };
        assert_eq!(
            ivf_difference(&base, &offsets).as_deref(),
            Some("partition offsets")
        );
        let lengths = pb::Ivf {
            lengths: vec![2],
            ..base.clone()
        };
        assert_eq!(
            ivf_difference(&base, &lengths).as_deref(),
            Some("partition lengths")
        );
    }

    #[test]
    fn every_column_of_the_partition_table_is_compared() {
        let table = |partition: u32, medoid: u32, rows: u32, file: &str| {
            RecordBatch::try_new(
                Arc::new(index_schema()),
                vec![
                    Arc::new(UInt32Array::from(vec![partition])),
                    Arc::new(UInt32Array::from(vec![medoid])),
                    Arc::new(UInt32Array::from(vec![rows])),
                    Arc::new(StringArray::from(vec![file])),
                ],
            )
            .unwrap()
        };
        let base = table(0, 5, 100, "part_00000.idx");
        assert!(table_difference(&base, &base).is_empty());
        for (changed, column) in [
            (table(1, 5, 100, "part_00000.idx"), PARTITION_ID_COLUMN),
            (table(0, 6, 100, "part_00000.idx"), MEDOID_COLUMN),
            (table(0, 5, 101, "part_00000.idx"), NUM_ROWS_COLUMN),
            (table(0, 5, 100, "part_00001.idx"), FILE_COLUMN),
        ] {
            assert_eq!(table_difference(&base, &changed), [column]);
        }
    }

    #[tokio::test]
    async fn a_rebuild_is_equal_and_a_changed_build_differs_where_it_changed() {
        let dir = tempfile::tempdir().unwrap();
        let first = indexed(&dir, "first", Build::default()).await;
        let again = indexed(&dir, "again", Build::default()).await;
        let graph = BuildParams { seed: 43, ..GRAPH };
        let reseeded = indexed(
            &dir,
            "reseeded",
            Build {
                graph,
                ..Default::default()
            },
        )
        .await;
        let graph = BuildParams {
            search_list_size: 24,
            ..GRAPH
        };
        let wider = indexed(
            &dir,
            "wider",
            Build {
                graph,
                ..Default::default()
            },
        )
        .await;

        let same = compare(&first, &again, INDEX, false).await;
        assert!(same.differing.is_empty(), "{:?}", same.lines);
        // The router trains on a sample drawn with the graph's seed, so its
        // centroid moves with the seed too; the codes are scalar, bounded by
        // every vector, and do not.
        let seeds = compare(&first, &reseeded, INDEX, false).await;
        assert_eq!(
            seeds.differing,
            ["ivf model", "part_00000.idx __neighbors"],
            "{:?}",
            seeds.lines
        );
        let beams = compare(&first, &wider, INDEX, false).await;
        assert_eq!(
            beams.differing,
            ["metadata", "part_00000.idx __neighbors"],
            "{:?}",
            beams.lines
        );
    }

    #[tokio::test]
    async fn a_changed_dataset_differs_where_it_changed() {
        let dir = tempfile::tempdir().unwrap();
        let first = indexed(&dir, "first", Build::default()).await;
        let rows_per_file = ROWS / 2;
        let relaid = indexed(
            &dir,
            "relaid",
            Build {
                rows_per_file,
                ..Default::default()
            },
        )
        .await;
        let mut values = Build::default().values;
        let nudged_row = 170;
        values[nudged_row * DIMENSION as usize] = 1.0 - values[nudged_row * DIMENSION as usize];
        let nudged = indexed(
            &dir,
            "nudged",
            Build {
                values,
                ..Default::default()
            },
        )
        .await;

        // Two fragments instead of three: the same vectors at other addresses,
        // and a fragment list of its own.
        let layouts = compare(&first, &relaid, INDEX, false).await;
        assert_eq!(
            layouts.differing,
            ["metadata", "part_00000.idx __row_id"],
            "{:?}",
            layouts.lines
        );
        let nudge = compare(&first, &nudged, INDEX, false).await;
        for column in [ROW_ID_COLUMN, "metadata"] {
            assert!(
                !nudge.differing.iter().any(|item| item.ends_with(column)),
                "{:?}",
                nudge.lines
            );
        }
        for column in [CODE_COLUMN, VECTOR_COLUMN] {
            let line = format!(
                "part_00000.idx {column}: DIFFERS, 1 of {ROWS} rows, the first at {nudged_row}"
            );
            assert!(nudge.lines.contains(&line), "{:?}", nudge.lines);
        }
    }

    #[tokio::test]
    async fn a_vector_less_twin_differs_only_by_its_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let full = indexed(&dir, "full", Build::default()).await;
        let source = VectorSource::Dataset;
        let bare = indexed(
            &dir,
            "bare",
            Build {
                source,
                ..Default::default()
            },
        )
        .await;

        let expected = compare(&full, &bare, INDEX, true).await;
        assert!(expected.differing.is_empty(), "{:?}", expected.lines);
        let unexpected = compare(&full, &bare, INDEX, false).await;
        assert_eq!(
            unexpected.differing,
            ["vector sources", "part_00000.idx __vector"],
            "{:?}",
            unexpected.lines
        );
        let reversed = compare(&bare, &full, INDEX, true).await;
        assert!(
            reversed.differing.contains(&"vector sources".to_string()),
            "{:?}",
            reversed.lines
        );
    }
}
