import lance
import pyarrow as pa
import numpy as np
import tempfile
import os

# Insert two batches of data where the second batch has only one row, create an index,
# execute update_columns to update the data in the second batch,
# then optimize the index, and finally query — the updated data {10.8,...,10.8} appears twice.

np.random.seed(42)

n_unique = 768
n_dims = 16
unique_vectors = [np.random.uniform(-1, 1, n_dims).astype(np.float32) for _ in range(n_unique)]
vectors = unique_vectors + unique_vectors
ids = list(range(n_unique * 2))

table1 = pa.table(
    {
        "id": pa.array(ids, type=pa.int32()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate(vectors), type=pa.float32()), list_size=n_dims
        ),
    }
)
table2 = pa.table(
    {
        "id": pa.array([10000], type=pa.int32()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(np.full(n_dims, 0.5, dtype=np.float32()), type=pa.float32()),
            list_size=n_dims,
        ),
    }
)

URI = os.path.join(tempfile.mkdtemp(), "lance_vector_upd")
dataset = lance.write_dataset(table1, URI, mode="create")
dataset = lance.write_dataset(table2, URI, mode="append")

dataset.create_index(
    "vector",
    index_type="IVF_PQ",
    metric="l2",
    num_partitions=1,
    num_sub_vectors=n_dims,
)

fragment = dataset.get_fragment(1)
full_table = fragment.to_table(columns=["vector", "id"], with_row_id=True)
matched_rowids = full_table["_rowid"].to_pylist()
new_vector = [10.8] * n_dims
vector_type = pa.list_(pa.float32(), n_dims)
update_data = pa.table(
    {
        "_rowid": pa.array(matched_rowids, type=pa.uint64()),
        "vector": pa.array([new_vector] * len(matched_rowids), type=vector_type),
    }
)

updated_frag, fields_modified = fragment.update_columns(
    update_data,
    left_on="_rowid",
    right_on="_rowid",
)

op = lance.LanceOperation.Update(
    updated_fragments=[updated_frag],
    fields_modified=fields_modified,
)
dataset = lance.LanceDataset.commit(
    dataset.uri,
    op,
    read_version=dataset.version,
)

dataset.optimize.optimize_indices(num_indices_to_merge=0)

indices = dataset.describe_indices()
all_segs = []
for idx in indices:
    for seg in idx.segments:
        all_segs.append(seg.uuid)

q108 = np.full(n_dims, 10.8, dtype=np.float32)
result = dataset.scanner(
    columns=["id", "vector", "_rowid", "_distance"],
    nearest={"column": "vector", "q": q108, "k": 10},
    index_segments=all_segs,
).to_pandas()

dupes = result[result["id"] == 10000]
assert len(dupes) == 2, f"Expected 2 rows with id=10000, got {len(dupes)}"
print(f"Bug reproduced: id=10000 appears {len(dupes)} times")
print(dupes[["id", "_rowid", "_distance"]])
