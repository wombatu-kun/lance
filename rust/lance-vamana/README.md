# lance-vamana

A disk-resident Vamana (DiskANN) vector index for Lance datasets, built and
queried entirely through Lance's published API. The crate lives in the Lance
tree for convenience but is not a member of its workspace, so the boundary it
compiles against is the one an out-of-tree crate sees.

```rust
lance_vamana::create_index(&mut dataset, "vamana_idx", &IndexParams::new("vec", 64)).await?;

let index = VamanaIndex::open(&dataset, "vamana_idx").await?;
let answer = index.search(&query, &SearchParams::new(10).with_nprobes(8)).await?;
for neighbor in &answer.neighbors {
    // `neighbor.row_id` is a Lance row address; fetch with `Dataset::take_rows`.
}
```

## What this costs the dataset

**Committing a Vamana index breaks Lance's own vector search on the indexed
column, and Lance's index maintenance on the whole dataset.** This is not a
rough edge to be tidied later; it follows from there being no way to register an
external vector index type with Lance, and it is the reason this crate ships its
own query driver.

Measured on a freshly indexed dataset (`a_committed_index_shadows_lances_own_vector_paths`):

| Call | Before `create_index` | After |
|---|---|---|
| `scan().nearest(col, q, k)` | works | **errors**: `Index Metadata not found` |
| `scan().nearest(...).use_index(false)` | works | works |
| `optimize_indices()` | works | **errors**, for *every* index on the dataset |
| `index_statistics(name)` | works | **errors** |
| plain `scan()` | works | works |

The mechanism: Lance's scanner picks a vector index by field id alone, with no
type check, so it selects the Vamana segment and then fails to read it as one of
its own. `optimize_indices` classifies an index as a vector index by the presence
of `index.idx` - the file name this format is obliged to use - and propagates the
failure out of the loop over every index.

Consequences to plan around:

- Do not put a Vamana index on a column that is also served by a Lance
  `IVF_HNSW_*` index; whichever appears first in the manifest wins the lookup,
  and a Vamana segment can shadow a working one.
- Anything that calls `optimize_indices` on the dataset - including routine
  maintenance of unrelated scalar indices - will fail while a Vamana index
  exists. Drop the index, maintain, rebuild.
- `use_index(false)` is the escape hatch for Lance-side vector queries.

## What the query path does not do

- **The delete list is a snapshot taken at open.** Deleted rows are excluded
  from answers, but the list is read once, when the index is opened. A row
  deleted afterwards keeps coming back until the index is reopened, and nothing
  about the answer reveals it.
- **Fewer than `k` rows come back when a probed partition is mostly deleted.**
  Deleted vertices are still walked - they carry the edges that hold the graph
  together - but they are dropped from the answer, and a walk only produces
  `search_list_size` candidates to draw from.
- **Rows added after the build are invisible.** The index answers from the
  fragments it was built over. Lance's scanner would scan the unindexed
  remainder; this driver does not.
- **No predicate prefilter and no refine step.** Both live in Lance's scanner,
  which this driver bypasses.
- An index whose fragments have been compacted away is **refused** at open
  rather than answering from what is left; rebuild it. So is one whose coverage
  the dataset has edited under it - an in-place column update prunes the
  rewritten fragments out of the index's `fragment_bitmap` while leaving every
  row address valid, which no liveness check can see.

## Building

- The whole vector column is held in memory for the duration of a build - twice
  over, briefly, while the batches are concatenated. A build is a builder-side
  cost; a query reads one partition at a time.
- `L2` and `Cosine` only. Cosine normalises the vectors it stores, so what the
  index holds is not bit-identical to the dataset's column. `Dot` is refused: see
  `supported_distance_type` for why.
- Address-style row ids only. A dataset created with `enable_stable_row_ids` is
  refused at build and at open.
- A build is reproducible from `BuildParams::seed`, with one hole outside this
  crate's control: Lance re-seeds from the OS when a k-means iteration leaves a
  cluster empty.

## Testing

```
CARGO_INCREMENTAL=0 cargo test
CARGO_INCREMENTAL=0 cargo clippy --all-targets -- -D warnings
```

Nothing here runs in Lance's CI: the crate is not a workspace member. `tests/spike.rs`
is executable documentation of what Lance's public API permits an external index
to do, and it is where the two facts this design rests on are pinned - that an
index with an unresolvable details `type_url` survives a reopen, and that a
compaction strands an index it cannot read rather than deleting it.
