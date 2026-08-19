# lance-vamana

A disk-resident Vamana (DiskANN) vector index for Lance datasets, built and
queried entirely through Lance's published API. The crate lives in the Lance
tree for convenience but is not a member of its workspace, so the boundary it
compiles against is the one an out-of-tree crate sees.

```rust
let built = lance_vamana::create_index(&mut dataset, "vamana_idx", &IndexParams::new("vec", 64)).await?;
println!("{} vectors cost {} distance computations to index", built.vectors, built.comparisons);

let index = VamanaIndex::open(&dataset, "vamana_idx").await?;
let answer = index.search(&query, &SearchParams::new(10).with_nprobes(8)).await?;
println!("answered in {} distance computations", answer.comparisons);
for neighbor in &answer.neighbors {
    // `neighbor.row_addr` is a Lance row address; fetch with `Dataset::take_rows`.
}
```

Both halves of the cost are returned rather than logged: a graph is a trade
between what a build pays and what a query pays, and a change that improves one
by spending the other is not visible from either number alone.

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

The mechanism, in both cases, is that Lance decides what an index *is* from the
column it sits on rather than from anything the index says about itself. The
scanner picks a vector index by field id alone, with no type check, so it selects
the Vamana segment and then fails to read it as one of its own. `optimize_indices`
groups indices the same way - `index_group_is_scalar` asks
`is_vector_field(field.data_type())` - and propagates the failure out of the loop
over every index. Renaming `index.idx` would change neither: the file name
decides nothing here.

Consequences to plan around:

- Do not put a Vamana index on a column that is also served by a Lance
  `IVF_HNSW_*` index; whichever appears first in the manifest wins the lookup,
  and a Vamana segment can shadow a working one.
- Anything that calls `optimize_indices` on the dataset - including routine
  maintenance of unrelated scalar indices - will fail while a Vamana index
  exists. Drop the index, maintain, rebuild.
- `use_index(false)` is the escape hatch for Lance-side vector queries.

A fourth path is broken for a different reason, and it is the only one that
breaks **writing**. When the manifest a commit starts from predates Lance 0.8.15
- whose fragment bitmaps could be wrong - or records no writer at all,
`migrate_indices` recalculates every index's fragment coverage, and it does that
by *opening* the index. Lance cannot open this format, so the commit fails
outright. `build_index_segment` therefore refuses such a dataset up front rather
than after the graph has been built, and names the remedy: one commit by any
current Lance build rewrites the manifest with a current writer version, and the
recalculation is gated on the manifest rather than on the age of the data.
Pinned by `a_dataset_older_than_lances_bitmap_fix_is_refused_before_the_build`
against the checked-in `test_data/v0.8.14` fixture; without the refusal that test
fails with `Index with id ... does not exist` after a full build.

## What the query path does not do

The same list lives on the `query` module, next to the code it describes; the
two are meant to say the same thing.

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
- **A fragment the dataset has dropped is answered for by nobody.** A delete that
  empties a fragment, and a compaction that rewrites one, both take it out of the
  dataset, and every vertex stored for it becomes unreachable rather than wrong -
  fragment ids are a high water mark in the manifest, so no stored address can
  ever come to mean another row. The index narrows itself to what is left and
  reports the result from `VamanaIndex::covered_fragments`, which is what the
  unindexed remainder should be computed against. After a compaction the rows
  are all still in the dataset, in fragments this index does not cover.
- **No predicate prefilter and no refine step.** Both live in Lance's scanner,
  which this driver bypasses.
- **Partitions are read whole, and nothing is cached between queries.** A query
  keeps a few reads in flight, so its working set is a few partitions rather
  than every partition it probes - but a lazy per-vertex traversal and a cache
  budget are both still ahead.

An index is **refused** at open, rather than answering from what is left, when:

- the dataset has edited a segment's coverage while the fragments are still
  there - an in-place column update prunes the rewritten fragments out of the
  index's `fragment_bitmap` while leaving every row address valid, which no
  liveness check can see;
- the dataset credits a segment with a fragment that segment never read, so it
  is expected to answer for rows it has never seen;
- an overlay has replaced the indexed values under a covered fragment, so the
  vectors ranked are not the ones the rows now hold;
- the manifest records a format version this build does not read;
- a segment was inherited from another dataset by a shallow clone, so its files
  live under a base path this crate cannot resolve;
- its segments disagree about the dimension, the metric or the identifier space,
  because a query merges their answers.

In every case the answer is to rebuild the index.

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

Run these from `rust/lance-vamana`. The crate is its own workspace root, so the
repository's own `cargo test --workspace`, `cargo clippy --all` and
`cargo fmt --all` do not reach it - a change here can leave the root workspace
green and this crate broken.

```
cd rust/lance-vamana
CARGO_INCREMENTAL=0 cargo fmt --all
CARGO_INCREMENTAL=0 cargo clippy --all-targets -- -D warnings
CARGO_INCREMENTAL=0 cargo test
```

`CARGO_INCREMENTAL=0` because the incremental cache across the fork's three
workspaces grows to tens of gigabytes.

Nothing here runs in Lance's CI, for the same reason. `tests/spike.rs`
is executable documentation of what Lance's public API permits an external index
to do, and it is where the two facts this design rests on are pinned - that an
index with an unresolvable details `type_url` survives a reopen, and that a
compaction strands an index it cannot read rather than deleting it.
