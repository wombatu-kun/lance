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
- **Rows added after the build are invisible** until they are indexed. The index
  answers from the fragments it was built over. Lance's scanner would scan the
  unindexed remainder; this driver does not. `insert_as_segment` is how they
  stop being invisible.
- **A fragment the dataset has dropped is answered for by nobody.** A delete that
  empties a fragment, and a compaction that rewrites one, both take it out of the
  dataset, and every vertex stored for it becomes unreachable rather than wrong -
  fragment ids are a high water mark in the manifest, so no stored address can
  ever come to mean another row. The index narrows itself to what is left and
  reports the result from `VamanaIndex::covered_fragments`, which is what the
  unindexed remainder should be computed against. After a compaction the rows
  are all still in the dataset, in fragments this index does not cover - which
  makes them ordinary new rows, and `insert_as_segment` brings them back.
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

## Maintenance

Four calls, and none of them takes a parameter beyond the dataset and the index
name. Everything else - the column, the metric, the width, the degree, the beam,
the pruning slack - is already recorded in the index, and taking it from anywhere
else would be a second copy of one number for the two to disagree about.

- `insert_as_segment` indexes every row the index does not cover yet, as a new
  segment beside the base. It inherits the base's centroids, so every segment of
  an index shares one partition numbering. Cheap to run and cheap in recall; what
  it costs is that a query probes `nprobes` partitions **per segment**.
- `insert_in_place` puts those same rows into the base's own graphs instead:
  routed by the base's centroids, each partition that drew any of them read,
  grown and rewritten, and the rest copied across undecoded. One segment stays
  one segment.
- `consolidate_index` takes the dataset's deleted rows out of the graphs that
  still hold them. On SIFT 100k it returns the deleted share in bytes to within
  half a percentage point and returns almost nothing in recall: a tombstone is
  nearly free to search past while the beam is wide next to the reciprocal of
  the live fraction.
- `merge_index` does all of that in one pass, and folds delta segments back into
  the base, which is the one thing none of the others can do. A partition is read
  once from each segment holding a piece of it and written once, whatever had to
  happen to it, and there is no order to get wrong.

**Which one to call.** Delta segments to be rid of: `merge_index`, because
nothing else removes them. Only new rows: `insert_as_segment` to make them
searchable now, `insert_in_place` to keep the read cost flat. Only deletions:
`consolidate_index`, which is cheaper than merging because it never reads the
dataset's vector column and never routes. Anything else, or more than one of them
at once: `merge_index`.

Which insert to use is a question about read operations, not about recall.
Measured on SIFT 100k over indices covering the same rows and differing only in
how they came to exist, eight segments against one cost 8x the partition reads,
8x the files and about 3x the latency, for +10% bytes and **no** loss of recall.
A delta is nearly free to a query's bandwidth and expensive to its IOPS.

**And a delta is cheap to undo.** Folding eight segments into one costs 4.5s and
takes 8.47ms off every query, so it pays for itself after **533 queries**; four
cost 4.3s and pay back after 1345, two cost 3.5s and pay back after 6509. The
fold leaves an index a query cannot tell from a one-pass build - the same 74 MiB
in the same 101 files, the same ten partitions and fifty reads per query, at
recall 0.9782 against 0.9777 - for a third of what rebuilding costs. What it does
not give back is the 8% more distances per query a grown graph spends: it moves
the vertices, it does not retrain the router or rebuild the base. That is why
there is no threshold inside `merge_index` and none is wanted. At 533 queries the
policy is "fold if anyone reads this index at all".

**Order matters between the two calls: consolidate, then insert.** A delete that
empties a fragment takes it out of the dataset, and `insert_in_place` refuses a
segment built over a fragment that is gone - rewriting one would store its
vertices under a coverage that no longer names them, where nothing would keep
them out of an answer. Consolidation removes exactly those vertices. The other
order works until the first fragment empties and then stops working. There is no
such order to `merge_index`: the pass that adds the new vertices is the pass that
drops the ones whose fragment left.

Run either and the index survives being replaced outright. Five rounds of "delete
a fifth, append as many rows as were removed, bring the index back up to date" on
SIFT 100k leave nothing of the original data and cost **0.14 of a percentage
point** of recall, against the roughly one point the FreshVamana paper allows.
What churn actually costs is arithmetic: the worn graph spends 8.5% more
distances per query than a rebuild, which also takes the recall back. The cycle
costs 2.4 rebuilds, a round 0.3 to 0.6 of one.

The two are the same pipeline, and the measurement says so: round for round,
`merge_index` answers with the same recall and the same distances per query as
`consolidate_index` followed by `insert_in_place`, to the last digit, in every
round of that cycle. It runs the same operations over the same data without
putting the partition on disk in between, so what the one pass saves is one read
and one write of the index per round - **3%** of the work here, 35.9s against
37.0s over five rounds. Maintenance is bound by the arithmetic of the graph
rather than by the disk, which is why fusing the passes is worth having and is
not where the money is. Where a partition read is a network round trip the same
two passes are not 3%.

They also answer compaction, which used to need a rebuild. Compaction strands the
index over fragments that no longer exist, and the rows it moved are then rows
this index does not cover - so indexing them again is the whole of the repair,
and the stranded segment goes with the same commit. That is one `merge_index`
call, or `consolidate_index` and then `insert_in_place`.

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
