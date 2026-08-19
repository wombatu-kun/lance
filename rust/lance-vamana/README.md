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
- **Nothing is cached between queries unless the index is given a cache.** A
  query keeps a few reads in flight, so its working set is a few partitions
  rather than every partition it probes - and by default the next query pays for
  those same partitions again. `VamanaIndex::with_cache(LanceCache)` changes
  that, and holds the part of a partition that does not depend on the query: the
  layout of its file, and for a lazy walk the codes and row ids it steers by.
  Nothing needs invalidating, because nothing an entry describes can change -
  deleting rows edits no index file, and adding rows or consolidating writes a
  *new* segment under a new uuid.

  The budget is the caller's to set and is in bytes of *resident* form, which is
  more than the codes weigh on disk: they are stored one contiguous stride a
  vertex and read back into the seven columns Lance's estimator wants, with the
  row ids beside them. On SIFT1M that is **89 MB a million rows** against 68 MB
  on disk, and against 776 bytes a row for the vectors themselves.
  An index given no cache reads every time - it holds no empty cache, because a
  cache of capacity zero is not the same thing as none: it admits an entry and
  reclaims it later, so it serves the occasional hit out of what is meant to be
  nothing.
- **A partition is read whole unless the walk is told otherwise.** Reading only
  the vertices a walk touches was measured instead of assumed
  (`examples/memory_gate.rs`): on its own it halves the pages moved at best and
  costs *more* CPU at fine granularity, because a walk scores `R` neighbours for
  every vertex it expands and so touches twenty-five to forty times as many as it
  expands. It pays with quantised codes standing in for those vectors, and only
  while the cache holds a fraction of the index - replaying real probe sequences
  through an LRU that holds all of it serves 25 to 250 queries per load, far past
  the crossover where reading whole was cheaper. Three bits a dimension is what
  "codes" has to mean (`examples/coded_walk.rs`): at three the walk spends two to
  thirteen per cent more comparisons than an exact one at equal recall, at one it
  needs a beam one and a half to three and a half times wider, and either way the
  answer has to be re-scored from the whole candidate list rather than from its
  nearest `K`. Reading a vertex's vector as it is expanded - which DiskANN gets
  free, because one page carries a vertex's edges next to its vector - was
  measured too, and does not pay: correcting a distance seats that vertex at the
  back of the list, the back of the list is the bar a new candidate has to beat,
  and so the walk expands more for it - three times more at one bit. At equal
  work a wider beam on plain codes reaches higher recall.

  Both halves are here. `IndexParams::with_code_bits(3)` builds a partition file
  with a `__code` column beside its vectors and its edges;
  `SearchParams::with_mode(WalkMode::Coded)` walks by it and re-scores the whole
  candidate list exactly, still reading the partition whole; and
  `WalkMode::Lazy` keeps the row ids and the codes and fetches the rest as it
  goes - the out-edges of a vertex when it expands one,
  `SearchParams::with_beam_width` vertices to a request, then the vectors of the
  candidate list in one more.

  On SIFT1M at 65536 rows a partition, four probes and equal recall
  (`examples/lazy_walk.rs`), that is **18.2 MB a query against 198.6 MB** read
  whole, and **18.6 ms of warm CPU against 131.0 ms** - decoding two hundred
  megabytes costs more than fetching eighteen, even with every byte already in
  the page cache. What it pays is round trips: 20 requests become 54, and 28
  iops become 197.

  Nine tenths of that 18.2 MB is the code column, re-read by every query, so the
  mode is only half of the design: **with a cache the same query reads 71.9 kB**,
  0.0004x of reading whole and 254 times less than reading lazily without one,
  in **3.2 ms** against 131.0, holding 89 MB to do it. What the walk itself
  chooses to fetch - the out-edges of the vertices it expands, the vectors of the
  candidates it ends with - is that 71.9 kB and nothing more; the 640 kB between
  it and the code column is `__row_id`, which is read whole beside the codes and
  cached with them. The distances are identical with and without the cache, which
  is the point: it changes what a query reads and nothing else.

  The cache also reverses which granularity is cheaper. Without one, 8192-row
  partitions read less than 65536-row ones (4.5 MB against 18.2); with one it is
  the other way round - **71.9 kB against 119.7 kB, and 3.2 ms against 4.9** -
  because the resident part is paid once while seven probes cost seven entry
  points, seven sets of edges and seven candidate lists against four.

  What that is worth needs something to compare it against, and the nearest
  measured one is Lance's own `IVF_HNSW_SQ` over the same vectors at the same
  recall, counted through `Scanner::scan_stats_callback`: **12.29 MB a query**
  read cold with its partitions tuned to 8192 rows, and 197.78 MB at the shipped
  default of one partition to a million rows. At that same 8192 rows this reads
  4.5 MB cold and 119.7 kB warm. Bytes are all that compares - those numbers
  come from Lance's scanner and these from this driver, the quantisers differ,
  eight-bit scalar against three-bit RaBitQ, and what is held equal is the
  dataset and the recall rather than the index.

  Warm against cold is not the comparison, though, because Lance's reads
  collapse too once its cache holds the partitions. What each has to hold to get
  there is: 257 to 297 bytes a row on disk for that index, and more again in
  Lance's cache, which holds an unpacked form - against 89 bytes a row here,
  which still reads 71.9 kB a query rather than nothing. The saving is the ratio
  between those two resident figures, and it only becomes a saving in bytes read
  when neither budget covers the index.

  Which is what makes the cache load-bearing rather than an optimisation.
  Without one a lazy walk over 65536-row partitions reads 18.2 MB a query, worse
  than the tuned baseline it is meant to beat; at 8192 rows it reads 4.5 MB,
  which is better. Granularity and the cache are chosen together, and neither
  choice survives the other being changed.

  Codes are off by default, and refused rather than skipped for a dimension that
  is not a multiple of eight, which is what RaBitQ packs a bit a dimension into.

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
- its segments disagree about the dimension, the metric, the identifier space or
  the codes, because a query merges their answers.

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

**And a delta is cheap to undo.** Folding eight segments into one costs 1.2s and
takes 9.70ms off every query, so it pays for itself after **123 queries**; four
cost 1.1s and pay back after 296, two cost 0.8s and pay back after 905. The
fold leaves an index a query cannot tell from a one-pass build - the same 74 MiB
in the same 101 files, the same ten partitions and fifty reads per query, at
recall 0.9782 against 0.9777 - for a third of what rebuilding costs. What it does
not give back is the 8% more distances per query a grown graph spends: it moves
the vertices, it does not retrain the router or rebuild the base. That is why
there is no threshold inside `merge_index` and none is wanted. At 123 queries the
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
costs 2.6 rebuilds, a round 0.3 to 0.7 of one.

The two are the same pipeline, and the measurement says so: round for round,
`merge_index` answers with the same recall and the same distances per query as
`consolidate_index` followed by `insert_in_place`, to the last digit, in every
round of that cycle. It runs the same operations over the same data without
putting the partition on disk in between, so what the one pass saves is one read
and one write of the index per round - **6%** of the work here, 8.2s against
8.7s over five rounds. Maintenance is bound by the arithmetic of the graph
rather than by the disk, which is why fusing the passes is worth having and is
not where the money is. Where a partition read is a network round trip the same
two passes are not 6%.

They also answer compaction, which used to need a rebuild. Compaction strands the
index over fragments that no longer exist, and the rows it moved are then rows
this index does not cover - so indexing them again is the whole of the repair,
and the stranded segment goes with the same commit. That is one `merge_index`
call, or `consolidate_index` and then `insert_in_place`.

## Building

- The whole vector column is held in memory for the duration of a build - twice
  over, briefly, while the batches are concatenated. A build is a builder-side
  cost; a query reads one partition at a time.
- Building and every maintenance call work on as many partitions at once as Lance
  gives the compute pool cores, and write them one at a time in id order. That is
  a fivefold saving on twelve cores and a working set of that many partitions:
  `num_partitions` sets both. A query's own bound is separate and unchanged.
- `L2` and `Cosine` only. Cosine normalises the vectors it stores, so what the
  index holds is not bit-identical to the dataset's column. `Dot` is refused: see
  `supported_distance_type` for why.
- Address-style row ids only. A dataset created with `enable_stable_row_ids` is
  refused at build and at open.
- `with_code_bits` mints one RaBitQ rotation for the whole index and every later
  segment inherits it, because a partition copied between two segments carries
  its codes unchanged and a code says nothing about the rotation it was built
  under. A copy between segments that disagree is refused.
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
