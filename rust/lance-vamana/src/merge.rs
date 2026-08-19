// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Applying everything a partition has pending, in one pass over it.
//!
//! [`crate::consolidate`] takes a partition's dead rows out and
//! [`crate::insert`] puts new ones in. Doing both, as a round of maintenance
//! does, means reading the partition, writing it, reading it back and writing it
//! again - and a partition file is the unit of rewrite, so that is twice the I/O
//! of the work itself. This is the two composed into one call: consolidate,
//! insert, and decide once whether what came out is still a graph.
//!
//! The order inside is forced, and the other way round is not merely slower.
//! Consolidation compacts the local ids, so no vertex keeps its number and no
//! entry point keeps its meaning; insertion appends and moves nothing. Inserting
//! first would renumber the rows just inserted for no reason, and would run the
//! one-hop repair across edges the insertion had only just pruned.
//!
//! This is the `StreamingMerge` of the FreshDiskANN paper without its third
//! phase. The paper deletes, inserts, and then patches the back-edges in
//! separately, because it edits an SSD-resident index in place while queries are
//! reading it. Here a partition is read into memory, rewritten and committed as
//! a new file, so the back-edges are applied where they are computed and there
//! is nothing left to patch.
//!
//! # The reachability check stays where consolidation put it
//!
//! [`crate::consolidate`] measured how the one-hop repair fails: it guarantees
//! that no edge dangles, not that the graph stays in one piece, and 90% of a
//! 1000-vertex partition removed evenly leaves 2 of 100 survivors reachable. So
//! a consolidated graph is walked, and one that came apart is built again -
//! here, over the survivors **and the newcomers together**, which is the one
//! thing this ordering buys over running the two steps in turn. The two-pass
//! round rebuilt over the survivors alone and then inserted into the result.
//!
//! Insertion can leave a vertex unreachable too, and it was tempting to walk the
//! graph once at the very end instead. Measured, that is the wrong trade. A back
//! edge is fought for through a prune, so a new point whose every chosen
//! neighbour rejected it ends up with out-edges and no in-edges - and the same
//! is true of a *build*, which is repeated insertion: 1000 uniform points at
//! `R=12, L=40` come out of `build_partition` with 999 of them reachable, and
//! inserting 500 more leaves 1496 of 1500. A check at the end would answer "not
//! whole" for perfectly ordinary partitions and pay a full rebuild to recover
//! one vertex in a thousand.
//!
//! So an orphaned vertex is left where the algorithm put it: stored, walked
//! past, and returned only when a query happens to reach it another way. What is
//! lost is a fraction of a percent of the rows, against a rebuild of everything;
//! the five-round churn measurement behind [`crate::inserter::insert_in_place`]
//! is the evidence that it does not accumulate into anything.

use arrow_array::FixedSizeListArray;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use roaring::RoaringBitmap;

use crate::build::{BuildParams, build_partition};
use crate::consolidate::consolidate_partition;
use crate::insert::{concat_vectors, insert_into_partition};
use crate::partition::Partition;
use crate::search::{Comparisons, flat_storage};

/// The rows joining a partition, and the vectors that place them.
///
/// A pair rather than two parameters, so that a merge can say it has none: a
/// partition that only lost rows is the consolidation case, and saying so with
/// an empty slice would still need an empty vector array of the right dimension
/// fabricated at every call site to go with it.
#[derive(Debug, Clone, Copy)]
pub struct Newcomers<'a> {
    pub row_ids: &'a [u64],
    pub vectors: &'a FixedSizeListArray,
}

/// A partition with its deletions taken out and its new rows put in.
#[derive(Debug)]
pub struct Merged {
    pub partition: Partition,
    /// Where a search of the result should start.
    pub medoid: u32,
    /// Whether the one-hop repair left the graph in pieces, so that it had to be
    /// built from scratch over what survived and what was joining it.
    ///
    /// Not a failure and not a fallback: a graph in pieces is one a search
    /// reaches one island of, and every vertex outside that island would be
    /// stored, read and never returned.
    pub rebuilt: bool,
}

/// Take `dead` out of `base` and put `newcomers` in.
///
/// `entry_point` is where the insertion's searches start, and it is read only
/// when nothing is dead: consolidation moves every local id and returns the
/// entry point of the compacted partition, which is the one the insertion then
/// uses. Passing the partition's recorded medoid is always right.
///
/// The pruning slack, the beam and the degree all come from `params`, which is
/// the caller's copy of what the segment records - a partition that is rewritten
/// has to end up matching the siblings it will sit beside. `params.seed` decides
/// the order the newcomers are linked in and, should it come to that, the order
/// a rebuild sweeps in; see [`crate::build::MAINTENANCE_SEED`].
///
/// A partition with nothing dead and nothing new is refused rather than returned
/// unchanged, because there is a cheaper answer for one of those than anything
/// this function can do: it is carried into the new segment as the bytes it
/// already is.
pub fn merge_partition(
    base: &Partition,
    entry_point: u32,
    dead: &RoaringBitmap,
    newcomers: Option<Newcomers<'_>>,
    distance_type: DistanceType,
    params: &BuildParams,
    comparisons: &Comparisons,
) -> Result<Merged> {
    let consolidated = if dead.is_empty() {
        None
    } else {
        let consolidated =
            consolidate_partition(base, dead, distance_type, params.alpha, comparisons)?;
        let graph = consolidated.partition.graph();
        if graph.reachable_from(consolidated.medoid)? != graph.len() {
            return rebuild(
                consolidated.partition,
                newcomers,
                distance_type,
                params,
                comparisons,
            );
        }
        Some(consolidated)
    };

    let (partition, medoid) = match (consolidated, newcomers) {
        (None, None) => {
            return Err(Error::invalid_input(
                "Vamana was asked to merge a partition with nothing deleted from it and nothing \
                 to add to it; such a partition is carried across as the bytes it already is"
                    .to_string(),
            ));
        }
        (Some(consolidated), None) => (consolidated.partition, consolidated.medoid),
        (None, Some(newcomers)) => {
            let inserted = insert_into_partition(
                base,
                newcomers.row_ids,
                newcomers.vectors,
                entry_point,
                distance_type,
                params,
                comparisons,
            )?;
            (inserted.partition, inserted.medoid)
        }
        (Some(consolidated), Some(newcomers)) => {
            let inserted = insert_into_partition(
                &consolidated.partition,
                newcomers.row_ids,
                newcomers.vectors,
                consolidated.medoid,
                distance_type,
                params,
                comparisons,
            )?;
            (inserted.partition, inserted.medoid)
        }
    };

    Ok(Merged {
        partition,
        medoid,
        rebuilt: false,
    })
}

/// Build a graph over what a torn partition still holds, plus whatever was
/// joining it.
///
/// One build rather than a build followed by an insertion. The newcomers were
/// going to be linked into this graph anyway, and a build that has them from the
/// start sweeps them twice like everything else instead of once into a graph
/// they had no part in shaping.
fn rebuild(
    survivors: Partition,
    newcomers: Option<Newcomers<'_>>,
    distance_type: DistanceType,
    params: &BuildParams,
    comparisons: &Comparisons,
) -> Result<Merged> {
    let (graph, kept) = survivors.into_parts();
    let mut row_ids = graph.row_ids().to_vec();
    let vectors = match newcomers {
        Some(newcomers) => {
            row_ids.extend_from_slice(newcomers.row_ids);
            concat_vectors(&[kept, newcomers.vectors.clone()])?
        }
        None => kept,
    };
    let built = {
        let store = flat_storage(&row_ids, &vectors, distance_type)?;
        build_partition(&store, params, comparisons)?
    };
    Ok(Merged {
        partition: Partition::try_new(built.graph, vectors)?,
        medoid: built.medoid,
        rebuilt: true,
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::graph::OrderedFloat;
    use lance_index::vector::storage::{DistCalculator, VectorStore};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;
    use std::sync::Arc;

    use super::*;
    use crate::search::{SearchScratch, greedy_search};

    const MAX_DEGREE: u32 = 16;

    fn params() -> BuildParams {
        BuildParams {
            max_degree: MAX_DEGREE,
            search_list_size: 32,
            alpha: 1.2,
            seed: 42,
        }
    }

    fn merge(
        base: &Partition,
        entry_point: u32,
        dead: &RoaringBitmap,
        newcomers: Option<Newcomers<'_>>,
    ) -> Result<Merged> {
        merge_partition(
            base,
            entry_point,
            dead,
            newcomers,
            DistanceType::L2,
            &params(),
            &Comparisons::default(),
        )
    }

    /// The 1000-vertex fixture the one-hop repair is characterised on, in
    /// [`crate::consolidate`] and in [`crate::consolidator`] before this module
    /// took its rebuild over. The reachability numbers quoted in both apply to
    /// exactly this graph, so it is copied rather than improved: a different
    /// cloud tears at a different fraction and the tests below would be pinning
    /// nothing in particular.
    mod characterised {
        use super::*;

        pub const VERTICES: usize = 1000;
        pub const DIMENSION: usize = 4;

        pub fn partition() -> Partition {
            let values = Float32Array::from(
                (0..VERTICES * DIMENSION)
                    .map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0)
                    .collect::<Vec<_>>(),
            );
            let vectors =
                FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
            let storage = FlatFloatStorage::new(vectors.clone(), DistanceType::L2);
            let built = build_partition(&storage, &params(), &Comparisons::default()).unwrap();
            Partition::try_new(built.graph, vectors).unwrap()
        }

        /// The `count` vertices nearest to vertex 0, which is what deleting a
        /// class or a category looks like in the space the index measures.
        pub fn clustered(partition: &Partition, count: usize) -> RoaringBitmap {
            let store = flat_storage(
                partition.graph().row_ids(),
                partition.vectors(),
                DistanceType::L2,
            )
            .unwrap();
            let from_center = store.dist_calculator_from_id(0);
            let mut scored = (0..partition.len() as u32)
                .map(|id| (OrderedFloat(from_center.distance(id)), id))
                .collect::<Vec<_>>();
            scored.sort_unstable();
            scored.into_iter().take(count).map(|(_, id)| id).collect()
        }

        /// Vectors from the same cloud, offset far enough along the sequence
        /// that they repeat none of it.
        pub fn newcomers(count: usize) -> FixedSizeListArray {
            let values = Float32Array::from(
                (0..count * DIMENSION)
                    .map(|i| (((i + VERTICES * DIMENSION) * 2654435761) % 1000) as f32 / 1000.0)
                    .collect::<Vec<_>>(),
            );
            FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap()
        }
    }

    /// Uniform noise at a width where a graph has something to get wrong, for
    /// the tests that are about the merge itself rather than about tearing.
    ///
    /// The lattice above answers every query perfectly at any beam, which is
    /// what makes it a clean reachability fixture and a useless search one.
    mod noise {
        use super::*;

        pub const DIMENSION: usize = 32;

        pub fn vectors(count: usize, offset: usize) -> FixedSizeListArray {
            let mut rng = SmallRng::seed_from_u64(offset as u64 + 1);
            let values = (0..count * DIMENSION)
                .map(|_| rng.random::<f32>())
                .collect::<Vec<_>>();
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIMENSION as i32)
                .unwrap()
        }

        /// A partition over `count` points whose row ids are `1000 + id`, so
        /// that a local id read where a row id belongs is visible.
        pub fn partition(count: usize) -> (Partition, u32) {
            let vectors = vectors(count, 0);
            let row_ids = (0..count as u64).map(|id| 1000 + id).collect::<Vec<_>>();
            let store = flat_storage(&row_ids, &vectors, DistanceType::L2).unwrap();
            let built = build_partition(&store, &params(), &Comparisons::default()).unwrap();
            (
                Partition::try_new(built.graph, vectors).unwrap(),
                built.medoid,
            )
        }
    }

    /// Whether a walk from `medoid` finds the vertex holding `row_id`.
    fn finds(partition: &Partition, medoid: u32, row_id: u64) -> bool {
        let vertex = partition
            .graph()
            .row_ids()
            .iter()
            .position(|id| *id == row_id)
            .expect("the row id is not in the partition at all") as u32;
        let vector: ArrayRef = Arc::new(Float32Array::from(
            partition.vector(vertex).unwrap().to_vec(),
        ));
        let store = flat_storage(
            partition.graph().row_ids(),
            partition.vectors(),
            DistanceType::L2,
        )
        .unwrap();
        let mut scratch = SearchScratch::new(partition.len());
        greedy_search(
            partition.graph(),
            &store.dist_calculator(vector, 0.0),
            medoid,
            params().search_list_size,
            &mut scratch,
            &Comparisons::default(),
        )
        .unwrap()
        .candidates
        .iter()
        .any(|node| node.id == vertex)
    }

    /// Every stored edge points at a vertex that exists and no list is wider
    /// than the format's slot count.
    fn assert_well_formed(partition: &Partition, medoid: u32) {
        let graph = partition.graph();
        for vertex in 0..graph.len() as u32 {
            let neighbors = graph.neighbors(vertex).unwrap();
            assert!(
                neighbors.len() <= MAX_DEGREE as usize,
                "vertex {vertex} has degree {}",
                neighbors.len()
            );
            for neighbor in neighbors {
                assert!(
                    (*neighbor as usize) < graph.len(),
                    "vertex {vertex} points at {neighbor} in a graph of {}",
                    graph.len()
                );
            }
        }
        assert_eq!(
            graph.reachable_from(medoid).unwrap(),
            graph.len(),
            "the merged graph is in pieces"
        );
    }

    /// Nothing to take out and nothing to put in has a cheaper answer than
    /// anything this can do, so it is the caller's mistake rather than a
    /// no-op returned quietly.
    #[test]
    fn merging_a_partition_with_nothing_pending_is_refused() {
        let (partition, medoid) = noise::partition(100);
        let error = merge(&partition, medoid, &RoaringBitmap::new(), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("nothing deleted"), "{error}");
    }

    /// With no newcomers the merge *is* the consolidation, and the graph it
    /// writes has to be the one the consolidation produced rather than something
    /// re-derived on the way through.
    #[test]
    fn with_nothing_to_insert_it_consolidates_and_nothing_more() {
        let partition = characterised::partition();
        let dead = characterised::clustered(&partition, characterised::VERTICES / 2);

        let expected = consolidate_partition(
            &partition,
            &dead,
            DistanceType::L2,
            params().alpha,
            &Comparisons::default(),
        )
        .unwrap();
        let merged = merge(&partition, 0, &dead, None).unwrap();

        assert!(!merged.rebuilt, "a graph that held together was rebuilt");
        assert_eq!(merged.medoid, expected.medoid);
        assert_eq!(
            merged.partition.graph().row_ids(),
            expected.partition.graph().row_ids()
        );
        for vertex in 0..expected.partition.len() as u32 {
            assert_eq!(
                merged.partition.graph().neighbors(vertex).unwrap(),
                expected.partition.graph().neighbors(vertex).unwrap(),
                "vertex {vertex} came out of the merge with different edges"
            );
        }
    }

    /// And with nothing deleted the merge is the insertion, from the entry point
    /// it was handed - the one case where that argument is read at all.
    #[test]
    fn with_nothing_deleted_it_inserts_from_the_entry_point_it_was_given() {
        const OLD: usize = 400;
        const NEW: usize = 100;
        let (partition, medoid) = noise::partition(OLD);
        let vectors = noise::vectors(NEW, OLD);
        let row_ids = (0..NEW as u64).map(|id| 5000 + id).collect::<Vec<_>>();

        let expected = insert_into_partition(
            &partition,
            &row_ids,
            &vectors,
            medoid,
            DistanceType::L2,
            &params(),
            &Comparisons::default(),
        )
        .unwrap();
        let merged = merge(
            &partition,
            medoid,
            &RoaringBitmap::new(),
            Some(Newcomers {
                row_ids: &row_ids,
                vectors: &vectors,
            }),
        )
        .unwrap();

        assert!(!merged.rebuilt);
        assert_eq!(merged.medoid, expected.medoid);
        assert_eq!(
            merged.partition.graph().row_ids(),
            expected.partition.graph().row_ids()
        );
        for vertex in 0..expected.partition.len() as u32 {
            assert_eq!(
                merged.partition.graph().neighbors(vertex).unwrap(),
                expected.partition.graph().neighbors(vertex).unwrap()
            );
        }
    }

    /// Both halves in one call: the dead are gone, the new are in, the survivors
    /// kept their rows, and everything left is findable from the entry point
    /// that comes back.
    ///
    /// The row ids are what makes this more than a graph test. Consolidation
    /// renumbers every vertex and insertion appends after the renumbering, so a
    /// merge that ran the two in the other order would produce a graph passing
    /// every structural assertion here while answering with the wrong rows.
    #[test]
    fn a_merge_takes_the_dead_out_and_puts_the_new_in() {
        const OLD: usize = 600;
        const NEW: usize = 150;
        let (partition, medoid) = noise::partition(OLD);
        let dead = (0..OLD as u32)
            .filter(|id| id % 4 == 0)
            .collect::<RoaringBitmap>();
        let vectors = noise::vectors(NEW, OLD);
        let row_ids = (0..NEW as u64).map(|id| 5000 + id).collect::<Vec<_>>();

        let merged = merge(
            &partition,
            medoid,
            &dead,
            Some(Newcomers {
                row_ids: &row_ids,
                vectors: &vectors,
            }),
        )
        .unwrap();

        let survivors = OLD - dead.len() as usize;
        assert_eq!(merged.partition.len(), survivors + NEW);
        assert_well_formed(&merged.partition, merged.medoid);

        let held = merged
            .partition
            .graph()
            .row_ids()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(held.len(), survivors + NEW, "a row id is stored twice");
        for old in 0..OLD as u32 {
            let row_id = partition.graph().row_ids()[old as usize];
            assert_eq!(
                held.contains(&row_id),
                !dead.contains(old),
                "row {row_id} is on the wrong side of the deletion"
            );
        }
        for row_id in &row_ids {
            assert!(held.contains(row_id), "newcomer {row_id} was not stored");
        }
        assert!(
            finds(&merged.partition, merged.medoid, row_ids[NEW / 2]),
            "a newcomer is stored but the walk cannot reach it"
        );
        assert!(
            finds(
                &merged.partition,
                merged.medoid,
                partition.graph().row_ids()[1]
            ),
            "a survivor is stored but the walk cannot reach it"
        );
    }

    /// 90% removed evenly is the case the one-hop repair cannot hold together -
    /// 2 of 100 survivors reachable, measured in [`crate::consolidate`]. What
    /// comes out is whole because it was built again, and the newcomers are in
    /// it: a rebuild that ran over the survivors alone, as the two-pass round
    /// did, would have thrown the batch away.
    #[test]
    fn a_partition_the_repair_tore_apart_is_rebuilt_with_its_newcomers() {
        const NEW: usize = 40;
        let partition = characterised::partition();
        let dead = (0..characterised::VERTICES as u32)
            .filter(|id| id % 10 != 0)
            .collect::<RoaringBitmap>();
        let vectors = characterised::newcomers(NEW);
        let row_ids = (0..NEW as u64).map(|id| 5000 + id).collect::<Vec<_>>();

        let merged = merge(
            &partition,
            0,
            &dead,
            Some(Newcomers {
                row_ids: &row_ids,
                vectors: &vectors,
            }),
        )
        .unwrap();

        assert!(merged.rebuilt, "the repair held, so nothing was rebuilt");
        assert_eq!(
            merged.partition.len(),
            characterised::VERTICES / 10 + NEW,
            "the rebuild lost the newcomers"
        );
        assert_well_formed(&merged.partition, merged.medoid);
    }

    /// The same 90%, removed as a region of the space instead: the survivors
    /// outside it keep their neighbourhoods, the repair holds, and the rebuild
    /// must *not* fire. Without this case "rebuild always" would pass the test
    /// above.
    #[test]
    fn a_partition_the_repair_held_together_is_not_rebuilt() {
        let partition = characterised::partition();
        let dead = characterised::clustered(&partition, characterised::VERTICES * 9 / 10);

        let merged = merge(&partition, 0, &dead, None).unwrap();

        assert!(!merged.rebuilt, "a graph that was whole was rebuilt anyway");
        assert_eq!(merged.partition.len(), characterised::VERTICES / 10);
        assert_well_formed(&merged.partition, merged.medoid);
    }

    /// A rebuild is a build, and it runs with the parameters it was handed
    /// rather than with a default of its own: the partition it replaces sits
    /// beside siblings built with those numbers.
    ///
    /// The degree is visible in the graph. The beam is not, so it is pinned
    /// through what the build spends: a wider one visits more candidates per
    /// insertion, and a rebuild that ignored the value would cost the same
    /// either way.
    #[test]
    fn a_rebuild_uses_the_parameters_it_was_handed() {
        let partition = characterised::partition();
        let dead = (0..characterised::VERTICES as u32)
            .filter(|id| id % 10 != 0)
            .collect::<RoaringBitmap>();
        let rebuild_with = |params: BuildParams| {
            let comparisons = Comparisons::default();
            let merged = merge_partition(
                &partition,
                0,
                &dead,
                None,
                DistanceType::L2,
                &params,
                &comparisons,
            )
            .unwrap();
            assert!(merged.rebuilt, "this case is supposed to reach a rebuild");
            (merged, comparisons.get())
        };

        let (narrow, _) = rebuild_with(BuildParams {
            max_degree: 4,
            ..params()
        });
        assert_eq!(
            narrow.partition.graph().max_degree(),
            4,
            "the rebuild ignored the degree it was handed"
        );

        let (_, recorded) = rebuild_with(params());
        let (_, wide) = rebuild_with(BuildParams {
            search_list_size: 200,
            ..params()
        });
        assert!(
            wide > recorded,
            "a beam of 200 cost {wide} distances and a beam of 32 cost {recorded}"
        );
    }
}
