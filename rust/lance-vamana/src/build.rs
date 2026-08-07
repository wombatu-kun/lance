// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building a partition's graph.

use std::collections::VecDeque;

use lance_core::{Error, Result};
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::storage::{DistCalculator, VectorStore};

use crate::search::Comparisons;

/// Choose up to `max_degree` out-edges for `point` from `candidates`.
///
/// Algorithm 2 of the DiskANN paper. Candidates are taken nearest first, and
/// after each pick the rest are swept: a candidate that sits `alpha` times
/// closer to the vertex just picked than to `point` is dropped, because the
/// picked vertex already routes there. That sweep is what makes the result a
/// spread of directions rather than the `max_degree` nearest points, and it is
/// what lets a walk make progress instead of circling.
///
/// `alpha` is the slack in that test. At `1.0` it reduces exactly to the
/// diversity heuristic Lance applies to HNSW; above `1.0` fewer candidates are
/// dropped, so the graph keeps more of its short edges. `candidates` carry
/// their distance to `point` and may contain `point` itself and duplicates.
pub fn robust_prune<S: VectorStore>(
    store: &S,
    point: u32,
    candidates: Vec<OrderedNode>,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Vec<u32>> {
    if alpha.is_nan() || alpha < 1.0 {
        return Err(Error::invalid_input(format!(
            "Vamana alpha must be at least 1.0, got {alpha}"
        )));
    }
    if max_degree == 0 {
        return Err(Error::invalid_input(
            "Vamana max_degree must be greater than zero".to_string(),
        ));
    }

    let mut pool = candidates;
    pool.retain(|candidate| candidate.id != point);
    pool.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
    pool.dedup_by_key(|candidate| candidate.id);
    let mut pool = VecDeque::from(pool);

    let mut selected = Vec::with_capacity(max_degree);
    while let Some(nearest) = pool.pop_front() {
        selected.push(nearest.id);
        if selected.len() == max_degree {
            break;
        }
        let from_nearest = store.dist_calculator_from_id(nearest.id);
        comparisons.record(pool.len() as u64);
        pool.retain(|candidate| alpha * from_nearest.distance(candidate.id) > candidate.dist.0);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::graph::OrderedFloat;
    use lance_linalg::distance::DistanceType;

    use super::*;

    /// Deterministic pseudo-random vectors: a fixed multiplicative congruential
    /// sequence, so the cross-check against Lance runs on the same points every
    /// time without pulling in an RNG.
    fn scattered_storage(num_vertices: usize, dimension: usize) -> FlatFloatStorage {
        let mut state = 12345u64;
        let values = Float32Array::from(
            (0..num_vertices * dimension)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect::<Vec<_>>(),
        );
        FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap(),
            DistanceType::L2,
        )
    }

    fn all_candidates(
        storage: &FlatFloatStorage,
        point: u32,
        num_vertices: usize,
    ) -> Vec<OrderedNode> {
        let calculator = storage.dist_calculator_from_id(point);
        (0..num_vertices as u32)
            .map(|id| OrderedNode::new(id, OrderedFloat(calculator.distance(id))))
            .collect()
    }

    /// What Lance's own diversity heuristic would select, walking candidates
    /// nearest first. `prefers_candidate` is the same predicate at `alpha = 1`,
    /// so the two must agree vertex for vertex.
    fn lance_selection(
        storage: &FlatFloatStorage,
        point: u32,
        candidates: &[OrderedNode],
        max_degree: usize,
    ) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.retain(|candidate| candidate.id != point);
        sorted.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));

        let mut selected: Vec<OrderedNode> = Vec::with_capacity(max_degree);
        for candidate in sorted {
            if selected.len() == max_degree {
                break;
            }
            if storage.prefers_candidate(&candidate, &selected) {
                selected.push(candidate);
            }
        }
        selected.into_iter().map(|node| node.id).collect()
    }

    #[test]
    fn alpha_one_reproduces_lance_diversity_exactly() {
        const VERTICES: usize = 200;
        let storage = scattered_storage(VERTICES, 8);

        for point in [0u32, 7, 63, 199] {
            let candidates = all_candidates(&storage, point, VERTICES);
            let ours = robust_prune(
                &storage,
                point,
                candidates.clone(),
                1.0,
                16,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(
                ours,
                lance_selection(&storage, point, &candidates, 16),
                "vertex {point}"
            );
        }
    }

    /// With enough slack nothing is ever dropped, so the result must be plain
    /// nearest-neighbours. This pins which way the alpha test points: an
    /// inverted comparison would prune everything here instead of nothing.
    #[test]
    fn unbounded_alpha_selects_the_nearest_candidates() {
        const VERTICES: usize = 64;
        let storage = scattered_storage(VERTICES, 4);
        let candidates = all_candidates(&storage, 0, VERTICES);

        let mut nearest = candidates.clone();
        nearest.retain(|candidate| candidate.id != 0);
        nearest.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        let nearest = nearest
            .iter()
            .take(8)
            .map(|node| node.id)
            .collect::<Vec<_>>();

        let selected = robust_prune(
            &storage,
            0,
            candidates,
            f32::MAX,
            8,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(selected, nearest);
    }

    /// The other end of the same axis: at `alpha = 1` on scattered data the
    /// sweep must actually throw candidates away, or the test above is
    /// measuring nothing.
    #[test]
    fn alpha_one_prunes_more_than_unbounded_alpha() {
        const VERTICES: usize = 200;
        let storage = scattered_storage(VERTICES, 8);
        let candidates = all_candidates(&storage, 0, VERTICES);

        let diverse = robust_prune(
            &storage,
            0,
            candidates.clone(),
            1.0,
            VERTICES,
            &Comparisons::default(),
        )
        .unwrap();
        assert!(
            diverse.len() < VERTICES - 1,
            "nothing was pruned: {} of {} candidates survived",
            diverse.len(),
            VERTICES - 1
        );
    }

    #[test]
    fn the_nearest_candidate_is_always_kept() {
        const VERTICES: usize = 64;
        let storage = scattered_storage(VERTICES, 4);
        let candidates = all_candidates(&storage, 3, VERTICES);
        let nearest = candidates
            .iter()
            .filter(|candidate| candidate.id != 3)
            .min_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)))
            .unwrap()
            .id;

        for alpha in [1.0, 1.2, 2.0] {
            let selected = robust_prune(
                &storage,
                3,
                candidates.clone(),
                alpha,
                8,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(selected[0], nearest, "at alpha {alpha}");
        }
    }

    #[test]
    fn the_point_and_its_duplicates_never_become_edges() {
        const VERTICES: usize = 32;
        let storage = scattered_storage(VERTICES, 4);
        let mut candidates = all_candidates(&storage, 5, VERTICES);
        candidates.extend(all_candidates(&storage, 5, VERTICES));

        let selected =
            robust_prune(&storage, 5, candidates, 1.2, 16, &Comparisons::default()).unwrap();
        assert!(!selected.contains(&5), "the vertex selected itself");
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            selected.len(),
            "duplicate out-edge: {selected:?}"
        );
        assert!(selected.len() <= 16);
    }

    #[test]
    fn an_alpha_below_one_is_rejected() {
        let storage = scattered_storage(8, 2);
        let error = robust_prune(
            &storage,
            0,
            all_candidates(&storage, 0, 8),
            0.9,
            4,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least 1.0"), "{error}");
    }

    /// A NaN alpha would make every prune test false and silently keep the
    /// nearest `max_degree` candidates, so it is rejected rather than compared.
    #[test]
    fn a_nan_alpha_is_rejected() {
        let storage = scattered_storage(8, 2);
        let error = robust_prune(
            &storage,
            0,
            all_candidates(&storage, 0, 8),
            f32::NAN,
            4,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least 1.0"), "{error}");
    }
}
