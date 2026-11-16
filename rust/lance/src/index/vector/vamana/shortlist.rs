// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::{DIST_COL, VECTOR_RESULT_SCHEMA};

/// A query retains only these two values per candidate, independently of the
/// number of partitions searched. Equal distances are ordered by row id.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    distance: f32,
    row_id: u64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.row_id.cmp(&other.row_id))
    }
}

pub(super) struct Shortlist {
    candidates: BinaryHeap<Candidate>,
    limit: usize,
}

impl Shortlist {
    pub fn new(limit: usize) -> Self {
        Self {
            candidates: BinaryHeap::new(),
            limit,
        }
    }

    pub fn extend(&mut self, batch: &RecordBatch) -> Result<()> {
        let distances = batch
            .column_by_name(DIST_COL)
            .ok_or_else(|| Error::internal(format!("Vamana candidates are missing {DIST_COL}")))?
            .as_primitive::<Float32Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal(format!("Vamana candidates are missing {ROW_ID}")))?
            .as_primitive::<UInt64Type>();
        for (distance, row_id) in distances.values().iter().zip(row_ids.values()) {
            self.insert(Candidate {
                distance: *distance,
                row_id: *row_id,
            });
        }
        Ok(())
    }

    fn insert(&mut self, candidate: Candidate) {
        if self.candidates.len() < self.limit {
            self.candidates.push(candidate);
        } else if let Some(mut worst) = self.candidates.peek_mut()
            && candidate < *worst
        {
            *worst = candidate;
        }
    }

    pub fn into_row_ids(self) -> Vec<u64> {
        self.candidates
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| candidate.row_id)
            .collect()
    }

    pub fn into_batch(self) -> Result<RecordBatch> {
        let (distances, row_ids): (Vec<_>, Vec<_>) = self
            .candidates
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| (candidate.distance, candidate.row_id))
            .unzip();
        Ok(RecordBatch::try_new(
            VECTOR_RESULT_SCHEMA.clone(),
            vec![
                Arc::new(Float32Array::from(distances)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )?)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty(0)]
    #[case::one(1)]
    #[case::short(7)]
    #[case::all(1000)]
    fn bounded_selection_matches_full_sort(#[case] limit: usize) {
        let distances = [
            f32::NAN,
            f32::INFINITY,
            -0.0,
            0.0,
            1.0,
            -1.0,
            f32::NEG_INFINITY,
        ];
        let mut reference = (0..512)
            .rev()
            .map(|row_id| Candidate {
                distance: distances[(row_id * 13 % distances.len() as u64) as usize],
                row_id,
            })
            .collect::<Vec<_>>();
        let mut shortlist = Shortlist::new(limit);
        for candidate in &reference {
            shortlist.insert(*candidate);
            assert!(shortlist.candidates.len() <= limit);
        }
        reference.sort();
        reference.truncate(limit);
        assert_eq!(shortlist.candidates.into_sorted_vec(), reference);
    }
}
