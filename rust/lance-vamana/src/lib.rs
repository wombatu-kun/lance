// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A disk-resident Vamana vector index for Lance datasets.
//!
//! This crate builds and reads its own index files through Lance's published
//! `lance-file` / `lance-io` crates and commits them with the public index
//! segment API. It deliberately depends on nothing private: it lives inside the
//! Lance fork's tree for convenience, but Cargo grants workspace members no
//! extra visibility and this crate is not a member anyway, so the boundary it
//! compiles against is the same one an out-of-tree crate sees.

pub mod build;
pub mod builder;
pub mod consolidate;
pub mod consolidator;
pub mod format;
pub mod insert;
pub mod inserter;
pub mod io;
pub mod merge;
pub mod merger;
pub mod partition;
pub mod query;
pub mod search;
pub mod segment;

pub use builder::{BuildStats, IndexParams, create_index};
pub use consolidate::{Consolidated, consolidate_partition};
pub use consolidator::{ConsolidateStats, consolidate_index};
pub use format::{IndexMetadata, RowIdMode};
pub use insert::{Inserted, insert_into_partition};
pub use inserter::{InsertStats, insert_as_segment, insert_in_place};
pub use merge::{Merged, Newcomers, merge_partition};
pub use merger::{MergeStats, merge_index};
pub use partition::{Partition, PartitionGraph};
pub use query::{Neighbor, QueryResult, SearchParams, VamanaIndex};
pub use segment::{PartitionEntry, SegmentManifest};
