// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared plumbing for the benchmark examples.
//!
//! A directory rather than a file, because Cargo compiles every *file* under
//! `examples/` as its own binary and a subdirectory is the one place a module
//! two examples both include can live. Each binary compiles it whole, so a
//! helper only one of them uses still lands in the other.
#![allow(dead_code)]

/// Read an `.fvecs` file: `(values, dimension, count)`, values row-major.
pub fn read_fvecs(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert_eq!(bytes.len() % record, 0);
    let count = bytes.len() / record;
    let mut values = Vec::with_capacity(count * dim);
    for row in 0..count {
        let start = row * record + 4;
        for i in 0..dim {
            let offset = start + i * 4;
            values.push(f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
        }
    }
    (values, dim, count)
}

/// Read an `.ivecs` file: one row of ids per query.
pub fn read_ivecs(path: &str) -> Vec<Vec<u32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert_eq!(bytes.len() % record, 0);
    (0..bytes.len() / record)
        .map(|row| {
            let start = row * record + 4;
            (0..dim)
                .map(|i| {
                    u32::from_le_bytes(bytes[start + i * 4..start + i * 4 + 4].try_into().unwrap())
                })
                .collect()
        })
        .collect()
}

pub fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|raw| {
            raw.parse()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(fallback)
}
