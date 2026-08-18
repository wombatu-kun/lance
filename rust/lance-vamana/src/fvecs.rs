// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Readers for the two file formats the ANN benchmark datasets ship in. Part of
//! the `vamana` binary, not of the library: `src/lib.rs` does not declare it.
//!
//! `.fvecs` and `.ivecs` are one layout over two element types: a record is a
//! little-endian `i32` width followed by that many values, repeated with no
//! header and no count of its own.
//!
//! The examples have a reader of their own and it panics on anything
//! unexpected, which suits a benchmark that owns its inputs. This one is handed
//! files by someone else, and streams because GIST1M is 3.8 GB.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use lance_core::{Error, Result};

/// A `.fvecs` file opened for reading, a batch of rows at a time.
#[derive(Debug)]
pub struct Fvecs {
    reader: BufReader<File>,
    dim: usize,
    rows: usize,
    read: usize,
}

impl Fvecs {
    /// Shape comes from the first record's width and the file's length: the
    /// format records no count, so a length that does not divide is refused.
    pub fn open(path: &Path) -> Result<Self> {
        let (reader, dim, rows) = open_records(path, size_of::<f32>())?;
        Ok(Self {
            reader,
            dim,
            rows,
            read: 0,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Records this reader will hand out, after any [`Self::take`].
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Stop after `rows` records. Above what the file holds is a ceiling, not
    /// an error.
    pub fn take(mut self, rows: usize) -> Self {
        self.rows = self.rows.min(rows);
        self
    }

    /// The next `rows` records as row-major values, or `None` at the end. Short
    /// only where the file ran out, never where the read did.
    pub fn next_batch(&mut self, rows: usize) -> Result<Option<Vec<f32>>> {
        let take = rows.min(self.rows - self.read);
        if take == 0 {
            return Ok(None);
        }
        let record = record_bytes(self.dim, size_of::<f32>());
        let mut raw = vec![0u8; take * record];
        self.reader
            .read_exact(&mut raw)
            .map_err(|e| Error::io(format!("reading {take} records of {record} bytes: {e}")))?;

        let mut values = Vec::with_capacity(take * self.dim);
        for row in 0..take {
            let start = row * record;
            check_width(&raw[start..start + 4], self.dim, self.read + row)?;
            for element in 0..self.dim {
                let at = start + 4 + element * size_of::<f32>();
                values.push(f32::from_le_bytes(
                    raw[at..at + 4].try_into().expect("a four-byte window"),
                ));
            }
        }
        self.read += take;
        Ok(Some(values))
    }

    /// Every remaining record as its own row, for query files.
    pub fn rest(&mut self) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(self.rows - self.read);
        while let Some(values) = self.next_batch(1024)? {
            rows.extend(values.chunks_exact(self.dim).map(<[f32]>::to_vec));
        }
        Ok(rows)
    }
}

/// Every record of an `.ivecs` file, one row of neighbour ids per query. Read
/// whole because a ground truth is megabytes where the base file is gigabytes.
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<u32>>> {
    let (mut reader, dim, rows) = open_records(path, size_of::<u32>())?;
    let record = record_bytes(dim, size_of::<u32>());
    let mut raw = vec![0u8; rows * record];
    reader
        .read_exact(&mut raw)
        .map_err(|e| Error::io(format!("reading {}: {e}", path.display())))?;

    (0..rows)
        .map(|row| {
            let start = row * record;
            check_width(&raw[start..start + 4], dim, row)?;
            Ok((0..dim)
                .map(|element| {
                    let at = start + 4 + element * size_of::<u32>();
                    u32::from_le_bytes(raw[at..at + 4].try_into().expect("a four-byte window"))
                })
                .collect())
        })
        .collect()
}

/// The width of a record and the number of them.
fn open_records(path: &Path, element: usize) -> Result<(BufReader<File>, usize, usize)> {
    let file = File::open(path)
        .map_err(|e| Error::invalid_input(format!("opening {}: {e}", path.display())))?;
    let length = file
        .metadata()
        .map_err(|e| Error::io(format!("sizing {}: {e}", path.display())))?
        .len() as usize;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 4];
    reader.read_exact(&mut header).map_err(|e| {
        Error::invalid_input(format!(
            "{} is too short to hold a record header: {e}",
            path.display()
        ))
    })?;
    let dim = i32::from_le_bytes(header);
    if dim <= 0 {
        return Err(Error::invalid_input(format!(
            "{} declares a width of {dim}, which is not a vector file",
            path.display()
        )));
    }
    let dim = dim as usize;

    let record = record_bytes(dim, element);
    if !length.is_multiple_of(record) {
        return Err(Error::invalid_input(format!(
            "{} is {length} bytes, which is not a whole number of {record}-byte records at width \
             {dim} - the file is truncated or is not this format",
            path.display()
        )));
    }

    // Reopened rather than rewound: the header belongs to the first record,
    // which the caller has not been handed yet.
    let file = File::open(path)
        .map_err(|e| Error::invalid_input(format!("reopening {}: {e}", path.display())))?;
    Ok((BufReader::new(file), dim, length / record))
}

fn record_bytes(dim: usize, element: usize) -> usize {
    4 + dim * element
}

/// Checked per record rather than once: a file of mixed widths parses into
/// plausible nonsense otherwise, since nothing else in the layout would object.
fn check_width(header: &[u8], dim: usize, row: usize) -> Result<()> {
    let width = i32::from_le_bytes(header.try_into().expect("a four-byte window"));
    if width as usize != dim {
        return Err(Error::invalid_input(format!(
            "record {row} declares width {width}, but the file opened at width {dim}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;

    use tempfile::TempDir;

    /// `rows` records of `dim` values each, values counting up across the file.
    fn write_fvecs(dir: &TempDir, name: &str, dim: usize, rows: usize) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut file = File::create(&path).unwrap();
        for row in 0..rows {
            file.write_all(&(dim as i32).to_le_bytes()).unwrap();
            for element in 0..dim {
                file.write_all(&((row * dim + element) as f32).to_le_bytes())
                    .unwrap();
            }
        }
        path
    }

    #[test]
    fn a_file_reads_back_the_values_it_was_written_with() {
        let dir = TempDir::new().unwrap();
        let path = write_fvecs(&dir, "base.fvecs", 3, 5);
        let mut file = Fvecs::open(&path).unwrap();
        assert_eq!((file.dim(), file.rows()), (3, 5));

        let rows = file.rest().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(rows[4], vec![12.0, 13.0, 14.0]);
    }

    #[test]
    fn batches_are_short_only_at_the_end() {
        let dir = TempDir::new().unwrap();
        let path = write_fvecs(&dir, "base.fvecs", 2, 5);
        let mut file = Fvecs::open(&path).unwrap();

        assert_eq!(file.next_batch(2).unwrap().unwrap().len(), 4);
        assert_eq!(file.next_batch(2).unwrap().unwrap().len(), 4);
        assert_eq!(file.next_batch(2).unwrap().unwrap().len(), 2);
        assert!(file.next_batch(2).unwrap().is_none());
    }

    #[test]
    fn a_row_limit_stops_before_the_end_of_the_file() {
        let dir = TempDir::new().unwrap();
        let path = write_fvecs(&dir, "base.fvecs", 2, 100);

        assert_eq!(Fvecs::open(&path).unwrap().take(7).rest().unwrap().len(), 7);
        assert_eq!(
            Fvecs::open(&path).unwrap().take(1000).rest().unwrap().len(),
            100
        );
    }

    #[test]
    fn a_length_that_is_not_a_whole_number_of_records_is_refused() {
        let dir = TempDir::new().unwrap();
        let path = write_fvecs(&dir, "base.fvecs", 4, 3);
        let mut truncated = std::fs::read(&path).unwrap();
        truncated.truncate(truncated.len() - 4);
        std::fs::write(&path, truncated).unwrap();

        let error = Fvecs::open(&path).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("whole number"),
            "the error must say what is wrong with the file: {error}"
        );
    }

    #[test]
    fn a_width_no_file_could_have_is_refused() {
        let dir = TempDir::new().unwrap();
        for width in [0i32, -1] {
            let path = dir.path().join(format!("{width}.fvecs"));
            std::fs::write(&path, width.to_le_bytes()).unwrap();

            let error = Fvecs::open(&path).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains(&format!("width of {width}")),
                "{error}"
            );
        }
    }

    #[test]
    fn a_record_that_changes_width_halfway_is_refused() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mixed.fvecs");
        let mut file = File::create(&path).unwrap();
        // The length still divides after the edit below, so only the
        // per-record check can catch it.
        for _ in 0..3 {
            file.write_all(&2i32.to_le_bytes()).unwrap();
            file.write_all(&[0u8; 8]).unwrap();
        }
        drop(file);
        let mut raw = std::fs::read(&path).unwrap();
        raw[24..28].copy_from_slice(&3i32.to_le_bytes());
        std::fs::write(&path, raw).unwrap();

        // Two at a time, so the bad record is the first of its batch and the
        // number reported has to be the file's rather than the batch's.
        let mut reader = Fvecs::open(&path).unwrap();
        assert_eq!(reader.next_batch(2).unwrap().unwrap().len(), 4);
        let error = reader.next_batch(2).unwrap_err();
        assert!(error.to_string().contains("record 2"), "{error}");
    }

    #[test]
    fn a_ground_truth_that_changes_width_halfway_is_refused() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mixed.ivecs");
        let mut file = File::create(&path).unwrap();
        for _ in 0..3 {
            file.write_all(&2i32.to_le_bytes()).unwrap();
            file.write_all(&[0u8; 8]).unwrap();
        }
        drop(file);
        let mut raw = std::fs::read(&path).unwrap();
        raw[24..28].copy_from_slice(&3i32.to_le_bytes());
        std::fs::write(&path, raw).unwrap();

        let error = read_ivecs(&path).unwrap_err();
        assert!(error.to_string().contains("record 2"), "{error}");
    }

    #[test]
    fn ground_truth_reads_at_its_own_width() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("truth.ivecs");
        let mut file = File::create(&path).unwrap();
        for row in 0..4u32 {
            file.write_all(&5i32.to_le_bytes()).unwrap();
            for element in 0..5u32 {
                file.write_all(&(row * 5 + element).to_le_bytes()).unwrap();
            }
        }
        drop(file);

        let truth = read_ivecs(&path).unwrap();
        assert_eq!(truth.len(), 4);
        assert_eq!(truth[3], vec![15, 16, 17, 18, 19]);
    }
}
