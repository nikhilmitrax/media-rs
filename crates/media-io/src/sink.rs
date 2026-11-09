//! Writable byte destinations with explicit completion and partial-output state.

use std::fs::File;
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use media_core::error::{Error, Result};

/// Byte-level destination capabilities. Container compatibility is separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SinkCapabilities {
    /// Bytes can be written sequentially.
    pub sequential: bool,
    /// Existing output can be revisited to finalize headers.
    pub seekable: bool,
}

enum SinkKind {
    File { path: PathBuf, file: File },
    Memory(Cursor<Vec<u8>>),
}

/// A writable destination that does not imply successful completion on drop.
///
/// The codec/muxer above it must drain and finalize before [`Sink::finish`].
/// Finishing flushes bytes but does not promise fsync durability. Dropping or
/// aborting a file sink leaves its partial file in place.
pub struct Sink {
    kind: SinkKind,
    bytes_written: u64,
    failed: bool,
}

impl Sink {
    /// Wrap an already opened, owned file without opening or truncating its path.
    /// The caller supplies the path used in completion/partial-output reports;
    /// it must identify the same file handle. No path validation or extra I/O occurs.
    pub fn from_file(file: File, path: impl Into<PathBuf>) -> Self {
        Self {
            kind: SinkKind::File {
                path: path.into(),
                file,
            },
            bytes_written: 0,
            failed: false,
        }
    }
    /// Create or truncate a local output file.
    pub fn file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(Error::io)?;
        Ok(Self {
            kind: SinkKind::File { path, file },
            bytes_written: 0,
            failed: false,
        })
    }

    /// Create a seekable in-memory output buffer.
    pub fn memory() -> Self {
        Self {
            kind: SinkKind::Memory(Cursor::new(Vec::new())),
            bytes_written: 0,
            failed: false,
        }
    }

    /// Inspect operations available to a container writer.
    pub fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            sequential: true,
            seekable: true,
        }
    }

    /// Total bytes accepted, including overwritten bytes after seeks.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Whether an earlier write, flush, or seek failed.
    pub fn has_failed(&self) -> bool {
        self.failed
    }

    /// Flush output and return explicit completion evidence.
    ///
    /// A previous I/O error prevents success even if later writes succeeded.
    pub fn finish(mut self) -> Result<FinishedSink> {
        if self.failed {
            return Err(Error::invalid(
                "sink previously failed; output is partial and cannot be finalized successfully",
            ));
        }
        self.flush().map_err(Error::io)?;
        Ok(match self.kind {
            SinkKind::File { path, .. } => FinishedSink::File {
                path,
                bytes_written: self.bytes_written,
            },
            SinkKind::Memory(cursor) => FinishedSink::Memory(cursor.into_inner()),
        })
    }

    /// Abort without declaring success and describe the partial output.
    pub fn abort(self) -> PartialOutput {
        PartialOutput {
            path: match self.kind {
                SinkKind::File { path, .. } => Some(path),
                SinkKind::Memory(_) => None,
            },
            bytes_written: self.bytes_written,
            io_failed: self.failed,
        }
    }
}

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let result = match &mut self.kind {
            SinkKind::File { file, .. } => file.write(bytes),
            SinkKind::Memory(cursor) => cursor.write(bytes),
        };
        match result {
            Ok(count) => match self.bytes_written.checked_add(count as u64) {
                Some(total) => {
                    self.bytes_written = total;
                    Ok(count)
                }
                None => {
                    self.failed = true;
                    Err(io::Error::other("sink byte counter overflowed"))
                }
            },
            Err(error) => {
                // Write::write_all retries Interrupted without reporting failure.
                self.failed |= error.kind() != io::ErrorKind::Interrupted;
                Err(error)
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        let result = match &mut self.kind {
            SinkKind::File { file, .. } => file.flush(),
            SinkKind::Memory(cursor) => cursor.flush(),
        };
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

impl Seek for Sink {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let result = match &mut self.kind {
            SinkKind::File { file, .. } => file.seek(position),
            SinkKind::Memory(cursor) => cursor.seek(position),
        };
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

/// Evidence that byte output was explicitly flushed successfully.
#[derive(Debug)]
pub enum FinishedSink {
    /// A completed byte file, not necessarily fsynced.
    File {
        /// Destination path.
        path: PathBuf,
        /// Bytes accepted, including rewritten headers.
        bytes_written: u64,
    },
    /// Completed in-memory bytes.
    Memory(Vec<u8>),
}

impl FinishedSink {
    /// Recover completed in-memory bytes; reject file output.
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        match self {
            Self::Memory(bytes) => Ok(bytes),
            Self::File { .. } => Err(Error::unsupported(
                "file output has no in-memory byte buffer",
            )),
        }
    }
}

/// Information about an output deliberately left unfinished.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartialOutput {
    /// Retained partial file, if file-backed.
    pub path: Option<PathBuf>,
    /// Bytes accepted before abort, including overwritten bytes.
    pub bytes_written: u64,
    /// Whether I/O failed before abort.
    pub io_failed: bool,
}
