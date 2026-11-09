//! Independently opened byte sources and explicit access capabilities.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use crate::range::{RangeOptions, RangeSource, RangeState, RangeStatistics};
use media_core::error::{Error, Result};

/// A byte reader with random navigation.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

/// Actual source properties, independent of container access support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceCapabilities {
    /// Bytes may be read sequentially.
    pub sequential: bool,
    /// Each opened reader permits seeking.
    pub seekable: bool,
    /// Independent ranges do not change another cursor.
    pub range_readable: bool,
    /// Established byte length, if known.
    pub known_length: Option<u64>,
    /// A source version is available for checking cached state.
    pub stable_version: bool,
    /// Capture-clock information is supplied by this adapter.
    pub live_clock: bool,
}

/// Source identity. In-memory identity compares the full immutable content.
#[derive(Clone, Eq, PartialEq)]
pub enum SourceIdentity {
    /// Canonical local path.
    File(PathBuf),
    /// Immutable content retained by a shared lease.
    Memory(Arc<[u8]>),
    /// Caller-defined identity, such as an object URL.
    Custom(String),
    /// Private adapter identity, compared exactly but never exposed in Debug.
    Private(Arc<[u8]>),
}

impl fmt::Debug for SourceIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
            Self::Memory(bytes) => f
                .debug_struct("Memory")
                .field("length", &bytes.len())
                .finish(),
            Self::Custom(identity) => f.debug_tuple("Custom").field(identity).finish(),
            Self::Private(_) => f.write_str("Private(redacted)"),
        }
    }
}

/// Freshness evidence. Local metadata is not a cryptographic content hash.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceVersion {
    /// Local file metadata.
    File {
        /// Observed length.
        length: u64,
        /// Modification timestamp if supported.
        modified: Option<SystemTime>,
    },
    /// Immutable memory whose complete content is retained in the identity.
    Immutable,
    /// Caller-supplied object version, for example an ETag.
    Custom(String),
    /// Conditional range object, with both immutable version and length.
    Range {
        /// Established object length.
        length: u64,
        /// Immutable object version.
        version: String,
    },
    /// No reliable version is available.
    Unversioned,
}

/// Identity and observed version of one byte object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceFingerprint {
    /// Object identity.
    pub identity: SourceIdentity,
    /// Version at observation time.
    pub version: SourceVersion,
}

impl SourceFingerprint {
    /// Whether version-checked cache reuse is permitted.
    pub fn is_versioned(&self) -> bool {
        !matches!(
            self.version,
            SourceVersion::Unversioned | SourceVersion::File { modified: None, .. }
        )
    }
}

type ReaderFactory = dyn Fn() -> io::Result<Box<dyn Read + Send>> + Send + Sync;
type SeekableFactory = dyn Fn() -> io::Result<Box<dyn ReadSeek + Send>> + Send + Sync;

enum SourceKind {
    Range {
        identity: SourceIdentity,
        label: String,
        state: Arc<RangeState>,
    },
    File(PathBuf),
    Memory(Arc<[u8]>),
    Reader {
        identity: String,
        version: Option<String>,
        length: Option<u64>,
        factory: Arc<ReaderFactory>,
    },
    Seekable {
        identity: String,
        version: Option<String>,
        length: Option<u64>,
        factory: Arc<SeekableFactory>,
    },
}

/// A cheap, cloneable descriptor with independently opened read cursors.
///
/// Factories must return a new cursor each time. A custom version must identify
/// exactly those bytes; construct a new descriptor when it changes. Files refresh
/// metadata for each fingerprint. Concurrent mutation still requires application
/// coordination: metadata cannot detect edits preserving length and timestamps.
#[derive(Clone)]
pub struct Source {
    kind: Arc<SourceKind>,
}

impl fmt::Debug for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Source")
            .field("label", &self.label())
            .finish()
    }
}

impl Source {
    /// Adapt a concurrent conditional-range provider. Metadata is refreshed for
    /// fingerprints and every cursor open; each cursor then pins that version.
    pub fn from_range_source(
        identity: impl Into<String>,
        provider: Arc<dyn RangeSource>,
        options: RangeOptions,
    ) -> Result<Self> {
        let identity = validate_identity(identity.into(), None)?;
        Self::range_source(
            SourceIdentity::Custom(identity.clone()),
            identity,
            provider,
            options,
        )
    }
    pub(crate) fn range_source(
        identity: SourceIdentity,
        label: String,
        provider: Arc<dyn RangeSource>,
        options: RangeOptions,
    ) -> Result<Self> {
        Ok(Self {
            kind: Arc::new(SourceKind::Range {
                identity,
                label,
                state: RangeState::new(provider, options)?,
            }),
        })
    }
    /// Shared range transport counters, when this is a range-backed source.
    pub fn range_statistics(&self) -> Option<RangeStatistics> {
        match self.kind.as_ref() {
            SourceKind::Range { state, .. } => Some(state.statistics()),
            _ => None,
        }
    }
    /// Describe an existing regular file using its canonical path.
    pub fn file(path: impl AsRef<Path>) -> Result<Self> {
        let path = fs::canonicalize(path).map_err(Error::io)?;
        if !fs::metadata(&path).map_err(Error::io)?.is_file() {
            return Err(Error::invalid("a file source must be a regular file"));
        }
        Ok(Self {
            kind: Arc::new(SourceKind::File(path)),
        })
    }

    /// Retain immutable bytes. Each reader starts at offset zero.
    pub fn memory(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            kind: Arc::new(SourceKind::Memory(bytes.into())),
        }
    }

    /// Adapt independently opened sequential readers without claiming seeking.
    pub fn from_reader_factory<F, R>(
        identity: impl Into<String>,
        version: Option<String>,
        known_length: Option<u64>,
        factory: F,
    ) -> Result<Self>
    where
        F: Fn() -> io::Result<R> + Send + Sync + 'static,
        R: Read + Send + 'static,
    {
        let identity = validate_identity(identity.into(), version.as_deref())?;
        Ok(Self {
            kind: Arc::new(SourceKind::Reader {
                identity,
                version,
                length: known_length,
                factory: Arc::new(move || factory().map(|r| Box::new(r) as _)),
            }),
        })
    }

    /// Adapt independently opened seekable readers.
    pub fn from_seekable_factory<F, R>(
        identity: impl Into<String>,
        version: Option<String>,
        known_length: Option<u64>,
        factory: F,
    ) -> Result<Self>
    where
        F: Fn() -> io::Result<R> + Send + Sync + 'static,
        R: Read + Seek + Send + 'static,
    {
        let identity = validate_identity(identity.into(), version.as_deref())?;
        Ok(Self {
            kind: Arc::new(SourceKind::Seekable {
                identity,
                version,
                length: known_length,
                factory: Arc::new(move || factory().map(|r| Box::new(r) as _)),
            }),
        })
    }

    /// Canonical path for a file-backed source.
    pub fn path(&self) -> Option<&Path> {
        match self.kind.as_ref() {
            SourceKind::File(path) => Some(path),
            _ => None,
        }
    }

    /// Borrow immutable bytes without copying for a memory-backed source.
    pub fn memory_bytes(&self) -> Option<&[u8]> {
        match self.kind.as_ref() {
            SourceKind::Memory(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Human-readable label; memory contents are never included.
    pub fn label(&self) -> String {
        match self.kind.as_ref() {
            SourceKind::File(path) => path.display().to_string(),
            SourceKind::Memory(bytes) => format!("memory ({} bytes)", bytes.len()),
            SourceKind::Reader { identity, .. } | SourceKind::Seekable { identity, .. } => {
                identity.clone()
            }
            SourceKind::Range { label, .. } => label.clone(),
        }
    }

    /// Inspect current access properties and refresh file length.
    pub fn capabilities(&self) -> Result<SourceCapabilities> {
        let fingerprint = self.fingerprint()?;
        let (seekable, known_length) = match self.kind.as_ref() {
            SourceKind::File(_) => match &fingerprint.version {
                SourceVersion::File { length, .. } => (true, Some(*length)),
                _ => (true, None),
            },
            SourceKind::Memory(bytes) => (true, Some(bytes.len() as u64)),
            SourceKind::Reader { length, .. } => (false, *length),
            SourceKind::Seekable { length, .. } => (true, *length),
            SourceKind::Range { .. } => match &fingerprint.version {
                SourceVersion::Range { length, .. } => (true, Some(*length)),
                _ => (true, None),
            },
        };
        Ok(SourceCapabilities {
            sequential: true,
            seekable,
            range_readable: seekable,
            known_length,
            stable_version: fingerprint.is_versioned(),
            live_clock: false,
        })
    }

    /// Observe identity/version evidence, refreshing local file metadata.
    pub fn fingerprint(&self) -> Result<SourceFingerprint> {
        let (identity, version) = match self.kind.as_ref() {
            SourceKind::Range {
                identity, state, ..
            } => {
                let metadata = state.metadata()?;
                (
                    identity.clone(),
                    SourceVersion::Range {
                        length: metadata.length,
                        version: metadata.version,
                    },
                )
            }
            SourceKind::File(path) => {
                let metadata = fs::metadata(path).map_err(Error::io)?;
                if !metadata.is_file() {
                    return Err(Error::invalid("source is no longer a regular file"));
                }
                (
                    SourceIdentity::File(path.clone()),
                    SourceVersion::File {
                        length: metadata.len(),
                        modified: metadata.modified().ok(),
                    },
                )
            }
            SourceKind::Memory(bytes) => (
                SourceIdentity::Memory(bytes.clone()),
                SourceVersion::Immutable,
            ),
            SourceKind::Reader {
                identity, version, ..
            }
            | SourceKind::Seekable {
                identity, version, ..
            } => (
                SourceIdentity::Custom(identity.clone()),
                version
                    .as_ref()
                    .map(|v| SourceVersion::Custom(v.clone()))
                    .unwrap_or(SourceVersion::Unversioned),
            ),
        };
        Ok(SourceFingerprint { identity, version })
    }

    /// Open a fresh seekable cursor, rejecting sequential-only adapters.
    pub fn open(&self) -> Result<Box<dyn ReadSeek + Send>> {
        match self.kind.as_ref() {
            SourceKind::File(path) => Ok(Box::new(File::open(path).map_err(Error::io)?)),
            SourceKind::Memory(bytes) => Ok(Box::new(Cursor::new(bytes.clone()))),
            SourceKind::Seekable { factory, .. } => factory().map_err(Error::io),
            SourceKind::Range { state, .. } => Ok(Box::new(state.open()?)),
            SourceKind::Reader { .. } => Err(Error::unsupported(
                "source is sequential-only; random access requires explicit buffering or spooling",
            )),
        }
    }

    /// Open a fresh sequential cursor without requiring seek support.
    pub fn open_reader(&self) -> Result<Box<dyn Read + Send>> {
        match self.kind.as_ref() {
            SourceKind::Reader { factory, .. } => factory().map_err(Error::io),
            _ => Ok(Box::new(SequentialView(self.open()?))),
        }
    }

    /// Read exactly one range using an independent cursor; never pad short input.
    pub fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(u64::try_from(length).map_err(|_| Error::overflow("range length"))?)
            .ok_or_else(|| Error::overflow("byte range exceeds u64"))?;
        let capabilities = self.capabilities()?;
        if !capabilities.range_readable {
            return Err(Error::unsupported(
                "source does not support independent ranges",
            ));
        }
        if capabilities.known_length.is_some_and(|size| end > size) {
            return Err(Error::invalid("requested byte range exceeds source length"));
        }
        let mut reader = self.open()?;
        reader.seek(SeekFrom::Start(offset)).map_err(Error::io)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|error| Error::invalid(format!("cannot allocate byte range: {error}")))?;
        bytes.resize(length, 0);
        reader.read_exact(&mut bytes).map_err(Error::io)?;
        Ok(bytes)
    }
}

struct SequentialView(Box<dyn ReadSeek + Send>);
impl Read for SequentialView {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.0.read(bytes)
    }
}

fn validate_identity(identity: String, version: Option<&str>) -> Result<String> {
    if identity.is_empty() {
        return Err(Error::invalid("source identity must not be empty"));
    }
    if version == Some("") {
        return Err(Error::invalid(
            "source version must not be empty when supplied",
        ));
    }
    Ok(identity)
}
