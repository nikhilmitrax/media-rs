//! Index reuse guarded by source version, track, and interpretation settings.

use media_core::error::{Error, Result};

use crate::source::{Source, SourceFingerprint};

/// Complete interpretation key required to reuse one index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexKey {
    /// Source identity and observed version at index creation.
    pub source: SourceFingerprint,
    /// Logical track identity, independent of default-track selection.
    pub track: String,
    /// Versioned settings describing time mapping and interpretation.
    pub interpretation: String,
}

impl IndexKey {
    /// Observe a source's current version and form a cache key.
    pub fn new(
        source: &Source,
        track: impl Into<String>,
        interpretation: impl Into<String>,
    ) -> Result<Self> {
        let source = source.fingerprint()?;
        if !source.is_versioned() {
            return Err(Error::unsupported(
                "index reuse requires a stable source version",
            ));
        }
        let track = track.into();
        let interpretation = interpretation.into();
        if track.is_empty() || interpretation.is_empty() {
            return Err(Error::invalid(
                "index track and interpretation identities must not be empty",
            ));
        }
        Ok(Self {
            source,
            track,
            interpretation,
        })
    }
}

/// Typed or opaque index entries associated with their original source version.
///
/// A stale lookup returns `None`. Rebuilding follows an explicit caller policy;
/// this container never scans or rebuilds implicitly.
#[derive(Clone, Debug)]
pub struct VersionedIndex<T> {
    key: IndexKey,
    entries: T,
}

impl<T> VersionedIndex<T> {
    /// Associate index entries with the current source and settings.
    pub fn new(
        source: &Source,
        track: impl Into<String>,
        interpretation: impl Into<String>,
        entries: T,
    ) -> Result<Self> {
        Ok(Self {
            key: IndexKey::new(source, track, interpretation)?,
            entries,
        })
    }

    /// Inspect the original cache key without reading the source again.
    pub fn key(&self) -> &IndexKey {
        &self.key
    }

    /// Return entries only if current source and interpretation still match.
    pub fn get(&self, source: &Source, track: &str, interpretation: &str) -> Result<Option<&T>> {
        let current = IndexKey::new(source, track, interpretation)?;
        Ok((self.key == current).then_some(&self.entries))
    }

    /// Consume the container without claiming its key remains current.
    pub fn into_inner(self) -> T {
        self.entries
    }
}
