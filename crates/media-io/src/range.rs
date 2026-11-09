//! Concurrent version-conditional byte ranges with independent seek cursors.
//!
//! Providers must atomically enforce the expected object version when reading.
//! A metadata check followed by an unconditional object read is insufficient.
//! Each cursor pins its opening length/version, and every response is validated
//! before bytes reach the caller. No range cache or implicit spool is created.
use media_core::error::{Error, ErrorKind, Result};
use std::{
    io::{self, Read, Seek, SeekFrom},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

/// Fresh object metadata. A version identifies exactly one immutable byte value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeMetadata {
    /// Established object size.
    pub length: u64,
    /// Immutable object version (for HTTP, a strong ETag including its quotes).
    pub version: String,
}
impl RangeMetadata {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.version.is_empty() {
            return Err(Error::invalid(
                "range provider returned an empty object version",
            ));
        }
        Ok(())
    }
}
/// Exactly one conditional object-range response.
#[derive(Debug, Clone)]
pub struct RangeResponse {
    /// Actual version/length associated atomically with these bytes.
    pub metadata: RangeMetadata,
    /// Exactly the requested range; short/long responses are errors.
    pub bytes: Vec<u8>,
}
/// Payload counters. HTTP transport additionally counts failed/probe body reads.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RangeStatistics {
    /// Metadata refresh requests.
    pub metadata_requests: u64,
    /// Conditional range requests, including failed requests.
    pub range_requests: u64,
    /// Payload bytes consumed from response-body reads; excludes protocol headers,
    /// TLS bytes, and unread transport-prefetched data.
    /// Generic providers without transport counters count returned response bytes.
    pub payload_bytes_received: u64,
}
/// Object-store client boundary with dynamic metadata and immutable read versions.
pub trait RangeSource: Send + Sync {
    /// Obtain current length/version without reading the complete object.
    fn metadata(&self) -> Result<RangeMetadata>;
    /// Atomically require `expected.version` and `expected.length`, then return
    /// exactly `length` bytes beginning at `offset`. Do not silently retry a
    /// failed version precondition against a newer object.
    fn read_range(
        &self,
        expected: &RangeMetadata,
        offset: u64,
        length: usize,
    ) -> Result<RangeResponse>;
    /// Optional transport counters, including failed response bodies and probes.
    /// Providers without these counters use adapter-observed payload statistics.
    fn statistics(&self) -> Option<RangeStatistics> {
        None
    }
}
/// Shared cooperative cancellation, checked between requests (not a thread abort).
#[derive(Debug, Default, Clone)]
pub struct RangeCancellation(Arc<AtomicBool>);
impl RangeCancellation {
    /// Request cancellation for all cursors using this token.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(Error::new(
                ErrorKind::Cancelled,
                "range source operation cancelled",
            ));
        }
        Ok(())
    }
}
/// Per-source range request controls; there is no implicit read-ahead allocation.
#[derive(Debug, Clone)]
pub struct RangeOptions {
    /// Maximum bytes requested in any single object request, nonzero.
    pub max_request_bytes: usize,
    /// Shared cancellation checked before/after each request.
    pub cancellation: Option<RangeCancellation>,
}
impl Default for RangeOptions {
    fn default() -> Self {
        Self {
            max_request_bytes: 256 * 1024,
            cancellation: None,
        }
    }
}
impl RangeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_request_bytes == 0 {
            return Err(Error::invalid("range request limit must be nonzero"));
        }
        Ok(())
    }
    pub(crate) fn check_cancelled(&self) -> Result<()> {
        if let Some(token) = &self.cancellation {
            token.check()?;
        }
        Ok(())
    }
}
#[derive(Default)]
pub(crate) struct Counters {
    pub metadata: AtomicU64,
    pub requests: AtomicU64,
    pub bytes: AtomicU64,
}
impl Counters {
    pub fn snapshot(&self) -> RangeStatistics {
        RangeStatistics {
            metadata_requests: self.metadata.load(Ordering::Relaxed),
            range_requests: self.requests.load(Ordering::Relaxed),
            payload_bytes_received: self.bytes.load(Ordering::Relaxed),
        }
    }
}
pub(crate) struct RangeState {
    provider: Arc<dyn RangeSource>,
    options: RangeOptions,
    counters: Counters,
}
impl RangeState {
    pub fn new(provider: Arc<dyn RangeSource>, options: RangeOptions) -> Result<Arc<Self>> {
        options.validate()?;
        let state = Arc::new(Self {
            provider,
            options,
            counters: Counters::default(),
        });
        state.metadata()?;
        Ok(state)
    }
    pub fn metadata(&self) -> Result<RangeMetadata> {
        self.options.check_cancelled()?;
        self.counters.metadata.fetch_add(1, Ordering::Relaxed);
        let metadata = self.provider.metadata()?;
        metadata.validate()?;
        self.options.check_cancelled()?;
        Ok(metadata)
    }
    pub fn open(self: &Arc<Self>) -> Result<RangeCursor> {
        Ok(RangeCursor {
            source: Arc::clone(self),
            expected: self.metadata()?,
            position: 0,
        })
    }
    pub fn statistics(&self) -> RangeStatistics {
        self.provider
            .statistics()
            .unwrap_or_else(|| self.counters.snapshot())
    }
}
pub(crate) struct RangeCursor {
    source: Arc<RangeState>,
    expected: RangeMetadata,
    position: u64,
}
impl RangeCursor {
    fn read_into(&mut self, buffer: &mut [u8]) -> Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        self.source.options.check_cancelled()?;
        if self.position >= self.expected.length {
            return Ok(0);
        }
        let count = buffer
            .len()
            .min(self.source.options.max_request_bytes)
            .min(usize::try_from(self.expected.length - self.position).unwrap_or(usize::MAX));
        self.source
            .counters
            .requests
            .fetch_add(1, Ordering::Relaxed);
        let response = self
            .source
            .provider
            .read_range(&self.expected, self.position, count)?;
        self.source
            .counters
            .bytes
            .fetch_add(response.bytes.len() as u64, Ordering::Relaxed);
        self.source.options.check_cancelled()?;
        if response.metadata != self.expected {
            return Err(Error::incompatible(
                "range object changed from the cursor's pinned version/length",
            )
            .with_constraint("source_version"));
        }
        if response.bytes.len() != count {
            return Err(Error::io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "range provider did not return exactly the requested bytes",
            )));
        }
        buffer[..count].copy_from_slice(&response.bytes);
        self.position = self
            .position
            .checked_add(count as u64)
            .ok_or_else(|| Error::overflow("range cursor position overflow"))?;
        Ok(count)
    }
}
impl Read for RangeCursor {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.read_into(buffer).map_err(io::Error::other)
    }
}
impl Seek for RangeCursor {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.source
            .options
            .check_cancelled()
            .map_err(io::Error::other)?;
        let position = match position {
            SeekFrom::Start(v) => i128::from(v),
            SeekFrom::Current(v) => i128::from(self.position) + i128::from(v),
            SeekFrom::End(v) => i128::from(self.expected.length) + i128::from(v),
        };
        self.position = u64::try_from(position).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "range seek is outside unsigned object offsets",
            )
        })?;
        Ok(self.position)
    }
}
