//! Verified HTTP object ranges (optional `http` / HTTPS `http-tls` features).
//!
//! HEAD must establish a length and strong ASCII ETag. A conditional range probe
//! must then prove 206/Content-Range support before a Source is returned. Every
//! read sends If-Match and Accept-Encoding: identity; ignored ranges, changing
//! versions, transformed encodings, redirects, and short bodies fail explicitly.
//! There is no automatic full download, spool, or redirect. URLs and credential
//! header values never appear in this adapter's Debug/error output.
//! Application-enabled ureq TRACE logging is outside this diagnostic redaction
//! and may expose URL paths/queries; the library does not initialize a logger.
use crate::{
    range::{Counters, RangeMetadata, RangeOptions, RangeResponse, RangeSource, RangeStatistics},
    source::{Source, SourceIdentity},
};
use media_core::error::{Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Read,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

/// Synchronous HTTP timeouts, request bounds, and optional request credentials.
#[derive(Clone)]
pub struct HttpOptions {
    /// Range bounds and cooperative cancellation.
    pub range: RangeOptions,
    /// Total deadline for each exchange, including reading its response body.
    pub request_timeout: Duration,
    /// Connection/DNS/TLS establishment deadline, additionally bounded by total.
    pub connect_timeout: Duration,
    /// Per-transport-read deadline, additionally bounded by total.
    pub read_timeout: Duration,
    /// Caller-owned headers such as Authorization; reserved range/host/body
    /// headers cannot override this adapter's protocol constraints.
    pub headers: BTreeMap<String, String>,
}
impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            range: RangeOptions::default(),
            request_timeout: Duration::from_secs(15),
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(3),
            headers: BTreeMap::new(),
        }
    }
}
impl fmt::Debug for HttpOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpOptions")
            .field("range", &self.range)
            .field("request_timeout", &self.request_timeout)
            .field("connect_timeout", &self.connect_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}
/// A checked, version-conditional HTTP provider. Prefer `connect` to build Source.
pub struct HttpRangeSource {
    url: String,
    agent: ureq::Agent,
    options: HttpOptions,
    counters: Counters,
}
impl fmt::Debug for HttpRangeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRangeSource")
            .field("url", &"[redacted]")
            .field("options", &self.options)
            .finish()
    }
}
impl HttpRangeSource {
    /// Verify HEAD and conditional range support, then return independent cursors.
    ///
    /// Signed query strings and Authorization headers are supported. URL userinfo
    /// and fragments are rejected. HTTPS additionally requires `http-tls`.
    /// Cooperative cancellation is checked between requests and body reads;
    /// an in-flight network call is bounded by the configured transport deadline.
    pub fn connect(url: impl Into<String>, options: HttpOptions) -> Result<Source> {
        options.range.validate()?;
        if options.request_timeout.is_zero()
            || options.connect_timeout.is_zero()
            || options.read_timeout.is_zero()
        {
            return Err(Error::invalid("HTTP timeouts must be nonzero"));
        }
        let url = url.into();
        let uri = url
            .parse::<ureq::http::Uri>()
            .map_err(|_| Error::invalid("HTTP object URL is invalid (redacted)"))?;
        if url.contains('#')
            || uri.authority().is_none_or(|a| a.as_str().contains('@'))
            || !matches!(uri.scheme_str(), Some("http" | "https"))
        {
            return Err(Error::invalid(
                "HTTP object URL must be absolute http(s), without userinfo or fragments",
            ));
        }
        if uri.scheme_str() == Some("https") && !cfg!(feature = "http-tls") {
            return Err(Error::unsupported(
                "HTTPS range sources require the http-tls feature",
            ));
        }
        let mut header_names = BTreeSet::new();
        for (name, value) in &options.headers {
            ureq::http::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| Error::invalid("HTTP header name is invalid"))?;
            ureq::http::HeaderValue::from_str(value)
                .map_err(|_| Error::invalid("HTTP header value is invalid (redacted)"))?;
            if !header_names.insert(name.to_ascii_lowercase()) {
                return Err(Error::invalid(
                    "HTTP duplicate header names are not permitted",
                ));
            }
            if [
                "range",
                "if-match",
                "if-range",
                "accept-encoding",
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
            ]
            .contains(&name.to_ascii_lowercase().as_str())
            {
                return Err(Error::invalid(
                    "HTTP reserved range/host/body headers cannot be overridden",
                ));
            }
        }
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .max_response_header_size(32 * 1024)
            .proxy(None)
            .timeout_global(Some(options.request_timeout))
            .timeout_connect(Some(options.connect_timeout))
            .timeout_per_read(Some(options.read_timeout))
            .build();
        let provider = Arc::new(Self {
            url: url.clone(),
            agent: config.into(),
            options: options.clone(),
            counters: Counters::default(),
        });
        let initial = provider.metadata()?;
        if initial.length == 0 {
            provider.verify_empty(&initial)?;
        } else {
            provider.read_range(&initial, 0, 1)?;
        }
        // Representation identity includes all caller headers (including auth),
        // not only a URL whose negotiated representations may have equal ETags.
        // Bytes compare exactly; their Debug projection is always redacted.
        let mut identity = Vec::new();
        for value in std::iter::once(url.as_str()).chain(
            options
                .headers
                .iter()
                .flat_map(|(name, value)| [name.as_str(), value.as_str()]),
        ) {
            identity.extend_from_slice(&(value.len() as u64).to_le_bytes());
            identity.extend_from_slice(value.as_bytes());
        }
        Source::range_source(
            SourceIdentity::Private(Arc::from(identity)),
            "HTTP range object (URL redacted)".into(),
            provider,
            options.range,
        )
    }
    fn request(
        &self,
        head: bool,
        expected: Option<&RangeMetadata>,
        range: Option<(u64, u64)>,
    ) -> Result<ureq::http::Response<ureq::Body>> {
        self.options.range.check_cancelled()?;
        let mut request = if head {
            self.agent.head(&self.url)
        } else {
            self.agent.get(&self.url)
        };
        request = request.header("Accept-Encoding", "identity");
        for (name, value) in &self.options.headers {
            request = request.header(name, value);
        }
        if let Some(expected) = expected {
            request = request.header("If-Match", &expected.version);
        }
        if let Some((start, end)) = range {
            request = request.header("Range", format!("bytes={start}-{end}"));
        }
        let response = request.call().map_err(|_| {
            Error::backend(
                "http",
                "HTTP range transport failed (URL and credentials redacted)",
            )
        })?;
        self.options.range.check_cancelled()?;
        if response.status().as_u16() == 412 {
            return Err(Error::incompatible(
                "HTTP object no longer matches the cursor's pinned version",
            )
            .with_constraint("source_version"));
        }
        if response.status().is_redirection() {
            return Err(Error::unsupported(
                "HTTP range redirects are disabled; provide the final object URL explicitly",
            ));
        }
        Ok(response)
    }
    fn verify_empty(&self, expected: &RangeMetadata) -> Result<()> {
        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        let response = self.request(false, Some(expected), Some((0, 0)))?;
        if response.status().as_u16() != 416
            || content_range(header(response.headers(), "content-range")?)? != (None, 0)
            || strong_etag(response.headers())? != expected.version
        {
            return Err(Error::unsupported(
                "empty HTTP object did not verify a version-conditional unsatisfied range",
            ));
        }
        Ok(())
    }
}
impl RangeSource for HttpRangeSource {
    fn metadata(&self) -> Result<RangeMetadata> {
        self.counters.metadata.fetch_add(1, Ordering::Relaxed);
        let response = self.request(true, None, None)?;
        if response.status().as_u16() != 200 {
            return Err(Error::unsupported(format!(
                "HTTP HEAD must return 200, received {}",
                response.status().as_u16()
            )));
        }
        check_encoding(response.headers())?;
        let length = decimal(header(response.headers(), "content-length")?)?;
        Ok(RangeMetadata {
            length,
            version: strong_etag(response.headers())?,
        })
    }
    fn read_range(
        &self,
        expected: &RangeMetadata,
        offset: u64,
        length: usize,
    ) -> Result<RangeResponse> {
        expected.validate()?;
        self.options.range.check_cancelled()?;
        if length == 0 || length > self.options.range.max_request_bytes {
            return Err(Error::invalid(
                "HTTP range length is zero or exceeds the per-request bound",
            ));
        }
        let exclusive = offset
            .checked_add(
                u64::try_from(length).map_err(|_| Error::overflow("HTTP range length overflow"))?,
            )
            .ok_or_else(|| Error::overflow("HTTP range end overflow"))?;
        if exclusive > expected.length {
            return Err(Error::invalid("HTTP range exceeds pinned object length"));
        }
        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        let mut response = self.request(false, Some(expected), Some((offset, exclusive - 1)))?;
        if response.status().as_u16() != 206 {
            return Err(Error::unsupported(format!(
                "HTTP object ignored/rejected byte range (status {}); full-body fallback is disabled",
                response.status().as_u16()
            )));
        }
        check_encoding(response.headers())?;
        let version = strong_etag(response.headers())?;
        if version != expected.version {
            return Err(
                Error::incompatible("HTTP range response changed the pinned strong ETag")
                    .with_constraint("source_version"),
            );
        }
        if content_range(header(response.headers(), "content-range")?)?
            != (Some((offset, exclusive - 1)), expected.length)
        {
            return Err(Error::incompatible(
                "HTTP Content-Range does not match the exact requested offset/length/total",
            ));
        }
        if response.headers().contains_key("content-length")
            && decimal(header(response.headers(), "content-length")?)? != length as u64
        {
            return Err(Error::incompatible(
                "HTTP Content-Length contradicts the requested range length",
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Error::exhausted("cannot allocate bounded HTTP response"))?;
        let mut reader = response.body_mut().as_reader();
        let mut buffer = [0u8; 8192];
        loop {
            self.options.range.check_cancelled()?;
            let allowed = buffer
                .len()
                .min(length.saturating_sub(bytes.len()).saturating_add(1));
            let read = reader.read(&mut buffer[..allowed]).map_err(|_| {
                Error::backend(
                    "http",
                    "HTTP range body ended early or its bounded transport read failed",
                )
            })?;
            self.counters
                .bytes
                .fetch_add(read as u64, Ordering::Relaxed);
            if read == 0 {
                break;
            }
            if bytes.len().checked_add(read).is_none_or(|v| v > length) {
                return Err(Error::incompatible(
                    "HTTP range body exceeds its declared/requested range",
                ));
            }
            bytes.extend_from_slice(&buffer[..read]);
        }
        self.options.range.check_cancelled()?;
        if bytes.len() != length {
            return Err(Error::backend(
                "http",
                "HTTP range body is shorter than requested; no padding is permitted",
            ));
        }
        Ok(RangeResponse {
            metadata: RangeMetadata {
                length: expected.length,
                version,
            },
            bytes,
        })
    }
    fn statistics(&self) -> Option<RangeStatistics> {
        Some(self.counters.snapshot())
    }
}
fn header<'a>(headers: &'a ureq::http::HeaderMap, name: &str) -> Result<&'a str> {
    if headers.get_all(name).iter().count() != 1 {
        return Err(Error::unsupported(format!(
            "HTTP response requires exactly one {name} header"
        )));
    }
    headers
        .get(name)
        .ok_or_else(|| Error::unsupported(format!("HTTP response lacks {name}")))?
        .to_str()
        .map_err(|_| Error::unsupported(format!("HTTP {name} must be ASCII")))
}
fn strong_etag(headers: &ureq::http::HeaderMap) -> Result<String> {
    let value = header(headers, "etag")?;
    let bytes = value.as_bytes();
    if bytes.len() < 2
        || bytes[0] != b'"'
        || bytes[bytes.len() - 1] != b'"'
        || !bytes[1..bytes.len() - 1]
            .iter()
            .all(|v| *v == 0x21 || (0x23..=0x7e).contains(v))
    {
        return Err(Error::unsupported(
            "HTTP range source requires a strong ASCII ETag; weak/missing validators cannot pin immutable reads",
        ));
    }
    Ok(value.to_owned())
}
fn check_encoding(headers: &ureq::http::HeaderMap) -> Result<()> {
    if headers.contains_key("content-encoding")
        && !header(headers, "content-encoding")?.eq_ignore_ascii_case("identity")
    {
        return Err(Error::unsupported(
            "HTTP range source requires identity content encoding",
        ));
    }
    Ok(())
}

fn decimal(value: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|v| v.is_ascii_digit()) {
        return Err(Error::incompatible(
            "HTTP range length/offset is not an unsigned decimal",
        ));
    }
    value
        .parse()
        .map_err(|_| Error::overflow("HTTP range length/offset exceeds u64"))
}
fn content_range(value: &str) -> Result<(Option<(u64, u64)>, u64)> {
    let (unit, value) = value
        .split_once(' ')
        .ok_or_else(|| Error::incompatible("invalid HTTP Content-Range unit"))?;
    if !unit.eq_ignore_ascii_case("bytes") {
        return Err(Error::unsupported("HTTP Content-Range requires byte units"));
    }
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| Error::incompatible("invalid HTTP Content-Range total"))?;
    let total = decimal(total)?;
    if range == "*" {
        return Ok((None, total));
    }
    let (start, end) = range
        .split_once('-')
        .ok_or_else(|| Error::incompatible("invalid HTTP Content-Range interval"))?;
    let (start, end) = (decimal(start)?, decimal(end)?);
    if start > end || end >= total {
        return Err(Error::incompatible(
            "HTTP Content-Range interval is outside its object total",
        ));
    }
    Ok((Some((start, end)), total))
}
