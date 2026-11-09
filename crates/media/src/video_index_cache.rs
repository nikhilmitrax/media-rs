//! Bounded local cache encoding. Checksums detect damage, not hostile forgery;
//! callers supply trusted cache files, and source metadata versions are checked.
use super::*;
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::UNIX_EPOCH,
};
const MAGIC: &[u8; 8] = b"MRVIDX02";
const MAX_KEY: usize = 64 * 1024;
const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
fn hash(mut state: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x100_0000_01b3);
    }
    state
}
struct Encoder<W> {
    io: W,
    hash: u64,
}
impl<W: Write> Encoder<W> {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.io.write_all(bytes).map_err(Error::io)?;
        self.hash = hash(self.hash, bytes);
        Ok(())
    }
    fn time(&mut self, time: Time) -> Result<()> {
        self.put(&time.numerator().to_le_bytes())?;
        self.put(&time.denominator().to_le_bytes())
    }
    fn optional(&mut self, time: Option<Time>) -> Result<()> {
        self.put(&[u8::from(time.is_some())])?;
        if let Some(time) = time {
            self.time(time)?;
        }
        Ok(())
    }
}
struct Decoder<R> {
    io: R,
    hash: u64,
}
impl<R: Read> Decoder<R> {
    fn get<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.io.read_exact(&mut bytes).map_err(Error::io)?;
        self.hash = hash(self.hash, &bytes);
        Ok(bytes)
    }
    fn time(&mut self) -> Result<Time> {
        Time::new(
            i64::from_le_bytes(self.get()?),
            u32::from_le_bytes(self.get()?),
        )
    }
    fn flag(&mut self) -> Result<bool> {
        match self.get::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::invalid("invalid video index flag")),
        }
    }
    fn optional(&mut self) -> Result<Option<Time>> {
        if !self.flag()? {
            return Ok(None);
        }
        let time = self.time()?;
        if time <= Time::ZERO {
            return Err(Error::invalid(
                "cached native/display duration must be positive",
            ));
        }
        Ok(Some(time))
    }
}
fn key(source: &SourceFingerprint) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut bytes = |tag: u8, identity: &[u8], version: &[u8]| -> Result<()> {
        let length = identity
            .len()
            .checked_add(version.len())
            .and_then(|n| n.checked_add(17))
            .ok_or_else(|| Error::overflow("index source key"))?;
        if length > MAX_KEY {
            return Err(Error::exhausted(
                "persisted video index source key exceeds its bound",
            ));
        }
        out.try_reserve_exact(length)
            .map_err(|_| Error::exhausted("index key allocation failed"))?;
        out.push(tag);
        out.extend_from_slice(&(identity.len() as u64).to_le_bytes());
        out.extend_from_slice(identity);
        out.extend_from_slice(&(version.len() as u64).to_le_bytes());
        out.extend_from_slice(version);
        Ok(())
    };
    match (&source.identity, &source.version) {
        (
            SourceIdentity::File(path),
            SourceVersion::File {
                length,
                modified: Some(modified),
            },
        ) => {
            let (negative, duration) = match modified.duration_since(UNIX_EPOCH) {
                Ok(duration) => (false, duration),
                Err(error) => (true, error.duration()),
            };
            let mut version = [0u8; 21];
            version[0..8].copy_from_slice(&length.to_le_bytes());
            version[8] = u8::from(negative);
            version[9..17].copy_from_slice(&duration.as_secs().to_le_bytes());
            version[17..21].copy_from_slice(&duration.subsec_nanos().to_le_bytes());
            bytes(1, path.as_os_str().as_encoded_bytes(), &version)?;
        }
        (SourceIdentity::Custom(identity), SourceVersion::Custom(version)) => {
            bytes(2, identity.as_bytes(), version.as_bytes())?
        }
        _ => {
            return Err(Error::unsupported(
                "persisted video indexes currently require a versioned local file or explicit public custom identity; private/memory identities stay in-process",
            ));
        }
    }
    Ok(out)
}
impl VideoIndex {
    /// Persist a trusted local cache without overwriting an existing file.
    /// Local path/mtime/length or public custom identity/version is stored exactly;
    /// private URL/header identities and memory bytes are never written. Checksums
    /// detect accidental corruption; this format does not authenticate a hostile
    /// cache. Failed writes may leave a partial cache which load rejects.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let _cpu = self.resources.reserve_cpu(0, 0)?;
        let scratch = source_charge(&self.source)?
            .checked_add(8192)
            .ok_or_else(|| Error::overflow("index persistence working bytes"))?;
        let _working = self.resources.host_budget().reserve(scratch)?;
        let key = key(&self.source)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(Error::io)?;
        let mut encoder = Encoder {
            io: BufWriter::with_capacity(4096, file),
            hash: BASIS,
        };
        encoder.put(MAGIC)?;
        encoder.put(&(key.len() as u64).to_le_bytes())?;
        encoder.put(&key)?;
        encoder.put(&self.track.0.to_le_bytes())?;
        encoder.put(&[u8::from(self.policy == FrameDurationPolicy::TrustReported)])?;
        encoder.time(self.origin)?;
        encoder.put(&(self.frames.len() as u64).to_le_bytes())?;
        for frame in &self.frames {
            encoder.time(
                frame
                    .timing
                    .source_pts
                    .ok_or_else(|| Error::incompatible("index source time absent"))?,
            )?;
            encoder.optional(frame.timing.duration)?;
            encoder.optional(frame.timing.native_duration)?;
            encoder.put(&[u8::from(frame.keyframe)])?;
            encoder.optional(frame.seek_timestamp)?;
        }
        encoder
            .io
            .write_all(&encoder.hash.to_le_bytes())
            .map_err(Error::io)?;
        encoder.io.flush().map_err(Error::io)
    }
    pub(crate) fn load(
        source: &Source,
        resources: &Resources,
        track: Option<TrackId>,
        policy: FrameDurationPolicy,
        path: &Path,
        limits: IndexBuildLimits,
    ) -> Result<Arc<Self>> {
        let _cpu = resources.reserve_cpu(0, 0)?;
        let (fingerprint, reservation) = prepare_build(source, resources, limits)?;
        let expected_key = key(&fingerprint)?;
        let file = File::open(path).map_err(Error::io)?;
        if file.metadata().map_err(Error::io)?.len() > limits.max_index_bytes as u64 {
            return Err(
                Error::exhausted("persisted index file exceeds metadata limit")
                    .with_constraint("video_index_bytes"),
            );
        }
        let mut decoder = Decoder {
            io: BufReader::with_capacity(4096, file),
            hash: BASIS,
        };
        if decoder.get::<8>()? != *MAGIC {
            return Err(Error::unsupported(
                "unsupported video index cache version; rebuild the timing index",
            ));
        }
        let length = usize::try_from(u64::from_le_bytes(decoder.get()?))
            .map_err(|_| Error::overflow("cached source key length"))?;
        if length > MAX_KEY || length != expected_key.len() {
            return Err(Error::incompatible(
                "cached video source identity/version differs",
            ));
        }
        // Compare keys incrementally without a second identity buffer.
        for &expected in &expected_key {
            if decoder.get::<1>()?[0] != expected {
                return Err(
                    Error::incompatible("cached video source identity/version is stale")
                        .with_constraint("video_index_source"),
                );
            }
        }
        let cached_track = TrackId(u32::from_le_bytes(decoder.get()?));
        let cached_policy = if decoder.flag()? {
            FrameDurationPolicy::TrustReported
        } else {
            FrameDurationPolicy::Unknown
        };
        if track.is_some_and(|track| track != cached_track) || cached_policy != policy {
            return Err(
                Error::incompatible("persisted video index interpretation differs")
                    .with_constraint("video_index_interpretation"),
            );
        }
        let origin = decoder.time()?;
        let count = usize::try_from(u64::from_le_bytes(decoder.get()?))
            .map_err(|_| Error::overflow("cached video frame count"))?;
        let maximum = limits.max_frames.min(
            limits.max_index_bytes.saturating_sub(
                source_charge(&fingerprint)?
                    .saturating_add(8192)
                    .saturating_add(expected_key.len()),
            ) / record_charge(),
        );
        if count == 0 || count > maximum {
            return Err(Error::exhausted(
                "cached video frame count exceeds explicit metadata/frame bounds",
            )
            .with_constraint("video_index_bytes"));
        }
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(count)
            .map_err(|_| Error::exhausted("cached index allocation failed"))?;
        for ordinal in 0..count {
            let source_pts = decoder.time()?;
            let duration = decoder.optional()?;
            let native_duration = decoder.optional()?;
            let keyframe = decoder.flag()?;
            // Navigation DTS can be zero or negative; duration validation does
            // not apply to these original compressed timestamps.
            let seek_timestamp = if decoder.flag()? {
                Some(decoder.time()?)
            } else {
                None
            };
            if policy == FrameDurationPolicy::Unknown && duration.is_some()
                || policy == FrameDurationPolicy::TrustReported && duration != native_duration
            {
                return Err(Error::incompatible(
                    "cached display-duration interpretation is inconsistent",
                ));
            }
            frames.push(IndexedTiming {
                timing: FrameTiming {
                    source_pts: Some(source_pts),
                    pts: Some(source_pts.checked_sub(origin)?),
                    duration,
                    native_duration,
                    ordinal: Some(ordinal as u64),
                },
                keyframe,
                seek_timestamp,
            });
        }
        let mut checksum = [0u8; 8];
        decoder.io.read_exact(&mut checksum).map_err(Error::io)?;
        if u64::from_le_bytes(checksum) != decoder.hash {
            return Err(Error::incompatible("video index checksum mismatch"));
        }
        let mut extra = [0u8; 1];
        if decoder.io.read(&mut extra).map_err(Error::io)? != 0 {
            return Err(Error::invalid("video index cache has trailing payload"));
        }
        drop(decoder);
        drop(expected_key);
        if source.fingerprint()? != fingerprint {
            return Err(Error::incompatible(
                "source changed while loading video index",
            ));
        }
        Self::finish(
            resources.clone(),
            fingerprint,
            cached_track,
            origin,
            policy,
            frames,
            reservation,
            IndexBuildReport {
                loaded: true,
                ..Default::default()
            },
        )
    }
}
