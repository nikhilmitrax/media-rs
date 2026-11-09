//! Structured failures with source, backend, policy, and native-cause context.
use std::{error::Error as StdError, fmt};
/// Library operation result.
pub type Result<T> = std::result::Result<T, Error>;
/// Stable failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Malformed request or description.
    InvalidInput,
    /// Arithmetic/address overflow.
    Overflow,
    /// Unavailable capability.
    Unsupported,
    /// Input/output failure.
    Io,
    /// Native implementation failure.
    Backend,
    /// An enforced allocation/queue/pool budget was reached.
    ResourceExhausted,
    /// Execution cancelled by its caller.
    Cancelled,
    /// Timing cannot establish one unique answer.
    AmbiguousTiming,
    /// Missing coverage or sample.
    MissingData,
    /// Incompatible representation or policy.
    Incompatible,
}
/// Structured operation context.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ErrorContext {
    /// Source path/URL/identity.
    pub source: Option<String>,
    /// Logical stream identifier.
    pub track: Option<u32>,
    /// Failing operation.
    pub operation: Option<String>,
    /// Implementation identity.
    pub backend: Option<String>,
    /// Hard constraint or limit.
    pub constraint: Option<String>,
    /// Media time when known.
    pub time: Option<crate::time::Time>,
}
/// Actionable error retaining its native cause.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    context: Box<ErrorContext>,
    cause: Option<Box<dyn StdError + Send + Sync>>,
}
impl Error {
    /// Construct a categorized error.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            context: Box::default(),
            cause: None,
        }
    }
    /// Invalid input.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }
    /// Checked arithmetic failed.
    pub fn overflow(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Overflow, message)
    }
    /// A capability is unavailable.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }
    /// Representation/policy mismatch.
    pub fn incompatible(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Incompatible, message)
    }
    /// A bounded resource could not be acquired.
    pub fn exhausted(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ResourceExhausted, message)
    }
    /// I/O failure retaining its native cause.
    pub fn io(cause: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, cause.to_string()).with_cause(cause)
    }
    /// Backend failure.
    pub fn backend(backend: &str, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Backend, message).with_backend(backend)
    }
    /// Attach the native cause.
    pub fn with_cause(mut self, cause: impl StdError + Send + Sync + 'static) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }
    /// Attach source identity.
    pub fn with_source(mut self, value: impl Into<String>) -> Self {
        self.context.source = Some(value.into());
        self
    }
    /// Attach logical track.
    pub fn with_track(mut self, value: u32) -> Self {
        self.context.track = Some(value);
        self
    }
    /// Attach operation.
    pub fn with_operation(mut self, value: impl Into<String>) -> Self {
        self.context.operation = Some(value.into());
        self
    }
    /// Attach backend.
    pub fn with_backend(mut self, value: impl Into<String>) -> Self {
        self.context.backend = Some(value.into());
        self
    }
    /// Attach constraint.
    pub fn with_constraint(mut self, value: impl Into<String>) -> Self {
        self.context.constraint = Some(value.into());
        self
    }
    /// Attach media time.
    pub fn with_time(mut self, value: crate::time::Time) -> Self {
        self.context.time = Some(value);
        self
    }
    /// Machine-readable category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    /// Explanation.
    pub fn message(&self) -> &str {
        &self.message
    }
    /// Structured context.
    pub fn context(&self) -> &ErrorContext {
        &self.context
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(v) = &self.context.operation {
            write!(f, " [operation={v}]")?;
        }
        if let Some(v) = &self.context.source {
            write!(f, " [source={v}]")?;
        }
        if let Some(v) = self.context.track {
            write!(f, " [track={v}]")?;
        }
        if let Some(v) = &self.context.backend {
            write!(f, " [backend={v}]")?;
        }
        if let Some(v) = &self.context.constraint {
            write!(f, " [constraint={v}]")?;
        }
        if let Some(v) = self.context.time {
            write!(f, " [time={v:?}]")?;
        }
        Ok(())
    }
}
impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.cause
            .as_ref()
            .map(|v| v.as_ref() as &(dyn StdError + 'static))
    }
}
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::io(value)
    }
}
