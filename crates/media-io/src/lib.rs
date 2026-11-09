//! Independent local/memory/versioned-range sources, optional HTTP objects,
//! source-version guarded indexes, and explicitly finalized destinations.

#[cfg(feature = "http")]
pub mod http;
pub mod index;
pub mod range;
pub mod sink;
pub mod source;
