//! Multimedia operations, planning, and execution boundaries.
//!
//! Exact temporal selection, deterministic dataset scheduling, bounded execution,
//! and explicit operation planning shared by native backends and facade requests.

pub mod clip_sampling;
pub mod dataset;
pub mod diagnostics;
pub mod execution;
pub mod operation;
pub mod plan;
pub mod sampling;
pub mod session;
