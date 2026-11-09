//! Reusable plan descriptions with explicit materializations.
use crate::diagnostics::PlanStep;
use std::fmt;
/// Backend choices and operation boundaries resolved before execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanDescription {
    /// Ordered operations.
    pub steps: Vec<PlanStep>,
    /// Decisions that still require input/runtime information.
    pub unresolved: Vec<String>,
}
impl PlanDescription {
    /// Human-readable plan.
    pub fn explain(&self) -> String {
        self.to_string()
    }
}
impl fmt::Display for PlanDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.steps.iter().enumerate() {
            writeln!(
                f,
                "{i}: {} via {} -> {:?} {:?}; allocation={}",
                s.operation, s.backend, s.output, s.format, s.materializes
            )?;
        }
        for decision in &self.unresolved {
            writeln!(f, "runtime: {decision}")?;
        }
        Ok(())
    }
}
