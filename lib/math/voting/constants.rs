//! Bitcoin Hivemind Consensus Constants
//!
//! # Single Source of Truth for All Consensus-Critical Constants
//!
//! This module defines all mathematical constants used in the Bitcoin Hivemind
//! voting consensus algorithm. These values are consensus-critical and must
//! remain consistent across all nodes in the network.
//!
//! ## Bitcoin Hivemind Whitepaper References
//! - Section 4.2: "Consensus Algorithm" - Neutral values and convergence thresholds
//! - Appendix A: "Mathematical Details" - Numerical tolerances and smoothing parameters
//!
//! ## Architectural Principle
//! All hard-coded numeric values in the voting system MUST be defined here
//! to prevent inconsistencies and ensure easy auditing of consensus rules.

/// Neutral value representing "inconclusive" or "unknown" state
///
/// # Bitcoin Hivemind Specification
/// Per the whitepaper, 0.5 represents the neutral midpoint between
/// binary outcomes (0.0 = No, 1.0 = Yes). This is used for:
/// - Initial voter reputation (no history to judge performance)
/// - Missing vote imputation (no opinion expressed)
/// - Uncertain outcome representation
///
/// # Whitepaper Reference
/// Section 4.2: "New voters start with neutral reputation to prevent
/// Sybil attacks through identity creation."
pub const BITCOIN_HIVEMIND_NEUTRAL_VALUE: f64 = 0.5;

/// Tolerance threshold for "Catch" function in consensus algorithm
///
/// # Bitcoin Hivemind Specification
/// The Catch function rounds outcomes to discrete values (0, 0.5, 1)
/// based on proximity to these values. A tolerance of 0.2 means:
/// - Values in [0.0, 0.3] round to 0.0
/// - Values in [0.3, 0.7] round to 0.5
/// - Values in [0.7, 1.0] round to 1.0
///
/// This prevents outcomes from being overly precise when voter consensus
/// is unclear and forces decisions toward discrete states.
///
/// # Whitepaper Reference
/// Section 4.2: "Catch Parameter" - Discretizes outcomes based on consensus strength
pub const CONSENSUS_CATCH_TOLERANCE: f64 = 0.2;

/// Numerical tolerance for SVD (Singular Value Decomposition) operations
///
/// # Technical Context
/// Used to determine:
/// - Zero singular values (rank estimation)
/// - Near-zero matrix elements (numerical stability)
/// - Convergence thresholds for iterative algorithms
///
/// Value of 1e-10 provides sufficient precision while avoiding
/// floating-point rounding issues in typical voting matrices.
///
/// # Whitepaper Reference
/// Appendix A: "Numerical Methods" - Precision requirements for PCA operations
pub const SVD_NUMERICAL_TOLERANCE: f64 = 1e-10;

/// Smoothing parameter for reputation updates (alpha in exponential smoothing)
///
/// # Bitcoin Hivemind Specification
/// Controls how quickly reputation responds to new voting performance:
/// - new_reputation = alpha × new_score + (1 - alpha) × old_reputation
///
/// A value of 0.1 means:
/// - 10% weight on new performance
/// - 90% weight on historical reputation
///
/// This prevents rapid reputation swings while allowing gradual correction.
///
/// # Whitepaper Reference
/// Section 4.3: "Reputation Updates" - Exponential smoothing prevents gaming
pub const REPUTATION_SMOOTHING_ALPHA: f64 = 0.1;

/// Minimum allowed reputation value
///
/// # Bitcoin Hivemind Specification
/// Reputation is clamped to [0.0, 1.0] range where:
/// - 0.0 = Consistently incorrect voter (minimum influence)
/// - 1.0 = Perfectly accurate voter (maximum influence)
///
/// Zero reputation means zero voting weight (no influence on outcomes).
///
/// # Whitepaper Reference
/// Section 4.3: "Reputation bounds prevent negative or unbounded influence"
pub const REPUTATION_MIN: f64 = 0.0;

/// Maximum allowed reputation value
///
/// # Bitcoin Hivemind Specification
/// Reputation is clamped to [0.0, 1.0] range where:
/// - 0.0 = Consistently incorrect voter (minimum influence)
/// - 1.0 = Perfectly accurate voter (maximum influence)
///
/// Maximum reputation provides upper bound on voting influence.
///
/// # Whitepaper Reference
/// Section 4.3: "Reputation bounds prevent negative or unbounded influence"
pub const REPUTATION_MAX: f64 = 1.0;

/// Production period duration in seconds (approximately 3 months)
///
/// # Bitcoin Hivemind Specification
/// Production periods follow quarterly (3-month) cycles:
/// - 3 months × 30 days × 24 hours × 60 minutes × 60 seconds
/// - Approximately one quarter-year per voting period
///
/// This provides sufficient time for markets to develop price discovery
/// and for voters to research outcomes before voting.
///
/// # Whitepaper Reference
/// Section 4.1: "Voting Periods" - Quarter-year cycles for production
pub const PRODUCTION_PERIOD_DURATION_SECONDS: u64 = 3 * 30 * 24 * 60 * 60;

/// Blocks between Votecoin proportion refreshes
///
/// # Implementation Detail
/// Votecoin holdings can change due to transfers or new issuance.
/// To balance accuracy with performance, we refresh proportions
/// every N blocks rather than on every vote.
///
/// Value of 10 blocks provides:
/// - Reasonable freshness (every ~10 minutes with 1-minute blocks)
/// - Minimal database query overhead
///
/// # Design Tradeoff
/// Lower values = more accurate weights but more database queries
/// Higher values = fewer queries but potentially stale weights
pub const VOTECOIN_STALENESS_BLOCKS: u64 = 10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_value_is_midpoint() {
        // Neutral value should be exactly halfway between min and max
        assert_eq!(BITCOIN_HIVEMIND_NEUTRAL_VALUE, 0.5);
        assert_eq!(
            BITCOIN_HIVEMIND_NEUTRAL_VALUE,
            (REPUTATION_MIN + REPUTATION_MAX) / 2.0
        );
    }

    #[test]
    fn test_catch_tolerance_valid_range() {
        // Catch tolerance should allow symmetric discretization around 0.5
        assert!(CONSENSUS_CATCH_TOLERANCE > 0.0);
        assert!(CONSENSUS_CATCH_TOLERANCE < 0.5);

        // Verify discretization boundaries make sense
        let lower_boundary = 0.5 - CONSENSUS_CATCH_TOLERANCE / 2.0;
        let upper_boundary = 0.5 + CONSENSUS_CATCH_TOLERANCE / 2.0;

        assert_eq!(lower_boundary, 0.4);
        assert_eq!(upper_boundary, 0.6);
    }

    #[test]
    fn test_reputation_bounds() {
        assert_eq!(REPUTATION_MIN, 0.0);
        assert_eq!(REPUTATION_MAX, 1.0);
        assert!(REPUTATION_MIN < REPUTATION_MAX);
    }

    #[test]
    fn test_numerical_tolerance_reasonable() {
        // Should be small enough for precision but not cause underflow
        assert!(SVD_NUMERICAL_TOLERANCE > 0.0);
        assert!(SVD_NUMERICAL_TOLERANCE < 1e-6);
    }

    #[test]
    fn test_smoothing_alpha_valid() {
        // Alpha should be in (0, 1) for valid exponential smoothing
        assert!(REPUTATION_SMOOTHING_ALPHA > 0.0);
        assert!(REPUTATION_SMOOTHING_ALPHA < 1.0);
    }

    #[test]
    fn test_period_duration_reasonable() {
        // Should be approximately 3 months in seconds
        let three_months_approx = 90 * 24 * 60 * 60; // 90 days
        let tolerance = 5 * 24 * 60 * 60; // 5 days tolerance

        assert!(
            (PRODUCTION_PERIOD_DURATION_SECONDS as i64 - three_months_approx as i64).abs()
            < tolerance as i64
        );
    }

    #[test]
    fn test_votecoin_staleness_reasonable() {
        // Should be positive and reasonable (not too frequent, not too stale)
        assert!(VOTECOIN_STALENESS_BLOCKS > 0);
        assert!(VOTECOIN_STALENESS_BLOCKS < 100); // Not more than 100 blocks
    }
}
