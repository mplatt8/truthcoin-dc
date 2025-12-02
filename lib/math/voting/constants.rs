//! Bitcoin Hivemind Consensus Constants
//!
//! All consensus-critical constants defined here to ensure network-wide consistency.

/// Neutral value (0.5) used for initial reputation and missing votes.
/// Prevents Sybil attacks by not favoring new voters.
pub const BITCOIN_HIVEMIND_NEUTRAL_VALUE: f64 = 0.5;

/// Catch function tolerance (0.2) discretizes outcomes to {0, 0.5, 1}.
/// Forces decisions toward discrete states when consensus is unclear.
pub const CONSENSUS_CATCH_TOLERANCE: f64 = 0.2;

/// SVD numerical tolerance (1e-10) for rank estimation and numerical stability.
pub const SVD_NUMERICAL_TOLERANCE: f64 = 1e-10;

/// Reputation smoothing alpha (0.3 = 30% new, 70% historical).
/// Balances responsiveness with stability to prevent gaming.
pub const REPUTATION_SMOOTHING_ALPHA: f64 = 0.3;

pub const REPUTATION_MIN: f64 = 0.0;
pub const REPUTATION_MAX: f64 = 1.0;

/// Production period duration (~3 months) allows for price discovery and voter research.
pub const PRODUCTION_PERIOD_DURATION_SECONDS: u64 = 3 * 30 * 24 * 60 * 60;

/// Refresh Votecoin proportions every 10 blocks to balance accuracy with performance.
pub const VOTECOIN_STALENESS_BLOCKS: u64 = 10;

/// Precision for rounding consensus outcomes (8 decimal places = satoshi-level precision).
/// This ensures cross-platform determinism by bounding any floating-point micro-divergence.
pub const CONSENSUS_PRECISION_DECIMALS: u32 = 8;

/// Precision for rounding reputation values.
pub const REPUTATION_PRECISION_DECIMALS: u32 = 10;

/// Precision for rounding SVD intermediate results.
pub const SVD_PRECISION_DECIMALS: u32 = 12;

/// Round a f64 value to a fixed number of decimal places for deterministic storage.
/// Uses "round half away from zero" semantics for consistency.
#[inline]
pub fn round_to_precision(value: f64, decimals: u32) -> f64 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }
    let multiplier = 10f64.powi(decimals as i32);
    (value * multiplier).round() / multiplier
}

/// Round a consensus outcome value to standard precision.
#[inline]
pub fn round_outcome(value: f64) -> f64 {
    round_to_precision(value, CONSENSUS_PRECISION_DECIMALS)
}

/// Round a reputation value to standard precision.
#[inline]
pub fn round_reputation(value: f64) -> f64 {
    round_to_precision(value, REPUTATION_PRECISION_DECIMALS)
}

/// Round SVD intermediate values to standard precision.
#[inline]
pub fn round_svd(value: f64) -> f64 {
    round_to_precision(value, SVD_PRECISION_DECIMALS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_value_is_midpoint() {
        assert_eq!(BITCOIN_HIVEMIND_NEUTRAL_VALUE, 0.5);
        assert_eq!(
            BITCOIN_HIVEMIND_NEUTRAL_VALUE,
            (REPUTATION_MIN + REPUTATION_MAX) / 2.0
        );
    }

    #[test]
    fn test_catch_tolerance_valid_range() {
        assert!(CONSENSUS_CATCH_TOLERANCE > 0.0);
        assert!(CONSENSUS_CATCH_TOLERANCE < 0.5);

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
        assert!(SVD_NUMERICAL_TOLERANCE > 0.0);
        assert!(SVD_NUMERICAL_TOLERANCE < 1e-6);
    }

    #[test]
    fn test_smoothing_alpha_valid() {
        assert!(REPUTATION_SMOOTHING_ALPHA > 0.0);
        assert!(REPUTATION_SMOOTHING_ALPHA < 1.0);
    }

    #[test]
    fn test_period_duration_reasonable() {
        let three_months_approx = 90 * 24 * 60 * 60;
        let tolerance = 5 * 24 * 60 * 60;

        assert!(
            (PRODUCTION_PERIOD_DURATION_SECONDS as i64
                - three_months_approx as i64)
                .abs()
                < tolerance as i64
        );
    }

    #[test]
    fn test_votecoin_staleness_reasonable() {
        assert!(VOTECOIN_STALENESS_BLOCKS > 0);
        assert!(VOTECOIN_STALENESS_BLOCKS < 100);
    }

    #[test]
    fn test_round_to_precision() {
        // Basic rounding
        assert_eq!(round_to_precision(0.123456789, 4), 0.1235);
        assert_eq!(round_to_precision(0.123456789, 8), 0.12345679);

        // Round half away from zero
        assert_eq!(round_to_precision(0.5555, 3), 0.556);
        assert_eq!(round_to_precision(-0.5555, 3), -0.556);

        // Edge cases
        assert_eq!(round_to_precision(0.0, 8), 0.0);
        assert_eq!(round_to_precision(1.0, 8), 1.0);

        // NaN and infinity pass through
        assert!(round_to_precision(f64::NAN, 8).is_nan());
        assert!(round_to_precision(f64::INFINITY, 8).is_infinite());
    }

    #[test]
    fn test_round_outcome() {
        assert_eq!(round_outcome(0.123456789012), 0.12345679);
        assert_eq!(round_outcome(0.999999999), 1.0);
    }

    #[test]
    fn test_round_reputation() {
        assert_eq!(round_reputation(0.12345678901234), 0.1234567890);
    }

    #[test]
    fn test_rounding_determinism() {
        // These values should always round the same way
        let test_values = [
            0.1 + 0.2,  // Classic floating point case
            1.0 / 3.0,
            std::f64::consts::PI,
            0.123456789012345,
        ];

        for val in test_values {
            let rounded = round_outcome(val);
            // Re-rounding should be idempotent
            assert_eq!(rounded, round_outcome(rounded));
        }
    }
}
