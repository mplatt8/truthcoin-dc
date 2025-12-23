//! Satoshi conversion utilities with standardized rounding.
//!
//! This module provides consistent float-to-integer conversion for financial
//! calculations, with explicit rounding modes and validation.
//!
//! # Rounding Conventions
//! - `Rounding::Up` (ceil): Use for costs/fees charged TO user
//! - `Rounding::Down` (floor): Use for payouts TO user
//! - `Rounding::Nearest` (round): Use for neutral/balanced calculations

use thiserror::Error;

/// Errors that can occur during satoshi conversion.
#[derive(Debug, Clone, Error)]
pub enum SatoshiError {
    #[error("Non-finite value: {0}")]
    NonFinite(f64),
    #[error("Negative value not allowed: {0}")]
    Negative(f64),
    #[error("Value exceeds maximum: {0}")]
    Overflow(f64),
}

/// Rounding strategy for satoshi conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Round up (ceil) - use for costs/fees charged TO user.
    Up,
    /// Round down (floor) - use for payouts TO user.
    Down,
    /// Round to nearest - use for neutral calculations.
    Nearest,
}

/// Convert f64 to u64 satoshis with validation and explicit rounding.
///
/// # Arguments
/// * `value` - The floating-point value (already in satoshis)
/// * `mode` - The rounding strategy to apply
///
/// # Errors
/// Returns `SatoshiError` if:
/// - Value is NaN or infinite
/// - Value is negative
/// - Rounded value exceeds u64::MAX
///
/// # Examples
/// ```
/// use truthcoin_dc::math::satoshi::{to_sats, Rounding};
///
/// // Costs round up
/// assert_eq!(to_sats(100.3, Rounding::Up).unwrap(), 101);
///
/// // Payouts round down
/// assert_eq!(to_sats(100.9, Rounding::Down).unwrap(), 100);
/// ```
pub fn to_sats(value: f64, mode: Rounding) -> Result<u64, SatoshiError> {
    if !value.is_finite() {
        return Err(SatoshiError::NonFinite(value));
    }
    if value < 0.0 {
        return Err(SatoshiError::Negative(value));
    }
    let rounded = match mode {
        Rounding::Up => value.ceil(),
        Rounding::Down => value.floor(),
        Rounding::Nearest => value.round(),
    };
    if rounded > u64::MAX as f64 {
        return Err(SatoshiError::Overflow(value));
    }
    Ok(rounded as u64)
}

/// Convert f64 to i64 satoshis (for costs that can be negative).
///
/// For signed values, rounding direction is applied based on the sign:
/// - `Up` rounds away from zero (positive: ceil, negative: floor)
/// - `Down` rounds toward zero (positive: floor, negative: ceil)
/// - `Nearest` always rounds to the nearest integer
///
/// # Arguments
/// * `value` - The floating-point value (already in satoshis)
/// * `mode` - The rounding strategy to apply
///
/// # Errors
/// Returns `SatoshiError` if:
/// - Value is NaN or infinite
/// - Rounded value exceeds i64 bounds
pub fn to_sats_signed(value: f64, mode: Rounding) -> Result<i64, SatoshiError> {
    if !value.is_finite() {
        return Err(SatoshiError::NonFinite(value));
    }
    let rounded = match mode {
        Rounding::Up => {
            if value >= 0.0 {
                value.ceil()
            } else {
                value.floor()
            }
        }
        Rounding::Down => {
            if value >= 0.0 {
                value.floor()
            } else {
                value.ceil()
            }
        }
        Rounding::Nearest => value.round(),
    };
    if rounded > i64::MAX as f64 || rounded < i64::MIN as f64 {
        return Err(SatoshiError::Overflow(value));
    }
    Ok(rounded as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_sats_rounding_up() {
        assert_eq!(to_sats(100.0, Rounding::Up).unwrap(), 100);
        assert_eq!(to_sats(100.1, Rounding::Up).unwrap(), 101);
        assert_eq!(to_sats(100.9, Rounding::Up).unwrap(), 101);
        assert_eq!(to_sats(0.0, Rounding::Up).unwrap(), 0);
    }

    #[test]
    fn test_to_sats_rounding_down() {
        assert_eq!(to_sats(100.0, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats(100.1, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats(100.9, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats(0.0, Rounding::Down).unwrap(), 0);
    }

    #[test]
    fn test_to_sats_rounding_nearest() {
        assert_eq!(to_sats(100.0, Rounding::Nearest).unwrap(), 100);
        assert_eq!(to_sats(100.4, Rounding::Nearest).unwrap(), 100);
        assert_eq!(to_sats(100.5, Rounding::Nearest).unwrap(), 101); // round half to even: 100.5 -> 100 or 101
        assert_eq!(to_sats(100.6, Rounding::Nearest).unwrap(), 101);
    }

    #[test]
    fn test_to_sats_negative_error() {
        assert!(matches!(
            to_sats(-1.0, Rounding::Up),
            Err(SatoshiError::Negative(_))
        ));
    }

    #[test]
    fn test_to_sats_non_finite_error() {
        assert!(matches!(
            to_sats(f64::NAN, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
        assert!(matches!(
            to_sats(f64::INFINITY, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
        assert!(matches!(
            to_sats(f64::NEG_INFINITY, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
    }

    #[test]
    fn test_to_sats_signed_positive() {
        assert_eq!(to_sats_signed(100.3, Rounding::Up).unwrap(), 101);
        assert_eq!(to_sats_signed(100.3, Rounding::Down).unwrap(), 100);
        assert_eq!(to_sats_signed(100.5, Rounding::Nearest).unwrap(), 101);
    }

    #[test]
    fn test_to_sats_signed_negative() {
        // Up rounds away from zero: -100.3 -> -101
        assert_eq!(to_sats_signed(-100.3, Rounding::Up).unwrap(), -101);
        // Down rounds toward zero: -100.3 -> -100
        assert_eq!(to_sats_signed(-100.3, Rounding::Down).unwrap(), -100);
        // Nearest: -100.5 -> -101 (or -100, depending on tie-breaking)
        assert_eq!(to_sats_signed(-100.6, Rounding::Nearest).unwrap(), -101);
    }

    #[test]
    fn test_to_sats_signed_non_finite_error() {
        assert!(matches!(
            to_sats_signed(f64::NAN, Rounding::Up),
            Err(SatoshiError::NonFinite(_))
        ));
    }
}
