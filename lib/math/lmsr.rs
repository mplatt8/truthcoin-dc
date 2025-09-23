/// Optimized LMSR Mathematical Operations using ndarray for Bitcoin Hivemind Sidechain
///
/// This module provides LMSR operations optimized for dense array representations,
/// leveraging ndarray's vectorized operations for better performance while maintaining
/// the numerical stability of the original implementation.
use ndarray::{Array, Array1, ArrayView1, IxDyn};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Precision constants for LMSR calculations
const SATOSHI_PRECISION: f64 = 100_000_000.0;
const LMSR_PRECISION: f64 = 1e-8;
const MAX_BETA: f64 = 1e12;
const MIN_BETA: f64 = 1e-6;
const FEE_SCALE: u64 = 10000;

/// LMSR calculation errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LmsrError {
    InvalidBeta {
        beta: f64,
        min: f64,
        max: f64,
    },
    ShareQuantityOverflow,
    InvalidCostCalculation,
    InsufficientTreasury {
        required: u64,
        available: u64,
    },
    InvalidOutcomeCount {
        count: usize,
        min: usize,
        max: usize,
    },
    PrecisionLoss,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for LmsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LmsrError::InvalidBeta { beta, min, max } => write!(
                f,
                "Beta {} outside valid range [{}, {}]",
                beta, min, max
            ),
            LmsrError::ShareQuantityOverflow => {
                write!(f, "Share quantity overflow")
            }
            LmsrError::InvalidCostCalculation => {
                write!(f, "Invalid cost calculation")
            }
            LmsrError::InsufficientTreasury {
                required,
                available,
            } => write!(
                f,
                "Insufficient treasury: required {}, available {}",
                required, available
            ),
            LmsrError::InvalidOutcomeCount { count, min, max } => write!(
                f,
                "Invalid outcome count {}: must be between {} and {}",
                count, min, max
            ),
            LmsrError::PrecisionLoss => write!(f, "Numerical precision loss"),
            LmsrError::DimensionMismatch { expected, actual } => write!(
                f,
                "Dimension mismatch: expected {}, got {}",
                expected, actual
            ),
        }
    }
}

impl std::error::Error for LmsrError {}

/// LMSR state using ndarray for efficient vector operations
#[derive(Debug, Clone, PartialEq)]
pub struct LmsrState {
    pub beta: f64,
    pub shares: Array1<f64>,
    pub treasury_balance: u64,
    pub trading_fee: f64,
}

/// Result of an LMSR trade calculation
#[derive(Debug, Clone, PartialEq)]
pub struct LmsrTradeResult {
    pub cost_satoshis: i64,
    pub fee_satoshis: u64,
    pub new_shares: Array1<f64>,
    pub new_treasury_balance: u64,
}

/// Price information for market outcomes
#[derive(Debug, Clone, PartialEq)]
pub struct LmsrPriceInfo {
    pub prices: Array1<f64>,
    pub marginal_costs: Array1<u64>,
}

/// LMSR implementation optimized for ndarray
pub struct Lmsr {
    max_outcomes: usize,
}

impl Lmsr {
    pub fn new(max_outcomes: usize) -> Self {
        Self { max_outcomes }
    }

    /// Core LMSR cost function with numerical stability
    /// C(q) = b * ln(Σ exp(q_i / b))
    pub fn cost_function(
        &self,
        beta: f64,
        shares: &ArrayView1<f64>,
    ) -> Result<f64, LmsrError> {
        if beta <= MIN_BETA || beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        if shares.is_empty() {
            return Ok(0.0);
        }

        // Log-sum-exp trick for numerical stability
        let max_share = shares.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        // Vectorized computation of exp((shares - max_share) / beta)
        let shifted_shares = (shares - max_share) / beta;
        let exp_shares = shifted_shares.mapv(f64::exp);

        // Check for overflow
        if !exp_shares.iter().all(|x| x.is_finite()) {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        let sum_exp = exp_shares.sum();

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        let cost = beta * (sum_exp.ln() + max_share / beta);

        if !cost.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        Ok(cost)
    }

    /// Calculate prices: p_i = exp(q_i / b) / Σ exp(q_j / b)
    pub fn calculate_prices(
        &self,
        beta: f64,
        shares: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, LmsrError> {
        if shares.is_empty() {
            return Ok(Array1::zeros(0));
        }

        // Log-sum-exp for numerical stability
        let max_share = shares.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        // Vectorized operations
        let shifted_shares = (shares - max_share) / beta;
        let exp_shares = shifted_shares.mapv(f64::exp);

        if !exp_shares.iter().all(|x| x.is_finite()) {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        let sum_exp = exp_shares.sum();

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        // Normalize to get probabilities
        let prices = exp_shares / sum_exp;

        // Verify prices sum to 1
        let price_sum: f64 = prices.sum();
        if (price_sum - 1.0).abs() > LMSR_PRECISION {
            return Err(LmsrError::PrecisionLoss);
        }

        Ok(prices)
    }

    /// Calculate the cost of buying shares
    pub fn calculate_buy_cost(
        &self,
        state: &LmsrState,
        outcome: usize,
        shares_to_buy: f64,
    ) -> Result<LmsrTradeResult, LmsrError> {
        self.validate_state(state)?;

        if shares_to_buy <= 0.0 {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        if outcome >= state.shares.len() {
            return Err(LmsrError::DimensionMismatch {
                expected: state.shares.len(),
                actual: outcome + 1,
            });
        }

        // Calculate cost before trade
        let cost_before =
            self.cost_function(state.beta, &state.shares.view())?;

        // Calculate cost after adding shares
        let mut new_shares = state.shares.clone();
        new_shares[outcome] += shares_to_buy;

        let cost_after = self.cost_function(state.beta, &new_shares.view())?;

        // Trade cost is the difference
        let trade_cost = cost_after - cost_before;
        let cost_satoshis = self.cost_to_satoshis(trade_cost)?;

        if cost_satoshis <= 0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        // Calculate fee
        let fee_satoshis =
            self.calculate_fee(cost_satoshis as u64, state.trading_fee);
        let total_cost = cost_satoshis + (fee_satoshis as i64);

        // Check treasury sufficiency
        if total_cost > (state.treasury_balance as i64) {
            return Err(LmsrError::InsufficientTreasury {
                required: total_cost as u64,
                available: state.treasury_balance,
            });
        }

        let new_treasury_balance = state.treasury_balance - (total_cost as u64);

        Ok(LmsrTradeResult {
            cost_satoshis: total_cost,
            fee_satoshis,
            new_shares,
            new_treasury_balance,
        })
    }

    /// Calculate the payout from selling shares
    pub fn calculate_sell_payout(
        &self,
        state: &LmsrState,
        outcome: usize,
        shares_to_sell: f64,
    ) -> Result<LmsrTradeResult, LmsrError> {
        self.validate_state(state)?;

        if shares_to_sell <= 0.0 {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        if outcome >= state.shares.len() {
            return Err(LmsrError::DimensionMismatch {
                expected: state.shares.len(),
                actual: outcome + 1,
            });
        }

        if shares_to_sell > state.shares[outcome] {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        // Calculate as negative buy
        let mut new_shares = state.shares.clone();
        new_shares[outcome] -= shares_to_sell;

        let cost_before =
            self.cost_function(state.beta, &state.shares.view())?;
        let cost_after = self.cost_function(state.beta, &new_shares.view())?;

        let payout = cost_before - cost_after;
        let payout_satoshis = self.cost_to_satoshis(payout)?;

        if payout_satoshis <= 0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        let fee_satoshis =
            self.calculate_fee(payout_satoshis as u64, state.trading_fee);
        let net_payout = payout_satoshis - (fee_satoshis as i64);

        let new_treasury_balance = state.treasury_balance + (net_payout as u64);

        Ok(LmsrTradeResult {
            cost_satoshis: -net_payout,
            fee_satoshis,
            new_shares,
            new_treasury_balance,
        })
    }

    /// Calculate current market prices
    pub fn calculate_current_prices(
        &self,
        state: &LmsrState,
    ) -> Result<LmsrPriceInfo, LmsrError> {
        self.validate_state(state)?;

        let prices = self.calculate_prices(state.beta, &state.shares.view())?;

        // Calculate marginal costs (vectorized)
        let mut marginal_costs = Array1::zeros(state.shares.len());
        for outcome in 0..state.shares.len() {
            match self.calculate_buy_cost(state, outcome, 1.0) {
                Ok(result) => {
                    marginal_costs[outcome] = result.cost_satoshis as u64;
                }
                Err(_) => {
                    marginal_costs[outcome] = u64::MAX;
                }
            }
        }

        Ok(LmsrPriceInfo {
            prices,
            marginal_costs,
        })
    }

    /// Calculate required treasury funding
    pub fn calculate_required_treasury(
        &self,
        beta: f64,
        outcome_count: usize,
    ) -> Result<u64, LmsrError> {
        if beta <= MIN_BETA || beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        if outcome_count == 0 || outcome_count > self.max_outcomes {
            return Err(LmsrError::InvalidOutcomeCount {
                count: outcome_count,
                min: 1,
                max: self.max_outcomes,
            });
        }

        // Maximum loss: b * ln(outcome_count)
        // Beta is expected to be in satoshi units for treasury calculation
        let max_cost = beta * (outcome_count as f64).ln();

        if !max_cost.is_finite() || max_cost < 0.0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        Ok(max_cost.round() as u64)
    }

    /// Validate LMSR state consistency
    pub fn validate_state(&self, state: &LmsrState) -> Result<(), LmsrError> {
        // Validate beta
        if state.beta <= MIN_BETA || state.beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta: state.beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        // Validate fee
        if state.trading_fee < 0.0 || state.trading_fee >= 1.0 {
            return Err(LmsrError::InvalidCostCalculation);
        }

        // Validate shares
        if !state.shares.iter().all(|&x| x >= 0.0 && x.is_finite()) {
            return Err(LmsrError::ShareQuantityOverflow);
        }

        // Check outcome count
        if state.shares.len() > self.max_outcomes {
            return Err(LmsrError::InvalidOutcomeCount {
                count: state.shares.len(),
                min: 1,
                max: self.max_outcomes,
            });
        }

        // Validate prices sum to 1
        if state.shares.len() > 0 {
            let prices =
                self.calculate_prices(state.beta, &state.shares.view())?;
            let price_sum: f64 = prices.sum();
            if (price_sum - 1.0).abs() > LMSR_PRECISION {
                return Err(LmsrError::PrecisionLoss);
            }
        }

        Ok(())
    }

    /// Convert cost to satoshis
    fn cost_to_satoshis(&self, cost: f64) -> Result<i64, LmsrError> {
        let satoshis = cost * SATOSHI_PRECISION;

        if !satoshis.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        Ok(satoshis.round() as i64)
    }

    /// Calculate trading fee
    fn calculate_fee(&self, cost_satoshis: u64, fee_rate: f64) -> u64 {
        let fee_bp = (fee_rate * (FEE_SCALE as f64)).round() as u64;
        cost_satoshis.saturating_mul(fee_bp) / FEE_SCALE
    }
}

impl Default for Lmsr {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Support for multidimensional markets using IxDyn
pub struct LmsrMultidim {
    max_total_outcomes: usize,
}

impl LmsrMultidim {
    pub fn new(max_total_outcomes: usize) -> Self {
        Self { max_total_outcomes }
    }

    /// Calculate cost for multidimensional market
    pub fn cost_function_multidim(
        &self,
        beta: f64,
        shares: &Array<f64, IxDyn>,
    ) -> Result<f64, LmsrError> {
        if beta <= MIN_BETA || beta >= MAX_BETA {
            return Err(LmsrError::InvalidBeta {
                beta,
                min: MIN_BETA,
                max: MAX_BETA,
            });
        }

        let flat_shares =
            shares.as_slice().ok_or(LmsrError::InvalidCostCalculation)?;

        if flat_shares.is_empty() {
            return Ok(0.0);
        }

        // Log-sum-exp on flattened array
        let max_share = flat_shares
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        let mut sum_exp = 0.0;
        for &share in flat_shares {
            let exp_val = ((share - max_share) / beta).exp();
            if !exp_val.is_finite() {
                return Err(LmsrError::ShareQuantityOverflow);
            }
            sum_exp += exp_val;
        }

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        let cost = beta * (sum_exp.ln() + max_share / beta);

        if !cost.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        Ok(cost)
    }

    /// Calculate prices for multidimensional market
    pub fn calculate_prices_multidim(
        &self,
        beta: f64,
        shares: &Array<f64, IxDyn>,
    ) -> Result<Array<f64, IxDyn>, LmsrError> {
        let flat_shares =
            shares.as_slice().ok_or(LmsrError::InvalidCostCalculation)?;

        if flat_shares.is_empty() {
            return Ok(Array::zeros(shares.raw_dim()));
        }

        // Log-sum-exp on flattened array
        let max_share = flat_shares
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        let mut exp_values = Vec::with_capacity(flat_shares.len());
        let mut sum_exp = 0.0;

        for &share in flat_shares {
            let exp_val = ((share - max_share) / beta).exp();
            if !exp_val.is_finite() {
                return Err(LmsrError::ShareQuantityOverflow);
            }
            exp_values.push(exp_val);
            sum_exp += exp_val;
        }

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(LmsrError::InvalidCostCalculation);
        }

        // Normalize
        let prices_vec: Vec<f64> =
            exp_values.iter().map(|&x| x / sum_exp).collect();

        // Reshape to original dimensions
        let prices = Array::from_shape_vec(shares.raw_dim(), prices_vec)
            .map_err(|_| LmsrError::InvalidCostCalculation)?;

        Ok(prices)
    }
}

/// Standalone functions for backward compatibility
pub fn calculate_cost(shares: &[f64], beta: f64) -> Result<f64, LmsrError> {
    let shares_array = Array1::from_vec(shares.to_vec());
    let lmsr = Lmsr::new(shares.len());
    lmsr.cost_function(beta, &shares_array.view())
}

pub fn calculate_prices(
    shares: &[f64],
    beta: f64,
) -> Result<Vec<f64>, LmsrError> {
    let shares_array = Array1::from_vec(shares.to_vec());
    let lmsr = Lmsr::new(shares.len());
    let prices = lmsr.calculate_prices(beta, &shares_array.view())?;
    Ok(prices.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cost_calculation() {
        let lmsr = Lmsr::default();
        let state = LmsrState {
            beta: 14400.0,
            shares: array![10.0, 5.0],
            treasury_balance: 1000000,
            trading_fee: 0.01,
        };

        let result = lmsr.calculate_buy_cost(&state, 0, 1.0).unwrap();
        assert!(result.cost_satoshis > 0);
        assert!(result.fee_satoshis > 0);
        assert_eq!(result.new_shares[0], 11.0);
    }

    #[test]
    fn test_price_calculation() {
        let lmsr = Lmsr::default();
        let shares = array![10.0, 5.0];
        let prices = lmsr.calculate_prices(100.0, &shares.view()).unwrap();

        // Prices should sum to 1.0
        let price_sum: f64 = prices.sum();
        assert!((price_sum - 1.0).abs() < LMSR_PRECISION);

        // Outcome 0 should have higher price (more shares)
        assert!(prices[0] > prices[1]);
    }

    #[test]
    fn test_multidim_market() {
        let lmsr = LmsrMultidim::new(256);

        // 2x3x2 market
        let shares = Array::zeros(IxDyn(&[2, 3, 2]));
        let beta = 7.0;

        let cost = lmsr.cost_function_multidim(beta, &shares).unwrap();
        let expected_cost = beta * (12.0_f64).ln(); // 12 total outcomes
        assert!((cost - expected_cost).abs() < LMSR_PRECISION);

        let prices = lmsr.calculate_prices_multidim(beta, &shares).unwrap();
        let price_sum: f64 = prices.sum();
        assert!((price_sum - 1.0).abs() < LMSR_PRECISION);
    }

    #[test]
    fn test_numerical_stability() {
        let lmsr = Lmsr::default();

        // Test with large share values that would overflow naive implementation
        let shares = array![1000.0, 999.0, 998.0];
        let beta = 14400.0;

        let result = lmsr.calculate_prices(beta, &shares.view());
        assert!(result.is_ok());

        let prices = result.unwrap();
        assert!(
            prices
                .iter()
                .all(|&p| p.is_finite() && p >= 0.0 && p <= 1.0)
        );
    }
}
