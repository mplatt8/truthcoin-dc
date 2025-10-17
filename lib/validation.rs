use crate::state::Error;
use crate::state::markets::{
    DFunction, DimensionSpec, MarketError, MarketState,
};
use crate::state::slots::{Decision, SlotId};
use crate::types::{Address, FilledTransaction};
use sneed::RoTxn;
use std::collections::HashSet;

pub trait SlotValidationInterface {
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error>;

    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error>;
}

pub struct SlotValidator;

impl SlotValidator {
    pub fn parse_slot_id_from_hex(slot_id_hex: &str) -> Result<SlotId, Error> {
        SlotId::from_hex(slot_id_hex)
    }

    pub fn validate_slot_id_consistency(
        slot_id: &SlotId,
        slot_id_bytes: [u8; 3],
    ) -> Result<(), Error> {
        if slot_id.as_bytes() != slot_id_bytes {
            return Err(Error::InvalidSlotId {
                reason: "Slot ID bytes don't match computed slot ID"
                    .to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_decision_structure(
        market_maker_address_bytes: [u8; 20],
        slot_id_bytes: [u8; 3],
        is_standard: bool,
        is_scaled: bool,
        question: &str,
        min: Option<u16>,
        max: Option<u16>,
    ) -> Result<Decision, Error> {
        Decision::new(
            market_maker_address_bytes,
            slot_id_bytes,
            is_standard,
            is_scaled,
            question.to_string(),
            min,
            max,
        )
    }

    pub fn validate_complete_decision_slot_claim<T>(
        slots_db: &T,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error>
    where
        T: SlotValidationInterface,
    {
        let claim = tx.claim_decision_slot().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a decision slot claim transaction".to_string(),
            }
        })?;

        let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;
        Self::validate_slot_id_consistency(&slot_id, claim.slot_id_bytes)?;

        let market_maker_address =
            MarketValidator::validate_market_maker_authorization(tx)?;
        let decision = Self::validate_decision_structure(
            market_maker_address.0,
            claim.slot_id_bytes,
            claim.is_standard,
            claim.is_scaled,
            &claim.question,
            claim.min,
            claim.max,
        )?;

        let current_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        let current_height = override_height
            .or_else(|| slots_db.try_get_height(rotxn).ok().flatten());

        slots_db
            .validate_slot_claim(
                rotxn,
                slot_id,
                &decision,
                current_ts,
                current_height,
            )
            .map_err(|e| match e {
                Error::SlotNotAvailable { slot_id: _, reason } => {
                    Error::InvalidSlotId { reason }
                }
                Error::SlotAlreadyClaimed { slot_id: _ } => {
                    Error::InvalidSlotId {
                        reason: "Slot already claimed".to_string(),
                    }
                }
                other => other,
            })
    }
}

/// Market validation utilities for Bitcoin Hivemind prediction markets.
///
/// Provides centralized validation for all market operations following the
/// single-source-of-truth pattern established for slots and voting validation.
pub struct MarketValidator;

impl MarketValidator {
    /// Validate market maker/trader authorization from transaction inputs.
    ///
    /// Extracts and validates the address of the market participant (maker or trader)
    /// from the first UTXO in the transaction, following Bitcoin's standard pattern
    /// of using the first input to identify the transaction originator.
    ///
    /// # Arguments
    /// * `tx` - Filled transaction containing spent UTXOs
    ///
    /// # Returns
    /// * `Ok(Address)` - Validated market participant address
    /// * `Err(Error)` - Invalid transaction structure
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on market participant identification
    pub fn validate_market_maker_authorization(
        tx: &FilledTransaction,
    ) -> Result<Address, Error> {
        if tx.inputs().is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Transaction must have at least one input".to_string(),
            });
        }

        if tx.spent_utxos.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "No spent UTXOs found".to_string(),
            });
        }

        // Extract market maker address from first UTXO
        let first_utxo = &tx.spent_utxos[0];
        let market_maker_address = first_utxo.address;

        Ok(market_maker_address)
    }

    /// Validate complete market creation transaction.
    ///
    /// Ensures all Bitcoin Hivemind requirements are met for market creation:
    /// 1. Valid market type (independent or categorical)
    /// 2. At least one decision slot referenced
    /// 3. All decision slots exist and have decisions
    /// 4. Categorical markets use only binary decisions
    /// 5. LMSR parameters are valid (beta > 0, 0 <= fee <= 1)
    /// 6. Market maker is properly authorized
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `tx` - Filled transaction to validate
    /// * `_override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid market creation
    /// * `Err(Error)` - Invalid market with detailed reason
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on market creation and LMSR parameters
    pub fn validate_market_creation(
        state: &crate::state::State,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        let market_data = tx.create_market().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a market creation transaction".to_string(),
            }
        })?;

        // Validate market type per whitepaper specifications
        if market_data.market_type != "independent"
            && market_data.market_type != "categorical"
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Invalid market type '{}': must be 'independent' or 'categorical'",
                    market_data.market_type
                ),
            });
        }

        // Validate decision slots are provided
        if market_data.decision_slots.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: "Market must reference at least one decision slot"
                    .to_string(),
            });
        }

        // Validate slot IDs and ensure they exist with decisions
        let mut slot_ids = Vec::new();
        for slot_hex in &market_data.decision_slots {
            // Use common validation utility for slot ID parsing
            let slot_id = SlotValidator::parse_slot_id_from_hex(slot_hex)?;

            // Verify slot exists
            let slot = state
                .slots()
                .get_slot(rotxn, slot_id)?
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: format!(
                        "Referenced decision slot {} does not exist",
                        slot_hex
                    ),
                })?;

            // Verify slot has a decision
            if slot.decision.is_none() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Referenced slot {} has no decision claimed",
                        slot_hex
                    ),
                });
            }

            slot_ids.push(slot_id);
        }

        // Validate categorical market constraints per whitepaper
        if market_data.market_type == "categorical" {
            // All decisions must be binary for categorical markets
            for slot_id in &slot_ids {
                let slot = state.slots().get_slot(rotxn, *slot_id)?.unwrap();
                let decision = slot.decision.unwrap();
                if decision.is_scaled {
                    return Err(Error::InvalidTransaction {
                        reason:
                            "Categorical markets can only use binary decisions"
                                .to_string(),
                    });
                }
            }
        }

        // Validate LMSR beta parameter (liquidity sensitivity)
        let beta = market_data.b;
        if beta <= 0.0 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "LMSR beta parameter must be positive, got {}",
                    beta
                ),
            });
        }

        // Validate trading fee if specified
        if let Some(fee) = market_data.trading_fee {
            if fee < 0.0 || fee > 1.0 {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Trading fee must be between 0.0 and 1.0, got {}",
                        fee
                    ),
                });
            }
        }

        // Validate market maker authorization
        let _market_maker_address =
            Self::validate_market_maker_authorization(tx)?;

        Ok(())
    }

    /// Validate share purchase transaction (buy_shares).
    ///
    /// Ensures all Bitcoin Hivemind requirements are met for buying shares:
    /// 1. Market exists
    /// 2. Market is in Trading state
    /// 3. Outcome index is valid for market
    /// 4. Shares to buy amount is positive
    /// 5. Max cost is positive
    /// 6. LMSR trade cost doesn't exceed max cost
    /// 7. Trader is properly authorized
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `tx` - Filled transaction to validate
    /// * `_override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid share purchase
    /// * `Err(Error)` - Invalid trade with detailed reason
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on LMSR trading and market operations
    pub fn validate_buy_shares(
        state: &crate::state::State,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::markets::MarketState;

        let buy_data = tx.buy_shares().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a buy shares transaction".to_string(),
            }
        })?;

        // Validate market exists
        let market = state
            .markets()
            .get_market(rotxn, &buy_data.market_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "Market {:?} does not exist",
                    buy_data.market_id
                ),
            })?;

        // Validate market is in trading state
        if market.state() != MarketState::Trading {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market not in trading state (current: {:?})",
                    market.state()
                ),
            });
        }

        // Validate outcome index
        if buy_data.outcome_index as usize >= market.shares().len() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Outcome index {} out of range (market has {} outcomes)",
                    buy_data.outcome_index,
                    market.shares().len()
                ),
            });
        }

        // Validate shares amount is positive
        if buy_data.shares_to_buy <= 0.0 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Shares to buy must be positive, got {}",
                    buy_data.shares_to_buy
                ),
            });
        }

        // Validate max cost is positive
        if buy_data.max_cost <= 0 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Max cost must be positive, got {}",
                    buy_data.max_cost
                ),
            });
        }

        // Calculate new share quantities after the trade
        let mut new_shares = market.shares().clone();
        new_shares[buy_data.outcome_index as usize] += buy_data.shares_to_buy;

        // Validate LMSR constraints using centralized LmsrService
        // This ensures single source of truth for all LMSR calculations
        // per Bitcoin Hivemind whitepaper section on market maker algorithm
        use crate::math::lmsr::LmsrService;
        let trade_cost = LmsrService::calculate_update_cost(
            &market.shares(),
            &new_shares,
            market.b(),
        )
        .map_err(|e| Error::InvalidTransaction {
            reason: format!("LMSR trade cost calculation failed: {:?}", e),
        })?;

        // Validate trade cost doesn't exceed max cost
        if trade_cost > buy_data.max_cost as f64 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Trade cost {:.4} exceeds max cost {}",
                    trade_cost, buy_data.max_cost
                ),
            });
        }

        // Validate trader authorization
        let _trader_address = Self::validate_market_maker_authorization(tx)?;

        Ok(())
    }

    /// Validate share redemption transaction
    ///
    /// # Bitcoin Hivemind Compliance
    /// Implements redemption validation per whitepaper section 4:
    /// - Market must be in Resolved state (voting complete, outcomes determined)
    /// - User must hold sufficient shares in the specified outcome
    /// - Payout value calculated from final_prices array
    /// - Ensures proper economic settlement post-resolution
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `tx` - Filled transaction to validate
    ///
    /// # Returns
    /// * `Ok(())` - Redemption is valid
    /// * `Err(Error)` - Invalid redemption with detailed reason
    ///
    /// # Validation Rules
    /// 1. Transaction must contain RedeemShares data
    /// 2. Market must exist
    /// 3. Market must be in Resolved state (not Trading/Voting)
    /// 4. Outcome index must be valid for the market
    /// 5. User must hold sufficient shares for redemption
    /// 6. Shares amount must be positive
    ///
    /// # Bitcoin Hivemind Specification
    /// Per whitepaper section 4, share redemption occurs after consensus resolution:
    /// - Shares are redeemed for Bitcoin based on final_prices
    /// - Payout = shares_redeemed × final_price[outcome_index]
    /// - Only available in Resolved state to prevent premature redemption
    pub fn validate_share_redemption(
        state: &crate::state::State,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
    ) -> Result<(), Error> {
        use crate::state::markets::MarketState;

        let redeem_data = tx.redeem_shares().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a share redemption transaction".to_string(),
            }
        })?;

        // Validate market exists
        let market = state
            .markets()
            .get_market(rotxn, &redeem_data.market_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "Market {:?} does not exist",
                    redeem_data.market_id
                ),
            })?;

        // CRITICAL: Validate market is in Resolved state
        // This is the core whitepaper requirement - shares can only be redeemed after resolution
        if market.state() != MarketState::Resolved {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market not in Resolved state (current: {:?}). Shares can only be redeemed after market resolution.",
                    market.state()
                ),
            });
        }

        // Validate outcome index
        if redeem_data.outcome_index as usize >= market.shares().len() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Outcome index {} out of range (market has {} outcomes)",
                    redeem_data.outcome_index,
                    market.shares().len()
                ),
            });
        }

        // Validate shares amount is positive
        if redeem_data.shares_to_redeem <= 0.0 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Shares to redeem must be positive, got {}",
                    redeem_data.shares_to_redeem
                ),
            });
        }

        // Get trader address from transaction authorization
        let trader_address = Self::validate_market_maker_authorization(tx)?;

        // Validate user has sufficient shares
        let share_account = state
            .markets()
            .get_user_share_account(rotxn, &trader_address)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!(
                    "No share account found for address {}",
                    trader_address
                ),
            })?;

        let current_shares = share_account.get_position(
            &redeem_data.market_id,
            redeem_data.outcome_index,
        );

        if current_shares < redeem_data.shares_to_redeem {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Insufficient shares for redemption. Has {:.4} shares, trying to redeem {:.4}",
                    current_shares, redeem_data.shares_to_redeem
                ),
            });
        }

        // Calculate redemption value based on final_prices (for informational purposes)
        // The actual payout happens during block application
        let final_price = market.final_prices()[redeem_data.outcome_index as usize];
        let _redemption_value = redeem_data.shares_to_redeem * final_price;

        // Validate final_prices are populated (should always be true for Resolved state)
        let final_prices_sum: f64 = market.final_prices().iter().sum();
        if (final_prices_sum - 1.0).abs() > 1e-6 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market final_prices not properly normalized (sum = {:.6}). Market may not be properly resolved.",
                    final_prices_sum
                ),
            });
        }

        Ok(())
    }

    /// Validate batched market trades for atomic processing.
    ///
    /// Performs validation for a batch of market trades that will be applied atomically.
    /// This is used during block application to ensure all trades in a block are valid
    /// before applying any changes.
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `batched_trades` - Vector of batched market trades to validate
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - Vector of validated trade costs
    /// * `Err(Error)` - Invalid batch with detailed reason
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on atomic market operations
    pub fn validate_batched_trades(
        state: &crate::state::State,
        rotxn: &RoTxn,
        batched_trades: &[crate::state::markets::BatchedMarketTrade],
    ) -> Result<Vec<f64>, Error> {
        use crate::state::markets::{MarketError, MarketState};

        let mut trade_costs = Vec::with_capacity(batched_trades.len());

        for (trade_index, trade) in batched_trades.iter().enumerate() {
            // Validate market exists
            let market = state
                .markets()
                .get_market(rotxn, &trade.market_id)
                .map_err(|e| Error::DatabaseError(format!("Market access failed: {}", e)))?
                .ok_or_else(|| {
                    Error::Market(MarketError::MarketNotFound {
                        id: trade.market_id.clone(),
                    })
                })?;

            // Validate market state allows trading
            if market.state() != MarketState::Trading {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Market {:?} not in trading state (current: {:?})",
                        trade_index,
                        trade.market_id,
                        market.state()
                    ),
                });
            }

            // Validate outcome index is valid for market
            if trade.outcome_index as usize >= market.shares().len() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Outcome index {} out of range (market has {} outcomes)",
                        trade_index,
                        trade.outcome_index,
                        market.shares().len()
                    ),
                });
            }

            // Calculate and validate trade cost using market snapshot
            let cost = trade.calculate_trade_cost().map_err(|e| {
                Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Trade cost calculation failed: {}",
                        trade_index, e
                    ),
                }
            })?;

            // Validate cost against trader's maximum
            if cost > trade.max_cost as f64 {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch trade {}: Trade cost {:.4} exceeds max cost {}",
                        trade_index, cost, trade.max_cost
                    ),
                });
            }

            trade_costs.push(cost);
        }

        Ok(trade_costs)
    }

    /// Validate LMSR parameters for market integrity.
    ///
    /// Ensures LMSR parameters (beta and share quantities) are valid and within
    /// acceptable ranges to prevent numerical instability or overflow.
    ///
    /// # Arguments
    /// * `beta` - LMSR beta parameter (liquidity sensitivity)
    /// * `shares` - Current share quantities for all outcomes
    ///
    /// # Returns
    /// * `Ok(())` - Valid LMSR parameters
    /// * `Err(Error)` - Invalid parameters with detailed reason
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on LMSR market maker algorithm
    pub fn validate_lmsr_parameters(
        beta: f64,
        shares: &ndarray::Array1<f64>,
    ) -> Result<(), Error> {
        // Validate beta is positive
        if beta <= 0.0 || !beta.is_finite() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "LMSR beta must be positive and finite, got {}",
                    beta
                ),
            });
        }

        // Validate share quantities are non-negative and finite
        for (idx, &share_qty) in shares.iter().enumerate() {
            if share_qty < 0.0 || !share_qty.is_finite() {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Share quantity at index {} must be non-negative and finite, got {}",
                        idx, share_qty
                    ),
                });
            }
        }

        // Validate we have at least 2 outcomes for a meaningful market
        if shares.len() < 2 {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Market must have at least 2 outcomes, got {}",
                    shares.len()
                ),
            });
        }

        Ok(())
    }
}

/// D-function validation utilities for market constraints.
pub struct DFunctionValidator;

impl DFunctionValidator {
    /// Validate D-function constraints against market dimensions.
    ///
    /// Ensures that D-function references are within bounds and structurally valid.
    ///
    /// # Arguments
    /// * `d_function` - The D-function to validate
    /// * `max_decision_index` - Maximum valid decision index
    /// * `decision_slots` - Available decision slots
    ///
    /// # Returns
    /// * `Ok(())` - D-function is valid
    /// * `Err(MarketError)` - Invalid D-function with detailed reason
    pub fn validate_constraint(
        d_function: &DFunction,
        max_decision_index: usize,
        decision_slots: &[SlotId],
    ) -> Result<(), MarketError> {
        match d_function {
            DFunction::Decision(idx) => {
                if *idx >= max_decision_index {
                    return Err(MarketError::InvalidDimensions);
                }
                Ok(())
            }
            DFunction::Equals(func, value) => {
                // Validate the nested function
                Self::validate_constraint(
                    func,
                    max_decision_index,
                    decision_slots,
                )?;
                // For decision equality, ensure value is within valid range
                if let DFunction::Decision(_) = func.as_ref() {
                    // For binary decisions, valid values are 0, 1, 2 (No, Yes, Invalid)
                    // For scalar decisions, valid values are 0 to range+1
                    // For now, we use a simple check - values > 2 are invalid for most cases
                    if *value > 2 {
                        return Err(MarketError::InvalidOutcomeCombination);
                    }
                }
                Ok(())
            }
            DFunction::And(left, right) => {
                Self::validate_constraint(
                    left,
                    max_decision_index,
                    decision_slots,
                )?;
                Self::validate_constraint(
                    right,
                    max_decision_index,
                    decision_slots,
                )?;
                Ok(())
            }
            DFunction::Or(left, right) => {
                Self::validate_constraint(
                    left,
                    max_decision_index,
                    decision_slots,
                )?;
                Self::validate_constraint(
                    right,
                    max_decision_index,
                    decision_slots,
                )?;
                Ok(())
            }
            DFunction::Not(func) => {
                Self::validate_constraint(
                    func,
                    max_decision_index,
                    decision_slots,
                )?;
                Ok(())
            }
            DFunction::True => Ok(()),
        }
    }

    /// Check if this D-function creates valid categorical constraints.
    ///
    /// For categorical dimensions, exactly one option should be true, with all others false.
    /// This validates that the D-function properly enforces mutual exclusivity.
    ///
    /// # Arguments
    /// * `d_function` - The D-function to validate
    /// * `categorical_slots` - Slot indices that form a categorical dimension
    /// * `combo` - The outcome combination to validate
    /// * `decision_slots` - Available decision slots
    ///
    /// # Returns
    /// * `Ok(true)` - Valid categorical constraint (exactly one true)
    /// * `Ok(false)` - Invalid categorical constraint (zero or multiple true)
    /// * `Err(MarketError)` - Evaluation error
    pub fn validate_categorical_constraint(
        d_function: &DFunction,
        categorical_slots: &[usize],
        combo: &[usize],
        decision_slots: &[SlotId],
    ) -> Result<bool, MarketError> {
        let mut true_count = 0;

        for &slot_idx in categorical_slots {
            if slot_idx >= combo.len() {
                return Err(MarketError::InvalidDimensions);
            }

            // Check if this slot is true in the combination
            if combo[slot_idx] == 1 {
                true_count += 1;
            }
        }

        // For valid categorical constraint, exactly one should be true
        // Unless all are false (residual case)
        Ok(true_count <= 1)
    }

    /// Validate dimensional consistency across all D-functions.
    ///
    /// Ensures that D-functions properly represent the market's dimensional structure
    /// and that all outcome combinations are valid according to whitepaper specifications.
    ///
    /// # Arguments
    /// * `d_functions` - All D-functions for the market
    /// * `dimension_specs` - Market dimension specifications
    /// * `decision_slots` - Available decision slots
    /// * `all_combos` - All possible outcome combinations
    ///
    /// # Returns
    /// * `Ok(())` - All constraints are dimensionally consistent
    /// * `Err(MarketError)` - Inconsistent dimensional constraints
    pub fn validate_dimensional_consistency(
        d_functions: &[DFunction],
        dimension_specs: &[DimensionSpec],
        decision_slots: &[SlotId],
        all_combos: &[Vec<usize>],
    ) -> Result<(), MarketError> {
        if d_functions.len() != all_combos.len() {
            return Err(MarketError::InvalidDimensions);
        }

        // Validate each D-function against its corresponding combination
        for (df, combo) in d_functions.iter().zip(all_combos.iter()) {
            // Basic constraint validation
            Self::validate_constraint(
                df,
                decision_slots.len(),
                decision_slots,
            )?;

            // Evaluate the D-function against its own combination - should be true
            if !df.evaluate(combo, decision_slots)? {
                return Err(MarketError::InvalidOutcomeCombination);
            }

            // Check categorical constraints for each dimension
            let mut slot_idx = 0;
            for spec in dimension_specs {
                match spec {
                    DimensionSpec::Single(_) => {
                        slot_idx += 1;
                    }
                    DimensionSpec::Categorical(slots) => {
                        let categorical_indices: Vec<usize> =
                            (slot_idx..slot_idx + slots.len()).collect();
                        if !Self::validate_categorical_constraint(
                            df,
                            &categorical_indices,
                            combo,
                            decision_slots,
                        )? {
                            return Err(MarketError::InvalidOutcomeCombination);
                        }
                        slot_idx += slots.len();
                    }
                }
            }
        }

        // Additional validation: ensure D-functions are mutually exclusive
        // (each combination should satisfy exactly one D-function)
        for combo in all_combos {
            let mut satisfied_count = 0;
            for df in d_functions {
                if df.evaluate(combo, decision_slots)? {
                    satisfied_count += 1;
                }
            }
            if satisfied_count != 1 {
                return Err(MarketError::InvalidOutcomeCombination);
            }
        }

        Ok(())
    }
}

/// Market state transition validation.
pub struct MarketStateValidator;

impl MarketStateValidator {
    /// Validate market state transition according to Bitcoin Hivemind specification.
    ///
    /// Ensures state transitions follow valid paths per whitepaper requirements:
    /// - Trading -> Voting -> Resolved -> Ossified
    /// - Trading -> Cancelled (if no trades occurred)
    /// - Trading/Voting -> Invalid (governance action)
    ///
    /// # Arguments
    /// * `from_state` - Current market state
    /// * `to_state` - Proposed new state
    ///
    /// # Returns
    /// * `Ok(())` - Valid state transition
    /// * `Err(Error)` - Invalid transition with detailed reason
    pub fn validate_market_state_transition(
        from_state: MarketState,
        to_state: MarketState,
    ) -> Result<(), Error> {
        use MarketState::*;

        let valid_transition = match (from_state, to_state) {
            // No change is always valid
            (a, b) if a == b => true,

            // Valid forward transitions
            (Trading, Voting) => true,
            (Trading, Cancelled) => true, // Only if no trades occurred (checked elsewhere)
            (Trading, Invalid) => true,   // Governance action
            (Voting, Resolved) => true,
            (Voting, Invalid) => true, // Governance action
            (Resolved, Ossified) => true,

            // All other transitions are invalid
            _ => false,
        };

        if !valid_transition {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Invalid market state transition from {:?} to {:?}",
                    from_state, to_state
                ),
            });
        }

        Ok(())
    }

    /// Check if market has entered voting period based on slot states.
    ///
    /// A market enters voting when any of its decision slots enter voting.
    ///
    /// # Arguments
    /// * `market_slots` - Set of slot IDs used by this market
    /// * `slots_in_voting` - Set of slot IDs currently in voting
    ///
    /// # Returns
    /// * `true` if market should transition to voting
    /// * `false` otherwise
    pub fn should_enter_voting(
        market_slots: &HashSet<SlotId>,
        slots_in_voting: &HashSet<SlotId>,
    ) -> bool {
        market_slots
            .iter()
            .any(|slot_id| slots_in_voting.contains(slot_id))
    }

    /// Check if all decision slots are ossified.
    ///
    /// A market is ready for resolution when all its decision slots are ossified.
    ///
    /// # Arguments
    /// * `market_slots` - Set of slot IDs used by this market
    /// * `slot_states` - Map of slot IDs to their ossification status
    ///
    /// # Returns
    /// * `true` if all slots are ossified
    /// * `false` otherwise
    pub fn all_slots_ossified(
        market_slots: &HashSet<SlotId>,
        slot_states: &std::collections::HashMap<SlotId, bool>,
    ) -> bool {
        market_slots
            .iter()
            .all(|slot_id| slot_states.get(slot_id).copied().unwrap_or(false))
    }
}

/// Vote submission validation utilities.
///
/// This validator ensures vote submissions comply with Bitcoin Hivemind
/// specifications for the consensus mechanism.
pub struct VoteValidator;

impl VoteValidator {
    /// Validate complete vote submission transaction
    ///
    /// Ensures all Bitcoin Hivemind requirements are met:
    /// 1. Voter has Votecoin balance > 0 (voting rights)
    /// 2. Voting period exists and is active
    /// 3. Decision slot exists and is in voting period
    /// 4. Vote value is valid for decision type
    /// 5. No duplicate votes (one per voter per decision per period)
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `filled_tx` - Filled transaction to validate
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid vote submission
    /// * `Err(Error)` - Invalid vote with detailed reason
    pub fn validate_vote_submission(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{
            slots::SlotId,
            voting::types::{VoterId, VotingPeriodId},
        };

        let vote_data = filled_tx.submit_vote().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            }
        })?;

        // Extract and validate voter address
        let voter_address = filled_tx
            .spent_utxos
            .first()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Vote transaction must have inputs".to_string(),
            })?
            .address;

        // Validate voter has Votecoin balance
        let votecoin_balance = state.get_votecoin_balance(rotxn, &voter_address)?;
        if votecoin_balance == 0 {
            return Err(Error::InvalidTransaction {
                reason: "Voter has no Votecoin balance (voting rights)".to_string(),
            });
        }

        // Validate decision slot exists
        let decision_id = SlotId::from_bytes(vote_data.slot_id_bytes)?;
        let slot = state
            .slots()
            .get_slot(rotxn, decision_id)?
            .ok_or_else(|| Error::InvalidSlotId {
                reason: format!("Decision slot {:?} does not exist", decision_id),
            })?;

        // Validate slot has a decision
        let decision = slot.decision.ok_or_else(|| Error::InvalidSlotId {
            reason: format!("Slot {:?} has no decision", decision_id),
        })?;

        // Validate vote value is appropriate for decision type
        if decision.is_scaled {
            // Scalar decision - validate value is within range
            let min = decision.min.unwrap_or(0) as f64;
            let max = decision.max.unwrap_or(1) as f64;
            if !vote_data.vote_value.is_nan()
                && (vote_data.vote_value < min || vote_data.vote_value > max)
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Vote value {} outside valid range [{}, {}]",
                        vote_data.vote_value, min, max
                    ),
                });
            }
        } else {
            // Binary decision - validate value is 0.0, 1.0, or NaN (abstain)
            if !vote_data.vote_value.is_nan()
                && vote_data.vote_value != 0.0
                && vote_data.vote_value != 1.0
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Binary decision vote must be 0.0, 1.0, or NaN (abstain), got {}",
                        vote_data.vote_value
                    ),
                });
            }
        }

        // Validate voting period (implicit from slot state)
        // The slot must be in voting period for votes to be accepted
        let current_ts = state
            .try_get_mainchain_timestamp(rotxn)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            })?;
        let current_height = state.try_get_height(rotxn)?;

        if !state
            .slots()
            .is_slot_in_voting(decision_id, current_ts, current_height)
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision slot {:?} is not in voting period",
                    decision_id
                ),
            });
        }

        // Check for duplicate votes
        let voter_id = VoterId::from_address(&voter_address);
        let period_id = VotingPeriodId::new(vote_data.voting_period);

        if state
            .voting()
            .databases()
            .get_vote(rotxn, period_id, voter_id, decision_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: "Duplicate vote: voter already voted on this decision in this period"
                    .to_string(),
            });
        }

        Ok(())
    }

    /// Validate batch vote submission transaction
    ///
    /// Validates all votes in a batch submission according to Bitcoin Hivemind
    /// specifications.
    ///
    /// # Arguments
    /// * `state` - Blockchain state for validation queries
    /// * `rotxn` - Read-only transaction
    /// * `filled_tx` - Filled transaction containing batch votes
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - All votes in batch are valid
    /// * `Err(Error)` - Invalid batch with detailed reason
    pub fn validate_vote_batch(
        state: &crate::state::State,
        rotxn: &RoTxn,
        filled_tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::{
            slots::SlotId,
            voting::types::{VoterId, VotingPeriodId},
        };

        let batch_data = filled_tx.submit_vote_batch().ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "Not a vote batch submission transaction".to_string(),
            }
        })?;

        // Extract and validate voter address
        let voter_address = filled_tx
            .spent_utxos
            .first()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Vote batch transaction must have inputs".to_string(),
            })?
            .address;

        // Validate voter has Votecoin balance
        let votecoin_balance = state.get_votecoin_balance(rotxn, &voter_address)?;
        if votecoin_balance == 0 {
            return Err(Error::InvalidTransaction {
                reason: "Voter has no Votecoin balance (voting rights)".to_string(),
            });
        }

        let voter_id = VoterId::from_address(&voter_address);
        let period_id = VotingPeriodId::new(batch_data.voting_period);

        let current_ts = state
            .try_get_mainchain_timestamp(rotxn)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            })?;
        let current_height = state.try_get_height(rotxn)?;

        // Validate each vote in the batch
        for (idx, vote_item) in batch_data.votes.iter().enumerate() {
            let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

            // Validate decision slot exists
            let slot = state
                .slots()
                .get_slot(rotxn, decision_id)?
                .ok_or_else(|| Error::InvalidSlotId {
                    reason: format!(
                        "Vote batch item {}: Decision slot {:?} does not exist",
                        idx, decision_id
                    ),
                })?;

            // Validate slot has a decision
            let decision = slot.decision.ok_or_else(|| Error::InvalidSlotId {
                reason: format!(
                    "Vote batch item {}: Slot {:?} has no decision",
                    idx, decision_id
                ),
            })?;

            // Validate vote value is appropriate for decision type
            if decision.is_scaled {
                let min = decision.min.unwrap_or(0) as f64;
                let max = decision.max.unwrap_or(1) as f64;
                if !vote_item.vote_value.is_nan()
                    && (vote_item.vote_value < min || vote_item.vote_value > max)
                {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Vote batch item {}: Vote value {} outside valid range [{}, {}]",
                            idx, vote_item.vote_value, min, max
                        ),
                    });
                }
            } else {
                if !vote_item.vote_value.is_nan()
                    && vote_item.vote_value != 0.0
                    && vote_item.vote_value != 1.0
                {
                    return Err(Error::InvalidTransaction {
                        reason: format!(
                            "Vote batch item {}: Binary decision vote must be 0.0, 1.0, or NaN (abstain), got {}",
                            idx, vote_item.vote_value
                        ),
                    });
                }
            }

            // Validate slot is in voting period
            if !state
                .slots()
                .is_slot_in_voting(decision_id, current_ts, current_height)
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Vote batch item {}: Decision slot {:?} is not in voting period",
                        idx, decision_id
                    ),
                });
            }

            // Check for duplicate votes
            if state
                .voting()
                .databases()
                .get_vote(rotxn, period_id, voter_id, decision_id)?
                .is_some()
            {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Vote batch item {}: Duplicate vote on decision {:?}",
                        idx, decision_id
                    ),
                });
            }
        }

        Ok(())
    }
}

/// Voter registration and reputation validation
pub struct VoterValidator;

impl VoterValidator {
    pub fn validate_voter_not_registered(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_id: crate::state::voting::types::VoterId,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_voter_reputation(rotxn, voter_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: "Voter already registered".to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_voter_exists(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_id: crate::state::voting::types::VoterId,
    ) -> Result<(), Error> {
        if state
            .voting()
            .databases()
            .get_voter_reputation(rotxn, voter_id)?
            .is_none()
        {
            return Err(Error::InvalidTransaction {
                reason: "Voter not found".to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_reputation_update(
        state: &crate::state::State,
        rotxn: &RoTxn,
        voter_id: crate::state::voting::types::VoterId,
        period_id: crate::state::voting::types::VotingPeriodId,
    ) -> Result<(), Error> {
        Self::validate_voter_exists(state, rotxn, voter_id)?;

        let consensus_outcomes = state
            .voting()
            .databases()
            .get_consensus_outcomes_for_period(rotxn, period_id)?;

        if consensus_outcomes.is_empty() {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "No consensus outcomes found for period {:?}",
                    period_id
                ),
            });
        }

        let voter_votes = state
            .voting()
            .databases()
            .get_votes_by_voter(rotxn, voter_id)?;

        let has_votes_in_period = voter_votes
            .iter()
            .any(|(key, _)| key.period_id == period_id);

        if !has_votes_in_period {
            return Err(Error::InvalidTransaction {
                reason: "Voter has no votes in this period".to_string(),
            });
        }

        Ok(())
    }
}

/// Voting period lifecycle validation
pub struct PeriodValidator;

impl PeriodValidator {
    pub fn validate_period_can_close(
        period: &crate::state::voting::types::VotingPeriod,
        current_timestamp: u64,
    ) -> Result<(), Error> {
        use crate::state::voting::types::VotingPeriodStatus;

        if current_timestamp < period.end_timestamp {
            return Err(Error::InvalidTransaction {
                reason: "Cannot close period before end time".to_string(),
            });
        }

        if period.status != VotingPeriodStatus::Active {
            return Err(Error::InvalidTransaction {
                reason: format!("Period {:?} is not active", period.id),
            });
        }

        Ok(())
    }

    pub fn validate_period_is_active(
        period: &crate::state::voting::types::VotingPeriod,
        timestamp: u64,
    ) -> Result<(), Error> {
        use crate::state::voting::types::VotingPeriodStatus;

        if period.status != VotingPeriodStatus::Active {
            return Err(Error::InvalidTransaction {
                reason: format!("Period {:?} is not active for voting", period.id),
            });
        }

        if !period.is_active(timestamp) {
            return Err(Error::InvalidTransaction {
                reason: "Timestamp is outside period window".to_string(),
            });
        }

        Ok(())
    }

    pub fn validate_decision_in_period(
        period: &crate::state::voting::types::VotingPeriod,
        decision_id: SlotId,
    ) -> Result<(), Error> {
        if !period.decision_slots.contains(&decision_id) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision {:?} not available in period {:?}",
                    decision_id, period.id
                ),
            });
        }
        Ok(())
    }
}

/// Period calculation utilities.
pub struct PeriodCalculator;

impl PeriodCalculator {
    /// Convert block height to testing period.
    ///
    /// # Single Source of Truth
    /// Block heights are 0-indexed and directly map to periods:
    /// - Heights 0-9   → Period 1
    /// - Heights 10-19 → Period 2
    /// - Heights 20-29 → Period 3
    /// - Heights 30-39 → Period 4
    ///
    /// This ensures consistent period boundaries across all subsystems.
    #[inline(always)]
    pub const fn block_height_to_testing_period(
        block_height: u32,
        testing_blocks_per_period: u32,
    ) -> u32 {
        // Use const function for compile-time optimization when possible
        if testing_blocks_per_period == 0 {
            0
        } else {
            // Direct division: heights 0-9 = period 1, 10-19 = period 2, etc.
            (block_height / testing_blocks_per_period) + 1
        }
    }

    /// Convert L1 timestamp to production period.
    #[inline]
    pub fn timestamp_to_period(timestamp: u64) -> u32 {
        // Handle pre-genesis timestamps
        if timestamp < crate::types::BITCOIN_GENESIS_TIMESTAMP {
            return 0;
        }

        let elapsed_seconds =
            timestamp - crate::types::BITCOIN_GENESIS_TIMESTAMP;

        (elapsed_seconds / crate::types::SECONDS_PER_QUARTER) as u32
    }

    /// Get human-readable period name for timestamp with stack allocation
    ///
    /// Provides quarter-year formatting for period display using stack-based string building
    #[inline]
    pub fn period_to_name(period_index: u32) -> String {
        let year = 2009_u32.wrapping_add(period_index / 4);
        let quarter = (period_index % 4) + 1;

        // Use format! which is more efficient than string concatenation
        format!("Q{} {}", quarter, year)
    }

    /// Validate period index is within reasonable bounds (performance optimization)
    #[inline(always)]
    pub const fn is_valid_period_index(period_index: u32) -> bool {
        // Reasonable upper bound: ~1000 years from 2009
        period_index < 4000
    }
}
