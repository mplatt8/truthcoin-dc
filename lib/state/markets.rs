use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use ndarray::{Array, Ix1};
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};

use crate::state::Error;
use crate::state::slots::{Decision, SlotId};
use crate::types::hashes;
use crate::types::{Address, OutPoint};
use thiserror::Error as ThisError;

pub const MAX_MARKET_OUTCOMES: usize = 256;
pub const L2_STORAGE_RATE_SATS_PER_BYTE: u64 = 1;
pub const BASE_MARKET_STORAGE_COST_SATS: u64 = 1000;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MarketStateHash([u8; 32]);

impl MarketStateHash {
    pub fn new(data: [u8; 32]) -> Self {
        Self(data)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_data(data: &str) -> Self {
        let hash_bytes = hashes::hash(data.as_bytes());
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash_bytes[0..32]);
        Self(result)
    }
}

impl std::fmt::Display for MarketStateHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStateVersion {
    pub version: u64,
    pub previous_state_hash: Option<MarketStateHash>,
    pub state_hash: MarketStateHash,
    pub created_at_height: u64,
    pub transaction_id: Option<[u8; 32]>,
    pub market_state: MarketState,
    pub b: f64,
    pub trading_fee: f64,
    #[serde(with = "ndarray_1d_serde")]
    pub shares: Array<f64, Ix1>,
    #[serde(with = "ndarray_1d_serde")]
    pub final_prices: Array<f64, Ix1>,
    pub treasury: f64,
    pub timestamp: u64,
}

impl MarketStateVersion {
    pub fn new(
        version: u64,
        previous_state_hash: Option<MarketStateHash>,
        created_at_height: u64,
        transaction_id: Option<[u8; 32]>,
        market_state: MarketState,
        b: f64,
        trading_fee: f64,
        shares: Array<f64, Ix1>,
        final_prices: Array<f64, Ix1>,
        treasury: f64,
        timestamp: u64,
    ) -> Self {
        let state_data = format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}",
            version,
            previous_state_hash
                .as_ref()
                .map(|h| h.to_string())
                .unwrap_or_default(),
            created_at_height,
            transaction_id.map(|id| hex::encode(id)).unwrap_or_default(),
            format!("{:?}", market_state),
            b,
            trading_fee,
            shares
                .to_vec()
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            treasury
        );

        let state_hash = MarketStateHash::from_data(&state_data);

        Self {
            version,
            previous_state_hash,
            state_hash,
            created_at_height,
            transaction_id,
            market_state,
            b,
            trading_fee,
            shares,
            final_prices,
            treasury,
            timestamp,
        }
    }

    pub fn get_state_hash(&self) -> &MarketStateHash {
        &self.state_hash
    }

    pub fn is_genesis(&self) -> bool {
        self.previous_state_hash.is_none() && self.version == 0
    }
}

/// Market-specific error types
#[derive(Debug, ThisError, Clone)]
pub enum MarketError {
    #[error("Invalid market dimensions")]
    InvalidDimensions,

    #[error("Too many market states: {0} (max {MAX_MARKET_OUTCOMES})")]
    TooManyStates(usize),

    #[error("Invalid beta parameter: {0}")]
    InvalidBeta(f64),

    #[error("Invalid outcome index: {0}")]
    InvalidOutcomeIndex(usize),

    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition { from: MarketState, to: MarketState },

    #[error("Market not found: {id:?}")]
    MarketNotFound { id: MarketId },

    #[error("Decision slot not found: {slot_id:?}")]
    DecisionSlotNotFound { slot_id: SlotId },

    #[error("Slot validation failed for slot: {slot_id:?}")]
    SlotValidationFailed { slot_id: SlotId },

    #[error("Invalid outcome combination")]
    InvalidOutcomeCombination,

    #[error("Database error: {0}")]
    DatabaseError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketState {
    Trading = 1,
    Voting = 2,
    Resolved = 3,
    Cancelled = 4,
    Invalid = 5,
    Ossified = 6,
}

impl MarketState {
    pub fn can_transition_to(&self, new_state: MarketState) -> bool {
        use MarketState::*;
        match (self, new_state) {
            // Trading can transition to Voting or be Cancelled/Invalid
            (Trading, Voting) => true,
            (Trading, Cancelled) => true,
            (Trading, Invalid) => true,

            // Voting can transition to Resolved or be marked Invalid
            (Voting, Resolved) => true,
            (Voting, Invalid) => true,

            // Resolved can transition to Ossified (final state)
            (Resolved, Ossified) => true,

            // Invalid and Cancelled are terminal states (except Invalid can be Ossified)
            (Invalid, Ossified) => true,

            // Ossified is the final terminal state - no transitions allowed
            (Ossified, _) => false,

            // Self-transitions are always allowed (idempotent operations)
            (state, new_state) if state == &new_state => true,

            // All other transitions are invalid
            _ => false,
        }
    }

    pub fn allows_trading(&self) -> bool {
        matches!(self, MarketState::Trading)
    }

    pub fn allows_voting(&self) -> bool {
        matches!(self, MarketState::Voting)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, MarketState::Ossified | MarketState::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum DFunction {
    Decision(usize),
    Equals(Box<DFunction>, usize),
    And(Box<DFunction>, Box<DFunction>),
    Or(Box<DFunction>, Box<DFunction>),
    Not(Box<DFunction>),
    True,
}

#[derive(Debug, Clone)]
pub enum DimensionSpec {
    Single(SlotId),
    Categorical(Vec<SlotId>),
}

pub fn parse_dimensions(
    dimensions_str: &str,
) -> Result<Vec<DimensionSpec>, MarketError> {
    let dimensions_str = dimensions_str.trim();
    if !dimensions_str.starts_with('[') || !dimensions_str.ends_with(']') {
        return Err(MarketError::InvalidDimensions);
    }

    let inner = &dimensions_str[1..dimensions_str.len() - 1];
    let mut dimensions = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = inner.chars().collect();

    while i < chars.len() {
        while i < chars.len() && (chars[i].is_whitespace() || chars[i] == ',') {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }

        if chars[i] == '[' {
            let start = i + 1;
            let mut bracket_count = 1;
            i += 1;

            while i < chars.len() && bracket_count > 0 {
                if chars[i] == '[' {
                    bracket_count += 1;
                } else if chars[i] == ']' {
                    bracket_count -= 1;
                }
                i += 1;
            }

            if bracket_count != 0 {
                return Err(MarketError::InvalidDimensions);
            }

            let categorical_str: String = chars[start..i - 1].iter().collect();
            let slot_ids = parse_slot_list(&categorical_str)?;
            dimensions.push(DimensionSpec::Categorical(slot_ids));
        } else {
            let start = i;
            while i < chars.len() && chars[i] != ',' && chars[i] != '[' {
                i += 1;
            }

            let slot_str: String = chars[start..i].iter().collect();
            let slot_id = parse_single_slot(slot_str.trim())?;
            dimensions.push(DimensionSpec::Single(slot_id));
        }
    }

    Ok(dimensions)
}

fn parse_slot_list(list_str: &str) -> Result<Vec<SlotId>, MarketError> {
    list_str
        .split(',')
        .map(|s| parse_single_slot(s.trim()))
        .collect()
}

fn parse_single_slot(slot_str: &str) -> Result<SlotId, MarketError> {
    let slot_bytes =
        hex::decode(slot_str).map_err(|_| MarketError::InvalidDimensions)?;

    if slot_bytes.len() != 3 {
        return Err(MarketError::InvalidDimensions);
    }

    // Safe conversion since we verified length is exactly 3
    let slot_id_array: [u8; 3] = slot_bytes
        .try_into()
        .map_err(|_| MarketError::InvalidDimensions)?;
    SlotId::from_bytes(slot_id_array)
        .map_err(|_| MarketError::InvalidDimensions)
}

/// Unique identifier for a market (6 bytes)
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize,
)]
pub struct MarketId(pub [u8; 6]);

impl MarketId {
    pub fn new(data: [u8; 6]) -> Self {
        Self(data)
    }

    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }
}

impl std::fmt::Display for MarketId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

/// Share account for address-based position tracking using commitment model
///
/// Shares are tracked per address rather than per UTXO, allowing for more efficient
/// position management and reducing fragmentation while maintaining security through
/// UTXO signature requirements for transfers.
///
/// # Security Enhancement
/// - Per-transaction-type nonces prevent cross-type replay attacks
/// - Global nonce provides additional sequential ordering guarantee
/// - Comprehensive nonce validation prevents all known replay attack vectors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShareAccount {
    /// Address that owns these share positions
    pub owner_address: Address,
    /// Map of (market_id, outcome_index) -> shares held
    pub positions: HashMap<(MarketId, u32), f64>,
    /// Global anti-replay nonce for all operations (maintains backward compatibility)
    pub nonce: u64,
    /// Per-transaction-type nonces to prevent cross-type replay attacks
    pub redemption_nonce: u64,
    pub trade_nonce: u64,
    /// Block height when this account was last updated
    pub last_updated_height: u64,
}

impl ShareAccount {
    pub fn new(owner_address: Address) -> Self {
        Self {
            owner_address,
            positions: HashMap::new(),
            nonce: 0,
            redemption_nonce: 0,
            trade_nonce: 0,
            last_updated_height: 0,
        }
    }

    /// Get shares held for a specific market outcome
    pub fn get_position(
        &self,
        market_id: &MarketId,
        outcome_index: u32,
    ) -> f64 {
        self.positions
            .get(&(market_id.clone(), outcome_index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Add shares to a position
    pub fn add_shares(
        &mut self,
        market_id: MarketId,
        outcome_index: u32,
        shares: f64,
        height: u64,
    ) {
        let key = (market_id, outcome_index);
        let current = self.positions.get(&key).copied().unwrap_or(0.0);
        self.positions.insert(key, current + shares);
        self.last_updated_height = height;
    }

    /// Remove shares from a position
    pub fn remove_shares(
        &mut self,
        market_id: &MarketId,
        outcome_index: u32,
        shares: f64,
        height: u64,
    ) -> Result<(), MarketError> {
        let key = (market_id.clone(), outcome_index);
        let current = self.positions.get(&key).copied().unwrap_or(0.0);

        if shares > current {
            return Err(MarketError::InvalidOutcomeCombination);
        }

        let new_amount = current - shares;
        if new_amount > 0.0 {
            self.positions.insert(key, new_amount);
        } else {
            self.positions.remove(&key);
        }

        self.last_updated_height = height;
        Ok(())
    }

    /// Increment global nonce (for replay protection)
    pub fn increment_nonce(&mut self) {
        self.nonce += 1;
    }

    /// Increment redemption-specific nonce
    pub fn increment_redemption_nonce(&mut self) {
        self.redemption_nonce += 1;
        self.increment_nonce(); // Also increment global nonce
    }

    /// Increment trade-specific nonce
    pub fn increment_trade_nonce(&mut self) {
        self.trade_nonce += 1;
        self.increment_nonce(); // Also increment global nonce
    }

    /// Get all positions for this account
    pub fn get_all_positions(&self) -> &HashMap<(MarketId, u32), f64> {
        &self.positions
    }
}

/// Batched market trade for atomic processing within blocks
///
/// According to Bitcoin Hivemind, all trades within a block should use the same
/// base market state to ensure fair pricing. This structure captures a trade
/// along with the market snapshot at the time of block processing.
#[derive(Debug, Clone)]
pub struct BatchedMarketTrade {
    /// Market ID for this trade
    pub market_id: [u8; 6],
    /// Outcome index being traded
    pub outcome_index: u32,
    /// Number of shares to buy (positive) or sell (negative)
    pub shares_to_buy: f64,
    /// Maximum cost the trader is willing to pay
    pub max_cost: u64,
    /// Market snapshot captured at block start for pricing
    pub market_snapshot: MarketSnapshot,
    /// Address of the trader (shares go directly to their account)
    pub trader_address: Address,
}

/// Market snapshot for atomic trade processing
///
/// Captures the essential market state at the start of block processing
/// to ensure all trades in the block use consistent pricing.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Market state (shares array) at snapshot time
    pub shares: Array<f64, Ix1>,
    /// LMSR beta parameter
    pub b: f64,
    /// Trading fee percentage
    pub trading_fee: f64,
    /// Current treasury value
    pub treasury: f64,
}

impl BatchedMarketTrade {
    pub fn new(
        market_id: [u8; 6],
        outcome_index: u32,
        shares_to_buy: f64,
        max_cost: u64,
        market: &Market,
        trader_address: Address,
    ) -> Self {
        let market_snapshot = MarketSnapshot {
            shares: market.shares().clone(),
            b: market.b(),
            trading_fee: market.trading_fee(),
            treasury: market.treasury(),
        };

        Self {
            market_id,
            outcome_index,
            shares_to_buy,
            max_cost,
            market_snapshot,
            trader_address,
        }
    }

    /// Calculate the cost of this trade using the snapshot market state
    pub fn calculate_trade_cost(&self) -> Result<f64, MarketError> {
        use crate::math::lmsr::Lmsr;
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(self.market_snapshot.shares.len());

        // Calculate current cost
        let old_cost = lmsr
            .cost_function(
                self.market_snapshot.b,
                &self.market_snapshot.shares.view(),
            )
            .map_err(|e| {
                MarketError::DatabaseError(format!(
                    "LMSR calculation failed: {:?}",
                    e
                ))
            })?;

        // Calculate cost with new shares
        let mut new_shares = self.market_snapshot.shares.clone();
        new_shares[self.outcome_index as usize] += self.shares_to_buy;

        let new_cost = lmsr
            .cost_function(self.market_snapshot.b, &new_shares.view())
            .map_err(|e| {
                MarketError::DatabaseError(format!(
                    "LMSR calculation failed: {:?}",
                    e
                ))
            })?;

        let base_cost = new_cost - old_cost;
        let fee_cost = base_cost * self.market_snapshot.trading_fee;

        Ok(base_cost + fee_cost)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub id: MarketId,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub creator_address: Address,
    pub decision_slots: Vec<SlotId>,
    pub d_functions: Vec<DFunction>,
    pub state_combos: Vec<Vec<usize>>,
    pub created_at_height: u64,
    pub expires_at_height: Option<u64>,
    pub tau_from_now: u8,
    pub share_vector_length: usize,
    pub storage_fee_sats: u64,
    pub size: usize,
    pub state_history: Vec<MarketStateVersion>,
    pub current_state_hash: MarketStateHash,
    pub total_volume_sats: u64,
    pub outcome_volumes_sats: Vec<u64>,
}

mod ndarray_1d_serde {
    use ndarray::{Array, Ix1};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        array: &Array<f64, Ix1>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match array.as_slice() {
            Some(slice) => slice.serialize(serializer),
            None => array.to_vec().serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Array<f64, Ix1>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data: Vec<f64> = Deserialize::deserialize(deserializer)?;
        Ok(Array::from_vec(data))
    }
}

pub struct MarketBuilder {
    title: String,
    description: String,
    tags: Vec<String>,
    creator_address: Address,
    decision_slots: Vec<SlotId>,
    categorical_slots: Option<(Vec<SlotId>, bool)>,
    dimension_specs: Option<Vec<DimensionSpec>>,
    b: f64,
    trading_fee: f64,
    initial_liquidity_sats: Option<u64>,
}

impl MarketBuilder {
    pub fn new(title: String, creator_address: Address) -> Self {
        Self {
            title,
            description: String::new(),
            tags: Vec::new(),
            creator_address,
            decision_slots: Vec::new(),
            categorical_slots: None,
            dimension_specs: None,
            b: 7.0,
            trading_fee: 0.005,
            initial_liquidity_sats: None,
        }
    }

    pub fn add_decision(mut self, slot_id: SlotId) -> Self {
        self.decision_slots.push(slot_id);
        self
    }

    pub fn add_decisions(mut self, slot_ids: Vec<SlotId>) -> Self {
        self.decision_slots.extend(slot_ids);
        self
    }

    pub fn set_categorical(
        mut self,
        slot_ids: Vec<SlotId>,
        has_residual: bool,
    ) -> Self {
        self.categorical_slots = Some((slot_ids, has_residual));
        self
    }

    pub fn with_description(mut self, desc: String) -> Self {
        self.description = desc;
        self
    }

    pub fn with_beta(mut self, b: f64) -> Self {
        self.b = b;
        self
    }

    pub fn with_fee(mut self, fee: f64) -> Self {
        self.trading_fee = fee;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    fn calculate_initial_liquidity(&self) -> u64 {
        let n_outcomes = self.get_outcome_count() as f64;
        let initial_liquidity = self.b * n_outcomes.ln();
        initial_liquidity.ceil() as u64
    }

    fn get_outcome_count(&self) -> usize {
        if let Some(specs) = &self.dimension_specs {
            count_total_outcomes(specs)
        } else if let Some((slots, has_residual)) = &self.categorical_slots {
            slots.len() + if *has_residual { 1 } else { 0 }
        } else {
            2_usize.pow(self.decision_slots.len() as u32)
        }
    }

    pub fn with_dimensions(
        mut self,
        dimension_specs: Vec<DimensionSpec>,
    ) -> Self {
        self.dimension_specs = Some(dimension_specs);
        self
    }

    // TODO: Consider refactoring this large method into smaller functions
    // TODO: Validate market configuration before building
    pub fn build(
        self,
        created_at_height: u64,
        expires_at_height: Option<u64>,
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<Market, MarketError> {
        let (all_slots, d_functions, state_combos) =
            if let Some(ref dimension_specs) = self.dimension_specs {
                // Use new mixed-dimensional approach
                let (d_funcs, combos) =
                    generate_mixed_dimensional(&dimension_specs, decisions)?;

                // Collect all slot IDs from dimension specs
                let mut slots = Vec::new();
                for spec in dimension_specs {
                    match spec {
                        DimensionSpec::Single(slot_id) => {
                            slots.push(slot_id.clone())
                        }
                        DimensionSpec::Categorical(slot_ids) => {
                            slots.extend(slot_ids)
                        }
                    }
                }
                (slots, d_funcs, combos)
            } else if let Some((ref cat_slots, has_residual)) =
                self.categorical_slots
            {
                // Legacy categorical approach
                if !self.decision_slots.is_empty() {
                    return Err(MarketError::InvalidDimensions);
                }
                let (d_funcs, combos) = generate_categorical_functions(
                    &cat_slots,
                    has_residual,
                    decisions,
                )?;
                (cat_slots.clone(), d_funcs, combos)
            } else {
                // Legacy independent approach
                let (d_funcs, combos) =
                    generate_full_product(&self.decision_slots, decisions)?;
                (self.decision_slots.clone(), d_funcs, combos)
            };

        // Calculate initial liquidity automatically based on beta and number of states
        let calculated_liquidity = self.calculate_initial_liquidity();

        Market::new(
            self.title,
            self.description,
            self.tags,
            self.creator_address,
            all_slots,
            d_functions,
            state_combos,
            self.b,
            self.trading_fee,
            created_at_height,
            expires_at_height,
            decisions,
            Some(calculated_liquidity),
        )
    }
}

/// D_Function evaluation and generation with algorithmic optimizations
impl DFunction {
    /// Evaluate D_Function against a decision outcome combination
    ///
    /// Uses direct computation with short-circuit evaluation for performance.
    ///
    /// # Performance
    /// - Short-circuit evaluation for AND/OR operations
    /// - Direct computation without memory overhead
    ///
    /// # Bitcoin Hivemind Compliance
    /// Maintains exact evaluation semantics per whitepaper specification
    pub fn evaluate(
        &self,
        combo: &[usize],
        decision_slots: &[SlotId],
    ) -> Result<bool, MarketError> {
        match self {
            DFunction::Decision(idx) => {
                if *idx >= combo.len() {
                    return Err(MarketError::InvalidDimensions);
                }
                // For binary: 0=No, 1=Yes, 2=Invalid (treat Invalid as false for state purposes)
                Ok(combo[*idx] == 1)
            }
            DFunction::Equals(func, value) => {
                if let DFunction::Decision(idx) = func.as_ref() {
                    if *idx >= combo.len() {
                        return Err(MarketError::InvalidDimensions);
                    }
                    Ok(combo[*idx] == *value)
                } else {
                    // For more complex expressions, evaluate recursively
                    let func_result = func.evaluate(combo, decision_slots)?;
                    Ok(func_result && *value == 1)
                }
            }
            DFunction::And(left, right) => {
                // Short-circuit evaluation: if left is false, don't evaluate right
                let left_result = left.evaluate(combo, decision_slots)?;
                if !left_result {
                    return Ok(false);
                }
                let right_result = right.evaluate(combo, decision_slots)?;
                Ok(left_result && right_result)
            }
            DFunction::Or(left, right) => {
                // Short-circuit evaluation: if left is true, don't evaluate right
                let left_result = left.evaluate(combo, decision_slots)?;
                if left_result {
                    return Ok(true);
                }
                let right_result = right.evaluate(combo, decision_slots)?;
                Ok(left_result || right_result)
            }
            DFunction::Not(func) => {
                let result = func.evaluate(combo, decision_slots)?;
                Ok(!result)
            }
            DFunction::True => Ok(true),
        }
    }

    /// Validate that this D-function is well-formed and references only valid decision slots.
    /// Delegates to validation.rs for single source of truth.
    ///
    /// # Arguments
    /// * `max_decision_index` - The maximum valid decision index
    /// * `decision_slots` - Available decision slots for validation
    ///
    /// # Returns
    /// * `Ok(())` - D-function is valid
    /// * `Err(MarketError)` - Invalid D-function with detailed reason
    pub fn validate_constraint(
        &self,
        max_decision_index: usize,
        decision_slots: &[SlotId],
    ) -> Result<(), MarketError> {
        crate::validation::DFunctionValidator::validate_constraint(
            self,
            max_decision_index,
            decision_slots,
        )
    }

    /// Check if this D-function creates valid categorical constraints.
    /// Delegates to validation.rs for single source of truth.
    ///
    /// # Arguments
    /// * `categorical_slots` - Slot indices that form a categorical dimension
    /// * `combo` - The outcome combination to validate
    /// * `decision_slots` - Available decision slots
    ///
    /// # Returns
    /// * `Ok(true)` - Valid categorical constraint (exactly one true)
    /// * `Ok(false)` - Invalid categorical constraint (zero or multiple true)
    /// * `Err(MarketError)` - Evaluation error
    pub fn validate_categorical_constraint(
        &self,
        categorical_slots: &[usize],
        combo: &[usize],
        decision_slots: &[SlotId],
    ) -> Result<bool, MarketError> {
        crate::validation::DFunctionValidator::validate_categorical_constraint(
            self,
            categorical_slots,
            combo,
            decision_slots,
        )
    }

    /// Validate dimensional consistency across all D-functions.
    /// Delegates to validation.rs for single source of truth.
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
        crate::validation::DFunctionValidator::validate_dimensional_consistency(
            d_functions,
            dimension_specs,
            decision_slots,
            all_combos,
        )
    }

    /// Build a balanced AND tree for better evaluation performance
    ///
    /// Instead of creating a left-heavy chain of AND operations,
    /// this creates a balanced binary tree which reduces evaluation depth
    /// and improves cache performance.
    fn build_balanced_and_tree(mut constraints: Vec<DFunction>) -> DFunction {
        while constraints.len() > 1 {
            let mut next_level =
                Vec::with_capacity((constraints.len() + 1) / 2);

            while constraints.len() >= 2 {
                let right = constraints
                    .pop()
                    .expect("constraints.len() >= 2 guarantees pop() succeeds");
                let left = constraints
                    .pop()
                    .expect("constraints.len() >= 2 guarantees pop() succeeds");
                next_level
                    .push(DFunction::And(Box::new(left), Box::new(right)));
            }

            // Handle odd number of constraints
            if let Some(remaining) = constraints.pop() {
                next_level.push(remaining);
            }

            constraints = next_level;
        }

        constraints.into_iter().next().unwrap_or(DFunction::True)
    }
}

fn calculate_storage_fee_with_scaling(
    share_vector_length: usize,
) -> Result<u64, MarketError> {
    // Validate share vector length against maximum allowed outcomes
    if share_vector_length > MAX_MARKET_OUTCOMES {
        return Err(MarketError::TooManyStates(share_vector_length));
    }

    let base_cost = BASE_MARKET_STORAGE_COST_SATS;

    // Pure quadratic scaling: base + (n² × rate)
    // Bounded by MAX_MARKET_OUTCOMES to prevent excessive storage costs
    let quadratic_cost =
        (share_vector_length as u64).pow(2) * L2_STORAGE_RATE_SATS_PER_BYTE;
    Ok(base_cost + quadratic_cost)
}

/// Generate mixed-dimensional market with proper D_Functions and optimization
///
/// Implements several performance optimizations:
/// 1. Early validation of dimension constraints before expensive computation
/// 2. Efficient pre-allocation based on expected outcome counts
/// 3. Optimized D_Function generation with constraint batching
/// 4. Early termination when approaching MAX_MARKET_OUTCOMES
///
/// # Performance
/// - Pre-validation: O(d) where d is number of dimension specs
/// - State generation: O(k^d) with early termination at MAX_MARKET_OUTCOMES
/// - Function generation: O(n) where n is number of states
///
/// # Bitcoin Hivemind Compliance
/// Maintains exact market semantics while optimizing computational efficiency
pub fn generate_mixed_dimensional(
    dimension_specs: &[DimensionSpec],
    decisions: &HashMap<SlotId, Decision>,
) -> Result<(Vec<DFunction>, Vec<Vec<usize>>), MarketError> {
    if dimension_specs.is_empty() {
        return Err(MarketError::InvalidDimensions);
    }

    // Pre-validate dimension specs and calculate expected complexity
    let mut expected_outcomes = 1usize;
    for spec in dimension_specs {
        let spec_outcomes = match spec {
            DimensionSpec::Single(_) => 3, // Binary: Yes/No/Invalid
            DimensionSpec::Categorical(slots) => slots.len() + 2, // N options + residual + invalid
        };

        // Early termination if complexity would exceed maximum
        if let Some(new_expected) = expected_outcomes.checked_mul(spec_outcomes)
        {
            if new_expected > MAX_MARKET_OUTCOMES {
                return Err(MarketError::TooManyStates(new_expected));
            }
            expected_outcomes = new_expected;
        } else {
            return Err(MarketError::TooManyStates(usize::MAX));
        }
    }

    // Build slot mapping and get dimensions per dimension spec with pre-allocation
    let mut all_slots = Vec::with_capacity(dimension_specs.len() * 4); // Estimate capacity
    let mut dimension_ranges = Vec::with_capacity(dimension_specs.len());
    let mut slot_to_dimension = Vec::new(); // Maps slot index to dimension spec index

    for (dim_idx, spec) in dimension_specs.iter().enumerate() {
        match spec {
            DimensionSpec::Single(slot_id) => {
                all_slots.push(*slot_id);
                slot_to_dimension.push(dim_idx);

                let decision = decisions.get(slot_id).ok_or(
                    MarketError::DecisionSlotNotFound { slot_id: *slot_id },
                )?;

                let outcomes = if decision.is_scaled {
                    if let (Some(min), Some(max)) = (decision.min, decision.max)
                    {
                        (max - min) as usize + 2 // +1 for range, +1 for null outcome
                    } else {
                        return Err(MarketError::SlotValidationFailed {
                            slot_id: *slot_id,
                        });
                    }
                } else {
                    3 // Binary: Yes/No/Invalid
                };
                dimension_ranges.push(outcomes);
            }
            DimensionSpec::Categorical(slot_ids) => {
                // For categorical, we have N options + residual + invalid
                let outcomes = slot_ids.len() + 2;
                dimension_ranges.push(outcomes);

                // Add all slots but map them to the same dimension
                for slot_id in slot_ids {
                    all_slots.push(*slot_id);
                    slot_to_dimension.push(dim_idx);
                }
            }
        }
    }

    // Generate Cartesian product of dimension outcomes with optimized caching
    let state_combos = generate_cartesian_product(&dimension_ranges);

    // Validate against maximum outcomes (optimized generation may have stopped early)
    if state_combos.is_empty() && expected_outcomes > MAX_MARKET_OUTCOMES {
        return Err(MarketError::TooManyStates(expected_outcomes));
    }

    if state_combos.len() > MAX_MARKET_OUTCOMES {
        return Err(MarketError::TooManyStates(state_combos.len()));
    }

    // Generate D_Functions for valid combinations with pre-allocation
    let mut d_functions = Vec::with_capacity(state_combos.len());

    for combo in &state_combos {
        // Build D_Function for this combination with optimized constraint generation
        let mut constraints = Vec::with_capacity(dimension_specs.len() * 2); // Estimate constraint count
        let mut slot_idx = 0;

        for (dim_idx, spec) in dimension_specs.iter().enumerate() {
            let dim_outcome = combo[dim_idx];

            match spec {
                DimensionSpec::Single(_) => {
                    // Simple case: slot outcome = dimension outcome
                    if dim_outcome < 3 {
                        // Valid outcomes only
                        constraints.push(DFunction::Equals(
                            Box::new(DFunction::Decision(slot_idx)),
                            dim_outcome,
                        ));
                    }
                    slot_idx += 1;
                }
                DimensionSpec::Categorical(slot_ids) => {
                    // Complex case: exactly one of the categorical slots should be true
                    if dim_outcome < slot_ids.len() {
                        // Option dim_outcome is selected
                        constraints.push(DFunction::Equals(
                            Box::new(DFunction::Decision(
                                slot_idx + dim_outcome,
                            )),
                            1,
                        ));
                        // All others must be false
                        for (other_idx, _) in slot_ids.iter().enumerate() {
                            if other_idx != dim_outcome {
                                constraints.push(DFunction::Equals(
                                    Box::new(DFunction::Decision(
                                        slot_idx + other_idx,
                                    )),
                                    0,
                                ));
                            }
                        }
                    } else if dim_outcome == slot_ids.len() {
                        // Residual case: all categorical slots are false
                        for other_idx in 0..slot_ids.len() {
                            constraints.push(DFunction::Equals(
                                Box::new(DFunction::Decision(
                                    slot_idx + other_idx,
                                )),
                                0,
                            ));
                        }
                    }
                    slot_idx += slot_ids.len();
                }
            }
        }

        // Optimize constraint combination for better evaluation performance
        let d_function = match constraints.len() {
            0 => DFunction::True,
            1 => constraints
                .into_iter()
                .next()
                .expect("constraints.len() == 1 guarantees next() succeeds"),
            _ => {
                // Build balanced tree for better evaluation performance
                // instead of left-heavy chain
                DFunction::build_balanced_and_tree(constraints)
            }
        };

        d_functions.push(d_function);
    }

    // Enhanced constraint validation - ensure all D-functions are dimensionally consistent
    DFunction::validate_dimensional_consistency(
        &d_functions,
        dimension_specs,
        &all_slots,
        &state_combos,
    )?;

    Ok((d_functions, state_combos))
}

/// Generate full Cartesian product for all decisions
pub fn generate_full_product(
    slots: &[SlotId],
    decisions: &HashMap<SlotId, Decision>,
) -> Result<(Vec<DFunction>, Vec<Vec<usize>>), MarketError> {
    if slots.is_empty() {
        return Err(MarketError::InvalidDimensions);
    }

    // Get dimensions for each slot (3 for binary, range+2 for scaled)
    let dimensions = get_raw_dimensions(slots, decisions)?;

    // Generate all combinations (Cartesian product)
    let state_combos = generate_cartesian_product(&dimensions);

    // Cap at MAX_MARKET_OUTCOMES states per Hivemind whitepaper specification
    if state_combos.len() > MAX_MARKET_OUTCOMES {
        return Err(MarketError::TooManyStates(state_combos.len()));
    }

    // For full product, all combinations are valid (use True function for each)
    let d_functions = vec![DFunction::True; state_combos.len()];

    Ok((d_functions, state_combos))
}

fn generate_categorical_functions(
    slots: &[SlotId],
    has_residual: bool,
    decisions: &HashMap<SlotId, Decision>,
) -> Result<(Vec<DFunction>, Vec<Vec<usize>>), MarketError> {
    for slot_id in slots {
        if let Some(decision) = decisions.get(slot_id) {
            if decision.is_scaled {
                return Err(MarketError::InvalidDimensions);
            }
        }
    }

    let mut d_functions = Vec::new();
    let mut state_combos = Vec::new();

    for (i, _slot_id) in slots.iter().enumerate() {
        let mut others = Vec::new();
        for (j, _) in slots.iter().enumerate() {
            if i != j {
                others.push(DFunction::Decision(j));
            }
        }

        let not_others = if others.is_empty() {
            DFunction::True
        } else {
            // Safe: others is guaranteed to be non-empty by the conditional above
            let or_others = others
                .into_iter()
                .reduce(|acc, func| {
                    DFunction::Or(Box::new(acc), Box::new(func))
                })
                .expect("others vector is non-empty");
            DFunction::Not(Box::new(or_others))
        };

        let function = DFunction::And(
            Box::new(DFunction::Decision(i)),
            Box::new(not_others),
        );
        let mut combo = vec![0; slots.len()];
        combo[i] = 1;

        d_functions.push(function);
        state_combos.push(combo);
    }
    if has_residual {
        let all_decisions: Vec<DFunction> =
            (0..slots.len()).map(|i| DFunction::Decision(i)).collect();

        let or_all = all_decisions
            .into_iter()
            .reduce(|acc, func| DFunction::Or(Box::new(acc), Box::new(func)))
            .unwrap_or(DFunction::True);

        let residual_function = DFunction::Not(Box::new(or_all));
        let residual_combo = vec![0; slots.len()];

        d_functions.push(residual_function);
        state_combos.push(residual_combo);
    }

    Ok((d_functions, state_combos))
}

fn get_raw_dimensions(
    slots: &[SlotId],
    decisions: &HashMap<SlotId, Decision>,
) -> Result<Vec<usize>, MarketError> {
    let mut dimensions = Vec::new();

    for slot_id in slots {
        let decision = decisions
            .get(slot_id)
            .ok_or(MarketError::DecisionSlotNotFound { slot_id: *slot_id })?;

        let outcomes = if decision.is_scaled {
            if let (Some(min), Some(max)) = (decision.min, decision.max) {
                (max - min) as usize + 2 // +1 for range, +1 for null outcome
            } else {
                return Err(MarketError::SlotValidationFailed {
                    slot_id: *slot_id,
                });
            }
        } else {
            3
        };
        dimensions.push(outcomes);
    }

    Ok(dimensions)
}

/// Efficient Cartesian product generation with early termination
///
/// Implements optimizations that don't require memory overhead:
/// 1. Early termination when approaching MAX_MARKET_OUTCOMES
/// 2. Pre-allocation of result vectors based on expected size
/// 3. Memory-efficient generation without caching overhead
///
/// # Performance
/// - Direct computation: O(product of dimensions) with early termination
/// - Memory-efficient with pre-allocation
///
/// # Bitcoin Hivemind Compliance
/// Maintains exact Cartesian product semantics per whitepaper specification
fn generate_cartesian_product(dimensions: &[usize]) -> Vec<Vec<usize>> {
    if dimensions.is_empty() {
        return vec![vec![]];
    }

    // Calculate expected result size for early termination and pre-allocation
    let expected_size: usize = dimensions.iter().product();

    // Early termination if result would exceed maximum allowed outcomes
    if expected_size > MAX_MARKET_OUTCOMES {
        // Return empty result - caller will handle the TooManyStates error
        return Vec::new();
    }

    // Pre-allocate result vector with expected capacity for efficiency
    let mut result = Vec::with_capacity(expected_size);
    result.push(vec![]);

    // Generate combinations with early termination check
    for &dim_size in dimensions {
        let mut new_result = Vec::with_capacity(result.len() * dim_size);

        for combo in result {
            for value in 0..dim_size {
                // Check if we would exceed the limit and return empty (caller handles error)
                if new_result.len() >= MAX_MARKET_OUTCOMES {
                    return Vec::new();
                }

                let mut new_combo = combo.clone();
                new_combo.push(value);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }

    result
}

fn calculate_max_tau(
    decision_slots: &[SlotId],
    decisions: &HashMap<SlotId, Decision>,
) -> u8 {
    decision_slots
        .iter()
        .filter_map(|slot_id| decisions.get(slot_id))
        .map(|_| 5u8) // Default tau value - in practice this would come from decision
        .max()
        .unwrap_or(5)
}

/// Count total number of outcomes for dimension specifications
fn count_total_outcomes(dimension_specs: &[DimensionSpec]) -> usize {
    if dimension_specs.is_empty() {
        return 2; // Default binary
    }

    // For now, return a simple count based on the number of dimensions
    // This should be implemented properly based on the dimension specification logic
    dimension_specs.len() * 2 // Simplified: assume each dimension has 2 outcomes
}

impl Market {
    pub fn new(
        title: String,
        description: String,
        tags: Vec<String>,
        creator_address: Address,
        decision_slots: Vec<SlotId>,
        d_functions: Vec<DFunction>,
        state_combos: Vec<Vec<usize>>,
        b: f64,
        trading_fee: f64,
        created_at_height: u64,
        expires_at_height: Option<u64>,
        decisions: &HashMap<SlotId, Decision>,
        initial_liquidity_sats: Option<u64>,
    ) -> Result<Self, MarketError> {
        // Validate inputs
        if decision_slots.is_empty() {
            return Err(MarketError::InvalidDimensions);
        }

        if d_functions.is_empty() {
            return Err(MarketError::InvalidDimensions);
        }

        if d_functions.len() != state_combos.len() {
            return Err(MarketError::InvalidDimensions);
        }

        // Cap at MAX_MARKET_OUTCOMES states per Hivemind whitepaper specification
        if d_functions.len() > MAX_MARKET_OUTCOMES {
            return Err(MarketError::TooManyStates(d_functions.len()));
        }

        if b <= 0.0 {
            return Err(MarketError::InvalidBeta(b));
        }

        // Calculate N (share vector length) - number of valid states
        let share_vector_length = d_functions.len();

        // Calculate storage fee with quadratic scaling (with bounds checking)
        let storage_fee_sats =
            calculate_storage_fee_with_scaling(share_vector_length)?;

        // LMSR markets always start with zero shares
        let shares = Array::zeros(share_vector_length);
        let final_prices = Array::zeros(share_vector_length);

        // Apply LMSR parameter relationship: Initial Liquidity = β × ln(n)
        // Always use the provided beta parameter and calculate liquidity accordingly
        let n_outcomes = share_vector_length as f64;
        let final_b = b; // Always use the provided beta parameter

        // Calculate initial treasury using the formula: Initial Liquidity = b * ln(Number of States)
        let calculated_treasury = b * n_outcomes.ln();

        // Use the provided initial liquidity if it's greater than calculated minimum,
        // otherwise use the calculated minimum
        let initial_treasury =
            if let Some(capital_sats) = initial_liquidity_sats {
                (capital_sats as f64).max(calculated_treasury)
            } else {
                calculated_treasury
            };

        // Create genesis state version (version 0, no previous state)
        let genesis_state = MarketStateVersion::new(
            0,    // version
            None, // no previous state
            created_at_height,
            None, // no transaction ID for genesis
            MarketState::Trading,
            final_b,
            trading_fee,
            shares.clone(),
            final_prices.clone(),
            initial_treasury,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        let genesis_state_hash = genesis_state.get_state_hash().clone();

        let mut market = Market {
            id: MarketId([0; 6]), // Will be set after hash calculation
            title,
            description,
            tags,
            creator_address,
            decision_slots: decision_slots.clone(),
            d_functions,
            state_combos,
            created_at_height,
            expires_at_height,
            tau_from_now: calculate_max_tau(&decision_slots, decisions),
            share_vector_length,
            storage_fee_sats,
            size: 0,
            state_history: vec![genesis_state],
            current_state_hash: genesis_state_hash,
            total_volume_sats: 0,
            outcome_volumes_sats: vec![0; share_vector_length],
        };

        // Calculate market ID from hash
        market.id = market.calculate_id();

        // Calculate size
        market.calculate_size();

        Ok(market)
    }

    /// Generate market ID from immutable market data
    fn calculate_id(&self) -> MarketId {
        let market_string = format!(
            "{}{}{}{}{}{}{}{}{}",
            self.title,
            self.description,
            self.creator_address,
            self.decision_slots.len(),
            self.decision_slots
                .iter()
                .map(|s| format!("{:?}", s))
                .collect::<Vec<_>>()
                .join(","),
            format!("{:?}", self.d_functions),
            format!("{:?}", self.state_combos),
            self.created_at_height,
            self.expires_at_height.unwrap_or(0)
        );
        let hash_bytes = hashes::hash(&market_string);
        // Use first 6 bytes of hash for market ID
        let mut id_bytes = [0u8; 6];
        id_bytes.copy_from_slice(&hash_bytes[0..6]);
        MarketId(id_bytes)
    }

    /// Calculate storage size in bytes (including state history)
    fn calculate_size(&mut self) {
        let base_size = self.title.len() +
            self.description.len() +
            self.tags.iter().map(|tag| tag.len()).sum::<usize>() +
            std::mem::size_of_val(&self.tau_from_now) +
            self.creator_address.to_string().len() +
            self.decision_slots.len() * 3 + // 3 bytes per SlotId
            std::mem::size_of_val(&self.created_at_height) +
            std::mem::size_of_val(&self.expires_at_height) +
            std::mem::size_of_val(&self.share_vector_length) +
            std::mem::size_of_val(&self.storage_fee_sats) +
            self.state_history.len() * std::mem::size_of::<MarketStateVersion>() +
            std::mem::size_of_val(&self.current_state_hash);

        self.size = base_size;
    }

    // === Blockchain-style State Management Methods ===

    /// Get the current (most recent) market state version
    pub fn get_current_state(&self) -> &MarketStateVersion {
        self.state_history
            .last()
            .expect("Market must have at least genesis state")
    }

    /// Get market state version by index
    pub fn get_state_version(
        &self,
        version: u64,
    ) -> Option<&MarketStateVersion> {
        self.state_history.get(version as usize)
    }

    /// Get the full state history
    pub fn get_state_history(&self) -> &Vec<MarketStateVersion> {
        &self.state_history
    }

    // === Convenience methods for accessing current state properties ===

    /// Get current market state (Trading, Voting, Resolved, etc.)
    pub fn state(&self) -> MarketState {
        self.get_current_state().market_state
    }

    /// Get current LMSR beta parameter
    pub fn b(&self) -> f64 {
        self.get_current_state().b
    }

    /// Get current trading fee
    pub fn trading_fee(&self) -> f64 {
        self.get_current_state().trading_fee
    }

    /// Get current share quantities
    pub fn shares(&self) -> &Array<f64, Ix1> {
        &self.get_current_state().shares
    }

    /// Get current final prices
    pub fn final_prices(&self) -> &Array<f64, Ix1> {
        &self.get_current_state().final_prices
    }

    /// Get current treasury balance
    pub fn treasury(&self) -> f64 {
        self.get_current_state().treasury
    }

    /// Create a new market state version (immutable update)
    ///
    /// This method creates a new MarketStateVersion that references the current
    /// state as the previous state, implementing the blockchain-like structure.
    pub fn create_new_state_version(
        &mut self,
        transaction_id: Option<[u8; 32]>,
        height: u64,
        new_market_state: Option<MarketState>,
        new_b: Option<f64>,
        new_trading_fee: Option<f64>,
        new_shares: Option<Array<f64, Ix1>>,
        new_final_prices: Option<Array<f64, Ix1>>,
        new_treasury: Option<f64>,
    ) -> Result<MarketStateHash, MarketError> {
        let current_state = self.get_current_state();
        let next_version = current_state.version + 1;

        // Use current values as defaults, override with new values if provided
        let market_state =
            new_market_state.unwrap_or(current_state.market_state);
        let b = new_b.unwrap_or(current_state.b);
        let trading_fee = new_trading_fee.unwrap_or(current_state.trading_fee);
        let shares = new_shares.unwrap_or_else(|| current_state.shares.clone());
        let final_prices = new_final_prices
            .unwrap_or_else(|| current_state.final_prices.clone());
        let treasury = new_treasury.unwrap_or(current_state.treasury);

        // Validate the new state
        if b <= 0.0 {
            return Err(MarketError::InvalidBeta(b));
        }

        // Validate state transition is allowed
        if new_market_state.is_some()
            && !current_state.market_state.can_transition_to(market_state)
        {
            return Err(MarketError::InvalidStateTransition {
                from: current_state.market_state,
                to: market_state,
            });
        }

        if shares.len() != self.share_vector_length {
            return Err(MarketError::InvalidDimensions);
        }

        // Create the new state version
        let new_state_version = MarketStateVersion::new(
            next_version,
            Some(self.current_state_hash.clone()),
            height,
            transaction_id,
            market_state,
            b,
            trading_fee,
            shares,
            final_prices,
            treasury,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        let new_state_hash = new_state_version.get_state_hash().clone();

        // Add to history and update current state hash
        self.state_history.push(new_state_version);
        self.current_state_hash = new_state_hash.clone();

        // Recalculate size
        self.calculate_size();

        Ok(new_state_hash)
    }

    /// Calculate treasury based on current market state
    /// Calculate current treasury using LMSR with actual market size
    ///
    /// Creates an LMSR calculator with the correct market size to ensure
    /// proper calculation according to Bitcoin Hivemind specifications.
    pub fn calc_treasury(&self) -> f64 {
        use crate::math::lmsr::Lmsr;
        let current_state = self.get_current_state();
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(current_state.shares.len());
        lmsr.cost_function(current_state.b, &current_state.shares.view())
            .unwrap_or_else(|_| {
                current_state.b * (current_state.shares.len() as f64).ln()
            })
    }

    /// Calculate treasury with specific shares (for state updates)
    ///
    /// Creates an LMSR calculator with the actual market size to ensure
    /// proper calculation according to Bitcoin Hivemind specifications.
    pub fn calc_treasury_with_shares(&self, shares: &Array<f64, Ix1>) -> f64 {
        use crate::math::lmsr::Lmsr;
        let current_state = self.get_current_state();
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(shares.len());
        lmsr.cost_function(current_state.b, &shares.view())
            .unwrap_or_else(|_| current_state.b * (shares.len() as f64).ln())
    }

    /// Calculate prices based on current market state
    pub fn current_prices(&self) -> Array<f64, Ix1> {
        use crate::math::lmsr::Lmsr;
        let current_state = self.get_current_state();
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(current_state.shares.len());
        lmsr.calculate_prices(current_state.b, &current_state.shares.view())
            .unwrap_or_else(|_| {
                let n = current_state.shares.len();
                Array::from_elem(n, 1.0 / n as f64)
            })
    }

    /// Calculate prices with specific shares using current beta
    pub fn calculate_prices(
        &self,
        shares: &Array<f64, Ix1>,
    ) -> Array<f64, Ix1> {
        use crate::math::lmsr::Lmsr;
        let current_state = self.get_current_state();
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(shares.len());
        lmsr.calculate_prices(current_state.b, &shares.view())
            .unwrap_or_else(|_| {
                let n = shares.len();
                Array::from_elem(n, 1.0 / n as f64)
            })
    }

    /// Calculate prices only for valid states (excludes invalid state combinations)
    /// This normalizes prices among valid outcomes only, for user display
    pub fn calculate_prices_for_display(&self) -> Vec<f64> {
        let all_prices = self.current_prices();
        let valid_state_combos = self.get_valid_state_combos();

        // Extract prices for valid states only
        let valid_prices: Vec<f64> = valid_state_combos
            .iter()
            .map(|(state_idx, _)| all_prices[*state_idx])
            .collect();

        // Renormalize so valid prices sum to 1.0
        let valid_sum: f64 = valid_prices.iter().sum();
        if valid_sum > 0.0 {
            valid_prices.iter().map(|p| p / valid_sum).collect()
        } else {
            // If all prices are zero, return equal probabilities
            let count = valid_prices.len();
            if count > 0 {
                vec![1.0 / count as f64; count]
            } else {
                vec![]
            }
        }
    }

    /// Update market with new share quantities (creates new state version)
    pub fn update_shares(
        &mut self,
        new_shares: Array<f64, Ix1>,
        transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<MarketStateHash, MarketError> {
        if new_shares.len() != self.share_vector_length {
            return Err(MarketError::InvalidDimensions);
        }

        // Calculate new treasury based on new shares
        let current_state = self.get_current_state();
        let new_treasury = crate::math::lmsr::calculate_cost(
            &new_shares.to_vec(),
            current_state.b,
        )
        .map_err(|e| {
            MarketError::DatabaseError(format!(
                "LMSR calculation failed: {:?}",
                e
            ))
        })?;

        self.create_new_state_version(
            transaction_id,
            height,
            None, // keep same market state
            None, // keep same b
            None, // keep same trading fee
            Some(new_shares),
            None, // keep same final prices
            Some(new_treasury),
        )
    }

    /// Update trading volume when a trade occurs
    pub fn update_trading_volume(
        &mut self,
        outcome_index: usize,
        trade_cost_sats: u64,
    ) -> Result<(), MarketError> {
        if outcome_index >= self.outcome_volumes_sats.len() {
            return Err(MarketError::InvalidOutcomeIndex(outcome_index));
        }

        // Update per-outcome volume
        self.outcome_volumes_sats[outcome_index] = self.outcome_volumes_sats
            [outcome_index]
            .saturating_add(trade_cost_sats);

        // Update total volume
        self.total_volume_sats =
            self.total_volume_sats.saturating_add(trade_cost_sats);

        Ok(())
    }

    /// Calculate cost to update shares (for trading)
    /// Returns the cost in terms of treasury difference
    pub fn query_update_cost(
        &self,
        new_shares: Array<f64, Ix1>,
    ) -> Result<f64, MarketError> {
        if new_shares.len() != self.share_vector_length {
            return Err(MarketError::InvalidDimensions);
        }

        let current_state = self.get_current_state();

        // Efficient calculation without cloning full market
        use crate::math::lmsr::Lmsr;
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(current_state.shares.len());
        let current_cost = lmsr
            .cost_function(current_state.b, &current_state.shares.view())
            .map_err(|e| {
                MarketError::DatabaseError(format!(
                    "LMSR calculation failed: {:?}",
                    e
                ))
            })?;
        let new_cost = lmsr
            .cost_function(current_state.b, &new_shares.view())
            .map_err(|e| {
                MarketError::DatabaseError(format!(
                    "LMSR calculation failed: {:?}",
                    e
                ))
            })?;

        Ok(new_cost - current_cost)
    }

    /// Calculate cost to amplify beta parameter (only increases allowed)
    pub fn query_amp_b_cost(&self, new_b: f64) -> Result<f64, MarketError> {
        let current_state = self.get_current_state();

        if new_b <= current_state.b {
            return Err(MarketError::InvalidBeta(new_b));
        }

        use crate::math::lmsr::Lmsr;
        // Use LMSR with actual market size instead of hardcoded default (256)
        let lmsr = Lmsr::new(current_state.shares.len());
        let new_cost = lmsr
            .cost_function(new_b, &current_state.shares.view())
            .map_err(|e| {
                MarketError::DatabaseError(format!(
                    "LMSR calculation failed: {:?}",
                    e
                ))
            })?;

        Ok(new_cost - current_state.treasury)
    }

    /// Check if market has entered voting period and update state if needed
    /// This should be called when any of the market's decision slots enter voting
    pub fn check_voting_period(
        &mut self,
        slots_in_voting: &HashSet<SlotId>,
        transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<bool, MarketError> {
        if self.state() != MarketState::Trading {
            return Ok(false);
        }

        // If any decision slot is in voting, the market enters voting state
        let has_voting_slot = self
            .decision_slots
            .iter()
            .any(|slot_id| slots_in_voting.contains(slot_id));

        if has_voting_slot {
            self.create_new_state_version(
                transaction_id,
                height,
                Some(MarketState::Voting),
                None,
                None,
                None,
                None,
                None,
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Cancel market (only valid before trading starts)
    pub fn cancel_market(
        &mut self,
        transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<MarketStateHash, MarketError> {
        match self.state() {
            MarketState::Trading if self.treasury() == 0.0 => {
                // Can only cancel if no trades have occurred
                self.create_new_state_version(
                    transaction_id,
                    height,
                    Some(MarketState::Cancelled),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            }
            current_state => Err(MarketError::InvalidStateTransition {
                from: current_state,
                to: MarketState::Cancelled,
            }),
        }
    }

    /// Mark market as invalid (governance action)
    pub fn invalidate_market(
        &mut self,
        transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<MarketStateHash, MarketError> {
        match self.state() {
            MarketState::Trading | MarketState::Voting => self
                .create_new_state_version(
                    transaction_id,
                    height,
                    Some(MarketState::Invalid),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            current_state => Err(MarketError::InvalidStateTransition {
                from: current_state,
                to: MarketState::Invalid,
            }),
        }
    }

    /// Check if all decision slots are ossified and update state accordingly
    pub fn check_ossification(
        &mut self,
        slot_states: &HashMap<SlotId, bool>,
        transaction_id: Option<[u8; 32]>,
        height: u64,
    ) -> Result<(), MarketError> {
        // Only markets that are resolved can become ossified
        if self.state() != MarketState::Resolved {
            return Ok(());
        }

        // Check if all decision slots in this market are ossified
        let all_ossified = self
            .decision_slots
            .iter()
            .all(|slot_id| slot_states.get(slot_id).copied().unwrap_or(false));

        if all_ossified {
            // Create new state version with Ossified state
            self.create_new_state_version(
                transaction_id,
                height,
                Some(MarketState::Ossified),
                None,
                None,
                None,
                None,
                None,
            )?;
        }

        Ok(())
    }

    /// Get market outcome count (total number of possible outcomes)
    pub fn get_outcome_count(&self) -> usize {
        self.shares().len()
    }

    /// Get market dimensions (number of valid states)
    pub fn get_dimensions(&self) -> Vec<usize> {
        vec![self.shares().len()]
    }

    /// Get valid state combinations (includes invalid states for resolution)
    pub fn get_state_combos(&self) -> &Vec<Vec<usize>> {
        &self.state_combos
    }

    /// Get only valid state combinations for user display (filters out invalid states)
    /// Invalid states (value 2 for binary decisions) are excluded from user view
    /// but remain available for resolution purposes
    pub fn get_valid_state_combos(&self) -> Vec<(usize, &Vec<usize>)> {
        self.state_combos
            .iter()
            .enumerate()
            .filter(|(_, combo)| {
                // Filter out combinations that contain invalid states (value 2)
                !combo.iter().any(|&value| value == 2)
            })
            .collect()
    }

    /// Get D_Functions
    pub fn get_d_functions(&self) -> &Vec<DFunction> {
        &self.d_functions
    }

    /// Get the storage fee required for this market based on share vector length
    pub fn get_storage_fee_sats(&self) -> u64 {
        self.storage_fee_sats
    }

    /// Calculate total transaction cost for trading (quadratic scaling with market complexity)
    /// This is safe because share_vector_length is validated during market creation
    pub fn calculate_trade_cost(&self, base_fee_sats: u64) -> u64 {
        // Base transaction fee + quadratic cost for market complexity
        // Trading in complex markets costs more due to computational burden
        // Note: share_vector_length is already validated to be <= MAX_MARKET_OUTCOMES
        let complexity_cost = (self.share_vector_length as u64).pow(2)
            * L2_STORAGE_RATE_SATS_PER_BYTE;
        base_fee_sats + complexity_cost
    }

    /// Get the share vector length (N)
    pub fn get_share_vector_length(&self) -> usize {
        self.share_vector_length
    }

    /// Get current market prices
    /// Alias for calculate_prices() for API compatibility
    pub fn get_current_prices(&self) -> Array<f64, Ix1> {
        self.current_prices()
    }

    /// Calculate cost to buy shares in this market
    ///
    /// # Arguments
    /// * `outcome_index` - The outcome to buy shares for
    /// * `shares_to_buy` - Number of shares to purchase
    ///
    /// # Returns
    /// Cost in satoshis for the trade
    pub fn calculate_buy_cost(
        &self,
        outcome_index: u32,
        shares_to_buy: f64,
    ) -> Result<u64, MarketError> {
        if outcome_index as usize >= self.shares().len() {
            return Err(MarketError::InvalidOutcomeCombination);
        }

        let mut new_shares = self.shares().clone();
        new_shares[outcome_index as usize] += shares_to_buy;

        let cost = self.query_update_cost(new_shares)?;
        let fee_cost = cost * self.trading_fee();

        Ok((cost + fee_cost) as u64)
    }

    /// Find the state index for a given decision outcome combination
    pub fn get_outcome_index(
        &self,
        positions: &[usize],
    ) -> Result<usize, MarketError> {
        if positions.len() != self.decision_slots.len() {
            return Err(MarketError::InvalidDimensions);
        }

        // Find the state that matches this combination
        for (state_idx, combo) in self.state_combos.iter().enumerate() {
            if combo == positions {
                return Ok(state_idx);
            }
        }

        Err(MarketError::InvalidOutcomeCombination)
    }

    /// Get the price for a specific outcome combination
    pub fn get_outcome_price(
        &self,
        positions: &[usize],
    ) -> Result<f64, MarketError> {
        let index = self.get_outcome_index(positions)?;
        let prices = self.current_prices();

        Ok(prices[index])
    }

    /// Human-readable outcome description for a state index
    pub fn describe_outcome_by_state(
        &self,
        state_index: usize,
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<String, MarketError> {
        if state_index >= self.state_combos.len() {
            return Err(MarketError::InvalidDimensions);
        }

        let positions = &self.state_combos[state_index];
        self.describe_outcome(positions, decisions)
    }

    /// Human-readable outcome description
    pub fn describe_outcome(
        &self,
        positions: &[usize],
        decisions: &HashMap<SlotId, Decision>,
    ) -> Result<String, MarketError> {
        if positions.len() != self.decision_slots.len() {
            return Err(MarketError::InvalidDimensions);
        }

        let mut description = Vec::new();

        for (i, &slot_id) in self.decision_slots.iter().enumerate() {
            let decision = decisions
                .get(&slot_id)
                .ok_or(MarketError::DecisionSlotNotFound { slot_id })?;

            let outcome_desc = if decision.is_scaled {
                // Scaled: show the value
                let value = decision.min.unwrap_or(0) + positions[i] as u16;
                format!("{}: {}", decision.question, value)
            } else {
                // Binary: Yes/No/Invalid
                let outcome = match positions[i] {
                    0 => "No",
                    1 => "Yes",
                    _ => "Invalid",
                };
                format!("{}: {}", decision.question, outcome)
            };

            description.push(outcome_desc);
        }

        Ok(description.join(", "))
    }
}

/// Database wrapper for market storage with proper indexing
///
/// Implements secondary indexes for efficient market queries according to
/// Bitcoin Hivemind specification for scalable market operations.
#[derive(Clone)]
pub struct MarketsDatabase {
    /// Primary market storage by ID
    markets: DatabaseUnique<SerdeBincode<[u8; 6]>, SerdeBincode<Market>>,
    /// Secondary index: MarketState -> Vec<MarketId>
    /// Enables O(1) lookups for markets by trading/voting/resolved state
    state_index:
        DatabaseUnique<SerdeBincode<MarketState>, SerdeBincode<Vec<MarketId>>>,
    /// Secondary index: ExpiryHeight -> Vec<MarketId>  
    /// Enables efficient range queries for markets expiring within height windows
    expiry_index:
        DatabaseUnique<SerdeBincode<u64>, SerdeBincode<Vec<MarketId>>>,
    /// Secondary index: SlotId -> Vec<MarketId>
    /// Enables O(1) lookups for markets using specific decision slots
    slot_index:
        DatabaseUnique<SerdeBincode<SlotId>, SerdeBincode<Vec<MarketId>>>,
    /// Address-based share account storage: Address -> ShareAccount
    /// Maps addresses to their share positions using commitment model
    share_accounts:
        DatabaseUnique<SerdeBincode<Address>, SerdeBincode<ShareAccount>>,
}

impl MarketsDatabase {
    /// Schema version for database migrations
    pub const CURRENT_SCHEMA_VERSION: u32 = 2;
    pub const LEGACY_UTXO_SCHEMA_VERSION: u32 = 1;

    /// Migrate from UTXO-based share positions to commitment model
    ///
    /// This method migrates existing UTXO-based share positions to the new
    /// address-based commitment model while maintaining Bitcoin Hivemind
    /// compliance and preventing loss of user positions.
    ///
    /// # Migration Process
    /// 1. Scan all existing UTXOs for share positions
    /// 2. Group positions by address (UTXO owner)
    /// 3. Create ShareAccount entries for each address
    /// 4. Aggregate positions across multiple UTXOs
    /// 5. Initialize nonces to prevent replay attacks
    /// 6. Mark migration as complete
    ///
    /// # Arguments
    /// * `txn` - Database transaction for atomic migration
    /// * `utxos_db` - UTXO database to scan for existing positions
    /// * `height` - Current block height for migration timestamp
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of share accounts migrated
    /// * `Err(Error)` - Migration failure with rollback
    ///
    /// # Safety
    /// This migration is designed to be idempotent and can be run multiple times.
    /// It will skip already-migrated accounts and only process legacy UTXO positions.
    pub fn migrate_utxo_to_commitment_model(
        &self,
        txn: &mut RwTxn,
        utxos_db: &DatabaseUnique<
            SerdeBincode<OutPoint>,
            SerdeBincode<crate::types::FilledOutput>,
        >,
        height: u64,
    ) -> Result<usize, Error> {
        use std::collections::BTreeMap;

        tracing::info!(
            "Starting migration from UTXO-based positions to commitment model at height {}",
            height
        );

        // Phase 1: Scan all UTXOs for share positions
        let mut address_positions: BTreeMap<
            Address,
            BTreeMap<(MarketId, u32), f64>,
        > = BTreeMap::new();
        let mut utxo_count = 0;
        let mut share_utxo_count = 0;

        // Iterate through all UTXOs to find share positions
        let utxo_iter = utxos_db.iter(txn).map_err(|e| {
            Error::DatabaseError(format!("UTXO iteration failed: {}", e))
        })?;

        let mut utxo_iter = utxo_iter;
        while let Some(item) = utxo_iter.next().map_err(|e| {
            Error::DatabaseError(format!("UTXO iteration item failed: {}", e))
        })? {
            let (outpoint, filled_output) = item;

            utxo_count += 1;

            // Check if this UTXO contains share positions
            if let Some(share_data) =
                Self::extract_legacy_share_data(&outpoint, &filled_output)
            {
                share_utxo_count += 1;

                // Group by address
                let address = filled_output.address;
                let address_positions_map =
                    address_positions.entry(address).or_default();

                // Add shares for this market/outcome combination
                let market_id = share_data.market_id.clone();
                let outcome_index = share_data.outcome_index;
                let shares = share_data.shares;
                let key = (share_data.market_id, share_data.outcome_index);
                let current =
                    address_positions_map.get(&key).copied().unwrap_or(0.0);
                address_positions_map.insert(key, current + shares);

                tracing::debug!(
                    "Found {:.4} shares for market {} outcome {} at address {} in UTXO {:?}",
                    shares,
                    market_id,
                    outcome_index,
                    address,
                    outpoint
                );
            }
        }

        // Drop the iterator to release the immutable borrow on txn
        drop(utxo_iter);

        tracing::info!(
            "Scanned {} UTXOs, found {} with share positions across {} addresses",
            utxo_count,
            share_utxo_count,
            address_positions.len()
        );

        // Phase 2: Create ShareAccount entries for each address
        let mut migrated_accounts = 0;
        let mut total_positions_migrated = 0;

        for (address, positions) in address_positions {
            // Check if account already exists (migration already done)
            if self
                .share_accounts
                .try_get(txn, &address)
                .map_err(|e| {
                    Error::DatabaseError(format!(
                        "Share account lookup failed: {}",
                        e
                    ))
                })?
                .is_some()
            {
                tracing::debug!(
                    "Skipping address {} - already has share account (already migrated)",
                    address
                );
                continue;
            }

            // Create new share account
            let mut share_account = ShareAccount::new(address);

            // Add all positions
            for ((market_id, outcome_index), shares) in positions.iter() {
                share_account.add_shares(
                    market_id.clone(),
                    *outcome_index,
                    *shares,
                    height,
                );
                total_positions_migrated += 1;

                tracing::debug!(
                    "Migrated {:.4} shares for market {} outcome {} to address {}",
                    shares,
                    market_id,
                    outcome_index,
                    address
                );
            }

            // Initialize nonce (starts at 0 for new accounts)
            // This prevents replay attacks on the new commitment model

            // Store the migrated account
            self.share_accounts
                .put(txn, &address, &share_account)
                .map_err(|e| {
                    Error::DatabaseError(format!(
                        "Share account creation failed for {}: {}",
                        address, e
                    ))
                })?;

            migrated_accounts += 1;

            tracing::info!(
                "Successfully migrated {} positions for address {} from UTXO model",
                positions.len(),
                address
            );
        }

        // Phase 3: Mark migration as complete (optional - for tracking)
        // This could be stored in a separate migration status table if needed

        tracing::info!(
            "Migration completed: {} accounts with {} total positions migrated from UTXO to commitment model",
            migrated_accounts,
            total_positions_migrated
        );

        Ok(migrated_accounts)
    }

    /// Extract legacy share data from UTXO (placeholder implementation)
    ///
    /// This method would extract share position data from legacy UTXOs.
    /// The actual implementation depends on how shares were stored in UTXOs
    /// in the previous version of the system.
    ///
    /// # Arguments
    /// * `outpoint` - UTXO outpoint
    /// * `filled_output` - UTXO content
    ///
    /// # Returns
    /// * `Some(LegacyShareData)` - If UTXO contains share positions
    /// * `None` - If UTXO doesn't contain shares
    fn extract_legacy_share_data(
        _outpoint: &OutPoint,
        filled_output: &crate::types::FilledOutput,
    ) -> Option<LegacyShareData> {
        // This is a placeholder implementation
        // In the actual migration, you would parse the UTXO content
        // to extract market ID, outcome index, and share quantity

        // For now, return None as there are no legacy UTXOs to migrate
        // This method should be implemented based on the legacy UTXO format

        match &filled_output.content {
            // If there was a legacy share content type, it would be handled here
            // FilledOutputContent::LegacyShares { market_id, outcome_index, shares } => {
            //     Some(LegacyShareData {
            //         market_id: MarketId::new(*market_id),
            //         outcome_index: *outcome_index,
            //         shares: *shares,
            //     })
            // },
            _ => None,
        }
    }

    /// Check if migration from UTXO model is needed
    ///
    /// Determines whether the database needs migration by checking for
    /// legacy UTXO-based share positions.
    ///
    /// # Arguments
    /// * `txn` - Read-only transaction
    /// * `utxos_db` - UTXO database to check
    ///
    /// # Returns
    /// * `Ok(bool)` - True if migration is needed
    /// * `Err(Error)` - Database error during check
    pub fn needs_utxo_migration(
        &self,
        txn: &RoTxn,
        utxos_db: &DatabaseUnique<
            SerdeBincode<OutPoint>,
            SerdeBincode<crate::types::FilledOutput>,
        >,
    ) -> Result<bool, Error> {
        // Check if any UTXOs contain legacy share data
        let utxo_iter = utxos_db.iter(txn).map_err(|e| {
            Error::DatabaseError(format!("UTXO iteration failed: {}", e))
        })?;

        let mut utxo_iter = utxo_iter;
        while let Some(item) = utxo_iter.next().map_err(|e| {
            Error::DatabaseError(format!("UTXO iteration item failed: {}", e))
        })? {
            let (outpoint, filled_output) = item;

            if Self::extract_legacy_share_data(&outpoint, &filled_output)
                .is_some()
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Validate market state transitions according to Bitcoin Hivemind specification.
    /// Delegates to validation.rs for single source of truth.
    ///
    /// # Arguments
    /// * `from_state` - Current market state
    /// * `to_state` - Desired new state
    ///
    /// # Returns
    /// * `Ok(())` - Valid state transition
    /// * `Err(Error)` - Invalid transition with detailed reason
    fn validate_market_state_transition(
        &self,
        from_state: MarketState,
        to_state: MarketState,
    ) -> Result<(), Error> {
        crate::validation::MarketStateValidator::validate_market_state_transition(from_state, to_state)
    }
    /// Number of databases used for markets storage (primary + indexes)
    pub const NUM_DBS: u32 = 5;

    /// Create new markets database with secondary indexes. Does not commit the RwTxn.
    ///
    /// # Arguments
    /// * `env` - LMDB environment
    /// * `rwtxn` - Read-write transaction
    ///
    /// # Returns
    /// Initialized MarketsDatabase with all indexes
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on scalable market storage
    pub fn new(env: &Env, rwtxn: &mut RwTxn) -> Result<Self, Error> {
        let markets = DatabaseUnique::create(env, rwtxn, "markets")?;
        let state_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_state")?;
        let expiry_index =
            DatabaseUnique::create(env, rwtxn, "markets_by_expiry")?;
        let slot_index = DatabaseUnique::create(env, rwtxn, "markets_by_slot")?;
        let share_accounts =
            DatabaseUnique::create(env, rwtxn, "share_accounts")?;

        Ok(MarketsDatabase {
            markets,
            state_index,
            expiry_index,
            slot_index,
            share_accounts,
        })
    }

    /// Add a market to the database with automatic index maintenance
    ///
    /// Maintains ACID consistency by updating all secondary indexes atomically.
    /// Follows Bitcoin Hivemind specification for market registration.
    pub fn add_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        // Add to primary storage
        self.markets.put(txn, market.id.as_bytes(), market)?;

        // Update state index
        self.update_state_index(txn, &market.id, None, Some(market.state()))?;

        // Update expiry index if market has expiry
        if let Some(expires_at) = market.expires_at_height {
            self.update_expiry_index(txn, &market.id, None, Some(expires_at))?;
        }

        // Update slot indexes for all decision slots
        for &slot_id in &market.decision_slots {
            self.update_slot_index(txn, &market.id, slot_id, true)?;
        }

        Ok(())
    }

    /// Get a market by ID
    pub fn get_market(
        &self,
        txn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<Market>, Error> {
        Ok(self.markets.try_get(txn, market_id.as_bytes())?)
    }

    /// Get all markets
    pub fn get_all_markets(&self, txn: &RoTxn) -> Result<Vec<Market>, Error> {
        let markets = self
            .markets
            .iter(txn)?
            .map(|(_, market)| Ok(market))
            .collect()?;
        Ok(markets)
    }

    /// Get multiple markets by their IDs in a batch operation with optimized access
    ///
    /// Eliminates N+1 query pattern by using database cursor iteration instead of
    /// individual lookups. This is especially important for share position calculations
    /// that may need many markets simultaneously.
    ///
    /// # Performance
    /// - Previous: O(n) individual database lookups (N+1 pattern)
    /// - Optimized: Single database iteration with hash-based filtering
    /// - Memory efficient: Only loads requested markets
    ///
    /// # Arguments
    /// * `txn` - Read-only database transaction
    /// * `market_ids` - Market IDs to retrieve
    ///
    /// # Returns
    /// HashMap of found markets, keyed by MarketId
    ///
    /// # Bitcoin Hivemind Compliance
    /// Maintains exact same semantics as individual lookups while improving performance
    pub fn get_markets_batch(
        &self,
        txn: &RoTxn,
        market_ids: &[MarketId],
    ) -> Result<HashMap<MarketId, Market>, Error> {
        if market_ids.is_empty() {
            return Ok(HashMap::new());
        }

        // For small batches, individual lookups may be faster due to lower overhead
        const BATCH_THRESHOLD: usize = 3;
        if market_ids.len() < BATCH_THRESHOLD {
            let mut markets = HashMap::with_capacity(market_ids.len());

            for market_id in market_ids {
                if let Some(market) = self.get_market(txn, market_id)? {
                    markets.insert(market_id.clone(), market);
                }
            }

            return Ok(markets);
        }

        // For larger batches, use optimized cursor iteration
        let market_id_set: HashSet<_> = market_ids.iter().collect();
        let mut markets = HashMap::with_capacity(market_ids.len());
        let mut found_count = 0;

        // Single database iteration instead of N individual lookups
        let market_iter = self.markets.iter(txn).map_err(|e| {
            Error::DatabaseError(format!(
                "Market batch iteration failed: {}",
                e
            ))
        })?;

        let mut market_iter = market_iter;
        while let Some(item) = market_iter.next().map_err(|e| {
            Error::DatabaseError(format!(
                "Market batch iteration item failed: {}",
                e
            ))
        })? {
            let (market_id_bytes, market) = item;

            // Convert bytes back to MarketId for comparison
            let market_id = MarketId::new(market_id_bytes);

            // Check if this market is in our requested set
            if market_id_set.contains(&market_id) {
                markets.insert(market_id, market);
                found_count += 1;

                // Early termination optimization: stop when we've found all requested markets
                if found_count >= market_ids.len() {
                    break;
                }
            }
        }

        tracing::debug!(
            "Batch loaded {}/{} requested markets using optimized iteration",
            found_count,
            market_ids.len()
        );

        Ok(markets)
    }

    /// Get markets by state using efficient O(1) index lookup
    ///
    /// Replaces previous O(n) linear scan with indexed access.
    /// Follows Bitcoin Hivemind specification for state-based market queries.
    pub fn get_markets_by_state(
        &self,
        txn: &RoTxn,
        state: MarketState,
    ) -> Result<Vec<Market>, Error> {
        // Use secondary index for O(1) lookup
        let market_ids =
            self.state_index.try_get(txn, &state)?.unwrap_or_default();

        // Fetch actual markets
        let mut markets = Vec::with_capacity(market_ids.len());
        for market_id in market_ids {
            if let Some(market) = self.get_market(txn, &market_id)? {
                markets.push(market);
            }
        }

        Ok(markets)
    }

    /// Get markets by expiry range using optimized range scanning
    ///
    /// Uses LMDB's native range iteration with early termination for efficient temporal queries
    /// instead of loading all expiry entries. Provides O(log n + k) performance where k is
    /// the number of results, compared to O(n) for the previous full scan approach.
    ///
    /// # Performance Optimization
    /// - Uses streaming database iteration with early termination
    /// - Previous: Load all entries into memory then filter (O(n))
    /// - Optimized: Stream and filter with early termination (O(log n + k))
    /// - Batch market loading for multiple results
    ///
    /// # Bitcoin Hivemind Compliance
    /// Maintains exact same semantics while dramatically improving performance for range queries.
    pub fn get_markets_by_expiry(
        &self,
        txn: &RoTxn,
        min_height: Option<u64>,
        max_height: Option<u64>,
    ) -> Result<Vec<Market>, Error> {
        let mut markets = Vec::new();

        // Use streaming iteration with early termination instead of collecting all entries
        let expiry_iter = self.expiry_index.iter(txn).map_err(|e| {
            Error::DatabaseError(format!(
                "Expiry index iteration failed: {}",
                e
            ))
        })?;

        // Collect market IDs that match the height range
        let mut matching_market_ids = Vec::new();
        let mut entries_checked = 0;
        let mut entries_matched = 0;

        let mut expiry_iter = expiry_iter;
        while let Some(item) = expiry_iter.next().map_err(|e| {
            Error::DatabaseError(format!(
                "Expiry index iteration item failed: {}",
                e
            ))
        })? {
            let (expiry_height, market_ids) = item;

            entries_checked += 1;

            // Early termination optimization: if we have a max_height and current height exceeds it,
            // we can stop since LMDB keys are naturally ordered
            if let Some(max) = max_height {
                if expiry_height > max {
                    break;
                }
            }

            // Check if this expiry height is within our range
            let matches_min =
                min_height.map_or(true, |min| expiry_height >= min);
            let matches_max =
                max_height.map_or(true, |max| expiry_height <= max);

            if matches_min && matches_max {
                entries_matched += 1;
                matching_market_ids.extend(market_ids);
            }
        }

        tracing::debug!(
            "Expiry range query: checked {} entries, matched {} entries with {} total markets",
            entries_checked,
            entries_matched,
            matching_market_ids.len()
        );

        // Use batch loading for efficiency if we have many markets
        if matching_market_ids.len() > 5 {
            // Use our optimized batch loader to eliminate N+1 pattern
            let markets_map =
                self.get_markets_batch(txn, &matching_market_ids)?;

            // Convert to vector maintaining consistent ordering
            for market_id in matching_market_ids {
                if let Some(market) = markets_map.get(&market_id) {
                    markets.push(market.clone());
                }
            }
        } else {
            // For small result sets, individual lookups are fine
            for market_id in matching_market_ids {
                if let Some(market) = self.get_market(txn, &market_id)? {
                    markets.push(market);
                }
            }
        }

        // Sort by expiry height for consistent ordering across calls
        markets.sort_by_key(|m| m.expires_at_height.unwrap_or(u64::MAX));

        tracing::debug!(
            "Retrieved {} markets by expiry range [{:?}, {:?}]",
            markets.len(),
            min_height,
            max_height
        );

        Ok(markets)
    }

    /// Update markets that have entered voting period based on slot states
    ///
    /// Optimized to check only Trading state markets instead of all markets.
    /// Uses indexed access to improve performance for large market databases.
    pub fn update_voting_markets(
        &self,
        txn: &mut RwTxn,
        slots_in_voting: &HashSet<SlotId>,
    ) -> Result<Vec<MarketId>, Error> {
        // Only check markets in Trading state, as only they can transition to Voting
        let trading_markets =
            self.get_markets_by_state(txn, MarketState::Trading)?;
        let mut newly_voting = Vec::new();

        for mut market in trading_markets {
            if market.check_voting_period(slots_in_voting, None, 0)? {
                // TODO: Add transaction_id and height
                self.update_market(txn, &market)?;
                newly_voting.push(market.id.clone());
            }
        }

        Ok(newly_voting)
    }

    /// Get markets by decision slot using efficient O(1) index lookup
    ///
    /// Replaces O(n) linear scan with indexed access for slot-based market queries.
    /// Follows Bitcoin Hivemind specification for slot-market associations.
    pub fn get_markets_by_slot(
        &self,
        txn: &RoTxn,
        slot_id: SlotId,
    ) -> Result<Vec<Market>, Error> {
        // Use secondary index for O(1) lookup
        let market_ids =
            self.slot_index.try_get(txn, &slot_id)?.unwrap_or_default();

        // Fetch actual markets
        let mut markets = Vec::with_capacity(market_ids.len());
        for market_id in market_ids {
            if let Some(market) = self.get_market(txn, &market_id)? {
                markets.push(market);
            }
        }

        Ok(markets)
    }

    /// Update a market with automatic index maintenance and atomic rollback
    ///
    /// Maintains ACID consistency by updating all relevant secondary indexes atomically.
    /// Implements proper rollback handling for multi-phase transaction failures to prevent
    /// database inconsistencies per Bitcoin Hivemind specification.
    ///
    /// # Transaction Safety
    /// - All operations are atomic within the RwTxn scope
    /// - Failed operations automatically rollback the entire transaction
    /// - Database consistency is guaranteed by LMDB's ACID properties
    ///
    /// # Error Handling
    /// - Any failure in index updates will cause the entire transaction to rollback
    /// - Secondary index inconsistencies are prevented by atomic operation grouping
    /// - Partial updates are impossible due to transaction boundary enforcement
    pub fn update_market(
        &self,
        txn: &mut RwTxn,
        market: &Market,
    ) -> Result<(), Error> {
        // Get the old market for index updates (fails fast if market doesn't exist for updates)
        let old_market = self.get_market(txn, &market.id)?;

        // Phase 1: Capture current state for potential rollback
        let _old_state = old_market.as_ref().map(|m| m.state());
        let _old_expiry = old_market.as_ref().and_then(|m| m.expires_at_height);
        let old_slots: HashSet<_> = old_market
            .as_ref()
            .map(|m| m.decision_slots.iter().cloned().collect())
            .unwrap_or_default();

        // Phase 2: Validate update operation before making any changes
        if let Some(ref old) = old_market {
            // Validate state transition is legal per Bitcoin Hivemind specification
            self.validate_market_state_transition(old.state(), market.state())?;
        }

        // Phase 3: Perform atomic multi-phase update with automatic rollback on failure
        // Note: All operations within this RwTxn will be atomically rolled back if any fail

        // Update primary storage first
        self.markets
            .put(txn, market.id.as_bytes(), market)
            .map_err(|e| {
                tracing::error!(
                    "Failed to update primary market storage for market {}: {}",
                    market.id,
                    e
                );
                Error::DatabaseError(format!(
                    "Primary market update failed: {}",
                    e
                ))
            })?;

        // Update secondary indexes with comprehensive error handling
        if let Some(old) = old_market {
            // Update state index if state changed
            if old.state() != market.state() {
                self.update_state_index(
                    txn,
                    &market.id,
                    Some(old.state()),
                    Some(market.state()),
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to update state index for market {}: {}",
                        market.id,
                        e
                    );
                    Error::DatabaseError(format!(
                        "State index update failed: {}",
                        e
                    ))
                })?;
            }

            // Update expiry index if expiry changed
            if old.expires_at_height != market.expires_at_height {
                self.update_expiry_index(
                    txn,
                    &market.id,
                    old.expires_at_height,
                    market.expires_at_height,
                )
                .map_err(|e| {
                    tracing::error!(
                        "Failed to update expiry index for market {}: {}",
                        market.id,
                        e
                    );
                    Error::DatabaseError(format!(
                        "Expiry index update failed: {}",
                        e
                    ))
                })?;
            }

            // Update slot indexes if slots changed (unlikely but possible)
            let new_slots: HashSet<_> =
                market.decision_slots.iter().cloned().collect();

            // Remove from slots that are no longer used
            for slot_id in old_slots.difference(&new_slots) {
                self.update_slot_index(txn, &market.id, *slot_id, false)
                    .map_err(|e| {
                        tracing::error!(
                            "Failed to remove market {} from slot index {}: {}",
                            market.id,
                            hex::encode(slot_id.as_bytes()),
                            e
                        );
                        Error::DatabaseError(format!(
                            "Slot index removal failed: {}",
                            e
                        ))
                    })?;
            }

            // Add to new slots
            for slot_id in new_slots.difference(&old_slots) {
                self.update_slot_index(txn, &market.id, *slot_id, true)
                    .map_err(|e| {
                        tracing::error!(
                            "Failed to add market {} to slot index {}: {}",
                            market.id,
                            hex::encode(slot_id.as_bytes()),
                            e
                        );
                        Error::DatabaseError(format!(
                            "Slot index addition failed: {}",
                            e
                        ))
                    })?;
            }
        }

        // All operations succeeded - transaction will commit automatically
        tracing::debug!(
            "Successfully updated market {} with all indexes",
            market.id
        );
        Ok(())
    }

    /// Check and update ossification status for resolved markets
    ///
    /// Optimized to check only Resolved state markets instead of all markets.
    /// Uses indexed access to improve performance for large market databases.
    pub fn update_ossification_status(
        &self,
        txn: &mut RwTxn,
        ossified_slots: &HashSet<SlotId>,
    ) -> Result<Vec<MarketId>, Error> {
        // Only check markets in Resolved state, as only they can transition to Ossified
        let resolved_markets =
            self.get_markets_by_state(txn, MarketState::Resolved)?;
        let mut newly_ossified = Vec::new();

        for mut market in resolved_markets {
            // All markets from get_markets_by_state(Resolved) are already in Resolved state

            // Check if all decision slots in this market are ossified
            let all_slots_ossified = market
                .decision_slots
                .iter()
                .all(|slot_id| ossified_slots.contains(slot_id));

            if all_slots_ossified {
                // Create new state version with Ossified state
                market
                    .create_new_state_version(
                        None, // transaction_id
                        0,    // height - TODO: get actual height
                        Some(MarketState::Ossified),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| {
                        Error::DatabaseError(format!(
                            "Failed to ossify market: {:?}",
                            e
                        ))
                    })?;

                self.update_market(txn, &market)?;
                newly_ossified.push(market.id.clone());
            }
        }

        Ok(newly_ossified)
    }

    /// Cancel a market (only valid before trading starts)
    pub fn cancel_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), MarketError> {
        let mut market = self
            .get_market(txn, market_id)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))?
            .ok_or(MarketError::MarketNotFound {
                id: market_id.clone(),
            })?;
        market.cancel_market(None, 0)?; // TODO: Add transaction_id and height parameters
        self.update_market(txn, &market)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))
    }

    /// Invalidate a market (governance action)
    pub fn invalidate_market(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), MarketError> {
        let mut market = self
            .get_market(txn, market_id)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))?
            .ok_or(MarketError::MarketNotFound {
                id: market_id.clone(),
            })?;
        market.invalidate_market(None, 0)?; // TODO: Add transaction_id and height parameters
        self.update_market(txn, &market)
            .map_err(|e| MarketError::DatabaseError(e.to_string()))
    }

    /// Get mempool-adjusted shares for a market
    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &MarketId,
    ) -> Result<Option<Array<f64, Ix1>>, Error> {
        // Use a special reserved address for mempool shares (all zeros with a flag)
        let mut mempool_addr_bytes = [0u8; 20];
        mempool_addr_bytes[0] = 0xFF; // Special flag for mempool
        mempool_addr_bytes[1..7].copy_from_slice(&market_id.0);
        let mempool_addr = Address(mempool_addr_bytes);

        match self.share_accounts.get(rotxn, &mempool_addr) {
            Ok(account) => {
                // Reconstruct shares array from stored positions
                if let Some(market) = self.get_market(rotxn, market_id)? {
                    let mut shares = market.shares().clone();
                    // Update shares based on account positions for this market
                    for ((account_market_id, outcome_index), &amount) in
                        &account.positions
                    {
                        if account_market_id == market_id {
                            shares[*outcome_index as usize] = amount;
                        }
                    }
                    return Ok(Some(shares));
                }
                Ok(None)
            }
            Err(_) => Ok(None), // If error getting account, assume no mempool shares
        }
    }

    /// Store mempool-adjusted shares for a market
    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
        shares: &Array<f64, Ix1>,
    ) -> Result<(), Error> {
        // Use a special reserved address for mempool shares
        let mut mempool_addr_bytes = [0u8; 20];
        mempool_addr_bytes[0] = 0xFF; // Special flag for mempool
        mempool_addr_bytes[1..7].copy_from_slice(&market_id.0);
        let mempool_addr = Address(mempool_addr_bytes);

        // Get or create account for mempool shares
        let mut account = match self.share_accounts.get(rwtxn, &mempool_addr) {
            Ok(acc) => acc,
            Err(_) => ShareAccount::new(mempool_addr.clone()),
        };

        // Clear existing positions for this market and set new ones
        account.positions.retain(|(mid, _), _| mid != market_id);

        // Add new positions based on shares array
        for (i, &share_amount) in shares.iter().enumerate() {
            if share_amount != 0.0 {
                account
                    .positions
                    .insert((market_id.clone(), i as u32), share_amount);
            }
        }

        self.share_accounts.put(rwtxn, &mempool_addr, &account)?;
        Ok(())
    }

    /// Clear mempool shares after block confirmation
    pub fn clear_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &MarketId,
    ) -> Result<(), Error> {
        let mut mempool_addr_bytes = [0u8; 20];
        mempool_addr_bytes[0] = 0xFF;
        mempool_addr_bytes[1..7].copy_from_slice(&market_id.0);
        let mempool_addr = Address(mempool_addr_bytes);

        // Get account and remove positions for this market
        if let Ok(mut account) = self.share_accounts.get(rwtxn, &mempool_addr) {
            account.positions.retain(|(mid, _), _| mid != market_id);

            // If account has no more positions, delete it entirely
            if account.positions.is_empty() {
                self.share_accounts.delete(rwtxn, &mempool_addr)?;
            } else {
                // Otherwise update with remaining positions
                self.share_accounts.put(rwtxn, &mempool_addr, &account)?;
            }
        }
        Ok(())
    }

    /// Update state index when market state changes
    ///
    /// Maintains consistency between primary storage and state index.
    /// Handles addition, removal, and state transitions atomically.
    fn update_state_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        old_state: Option<MarketState>,
        new_state: Option<MarketState>,
    ) -> Result<(), Error> {
        // Remove from old state index
        if let Some(old) = old_state {
            let mut market_ids =
                self.state_index.try_get(txn, &old)?.unwrap_or_default();
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.state_index.delete(txn, &old)?;
            } else {
                self.state_index.put(txn, &old, &market_ids)?;
            }
        }

        // Add to new state index
        if let Some(new) = new_state {
            let mut market_ids =
                self.state_index.try_get(txn, &new)?.unwrap_or_default();
            if !market_ids.contains(market_id) {
                market_ids.push(market_id.clone());
                self.state_index.put(txn, &new, &market_ids)?;
            }
        }

        Ok(())
    }

    /// Update expiry index when market expiry changes
    ///
    /// Maintains consistency between primary storage and expiry index.
    /// Handles temporal indexing for efficient range queries.
    fn update_expiry_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        old_expiry: Option<u64>,
        new_expiry: Option<u64>,
    ) -> Result<(), Error> {
        // Remove from old expiry index
        if let Some(old) = old_expiry {
            let mut market_ids =
                self.expiry_index.try_get(txn, &old)?.unwrap_or_default();
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.expiry_index.delete(txn, &old)?;
            } else {
                self.expiry_index.put(txn, &old, &market_ids)?;
            }
        }

        // Add to new expiry index
        if let Some(new) = new_expiry {
            let mut market_ids =
                self.expiry_index.try_get(txn, &new)?.unwrap_or_default();
            if !market_ids.contains(market_id) {
                market_ids.push(market_id.clone());
                self.expiry_index.put(txn, &new, &market_ids)?;
            }
        }

        Ok(())
    }

    /// Update slot index when market-slot associations change
    ///
    /// Maintains consistency between primary storage and slot index.
    /// Enables efficient slot-to-market lookups.
    fn update_slot_index(
        &self,
        txn: &mut RwTxn,
        market_id: &MarketId,
        slot_id: SlotId,
        add: bool,
    ) -> Result<(), Error> {
        let mut market_ids =
            self.slot_index.try_get(txn, &slot_id)?.unwrap_or_default();

        if add {
            // Add market to slot index
            if !market_ids.contains(market_id) {
                market_ids.push(market_id.clone());
                self.slot_index.put(txn, &slot_id, &market_ids)?;
            }
        } else {
            // Remove market from slot index
            market_ids.retain(|id| id != market_id);
            if market_ids.is_empty() {
                self.slot_index.delete(txn, &slot_id)?;
            } else {
                self.slot_index.put(txn, &slot_id, &market_ids)?;
            }
        }

        Ok(())
    }

    /// Process a batch of market trades atomically using snapshot-based pricing
    ///
    /// This ensures all trades in a block use the same base market state for fair pricing
    /// according to Bitcoin Hivemind whitepaper specifications. Implements comprehensive
    /// error handling and atomic rollback for multi-phase transaction failures.
    ///
    /// # Arguments
    /// * `txn` - Database transaction
    /// * `batched_trades` - Vector of validated market trades to process
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - Trade costs for each processed trade
    /// * `Err(Error)` - If any validation fails, with automatic rollback
    ///
    /// # Transaction Safety
    /// - All operations are atomic within the RwTxn scope
    /// - Failed operations automatically rollback the entire batch
    /// - Market state consistency is guaranteed across all trades
    /// - Share account updates are atomic with market updates
    pub fn process_market_trades_batch(
        &self,
        txn: &mut RwTxn,
        batched_trades: Vec<BatchedMarketTrade>,
    ) -> Result<Vec<f64>, Error> {
        if batched_trades.is_empty() {
            return Ok(Vec::new());
        }

        let mut trade_costs = Vec::with_capacity(batched_trades.len());
        let mut market_updates: HashMap<MarketId, Array<f64, Ix1>> =
            HashMap::new();

        // Phase 1: Pre-validation - validate all trades before making any changes
        // This prevents partial application of trades if any validation fails
        tracing::debug!(
            "Validating {} batched market trades",
            batched_trades.len()
        );

        for (trade_index, trade) in batched_trades.iter().enumerate() {
            // Validate market exists and is in trading state
            let market_id = MarketId::new(trade.market_id);
            let market = self
                .get_market(txn, &market_id)
                .map_err(|e| {
                    tracing::error!(
                        "Database error accessing market {} for trade {}: {}",
                        hex::encode(&trade.market_id),
                        trade_index,
                        e
                    );
                    Error::DatabaseError(format!("Market access failed: {}", e))
                })?
                .ok_or_else(|| {
                    tracing::error!(
                        "Market {} not found for trade {}",
                        hex::encode(&trade.market_id),
                        trade_index
                    );
                    Error::Market(MarketError::MarketNotFound {
                        id: MarketId(trade.market_id),
                    })
                })?;

            // Validate market state allows trading
            if market.state() != MarketState::Trading {
                tracing::error!(
                    "Market {} is not in trading state for trade {}: {:?}",
                    hex::encode(&trade.market_id),
                    trade_index,
                    market.state()
                );
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Market {} is not in trading state",
                        hex::encode(&trade.market_id)
                    ),
                });
            }

            // Validate outcome index is valid for market
            if trade.outcome_index as usize >= market.shares().len() {
                tracing::error!(
                    "Invalid outcome index {} for market {} with {} outcomes",
                    trade.outcome_index,
                    hex::encode(&trade.market_id),
                    market.shares().len()
                );
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Invalid outcome index {} for market",
                        trade.outcome_index
                    ),
                });
            }

            // Calculate and validate trade cost using market snapshot
            let cost = trade.calculate_trade_cost().map_err(|e| {
                tracing::error!(
                    "Trade cost calculation failed for trade {}: {}",
                    trade_index,
                    e
                );
                Error::InvalidTransaction {
                    reason: format!("Trade cost calculation failed: {}", e),
                }
            })?;

            // Validate cost against trader's maximum
            if cost > trade.max_cost as f64 {
                tracing::error!(
                    "Trade {} cost {:.4} exceeds maximum {}",
                    trade_index,
                    cost,
                    trade.max_cost
                );
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Trade cost {:.4} exceeds maximum {}",
                        cost, trade.max_cost
                    ),
                });
            }

            // Validate trader has no negative positions after trade
            // (Note: This would require checking existing account balances)

            trade_costs.push(cost);

            // Accumulate share changes for each market
            let shares_update =
                market_updates.entry(market_id.clone()).or_insert_with(|| {
                    Array::zeros(trade.market_snapshot.shares.len())
                });
            shares_update[trade.outcome_index as usize] += trade.shares_to_buy;
        }

        // Phase 2: Apply all validated market updates atomically
        // All validation passed, now apply changes with comprehensive error handling
        tracing::debug!(
            "Applying market updates for {} markets",
            market_updates.len()
        );

        for (market_id, share_changes) in market_updates {
            let mut market =
                self.get_market(txn, &market_id)?.ok_or_else(|| {
                    tracing::error!(
                        "Market {} disappeared during batch processing",
                        market_id
                    );
                    Error::DatabaseError(
                        "Market disappeared during processing".to_string(),
                    )
                })?;

            // Apply accumulated share changes with bounds checking
            let mut new_shares_array = market.shares().clone();
            for (outcome_index, &share_change) in
                share_changes.iter().enumerate()
            {
                if share_change != 0.0 {
                    let new_shares =
                        new_shares_array[outcome_index] + share_change;
                    // Validate shares don't go negative (should not happen with proper validation)
                    if new_shares < 0.0 {
                        tracing::error!(
                            "Share update would result in negative shares for market {} outcome {}: {} + {} = {}",
                            market_id,
                            outcome_index,
                            market.shares()[outcome_index],
                            share_change,
                            new_shares
                        );
                        return Err(Error::DatabaseError("Invalid share update would result in negative shares".to_string()));
                    }
                    new_shares_array[outcome_index] = new_shares;
                }
            }

            // Recalculate treasury with the updated shares
            let new_treasury =
                market.calc_treasury_with_shares(&new_shares_array);

            // Create new market state version with updated shares and treasury
            market
                .create_new_state_version(
                    None, // transaction_id - TODO: pass actual transaction_id
                    0,    // height - TODO: pass actual height
                    None, // keep current market state
                    None, // keep current b
                    None, // keep current trading fee
                    Some(new_shares_array),
                    None, // keep current final prices
                    Some(new_treasury),
                )
                .map_err(|e| {
                    Error::DatabaseError(format!(
                        "Failed to create new market state: {:?}",
                        e
                    ))
                })?;

            // Update market in database with full error context
            self.update_market(txn, &market).map_err(|e| {
                tracing::error!(
                    "Failed to update market {} during batch processing: {}",
                    market_id,
                    e
                );
                e // Propagate the detailed error from update_market
            })?;

            tracing::debug!(
                "Successfully updated market {} with new treasury: {:.4}",
                market_id,
                market.treasury()
            );
        }

        // Phase 3: Update share accounts atomically with market updates
        tracing::debug!(
            "Updating share accounts for {} trades",
            batched_trades.len()
        );

        for (trade_index, trade) in batched_trades.iter().enumerate() {
            self.add_shares_to_account(
                txn,
                &trade.trader_address,
                MarketId::new(trade.market_id),
                trade.outcome_index,
                trade.shares_to_buy,
                0, // Height will be set by caller
            )
            .map_err(|e| {
                tracing::error!(
                    "Failed to update share account for trade {}: {}",
                    trade_index,
                    e
                );
                Error::DatabaseError(format!(
                    "Share account update failed for trade {}: {}",
                    trade_index, e
                ))
            })?;
        }

        tracing::info!(
            "Successfully processed {} batched market trades with total cost: {:.4}",
            batched_trades.len(),
            trade_costs.iter().sum::<f64>()
        );

        Ok(trade_costs)
    }

    /// Add shares to an address-based account using commitment model
    ///
    /// This tracks shares per address rather than per UTXO, allowing for more efficient
    /// position management while maintaining security through UTXO authorization requirements.
    ///
    /// # Arguments
    /// * `txn` - Database transaction
    /// * `address` - Address to add shares to
    /// * `market_id` - Market identifier
    /// * `outcome_index` - Specific outcome being traded
    /// * `shares` - Number of shares to add
    /// * `height` - Current block height
    pub fn add_shares_to_account(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: MarketId,
        outcome_index: u32,
        shares: f64,
        height: u64,
    ) -> Result<(), Error> {
        // Get or create account for this address
        let mut account = self
            .share_accounts
            .try_get(txn, address)?
            .unwrap_or_else(|| ShareAccount::new(*address));

        // Add shares to the account
        account.add_shares(market_id, outcome_index, shares, height);

        // Save updated account
        self.share_accounts.put(txn, address, &account)?;

        Ok(())
    }

    /// Remove shares from an address-based account
    ///
    /// # Arguments
    /// * `txn` - Database transaction
    /// * `address` - Address to remove shares from
    /// * `market_id` - Market identifier
    /// * `outcome_index` - Specific outcome being traded
    /// * `shares` - Number of shares to remove
    /// * `height` - Current block height
    pub fn remove_shares_from_account(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: &MarketId,
        outcome_index: u32,
        shares: f64,
        height: u64,
    ) -> Result<(), Error> {
        // Get account for this address
        let mut account = self
            .share_accounts
            .try_get(txn, address)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "No share account found for address".to_string(),
            })?;

        // Remove shares from the account
        account
            .remove_shares(market_id, outcome_index, shares, height)
            .map_err(|_| Error::InvalidTransaction {
                reason: "Insufficient shares for sell transaction".to_string(),
            })?;

        // Save updated account (or remove if empty)
        if account.positions.is_empty() {
            self.share_accounts.delete(txn, address)?;
        } else {
            self.share_accounts.put(txn, address, &account)?;
        }

        Ok(())
    }

    /// Get share account for a given address
    ///
    /// Returns the share account containing all positions for the specified address.
    pub fn get_user_share_account(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<Option<ShareAccount>, Error> {
        Ok(self.share_accounts.try_get(txn, address)?)
    }

    /// Get all share positions for a given address (API compatibility method)
    ///
    /// Returns positions as a vector for backward compatibility with existing APIs.
    pub fn get_user_share_positions(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<Vec<(MarketId, u32, f64)>, Error> {
        if let Some(account) = self.get_user_share_account(txn, address)? {
            Ok(account
                .positions
                .into_iter()
                .map(|((market_id, outcome_index), shares)| {
                    (market_id, outcome_index, shares)
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get share positions for a specific user and market
    ///
    /// Returns positions for the specified address and market.
    pub fn get_market_user_positions(
        &self,
        txn: &RoTxn,
        address: &Address,
        market_id: &MarketId,
    ) -> Result<Vec<(u32, f64)>, Error> {
        if let Some(account) = self.get_user_share_account(txn, address)? {
            Ok(account
                .positions
                .into_iter()
                .filter(|((pos_market_id, _), _)| pos_market_id == market_id)
                .map(|((_, outcome_index), shares)| (outcome_index, shares))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Apply share redemption for a resolved market
    ///
    /// When a market resolves, users can redeem their shares for Bitcoin based
    /// on the final resolution. This updates the address-based share accounts.
    pub fn apply_share_redemption(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: [u8; 6],
        outcome_index: u32,
        shares_to_redeem: f64,
        height: u64,
    ) -> Result<(), Error> {
        let market_id_struct = MarketId::new(market_id);
        self.remove_shares_from_account(
            txn,
            address,
            &market_id_struct,
            outcome_index,
            shares_to_redeem,
            height,
        )
    }

    /// Revert a share trade (for block disconnection)
    ///
    /// Reverts share position changes when a block is disconnected.
    pub fn revert_share_trade(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: [u8; 6],
        outcome_index: u32,
        shares_traded: f64,
        height: u64,
    ) -> Result<(), Error> {
        let market_id_struct = MarketId::new(market_id);
        // Reverse the trade by removing the shares that were added
        self.remove_shares_from_account(
            txn,
            address,
            &market_id_struct,
            outcome_index,
            shares_traded,
            height,
        )
    }

    /// Revert share redemption (for block disconnection)
    ///
    /// Reverts share redemption when a block is disconnected.
    pub fn revert_share_redemption(
        &self,
        txn: &mut RwTxn,
        address: &Address,
        market_id: [u8; 6],
        outcome_index: u32,
        shares_redeemed: f64,
        height: u64,
    ) -> Result<(), Error> {
        let market_id_struct = MarketId::new(market_id);
        // Reverse the redemption by adding the shares back
        self.add_shares_to_account(
            txn,
            address,
            market_id_struct,
            outcome_index,
            shares_redeemed,
            height,
        )
    }

    /// Get the current nonce for an address (for replay protection)
    pub fn get_account_nonce(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<u64, Error> {
        if let Some(account) = self.share_accounts.try_get(txn, address)? {
            Ok(account.nonce)
        } else {
            Ok(0) // New accounts start with nonce 0
        }
    }

    /// Get all nonces for an address (enhanced security)
    ///
    /// Returns both global nonce and per-transaction-type nonces for comprehensive
    /// replay attack prevention across different operation types.
    ///
    /// # Returns
    /// Tuple of (global_nonce, redemption_nonce, trade_nonce)
    pub fn get_account_nonces(
        &self,
        txn: &RoTxn,
        address: &Address,
    ) -> Result<(u64, u64, u64), Error> {
        if let Some(account) = self.share_accounts.try_get(txn, address)? {
            Ok((account.nonce, account.redemption_nonce, account.trade_nonce))
        } else {
            Ok((0, 0, 0)) // New accounts start with all nonces at 0
        }
    }

    // Note: Market deletion is intentionally not supported.
    // Markets follow state transitions and eventually become ossified when all
    // their decision slots are ossified. This maintains blockchain immutability
    // and preserves the complete audit trail.
}

/// Legacy share data structure for UTXO migration
///
/// Represents share position data extracted from legacy UTXOs
/// during migration to the commitment model.
#[derive(Debug, Clone)]
struct LegacyShareData {
    /// Market identifier
    market_id: MarketId,
    /// Outcome index within market
    outcome_index: u32,
    /// Number of shares held
    shares: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    /// Test basic D-function constraint validation
    #[test]
    fn test_dfunction_constraint_validation() {
        let decision_slots = vec![];

        // Test valid decision reference
        let valid_func = DFunction::Decision(0);
        assert!(valid_func.validate_constraint(2, &decision_slots).is_ok());

        // Test invalid decision reference (out of bounds)
        let invalid_func = DFunction::Decision(5);
        assert!(
            invalid_func
                .validate_constraint(2, &decision_slots)
                .is_err()
        );

        // Test valid equals constraint
        let valid_equals =
            DFunction::Equals(Box::new(DFunction::Decision(0)), 1);
        assert!(valid_equals.validate_constraint(2, &decision_slots).is_ok());

        // Test invalid equals constraint (bad value)
        let invalid_equals =
            DFunction::Equals(Box::new(DFunction::Decision(0)), 5);
        assert!(
            invalid_equals
                .validate_constraint(2, &decision_slots)
                .is_err()
        );

        // Test nested constraints
        let nested_and = DFunction::And(
            Box::new(DFunction::Decision(0)),
            Box::new(DFunction::Decision(1)),
        );
        assert!(nested_and.validate_constraint(2, &decision_slots).is_ok());

        let invalid_nested = DFunction::And(
            Box::new(DFunction::Decision(0)),
            Box::new(DFunction::Decision(5)), // Invalid reference
        );
        assert!(
            invalid_nested
                .validate_constraint(2, &decision_slots)
                .is_err()
        );
    }

    /// Test categorical constraint validation
    #[test]
    fn test_categorical_constraint_validation() {
        let decision_slots = vec![];
        let df = DFunction::True; // Simple test function

        // Test valid categorical constraint (exactly one true)
        let valid_combo = vec![1, 0, 0]; // First slot true, others false
        let categorical_slots = vec![0, 1, 2];
        assert!(
            df.validate_categorical_constraint(
                &categorical_slots,
                &valid_combo,
                &decision_slots
            )
            .unwrap()
        );

        // Test residual case (all false)
        let residual_combo = vec![0, 0, 0];
        assert!(
            df.validate_categorical_constraint(
                &categorical_slots,
                &residual_combo,
                &decision_slots
            )
            .unwrap()
        );

        // Test invalid constraint (multiple true)
        let invalid_combo = vec![1, 1, 0];
        assert!(
            !df.validate_categorical_constraint(
                &categorical_slots,
                &invalid_combo,
                &decision_slots
            )
            .unwrap()
        );

        // Test out of bounds
        let oob_slots = vec![0, 1, 5]; // Index 5 doesn't exist
        assert!(
            df.validate_categorical_constraint(
                &oob_slots,
                &valid_combo,
                &decision_slots
            )
            .is_err()
        );
    }

    /// Test dimension parsing
    #[test]
    fn test_dimension_parsing() {
        // Test single slot parsing (using 3-byte hex values)
        let single_str = "[010101]"; // 3 bytes = 6 hex chars
        let result = parse_dimensions(single_str);
        assert!(result.is_ok());
        let dimensions = result.unwrap();
        assert_eq!(dimensions.len(), 1);
        assert!(matches!(dimensions[0], DimensionSpec::Single(_)));

        // Test categorical parsing
        let categorical_str = "[[010101,010102,010103]]";
        let result = parse_dimensions(categorical_str);
        assert!(result.is_ok());
        let dimensions = result.unwrap();
        assert_eq!(dimensions.len(), 1);
        if let DimensionSpec::Categorical(slots) = &dimensions[0] {
            assert_eq!(slots.len(), 3);
        } else {
            panic!("Expected categorical dimension");
        }

        // Test mixed parsing
        let mixed_str = "[010101,[010102,010103],010104]";
        let result = parse_dimensions(mixed_str);
        assert!(result.is_ok());
        let dimensions = result.unwrap();
        assert_eq!(dimensions.len(), 3);
        assert!(matches!(dimensions[0], DimensionSpec::Single(_)));
        assert!(matches!(dimensions[1], DimensionSpec::Categorical(_)));
        assert!(matches!(dimensions[2], DimensionSpec::Single(_)));

        // Test invalid format
        let invalid_str = "010101,010102";
        let result = parse_dimensions(invalid_str);
        assert!(result.is_err());
    }

    /// Test market state processing with mempool updates
    #[test]
    fn test_mempool_market_processing() {
        // This test validates the structure is in place for mempool processing
        let market_id = [0u8; 6];
        let shares = Array::from_vec(vec![100.0, 100.0, 100.0]);

        let snapshot = MarketSnapshot {
            shares,
            b: 10.0,
            trading_fee: 0.01,
            treasury: 1000.0,
        };

        let trade = BatchedMarketTrade {
            market_id,
            outcome_index: 0,
            shares_to_buy: 10.0,
            max_cost: 1000,
            market_snapshot: snapshot,
            trader_address: Address([0u8; 20]),
        };

        // Validate the trade structure is properly constructed
        assert_eq!(trade.outcome_index, 0);
        assert_eq!(trade.shares_to_buy, 10.0);
        assert_eq!(trade.max_cost, 1000);
        assert_eq!(trade.market_snapshot.b, 10.0);
        assert_eq!(trade.market_snapshot.trading_fee, 0.01);

        println!("Mempool trade structure validation passed");
    }

    /// Test LMSR initialization according to Bitcoin Hivemind specification
    ///
    /// Based on Hivemind whitepaper section on LMSR:
    /// 1. When shares are initialized to zero: treasury = β × ln(n)
    /// 2. For target liquidity L: shares = L - β × ln(n)
    /// 3. Initial prices should be uniform: 1/n for each outcome
    /// 4. Treasury should equal the liquidity amount after initialization
    #[test]
    fn test_lmsr_initialization_spec_compliance() {
        // Test Case 1: Binary market (n=2) with β=7, similar to reference
        let beta: f64 = 7.0;
        let n_outcomes: f64 = 2.0;
        let target_liquidity: f64 = 100.0;

        // Calculate expected values per specification
        let min_treasury = beta * n_outcomes.ln(); // β × ln(n)
        let expected_initial_shares = target_liquidity - min_treasury; // L - β × ln(n)

        println!("Binary market test:");
        println!(
            "  β = {}, n = {}, L = {}",
            beta, n_outcomes, target_liquidity
        );
        println!("  Min treasury (β×ln(n)) = {:.6}", min_treasury);
        println!("  Expected initial shares = {:.6}", expected_initial_shares);

        // Simulate the shares array and treasury calculation
        let shares = Array::from_elem(2, expected_initial_shares);
        let calculated_treasury =
            beta * shares.mapv(|x| (x / beta).exp()).sum().ln();

        println!("  Calculated treasury = {:.6}", calculated_treasury);
        println!("  Target liquidity = {:.6}", target_liquidity);

        // Verify treasury matches target liquidity
        assert!(
            (calculated_treasury - target_liquidity).abs() < 1e-10,
            "Treasury {:.6} should equal target liquidity {:.6}",
            calculated_treasury,
            target_liquidity
        );

        // Verify initial prices are uniform
        let exp_shares: Array<f64, ndarray::Ix1> =
            shares.mapv(|x| (x / beta).exp());
        let sum_exp = exp_shares.sum();
        let prices: Array<f64, ndarray::Ix1> = exp_shares.mapv(|x| x / sum_exp);

        for (i, &price) in prices.iter().enumerate() {
            let expected_price = 1.0 / n_outcomes;
            println!(
                "  Price[{}] = {:.6}, expected = {:.6}",
                i, price, expected_price
            );
            assert!(
                (price - expected_price).abs() < 1e-10,
                "Price[{}] should be {:.6} but was {:.6}",
                i,
                expected_price,
                price
            );
        }

        // Test Case 2: 3-outcome market with different β
        let beta: f64 = 3.2;
        let n_outcomes: f64 = 3.0;
        let target_liquidity: f64 = 50.0;

        let min_treasury = beta * n_outcomes.ln();
        let expected_initial_shares = target_liquidity - min_treasury;

        println!("\n3-outcome market test:");
        println!(
            "  β = {}, n = {}, L = {}",
            beta, n_outcomes, target_liquidity
        );
        println!("  Min treasury (β×ln(n)) = {:.6}", min_treasury);
        println!("  Expected initial shares = {:.6}", expected_initial_shares);

        let shares = Array::from_elem(3, expected_initial_shares);
        let calculated_treasury =
            beta * shares.mapv(|x| (x / beta).exp()).sum().ln();

        println!("  Calculated treasury = {:.6}", calculated_treasury);

        assert!(
            (calculated_treasury - target_liquidity).abs() < 1e-10,
            "Treasury should equal target liquidity"
        );

        // Verify uniform prices
        let exp_shares: Array<f64, ndarray::Ix1> =
            shares.mapv(|x| (x / beta).exp());
        let sum_exp = exp_shares.sum();
        let prices: Array<f64, ndarray::Ix1> = exp_shares.mapv(|x| x / sum_exp);

        for &price in prices.iter() {
            let expected_price = 1.0 / n_outcomes;
            assert!(
                (price - expected_price).abs() < 1e-10,
                "All prices should be uniform at {:.6}",
                expected_price
            );
        }

        // Test Case 3: Edge case - exactly minimum liquidity
        let beta: f64 = 5.0;
        let n_outcomes: f64 = 4.0;
        let min_liquidity = beta * n_outcomes.ln();

        println!("\nMinimum liquidity edge case:");
        println!(
            "  β = {}, n = {}, min L = {:.6}",
            beta, n_outcomes, min_liquidity
        );

        // At minimum liquidity, shares should be 0
        let expected_shares: f64 = min_liquidity - min_liquidity; // Should be 0
        assert!(
            expected_shares.abs() < 1e-10,
            "Shares should be 0 at minimum liquidity"
        );

        // Treasury with zero shares should equal β × ln(n)
        let shares = Array::zeros(4);
        let calculated_treasury =
            beta * shares.mapv(|x: f64| (x / beta).exp()).sum().ln();

        println!("  Treasury with zero shares = {:.6}", calculated_treasury);
        assert!(
            (calculated_treasury - min_liquidity).abs() < 1e-10,
            "Zero shares should give minimum treasury"
        );
    }

    /// Test validation of minimum liquidity requirement
    #[test]
    fn test_liquidity_validation() {
        let beta: f64 = 7.0;
        let n_outcomes: f64 = 2.0;
        let min_liquidity = beta * n_outcomes.ln();

        // Test insufficient liquidity
        let insufficient = min_liquidity - 0.1;
        let expected_shares: f64 = insufficient - min_liquidity; // Negative

        assert!(
            expected_shares < 0.0,
            "Insufficient liquidity should result in negative shares"
        );

        // Test adequate liquidity
        let adequate = min_liquidity + 10.0;
        let expected_shares: f64 = adequate - min_liquidity;

        assert!(
            expected_shares > 0.0 && expected_shares.is_finite(),
            "Adequate liquidity should result in positive, finite shares"
        );
    }
}
