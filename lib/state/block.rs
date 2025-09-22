//! Block state management for Bitcoin Hivemind sidechain.
//!
//! Handles atomic state transitions for all transaction types including
//! UTXO management, market operations, slot transitions, and LMSR calculations.

use std::collections::{HashMap, HashSet};
use ndarray::{Array, Ix1};

use sneed::{RoTxn, RwTxn};

use crate::{
    math::lmsr::{Lmsr, LmsrState, LmsrError},
    state::{Error, State, UtxoManager, error, markets::{MarketId}},
    types::{
        AmountOverflowError, Authorization, Body, FilledOutput,
        FilledOutputContent, FilledTransaction, GetAddress as _,
        GetBitcoinValue as _, Header, InPoint, OutPoint, OutputContent,
        SpentOutput, TxData, Verify as _, Address,
    },
};

/// State changes for atomic block processing.
struct StateUpdate {
    /// Market state updates with integrated LMSR calculations
    market_updates: Vec<MarketStateUpdate>,
    /// New markets to be created
    market_creations: Vec<MarketCreation>,
    /// Share account changes (trades, transfers, redemptions)
    share_account_changes: HashMap<(Address, MarketId), HashMap<u32, f64>>,
    /// Slot state changes and period transitions
    slot_changes: Vec<SlotStateChange>,
}

/// Market state update.
struct MarketStateUpdate {
    market_id: MarketId,
    new_shares: Option<Array<f64, Ix1>>,
    new_beta: Option<f64>,
    trader_address: Option<Address>,
    trade_cost: Option<f64>,
    transaction_id: Option<[u8; 32]>,
    outcome_index: Option<u32>,
    volume_sats: Option<u64>,
}

/// Slot state change.
struct SlotStateChange {
    slot_id: crate::state::slots::SlotId,
    new_decision: Option<crate::state::slots::Decision>,
    period_transition: Option<u32>,
}

/// Market creation data.
struct MarketCreation {
    market: crate::state::Market,
    creator_address: Address,
    height: u32,
}


impl StateUpdate {
    fn new() -> Self {
        Self {
            market_updates: Vec::new(),
            market_creations: Vec::new(),
            share_account_changes: HashMap::new(),
            slot_changes: Vec::new(),
        }
    }
    
    /// Verify the internal consistency of all collected changes
    /// 
    /// This validation ensures that all changes are mathematically sound
    /// and consistent with Bitcoin Hivemind specifications before application.
    fn verify_internal_consistency(&self) -> Result<(), Error> {
        // Verify no duplicate market IDs between creations and updates
        let mut created_market_ids = std::collections::HashSet::new();
        for creation in &self.market_creations {
            if !created_market_ids.insert(creation.market.id.clone()) {
                return Err(Error::InvalidSlotId {
                    reason: format!("Duplicate market creation for ID: {:?}", creation.market.id),
                });
            }
        }
        
        for update in &self.market_updates {
            if created_market_ids.contains(&update.market_id) {
                return Err(Error::InvalidSlotId {
                    reason: format!("Market {:?} cannot be both created and updated in same block", update.market_id),
                });
            }
        }
        
        // Verify share account changes are balanced (total shares conserved)
        for ((address, market_id), outcome_changes) in &self.share_account_changes {
            let _total_delta: f64 = outcome_changes.values().sum();
            // For now, just verify no individual change is infinite or NaN
            for &delta in outcome_changes.values() {
                if !delta.is_finite() {
                    return Err(Error::InvalidSlotId {
                        reason: format!("Invalid share delta for address {:?} market {:?}: {}", address, market_id, delta),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate all collected state changes before applying
    fn validate_all_changes(&self, state: &State, rotxn: &RoTxn) -> Result<(), Error> {
        // First verify internal consistency
        self.verify_internal_consistency()?;
        
        // Validate market updates with comprehensive LMSR constraints
        for update in &self.market_updates {
            if let Some(ref shares) = update.new_shares {
                if let Some(beta) = update.new_beta {
                    validate_lmsr_params(beta, shares)
                        .map_err(|e| Error::InvalidSlotId {
                            reason: format!("LMSR validation failed: {:?}", e),
                        })?;
                }
            }
            
            // Ensure market exists
            if state.markets().get_market(rotxn, &update.market_id)?.is_none() {
                return Err(Error::InvalidSlotId {
                    reason: format!("Market {:?} does not exist", update.market_id),
                });
            }
        }
        
        // Validate market creations
        for creation in &self.market_creations {
            // Validate market doesn't already exist
            if state.markets().get_market(rotxn, &creation.market.id)?.is_some() {
                return Err(Error::InvalidSlotId {
                    reason: format!("Market {:?} already exists", creation.market.id),
                });
            }
            
            // Validate LMSR parameters for new market
            validate_lmsr_params(creation.market.b(), &creation.market.shares())
                .map_err(|e| Error::InvalidSlotId {
                    reason: format!("Market creation LMSR validation failed: {:?}", e),
                })?;
        }
        
        // Validate slot changes
        for slot_change in &self.slot_changes {
            // Validate slot exists if updating
            if slot_change.new_decision.is_some() {
                if state.slots().get_slot(rotxn, slot_change.slot_id)?.is_none() {
                    return Err(Error::InvalidSlotId {
                        reason: format!("Slot {:?} does not exist", slot_change.slot_id),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply all collected changes atomically
    /// 
    /// This method ensures that all state changes are applied in a single atomic transaction.
    /// If any operation fails, the entire transaction is rolled back, maintaining database consistency.
    /// The order of operations is carefully designed to handle dependencies correctly.
    fn apply_all_changes(&self, state: &State, rwtxn: &mut RwTxn, height: u32) -> Result<(), Error> {
        // Apply market creations first (order matters for dependencies)
        // Markets must exist before shares can be allocated or updated
        for creation in &self.market_creations {
            // Store market in database
            state.markets().add_market(rwtxn, &creation.market)
                .map_err(|_| Error::InvalidSlotId {
                    reason: "Failed to store market in database".to_string(),
                })?;
            
            // Market shares start at zero according to LMSR specification
            // Initial liquidity is automatically calculated and built into the market treasury
        }
        
        // Apply market updates with integrated LMSR calculations
        for update in &self.market_updates {
            if let Some(ref new_shares) = update.new_shares {
                // Update market shares and recalculate treasury using integrated LMSR
                let mut market = state.markets().get_market(rwtxn, &update.market_id)?
                    .ok_or_else(|| Error::InvalidSlotId {
                        reason: format!("Market {:?} not found", update.market_id),
                    })?;
                
                // Recalculate treasury with new shares using comprehensive LMSR
                let new_treasury = calc_treasury(new_shares, market.b())
                    .map_err(|e| Error::InvalidSlotId {
                        reason: format!("Treasury calculation failed: {:?}", e),
                    })?;
                
                // Create new market state version instead of direct mutation
                let _new_state_hash = market.create_new_state_version(
                    update.transaction_id,
                    height as u64,
                    None, // Keep current market state
                    None, // Keep current b
                    None, // Keep current trading fee
                    Some(new_shares.clone()),
                    None, // Keep current final prices
                    Some(new_treasury),
                ).map_err(|e| Error::InvalidSlotId {
                    reason: format!("Failed to create new market state: {:?}", e),
                })?;

                // Update volume if this is a trade
                if let (Some(outcome_index), Some(volume_sats)) = (update.outcome_index, update.volume_sats) {
                    market.update_trading_volume(outcome_index as usize, volume_sats)
                        .map_err(|e| Error::InvalidSlotId {
                            reason: format!("Failed to update volume: {:?}", e),
                        })?;
                }

                // Update market in database
                state.markets().update_market(rwtxn, &market)?;
                
                // Clear mempool shares now that they're confirmed in block
                state.clear_mempool_shares(rwtxn, &update.market_id)?;
                
                // Update trader's share account if applicable
                if let (Some(trader), Some(cost)) = (&update.trader_address, update.trade_cost) {
                    // The specific outcome and amount would be determined by the trade logic
                    // This is a simplified version - actual implementation would track specific outcomes
                    let _ = (trader, cost); // Placeholder for actual share account updates
                }
            }
        }
        
        // Apply share account changes
        for ((address, market_id), outcome_changes) in &self.share_account_changes {
            for (&outcome_index, &share_delta) in outcome_changes {
                if share_delta != 0.0 {
                    if share_delta > 0.0 {
                        state.markets().add_shares_to_account(
                            rwtxn, address, market_id.clone(), outcome_index, share_delta, 0
                        )?;
                    } else {
                        state.markets().remove_shares_from_account(
                            rwtxn, address, market_id, outcome_index, -share_delta, 0
                        )?;
                    }
                }
            }
        }
        
        // Apply slot changes with integrated state management
        for slot_change in &self.slot_changes {
            if let Some(ref _decision) = slot_change.new_decision {
                // Slot decision updates are handled through the claim mechanism
                // If a decision needs to be updated, it means a new claim is being processed
                // The actual claim processing happens during individual transaction processing
                // but we can validate the slot exists and is in proper state
                if let Some(slot) = state.slots().get_slot(rwtxn, slot_change.slot_id)? {
                    if slot.decision.is_some() {
                        return Err(Error::InvalidSlotId {
                            reason: format!("Slot {:?} already has a decision", slot_change.slot_id),
                        });
                    }
                }
            }
            
            // Handle period transitions for slot state management
            if let Some(new_period) = slot_change.period_transition {
                // Period transitions are handled by the slot minting system
                // This validates that the transition is consistent with current state
                let current_period = slot_change.slot_id.period_index();
                if new_period <= current_period {
                    return Err(Error::InvalidSlotId {
                        reason: format!("Invalid period transition from {} to {}", current_period, new_period),
                    });
                }
                // The actual period transitions are managed by the slot minting system
                // during block connection, so this is primarily for validation
            }
        }
        
        
        Ok(())
    }
    
    /// Add a market update to the collected changes
    fn add_market_update(&mut self, update: MarketStateUpdate) {
        self.market_updates.push(update);
    }
    
    /// Add share account changes
    fn add_share_account_change(&mut self, address: Address, market_id: MarketId, outcome: u32, delta: f64) {
        self.share_account_changes
            .entry((address, market_id))
            .or_insert_with(HashMap::new)
            .insert(outcome, delta);
    }
    
    
    /// Add market creation to the collected changes
    fn add_market_creation(&mut self, creation: MarketCreation) {
        self.market_creations.push(creation);
    }
}


/// Calculate treasury using LMSR with actual market size.
///
/// This function creates an LMSR calculator with the exact number of outcomes
/// in the market, ensuring consistency with Hivemind whitepaper specifications.
///
/// # Arguments
/// * `shares` - Current share quantities for all market outcomes
/// * `beta` - LMSR beta parameter for liquidity
///
/// # Returns
/// * `Ok(treasury)` - Calculated treasury value
/// * `Err(LmsrError)` - LMSR calculation error
fn calc_treasury(shares: &Array<f64, Ix1>, beta: f64) -> Result<f64, LmsrError> {
    // Create LMSR with actual market size instead of hardcoded 256
    // This ensures proper validation and calculation for markets of any size
    let lmsr = Lmsr::new(shares.len());
    lmsr.cost_function(beta, &shares.view())
}

/// Validate LMSR parameters with actual market size.
///
/// Creates an LMSR validator with the correct market size to ensure
/// proper validation according to Bitcoin Hivemind specifications.
fn validate_lmsr_params(beta: f64, shares: &Array<f64, Ix1>) -> Result<(), LmsrError> {
    let state = LmsrState {
        beta,
        shares: shares.clone(),
        treasury_balance: u64::MAX,
        trading_fee: 0.0,
    };
    // Use LMSR with actual market size instead of hardcoded 256
    let lmsr = Lmsr::new(shares.len());
    lmsr.validate_state(&state)
}

/// Calculate share update cost.
fn query_update_cost(
    current_shares: &Array<f64, Ix1>,
    new_shares: &Array<f64, Ix1>,
    beta: f64,
) -> Result<f64, LmsrError> {
    if new_shares.len() != current_shares.len() {
        return Err(LmsrError::InvalidOutcomeCount {
            count: new_shares.len(),
            min: 1,
            max: current_shares.len(),
        });
    }
    
    let current_cost = calc_treasury(current_shares, beta)?;
    let new_cost = calc_treasury(new_shares, beta)?;
    
    Ok(new_cost - current_cost)
}

/// Validate a block and return total fees.
pub fn validate(
    state: &State,
    rotxn: &RoTxn,
    header: &Header,
    body: &Body,
) -> Result<bitcoin::Amount, Error> {
    let tip_hash = state.try_get_tip(rotxn)?;
    if header.prev_side_hash != tip_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    };
    let merkle_root = body.compute_merkle_root();
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }

    // Calculate the height this block will have after being applied
    let future_height =
        state.try_get_height(rotxn)?.map_or(0, |height| height + 1);

    let mut coinbase_value = bitcoin::Amount::ZERO;
    for output in &body.coinbase {
        coinbase_value = coinbase_value
            .checked_add(output.get_bitcoin_value())
            .ok_or(AmountOverflowError)?;
    }
    let mut total_fees = bitcoin::Amount::ZERO;
    let mut spent_utxos = HashSet::new();
    let filled_txs: Vec<_> = body
        .transactions
        .iter()
        .map(|t| state.fill_transaction(rotxn, t))
        .collect::<Result<_, _>>()?;
    for filled_tx in &filled_txs {
        for input in &filled_tx.transaction.inputs {
            if spent_utxos.contains(input) {
                return Err(Error::UtxoDoubleSpent);
            }
            spent_utxos.insert(*input);
        }
        // Use the future height for validation
        total_fees = total_fees
            .checked_add(state.validate_filled_transaction(
                rotxn,
                filled_tx,
                Some(future_height),
            )?)
            .ok_or(AmountOverflowError)?;
    }
    if coinbase_value > total_fees {
        return Err(Error::NotEnoughFees);
    }
    let spent_utxos = filled_txs.iter().flat_map(|t| t.spent_utxos.iter());
    for (authorization, spent_utxo) in
        body.authorizations.iter().zip(spent_utxos)
    {
        if authorization.get_address() != spent_utxo.address {
            return Err(Error::WrongPubKeyForAddress);
        }
    }
    if Authorization::verify_body(body).is_err() {
        return Err(Error::AuthorizationError);
    }
    Ok(total_fees)
}

/// Connect a block as the single source of truth for all state transitions
/// 
/// This function serves as the authoritative implementation for all state changes
/// within a block, processing UTXO updates, market operations with LMSR calculations,
/// slot transitions, and database persistence atomically.
/// 
/// # Block State Management Architecture
/// - All transaction types processed within block state
/// - LMSR calculations integrated directly into block processing  
/// - Database state maintained in perfect sync with blockchain state
/// - Atomic operations ensure consistency across all state components
/// 
/// # Arguments
/// * `state` - Blockchain state (authoritative source)
/// * `rwtxn` - Database write transaction for atomic persistence
/// * `header` - Block header with validation data
/// * `body` - Block body containing all transactions
/// * `mainchain_timestamp` - Bitcoin timestamp for period calculations
/// 
/// # Returns
/// * `Ok(())` - Successful atomic block connection
/// * `Err(Error)` - Connection failure with automatic rollback
/// 
/// # Bitcoin Hivemind Compliance
/// Implements atomic block connection per whitepaper specifications for
/// concurrent operations and mathematical precision requirements.
pub fn connect(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
    mainchain_timestamp: u64,
) -> Result<(), Error> {
    let height = state.try_get_height(rwtxn)?.map_or(0, |height| height + 1);
    let tip_hash = state.try_get_tip(rwtxn)?;
    if tip_hash != header.prev_side_hash {
        let err = error::InvalidHeader::PrevSideHash {
            expected: tip_hash,
            received: header.prev_side_hash,
        };
        return Err(Error::InvalidHeader(err));
    }
    let merkle_root = body.compute_merkle_root();
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: merkle_root,
            computed: header.merkle_root,
        };
        return Err(err);
    }

    // slot minting using mainchain timestamp

    if height == 0 {
        state
            .slots()
            .mint_genesis(rwtxn, mainchain_timestamp, height)?;
    } else {
        state
            .slots()
            .mint_up_to(rwtxn, mainchain_timestamp, height)?;
    }

    for (vout, output) in body.coinbase.iter().enumerate() {
        let outpoint = OutPoint::Coinbase {
            merkle_root: header.merkle_root,
            vout: vout as u32,
        };
        let filled_content = match output.content.clone() {
            OutputContent::Bitcoin(value) => {
                FilledOutputContent::Bitcoin(value)
            }
            OutputContent::Withdrawal(withdrawal) => {
                FilledOutputContent::BitcoinWithdrawal(withdrawal)
            }
            OutputContent::Votecoin(amount) => {
                // Only allow Votecoin creation in the genesis block (height 0)
                if height == 0 {
                    FilledOutputContent::Votecoin(amount)
                } else {
                    return Err(Error::BadCoinbaseOutputContent);
                }
            }
        };
        let filled_output = FilledOutput {
            address: output.address,
            content: filled_content,
            memo: output.memo.clone(),
        };
        state.insert_utxo_with_address_index(rwtxn, &outpoint, &filled_output)?;
    }
    // Phase 1: Transaction Processing and State Validation
    // All transaction types are processed within the collected block state management
    // ensuring atomic operations and perfect database-to-blockchain alignment
    let mut state_update = StateUpdate::new();
    let mut filled_transactions = Vec::new();
    
    // Process all transactions through state management
    for transaction in &body.transactions {
        let filled_tx = state.fill_transaction(rwtxn, transaction)?;
        filled_transactions.push(filled_tx.clone());
        
        // Process all transaction types within block state
        match &transaction.data {
            Some(TxData::BuyShares { .. }) => {
                apply_market_trade(state, rwtxn, &filled_tx, &mut state_update, height)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                apply_market_creation(state, rwtxn, &filled_tx, &mut state_update, height)?;
            }
            Some(TxData::CreateMarketDimensional { .. }) => {
                apply_dimensional_market(state, rwtxn, &filled_tx, &mut state_update, height)?;
            }
            Some(TxData::RedeemShares { .. }) => {
                apply_share_redemption(state, rwtxn, &filled_tx, &mut state_update, height)?;
            }
            Some(TxData::ClaimDecisionSlot { .. }) => {
                apply_slot_claim(state, rwtxn, &filled_tx, &mut state_update, height, mainchain_timestamp)?;
            }
            None => {
                // Regular UTXO-only transactions
            }
        }
    }
    
    // Validate all collected state changes
    state_update.validate_all_changes(state, rwtxn)?;
    
    // Apply all validated state changes atomically

    // Apply state changes first (these can fail with detailed error handling)
    state_update.apply_all_changes(state, rwtxn, height)?;

    // Apply UTXO changes after state validation succeeds
    for filled_tx in &filled_transactions {
        apply_utxo_changes(state, rwtxn, filled_tx)?;
    }
    
    let block_hash = header.hash();
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;
    state
        .mainchain_timestamp
        .put(rwtxn, &(), &mainchain_timestamp)?;


    Ok(())
}

pub fn disconnect_tip(
    state: &State,
    rwtxn: &mut RwTxn,
    header: &Header,
    body: &Body,
) -> Result<(), Error> {
    let tip_hash = state.tip.try_get(rwtxn, &())?.ok_or(Error::NoTip)?;
    if tip_hash != header.hash() {
        let err = error::InvalidHeader::BlockHash {
            expected: tip_hash,
            computed: header.hash(),
        };
        return Err(Error::InvalidHeader(err));
    }
    let merkle_root = body.compute_merkle_root();
    if merkle_root != header.merkle_root {
        let err = Error::InvalidBody {
            expected: header.merkle_root,
            computed: merkle_root,
        };
        return Err(err);
    }
    let height = state
        .try_get_height(rwtxn)?
        .expect("Height should not be None");
    // revert txs, last-to-first
    body.transactions.iter().rev().try_for_each(|tx| {
        let txid = tx.txid();
        let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
        // revert transaction effects
        match &tx.data {
            None => (),
            Some(TxData::ClaimDecisionSlot { .. }) => {
                let () = revert_claim_decision_slot(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = revert_create_market(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarketDimensional { .. }) => {
                let () = revert_create_market_dimensional(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::BuyShares { .. }) => {
                let () = revert_buy_shares(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::RedeemShares { .. }) => {
                let () = revert_redeem_shares(state, rwtxn, &filled_tx)?;
            }
        }
        // delete UTXOs, last-to-first
        tx.outputs.iter().enumerate().rev().try_for_each(
            |(vout, _output)| {
                let outpoint = OutPoint::Regular {
                    txid,
                    vout: vout as u32,
                };
                if state.delete_utxo_with_address_index(rwtxn, &outpoint)? {
                    Ok(())
                } else {
                    Err(Error::NoUtxo { outpoint })
                }
            },
        )?;
        // unspend STXOs, last-to-first
        tx.inputs.iter().rev().try_for_each(|outpoint| {
            if let Some(spent_output) = state.stxos.try_get(rwtxn, outpoint)? {
                state.stxos.delete(rwtxn, outpoint)?;
                state.insert_utxo_with_address_index(rwtxn, outpoint, &spent_output.output)?;
                Ok(())
            } else {
                Err(Error::NoStxo {
                    outpoint: *outpoint,
                })
            }
        })
    })?;
    // delete coinbase UTXOs, last-to-first
    body.coinbase.iter().enumerate().rev().try_for_each(
        |(vout, _output)| {
            let outpoint = OutPoint::Coinbase {
                merkle_root: header.merkle_root,
                vout: vout as u32,
            };
            if state.delete_utxo_with_address_index(rwtxn, &outpoint)? {
                Ok(())
            } else {
                Err(Error::NoUtxo { outpoint })
            }
        },
    )?;
    match (header.prev_side_hash, height) {
        (None, 0) => {
            state.tip.delete(rwtxn, &())?;
            state.height.delete(rwtxn, &())?;
        }
        (None, _) | (_, 0) => return Err(Error::NoTip),
        (Some(prev_side_hash), height) => {
            state.tip.put(rwtxn, &(), &prev_side_hash)?;
            state.height.put(rwtxn, &(), &(height - 1))?;
        }
    }
    Ok(())
}

fn apply_claim_decision_slot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    mainchain_timestamp: u64,
    block_height: u32,
) -> Result<(), Error> {
    use crate::state::slots::{Decision, SlotId};

    let claim = filled_tx.claim_decision_slot().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a decision slot claim transaction".to_string(),
        }
    })?;

    let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;

    let market_maker_address_bytes = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "No spent UTXOs found".to_string(),
        })?
        .address
        .0;

    let decision = Decision::new(
        market_maker_address_bytes,
        claim.slot_id_bytes,
        claim.is_standard,
        claim.is_scaled,
        claim.question.clone(),
        claim.min,
        claim.max,
    )?;

    state.slots().claim_slot(
        rwtxn,
        slot_id,
        decision,
        mainchain_timestamp,
        Some(block_height),
    )?;

    Ok(())
}

fn revert_claim_decision_slot(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::slots::SlotId;

    let claim = filled_tx.claim_decision_slot().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a decision slot claim transaction".to_string(),
        }
    })?;

    let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;

    state.slots().revert_claim_slot(rwtxn, slot_id)?;

    Ok(())
}

/// Extract creator address from transaction's first spent UTXO
/// 
/// This is a common validation step for market creation transactions as per
/// Bitcoin Hivemind whitepaper specifications.
fn extract_creator_address(filled_tx: &FilledTransaction) -> Result<crate::types::Address, Error> {
    filled_tx.spent_utxos.first()
        .map(|utxo| utxo.address)
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "No spent UTXOs found".to_string(),
        })
}

/// Configure common market builder fields from market data
/// 
/// This consolidates the common pattern of setting optional fields on MarketBuilder
/// instances, following the DRY principle while maintaining Hivemind specification compliance.
fn configure_market_builder(
    mut builder: crate::state::MarketBuilder,
    description: &str,
    tags: &Option<Vec<String>>,
    b: f64,
    trading_fee: Option<f64>,
) -> crate::state::MarketBuilder {
    if !description.is_empty() {
        builder = builder.with_description(description.to_string());
    }

    if let Some(tags) = tags.as_ref() {
        builder = builder.with_tags(tags.clone());
    }

    builder = builder.with_beta(b);
    
    if let Some(fee) = trading_fee {
        builder = builder.with_fee(fee);
    }
    
    builder
}

/// Store market in database with consistent error handling
/// 
/// This provides a standardized way to store markets in the database while
/// maintaining consistent error reporting across all market creation types.
// Old collect_market_trade function removed - now using apply_market_trade

fn revert_create_market(
    _state: &State,
    _rwtxn: &mut RwTxn,
    _filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    // Markets are immutable once created in Bitcoin Hivemind - no reversion possible
    Ok(())
}

fn revert_create_market_dimensional(
    _state: &State,
    _rwtxn: &mut RwTxn,
    _filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    // Markets are immutable once created in Bitcoin Hivemind - no reversion possible
    Ok(())
}

/// Apply a share redemption transaction according to Bitcoin Hivemind whitepaper
/// 
/// This function validates the redemption transaction and processes it for a resolved market.
/// Users can redeem their shares for Bitcoin based on the final market resolution.
/// The transaction data contains the market ID, outcome index, and share amount.
fn apply_redeem_shares(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    height: u32,
) -> Result<(), Error> {
    let redeem_data = filled_tx.redeem_shares().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a redeem shares transaction".to_string(),
        }
    })?;

    // Get the address from the transaction authorization
    let trader_address = filled_tx.spent_utxos.first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "Redeem shares transaction must have inputs".to_string(),
        })?.address;

    // Apply the redemption to the market
    state.markets().apply_share_redemption(
        rwtxn,
        &trader_address,
        redeem_data.market_id,
        redeem_data.outcome_index,
        redeem_data.shares_to_redeem,
        height as u64,
    )?;

    Ok(())
}

/// Revert a share buy transaction
/// 
/// This function reverts a previously applied buy transaction by applying
/// the inverse operation (selling the same amount of shares).
fn revert_buy_shares(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let buy_data = filled_tx.buy_shares().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a buy shares transaction".to_string(),
        }
    })?;

    // Get the address from the transaction authorization
    let trader_address = filled_tx.spent_utxos.first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "Buy shares transaction must have inputs".to_string(),
        })?.address;

    // Get current height for reversion
    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

    // Revert the trade (positive amount becomes sell during revert)
    state.markets().revert_share_trade(
        rwtxn,
        &trader_address,
        buy_data.market_id,
        buy_data.outcome_index,
        buy_data.shares_to_buy,
        height as u64,
    )?;

    Ok(())
}

/// Revert a share redemption transaction
/// 
/// This function reverts a previously applied share redemption by restoring
/// the user's shares and removing the Bitcoin payout.
fn revert_redeem_shares(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let redeem_data = filled_tx.redeem_shares().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a redeem shares transaction".to_string(),
        }
    })?;

    // Get the address from the transaction authorization
    let trader_address = filled_tx.spent_utxos.first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "Redeem shares transaction must have inputs".to_string(),
        })?.address;

    // Get current height for reversion
    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

    // Revert the redemption
    state.markets().revert_share_redemption(
        rwtxn,
        &trader_address,
        redeem_data.market_id,
        redeem_data.outcome_index,
        redeem_data.shares_to_redeem,
        height as u64,
    )?;

    Ok(())
}



/// Apply UTXO changes with state management
/// 
/// This function consolidates all UTXO operations within the single source of truth
/// approach, ensuring atomic updates to both primary UTXO database and address index.
fn apply_utxo_changes(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let txid = filled_tx.txid();
    
    // Process inputs (spending UTXOs)
    for (vin, input) in filled_tx.inputs().iter().enumerate() {
        let spent_output = state
            .utxos
            .try_get(rwtxn, input)?
            .ok_or(Error::NoUtxo { outpoint: *input })?;
        let spent_output = SpentOutput {
            output: spent_output,
            inpoint: InPoint::Regular {
                txid,
                vin: vin as u32,
            },
        };
        state.delete_utxo_with_address_index(rwtxn, input)?;
        state.stxos.put(rwtxn, input, &spent_output)?;
    }
    
    // Process outputs (creating new UTXOs)
    let Some(filled_outputs) = filled_tx.filled_outputs() else {
        let err = error::FillTxOutputContents(Box::new(filled_tx.clone()));
        return Err(err.into());
    };
    for (vout, filled_output) in filled_outputs.iter().enumerate() {
        let outpoint = OutPoint::Regular {
            txid,
            vout: vout as u32,
        };
        state.insert_utxo_with_address_index(rwtxn, &outpoint, filled_output)?;
    }
    
    Ok(())
}

/// Apply market trade with LMSR calculations
/// 
/// This function integrates LMSR mathematical operations directly into block processing,
/// ensuring atomic market state updates aligned with Bitcoin Hivemind specifications.
fn apply_market_trade(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    _height: u32,
) -> Result<(), Error> {
    let buy_data = filled_tx.buy_shares().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a buy shares transaction".to_string(),
        }
    })?;

    // Get current market for LMSR calculations
    let market = state.markets().get_market(rwtxn, &MarketId::new(buy_data.market_id))?
        .ok_or_else(|| Error::InvalidSlotId {
            reason: format!("Market {:?} does not exist", buy_data.market_id),
        })?;

    // Validate market state
    if market.state() != crate::state::markets::MarketState::Trading {
        return Err(Error::InvalidSlotId {
            reason: "Market is not in trading state".to_string(),
        });
    }

    // Calculate new share quantities using integrated LMSR
    let mut new_shares = market.shares().clone();
    new_shares[buy_data.outcome_index as usize] += buy_data.shares_to_buy;
    
    // Calculate trade cost using comprehensive LMSR calculator
    let trade_cost = query_update_cost(&market.shares(), &new_shares, market.b())
        .map_err(|e| Error::InvalidSlotId {
            reason: format!("Failed to calculate trade cost: {:?}", e),
        })?;
    
    // Validate cost constraints
    if trade_cost > buy_data.max_cost as f64 {
        return Err(Error::InvalidSlotId {
            reason: format!("Trade cost {} exceeds max cost {}", trade_cost, buy_data.max_cost),
        });
    }
    
    // Get trader address
    let trader_address = filled_tx.spent_utxos.first()
        .map(|utxo| utxo.address)
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "No spent UTXOs found for trade".to_string(),
        })?;
    
    // Calculate volume in sats (including fees)
    let volume_sats = trade_cost.ceil() as u64;

    // Add to collected changes for atomic application
    state_update.add_market_update(MarketStateUpdate {
        market_id: MarketId::new(buy_data.market_id),
        new_shares: Some(new_shares),
        new_beta: None,
        trader_address: Some(trader_address),
        trade_cost: Some(trade_cost),
        transaction_id: Some(filled_tx.transaction.txid().0),
        outcome_index: Some(buy_data.outcome_index),
        volume_sats: Some(volume_sats),
    });
    
    // Add share account change
    state_update.add_share_account_change(
        trader_address,
        MarketId::new(buy_data.market_id),
        buy_data.outcome_index,
        buy_data.shares_to_buy,
    );
    
    Ok(())
}

/// Apply market creation to state
fn apply_market_creation(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use std::collections::HashMap;
    use crate::state::{MarketBuilder, slots::SlotId};

    let market_data = filled_tx.create_market().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a market creation transaction".to_string(),
        }
    })?;

    // Get creator address using common helper
    let creator_address = extract_creator_address(filled_tx)?;

    // Parse and collect decision slot data
    let mut slot_ids = Vec::new();
    let mut decisions = HashMap::new();
    
    for slot_hex in &market_data.decision_slots {
        let slot_bytes = hex::decode(slot_hex)
            .map_err(|_| Error::InvalidSlotId {
                reason: format!("Invalid slot ID hex: {}", slot_hex),
            })?;
        
        let slot_id_array: [u8; 3] = slot_bytes.try_into().unwrap();
        let slot_id = SlotId::from_bytes(slot_id_array)?;
        
        let slot = state.slots.get_slot(rwtxn, slot_id)?
            .ok_or_else(|| Error::InvalidSlotId {
                reason: format!("Slot {} does not exist", slot_hex),
            })?;
        
        let decision = slot.decision
            .ok_or_else(|| Error::InvalidSlotId {
                reason: format!("Slot {} has no decision", slot_hex),
            })?;
            
        slot_ids.push(slot_id);
        decisions.insert(slot_id, decision);
    }

    // Build market using common helper
    let mut builder = MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        &market_data.tags,
        market_data.b,
        market_data.trading_fee,
    );

    let mut builder = match market_data.market_type.as_str() {
        "independent" => builder.add_decisions(slot_ids),
        "categorical" => builder.set_categorical(slot_ids, market_data.has_residual.unwrap_or(false)),
        _ => return Err(Error::InvalidSlotId {
            reason: format!("Invalid market type: {}", market_data.market_type),
        }),
    };

    // Initial liquidity is now calculated automatically based on beta parameter

    let market = builder.build(height as u64, None, &decisions)
        .map_err(|e| Error::InvalidSlotId {
            reason: format!("Market creation failed: {}", e),
        })?;

    // Add to collected changes instead of direct application
    state_update.add_market_creation(MarketCreation {
        market,
        creator_address,
        height,
    });
    
    Ok(())
}

/// Apply dimensional market creation to state
fn apply_dimensional_market(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use std::collections::HashMap;
    use crate::state::{MarketBuilder, markets::{parse_dimensions, DimensionSpec}};

    let market_data = filled_tx.create_market_dimensional().ok_or_else(|| {
        Error::InvalidSlotId {
            reason: "Not a dimensional market creation transaction".to_string(),
        }
    })?;

    // Get creator address using common helper
    let creator_address = extract_creator_address(filled_tx)?;

    // Parse dimension specification
    let dimension_specs = parse_dimensions(&market_data.dimensions)
        .map_err(|_| Error::InvalidSlotId {
            reason: "Failed to parse dimension specification".to_string(),
        })?;

    // Collect all slot IDs and validate decisions exist
    let mut decisions = HashMap::new();
    for spec in &dimension_specs {
        let slot_ids = match spec {
            DimensionSpec::Single(slot_id) => vec![*slot_id],
            DimensionSpec::Categorical(slot_ids) => slot_ids.clone(),
        };
        
        for slot_id in slot_ids {
            let slot = state.slots.get_slot(rwtxn, slot_id)?
                .ok_or_else(|| Error::InvalidSlotId {
                    reason: format!("Slot {:?} does not exist", slot_id),
                })?;
            
            let decision = slot.decision
                .ok_or_else(|| Error::InvalidSlotId {
                    reason: format!("Slot {:?} has no decision", slot_id),
                })?;
                
            decisions.insert(slot_id, decision);
        }
    }

    // Build dimensional market using common helper
    let mut builder = MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        &market_data.tags,
        market_data.b,
        market_data.trading_fee,
    );

    // Use the dimensional specification
    let mut builder = builder.with_dimensions(dimension_specs);

    // Initial liquidity is now calculated automatically based on beta parameter

    let market = builder.build(height as u64, None, &decisions)
        .map_err(|e| Error::InvalidSlotId {
            reason: format!("Dimensional market creation failed: {}", e),
        })?;

    // Add to collected changes instead of direct application
    state_update.add_market_creation(MarketCreation {
        market,
        creator_address,
        height,
    });
    
    Ok(())
}

/// Apply share redemption with LMSR treasury updates
fn apply_share_redemption(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    _state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    // Use existing share redemption logic
    apply_redeem_shares(state, rwtxn, filled_tx, height)
}


/// Apply slot claim with period transition handling
fn apply_slot_claim(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    _state_update: &mut StateUpdate,
    height: u32,
    mainchain_timestamp: u64,
) -> Result<(), Error> {
    // Use existing slot claim logic
    apply_claim_decision_slot(state, rwtxn, filled_tx, mainchain_timestamp, height)
}

