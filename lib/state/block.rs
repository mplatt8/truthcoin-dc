//! Connect and disconnect blocks

use std::collections::HashSet;

use sneed::{RoTxn, RwTxn};

use crate::{
    state::{Error, State, UtxoManager, amm, error},
    types::{
        AmountOverflowError, Authorization, Body, FilledOutput,
        FilledOutputContent, FilledTransaction, GetAddress as _,
        GetBitcoinValue as _, Header, InPoint, OutPoint, OutputContent,
        SpentOutput, TxData, Verify as _,
    },
};

/// Validate a block, returning the merkle root and fees
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
            OutputContent::AmmLpToken(_) => {
                return Err(Error::BadCoinbaseOutputContent);
            }
        };
        let filled_output = FilledOutput {
            address: output.address,
            content: filled_content,
            memo: output.memo.clone(),
        };
        state.insert_utxo_with_address_index(rwtxn, &outpoint, &filled_output)?;
    }
    for transaction in &body.transactions {
        let filled_tx = state.fill_transaction(rwtxn, transaction)?;
        let txid = filled_tx.txid();
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
        let Some(filled_outputs) = filled_tx.filled_outputs() else {
            let err = error::FillTxOutputContents(Box::new(filled_tx));
            return Err(err.into());
        };
        for (vout, filled_output) in filled_outputs.iter().enumerate() {
            let outpoint = OutPoint::Regular {
                txid,
                vout: vout as u32,
            };
            state.insert_utxo_with_address_index(rwtxn, &outpoint, filled_output)?;
        }
        match &transaction.data {
            None => (),
            Some(TxData::AmmBurn { .. }) => {
                let () = amm::apply_burn(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmMint { .. }) => {
                let () = amm::apply_mint(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmSwap { .. }) => {
                let () = amm::apply_swap(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::ClaimDecisionSlot { .. }) => {
                let () = apply_claim_decision_slot(
                    state,
                    rwtxn,
                    &filled_tx,
                    mainchain_timestamp,
                    height,
                )?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = apply_create_market(
                    state,
                    rwtxn,
                    &filled_tx,
                    height,
                )?;
            }
            Some(TxData::CreateMarketDimensional { .. }) => {
                let () = apply_create_market_dimensional(
                    state,
                    rwtxn,
                    &filled_tx,
                    height,
                )?;
            }
        }
    }
    let block_hash = header.hash();
    state.tip.put(rwtxn, &(), &block_hash)?;
    state.height.put(rwtxn, &(), &height)?;
    state
        .mainchain_timestamp
        .put(rwtxn, &(), &mainchain_timestamp)?;

    // No longer purging old slots - they become ossified instead

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
            Some(TxData::AmmBurn { .. }) => {
                let () = amm::revert_burn(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmMint { .. }) => {
                let () = amm::revert_mint(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::AmmSwap { .. }) => {
                let () = amm::revert_swap(&state.amm_pools, rwtxn, &filled_tx)?;
            }
            Some(TxData::ClaimDecisionSlot { .. }) => {
                let () = revert_claim_decision_slot(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = revert_create_market(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarketDimensional { .. }) => {
                let () = revert_create_market_dimensional(state, rwtxn, &filled_tx)?;
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
    b: Option<f64>,
    trading_fee: Option<f64>,
) -> crate::state::MarketBuilder {
    if !description.is_empty() {
        builder = builder.with_description(description.to_string());
    }
    
    if let Some(tags) = tags.as_ref() {
        builder = builder.with_tags(tags.clone());
    }
    
    if let Some(b) = b {
        builder = builder.with_liquidity(b);
    }
    
    if let Some(fee) = trading_fee {
        builder = builder.with_fee(fee);
    }
    
    builder
}

/// Store market in database with consistent error handling
/// 
/// This provides a standardized way to store markets in the database while
/// maintaining consistent error reporting across all market creation types.
fn store_market_in_db(
    state: &State,
    rwtxn: &mut RwTxn,
    market: &crate::state::Market,
    market_type: &str,
) -> Result<(), Error> {
    state.markets().add_market(rwtxn, market)
        .map_err(|_| Error::InvalidSlotId {
            reason: format!("Failed to store {} market in database", market_type),
        })
}

fn apply_create_market(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
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

    let builder = match market_data.market_type.as_str() {
        "independent" => builder.add_decisions(slot_ids),
        "categorical" => builder.set_categorical(slot_ids, market_data.has_residual.unwrap_or(false)),
        _ => return Err(Error::InvalidSlotId {
            reason: format!("Invalid market type: {}", market_data.market_type),
        }),
    };

    let market = builder.build(height as u64, None, &decisions)
        .map_err(|e| Error::InvalidSlotId {
            reason: format!("Market creation failed: {}", e),
        })?;

    // Store market in database using common helper
    store_market_in_db(state, rwtxn, &market, "independent/categorical")?;

    Ok(())
}

fn apply_create_market_dimensional(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
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

    // Use the new dimensional specification
    let builder = builder.with_dimensions(dimension_specs);

    let market = builder.build(height as u64, None, &decisions)
        .map_err(|e| Error::InvalidSlotId {
            reason: format!("Dimensional market creation failed: {}", e),
        })?;

    // Store market in database using common helper
    store_market_in_db(state, rwtxn, &market, "dimensional")?;

    Ok(())
}

fn revert_create_market(
    _state: &State,
    _rwtxn: &mut RwTxn,
    _filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    // For now, market reversion is not implemented as markets are immutable once created
    // In the future, this could be enhanced to support market cancellation in specific cases
    Ok(())
}

fn revert_create_market_dimensional(
    _state: &State,
    _rwtxn: &mut RwTxn,
    _filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    // For now, dimensional market reversion is not implemented as markets are immutable once created
    // In the future, this could be enhanced to support market cancellation in specific cases
    Ok(())
}
