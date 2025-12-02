use ndarray::{Array, Ix1};
use std::collections::{HashMap, HashSet};

use sneed::{RoTxn, RwTxn};

use crate::{
    math::lmsr::{LmsrError, LmsrService},
    state::{Error, State, UtxoManager, error, markets::MarketId},
    types::{
        Address, AmountOverflowError, Authorization, Body, FilledOutput,
        FilledOutputContent, FilledTransaction, GetAddress as _,
        GetBitcoinValue as _, Header, InPoint, OutPoint, OutputContent,
        SpentOutput, TxData, Verify as _,
    },
};

struct StateUpdate {
    market_updates: Vec<MarketStateUpdate>,
    market_creations: Vec<MarketCreation>,
    share_account_changes: HashMap<(Address, MarketId), HashMap<u32, f64>>,
    slot_changes: Vec<SlotStateChange>,
    vote_submissions: Vec<VoteSubmission>,
    voter_registrations: Vec<VoterRegistration>,
    reputation_updates: Vec<ReputationUpdate>,
}

struct MarketStateUpdate {
    market_id: MarketId,
    new_shares: Option<Array<f64, Ix1>>,
    new_beta: Option<f64>,
    trader_address: Option<Address>,
    trade_cost: Option<f64>,
    transaction_id: Option<[u8; 32]>,
    outcome_index: Option<u32>,
    volume_sats: Option<u64>,
    /// Trading fee collected for the market author (in satoshis)
    fee_sats: Option<u64>,
}

struct SlotStateChange {
    slot_id: crate::state::slots::SlotId,
    new_decision: Option<crate::state::slots::Decision>,
    period_transition: Option<u32>,
}

struct MarketCreation {
    market: crate::state::Market,
    creator_address: Address,
    height: u32,
}

struct VoteSubmission {
    vote: crate::state::voting::types::Vote,
}

struct VoterRegistration {
    address: crate::types::Address,
    initial_reputation: crate::state::voting::types::VoterReputation,
}

struct ReputationUpdate {
    address: crate::types::Address,
    updated_reputation: crate::state::voting::types::VoterReputation,
    old_reputation: crate::state::voting::types::VoterReputation,
}

impl StateUpdate {
    fn new() -> Self {
        Self {
            market_updates: Vec::new(),
            market_creations: Vec::new(),
            share_account_changes: HashMap::new(),
            slot_changes: Vec::new(),
            vote_submissions: Vec::new(),
            voter_registrations: Vec::new(),
            reputation_updates: Vec::new(),
        }
    }

    fn verify_internal_consistency(&self) -> Result<(), Error> {
        let mut created_market_ids = std::collections::HashSet::new();
        for creation in &self.market_creations {
            if !created_market_ids.insert(creation.market.id.clone()) {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "Duplicate market creation for ID: {:?}",
                        creation.market.id
                    ),
                });
            }
        }

        for update in &self.market_updates {
            if created_market_ids.contains(&update.market_id) {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "Market {:?} cannot be both created and updated in same block",
                        update.market_id
                    ),
                });
            }
        }

        for ((address, market_id), outcome_changes) in
            &self.share_account_changes
        {
            let _total_delta: f64 = outcome_changes.values().sum();
            for &delta in outcome_changes.values() {
                if !delta.is_finite() {
                    return Err(Error::InvalidSlotId {
                        reason: format!(
                            "Invalid share delta for address {:?} market {:?}: {}",
                            address, market_id, delta
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_all_changes(
        &self,
        state: &State,
        rotxn: &RoTxn,
    ) -> Result<(), Error> {
        self.verify_internal_consistency()?;

        for update in &self.market_updates {
            if let Some(ref shares) = update.new_shares {
                if let Some(beta) = update.new_beta {
                    crate::validation::MarketValidator::validate_lmsr_parameters(beta, shares)?;
                }
            }

            if state
                .markets()
                .get_market(rotxn, &update.market_id)?
                .is_none()
            {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "Market {:?} does not exist",
                        update.market_id
                    ),
                });
            }
        }

        for creation in &self.market_creations {
            if state
                .markets()
                .get_market(rotxn, &creation.market.id)?
                .is_some()
            {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "Market {:?} already exists",
                        creation.market.id
                    ),
                });
            }

            crate::validation::MarketValidator::validate_lmsr_parameters(
                creation.market.b(),
                &creation.market.shares(),
            )?;
        }

        for slot_change in &self.slot_changes {
            if slot_change.new_decision.is_some() {
                if state
                    .slots()
                    .get_slot(rotxn, slot_change.slot_id)?
                    .is_none()
                {
                    return Err(Error::InvalidSlotId {
                        reason: format!(
                            "Slot {:?} does not exist",
                            slot_change.slot_id
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn apply_all_changes(
        &self,
        state: &State,
        rwtxn: &mut RwTxn,
        height: u32,
    ) -> Result<(), Error> {
        for creation in &self.market_creations {
            state
                .markets()
                .add_market(rwtxn, &creation.market)
                .map_err(|_| Error::InvalidSlotId {
                    reason: "Failed to store market in database".to_string(),
                })?;
        }

        for update in &self.market_updates {
            if let Some(ref new_shares) = update.new_shares {
                let mut market = state
                    .markets()
                    .get_market(rwtxn, &update.market_id)?
                    .ok_or_else(|| Error::InvalidSlotId {
                        reason: format!(
                            "Market {:?} not found",
                            update.market_id
                        ),
                    })?;

                let new_treasury =
                    LmsrService::calculate_treasury(new_shares, market.b())
                        .map_err(|e| Error::InvalidSlotId {
                            reason: format!(
                                "Treasury calculation failed: {:?}",
                                e
                            ),
                        })?;

                // Use create_new_state_version_with_fees to accumulate trading fees for the market author
                let _new_state_hash = market
                    .create_new_state_version_with_fees(
                        update.transaction_id,
                        height as u64,
                        None,
                        None,
                        None,
                        Some(new_shares.clone()),
                        None,
                        Some(new_treasury),
                        update.fee_sats,
                    )
                    .map_err(|e| Error::InvalidSlotId {
                        reason: format!(
                            "Failed to create new market state: {:?}",
                            e
                        ),
                    })?;

                if let (Some(outcome_index), Some(volume_sats)) =
                    (update.outcome_index, update.volume_sats)
                {
                    market
                        .update_trading_volume(
                            outcome_index as usize,
                            volume_sats,
                        )
                        .map_err(|e| Error::InvalidSlotId {
                            reason: format!("Failed to update volume: {:?}", e),
                        })?;
                }

                state.markets().update_market(rwtxn, &market)?;

                state.clear_mempool_shares(rwtxn, &update.market_id)?;

                if let (Some(trader), Some(cost)) =
                    (&update.trader_address, update.trade_cost)
                {
                    let _ = (trader, cost);
                }
            }
        }

        for ((address, market_id), outcome_changes) in
            &self.share_account_changes
        {
            for (&outcome_index, &share_delta) in outcome_changes {
                if share_delta != 0.0 {
                    if share_delta > 0.0 {
                        state.markets().add_shares_to_account(
                            rwtxn,
                            address,
                            market_id.clone(),
                            outcome_index,
                            share_delta,
                            height as u64,
                        )?;
                    } else {
                        state.markets().remove_shares_from_account(
                            rwtxn,
                            address,
                            market_id,
                            outcome_index,
                            -share_delta,
                            height as u64,
                        )?;
                    }
                }
            }
        }

        for slot_change in &self.slot_changes {
            if let Some(ref _decision) = slot_change.new_decision {
                if let Some(slot) =
                    state.slots().get_slot(rwtxn, slot_change.slot_id)?
                {
                    if slot.decision.is_some() {
                        return Err(Error::InvalidSlotId {
                            reason: format!(
                                "Slot {:?} already has a decision",
                                slot_change.slot_id
                            ),
                        });
                    }
                }
            }

            if let Some(new_period) = slot_change.period_transition {
                let current_period = slot_change.slot_id.period_index();
                if new_period <= current_period {
                    return Err(Error::InvalidSlotId {
                        reason: format!(
                            "Invalid period transition from {} to {}",
                            current_period, new_period
                        ),
                    });
                }
            }
        }

        tracing::debug!(
            "apply_all_changes: Applying {} vote submissions",
            self.vote_submissions.len()
        );

        for submission in &self.vote_submissions {
            state
                .voting()
                .databases()
                .put_vote(rwtxn, &submission.vote)?;
        }

        for registration in &self.voter_registrations {
            state.voting().databases().put_voter_reputation(
                rwtxn,
                &registration.initial_reputation,
            )?;
        }

        for update in &self.reputation_updates {
            state
                .voting()
                .databases()
                .put_voter_reputation(rwtxn, &update.updated_reputation)?;
        }

        Ok(())
    }
    fn add_market_update(&mut self, update: MarketStateUpdate) {
        self.market_updates.push(update);
    }
    fn add_share_account_change(
        &mut self,
        address: Address,
        market_id: MarketId,
        outcome: u32,
        delta: f64,
    ) {
        self.share_account_changes
            .entry((address, market_id))
            .or_insert_with(HashMap::new)
            .insert(outcome, delta);
    }
    fn add_market_creation(&mut self, creation: MarketCreation) {
        self.market_creations.push(creation);
    }
    fn add_vote_submission(&mut self, vote: crate::state::voting::types::Vote) {
        self.vote_submissions.push(VoteSubmission { vote });
    }
    fn add_voter_registration(
        &mut self,
        address: crate::types::Address,
        reputation: crate::state::voting::types::VoterReputation,
    ) {
        self.voter_registrations.push(VoterRegistration {
            address,
            initial_reputation: reputation,
        });
    }
    fn add_reputation_update(
        &mut self,
        address: crate::types::Address,
        old_reputation: crate::state::voting::types::VoterReputation,
        updated_reputation: crate::state::voting::types::VoterReputation,
    ) {
        self.reputation_updates.push(ReputationUpdate {
            address,
            updated_reputation,
            old_reputation,
        });
    }
}
fn query_update_cost(
    current_shares: &Array<f64, Ix1>,
    new_shares: &Array<f64, Ix1>,
    beta: f64,
) -> Result<f64, LmsrError> {
    LmsrService::calculate_update_cost(current_shares, new_shares, beta)
}

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

    // Note: VoteCoin redistribution is now applied atomically with consensus
    // calculation in calculate_and_store_consensus(). This eliminates the
    // one-block window of inconsistent state that existed when redistribution
    // was pending.

    if height == 0 {
        state
            .slots()
            .mint_genesis(rwtxn, mainchain_timestamp, height)?;
    } else {
        state
            .slots()
            .mint_up_to(rwtxn, mainchain_timestamp, height)?;
    }

    for claimed_slot in state.slots().get_all_claimed_slots(rwtxn)? {
        let slot_id = claimed_slot.slot_id;
        let current_state =
            state.slots().get_slot_current_state(rwtxn, slot_id)?;

        if current_state == crate::state::slots::SlotState::Claimed {
            let voting_period = slot_id.voting_period();
            let period_info = crate::state::voting::period_calculator::calculate_voting_period(
                rwtxn,
                crate::state::voting::types::VotingPeriodId(voting_period),
                mainchain_timestamp,
                state.slots().get_config(),
                state.slots(),
                false,
            )?;

            if matches!(
                period_info.status,
                crate::state::voting::types::VotingPeriodStatus::Active
                    | crate::state::voting::types::VotingPeriodStatus::Closed
            ) {
                state.slots().transition_slot_to_voting(
                    rwtxn,
                    slot_id,
                    height as u64,
                    mainchain_timestamp,
                )?;
            }
        }
    }

    let all_periods = state.voting().get_all_periods(
        rwtxn,
        mainchain_timestamp,
        state.slots().get_config(),
        state.slots(),
    )?;

    for (period_id, period) in all_periods {
        if period.status
            == crate::state::voting::types::VotingPeriodStatus::Closed
        {
            let votes = state
                .voting()
                .databases()
                .get_votes_for_period(rwtxn, period_id)?;

            if !votes.is_empty() {
                let existing_outcomes = state
                    .voting()
                    .databases()
                    .get_consensus_outcomes_for_period(rwtxn, period_id)?;

                if existing_outcomes.is_empty() {
                    tracing::info!(
                        "Protocol: Automatically calculating consensus for period {} at block height {} (period ended at timestamp {})",
                        period_id.0,
                        height,
                        period.end_timestamp
                    );

                    state.voting().calculate_and_store_consensus(
                        rwtxn,
                        period_id,
                        state,
                        mainchain_timestamp,
                        height as u64,
                        state.slots(),
                    )?;

                    tracing::info!(
                        "Protocol: Successfully calculated consensus for period {} - pending redistribution will be applied in next block",
                        period_id.0
                    );
                }
            }
        }
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
        state.insert_utxo_with_address_index(
            rwtxn,
            &outpoint,
            &filled_output,
        )?;
    }
    let mut state_update = StateUpdate::new();
    let mut filled_transactions = Vec::new();

    for transaction in &body.transactions {
        let filled_tx = state.fill_transaction(rwtxn, transaction)?;
        filled_transactions.push(filled_tx.clone());

        match &transaction.data {
            Some(TxData::BuyShares { .. }) => {
                apply_market_trade(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::CreateMarket { .. }) => {
                apply_market_creation(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::CreateMarketDimensional { .. }) => {
                apply_dimensional_market(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::RedeemShares { .. }) => {
                apply_share_redemption(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::ClaimDecisionSlot { .. }) => {
                apply_slot_claim(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                    mainchain_timestamp,
                )?;
            }
            Some(TxData::SubmitVote { .. }) => {
                apply_submit_vote(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::SubmitVoteBatch { .. }) => {
                apply_submit_vote_batch(
                    state,
                    rwtxn,
                    &filled_tx,
                    &mut state_update,
                    height,
                )?;
            }
            Some(TxData::RegisterVoter { .. }) => {}
            None => {}
        }
    }

    state_update.validate_all_changes(state, rwtxn)?;

    state_update.apply_all_changes(state, rwtxn, height)?;

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
    body.transactions.iter().rev().try_for_each(|tx| {
        let txid = tx.txid();
        let filled_tx = state.fill_transaction_from_stxos(rwtxn, tx.clone())?;
        match &tx.data {
            None => (),
            Some(TxData::ClaimDecisionSlot { .. }) => {
                let () = revert_claim_decision_slot(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarket { .. }) => {
                let () = revert_create_market(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::CreateMarketDimensional { .. }) => {
                let () =
                    revert_create_market_dimensional(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::BuyShares { .. }) => {
                let () = revert_buy_shares(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::RedeemShares { .. }) => {
                let () = revert_redeem_shares(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::SubmitVote { .. }) => {
                let () = revert_submit_vote(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::SubmitVoteBatch { .. }) => {
                let () = revert_submit_vote_batch(state, rwtxn, &filled_tx)?;
            }
            Some(TxData::RegisterVoter { .. }) => {}
        }

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
        tx.inputs.iter().rev().try_for_each(|outpoint| {
            if let Some(spent_output) = state.stxos.try_get(rwtxn, outpoint)? {
                state.remove_stxo_caches(rwtxn, outpoint, &spent_output)?;
                state.stxos.delete(rwtxn, outpoint)?;
                state.insert_utxo_with_address_index(
                    rwtxn,
                    outpoint,
                    &spent_output.output,
                )?;
                Ok(())
            } else {
                Err(Error::NoStxo {
                    outpoint: *outpoint,
                })
            }
        })
    })?;
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

    if height > 0 {
        state
            .slots()
            .rollback_slot_states_to_height(rwtxn, (height - 1) as u64)?;

        tracing::info!(
            "Rolled back slot states to height {} during reorg",
            height - 1
        );
    }

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

    let slot_period = slot_id.period_index();
    let voting_period = slot_id.voting_period();

    tracing::debug!(
        "Claimed slot {} from period {} (votes in period {})",
        hex::encode(slot_id.as_bytes()),
        slot_period,
        voting_period
    );

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

fn extract_creator_address(
    filled_tx: &FilledTransaction,
) -> Result<crate::types::Address, Error> {
    filled_tx
        .spent_utxos
        .first()
        .map(|utxo| utxo.address)
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "No spent UTXOs found".to_string(),
        })
}

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

fn revert_create_market(
    _state: &State,
    _rwtxn: &mut RwTxn,
    _filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    Ok(())
}

fn revert_create_market_dimensional(
    _state: &State,
    _rwtxn: &mut RwTxn,
    _filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    Ok(())
}

fn apply_redeem_shares(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    height: u32,
) -> Result<(), Error> {
    let redeem_data =
        filled_tx
            .redeem_shares()
            .ok_or_else(|| Error::InvalidSlotId {
                reason: "Not a redeem shares transaction".to_string(),
            })?;

    let trader_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "Redeem shares transaction must have inputs".to_string(),
        })?
        .address;

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

fn revert_buy_shares(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let buy_data =
        filled_tx.buy_shares().ok_or_else(|| Error::InvalidSlotId {
            reason: "Not a buy shares transaction".to_string(),
        })?;

    let trader_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "Buy shares transaction must have inputs".to_string(),
        })?
        .address;

    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

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

fn revert_redeem_shares(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    let redeem_data =
        filled_tx
            .redeem_shares()
            .ok_or_else(|| Error::InvalidSlotId {
                reason: "Not a redeem shares transaction".to_string(),
            })?;

    let trader_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "Redeem shares transaction must have inputs".to_string(),
        })?
        .address;

    let height = state.try_get_height(rwtxn)?.unwrap_or(0);

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

fn apply_utxo_changes(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
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
        state.update_stxo_caches(rwtxn, input, &spent_output)?;
    }

    let Some(filled_outputs) = filled_tx.filled_outputs() else {
        let err = error::FillTxOutputContents(Box::new(filled_tx.clone()));
        return Err(err.into());
    };

    for (vout, filled_output) in filled_outputs.iter().enumerate() {
        let outpoint = OutPoint::Regular {
            txid,
            vout: vout as u32,
        };

        state.insert_utxo_with_address_index(
            rwtxn,
            &outpoint,
            filled_output,
        )?;
    }

    Ok(())
}

fn apply_market_trade(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    _height: u32,
) -> Result<(), Error> {
    let buy_data =
        filled_tx.buy_shares().ok_or_else(|| Error::InvalidSlotId {
            reason: "Not a buy shares transaction".to_string(),
        })?;

    let market = state
        .markets()
        .get_market(rwtxn, &buy_data.market_id)?
        .ok_or_else(|| Error::InvalidSlotId {
            reason: format!("Market {:?} does not exist", buy_data.market_id),
        })?;

    let market_state = market.compute_state(state.slots(), rwtxn)?;
    if !market_state.allows_trading() {
        return Err(Error::InvalidSlotId {
            reason: format!(
                "Cannot trade: market is in {:?} state",
                market_state
            ),
        });
    }

    let mut new_shares = market.shares().clone();
    new_shares[buy_data.outcome_index as usize] += buy_data.shares_to_buy;

    let base_cost =
        query_update_cost(&market.shares(), &new_shares, market.b()).map_err(
            |e| Error::InvalidSlotId {
                reason: format!("Failed to calculate trade cost: {:?}", e),
            },
        )?;

    // Calculate fee for market author
    let fee_amount = base_cost * market.trading_fee();
    let total_cost = base_cost + fee_amount;

    if total_cost > buy_data.max_cost as f64 {
        return Err(Error::InvalidSlotId {
            reason: format!(
                "Trade cost {} (base: {}, fee: {}) exceeds max cost {}",
                total_cost, base_cost, fee_amount, buy_data.max_cost
            ),
        });
    }

    let trader_address = filled_tx
        .spent_utxos
        .first()
        .map(|utxo| utxo.address)
        .ok_or_else(|| Error::InvalidSlotId {
            reason: "No spent UTXOs found for trade".to_string(),
        })?;

    let volume_sats = total_cost.ceil() as u64;
    let fee_sats = fee_amount.ceil() as u64;

    state_update.add_market_update(MarketStateUpdate {
        market_id: buy_data.market_id.clone(),
        new_shares: Some(new_shares),
        new_beta: None,
        trader_address: Some(trader_address),
        trade_cost: Some(total_cost),
        transaction_id: Some(filled_tx.transaction.txid().0),
        outcome_index: Some(buy_data.outcome_index),
        volume_sats: Some(volume_sats),
        fee_sats: Some(fee_sats),
    });

    state_update.add_share_account_change(
        trader_address,
        buy_data.market_id,
        buy_data.outcome_index,
        buy_data.shares_to_buy,
    );

    Ok(())
}
fn apply_market_creation(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{MarketBuilder, slots::SlotId};
    use std::collections::HashMap;

    let market_data =
        filled_tx
            .create_market()
            .ok_or_else(|| Error::InvalidSlotId {
                reason: "Not a market creation transaction".to_string(),
            })?;

    let creator_address = extract_creator_address(filled_tx)?;

    let mut slot_ids = Vec::new();
    let mut decisions = HashMap::new();

    for slot_hex in &market_data.decision_slots {
        let slot_bytes =
            hex::decode(slot_hex).map_err(|_| Error::InvalidSlotId {
                reason: format!("Invalid slot ID hex: {}", slot_hex),
            })?;

        let slot_id_array: [u8; 3] = slot_bytes.try_into().unwrap();
        let slot_id = SlotId::from_bytes(slot_id_array)?;

        let slot = state.slots.get_slot(rwtxn, slot_id)?.ok_or_else(|| {
            Error::InvalidSlotId {
                reason: format!("Slot {} does not exist", slot_hex),
            }
        })?;

        let decision = slot.decision.ok_or_else(|| Error::InvalidSlotId {
            reason: format!("Slot {} has no decision", slot_hex),
        })?;

        slot_ids.push(slot_id);
        decisions.insert(slot_id, decision);
    }

    let mut builder =
        MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        &market_data.tags,
        market_data.b,
        market_data.trading_fee,
    );

    let builder = match market_data.market_type.as_str() {
        "independent" => builder.add_decisions(slot_ids),
        "categorical" => builder.set_categorical(
            slot_ids,
            market_data.has_residual.unwrap_or(false),
        ),
        _ => {
            return Err(Error::InvalidSlotId {
                reason: format!(
                    "Invalid market type: {}",
                    market_data.market_type
                ),
            });
        }
    };

    let market =
        builder
            .build(height as u64, None, &decisions)
            .map_err(|e| Error::InvalidSlotId {
                reason: format!("Market creation failed: {}", e),
            })?;

    state_update.add_market_creation(MarketCreation {
        market,
        creator_address,
        height,
    });

    Ok(())
}
fn apply_dimensional_market(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        MarketBuilder,
        markets::{DimensionSpec, parse_dimensions},
    };
    use std::collections::HashMap;

    let market_data =
        filled_tx.create_market_dimensional().ok_or_else(|| {
            Error::InvalidSlotId {
                reason: "Not a dimensional market creation transaction"
                    .to_string(),
            }
        })?;

    let creator_address = extract_creator_address(filled_tx)?;

    let dimension_specs =
        parse_dimensions(&market_data.dimensions).map_err(|_| {
            Error::InvalidSlotId {
                reason: "Failed to parse dimension specification".to_string(),
            }
        })?;

    let mut decisions = HashMap::new();
    for spec in &dimension_specs {
        let slot_ids = match spec {
            DimensionSpec::Single(slot_id) => vec![*slot_id],
            DimensionSpec::Categorical(slot_ids) => slot_ids.clone(),
        };

        for slot_id in slot_ids {
            let slot =
                state.slots.get_slot(rwtxn, slot_id)?.ok_or_else(|| {
                    Error::InvalidSlotId {
                        reason: format!("Slot {:?} does not exist", slot_id),
                    }
                })?;

            let decision =
                slot.decision.ok_or_else(|| Error::InvalidSlotId {
                    reason: format!("Slot {:?} has no decision", slot_id),
                })?;

            decisions.insert(slot_id, decision);
        }
    }

    let mut builder =
        MarketBuilder::new(market_data.title.clone(), creator_address);
    builder = configure_market_builder(
        builder,
        &market_data.description,
        &market_data.tags,
        market_data.b,
        market_data.trading_fee,
    );

    let builder = builder.with_dimensions(dimension_specs);

    let market =
        builder
            .build(height as u64, None, &decisions)
            .map_err(|e| Error::InvalidSlotId {
                reason: format!("Dimensional market creation failed: {}", e),
            })?;

    state_update.add_market_creation(MarketCreation {
        market,
        creator_address,
        height,
    });

    Ok(())
}
fn apply_share_redemption(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    _state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    apply_redeem_shares(state, rwtxn, filled_tx, height)
}
fn apply_slot_claim(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    _state_update: &mut StateUpdate,
    height: u32,
    mainchain_timestamp: u64,
) -> Result<(), Error> {
    apply_claim_decision_slot(
        state,
        rwtxn,
        filled_tx,
        mainchain_timestamp,
        height,
    )
}

fn apply_submit_vote(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        slots::SlotId,
        voting::types::{Vote, VotingPeriodId},
    };

    let vote_data =
        filled_tx
            .submit_vote()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote transaction must have inputs".to_string(),
        })?
        .address;

    let decision_id = SlotId::from_bytes(vote_data.slot_id_bytes)?;

    let slot_claim_period = decision_id.period_index();
    let voting_period = decision_id.voting_period();
    let period_id = VotingPeriodId::new(voting_period);

    if vote_data.voting_period != voting_period {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Vote period mismatch: slot {} was claimed in period {} and must be voted on in period {}, but transaction specifies period {}",
                hex::encode(vote_data.slot_id_bytes),
                slot_claim_period,
                voting_period,
                vote_data.voting_period
            ),
        });
    }

    let timestamp =
        state.try_get_mainchain_timestamp(rwtxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            }
        })?;

    let vote_value = crate::validation::VoteValidator::convert_vote_value(
        vote_data.vote_value,
    );

    let vote = Vote::new(
        voter_address,
        period_id,
        decision_id,
        vote_value,
        timestamp,
        height as u64,
        filled_tx.txid().0,
    );

    state_update.add_vote_submission(vote);

    Ok(())
}

fn revert_submit_vote(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{slots::SlotId, voting::types::VotingPeriodId};

    let vote_data =
        filled_tx
            .submit_vote()
            .ok_or_else(|| Error::InvalidTransaction {
                reason: "Not a vote submission transaction".to_string(),
            })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote transaction must have inputs".to_string(),
        })?
        .address;

    let decision_id = SlotId::from_bytes(vote_data.slot_id_bytes)?;

    let voting_period = decision_id.voting_period();
    let period_id = VotingPeriodId::new(voting_period);

    state.voting().databases().delete_vote(
        rwtxn,
        period_id,
        voter_address,
        decision_id,
    )?;

    Ok(())
}

fn apply_submit_vote_batch(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
    state_update: &mut StateUpdate,
    height: u32,
) -> Result<(), Error> {
    use crate::state::{
        slots::SlotId,
        voting::types::{Vote, VotingPeriodId},
    };

    let batch_data = filled_tx.submit_vote_batch().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a vote batch submission transaction".to_string(),
        }
    })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote batch transaction must have inputs".to_string(),
        })?
        .address;

    let timestamp =
        state.try_get_mainchain_timestamp(rwtxn)?.ok_or_else(|| {
            Error::InvalidTransaction {
                reason: "No mainchain timestamp available".to_string(),
            }
        })?;

    let mut expected_voting_period: Option<u32> = None;

    for vote_item in &batch_data.votes {
        let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

        let voting_period = decision_id.voting_period();

        if let Some(expected) = expected_voting_period {
            if voting_period != expected {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch vote period mismatch: slot {} requires period {} but batch expects period {}",
                        hex::encode(vote_item.slot_id_bytes),
                        voting_period,
                        expected
                    ),
                });
            }
        } else {
            expected_voting_period = Some(voting_period);

            if batch_data.voting_period != voting_period {
                return Err(Error::InvalidTransaction {
                    reason: format!(
                        "Batch vote period mismatch: slots require period {} but transaction specifies period {}",
                        voting_period, batch_data.voting_period
                    ),
                });
            }
        }

        let period_id = VotingPeriodId::new(voting_period);

        let vote_value = crate::validation::VoteValidator::convert_vote_value(
            vote_item.vote_value,
        );

        let vote = Vote::new(
            voter_address,
            period_id,
            decision_id,
            vote_value,
            timestamp,
            height as u64,
            filled_tx.txid().0,
        );

        state_update.add_vote_submission(vote);
    }

    Ok(())
}

fn revert_submit_vote_batch(
    state: &State,
    rwtxn: &mut RwTxn,
    filled_tx: &FilledTransaction,
) -> Result<(), Error> {
    use crate::state::{slots::SlotId, voting::types::VotingPeriodId};

    let batch_data = filled_tx.submit_vote_batch().ok_or_else(|| {
        Error::InvalidTransaction {
            reason: "Not a vote batch submission transaction".to_string(),
        }
    })?;

    let voter_address = filled_tx
        .spent_utxos
        .first()
        .ok_or_else(|| Error::InvalidTransaction {
            reason: "Vote batch transaction must have inputs".to_string(),
        })?
        .address;

    for vote_item in &batch_data.votes {
        let decision_id = SlotId::from_bytes(vote_item.slot_id_bytes)?;

        let voting_period = decision_id.voting_period();
        let period_id = VotingPeriodId::new(voting_period);

        state.voting().databases().delete_vote(
            rwtxn,
            period_id,
            voter_address,
            decision_id,
        )?;
    }

    Ok(())
}
