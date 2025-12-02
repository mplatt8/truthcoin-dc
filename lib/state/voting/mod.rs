//! Bitcoin Hivemind Voting Module
//!
//! Implements the Bitcoin Hivemind voting mechanism with Votecoin economic stake integration.
//!
//! ## Voting Weight Formula
//! Final Voting Weight = Base Reputation × Votecoin Holdings Proportion

pub mod database;
pub mod period_calculator;
pub mod redistribution;
pub mod types;

use crate::state::{Error, slots::SlotId};
use database::VotingDatabases;
use sneed::{Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};
use types::{
    DecisionOutcome, DecisionResolution, Vote, VoteValue, VoterReputation,
    VotingPeriod, VotingPeriodId, VotingPeriodStats, VotingPeriodStatus,
};

#[derive(Clone)]
pub struct VotingSystem {
    databases: VotingDatabases,
    consensus_lock: std::sync::Arc<std::sync::Mutex<()>>,
}

impl VotingSystem {
    pub const NUM_DBS: u32 = VotingDatabases::NUM_DBS;

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        let databases = VotingDatabases::new(env, rwtxn)?;
        Ok(Self {
            databases,
            consensus_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
        })
    }

    pub fn databases(&self) -> &VotingDatabases {
        &self.databases
    }

    pub fn snapshot_votecoin_proportions(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_height: u64,
    ) -> Result<(), Error> {
        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        let voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();

        let votecoin_proportions =
            state.get_votecoin_proportions_batch(rwtxn, &voters)?;

        for voter_address in voters {
            if let Some(mut reputation) =
                self.databases.get_voter_reputation(rwtxn, voter_address)?
            {
                let proportion = votecoin_proportions
                    .get(&voter_address)
                    .copied()
                    .unwrap_or(0.0);

                reputation
                    .update_votecoin_proportion(proportion, current_height);
                self.databases.put_voter_reputation(rwtxn, &reputation)?;
            }
        }

        Ok(())
    }

    /// Calculate and store consensus outcomes for a voting period.
    ///
    /// SINGLE SOURCE OF TRUTH for consensus calculation and reputation updates.
    /// Uses SVD-based PCA as specified in Bitcoin Hivemind Section 4.2.
    pub(crate) fn calculate_and_store_consensus(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_timestamp: u64,
        current_height: u64,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        use crate::math::voting::{
            ReputationVector, SparseVoteMatrix, calculate_consensus,
        };

        let _consensus_guard = self.consensus_lock.lock().map_err(|_| {
            Error::DatabaseError("Failed to acquire consensus lock".to_string())
        })?;

        tracing::debug!(
            "calculate_and_store_consensus: Starting for period {} (lock acquired)",
            period_id.0
        );

        let existing_outcomes = self
            .databases
            .get_consensus_outcomes_for_period(rwtxn, period_id)?;
        if !existing_outcomes.is_empty() {
            tracing::warn!(
                "calculate_and_store_consensus: Consensus already calculated for period {} ({} outcomes exist), skipping",
                period_id.0,
                existing_outcomes.len()
            );
            return Ok(());
        }

        let all_votes =
            self.databases.get_votes_for_period(rwtxn, period_id)?;

        if all_votes.is_empty() {
            tracing::warn!(
                "calculate_and_store_consensus: No votes found for period {}, returning empty",
                period_id.0
            );
            return Ok(());
        }

        let mut voters_set = HashSet::new();
        let mut decisions_set = HashSet::new();
        let mut voter_reputations = HashMap::new();

        for vote_key in all_votes.keys() {
            if voters_set.insert(vote_key.voter_address) {
                if let Some(rep) = self
                    .databases
                    .get_voter_reputation(rwtxn, vote_key.voter_address)?
                {
                    voter_reputations.insert(vote_key.voter_address, rep);
                } else {
                    let timestamp = 0u64;
                    let default_rep = crate::state::voting::types::VoterReputation::new_default(
                        vote_key.voter_address,
                        timestamp,
                        period_id,
                    );
                    self.databases.put_voter_reputation(rwtxn, &default_rep)?;
                    voter_reputations
                        .insert(vote_key.voter_address, default_rep);
                }
            }
            decisions_set.insert(vote_key.decision_id);
        }

        if voter_reputations.is_empty() {
            return Ok(());
        }

        let voters: Vec<_> = voters_set.into_iter().collect();
        let decisions: Vec<_> = decisions_set.into_iter().collect();

        let mut vote_matrix = SparseVoteMatrix::new(voters, decisions);

        for (vote_key, vote_entry) in &all_votes {
            vote_matrix
                .set_vote(
                    vote_key.voter_address,
                    vote_key.decision_id,
                    vote_entry.to_f64(),
                )
                .map_err(|e| Error::InvalidTransaction {
                    reason: format!("Failed to set vote in matrix: {:?}", e),
                })?;
        }

        let reputation_vector =
            ReputationVector::from_voter_reputations(&voter_reputations);

        let consensus_result =
            calculate_consensus(&vote_matrix, &reputation_vector).map_err(
                |e| Error::InvalidTransaction {
                    reason: format!("Failed to calculate consensus: {:?}", e),
                },
            )?;

        let mut period_stats = self
            .databases
            .get_period_stats(rwtxn, period_id)?
            .unwrap_or_else(|| VotingPeriodStats::new(period_id, 0));

        period_stats.first_loading =
            Some(consensus_result.first_loading.clone());
        period_stats.explained_variance =
            Some(consensus_result.explained_variance);
        period_stats.certainty = Some(consensus_result.certainty);

        let mut reputation_changes = HashMap::new();

        for (voter_id, new_reputation) in
            consensus_result.updated_reputations.iter()
        {
            let mut voter_rep = self
                .databases
                .get_voter_reputation(rwtxn, *voter_id)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(*voter_id, 0, period_id)
                });

            let old_reputation = voter_rep.reputation;

            reputation_changes
                .insert(*voter_id, (old_reputation, *new_reputation));

            let consensus_txid = crate::types::Txid([0xff; 32]);
            voter_rep.reputation_history.push(
                old_reputation,
                consensus_txid,
                0,
            );

            voter_rep.reputation = *new_reputation;
            voter_rep.last_updated = 0;
            voter_rep.last_period = period_id;

            self.databases.put_voter_reputation(rwtxn, &voter_rep)?;
        }

        if !reputation_changes.is_empty() {
            period_stats.reputation_changes = Some(reputation_changes.clone());
        }

        self.databases.put_period_stats(rwtxn, &period_stats)?;

        for (slot_id, outcome_value) in &consensus_result.outcomes {
            // Handle unanimous abstention: if outcome is None, skip storing outcome
            // but still process the slot (it will be marked as unresolved)
            let Some(outcome_f64) = outcome_value else {
                tracing::warn!(
                    "Slot {} has unanimous abstention - no consensus outcome stored",
                    hex::encode(slot_id.as_bytes())
                );
                continue;
            };

            let mut resolution =
                DecisionResolution::new(*slot_id, period_id, 0, 1, 0, 0);
            resolution.mark_outcome_ready();

            let outcome = DecisionOutcome::new(
                *slot_id,
                period_id,
                *outcome_f64,
                0.0,
                1.0,
                1.0,
                all_votes.len() as u64,
                voter_reputations.values().map(|r| r.reputation).sum(),
                0,
                0,
                true,
                resolution,
            );

            self.databases.put_decision_outcome(rwtxn, &outcome)?;
        }

        let redistribution_summary =
            redistribution::redistribute_votecoin_after_consensus(
                state,
                rwtxn,
                period_id,
                &reputation_changes,
                current_timestamp,
                current_height,
            )?;

        // ATOMIC: Apply redistribution immediately in the same transaction
        redistribution::apply_votecoin_redistribution(
            state,
            rwtxn,
            &redistribution_summary,
            current_height,
        )?;

        let mut slots_in_period = Vec::new();
        for (slot_id, _) in consensus_result.outcomes.iter() {
            slots_in_period.push(*slot_id);
        }

        // Transition slots through the full lifecycle atomically
        // Slots with unanimous abstention (None outcome) are handled differently
        for slot_id in &slots_in_period {
            let outcome_opt = consensus_result.outcomes.get(slot_id).and_then(|v| *v);

            match outcome_opt {
                Some(outcome_value) => {
                    // Normal case: slot has consensus outcome
                    let consensus_outcome = outcome_value > 0.5;

                    slots_db.transition_slot_to_redistribution(
                        rwtxn,
                        *slot_id,
                        current_height,
                        current_timestamp,
                        consensus_outcome,
                    )?;

                    slots_db.transition_slot_to_resolved(
                        rwtxn,
                        *slot_id,
                        current_height,
                        current_timestamp,
                    )?;

                    slots_db.transition_slot_to_ossified(
                        rwtxn,
                        *slot_id,
                        current_height,
                        current_timestamp,
                    )?;

                    tracing::debug!(
                        "Atomically transitioned slot {} through Redistribution -> Resolved -> Ossified with outcome {}",
                        hex::encode(slot_id.as_bytes()),
                        consensus_outcome
                    );
                }
                None => {
                    // Unanimous abstention: mark slot as unresolved
                    // The slot remains in its current state - no transition occurs
                    // This allows the market to remain open or be handled by other mechanisms
                    tracing::warn!(
                        "Slot {} has unanimous abstention - not transitioning to resolved/ossified",
                        hex::encode(slot_id.as_bytes())
                    );
                }
            }
        }

        // Collect only the slots that were actually resolved (not abstained)
        let resolved_slot_ids: Vec<SlotId> = slots_in_period
            .iter()
            .filter(|slot_id| {
                consensus_result
                    .outcomes
                    .get(slot_id)
                    .and_then(|v| *v)
                    .is_some()
            })
            .copied()
            .collect();

        // Store redistribution record as already applied for auditability
        let mut period_redistribution = redistribution::PeriodRedistribution::new(
            period_id,
            resolved_slot_ids.clone(),
            redistribution_summary,
            current_height,
        );
        period_redistribution.mark_applied(current_height);

        self.databases
            .put_pending_redistribution(rwtxn, &period_redistribution)?;

        // Collect slot outcomes for final_prices calculation in market redemption
        let slot_outcomes: std::collections::HashMap<SlotId, f64> = consensus_result
            .outcomes
            .iter()
            .filter_map(|(slot_id, opt_outcome)| {
                opt_outcome.map(|v| (*slot_id, v))
            })
            .collect();

        // Update market ossification status (only for resolved slots)
        let ossified_slot_ids: std::collections::HashSet<_> =
            resolved_slot_ids.iter().copied().collect();
        let newly_ossified_markets = state
            .markets()
            .update_ossification_status(
                rwtxn,
                &ossified_slot_ids,
                &slot_outcomes,
                state.slots(),
            )?;

        if !newly_ossified_markets.is_empty() {
            tracing::info!(
                "Atomically transitioned {} markets to Ossified state: {:?}",
                newly_ossified_markets.len(),
                newly_ossified_markets
            );
        }

        let abstained_count = slots_in_period.len() - resolved_slot_ids.len();
        tracing::info!(
            "Atomically applied VoteCoin redistribution for period {} at height {}: {} resolved, {} abstained",
            period_id.0,
            current_height,
            resolved_slot_ids.len(),
            abstained_count
        );

        Ok(())
    }

    pub fn get_all_periods(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<HashMap<VotingPeriodId, VotingPeriod>, Error> {
        period_calculator::get_all_active_periods(
            rotxn,
            slots_db,
            config,
            current_timestamp,
            &self.databases,
        )
    }

    pub fn get_active_period(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<Option<VotingPeriod>, Error> {
        let all_periods =
            self.get_all_periods(rotxn, current_timestamp, config, slots_db)?;

        for period in all_periods.values() {
            if period.status == VotingPeriodStatus::Active {
                return Ok(Some(period.clone()));
            }
        }

        Ok(None)
    }

    pub fn cast_vote(
        &self,
        rwtxn: &mut RwTxn,
        voter_address: crate::types::Address,
        period_id: VotingPeriodId,
        decision_id: SlotId,
        value: VoteValue,
        timestamp: u64,
        block_height: u64,
        tx_hash: [u8; 32],
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        if !period_calculator::can_accept_votes(&period) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Period {:?} cannot accept votes (status: {:?}, timestamp: {})",
                    period_id, period.status, timestamp
                ),
            });
        }

        crate::validation::PeriodValidator::validate_decision_in_period(
            &period,
            decision_id,
        )?;

        let vote = Vote::new(
            voter_address,
            period_id,
            decision_id,
            value,
            timestamp,
            block_height,
            tx_hash,
        );

        self.databases.put_vote(rwtxn, &vote)?;

        Ok(())
    }

    pub fn get_votes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<(crate::types::Address, SlotId), VoteValue>, Error>
    {
        let vote_entries =
            self.databases.get_votes_for_period(rotxn, period_id)?;
        let mut votes = HashMap::new();

        for (key, entry) in vote_entries {
            votes.insert((key.voter_address, key.decision_id), entry.value);
        }

        Ok(votes)
    }

    pub fn get_vote_matrix(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<(crate::types::Address, SlotId), f64>, Error> {
        let vote_entries =
            self.databases.get_votes_for_period(rotxn, period_id)?;
        let mut matrix = HashMap::new();

        for (key, entry) in vote_entries {
            let vote_value = entry.to_f64();
            if !vote_value.is_nan() {
                matrix.insert((key.voter_address, key.decision_id), vote_value);
            }
        }

        Ok(matrix)
    }

    pub fn get_participation_stats(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(u64, u64, f64), Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;

        let decision_slots = period_calculator::get_decision_slots_for_period(
            rotxn, period_id, slots_db,
        )?;

        let total_votes = votes.len() as u64;
        let unique_voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();
        let total_voters = unique_voters.len() as u64;
        let total_decisions = decision_slots.len() as u64;

        let participation_rate = if total_voters > 0 && total_decisions > 0 {
            total_votes as f64 / (total_voters * total_decisions) as f64
        } else {
            0.0
        };

        Ok((total_voters, total_votes, participation_rate))
    }

    pub fn initialize_voter_reputation(
        &self,
        rwtxn: &mut RwTxn,
        voter_address: crate::types::Address,
        initial_reputation: f64,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Result<(), Error> {
        if self
            .databases
            .get_voter_reputation(rwtxn, voter_address)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voter {:?} already has reputation",
                    voter_address
                ),
            });
        }

        let reputation = VoterReputation::new(
            voter_address,
            initial_reputation,
            timestamp,
            period_id,
        );
        self.databases.put_voter_reputation(rwtxn, &reputation)?;

        Ok(())
    }

    pub fn get_reputation_weights(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<crate::types::Address, f64>, Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;
        let voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();
        let mut weights = HashMap::new();

        for voter_address in voters {
            let reputation = self
                .databases
                .get_voter_reputation(rotxn, voter_address)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(voter_address, 0, period_id)
                });

            weights.insert(voter_address, reputation.get_voting_weight());
        }

        Ok(weights)
    }

    /// Get voting weights with fresh Votecoin proportions from current UTXO set.
    /// Use this for all consensus calculations to ensure accuracy.
    pub fn get_fresh_reputation_weights(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_height: u64,
    ) -> Result<HashMap<crate::types::Address, f64>, Error> {
        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        let voters: HashSet<crate::types::Address> =
            votes.keys().map(|k| k.voter_address).collect();

        let votecoin_proportions =
            state.get_votecoin_proportions_batch(rwtxn, &voters)?;

        let mut weights = HashMap::new();

        for voter_address in voters {
            let mut reputation = self
                .databases
                .get_voter_reputation(rwtxn, voter_address)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(voter_address, 0, period_id)
                });

            if reputation.needs_votecoin_refresh(
                current_height,
                crate::math::voting::constants::VOTECOIN_STALENESS_BLOCKS,
            ) {
                let proportion = votecoin_proportions
                    .get(&voter_address)
                    .copied()
                    .unwrap_or(0.0);
                reputation
                    .update_votecoin_proportion(proportion, current_height);

                self.databases.put_voter_reputation(rwtxn, &reputation)?;
            }

            weights.insert(voter_address, reputation.get_voting_weight());
        }

        Ok(weights)
    }

    pub fn store_decision_outcome(
        &self,
        rwtxn: &mut RwTxn,
        outcome: DecisionOutcome,
    ) -> Result<(), Error> {
        if self
            .databases
            .get_decision_outcome(rwtxn, outcome.decision_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Outcome already exists for decision {:?} in period {:?}",
                    outcome.decision_id, outcome.period_id
                ),
            });
        }

        self.databases.put_decision_outcome(rwtxn, &outcome)?;
        Ok(())
    }

    pub fn resolve_period_decisions(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
        block_height: u64,
        state: &crate::state::State,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<Vec<DecisionOutcome>, Error> {
        use crate::types::Txid;

        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            current_timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        period_calculator::validate_transition(
            &period,
            VotingPeriodStatus::Resolved,
            current_timestamp,
        )?;

        let consensus_outcomes = self
            .databases
            .get_consensus_outcomes_for_period(rwtxn, period_id)?;

        if consensus_outcomes.is_empty() {
            return Err(Error::ConsensusNotYetCalculated(period_id));
        }

        let reputation_weights = self.get_fresh_reputation_weights(
            rwtxn,
            period_id,
            state,
            block_height,
        )?;

        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;

        let mut voter_reputations = HashMap::new();
        for vote_key in votes.keys() {
            if !voter_reputations.contains_key(&vote_key.voter_address) {
                if let Some(rep) = self
                    .databases
                    .get_voter_reputation(rwtxn, vote_key.voter_address)?
                {
                    voter_reputations.insert(vote_key.voter_address, rep);
                }
            }
        }

        let period_stats = self.databases.get_period_stats(rwtxn, period_id)?;
        let certainty = period_stats
            .and_then(|stats| stats.certainty)
            .unwrap_or(0.5);

        let mut outcomes = Vec::new();
        for decision_id in &period.decision_slots {
            let outcome_value =
                consensus_outcomes.get(decision_id).copied().unwrap_or(0.5);

            let decision_votes_count = votes
                .iter()
                .filter(|(key, _)| key.decision_id == *decision_id)
                .count();

            let mut resolution = DecisionResolution::new(
                *decision_id,
                period_id,
                period.end_timestamp,
                1,
                current_timestamp,
                block_height,
            );
            resolution.update_status(
                types::DecisionResolutionStatus::Resolved,
                current_timestamp,
                block_height,
                Some("Consensus reached via SVD".to_string()),
            );
            resolution.mark_outcome_ready();

            let outcome = DecisionOutcome::new(
                *decision_id,
                period_id,
                outcome_value,
                0.0,
                1.0,
                certainty,
                decision_votes_count as u64,
                reputation_weights.values().sum(),
                current_timestamp,
                block_height,
                true,
                resolution,
            );

            self.databases.put_decision_outcome(rwtxn, &outcome)?;
            outcomes.push(outcome);
        }

        Ok(outcomes)
    }

    pub fn get_period_outcomes(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<SlotId, DecisionOutcome>, Error> {
        self.databases.get_outcomes_for_period(rotxn, period_id)
    }

    pub fn calculate_period_statistics(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<VotingPeriodStats, Error> {
        let (total_voters, total_votes, participation_rate) =
            self.get_participation_stats(rotxn, period_id, config, slots_db)?;

        let has_outcomes = self.databases.has_consensus(rotxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rotxn,
            period_id,
            current_timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        let reputation_weights =
            self.get_reputation_weights(rotxn, period_id)?;
        let total_reputation_weight: f64 = reputation_weights.values().sum();

        let outcomes =
            self.databases.get_outcomes_for_period(rotxn, period_id)?;
        let consensus_decisions = outcomes
            .values()
            .filter(|outcome| outcome.is_consensus)
            .count() as u64;

        let mut stats = VotingPeriodStats::new(period_id, current_timestamp);
        stats.total_voters = total_voters;
        stats.total_votes = total_votes;
        stats.total_decisions = period.decision_slots.len() as u64;
        stats.avg_participation_rate = participation_rate;
        stats.total_reputation_weight = total_reputation_weight;
        stats.consensus_decisions = consensus_decisions;

        Ok(stats)
    }

    pub fn validate_consistency(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<String>, Error> {
        self.databases.check_consistency(rotxn)
    }

    pub fn get_system_stats(
        &self,
        rotxn: &RoTxn,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(u64, u64, u64, f64), Error> {
        let total_votes = self.databases.count_total_votes(rotxn)?;
        let all_voters = self.databases.get_all_voters(rotxn)?;
        let total_voters = all_voters.len() as u64;

        let (_, avg_reputation, _, _) =
            self.databases.get_reputation_stats(rotxn)?;

        let all_slots = slots_db.get_all_claimed_slots(rotxn)?;
        let mut unique_periods = std::collections::HashSet::new();
        for slot in all_slots {
            let voting_period = slot.slot_id.voting_period();
            unique_periods.insert(voting_period);
        }
        let total_periods = unique_periods.len() as u64;

        Ok((total_periods, total_votes, total_voters, avg_reputation))
    }
}
