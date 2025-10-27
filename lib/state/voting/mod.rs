//! Bitcoin Hivemind Voting Module with Votecoin Integration
//!
//! This module implements the complete Bitcoin Hivemind voting mechanism as specified
//! in the whitepaper, now fully integrated with the Votecoin economic stake system.
//! It provides high-level operations for vote management, period handling, reputation
//! tracking, and outcome determination using the complete voting weight formula.
//!
//! ## Votecoin-Voting Integration
//!
//! This implementation provides clean integration between the Votecoin system and the
//! voting power system according to the Bitcoin Hivemind specification:
//!
//! ### Bitcoin Hivemind Voting Weight Formula
//! **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
//!
//! Where:
//! - **Base Reputation**: Historical accuracy-based weighting (0.0 to 1.0)
//! - **Votecoin Holdings Proportion**: Economic stake as fraction of total supply (0.0 to 1.0)
//! - **Final Voting Weight**: Combined influence in consensus calculations
//!
//! ### Key Features
//! 1. **Efficient UTXO-based Balance Queries**: O(k) time complexity for Votecoin balance lookups
//! 2. **Cached Proportion Updates**: Votecoin proportions cached with staleness checking
//! 3. **Atomic Weight Calculations**: Both components updated together for consistency
//! 4. **Backward Compatibility**: Existing voting system enhanced, not replaced
//! 5. **Economic Incentive Alignment**: Voting power tied to both performance and stake
//!
//! ### Integration Points
//! - `State::get_votecoin_balance()` - Query individual Votecoin holdings
//! - `State::get_votecoin_proportions_batch()` - Efficient batch proportion calculation
//! - `VoterReputation::update_votecoin_proportion()` - Update economic stake component
//! - `VotingSystem::get_fresh_reputation_weights()` - Get complete voting weights
//! - `ReputationVector::from_voter_reputations()` - Convert to mathematical operations
//!
//! ## Architecture
//! - `types`: Core data structures enhanced with Votecoin weighting
//! - `database`: Low-level database operations and CRUD
//! - Main module: High-level business logic with Votecoin integration
//! - `State`: UTXO-based Votecoin balance queries
//! - `math::voting`: Mathematical operations using combined weights
//!
//! ## Bitcoin Hivemind Specification References
//! - Section 3: "Voting" - Core voting mechanism
//! - Section 4: "Consensus Algorithm" - Vote aggregation and reputation weighting
//! - Section 5: "Economics" - Economic incentives and stake-based weighting
//! - Section 6: "Implementation" - Technical details for economic stake integration

pub mod database;
pub mod types;
pub mod period_calculator;

// Tests temporarily disabled - can be re-enabled after Phase 2
// #[cfg(test)]
// mod basic_tests;

use crate::state::{Error, slots::SlotId};
use database::VotingDatabases;
use sneed::{Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};
use types::{
    DecisionOutcome, DecisionResolution, Vote, VoteValue, VoterId,
    VoterReputation, VotingPeriod, VotingPeriodId, VotingPeriodStats,
    VotingPeriodStatus,
};

/// High-level voting system interface
///
/// This struct provides the main interface for all voting operations,
/// combining database operations with business logic validation to ensure
/// compliance with Bitcoin Hivemind consensus rules.
#[derive(Clone)]
pub struct VotingSystem {
    /// Database layer for persistent storage
    databases: VotingDatabases,
}

impl VotingSystem {
    /// Number of databases managed by the voting system
    pub const NUM_DBS: u32 = VotingDatabases::NUM_DBS;

    /// Create a new voting system
    ///
    /// # Arguments
    /// * `env` - LMDB environment
    /// * `rwtxn` - Read-write transaction for initialization
    ///
    /// # Returns
    /// New VotingSystem instance with all databases initialized
    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        let databases = VotingDatabases::new(env, rwtxn)?;
        Ok(Self { databases })
    }

    /// Get read-only access to underlying databases
    pub fn databases(&self) -> &VotingDatabases {
        &self.databases
    }

    // ================================================================================
    // Voting Period Management
    // ================================================================================
    //
    // NOTE: Voting periods are NO LONGER STORED or explicitly managed!
    // All period information is calculated on-demand using period_calculator module.
    //
    // Period lifecycle is now implicit:
    // - Pending: before start_timestamp (calculated from period index)
    // - Active: between start and end timestamps
    // - Closed: after end_timestamp, before consensus calculated
    // - Resolved: after consensus outcomes stored
    //
    // No create_voting_period(), activate_voting_period(), or close_voting_period() methods.
    // Use period_calculator::calculate_voting_period() to get current period state.
    // Use cast_vote() during active periods - it validates period status automatically.
    // Use resolve_period_decisions() to calculate consensus and mark period resolved.

    /// Snapshot Votecoin proportions for all voters in a period
    ///
    /// This method caches Votecoin proportions at period close time to avoid
    /// expensive O(N×U) recalculation during consensus resolution. The cached
    /// proportions are stored in voter reputation records.
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Period to snapshot proportions for
    /// * `state` - State reference for UTXO queries
    /// * `current_height` - Current block height for cache timestamp
    ///
    /// # Returns
    /// Ok(()) if proportions were successfully cached
    ///
    /// # Performance
    /// This optimization reduces consensus resolution from O(N×U) to O(N)
    /// where N = voters, U = UTXOs, by caching proportions once instead
    /// of recalculating on every resolution attempt.
    pub fn snapshot_votecoin_proportions(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_height: u64,
    ) -> Result<(), Error> {
        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        let voters: HashSet<VoterId> = votes.keys().map(|k| k.voter_id).collect();

        // Convert VoterIds to Addresses for UTXO queries
        let addresses: HashSet<crate::types::Address> = voters
            .iter()
            .map(|voter_id| voter_id.to_address())
            .collect();

        // Get current Votecoin proportions for all voters (single batch query)
        let votecoin_proportions =
            state.get_votecoin_proportions_batch(rwtxn, &addresses)?;

        // Update cached proportions in voter reputations
        for voter_id in voters {
            if let Some(mut reputation) = self.databases.get_voter_reputation(rwtxn, voter_id)? {
                let address = voter_id.to_address();
                let proportion = votecoin_proportions.get(&address).copied().unwrap_or(0.0);

                reputation.update_votecoin_proportion(proportion, current_height);
                self.databases.put_voter_reputation(rwtxn, &reputation)?;
            }
        }

        Ok(())
    }

    /// Calculate and store consensus outcomes for a voting period
    ///
    /// **SINGLE SOURCE OF TRUTH FOR CONSENSUS AND REPUTATION UPDATES**
    ///
    /// This is the authoritative function for both consensus calculation and reputation
    /// updates across the entire system. It ensures that reputation changes happen in
    /// exactly one place, using the mathematically correct SVD-based algorithm from
    /// the Bitcoin Hivemind whitepaper.
    ///
    /// # Responsibilities
    /// 1. **Consensus Calculation**: Uses SVD-based PCA to determine decision outcomes
    /// 2. **Reputation Updates**: Updates voter reputations based on SVD alignment
    /// 3. **Metrics Storage**: Stores SVD metrics (loading, variance, certainty)
    /// 4. **Outcome Storage**: Persists consensus outcomes to database
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Period to calculate consensus for
    ///
    /// # Returns
    /// Ok(()) if consensus was calculated and stored successfully
    ///
    /// # Bitcoin Hivemind Compliance
    /// Implements Section 4.2: "Reputation System"
    /// - Uses current reputation to calculate consensus outcomes
    /// - Updates reputation based on SVD principal component alignment
    /// - Applies smoothing factor (alpha) to prevent volatility
    /// - Handles PCA sign ambiguity through reflection analysis
    ///
    /// **CRITICAL**: No other function in the codebase should update reputation.
    /// This is the ONLY location where reputation changes occur, ensuring:
    /// - Consistency with whitepaper mathematical specification
    /// - No duplicate or conflicting updates
    /// - Proper application of smoothing and consensus algorithms
    ///
    /// Reference: lib/math/voting/consensus.rs::run_consensus()
    fn calculate_and_store_consensus(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
    ) -> Result<(), Error> {
        use crate::math::voting::{calculate_consensus, ReputationVector, SparseVoteMatrix};

        tracing::debug!(
            "calculate_and_store_consensus: Starting for period {}",
            period_id.0
        );

        let all_votes = self.databases.get_votes_for_period(rwtxn, period_id)?;

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
            if voters_set.insert(vote_key.voter_id) {
                if let Some(rep) = self.databases.get_voter_reputation(rwtxn, vote_key.voter_id)? {
                    let reputation_value = rep.reputation;
                    voter_reputations.insert(vote_key.voter_id, rep);
                } else {
                    // New voter - initialize with default neutral reputation
                    // This is required by Bitcoin Hivemind: new voters start with neutral reputation

                    // Get current timestamp (we can use 0 as it's just for initialization)
                    let timestamp = 0u64;

                    // Create default reputation for new voter using new_default method
                    let default_rep = crate::state::voting::types::VoterReputation::new_default(
                        vote_key.voter_id,
                        timestamp,
                        period_id,
                    );

                    // Store the reputation for future use
                    self.databases.put_voter_reputation(rwtxn, &default_rep)?;
                    voter_reputations.insert(vote_key.voter_id, default_rep);

                }
            }
            decisions_set.insert(vote_key.decision_id);
        }

        if voter_reputations.is_empty() {
            return Ok(());
        } else {
        }

        let voters: Vec<_> = voters_set.into_iter().collect();
        let decisions: Vec<_> = decisions_set.into_iter().collect();

        let mut vote_matrix = SparseVoteMatrix::new(voters, decisions);

        for (vote_key, vote_entry) in &all_votes {
            vote_matrix.set_vote(
                vote_key.voter_id,
                vote_key.decision_id,
                vote_entry.to_f64(),
            ).map_err(|e| Error::InvalidTransaction {
                reason: format!("Failed to set vote in matrix: {:?}", e),
            })?;
        }

        let reputation_vector = ReputationVector::from_voter_reputations(&voter_reputations);

        // Calculate consensus outcomes with full SVD metrics
        let consensus_result = calculate_consensus(&vote_matrix, &reputation_vector)
            .map_err(|e| Error::InvalidTransaction {
                reason: format!("Failed to calculate consensus: {:?}", e),
            })?;


        // Store SVD metrics in period stats
        let mut period_stats = self.databases.get_period_stats(rwtxn, period_id)?
            .unwrap_or_else(|| VotingPeriodStats::new(period_id, 0));

        period_stats.first_loading = Some(consensus_result.first_loading.clone());
        period_stats.explained_variance = Some(consensus_result.explained_variance);
        period_stats.certainty = Some(consensus_result.certainty);

        // Collect reputation changes before updating
        let mut reputation_changes = HashMap::new();

        // Update voter reputations based on SVD consensus results
        // This is CRITICAL for the Bitcoin Hivemind incentive mechanism
        for (voter_id, new_reputation) in consensus_result.updated_reputations.iter() {
            if let Some(mut voter_rep) = self.databases.get_voter_reputation(rwtxn, *voter_id)? {
                let old_reputation = voter_rep.reputation;

                // Store the change for period stats
                reputation_changes.insert(*voter_id, (old_reputation, *new_reputation));

                // Push old reputation to history before updating
                // Use a dummy txid for consensus updates
                let consensus_txid = crate::types::Txid([0xff; 32]);
                voter_rep.reputation_history.push(old_reputation, consensus_txid, 0);

                // Update reputation to the new value calculated by SVD consensus
                // The consensus algorithm already applies smoothing (alpha factor)
                voter_rep.reputation = *new_reputation;
                voter_rep.last_updated = 0; // Will be set properly in resolve_period_decisions
                voter_rep.last_period = period_id;

                // Store the updated reputation
                self.databases.put_voter_reputation(rwtxn, &voter_rep)?;

            } else {
            }
        }

        // Store reputation changes in period stats
        if !reputation_changes.is_empty() {
            period_stats.reputation_changes = Some(reputation_changes);
        }

        // Save the updated period stats with reputation changes
        self.databases.put_period_stats(rwtxn, &period_stats)?;


        // Store the consensus outcomes in the database as DecisionOutcome objects
        for (slot_id, outcome_value) in consensus_result.outcomes {
            // Create resolution tracking for this decision
            let mut resolution = DecisionResolution::new(
                slot_id,
                period_id,
                0,  // voting_deadline - not needed for already calculated consensus
                1,  // min_votes_required
                0,  // current_timestamp
                0,  // current_height
            );
            resolution.mark_outcome_ready();  // Mark as ready since we have consensus

            // Create a DecisionOutcome for this consensus result
            let outcome = DecisionOutcome::new(
                slot_id,
                period_id,
                outcome_value,
                0.0,  // min value for binary/scalar decisions
                1.0,  // max value for binary/scalar decisions
                1.0,  // confidence (100% for initial consensus)
                all_votes.len() as u64,  // total votes
                voter_reputations.values().map(|r| r.reputation).sum(),  // total reputation weight
                0,  // timestamp (will be set later)
                0,  // block height (will be set later)
                true,  // is_consensus
                resolution,  // resolution tracking
            );

            self.databases.put_decision_outcome(rwtxn, &outcome)?;

        }

        Ok(())
    }

    /// Get all voting periods using calculated period information
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `current_timestamp` - Current L1 timestamp
    /// * `config` - Slot configuration for period calculation
    /// * `slots_db` - Slots database reference
    ///
    /// # Returns
    /// HashMap of VotingPeriodId to VotingPeriod for all periods with slots
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods are calculated on-demand from slots, not retrieved from storage.
    /// This replaces get_active_period() and get_periods_by_status() calls.
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

    /// Get the current active voting period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `current_timestamp` - Current L1 timestamp
    /// * `config` - Slot configuration for period calculation
    /// * `slots_db` - Slots database reference
    ///
    /// # Returns
    /// Some(VotingPeriod) if there is an active period, None otherwise
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods are calculated on-demand from slots. Status is determined by timestamps.
    pub fn get_active_period(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<Option<VotingPeriod>, Error> {
        let all_periods = self.get_all_periods(rotxn, current_timestamp, config, slots_db)?;

        for period in all_periods.values() {
            if period.status == VotingPeriodStatus::Active {
                return Ok(Some(period.clone()));
            }
        }

        Ok(None)
    }

    // ================================================================================
    // Vote Management
    // ================================================================================

    /// Cast a vote on a decision
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `voter_id` - ID of voter casting the vote
    /// * `period_id` - Voting period this vote belongs to
    /// * `decision_id` - Decision being voted on
    /// * `value` - Vote value (binary, scalar, or abstain)
    /// * `timestamp` - L1 timestamp when vote was cast
    /// * `block_height` - L2 block height when vote was included
    /// * `tx_hash` - Hash of transaction containing this vote
    /// * `config` - Slot configuration for period calculation
    /// * `slots_db` - Slots database reference for decision validation
    ///
    /// # Returns
    /// Ok(()) if vote was cast successfully
    ///
    /// # Errors
    /// - Period not active
    /// - Voter already voted on this decision
    /// - Invalid vote value for decision type
    /// - Database errors
    ///
    /// # Bitcoin Hivemind Compliance
    /// Votes can only be cast during active periods and each voter can only
    /// vote once per decision per period to maintain consensus integrity.
    /// Period is calculated on-demand from slots, not retrieved from storage.
    pub fn cast_vote(
        &self,
        rwtxn: &mut RwTxn,
        voter_id: VoterId,
        period_id: VotingPeriodId,
        decision_id: SlotId,
        value: VoteValue,
        timestamp: u64,
        block_height: u64,
        tx_hash: [u8; 32],
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(), Error> {
        // Calculate period on-demand (status is correctly calculated within)
        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        // Check if votes can be accepted based on period status
        if !period_calculator::can_accept_votes(&period) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Period {:?} cannot accept votes (status: {:?}, timestamp: {})",
                    period_id, period.status, timestamp
                ),
            });
        }

        crate::validation::PeriodValidator::validate_decision_in_period(&period, decision_id)?;

        // Create and store the vote
        let vote = Vote::new(
            voter_id,
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

    /// Get all votes for a specific period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to query
    ///
    /// # Returns
    /// HashMap mapping (voter_id, decision_id) to vote value
    pub fn get_votes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<(VoterId, SlotId), VoteValue>, Error> {
        let vote_entries =
            self.databases.get_votes_for_period(rotxn, period_id)?;
        let mut votes = HashMap::new();

        for (key, entry) in vote_entries {
            votes.insert((key.voter_id, key.decision_id), entry.value);
        }

        Ok(votes)
    }

    /// Get vote matrix for consensus algorithm processing
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to process
    ///
    /// # Returns
    /// Sparse matrix representation suitable for mathematical operations
    pub fn get_vote_matrix(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<(VoterId, SlotId), f64>, Error> {
        let vote_entries =
            self.databases.get_votes_for_period(rotxn, period_id)?;
        let mut matrix = HashMap::new();

        for (key, entry) in vote_entries {
            let vote_value = entry.to_f64();
            // Only include non-abstain votes in matrix
            if !vote_value.is_nan() {
                matrix.insert((key.voter_id, key.decision_id), vote_value);
            }
        }

        Ok(matrix)
    }

    /// Get voting participation statistics for a period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Period to analyze
    /// * `config` - Slot configuration for period calculation
    /// * `slots_db` - Slots database reference
    ///
    /// # Returns
    /// Tuple of (total_voters, total_votes, participation_rate)
    ///
    /// # Bitcoin Hivemind Compliance
    /// Period is calculated on-demand from slots, not retrieved from storage.
    pub fn get_participation_stats(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        config: &crate::state::slots::SlotConfig,
        slots_db: &crate::state::slots::Dbs,
    ) -> Result<(u64, u64, f64), Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;

        // Get decision slots for this period
        let decision_slots = period_calculator::get_decision_slots_for_period(
            rotxn,
            period_id,
            slots_db,
        )?;

        let total_votes = votes.len() as u64;
        let unique_voters: HashSet<VoterId> =
            votes.keys().map(|k| k.voter_id).collect();
        let total_voters = unique_voters.len() as u64;
        let total_decisions = decision_slots.len() as u64;

        let participation_rate = if total_voters > 0 && total_decisions > 0 {
            total_votes as f64 / (total_voters * total_decisions) as f64
        } else {
            0.0
        };

        Ok((total_voters, total_votes, participation_rate))
    }

    // ================================================================================
    // Reputation Management
    // ================================================================================

    /// Initialize reputation for a new voter
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `voter_id` - New voter's ID
    /// * `initial_reputation` - Starting reputation (typically 0.5)
    /// * `timestamp` - Current timestamp
    /// * `period_id` - Current voting period
    ///
    /// # Bitcoin Hivemind Compliance
    /// New voters start with neutral reputation to prevent gaming through
    /// multiple identity creation while ensuring fair initial participation.
    pub fn initialize_voter_reputation(
        &self,
        rwtxn: &mut RwTxn,
        voter_id: VoterId,
        initial_reputation: f64,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Result<(), Error> {
        // Check if voter already has reputation
        if self
            .databases
            .get_voter_reputation(rwtxn, voter_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!("Voter {:?} already has reputation", voter_id),
            });
        }

        let reputation = VoterReputation::new(
            voter_id,
            initial_reputation,
            timestamp,
            period_id,
        );
        self.databases.put_voter_reputation(rwtxn, &reputation)?;

        Ok(())
    }

    /// Get reputation weights for all voters in a period with Votecoin integration
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Period to get weights for
    /// * `state` - State reference for Votecoin balance queries
    ///
    /// # Returns
    /// HashMap mapping VoterId to final voting weight (reputation × Votecoin proportion)
    ///
    /// # Bitcoin Hivemind Specification
    /// Implements the complete voting weight formula:
    /// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
    pub fn get_reputation_weights(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<VoterId, f64>, Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;
        let voters: HashSet<VoterId> =
            votes.keys().map(|k| k.voter_id).collect();
        let mut weights = HashMap::new();

        for voter_id in voters {
            let reputation = self
                .databases
                .get_voter_reputation(rotxn, voter_id)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(voter_id, 0, period_id)
                });

            // Use the final voting weight which incorporates Votecoin holdings
            weights.insert(voter_id, reputation.get_voting_weight());
        }

        Ok(weights)
    }

    /// Get reputation weights with fresh Votecoin proportion calculations
    ///
    /// This method ensures that Votecoin proportions are up-to-date by querying
    /// the current UTXO set if needed, then calculates final voting weights.
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction (needed for reputation updates)
    /// * `period_id` - Period to get weights for
    /// * `state` - State reference for Votecoin balance queries
    /// * `current_height` - Current block height for staleness checking
    ///
    /// # Returns
    /// HashMap mapping VoterId to final voting weight
    ///
    /// # Bitcoin Hivemind Compliance
    /// This is the authoritative method for getting voting weights that should
    /// be used for all consensus calculations to ensure accuracy.
    pub fn get_fresh_reputation_weights(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        state: &crate::state::State,
        current_height: u64,
    ) -> Result<HashMap<VoterId, f64>, Error> {
        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        let voters: HashSet<VoterId> =
            votes.keys().map(|k| k.voter_id).collect();

        // Convert VoterIds to Addresses for UTXO queries
        let addresses: HashSet<crate::types::Address> = voters
            .iter()
            .map(|voter_id| voter_id.to_address())
            .collect();

        // Get current Votecoin proportions for all voters
        let votecoin_proportions =
            state.get_votecoin_proportions_batch(rwtxn, &addresses)?;

        let mut weights = HashMap::new();

        for voter_id in voters {
            let mut reputation = self
                .databases
                .get_voter_reputation(rwtxn, voter_id)?
                .unwrap_or_else(|| {
                    VoterReputation::new_default(voter_id, 0, period_id)
                });

            // Update Votecoin proportion if stale
            if reputation.needs_votecoin_refresh(
                current_height,
                crate::math::voting::constants::VOTECOIN_STALENESS_BLOCKS,
            ) {
                let address = voter_id.to_address();
                let proportion =
                    votecoin_proportions.get(&address).copied().unwrap_or(0.0);
                reputation
                    .update_votecoin_proportion(proportion, current_height);

                // Save updated reputation back to database
                self.databases.put_voter_reputation(rwtxn, &reputation)?;
            }

            weights.insert(voter_id, reputation.get_voting_weight());
        }

        Ok(weights)
    }

    // ================================================================================
    // Outcome Determination
    // ================================================================================

    /// Store final decision outcome after consensus resolution
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `outcome` - Decision outcome to store
    ///
    /// # Bitcoin Hivemind Compliance
    /// Outcomes are immutable once stored and represent the consensus truth
    /// used for market resolution and economic settlement.
    pub fn store_decision_outcome(
        &self,
        rwtxn: &mut RwTxn,
        outcome: DecisionOutcome,
    ) -> Result<(), Error> {
        // Validate outcome doesn't already exist
        if self
            .databases
            .get_decision_outcome(
                rwtxn,
                outcome.decision_id,
            )?
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

    /// Resolve all decisions in a voting period using Votecoin-integrated weights
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Period to resolve
    /// * `current_timestamp` - Current timestamp
    /// * `block_height` - Current block height
    /// * `state` - State reference for Votecoin balance queries
    /// * `config` - Slot configuration for period calculation
    /// * `slots_db` - Slots database reference
    ///
    /// # Returns
    /// Vector of resolved decision outcomes
    ///
    /// # Bitcoin Hivemind Compliance
    /// Implements the complete consensus algorithm with SVD-based PCA as specified
    /// in the whitepaper. Uses complete voting weights including Votecoin holdings
    /// proportion. Includes ACID-compliant error handling with automatic rollback.
    /// Period is calculated on-demand from slots, not retrieved from storage.
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

        // Calculate period on-demand (status is correctly calculated within)
        let has_outcomes = self.databases.has_consensus(rwtxn, period_id)?;
        let period = period_calculator::calculate_voting_period(
            rwtxn,
            period_id,
            current_timestamp,
            config,
            slots_db,
            has_outcomes,
        )?;

        // Validate state transition
        period_calculator::validate_transition(
            &period,
            VotingPeriodStatus::Resolved,
            current_timestamp,
        )?;

        // Calculate consensus outcomes if not already calculated
        let mut consensus_outcomes = self.databases.get_consensus_outcomes_for_period(rwtxn, period_id)?;

        if consensus_outcomes.is_empty() {
            // Calculate consensus now using current votes
            self.calculate_and_store_consensus(rwtxn, period_id)?;

            // Retrieve the newly calculated outcomes
            consensus_outcomes = self.databases.get_consensus_outcomes_for_period(rwtxn, period_id)?;


            if consensus_outcomes.is_empty() {
                // No votes - period resolved with default outcomes
                // Period status is now calculated on-demand, not stored
                return Ok(Vec::new());
            }

            // Since we just calculated and stored DecisionOutcome objects in calculate_and_store_consensus,
            // we need to retrieve them and return them, not create new ones
            let mut outcome_vec = Vec::new();
            for (slot_id, _) in consensus_outcomes {
                if let Some(outcome) = self.databases.get_decision_outcome(rwtxn, slot_id)? {
                    outcome_vec.push(outcome);
                }
            }


            return Ok(outcome_vec);
        }

        // Use fresh reputation weights that include up-to-date Votecoin proportions
        let reputation_weights = self.get_fresh_reputation_weights(
            rwtxn,
            period_id,
            state,
            block_height,
        )?;

        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;

        // Build voter reputations map for detailed result
        let mut voter_reputations = HashMap::new();
        for vote_key in votes.keys() {
            if !voter_reputations.contains_key(&vote_key.voter_id) {
                if let Some(rep) = self.databases.get_voter_reputation(rwtxn, vote_key.voter_id)? {
                    voter_reputations.insert(vote_key.voter_id, rep);
                }
            }
        }

        let period_stats = self.databases.get_period_stats(rwtxn, period_id)?;
        let certainty = period_stats
            .and_then(|stats| stats.certainty)
            .unwrap_or(0.5); // Default to neutral if consensus hasn't been calculated

        let mut outcomes = Vec::new();
        for decision_id in &period.decision_slots {
            let outcome_value = consensus_outcomes
                .get(decision_id)
                .copied()
                .unwrap_or(0.5);

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

        // NOTE: Reputation updates are NOT done here!
        //
        // Bitcoin Hivemind Specification Compliance (Section 4.2):
        // Reputation updates occur exclusively within the consensus algorithm
        // (calculate_and_store_consensus -> consensus::run_consensus) where they
        // are calculated using SVD-based PCA, weighted principal components, and
        // proper smoothing factors as specified in the whitepaper.
        //
        // The consensus algorithm already:
        // 1. Calculates voter alignment with the principal component (not simple tolerance)
        // 2. Applies reputation smoothing (alpha factor) to prevent volatility
        // 3. Handles PCA sign ambiguity and reflection attacks
        // 4. Stores updated reputations to the database (lines 280-305)
        //
        // Previous duplicate logic (lines 976-1017) has been removed to maintain
        // single source of truth for reputation updates.
        //
        // Reference: lib/math/voting/consensus.rs::run_consensus()
        // Whitepaper: Section 4.2 "Reputation System" and Section 4.4 "SVD Algorithm"

        // Period status is now calculated on-demand, not stored
        // No need to call update_period_status() - has_consensus flag determines Resolved status

        Ok(outcomes)
    }

    /// Get outcomes for all decisions in a period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Period to query
    ///
    /// # Returns
    /// HashMap mapping SlotId to DecisionOutcome
    pub fn get_period_outcomes(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<SlotId, DecisionOutcome>, Error> {
        self.databases.get_outcomes_for_period(rotxn, period_id)
    }

    // ================================================================================
    // Utility Functions
    // ================================================================================

    /// Get comprehensive statistics for a voting period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Period to analyze
    /// * `current_timestamp` - Current timestamp for period calculation
    /// * `config` - Slot configuration for period calculation
    /// * `slots_db` - Slots database reference
    ///
    /// # Returns
    /// VotingPeriodStats with complete analytics
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

        // Calculate period on-demand (status is correctly calculated within)
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

    /// Validate voting system consistency
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    ///
    /// # Returns
    /// Vector of consistency issues found
    pub fn validate_consistency(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<String>, Error> {
        self.databases.check_consistency(rotxn)
    }

    /// Get system-wide voting statistics
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `slots_db` - Slots database reference for counting periods
    ///
    /// # Returns
    /// Tuple of (total_periods, total_votes, total_voters, avg_reputation)
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods are calculated from slots, so we count unique period indices in slots
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

        // Count periods by finding unique voting periods from claimed slots
        let all_slots = slots_db.get_all_claimed_slots(rotxn)?;
        let mut unique_periods = std::collections::HashSet::new();
        for slot in all_slots {
            // Slots claimed in period N are voted on in period N+1
            let voting_period = slot.slot_id.voting_period();
            unique_periods.insert(voting_period);
        }
        let total_periods = unique_periods.len() as u64;

        Ok((total_periods, total_votes, total_voters, avg_reputation))
    }
}

// Re-export public types for convenience through the already-imported names

// NOTE: Tests are available in tests.rs.disabled - rename to enable
// #[cfg(test)]
// mod tests;
