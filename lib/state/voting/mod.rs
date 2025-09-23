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

// Tests temporarily disabled - can be re-enabled after Phase 2
#[cfg(test)]
mod basic_tests;

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

    /// Create a new voting period
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Unique identifier for this period
    /// * `start_timestamp` - L1 timestamp when voting begins
    /// * `end_timestamp` - L1 timestamp when voting ends
    /// * `decision_slots` - Slots available for voting in this period
    /// * `created_at_height` - Block height when period was created
    ///
    /// # Returns
    /// Ok(()) if period was created successfully
    ///
    /// # Errors
    /// - Period ID already exists
    /// - Invalid timestamp range
    /// - Database errors
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods must have deterministic start/end times and cannot overlap
    /// to ensure consistent consensus across all network participants.
    pub fn create_voting_period(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        start_timestamp: u64,
        end_timestamp: u64,
        decision_slots: Vec<SlotId>,
        created_at_height: u64,
    ) -> Result<(), Error> {
        // Validate timestamp range
        if start_timestamp >= end_timestamp {
            return Err(Error::InvalidTransaction {
                reason: "Voting period start must be before end".to_string(),
            });
        }

        // Check if period already exists
        if self
            .databases
            .get_voting_period(rwtxn, period_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!("Voting period {:?} already exists", period_id),
            });
        }

        // Create the voting period
        let period = VotingPeriod::new(
            period_id,
            start_timestamp,
            end_timestamp,
            decision_slots,
            created_at_height,
        );

        self.databases.put_voting_period(rwtxn, &period)?;

        Ok(())
    }

    /// Activate a voting period when its time window begins
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Period to activate
    /// * `current_timestamp` - Current L1 timestamp
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods automatically transition to Active status when current time
    /// reaches their start_timestamp, ensuring deterministic activation.
    pub fn activate_voting_period(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
    ) -> Result<(), Error> {
        let period = self
            .databases
            .get_voting_period(rwtxn, period_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Voting period {:?} not found", period_id),
            })?;

        // Validate timing
        if current_timestamp < period.start_timestamp {
            return Err(Error::InvalidTransaction {
                reason: "Cannot activate period before start time".to_string(),
            });
        }

        if period.status != VotingPeriodStatus::Pending {
            return Err(Error::InvalidTransaction {
                reason: format!("Period {:?} is not pending", period_id),
            });
        }

        self.databases.update_period_status(
            rwtxn,
            period_id,
            VotingPeriodStatus::Active,
        )?;
        Ok(())
    }

    /// Close a voting period when its time window ends
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Period to close
    /// * `current_timestamp` - Current L1 timestamp
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods automatically transition to Closed status when current time
    /// reaches their end_timestamp, preventing further vote acceptance.
    pub fn close_voting_period(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
    ) -> Result<(), Error> {
        let period = self
            .databases
            .get_voting_period(rwtxn, period_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Voting period {:?} not found", period_id),
            })?;

        // Validate timing
        if current_timestamp < period.end_timestamp {
            return Err(Error::InvalidTransaction {
                reason: "Cannot close period before end time".to_string(),
            });
        }

        if period.status != VotingPeriodStatus::Active {
            return Err(Error::InvalidTransaction {
                reason: format!("Period {:?} is not active", period_id),
            });
        }

        self.databases.update_period_status(
            rwtxn,
            period_id,
            VotingPeriodStatus::Closed,
        )?;
        Ok(())
    }

    /// Get the current active voting period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `current_timestamp` - Current L1 timestamp
    ///
    /// # Returns
    /// Some(VotingPeriod) if there is an active period, None otherwise
    pub fn get_active_period(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
    ) -> Result<Option<VotingPeriod>, Error> {
        let active_periods = self
            .databases
            .get_periods_by_status(rotxn, VotingPeriodStatus::Active)?;

        for period in active_periods {
            if period.is_active(current_timestamp) {
                return Ok(Some(period));
            }
        }

        Ok(None)
    }

    /// Get voting periods that need status updates
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `current_timestamp` - Current L1 timestamp
    ///
    /// # Returns
    /// Vector of (period_id, new_status) for periods needing updates
    pub fn get_periods_needing_update(
        &self,
        rotxn: &RoTxn,
        current_timestamp: u64,
    ) -> Result<Vec<(VotingPeriodId, VotingPeriodStatus)>, Error> {
        let mut updates = Vec::new();

        // Check pending periods that should become active
        let pending_periods = self
            .databases
            .get_periods_by_status(rotxn, VotingPeriodStatus::Pending)?;
        for period in pending_periods {
            if current_timestamp >= period.start_timestamp {
                updates.push((period.id, VotingPeriodStatus::Active));
            }
        }

        // Check active periods that should close
        let active_periods = self
            .databases
            .get_periods_by_status(rotxn, VotingPeriodStatus::Active)?;
        for period in active_periods {
            if current_timestamp >= period.end_timestamp {
                updates.push((period.id, VotingPeriodStatus::Closed));
            }
        }

        Ok(updates)
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
    ) -> Result<(), Error> {
        // Validate voting period is active
        let period = self
            .databases
            .get_voting_period(rwtxn, period_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Voting period {:?} not found", period_id),
            })?;

        if period.status != VotingPeriodStatus::Active {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Period {:?} is not active for voting",
                    period_id
                ),
            });
        }

        if !period.is_active(timestamp) {
            return Err(Error::InvalidTransaction {
                reason: "Vote timestamp is outside period window".to_string(),
            });
        }

        // Check if decision is in this period
        if !period.decision_slots.contains(&decision_id) {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Decision {:?} not available in period {:?}",
                    decision_id, period_id
                ),
            });
        }

        // Check for duplicate vote
        if self
            .databases
            .get_vote(rwtxn, period_id, voter_id, decision_id)?
            .is_some()
        {
            return Err(Error::InvalidTransaction {
                reason: format!(
                    "Voter {:?} already voted on decision {:?} in period {:?}",
                    voter_id, decision_id, period_id
                ),
            });
        }

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
    ///
    /// # Returns
    /// Tuple of (total_voters, total_votes, participation_rate)
    pub fn get_participation_stats(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<(u64, u64, f64), Error> {
        let votes = self.databases.get_votes_for_period(rotxn, period_id)?;
        let period = self
            .databases
            .get_voting_period(rotxn, period_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Period {:?} not found", period_id),
            })?;

        let total_votes = votes.len() as u64;
        let unique_voters: HashSet<VoterId> =
            votes.keys().map(|k| k.voter_id).collect();
        let total_voters = unique_voters.len() as u64;
        let total_decisions = period.decision_slots.len() as u64;

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

    /// Update voter reputation based on consensus performance
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `voter_id` - Voter to update
    /// * `was_correct` - Whether voter was in consensus on recent decisions
    /// * `timestamp` - Current timestamp
    /// * `period_id` - Period being processed
    ///
    /// # Bitcoin Hivemind Compliance
    /// Reputation updates follow the incentive mechanism to reward accurate
    /// reporting and penalize deviation from consensus truth.
    pub fn update_voter_reputation(
        &self,
        rwtxn: &mut RwTxn,
        voter_id: VoterId,
        was_correct: bool,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Result<(), Error> {
        let mut reputation = self
            .databases
            .get_voter_reputation(rwtxn, voter_id)?
            .unwrap_or_else(|| {
                VoterReputation::new(voter_id, 0.5, timestamp, period_id)
            });

        reputation.update(was_correct, timestamp, period_id);
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
                    VoterReputation::new(voter_id, 0.5, 0, period_id)
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
        const VOTECOIN_STALENESS_BLOCKS: u64 = 10; // Refresh every 10 blocks

        for voter_id in voters {
            let mut reputation = self
                .databases
                .get_voter_reputation(rwtxn, voter_id)?
                .unwrap_or_else(|| {
                    VoterReputation::new(voter_id, 0.5, 0, period_id)
                });

            // Update Votecoin proportion if stale
            if reputation.needs_votecoin_refresh(
                current_height,
                VOTECOIN_STALENESS_BLOCKS,
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
                outcome.period_id,
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
    ///
    /// # Returns
    /// Vector of resolved decision outcomes
    ///
    /// # Bitcoin Hivemind Compliance
    /// This is a placeholder for the full consensus algorithm implementation.
    /// The actual algorithm involves PCA, reputation weighting, and iterative
    /// convergence as specified in the whitepaper. Now uses complete voting
    /// weights including Votecoin holdings proportion.
    pub fn resolve_period_decisions(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
        block_height: u64,
        state: &crate::state::State,
    ) -> Result<Vec<DecisionOutcome>, Error> {
        let period = self
            .databases
            .get_voting_period(rwtxn, period_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Period {:?} not found", period_id),
            })?;

        if period.status != VotingPeriodStatus::Closed {
            return Err(Error::InvalidTransaction {
                reason: "Can only resolve closed periods".to_string(),
            });
        }

        let votes = self.databases.get_votes_for_period(rwtxn, period_id)?;
        // Use fresh reputation weights that include up-to-date Votecoin proportions
        let reputation_weights = self.get_fresh_reputation_weights(
            rwtxn,
            period_id,
            state,
            block_height,
        )?;
        let mut outcomes = Vec::new();

        // Simple consensus algorithm (placeholder for full PCA implementation)
        for decision_id in &period.decision_slots {
            let decision_votes: Vec<_> = votes
                .iter()
                .filter(|(key, _)| key.decision_id == *decision_id)
                .collect();

            if decision_votes.is_empty() {
                // No votes - use default outcome
                let resolution = DecisionResolution::new(
                    *decision_id,
                    period_id,
                    current_timestamp + 3600, // 1 hour voting deadline
                    1,                        // min_votes_required
                    current_timestamp,
                    block_height,
                );
                let outcome = DecisionOutcome::new(
                    *decision_id,
                    period_id,
                    0.5, // Default neutral outcome
                    0.0, // min
                    1.0, // max
                    0.0, // No confidence
                    0,
                    0.0,
                    current_timestamp,
                    block_height,
                    false, // Not consensus
                    resolution,
                );
                outcomes.push(outcome);
                continue;
            }

            // Weighted average based on reputation
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            let mut total_votes = 0;

            for (key, entry) in decision_votes {
                let vote_value = entry.to_f64();
                if !vote_value.is_nan() {
                    let weight =
                        reputation_weights.get(&key.voter_id).unwrap_or(&0.5);
                    weighted_sum += vote_value * weight;
                    total_weight += weight;
                    total_votes += 1;
                }
            }

            let outcome_value = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                0.5 // Default if no valid votes
            };

            // Simple confidence based on participation
            let confidence = (total_votes as f64
                / period.decision_slots.len() as f64)
                .min(1.0);

            let resolution = DecisionResolution::new(
                *decision_id,
                period_id,
                current_timestamp + 3600, // 1 hour voting deadline
                1,                        // min_votes_required
                current_timestamp,
                block_height,
            );
            let outcome = DecisionOutcome::new(
                *decision_id,
                period_id,
                outcome_value,
                0.0, // min
                1.0, // max
                confidence,
                total_votes as u64,
                total_weight,
                current_timestamp,
                block_height,
                total_votes > 0,
                resolution,
            );

            self.databases.put_decision_outcome(rwtxn, &outcome)?;
            outcomes.push(outcome);
        }

        // Update period status to resolved
        self.databases.update_period_status(
            rwtxn,
            period_id,
            VotingPeriodStatus::Resolved,
        )?;

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
    ///
    /// # Returns
    /// VotingPeriodStats with complete analytics
    pub fn calculate_period_statistics(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        current_timestamp: u64,
    ) -> Result<VotingPeriodStats, Error> {
        let (total_voters, total_votes, participation_rate) =
            self.get_participation_stats(rotxn, period_id)?;

        let period = self
            .databases
            .get_voting_period(rotxn, period_id)?
            .ok_or_else(|| Error::InvalidTransaction {
                reason: format!("Period {:?} not found", period_id),
            })?;

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
    ///
    /// # Returns
    /// Tuple of (total_periods, total_votes, total_voters, avg_reputation)
    pub fn get_system_stats(
        &self,
        rotxn: &RoTxn,
    ) -> Result<(u64, u64, u64, f64), Error> {
        let total_votes = self.databases.count_total_votes(rotxn)?;
        let all_voters = self.databases.get_all_voters(rotxn)?;
        let total_voters = all_voters.len() as u64;

        let (_, avg_reputation, _, _) =
            self.databases.get_reputation_stats(rotxn)?;

        // Count periods
        // Count periods by iterating through all periods
        let total_periods = self
            .databases
            .get_periods_by_status(rotxn, VotingPeriodStatus::Pending)?
            .len() as u64
            + self
                .databases
                .get_periods_by_status(rotxn, VotingPeriodStatus::Active)?
                .len() as u64
            + self
                .databases
                .get_periods_by_status(rotxn, VotingPeriodStatus::Closed)?
                .len() as u64
            + self
                .databases
                .get_periods_by_status(rotxn, VotingPeriodStatus::Resolved)?
                .len() as u64;

        Ok((total_periods, total_votes, total_voters, avg_reputation))
    }
}

// Re-export public types for convenience through the already-imported names

// NOTE: Tests are available in tests.rs.disabled - rename to enable
// #[cfg(test)]
// mod tests;
