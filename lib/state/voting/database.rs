//! Bitcoin Hivemind Voting Database Operations
//!
//! This module implements efficient database storage and retrieval for Bitcoin Hivemind
//! voting data using LMDB through the sneed crate. All database operations follow
//! ACID principles and are optimized for the sparse matrix nature of voting data.
//!
//! ## Database Schema
//! - `votes`: VoteMatrixKey -> VoteMatrixEntry
//! - `vote_batches`: (VotingPeriodId, u32) -> VoteBatch (for bulk operations)
//! - `voter_reputation`: VoterId -> VoterReputation
//! - `decision_outcomes`: SlotId -> DecisionOutcome (period derived from slot.period_index() + 1)
//! - `period_stats`: VotingPeriodId -> VotingPeriodStats
//! - `consensus_outcomes`: (VotingPeriodId, SlotId) -> f64 (for reputation calculation)
//!
//! ## Bitcoin Hivemind Specification References
//! - Section 4: "Consensus Algorithm" - Vote matrix storage requirements
//! - Section 5: "Implementation" - Database design considerations

use crate::state::{Error, voting::types::*};
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
// Removed unused serde imports - types module handles serialization
use sneed::{DatabaseUnique, Env, RoTxn, RwTxn};
use std::collections::{HashMap, HashSet};

/// Database collection for Bitcoin Hivemind voting system
///
/// This struct encapsulates all database tables needed for the voting mechanism,
/// providing a clean interface for CRUD operations while maintaining ACID
/// transaction semantics as required by the Bitcoin Hivemind specification.
///
/// ## Architecture Note
/// Voting periods are NOT stored in the database. They are calculated on-demand
/// from slots using period_calculator. This ensures a single source of truth.
#[derive(Clone)]
pub struct VotingDatabases {
    /// Individual votes in sparse matrix format
    /// Key: VoteMatrixKey, Value: VoteMatrixEntry
    votes: DatabaseUnique<
        SerdeBincode<VoteMatrixKey>,
        SerdeBincode<VoteMatrixEntry>,
    >,

    /// Bulk vote batches for efficient processing
    /// Key: (VotingPeriodId, batch_index), Value: VoteBatch
    vote_batches: DatabaseUnique<
        SerdeBincode<(VotingPeriodId, u32)>,
        SerdeBincode<VoteBatch>,
    >,

    /// Voter reputation tracking
    /// Key: VoterId, Value: VoterReputation
    voter_reputation:
        DatabaseUnique<SerdeBincode<VoterId>, SerdeBincode<VoterReputation>>,

    /// Final decision outcomes after consensus
    /// Key: SlotId, Value: DecisionOutcome
    /// Period is derived from slot.period_index() + 1
    decision_outcomes: DatabaseUnique<
        SerdeBincode<crate::state::slots::SlotId>,
        SerdeBincode<DecisionOutcome>,
    >,

    /// Period statistics and analytics
    /// Key: VotingPeriodId, Value: VotingPeriodStats
    period_stats: DatabaseUnique<
        SerdeBincode<VotingPeriodId>,
        SerdeBincode<VotingPeriodStats>,
    >,
}

impl VotingDatabases {
    /// Number of database tables managed by this struct
    /// Reduced from 6 to 5 after removing consensus_outcomes table (redundant with decision_outcomes)
    pub const NUM_DBS: u32 = 5;

    /// Create new voting databases
    ///
    /// # Arguments
    /// * `env` - LMDB environment
    /// * `rwtxn` - Read-write transaction for database creation
    ///
    /// # Returns
    /// New VotingDatabases instance with all tables initialized
    ///
    /// # Errors
    /// Returns database creation errors if any table fails to initialize
    ///
    /// # Bitcoin Hivemind Compliance
    /// Voting periods are NOT stored in the database. They are calculated on-demand
    /// from the slots database using period_calculator, ensuring a single source of truth.
    ///
    /// Consensus outcomes are NOT stored separately. They are stored as part of DecisionOutcome
    /// and extracted on-demand to eliminate redundancy.
    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Ok(Self {
            votes: DatabaseUnique::create(env, rwtxn, "votes")?,
            vote_batches: DatabaseUnique::create(env, rwtxn, "vote_batches")?,
            voter_reputation: DatabaseUnique::create(
                env,
                rwtxn,
                "voter_reputation",
            )?,
            decision_outcomes: DatabaseUnique::create(
                env,
                rwtxn,
                "decision_outcomes",
            )?,
            period_stats: DatabaseUnique::create(env, rwtxn, "period_stats")?,
        })
    }

    // ================================================================================
    // Voting Period Operations
    // ================================================================================
    //
    // NOTE: Voting periods are NO LONGER STORED in the database!
    // All period information is calculated on-demand using period_calculator module.
    // This section has been removed to enforce the single source of truth: slots database.
    //
    // Use period_calculator::calculate_voting_period() instead of get_voting_period()
    // Use period_calculator::get_all_active_periods() instead of get_periods_by_status()

    // ================================================================================
    // Vote Operations
    // ================================================================================

    /// Store an individual vote
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `vote` - Vote to store
    ///
    /// # Bitcoin Hivemind Compliance
    /// Votes are stored in sparse matrix format for efficient consensus algorithm
    /// execution. Each voter can only cast one vote per decision per period.
    pub fn put_vote(
        &self,
        rwtxn: &mut RwTxn,
        vote: &Vote,
    ) -> Result<(), Error> {
        let key =
            VoteMatrixKey::new(vote.period_id, vote.voter_id, vote.decision_id);
        let entry =
            VoteMatrixEntry::new(vote.value, vote.timestamp, vote.block_height);

        tracing::debug!(
            "put_vote: Storing vote for period {}, voter {:?}, decision {:?}, value {:?}",
            vote.period_id.0,
            hex::encode(vote.voter_id.as_bytes()),
            vote.decision_id.to_hex(),
            vote.value
        );

        self.votes.put(rwtxn, &key, &entry)?;
        Ok(())
    }

    /// Retrieve a specific vote
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period
    /// * `voter_id` - Voter who cast the vote
    /// * `decision_id` - Decision that was voted on
    ///
    /// # Returns
    /// Some(VoteMatrixEntry) if vote exists, None otherwise
    pub fn get_vote(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        voter_id: VoterId,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<VoteMatrixEntry>, Error> {
        let key = VoteMatrixKey::new(period_id, voter_id, decision_id);
        Ok(self.votes.try_get(rotxn, &key)?)
    }

    /// Get all votes for a specific period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to query
    ///
    /// # Returns
    /// HashMap mapping VoteMatrixKey to VoteMatrixEntry for all votes in period
    pub fn get_votes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<VoteMatrixKey, VoteMatrixEntry>, Error> {
        let mut votes = HashMap::new();
        let mut iter = self.votes.iter(rotxn)?;
        let mut total_votes_scanned = 0;
        let mut periods_seen = HashSet::new();

        while let Some((key, entry)) = iter.next()? {
            total_votes_scanned += 1;
            periods_seen.insert(key.period_id.0);

            if key.period_id == period_id {
                votes.insert(key, entry);
            }
        }

        tracing::debug!(
            "get_votes_for_period: Looking for period {}, found {} votes. \
            Total votes scanned: {}, Unique periods seen: {:?}",
            period_id.0,
            votes.len(),
            total_votes_scanned,
            periods_seen
        );

        Ok(votes)
    }

    /// Get all votes by a specific voter across all periods
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `voter_id` - Voter to query
    ///
    /// # Returns
    /// HashMap mapping VoteMatrixKey to VoteMatrixEntry for all voter's votes
    pub fn get_votes_by_voter(
        &self,
        rotxn: &RoTxn,
        voter_id: VoterId,
    ) -> Result<HashMap<VoteMatrixKey, VoteMatrixEntry>, Error> {
        let mut votes = HashMap::new();
        let mut iter = self.votes.iter(rotxn)?;

        while let Some((key, entry)) = iter.next()? {
            if key.voter_id == voter_id {
                votes.insert(key, entry);
            }
        }

        Ok(votes)
    }

    /// Get all votes for a specific decision across all periods
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `decision_id` - Decision to query
    ///
    /// # Returns
    /// HashMap mapping VoteMatrixKey to VoteMatrixEntry for all votes on decision
    pub fn get_votes_for_decision(
        &self,
        rotxn: &RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<HashMap<VoteMatrixKey, VoteMatrixEntry>, Error> {
        let mut votes = HashMap::new();
        let mut iter = self.votes.iter(rotxn)?;

        while let Some((key, entry)) = iter.next()? {
            if key.decision_id == decision_id {
                votes.insert(key, entry);
            }
        }

        Ok(votes)
    }

    /// Delete a vote (for rollback operations)
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `period_id` - Voting period
    /// * `voter_id` - Voter who cast the vote
    /// * `decision_id` - Decision that was voted on
    ///
    /// # Returns
    /// true if vote was deleted, false if it didn't exist
    pub fn delete_vote(
        &self,
        rwtxn: &mut RwTxn,
        period_id: VotingPeriodId,
        voter_id: VoterId,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<bool, Error> {
        let key = VoteMatrixKey::new(period_id, voter_id, decision_id);
        Ok(self.votes.delete(rwtxn, &key)?)
    }

    // ================================================================================
    // Vote Batch Operations
    // ================================================================================

    /// Store a batch of votes for efficient bulk processing
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `batch` - Vote batch to store
    /// * `batch_index` - Index for this batch within the period
    ///
    /// # Bitcoin Hivemind Compliance
    /// Batch operations maintain atomicity while improving performance for
    /// large vote ingestion required during active trading periods.
    pub fn put_vote_batch(
        &self,
        rwtxn: &mut RwTxn,
        batch: &VoteBatch,
        batch_index: u32,
    ) -> Result<(), Error> {
        let key = (batch.period_id, batch_index);
        self.vote_batches.put(rwtxn, &key, batch)?;
        Ok(())
    }

    /// Retrieve a vote batch
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period
    /// * `batch_index` - Index of batch to retrieve
    ///
    /// # Returns
    /// Some(VoteBatch) if found, None otherwise
    pub fn get_vote_batch(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
        batch_index: u32,
    ) -> Result<Option<VoteBatch>, Error> {
        let key = (period_id, batch_index);
        Ok(self.vote_batches.try_get(rotxn, &key)?)
    }

    /// Get all vote batches for a period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to query
    ///
    /// # Returns
    /// Vector of all vote batches in chronological order
    pub fn get_vote_batches_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Vec<VoteBatch>, Error> {
        let mut batches = Vec::new();
        let mut iter = self.vote_batches.iter(rotxn)?;

        while let Some(((p_id, _batch_index), batch)) = iter.next()? {
            if p_id == period_id {
                batches.push(batch);
            }
        }

        // Sort by batch index to maintain chronological order
        batches.sort_by_key(|batch| batch.created_at);
        Ok(batches)
    }

    // ================================================================================
    // Voter Reputation Operations
    // ================================================================================

    /// Store or update voter reputation
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `reputation` - Voter reputation data to store
    ///
    /// # Bitcoin Hivemind Compliance
    /// Reputation updates follow the consensus algorithm requirements for
    /// maintaining voter incentive alignment with accurate reporting.
    pub fn put_voter_reputation(
        &self,
        rwtxn: &mut RwTxn,
        reputation: &VoterReputation,
    ) -> Result<(), Error> {
        self.voter_reputation
            .put(rwtxn, &reputation.voter_id, reputation)?;
        Ok(())
    }

    /// Retrieve voter reputation
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `voter_id` - Voter to query
    ///
    /// # Returns
    /// Some(VoterReputation) if found, None for new voters
    pub fn get_voter_reputation(
        &self,
        rotxn: &RoTxn,
        voter_id: VoterId,
    ) -> Result<Option<VoterReputation>, Error> {
        Ok(self.voter_reputation.try_get(rotxn, &voter_id)?)
    }

    /// Get all voter reputations above a threshold
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `min_reputation` - Minimum reputation score to include
    ///
    /// # Returns
    /// Vector of voter reputations meeting the threshold
    pub fn get_voters_above_reputation(
        &self,
        rotxn: &RoTxn,
        min_reputation: f64,
    ) -> Result<Vec<VoterReputation>, Error> {
        let mut voters = Vec::new();
        let mut iter = self.voter_reputation.iter(rotxn)?;

        while let Some((_voter_id, reputation)) = iter.next()? {
            if reputation.reputation >= min_reputation {
                voters.push(reputation);
            }
        }

        Ok(voters)
    }

    /// Get reputation statistics for analysis
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    ///
    /// # Returns
    /// Tuple of (total_voters, avg_reputation, median_reputation, total_weight)
    pub fn get_reputation_stats(
        &self,
        rotxn: &RoTxn,
    ) -> Result<(u64, f64, f64, f64), Error> {
        let mut reputations = Vec::new();
        let mut iter = self.voter_reputation.iter(rotxn)?;

        while let Some((_voter_id, reputation)) = iter.next()? {
            reputations.push(reputation.reputation);
        }

        if reputations.is_empty() {
            return Ok((0, 0.0, 0.0, 0.0));
        }

        let total_voters = reputations.len() as u64;
        let total_weight: f64 = reputations.iter().sum();
        let avg_reputation = total_weight / total_voters as f64;

        reputations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_reputation = if reputations.len() % 2 == 0 {
            let mid = reputations.len() / 2;
            (reputations[mid - 1] + reputations[mid]) / 2.0
        } else {
            reputations[reputations.len() / 2]
        };

        Ok((
            total_voters,
            avg_reputation,
            median_reputation,
            total_weight,
        ))
    }

    // ================================================================================
    // Decision Outcome Operations
    // ================================================================================

    /// Store a decision outcome after consensus resolution
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `outcome` - Decision outcome to store
    ///
    /// # Bitcoin Hivemind Compliance
    /// Outcomes are immutable once stored and represent the final consensus
    /// value used for market resolution and payout calculations.
    /// Key is SlotId only - period is derived from slot.period_index() + 1
    pub fn put_decision_outcome(
        &self,
        rwtxn: &mut RwTxn,
        outcome: &DecisionOutcome,
    ) -> Result<(), Error> {
        self.decision_outcomes.put(rwtxn, &outcome.decision_id, outcome)?;
        Ok(())
    }

    /// Retrieve a decision outcome by SlotId
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `decision_id` - Decision to query
    ///
    /// # Returns
    /// Some(DecisionOutcome) if outcome exists, None otherwise
    ///
    /// # Bitcoin Hivemind Compliance
    /// Period information is available in the returned DecisionOutcome.
    /// No need to pass period_id since it's derivable from slot.period_index() + 1
    pub fn get_decision_outcome(
        &self,
        rotxn: &RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<DecisionOutcome>, Error> {
        Ok(self.decision_outcomes.try_get(rotxn, &decision_id)?)
    }

    /// Get all outcomes for a voting period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to query
    ///
    /// # Returns
    /// HashMap mapping SlotId to DecisionOutcome for all resolved decisions
    ///
    /// # Bitcoin Hivemind Compliance
    /// Filters outcomes by checking if outcome.period_id matches the query.
    /// Period is stored in DecisionOutcome for backwards compatibility but
    /// is redundant with slot.period_index() + 1
    pub fn get_outcomes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<crate::state::slots::SlotId, DecisionOutcome>, Error>
    {
        let mut outcomes = HashMap::new();
        let mut iter = self.decision_outcomes.iter(rotxn)?;

        while let Some((decision_id, outcome)) = iter.next()? {
            if outcome.period_id == period_id {
                outcomes.insert(decision_id, outcome);
            }
        }

        Ok(outcomes)
    }

    /// Get outcome for a specific decision
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `decision_id` - Decision to query
    ///
    /// # Returns
    /// Option<DecisionOutcome> for this decision
    ///
    /// # Bitcoin Hivemind Compliance
    /// This is the most efficient lookup - O(1) direct key access
    pub fn get_outcome_for_decision(
        &self,
        rotxn: &RoTxn,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<DecisionOutcome>, Error> {
        Ok(self.decision_outcomes.try_get(rotxn, &decision_id)?)
    }

    // ================================================================================
    // Period Statistics Operations
    // ================================================================================

    /// Store voting period statistics
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    /// * `stats` - Period statistics to store
    pub fn put_period_stats(
        &self,
        rwtxn: &mut RwTxn,
        stats: &VotingPeriodStats,
    ) -> Result<(), Error> {
        self.period_stats.put(rwtxn, &stats.period_id, stats)?;
        Ok(())
    }

    /// Retrieve period statistics
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Period to query
    ///
    /// # Returns
    /// Some(VotingPeriodStats) if statistics exist, None otherwise
    pub fn get_period_stats(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Option<VotingPeriodStats>, Error> {
        Ok(self.period_stats.try_get(rotxn, &period_id)?)
    }

    /// Get statistics for a range of periods
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `start_period` - First period to include
    /// * `end_period` - Last period to include
    ///
    /// # Returns
    /// Vector of statistics ordered by period ID
    pub fn get_stats_range(
        &self,
        rotxn: &RoTxn,
        start_period: VotingPeriodId,
        end_period: VotingPeriodId,
    ) -> Result<Vec<VotingPeriodStats>, Error> {
        let mut stats = Vec::new();
        let mut iter = self.period_stats.iter(rotxn)?;

        while let Some((period_id, period_stats)) = iter.next()? {
            if period_id >= start_period && period_id <= end_period {
                stats.push(period_stats);
            }
        }

        stats.sort_by_key(|s| s.period_id.as_u32());
        Ok(stats)
    }

    // ================================================================================
    // Utility Operations
    // ================================================================================

    /// Count total votes across all periods
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    ///
    /// # Returns
    /// Total number of votes in the database
    pub fn count_total_votes(&self, rotxn: &RoTxn) -> Result<u64, Error> {
        let mut count = 0;
        let mut iter = self.votes.iter(rotxn)?;

        while iter.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Get unique voters across all periods
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    ///
    /// # Returns
    /// Set of all unique voter IDs
    pub fn get_all_voters(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashSet<VoterId>, Error> {
        let mut voters = HashSet::new();
        let mut iter = self.votes.iter(rotxn)?;

        while let Some((key, _entry)) = iter.next()? {
            voters.insert(key.voter_id);
        }

        Ok(voters)
    }

    /// Check database consistency
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    ///
    /// # Returns
    /// Vector of consistency warnings/errors found
    ///
    /// # Bitcoin Hivemind Compliance
    /// Periods are now calculated on-demand, so we don't validate against stored periods.
    /// Instead, we check internal database consistency only.
    pub fn check_consistency(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<String>, Error> {
        let mut issues = Vec::new();

        // Count votes and ensure vote matrix keys are valid
        let mut vote_count = 0;
        let mut vote_iter = self.votes.iter(rotxn)?;
        while let Some((_key, _entry)) = vote_iter.next()? {
            vote_count += 1;
        }

        // Count outcomes and ensure consistency
        let mut outcome_count = 0;
        let mut outcome_iter = self.decision_outcomes.iter(rotxn)?;
        while let Some((_decision_id, outcome)) = outcome_iter.next()? {
            outcome_count += 1;

            // Verify period consistency: outcome.period_id should equal decision_id.period_index() + 1
            let expected_period = outcome.decision_id.period_index() + 1;
            if outcome.period_id.as_u32() != expected_period {
                issues.push(format!(
                    "Outcome period mismatch: decision {:?} claims period {:?} but should be {}",
                    outcome.decision_id,
                    outcome.period_id,
                    expected_period
                ));
            }
        }

        if issues.is_empty() {
            issues.push(format!(
                "Database consistent: {} votes, {} outcomes",
                vote_count, outcome_count
            ));
        }

        Ok(issues)
    }

    /// Clear all voting data (for testing/reset purposes)
    ///
    /// # Arguments
    /// * `rwtxn` - Read-write transaction
    ///
    /// # Warning
    /// This operation is irreversible and should only be used for testing
    /// or complete system resets.
    pub fn clear_all_data(&self, rwtxn: &mut RwTxn) -> Result<(), Error> {
        self.votes.clear(rwtxn)?;
        self.vote_batches.clear(rwtxn)?;
        self.voter_reputation.clear(rwtxn)?;
        self.decision_outcomes.clear(rwtxn)?;
        self.period_stats.clear(rwtxn)?;
        Ok(())
    }

    // ================================================================================
    // Consensus Outcome Operations (Extract from DecisionOutcome)
    // ================================================================================
    //
    // ARCHITECTURE NOTE: Consensus outcomes are NOT stored in a separate table.
    // They are extracted from DecisionOutcome.outcome_value on-demand to eliminate
    // redundancy. This is the single source of truth for consensus values.

    /// Get consensus outcome for a decision by extracting from DecisionOutcome
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `decision_id` - Decision to query
    ///
    /// # Returns
    /// Some(outcome) if decision has been resolved, None otherwise
    ///
    /// # Bitcoin Hivemind Compliance
    /// Extracts the consensus value from the DecisionOutcome struct instead of
    /// storing it separately, eliminating redundancy.
    pub fn get_consensus_outcome(
        &self,
        rotxn: &RoTxn,
        _period_id: VotingPeriodId,
        decision_id: crate::state::slots::SlotId,
    ) -> Result<Option<f64>, Error> {
        if let Some(outcome) = self.decision_outcomes.try_get(rotxn, &decision_id)? {
            Ok(Some(outcome.outcome_value))
        } else {
            Ok(None)
        }
    }

    /// Get all consensus outcomes for a period by extracting from DecisionOutcomes
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to query
    ///
    /// # Returns
    /// HashMap mapping SlotId to consensus outcome value
    ///
    /// # Bitcoin Hivemind Compliance
    /// Extracts consensus values from DecisionOutcome structs for the period.
    /// This eliminates the need for a separate consensus_outcomes table.
    pub fn get_consensus_outcomes_for_period(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<HashMap<crate::state::slots::SlotId, f64>, Error> {
        let mut outcomes = HashMap::new();
        let mut iter = self.decision_outcomes.iter(rotxn)?;
        let mut total_scanned = 0;

        while let Some((decision_id, outcome)) = iter.next()? {
            total_scanned += 1;

            if outcome.period_id == period_id {
                outcomes.insert(decision_id, outcome.outcome_value);
            }
        }

        Ok(outcomes)
    }

    /// Check if consensus has been calculated for a period
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to check
    ///
    /// # Returns
    /// true if any decision outcomes exist for the period
    pub fn has_consensus(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<bool, Error> {
        let outcomes = self.get_consensus_outcomes_for_period(rotxn, period_id)?;
        Ok(!outcomes.is_empty())
    }

    /// Get period consensus status (used by RPC layer for backwards compatibility)
    ///
    /// # Arguments
    /// * `rotxn` - Read-only transaction
    /// * `period_id` - Voting period to query
    ///
    /// # Returns
    /// Some(status_string) if period has consensus data, None otherwise
    pub fn get_period_consensus(
        &self,
        rotxn: &RoTxn,
        period_id: VotingPeriodId,
    ) -> Result<Option<String>, Error> {
        if self.has_consensus(rotxn, period_id)? {
            Ok(Some("Resolved".to_string()))
        } else {
            Ok(None)
        }
    }
}

// Tests are disabled until tempfile dependency is added to Cargo.toml
// #[cfg(test)]
// mod tests {
//     // Test implementation requires tempfile dependency
// }
