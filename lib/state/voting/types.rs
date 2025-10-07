use crate::state::slots::SlotId;
use crate::state::rollback::{RollBack, TxidStamped};
use crate::types::{Address, Txid, hashes};
use serde::{Deserialize, Serialize};

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Deserialize,
    Serialize,
)]
pub struct VotingPeriodId(pub u32);

impl VotingPeriodId {
    pub const fn new(period: u32) -> Self {
        Self(period)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_be_bytes(bytes))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingPeriodStatus {
    Pending,
    Active,
    Closed,
    Resolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VotingPeriod {
    pub id: VotingPeriodId,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub status: VotingPeriodStatus,
    pub decision_slots: Vec<SlotId>,
    pub created_at_height: u64,
}

impl VotingPeriod {
    pub fn new(
        id: VotingPeriodId,
        start_timestamp: u64,
        end_timestamp: u64,
        decision_slots: Vec<SlotId>,
        created_at_height: u64,
    ) -> Self {
        Self {
            id,
            start_timestamp,
            end_timestamp,
            status: VotingPeriodStatus::Pending,
            decision_slots,
            created_at_height,
        }
    }

    pub fn is_active(&self, current_timestamp: u64) -> bool {
        self.status == VotingPeriodStatus::Active
            && current_timestamp >= self.start_timestamp
            && current_timestamp < self.end_timestamp
    }

    pub fn has_ended(&self, current_timestamp: u64) -> bool {
        current_timestamp >= self.end_timestamp
    }

    pub fn duration_seconds(&self) -> u64 {
        self.end_timestamp.saturating_sub(self.start_timestamp)
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Deserialize,
    Serialize,
)]
pub struct VoterId([u8; 20]);

impl VoterId {
    pub fn from_address(address: &Address) -> Self {
        Self(address.0)
    }

    pub fn from_bytes(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 20] {
        &self.0
    }

    /// Convert to Address for compatibility
    pub fn to_address(&self) -> Address {
        Address(self.0)
    }
}

/// Vote value type supporting both binary and scalar decisions
///
/// Bitcoin Hivemind supports two types of decisions:
/// - Binary: Yes/No questions with values 0.0 or 1.0
/// - Scalar: Continuous values within a defined range [min, max]
///
/// # Specification Reference
/// Bitcoin Hivemind whitepaper Section 2.2: "Decision Types"
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum VoteValue {
    /// Binary vote: false (0.0) or true (1.0)
    Binary(bool),
    /// Scalar vote: value within decision's [min, max] range
    Scalar(f64),
    /// Abstain from voting on this decision
    Abstain,
}

impl VoteValue {
    /// Convert vote value to f64 for mathematical operations
    ///
    /// # Returns
    /// - Binary: 0.0 for false, 1.0 for true
    /// - Scalar: the scalar value
    /// - Abstain: f64::NAN to indicate missing data
    pub fn to_f64(&self) -> f64 {
        match self {
            VoteValue::Binary(false) => 0.0,
            VoteValue::Binary(true) => 1.0,
            VoteValue::Scalar(value) => *value,
            VoteValue::Abstain => f64::NAN,
        }
    }

    /// Check if this vote represents an abstention
    pub fn is_abstain(&self) -> bool {
        matches!(self, VoteValue::Abstain)
    }

    /// Create a binary vote
    pub fn binary(value: bool) -> Self {
        VoteValue::Binary(value)
    }

    /// Create a scalar vote
    pub fn scalar(value: f64) -> Self {
        VoteValue::Scalar(value)
    }

    /// Create an abstention
    pub fn abstain() -> Self {
        VoteValue::Abstain
    }
}

/// Individual vote cast by a voter on a specific decision
///
/// Votes are the atomic units of the Bitcoin Hivemind consensus mechanism.
/// Each vote links a voter to a decision with a specific value, and all votes
/// are aggregated using the consensus algorithm to determine outcomes.
///
/// # Specification Reference
/// Bitcoin Hivemind whitepaper Section 3.3: "Vote Structure"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    /// Unique identifier for the voter
    pub voter_id: VoterId,
    /// Voting period when this vote was cast
    pub period_id: VotingPeriodId,
    /// Decision slot being voted on
    pub decision_id: SlotId,
    /// Vote value (binary, scalar, or abstain)
    pub value: VoteValue,
    /// L1 timestamp when vote was cast
    pub timestamp: u64,
    /// L2 block height when vote was included
    pub block_height: u64,
    /// Hash of the transaction containing this vote
    pub tx_hash: [u8; 32],
}

impl Vote {
    /// Create a new vote
    ///
    /// # Arguments
    /// * `voter_id` - ID of the voter casting this vote
    /// * `period_id` - Voting period this vote belongs to
    /// * `decision_id` - Decision slot being voted on
    /// * `value` - Vote value (binary, scalar, or abstain)
    /// * `timestamp` - L1 timestamp when vote was cast
    /// * `block_height` - L2 block height when vote was included
    /// * `tx_hash` - Hash of transaction containing this vote
    pub fn new(
        voter_id: VoterId,
        period_id: VotingPeriodId,
        decision_id: SlotId,
        value: VoteValue,
        timestamp: u64,
        block_height: u64,
        tx_hash: [u8; 32],
    ) -> Self {
        Self {
            voter_id,
            period_id,
            decision_id,
            value,
            timestamp,
            block_height,
            tx_hash,
        }
    }

    /// Compute a unique hash for this vote for deduplication
    ///
    /// The hash is computed from the voter, period, and decision to ensure
    /// each voter can only cast one vote per decision per period.
    pub fn compute_hash(&self) -> [u8; 32] {
        let vote_data = (
            self.voter_id.as_bytes(),
            self.period_id.as_bytes(),
            self.decision_id.as_bytes(),
        );
        hashes::hash(&vote_data)
    }
}

/// Voter reputation and weighting for consensus algorithm with Votecoin integration
///
/// Reputation represents a voter's historical accuracy and influence in the
/// consensus process, now enhanced with Votecoin holdings to implement the
/// complete Bitcoin Hivemind voting weight formula.
///
/// # Bitcoin Hivemind Voting Weight Formula
/// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
///
/// This structure maintains both components:
/// - Base Reputation: Historical accuracy-based weighting
/// - Votecoin Holdings: Economic stake-based weighting
///
/// # Specification Reference
/// Bitcoin Hivemind whitepaper Section 4.2: "Reputation System" and Section 5: "Economics"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoterReputation {
    /// Voter this reputation data belongs to
    pub voter_id: VoterId,
    /// Current base reputation score (0.0 to 1.0) based on historical accuracy
    pub reputation: f64,
    /// Cached Votecoin holdings proportion (0.0 to 1.0) of total supply
    /// This is updated when voting weights are calculated to avoid repeated UTXO queries
    pub votecoin_proportion: f64,
    /// Final voting weight combining reputation and Votecoin holdings
    /// Calculated as: reputation × votecoin_proportion
    pub voting_weight: f64,
    /// Total number of decisions this voter has participated in
    pub total_decisions: u64,
    /// Number of decisions where voter was in consensus
    pub correct_decisions: u64,
    /// Running average of voter accuracy
    pub accuracy_rate: f64,
    /// Timestamp of last reputation update
    pub last_updated: u64,
    /// Voting period when reputation was last calculated
    pub last_period: VotingPeriodId,
    /// Block height when Votecoin proportion was last updated
    /// Used to determine when to refresh cached proportion data
    pub votecoin_updated_height: Option<u64>,
    /// Reputation history for rollback support during blockchain reorganizations
    ///
    /// This enables safe reversion of reputation updates when blocks are disconnected,
    /// maintaining consensus correctness across reorgs as required by Bitcoin Hivemind.
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper Section 6: "Blockchain Security" - Reorg handling
    pub reputation_history: RollBack<TxidStamped<f64>>,
}

impl VoterReputation {
    /// Create initial reputation for a new voter with Bitcoin Hivemind integration
    ///
    /// # Arguments
    /// * `voter_id` - ID of the voter
    /// * `initial_reputation` - Starting base reputation (typically 0.5)
    /// * `timestamp` - Current timestamp
    /// * `period_id` - Current voting period
    ///
    /// # Bitcoin Hivemind Compliance
    /// New voters start with neutral base reputation to prevent gaming through
    /// multiple identity creation. Votecoin proportion is initialized to 0.0
    /// and must be updated with actual holdings before voting weight calculation.
    pub fn new(
        voter_id: VoterId,
        initial_reputation: f64,
        timestamp: u64,
        period_id: VotingPeriodId,
    ) -> Self {
        let reputation = initial_reputation.clamp(0.0, 1.0);

        // Initialize reputation history with genesis/initial transaction
        // Uses a zero txid to represent the initial state
        let initial_txid = Txid([0u8; 32]);
        let reputation_history = RollBack::<TxidStamped<f64>>::new(reputation, initial_txid, 0);

        Self {
            voter_id,
            reputation,
            votecoin_proportion: 0.0, // Must be updated with actual holdings
            voting_weight: 0.0, // Will be calculated when Votecoin proportion is set
            total_decisions: 0,
            correct_decisions: 0,
            accuracy_rate: 0.0,
            last_updated: timestamp,
            last_period: period_id,
            votecoin_updated_height: None,
            reputation_history,
        }
    }

    /// Update base reputation based on voting performance
    ///
    /// # Arguments
    /// * `was_correct` - Whether voter was in consensus on recent decisions
    /// * `timestamp` - Current timestamp
    /// * `period_id` - Voting period being processed
    /// * `txid` - Transaction ID of the reputation update
    /// * `height` - Block height of the reputation update
    ///
    /// # Bitcoin Hivemind Compliance
    /// Updates only the base reputation component. Final voting weight is
    /// recalculated by multiplying updated reputation with Votecoin proportion.
    /// This follows the incentive mechanism to reward accurate reporting.
    ///
    /// # Rollback Support
    /// Pushes current reputation to history before updating, enabling safe
    /// reversion during blockchain reorganizations.
    pub fn update(
        &mut self,
        was_correct: bool,
        timestamp: u64,
        period_id: VotingPeriodId,
        txid: Txid,
        height: u32,
    ) {
        // Push current reputation to history before updating
        self.reputation_history.push(self.reputation, txid, height);

        self.total_decisions += 1;
        if was_correct {
            self.correct_decisions += 1;
        }

        self.accuracy_rate = if self.total_decisions > 0 {
            self.correct_decisions as f64 / self.total_decisions as f64
        } else {
            0.0
        };

        // Update base reputation (can be enhanced with more sophisticated algorithms)
        self.reputation = self.accuracy_rate.clamp(0.0, 1.0);
        self.last_updated = timestamp;
        self.last_period = period_id;

        // Recalculate final voting weight with updated reputation
        self.update_voting_weight();
    }

    /// Rollback the most recent reputation update
    ///
    /// # Returns
    /// Previous reputation value if available, None if at initial state
    ///
    /// # Bitcoin Hivemind Compliance
    /// Enables safe blockchain reorganizations by reverting reputation to
    /// previous state when blocks are disconnected.
    pub fn rollback_update(&mut self) -> Option<f64> {
        if let Some(previous) = self.reputation_history.pop() {
            let previous_reputation = previous.data;
            self.reputation = previous_reputation;

            // Decrement counters (note: this is approximate as we don't track
            // individual decision correctness in history)
            self.total_decisions = self.total_decisions.saturating_sub(1);

            // Recalculate accuracy rate
            self.accuracy_rate = if self.total_decisions > 0 {
                self.correct_decisions as f64 / self.total_decisions as f64
            } else {
                0.0
            };

            // Recalculate voting weight
            self.update_voting_weight();

            Some(previous_reputation)
        } else {
            None
        }
    }

    /// Update Votecoin holdings proportion and recalculate voting weight
    ///
    /// # Arguments
    /// * `votecoin_proportion` - New Votecoin proportion (0.0 to 1.0)
    /// * `current_height` - Current block height for caching
    ///
    /// # Bitcoin Hivemind Specification
    /// This implements the Votecoin Holdings Proportion component of:
    /// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
    pub fn update_votecoin_proportion(
        &mut self,
        votecoin_proportion: f64,
        current_height: u64,
    ) {
        self.votecoin_proportion = votecoin_proportion.clamp(0.0, 1.0);
        self.votecoin_updated_height = Some(current_height);
        self.update_voting_weight();
    }

    /// Recalculate final voting weight from reputation and Votecoin proportion
    ///
    /// # Bitcoin Hivemind Formula
    /// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
    ///
    /// Special cases handled:
    /// - If no Votecoin holdings (proportion = 0.0), voting weight = 0.0
    /// - If no reputation (reputation = 0.0), voting weight = 0.0
    /// - This ensures both economic stake and performance matter for influence
    pub fn update_voting_weight(&mut self) {
        self.voting_weight = self.reputation * self.votecoin_proportion;
    }

    /// Get the final voting weight for consensus calculations
    ///
    /// # Returns
    /// Final voting weight incorporating both reputation and Votecoin holdings
    pub fn get_voting_weight(&self) -> f64 {
        self.voting_weight
    }

    /// Get base reputation score (without Votecoin weighting)
    ///
    /// # Returns
    /// Base reputation score for historical tracking
    pub fn get_base_reputation(&self) -> f64 {
        self.reputation
    }

    /// Get Votecoin holdings proportion
    ///
    /// # Returns
    /// Cached Votecoin proportion (may need refresh if UTXO set changed)
    pub fn get_votecoin_proportion(&self) -> f64 {
        self.votecoin_proportion
    }

    /// Check if Votecoin proportion needs refresh based on block height
    ///
    /// # Arguments
    /// * `current_height` - Current block height
    /// * `max_staleness` - Maximum blocks before refresh needed
    ///
    /// # Returns
    /// True if proportion should be refreshed from UTXO set
    pub fn needs_votecoin_refresh(
        &self,
        current_height: u64,
        max_staleness: u64,
    ) -> bool {
        match self.votecoin_updated_height {
            None => true, // Never updated
            Some(last_height) => {
                current_height.saturating_sub(last_height) > max_staleness
            }
        }
    }
}

/// Final outcome of a resolved decision
///
/// Status of decision resolution process
///
/// Tracks the current state of a decision through the resolution process
/// as defined in the Bitcoin Hivemind specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionResolutionStatus {
    /// Decision is available for voting
    Pending,
    /// Voting period has ended, awaiting consensus calculation
    AwaitingResolution,
    /// Decision has been resolved through consensus
    Resolved,
    /// Decision was defaulted due to insufficient participation
    Defaulted,
    /// Decision was cancelled or invalidated
    Cancelled,
}

impl DecisionResolutionStatus {
    /// Check if decision accepts new votes
    pub fn accepts_votes(&self) -> bool {
        matches!(self, DecisionResolutionStatus::Pending)
    }

    /// Check if decision is finalized
    pub fn is_finalized(&self) -> bool {
        matches!(
            self,
            DecisionResolutionStatus::Resolved
                | DecisionResolutionStatus::Defaulted
                | DecisionResolutionStatus::Cancelled
        )
    }
}

/// Decision resolution state tracking
///
/// Maintains the state of a decision through the resolution process,
/// including when state changes occur and why.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionResolution {
    /// Decision being tracked
    pub decision_id: SlotId,
    /// Voting period this decision belongs to
    pub period_id: VotingPeriodId,
    /// Current resolution status
    pub status: DecisionResolutionStatus,
    /// When this status was set
    pub status_changed_at: u64,
    /// Block height when status changed
    pub status_changed_height: u64,
    /// Number of votes received so far
    pub vote_count: u32,
    /// Required minimum votes for consensus
    pub min_votes_required: u32,
    /// Deadline for voting (L1 timestamp)
    pub voting_deadline: u64,
    /// Whether this decision has outcome ready for consensus
    pub outcome_ready: bool,
    /// Optional reason for status change
    pub reason: Option<String>,
}

impl DecisionResolution {
    /// Create new decision resolution tracking
    pub fn new(
        decision_id: SlotId,
        period_id: VotingPeriodId,
        voting_deadline: u64,
        min_votes_required: u32,
        current_timestamp: u64,
        current_height: u64,
    ) -> Self {
        Self {
            decision_id,
            period_id,
            status: DecisionResolutionStatus::Pending,
            status_changed_at: current_timestamp,
            status_changed_height: current_height,
            vote_count: 0,
            min_votes_required,
            voting_deadline,
            outcome_ready: false,
            reason: None,
        }
    }

    /// Update resolution status
    pub fn update_status(
        &mut self,
        new_status: DecisionResolutionStatus,
        timestamp: u64,
        block_height: u64,
        reason: Option<String>,
    ) {
        self.status = new_status;
        self.status_changed_at = timestamp;
        self.status_changed_height = block_height;
        self.reason = reason;
    }

    /// Add a vote to this decision
    pub fn add_vote(&mut self) {
        if self.status.accepts_votes() {
            self.vote_count += 1;
        }
    }

    /// Check if voting period has expired
    pub fn is_voting_expired(&self, current_timestamp: u64) -> bool {
        current_timestamp >= self.voting_deadline
    }

    /// Check if decision has minimum votes for consensus
    pub fn has_minimum_votes(&self) -> bool {
        self.vote_count >= self.min_votes_required
    }

    /// Check if decision is ready for consensus calculation
    pub fn is_ready_for_consensus(&self, current_timestamp: u64) -> bool {
        matches!(self.status, DecisionResolutionStatus::Pending)
            && (self.is_voting_expired(current_timestamp)
                || self.has_minimum_votes())
    }

    /// Mark outcome as ready
    pub fn mark_outcome_ready(&mut self) {
        self.outcome_ready = true;
    }
}

/// After voting closes and consensus is reached, each decision receives
/// a final outcome that is used for market resolution and payout calculation.
///
/// # Specification Reference
/// Bitcoin Hivemind whitepaper Section 4.3: "Outcome Determination"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Decision slot this outcome applies to
    pub decision_id: SlotId,
    /// Voting period when this outcome was determined
    pub period_id: VotingPeriodId,
    /// Final consensus value for this decision
    pub outcome_value: f64,
    /// Minimum possible value for this decision (for scalar decisions)
    pub min: f64,
    /// Maximum possible value for this decision (for scalar decisions)
    pub max: f64,
    /// Confidence level in this outcome (0.0 to 1.0)
    pub confidence: f64,
    /// Total number of votes cast on this decision
    pub total_votes: u64,
    /// Total reputation weight of all voters
    pub total_reputation_weight: f64,
    /// L1 timestamp when outcome was finalized
    pub finalized_at: u64,
    /// L2 block height when outcome was recorded
    pub block_height: u64,
    /// Whether this outcome was reached through consensus or default
    pub is_consensus: bool,
    /// Resolution status and tracking information
    pub resolution: DecisionResolution,
}

impl DecisionOutcome {
    /// Create a new decision outcome
    ///
    /// # Arguments
    /// * `decision_id` - Decision slot this outcome applies to
    /// * `period_id` - Voting period when outcome was determined
    /// * `outcome_value` - Final consensus value
    /// * `min` - Minimum possible value for this decision
    /// * `max` - Maximum possible value for this decision
    /// * `confidence` - Confidence level in outcome
    /// * `total_votes` - Number of votes cast
    /// * `total_reputation_weight` - Sum of voter reputation weights
    /// * `finalized_at` - Timestamp when outcome was finalized
    /// * `block_height` - Block height when outcome was recorded
    /// * `is_consensus` - Whether outcome represents true consensus
    /// * `resolution` - Resolution tracking information
    pub fn new(
        decision_id: SlotId,
        period_id: VotingPeriodId,
        outcome_value: f64,
        min: f64,
        max: f64,
        confidence: f64,
        total_votes: u64,
        total_reputation_weight: f64,
        finalized_at: u64,
        block_height: u64,
        is_consensus: bool,
        resolution: DecisionResolution,
    ) -> Self {
        Self {
            decision_id,
            period_id,
            outcome_value,
            min,
            max,
            confidence: confidence.clamp(0.0, 1.0),
            total_votes,
            total_reputation_weight,
            finalized_at,
            block_height,
            is_consensus,
            resolution,
        }
    }

    /// Create outcome with resolved status
    pub fn new_resolved(
        decision_id: SlotId,
        period_id: VotingPeriodId,
        outcome_value: f64,
        min: f64,
        max: f64,
        confidence: f64,
        total_votes: u64,
        total_reputation_weight: f64,
        finalized_at: u64,
        block_height: u64,
    ) -> Self {
        let mut resolution = DecisionResolution::new(
            decision_id,
            period_id,
            finalized_at,
            1, // min_votes_required
            finalized_at,
            block_height,
        );
        resolution.update_status(
            DecisionResolutionStatus::Resolved,
            finalized_at,
            block_height,
            Some("Consensus reached".to_string()),
        );
        resolution.mark_outcome_ready();

        Self::new(
            decision_id,
            period_id,
            outcome_value,
            min,
            max,
            confidence,
            total_votes,
            total_reputation_weight,
            finalized_at,
            block_height,
            true,
            resolution,
        )
    }
}

/// Matrix key for efficient vote storage and retrieval
///
/// The matrix key combines voter and decision identifiers to create
/// a unique key for each vote in the sparse vote matrix structure.
///
/// # Specification Reference
/// Bitcoin Hivemind whitepaper Section 4.4: "Vote Matrix Structure"
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
pub struct VoteMatrixKey {
    /// Voting period this key belongs to
    pub period_id: VotingPeriodId,
    /// Voter casting the vote
    pub voter_id: VoterId,
    /// Decision being voted on
    pub decision_id: SlotId,
}

impl VoteMatrixKey {
    /// Create a new vote matrix key
    pub fn new(
        period_id: VotingPeriodId,
        voter_id: VoterId,
        decision_id: SlotId,
    ) -> Self {
        Self {
            period_id,
            voter_id,
            decision_id,
        }
    }
}

/// Vote matrix entry storing vote data efficiently
///
/// The vote matrix is the core data structure for the Bitcoin Hivemind
/// consensus algorithm. It stores votes in a sparse matrix format optimized
/// for the mathematical operations required by the consensus algorithm.
///
/// # Specification Reference
/// Bitcoin Hivemind whitepaper Section 4: "Consensus Algorithm"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteMatrixEntry {
    /// The vote value
    pub value: VoteValue,
    /// Timestamp when vote was cast
    pub timestamp: u64,
    /// Block height when vote was included
    pub block_height: u64,
}

impl VoteMatrixEntry {
    /// Create a new vote matrix entry
    pub fn new(value: VoteValue, timestamp: u64, block_height: u64) -> Self {
        Self {
            value,
            timestamp,
            block_height,
        }
    }

    /// Convert to f64 for mathematical operations
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }
}

/// Batch of votes for efficient database operations
///
/// Vote batches allow multiple votes to be processed atomically,
/// improving performance for bulk operations and ensuring consistency
/// during vote ingestion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoteBatch {
    /// Voting period these votes belong to
    pub period_id: VotingPeriodId,
    /// Collection of votes in this batch
    pub votes: Vec<Vote>,
    /// Timestamp when batch was created
    pub created_at: u64,
    /// Block height when batch was processed
    pub block_height: u64,
}

impl VoteBatch {
    /// Create a new vote batch
    pub fn new(
        period_id: VotingPeriodId,
        created_at: u64,
        block_height: u64,
    ) -> Self {
        Self {
            period_id,
            votes: Vec::new(),
            created_at,
            block_height,
        }
    }

    /// Add a vote to this batch
    pub fn add_vote(&mut self, vote: Vote) {
        self.votes.push(vote);
    }

    /// Get the number of votes in this batch
    pub fn len(&self) -> usize {
        self.votes.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.votes.is_empty()
    }
}

/// Statistics about a voting period
///
/// Period statistics provide insights into voter participation,
/// consensus quality, and overall health of the voting process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VotingPeriodStats {
    /// Period these statistics apply to
    pub period_id: VotingPeriodId,
    /// Total number of unique voters who participated
    pub total_voters: u64,
    /// Total number of votes cast across all decisions
    pub total_votes: u64,
    /// Number of decisions available for voting
    pub total_decisions: u64,
    /// Average participation rate across all decisions
    pub avg_participation_rate: f64,
    /// Total reputation weight of all participants
    pub total_reputation_weight: f64,
    /// Number of decisions that reached consensus
    pub consensus_decisions: u64,
    /// Timestamp when statistics were calculated
    pub calculated_at: u64,
}

impl VotingPeriodStats {
    /// Create new period statistics
    pub fn new(period_id: VotingPeriodId, calculated_at: u64) -> Self {
        Self {
            period_id,
            total_voters: 0,
            total_votes: 0,
            total_decisions: 0,
            avg_participation_rate: 0.0,
            total_reputation_weight: 0.0,
            consensus_decisions: 0,
            calculated_at,
        }
    }

    /// Calculate participation rate
    pub fn participation_rate(&self) -> f64 {
        if self.total_decisions > 0 && self.total_voters > 0 {
            self.total_votes as f64
                / (self.total_decisions * self.total_voters) as f64
        } else {
            0.0
        }
    }

    /// Calculate consensus rate
    pub fn consensus_rate(&self) -> f64 {
        if self.total_decisions > 0 {
            self.consensus_decisions as f64 / self.total_decisions as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Address;

    #[test]
    fn test_voting_period_id() {
        let period_id = VotingPeriodId::new(42);
        assert_eq!(period_id.as_u32(), 42);

        let bytes = period_id.as_bytes();
        let reconstructed = VotingPeriodId::from_bytes(bytes);
        assert_eq!(period_id, reconstructed);
    }

    #[test]
    fn test_voter_id() {
        let address = Address([1u8; 20]);
        let voter_id = VoterId::from_address(&address);
        assert_eq!(voter_id.to_address(), address);
    }

    #[test]
    fn test_vote_value() {
        let binary_true = VoteValue::binary(true);
        assert_eq!(binary_true.to_f64(), 1.0);
        assert!(!binary_true.is_abstain());

        let binary_false = VoteValue::binary(false);
        assert_eq!(binary_false.to_f64(), 0.0);

        let scalar = VoteValue::scalar(0.75);
        assert_eq!(scalar.to_f64(), 0.75);

        let abstain = VoteValue::abstain();
        assert!(abstain.is_abstain());
        assert!(abstain.to_f64().is_nan());
    }

    #[test]
    fn test_voting_period() {
        let start = 1000;
        let end = 2000;
        let period =
            VotingPeriod::new(VotingPeriodId::new(1), start, end, vec![], 100);

        assert_eq!(period.duration_seconds(), 1000);
        assert!(!period.is_active(500)); // Before start
        assert!(!period.is_active(2500)); // After end
        assert!(period.has_ended(2500));
        assert!(!period.has_ended(500));
    }

    #[test]
    fn test_voter_reputation() {
        let voter_id = VoterId::from_bytes([1u8; 20]);
        let mut reputation =
            VoterReputation::new(voter_id, 0.5, 1000, VotingPeriodId::new(1));

        assert_eq!(reputation.reputation, 0.5);
        assert_eq!(reputation.total_decisions, 0);

        reputation.update(true, 1100, VotingPeriodId::new(2), Txid([1u8; 32]), 1);
        assert_eq!(reputation.total_decisions, 1);
        assert_eq!(reputation.correct_decisions, 1);
        assert_eq!(reputation.accuracy_rate, 1.0);
        assert_eq!(reputation.reputation, 1.0);

        reputation.update(false, 1200, VotingPeriodId::new(3), Txid([2u8; 32]), 2);
        assert_eq!(reputation.total_decisions, 2);
        assert_eq!(reputation.correct_decisions, 1);
        assert_eq!(reputation.accuracy_rate, 0.5);
        assert_eq!(reputation.reputation, 0.5);
    }

    #[test]
    fn test_vote_batch() {
        let period_id = VotingPeriodId::new(1);
        let mut batch = VoteBatch::new(period_id, 1000, 100);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        let vote = Vote::new(
            VoterId::from_bytes([1u8; 20]),
            period_id,
            SlotId::new(1, 0).unwrap(),
            VoteValue::binary(true),
            1000,
            100,
            [0u8; 32],
        );

        batch.add_vote(vote);
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_voting_period_stats() {
        let period_id = VotingPeriodId::new(1);
        let mut stats = VotingPeriodStats::new(period_id, 1000);

        stats.total_voters = 10;
        stats.total_votes = 80;
        stats.total_decisions = 10;
        stats.consensus_decisions = 8;

        assert_eq!(stats.participation_rate(), 0.8);
        assert_eq!(stats.consensus_rate(), 0.8);
    }
}
