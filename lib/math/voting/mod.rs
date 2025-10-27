//! Bitcoin Hivemind Voting Mathematics with Votecoin Integration
//!
//! This module implements the mathematical foundations for the Bitcoin Hivemind
//! voting consensus algorithm as specified in the whitepaper, now enhanced with
//! complete Votecoin economic stake integration. It provides efficient operations
//! on vote matrices and reputation vectors using the full voting weight formula.
//!
//! ## Mathematical Background
//! The Bitcoin Hivemind consensus algorithm uses Principal Component Analysis (PCA)
//! and iterative convergence to determine truth from voter input. This implementation
//! now incorporates the complete voting weight formula that combines both historical
//! accuracy (reputation) and economic stake (Votecoin holdings).
//!
//! ## Votecoin Integration in Mathematical Operations
//!
//! ### Bitcoin Hivemind Voting Weight Formula
//! **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
//!
//! All mathematical operations now use the complete voting weights that incorporate:
//! - Historical accuracy through reputation scoring
//! - Economic stake through Votecoin holdings proportion
//! - Combined influence that ensures both performance and investment matter
//!
//! ### Key Mathematical Components
//! 1. **SparseVoteMatrix**: Efficient storage and manipulation of voting data
//! 2. **ReputationVector**: Enhanced to use final voting weights (reputation × Votecoin)
//! 3. **VoteAggregator**: Weighted calculations using complete economic-stake weights
//! 4. **ConsensusEngine**: PCA operations with Votecoin-integrated weighting
//!
//! ### Enhanced Features
//! - Vote aggregation weighted by both accuracy and economic stake
//! - Consensus calculations that reflect voter's total influence
//! - Outlier detection considering both reputation and Votecoin holdings
//! - Confidence scoring based on complete voting weight distribution
//!
//! ## Bitcoin Hivemind Specification References
//! - Section 4: "Consensus Algorithm" - Mathematical basis for truth-finding
//! - Section 5: "Economics" - Economic stake integration in consensus
//! - Appendix A: "Mathematical Details" - Complete algorithm specifications
//! - Appendix B: "Economic Weighting" - Stake-based influence calculations

pub mod constants;
pub mod matrix;
pub mod consensus;

use crate::state::slots::SlotId;
use crate::state::voting::types::VoterId;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Error types for voting mathematics operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum VotingMathError {
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Empty vote matrix provided")]
    EmptyMatrix,

    #[error("Invalid reputation values: {reason}")]
    InvalidReputation { reason: String },

    #[error("Numerical computation error: {reason}")]
    NumericalError { reason: String },

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
}

/// Sparse vote matrix representation optimized for Bitcoin Hivemind operations
///
/// The vote matrix is stored as a sparse structure since most voters do not
/// vote on all decisions. This representation is efficient for both storage
/// and the mathematical operations required by the consensus algorithm.
///
/// # Structure
/// - Rows represent voters
/// - Columns represent decisions
/// - Values are vote values (0.0, 1.0 for binary; continuous for scalar)
/// - Missing values represent abstentions
#[derive(Debug, Clone)]
pub struct SparseVoteMatrix {
    /// Mapping from (voter_index, decision_index) to vote value
    entries: HashMap<(usize, usize), f64>,
    /// Voter ID to index mapping
    voter_indices: HashMap<VoterId, usize>,
    /// Decision ID to index mapping
    decision_indices: HashMap<SlotId, usize>,
    /// Index to voter ID mapping (reverse lookup)
    index_to_voter: HashMap<usize, VoterId>,
    /// Index to decision ID mapping (reverse lookup)
    index_to_decision: HashMap<usize, SlotId>,
    /// Number of voters (matrix rows)
    num_voters: usize,
    /// Number of decisions (matrix columns)
    num_decisions: usize,
}

impl SparseVoteMatrix {
    /// Create a new sparse vote matrix
    ///
    /// # Arguments
    /// * `voters` - List of voter IDs
    /// * `decisions` - List of decision IDs
    ///
    /// # Returns
    /// Empty sparse matrix with specified dimensions
    pub fn new(voters: Vec<VoterId>, decisions: Vec<SlotId>) -> Self {
        let num_voters = voters.len();
        let num_decisions = decisions.len();

        let mut voter_indices = HashMap::new();
        let mut index_to_voter = HashMap::new();
        for (i, voter_id) in voters.into_iter().enumerate() {
            voter_indices.insert(voter_id, i);
            index_to_voter.insert(i, voter_id);
        }

        let mut decision_indices = HashMap::new();
        let mut index_to_decision = HashMap::new();
        for (j, decision_id) in decisions.into_iter().enumerate() {
            decision_indices.insert(decision_id, j);
            index_to_decision.insert(j, decision_id);
        }

        Self {
            entries: HashMap::new(),
            voter_indices,
            decision_indices,
            index_to_voter,
            index_to_decision,
            num_voters,
            num_decisions,
        }
    }

    /// Set a vote value in the matrix
    ///
    /// # Arguments
    /// * `voter_id` - Voter who cast the vote
    /// * `decision_id` - Decision being voted on
    /// * `value` - Vote value
    ///
    /// # Returns
    /// Ok(()) if vote was set, Err if voter or decision not found
    pub fn set_vote(
        &mut self,
        voter_id: VoterId,
        decision_id: SlotId,
        value: f64,
    ) -> Result<(), VotingMathError> {
        let voter_idx =
            *self.voter_indices.get(&voter_id).ok_or_else(|| {
                VotingMathError::InvalidReputation {
                    reason: format!("Voter {:?} not found in matrix", voter_id),
                }
            })?;

        let decision_idx = *self
            .decision_indices
            .get(&decision_id)
            .ok_or_else(|| VotingMathError::InvalidReputation {
                reason: format!(
                    "Decision {:?} not found in matrix",
                    decision_id
                ),
            })?;

        self.entries.insert((voter_idx, decision_idx), value);
        Ok(())
    }

    /// Get a vote value from the matrix
    ///
    /// # Arguments
    /// * `voter_id` - Voter to query
    /// * `decision_id` - Decision to query
    ///
    /// # Returns
    /// Some(value) if vote exists, None if abstention or not found
    pub fn get_vote(
        &self,
        voter_id: VoterId,
        decision_id: SlotId,
    ) -> Option<f64> {
        let voter_idx = *self.voter_indices.get(&voter_id)?;
        let decision_idx = *self.decision_indices.get(&decision_id)?;
        self.entries.get(&(voter_idx, decision_idx)).copied()
    }

    /// Convert to dense matrix for mathematical operations
    ///
    /// # Arguments
    /// * `fill_value` - Value to use for missing entries (typically NaN or 0.5)
    ///
    /// # Returns
    /// Dense ndarray matrix suitable for linear algebra operations
    pub fn to_dense(&self, fill_value: f64) -> Array2<f64> {
        let mut dense = Array2::from_elem(
            (self.num_voters, self.num_decisions),
            fill_value,
        );

        for (&(i, j), &value) in &self.entries {
            dense[[i, j]] = value;
        }

        dense
    }

    /// Get dimensions of the matrix
    ///
    /// # Returns
    /// Tuple of (num_voters, num_decisions)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.num_voters, self.num_decisions)
    }

    /// Get total number of non-missing votes
    pub fn num_votes(&self) -> usize {
        self.entries.len()
    }

    /// Get density (fraction of matrix filled)
    pub fn density(&self) -> f64 {
        if self.num_voters == 0 || self.num_decisions == 0 {
            return 0.0;
        }
        self.entries.len() as f64
            / (self.num_voters * self.num_decisions) as f64
    }

    /// Get votes for a specific voter
    ///
    /// # Arguments
    /// * `voter_id` - Voter to query
    ///
    /// # Returns
    /// HashMap mapping SlotId to vote value for this voter
    pub fn get_voter_votes(&self, voter_id: VoterId) -> HashMap<SlotId, f64> {
        let mut votes = HashMap::new();

        if let Some(&voter_idx) = self.voter_indices.get(&voter_id) {
            for (&(i, j), &value) in &self.entries {
                if i == voter_idx {
                    if let Some(&decision_id) = self.index_to_decision.get(&j) {
                        votes.insert(decision_id, value);
                    }
                }
            }
        }

        votes
    }

    /// Get votes for a specific decision
    ///
    /// # Arguments
    /// * `decision_id` - Decision to query
    ///
    /// # Returns
    /// HashMap mapping VoterId to vote value for this decision
    pub fn get_decision_votes(
        &self,
        decision_id: SlotId,
    ) -> HashMap<VoterId, f64> {
        let mut votes = HashMap::new();

        if let Some(&decision_idx) = self.decision_indices.get(&decision_id) {
            for (&(i, j), &value) in &self.entries {
                if j == decision_idx {
                    if let Some(&voter_id) = self.index_to_voter.get(&i) {
                        votes.insert(voter_id, value);
                    }
                }
            }
        }

        votes
    }

    /// Get list of all voters
    pub fn get_voters(&self) -> Vec<VoterId> {
        (0..self.num_voters)
            .filter_map(|i| self.index_to_voter.get(&i).copied())
            .collect()
    }

    /// Get list of all decisions
    pub fn get_decisions(&self) -> Vec<SlotId> {
        (0..self.num_decisions)
            .filter_map(|j| self.index_to_decision.get(&j).copied())
            .collect()
    }
}

/// Reputation vector for weighted consensus calculations with Votecoin integration
///
/// This vector now represents the complete Bitcoin Hivemind voting weights that
/// combine both historical accuracy (reputation) and economic stake (Votecoin holdings).
/// Higher combined weight voters have more influence in determining final outcomes.
///
/// # Bitcoin Hivemind Specification
/// Implements the complete voting weight formula:
/// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
#[derive(Debug, Clone)]
pub struct ReputationVector {
    /// Final voting weights indexed by voter (reputation × Votecoin proportion)
    reputations: HashMap<VoterId, f64>,
    /// Cached total reputation weight
    total_weight: Option<f64>,
}

impl ReputationVector {
    /// Create a new reputation vector
    pub fn new() -> Self {
        Self {
            reputations: HashMap::new(),
            total_weight: None,
        }
    }

    /// Set final voting weight for a voter (reputation × Votecoin proportion)
    ///
    /// # Arguments
    /// * `voter_id` - Voter to set weight for
    /// * `voting_weight` - Final voting weight incorporating both reputation and Votecoin
    ///
    /// # Bitcoin Hivemind Compliance
    /// This should be the result of: Base Reputation × Votecoin Holdings Proportion
    pub fn set_reputation(&mut self, voter_id: VoterId, voting_weight: f64) {
        self.reputations
            .insert(voter_id, voting_weight.clamp(0.0, 1.0));
        self.total_weight = None; // Invalidate cache
    }

    /// Get final voting weight for a voter
    ///
    /// # Arguments
    /// * `voter_id` - Voter to query
    ///
    /// # Returns
    /// Final voting weight, or 0.0 if not found (no influence without holdings)
    ///
    /// # Bitcoin Hivemind Specification
    /// Returns the complete voting weight that combines both reputation and Votecoin holdings.
    /// Zero weight means the voter either has no reputation or no Votecoin holdings.
    pub fn get_reputation(&self, voter_id: VoterId) -> f64 {
        self.reputations.get(&voter_id).copied().unwrap_or(0.0)
    }

    /// Get total reputation weight
    pub fn total_weight(&mut self) -> f64 {
        if let Some(weight) = self.total_weight {
            return weight;
        }

        let weight: f64 = self.reputations.values().sum();
        self.total_weight = Some(weight);
        weight
    }

    /// Normalize reputations to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.total_weight();
        if total > 0.0 {
            for reputation in self.reputations.values_mut() {
                *reputation /= total;
            }
            self.total_weight = Some(1.0);
        }
    }

    /// Convert to ndarray vector for mathematical operations
    ///
    /// # Arguments
    /// * `voters` - Ordered list of voters to include
    ///
    /// # Returns
    /// Array1 with reputation values in voter order
    pub fn to_array(&self, voters: &[VoterId]) -> Array1<f64> {
        Array1::from_iter(
            voters.iter().map(|&voter_id| self.get_reputation(voter_id)),
        )
    }

    /// Get number of voters with reputation
    pub fn len(&self) -> usize {
        self.reputations.len()
    }

    /// Check if reputation vector is empty
    pub fn is_empty(&self) -> bool {
        self.reputations.is_empty()
    }

    /// Create ReputationVector from VoterReputation structs with Votecoin integration
    ///
    /// # Arguments
    /// * `voter_reputations` - HashMap of VoterId to VoterReputation
    ///
    /// # Returns
    /// ReputationVector with final voting weights extracted from each VoterReputation
    ///
    /// # Bitcoin Hivemind Compliance
    /// Extracts the complete voting weights that already incorporate both
    /// reputation and Votecoin holdings from the VoterReputation structures.
    pub fn from_voter_reputations(
        voter_reputations: &HashMap<
            VoterId,
            crate::state::voting::types::VoterReputation,
        >,
    ) -> Self {
        let mut reputation_vector = Self::new();

        for (voter_id, voter_reputation) in voter_reputations {
            reputation_vector.set_reputation(
                *voter_id,
                voter_reputation.get_voting_weight(),
            );
        }

        reputation_vector
    }
}

impl Default for ReputationVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Basic vote aggregation functions
///
/// These functions provide simple aggregation methods that can be used
/// independently or as building blocks for more complex algorithms.
pub struct VoteAggregator;

impl VoteAggregator {
    /// Calculate simple majority vote for a decision
    ///
    /// # Arguments
    /// * `votes` - Mapping from VoterId to vote value
    ///
    /// # Returns
    /// Majority vote value (0.0 or 1.0 for binary decisions)
    pub fn simple_majority(votes: &HashMap<VoterId, f64>) -> f64 {
        if votes.is_empty() {
            return 0.5;
        }

        let total: f64 = votes.values().sum();
        let average = total / votes.len() as f64;

        // For binary decisions, round to 0 or 1
        if average >= 0.5 { 1.0 } else { 0.0 }
    }

    /// Calculate weighted average vote using reputation
    ///
    /// # Arguments
    /// * `votes` - Mapping from VoterId to vote value
    /// * `reputations` - Reputation vector
    ///
    /// # Returns
    /// Reputation-weighted average vote value
    pub fn weighted_average(
        votes: &HashMap<VoterId, f64>,
        reputations: &ReputationVector,
    ) -> f64 {
        if votes.is_empty() {
            return 0.5;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (&voter_id, &vote) in votes {
            let weight = reputations.get_reputation(voter_id);
            weighted_sum += vote * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.5
        }
    }

    /// Calculate median vote value
    ///
    /// # Arguments
    /// * `votes` - Mapping from VoterId to vote value
    ///
    /// # Returns
    /// Median vote value
    pub fn median_vote(votes: &HashMap<VoterId, f64>) -> f64 {
        if votes.is_empty() {
            return 0.5;
        }

        let mut values: Vec<f64> = votes.values().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = values.len();
        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }

    /// Calculate vote confidence based on agreement
    ///
    /// # Arguments
    /// * `votes` - Mapping from VoterId to vote value
    /// * `outcome` - Final outcome value
    ///
    /// # Returns
    /// Confidence score in [0.0, 1.0] based on voter agreement
    pub fn calculate_confidence(
        votes: &HashMap<VoterId, f64>,
        outcome: f64,
    ) -> f64 {
        if votes.is_empty() {
            return 0.0;
        }

        let mut agreement_count = 0;
        let total_votes = votes.len();

        for &vote in votes.values() {
            // For binary decisions, check if vote agrees with outcome
            let agrees = if outcome >= 0.5 {
                vote >= 0.5
            } else {
                vote < 0.5
            };

            if agrees {
                agreement_count += 1;
            }
        }

        agreement_count as f64 / total_votes as f64
    }
}

/// Utility functions for vote matrix operations
pub struct MatrixUtils;

impl MatrixUtils {
    /// Calculate participation rate for each decision
    ///
    /// # Arguments
    /// * `matrix` - Vote matrix to analyze
    ///
    /// # Returns
    /// HashMap mapping SlotId to participation rate [0.0, 1.0]
    pub fn calculate_participation_rates(
        matrix: &SparseVoteMatrix,
    ) -> HashMap<SlotId, f64> {
        let mut rates = HashMap::new();
        let (num_voters, _) = matrix.dimensions();

        if num_voters == 0 {
            return rates;
        }

        for decision_id in matrix.get_decisions() {
            let votes = matrix.get_decision_votes(decision_id);
            let rate = votes.len() as f64 / num_voters as f64;
            rates.insert(decision_id, rate);
        }

        rates
    }

    /// Calculate voter activity levels
    ///
    /// # Arguments
    /// * `matrix` - Vote matrix to analyze
    ///
    /// # Returns
    /// HashMap mapping VoterId to activity rate [0.0, 1.0]
    pub fn calculate_voter_activity(
        matrix: &SparseVoteMatrix,
    ) -> HashMap<VoterId, f64> {
        let mut activity = HashMap::new();
        let (_, num_decisions) = matrix.dimensions();

        if num_decisions == 0 {
            return activity;
        }

        for voter_id in matrix.get_voters() {
            let votes = matrix.get_voter_votes(voter_id);
            let rate = votes.len() as f64 / num_decisions as f64;
            activity.insert(voter_id, rate);
        }

        activity
    }

    /// Find outlier voters based on voting patterns
    ///
    /// # Arguments
    /// * `matrix` - Vote matrix to analyze
    /// * `threshold` - Outlier threshold (std deviations from mean)
    ///
    /// # Returns
    /// Set of voter IDs identified as outliers
    pub fn find_outlier_voters(
        matrix: &SparseVoteMatrix,
        threshold: f64,
    ) -> Result<Vec<VoterId>, VotingMathError> {
        let decisions = matrix.get_decisions();
        if decisions.is_empty() {
            return Ok(Vec::new());
        }

        let mut outliers = Vec::new();

        for voter_id in matrix.get_voters() {
            let mut deviations = Vec::new();

            // Calculate how much this voter deviates from majority on each decision
            for decision_id in &decisions {
                let decision_votes = matrix.get_decision_votes(*decision_id);
                if decision_votes.len() < 2 {
                    continue;
                }

                let majority = VoteAggregator::simple_majority(&decision_votes);
                if let Some(voter_vote) =
                    matrix.get_vote(voter_id, *decision_id)
                {
                    let deviation = (voter_vote - majority).abs();
                    deviations.push(deviation);
                }
            }

            if deviations.is_empty() {
                continue;
            }

            // Calculate mean and std deviation of this voter's deviations
            let mean: f64 =
                deviations.iter().sum::<f64>() / deviations.len() as f64;
            let variance: f64 =
                deviations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / deviations.len() as f64;
            let std_dev = variance.sqrt();

            // If voter consistently deviates more than threshold, mark as outlier
            if mean > threshold * std_dev {
                outliers.push(voter_id);
            }
        }

        Ok(outliers)
    }
}

/// Calculate consensus outcomes for decisions using Bitcoin Hivemind algorithm
///
/// This is the main entry point for Phase 3 consensus calculation.
/// Uses the reference-compliant consensus algorithm with SVD-based PCA.
///
/// # Arguments
/// * `vote_matrix` - Sparse vote matrix containing all votes for the period
/// * `reputation_vector` - Voter reputations (reputation × Votecoin)
///
/// # Returns
/// Detailed consensus result with SVD analysis
#[derive(Debug, Clone)]
pub struct DetailedConsensusResult {
    /// Decision outcomes mapped by SlotId
    pub outcomes: HashMap<SlotId, f64>,
    /// First principal component from SVD
    pub first_loading: Vec<f64>,
    /// Variance explained by the first component
    pub explained_variance: f64,
    /// Average certainty score across all decisions
    pub certainty: f64,
    /// Updated reputation weights after consensus
    pub updated_reputations: HashMap<VoterId, f64>,
    /// Outlier voters detected by the algorithm
    pub outliers: Vec<VoterId>,
}

/// HashMap mapping SlotId to consensus outcome value
///
/// # Bitcoin Hivemind Specification
/// Calculate consensus with full SVD information and metrics
/// Implements Section 4.2: Uses old reputation to calculate consensus outcomes.
/// These outcomes are then used to update reputation in the next period.
pub fn calculate_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &ReputationVector,
) -> Result<DetailedConsensusResult, VotingMathError> {
    let detailed = consensus::run_consensus(vote_matrix, reputation_vector)?;

    Ok(DetailedConsensusResult {
        outcomes: detailed.outcomes,
        first_loading: detailed.first_loading,
        explained_variance: detailed.explained_variance,
        certainty: detailed.certainty,
        updated_reputations: detailed.updated_reputations,
        outliers: detailed.outliers,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_reference_compat;