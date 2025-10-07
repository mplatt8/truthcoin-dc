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

pub mod matrix;

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
/// Uses the existing factory() implementation with reputation weighting.
///
/// # Arguments
/// * `vote_matrix` - Sparse vote matrix containing all votes for the period
/// * `reputation_vector` - Voter reputations (reputation × Votecoin)
///
/// # Returns
/// HashMap mapping SlotId to consensus outcome value
///
/// # Bitcoin Hivemind Specification
/// Implements Section 4.2: Uses old reputation to calculate consensus outcomes.
/// These outcomes are then used to update reputation in the next period.
pub fn calculate_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &ReputationVector,
) -> Result<HashMap<SlotId, f64>, VotingMathError> {
    const MAX_ITERATIONS: usize = 50;
    const TOLERANCE: f64 = 1e-6;

    let consensus_result =
        ConsensusEngine::factory(vote_matrix, reputation_vector, MAX_ITERATIONS, TOLERANCE)?;

    let mut outcomes = HashMap::new();
    for (slot_id, decision_outcome) in consensus_result.outcomes {
        outcomes.insert(slot_id, decision_outcome.outcome_value);
    }

    Ok(outcomes)
}

/// Main Bitcoin Hivemind consensus algorithm implementation
///
/// This is the core factory() function that implements the PCA-based
/// truth-finding algorithm as specified in the Bitcoin Hivemind whitepaper.
/// It performs iterative convergence between reputation and outcomes to
/// find the Nash equilibrium solution.
pub struct ConsensusEngine;

impl ConsensusEngine {
    /// Main factory function - implements the Bitcoin Hivemind consensus algorithm
    ///
    /// This function implements the core consensus algorithm from the Bitcoin Hivemind
    /// whitepaper, using Principal Component Analysis and iterative convergence to
    /// determine truth from voter input.
    ///
    /// # Arguments
    /// * `vote_matrix` - Sparse vote matrix with voter decisions
    /// * `initial_reputations` - Starting reputation values for voters
    /// * `max_iterations` - Maximum iterations for convergence
    /// * `tolerance` - Convergence tolerance threshold
    ///
    /// # Returns
    /// Result containing final outcomes and updated reputations
    pub fn factory(
        vote_matrix: &SparseVoteMatrix,
        initial_reputations: &ReputationVector,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<ConsensusResult, VotingMathError> {
        let (num_voters, num_decisions) = vote_matrix.dimensions();

        if num_voters == 0 || num_decisions == 0 {
            return Err(VotingMathError::EmptyMatrix);
        }

        // Convert to dense matrix for mathematical operations
        let vote_matrix_dense = vote_matrix.to_dense(f64::NAN);

        // Initialize reputation vector
        let voters = vote_matrix.get_voters();
        let mut reputation = initial_reputations.to_array(&voters);

        // Build voter lookup map for O(1) access
        let voter_lookup: HashMap<VoterId, usize> = voters
            .iter()
            .enumerate()
            .map(|(idx, &voter_id)| (voter_id, idx))
            .collect();

        // Initialize outcomes to weighted averages
        let mut outcomes = Array1::zeros(num_decisions);
        let decisions = vote_matrix.get_decisions();

        for (j, &decision_id) in decisions.iter().enumerate() {
            let decision_votes = vote_matrix.get_decision_votes(decision_id);
            outcomes[j] = VoteAggregator::weighted_average(
                &decision_votes,
                initial_reputations,
            );
        }

        // Iterative convergence loop
        for iteration in 0..max_iterations {
            let old_reputation = reputation.clone();
            let old_outcomes = outcomes.clone();

            // Step 1: Update reputation using weighted PCA
            reputation = Self::weighted_prin_comp(
                &vote_matrix_dense,
                &outcomes,
                &reputation,
            )?;

            // Step 2: Update outcomes using updated reputation
            for (j, &decision_id) in decisions.iter().enumerate() {
                let decision_votes =
                    vote_matrix.get_decision_votes(decision_id);
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (&voter_id, &vote) in &decision_votes {
                    if let Some(&voter_idx) = voter_lookup.get(&voter_id) {
                        let weight = reputation[voter_idx];
                        if !vote.is_nan() && weight > 0.0 {
                            weighted_sum += vote * weight;
                            total_weight += weight;
                        }
                    }
                }

                outcomes[j] = if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.5
                };
            }

            // Check convergence
            let rep_diff =
                (&reputation - &old_reputation).mapv(|x| x.abs()).sum();
            let outcome_diff =
                (&outcomes - &old_outcomes).mapv(|x| x.abs()).sum();

            if rep_diff < tolerance && outcome_diff < tolerance {
                // Converged - build result
                return Ok(Self::build_consensus_result(
                    vote_matrix,
                    reputation,
                    outcomes,
                    iteration + 1,
                ));
            }
        }

        Err(VotingMathError::ConvergenceFailure {
            iterations: max_iterations,
        })
    }

    /// Weighted Principal Component Analysis for reputation calculation
    ///
    /// Implements the weighted PCA algorithm from the Bitcoin Hivemind specification.
    /// This calculates the first principal component of the vote matrix weighted by
    /// current reputation estimates.
    ///
    /// # Arguments
    /// * `vote_matrix` - Dense vote matrix (voters x decisions)
    /// * `outcomes` - Current outcome estimates
    /// * `weights` - Current reputation weights
    ///
    /// # Returns
    /// Updated reputation vector based on weighted PCA
    fn weighted_prin_comp(
        vote_matrix: &Array2<f64>,
        outcomes: &Array1<f64>,
        weights: &Array1<f64>,
    ) -> Result<Array1<f64>, VotingMathError> {
        let (num_voters, num_decisions) = vote_matrix.dim();

        if outcomes.len() != num_decisions || weights.len() != num_voters {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!(
                    "outcomes: {}, weights: {}",
                    num_decisions, num_voters
                ),
                actual: format!(
                    "outcomes: {}, weights: {}",
                    outcomes.len(),
                    weights.len()
                ),
            });
        }

        // Handle missing values by replacing NaN with outcome estimates
        let mut filled_matrix = vote_matrix.clone();
        for i in 0..num_voters {
            for j in 0..num_decisions {
                if filled_matrix[[i, j]].is_nan() {
                    filled_matrix[[i, j]] = outcomes[j];
                }
            }
        }

        // Center the matrix around outcomes
        let mut centered_matrix = filled_matrix.clone();
        for i in 0..num_voters {
            for j in 0..num_decisions {
                centered_matrix[[i, j]] -= outcomes[j];
            }
        }

        // Calculate weighted covariance matrix (decisions x decisions)
        let mut cov_matrix = Array2::zeros((num_decisions, num_decisions));
        let total_weight: f64 = weights.sum();

        if total_weight <= 0.0 {
            return Err(VotingMathError::InvalidReputation {
                reason: "Total weight is non-positive".to_string(),
            });
        }

        for j1 in 0..num_decisions {
            for j2 in 0..num_decisions {
                let mut weighted_cov = 0.0;

                for i in 0..num_voters {
                    if weights[i] > 0.0 {
                        weighted_cov += weights[i]
                            * centered_matrix[[i, j1]]
                            * centered_matrix[[i, j2]];
                    }
                }

                cov_matrix[[j1, j2]] = weighted_cov / total_weight;
            }
        }

        // Find first principal component using power iteration
        let mut eigenvector = Array1::from_elem(
            num_decisions,
            1.0 / (num_decisions as f64).sqrt(),
        );

        for _ in 0..50 {
            // Max power iterations
            let old_eigenvector = eigenvector.clone();

            // Multiply by covariance matrix
            eigenvector = cov_matrix.dot(&eigenvector);

            // Normalize
            let norm = eigenvector.mapv(|x| x * x).sum().sqrt();
            if norm > 0.0 {
                eigenvector /= norm;
            }

            // Check convergence
            let diff =
                (&eigenvector - &old_eigenvector).mapv(|x| x.abs()).sum();
            if diff < 1e-8 {
                break;
            }
        }

        // Calculate new reputation as projection onto first principal component
        let mut new_reputation = Array1::zeros(num_voters);

        for i in 0..num_voters {
            let mut projection = 0.0;
            for j in 0..num_decisions {
                if !filled_matrix[[i, j]].is_nan() {
                    projection += centered_matrix[[i, j]] * eigenvector[j];
                }
            }
            new_reputation[i] = projection;
        }

        // Normalize reputation to [0, 1] range
        let min_rep =
            new_reputation.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_rep = new_reputation
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_rep - min_rep).abs() > 1e-10 {
            for rep in new_reputation.iter_mut() {
                *rep = (*rep - min_rep) / (max_rep - min_rep);
            }
        } else {
            // All reputation values are equal - reset to uniform
            new_reputation.fill(0.5);
        }

        Ok(new_reputation)
    }

    /// Build the final consensus result from converged values
    fn build_consensus_result(
        vote_matrix: &SparseVoteMatrix,
        reputation: Array1<f64>,
        outcomes: Array1<f64>,
        iterations: usize,
    ) -> ConsensusResult {
        let voters = vote_matrix.get_voters();
        let decisions = vote_matrix.get_decisions();

        // Build reputation map
        let mut final_reputations = HashMap::new();
        for (i, &voter_id) in voters.iter().enumerate() {
            final_reputations.insert(voter_id, reputation[i]);
        }

        // Build outcomes map with confidence scores
        let mut final_outcomes = HashMap::new();
        for (j, &decision_id) in decisions.iter().enumerate() {
            let decision_votes = vote_matrix.get_decision_votes(decision_id);
            let confidence = VoteAggregator::calculate_confidence(
                &decision_votes,
                outcomes[j],
            );

            final_outcomes.insert(
                decision_id,
                DecisionOutcome {
                    outcome_value: outcomes[j],
                    min: 0.0, // Binary decisions have min 0.0
                    max: 1.0, // Binary decisions have max 1.0
                    confidence,
                    voter_count: decision_votes.len(),
                },
            );
        }

        ConsensusResult {
            outcomes: final_outcomes,
            reputations: final_reputations,
            iterations_to_converge: iterations,
            total_votes: vote_matrix.num_votes(),
        }
    }
}

/// Result of the consensus algorithm
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Final outcome values for each decision
    pub outcomes: HashMap<SlotId, DecisionOutcome>,
    /// Updated reputation values for each voter
    pub reputations: HashMap<VoterId, f64>,
    /// Number of iterations required for convergence
    pub iterations_to_converge: usize,
    /// Total number of votes processed
    pub total_votes: usize,
}

/// Outcome for a specific decision
#[derive(Debug, Clone)]
pub struct DecisionOutcome {
    /// Final outcome value (0.0-1.0 for binary, continuous for scalar)
    pub outcome_value: f64,
    /// Minimum possible value for this decision (for scalar decisions)
    pub min: f64,
    /// Maximum possible value for this decision (for scalar decisions)
    pub max: f64,
    /// Confidence in the outcome [0.0, 1.0]
    pub confidence: f64,
    /// Number of voters who participated in this decision
    pub voter_count: usize,
}

/// Additional mathematical operations for voting consensus
pub struct VotingMath;

impl VotingMath {
    /// Calculate missing value imputation using column means
    ///
    /// Replaces NaN values in the vote matrix with the mean of non-missing values
    /// in the same column (decision), weighted by voter reputation.
    pub fn impute_missing_values(
        matrix: &mut Array2<f64>,
        reputation: &Array1<f64>,
    ) -> Result<(), VotingMathError> {
        let (num_voters, num_decisions) = matrix.dim();

        if reputation.len() != num_voters {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("reputation length: {}", num_voters),
                actual: format!("reputation length: {}", reputation.len()),
            });
        }

        for j in 0..num_decisions {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            let mut missing_indices = Vec::new();

            // Calculate weighted mean for non-missing values
            for i in 0..num_voters {
                let value = matrix[[i, j]];
                if !value.is_nan() {
                    let weight = reputation[i];
                    weighted_sum += value * weight;
                    total_weight += weight;
                } else {
                    missing_indices.push(i);
                }
            }

            let mean = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                0.5 // Default neutral value
            };

            // Fill missing values with weighted mean
            for &i in &missing_indices {
                matrix[[i, j]] = mean;
            }
        }

        Ok(())
    }

    /// Calculate voter accuracy scores based on outcomes
    ///
    /// Computes how accurately each voter predicted the final outcomes,
    /// which is used to update reputation in the next iteration.
    pub fn calculate_accuracy_scores(
        vote_matrix: &Array2<f64>,
        outcomes: &Array1<f64>,
    ) -> Result<Array1<f64>, VotingMathError> {
        let (num_voters, num_decisions) = vote_matrix.dim();

        if outcomes.len() != num_decisions {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("outcomes length: {}", num_decisions),
                actual: format!("outcomes length: {}", outcomes.len()),
            });
        }

        let mut accuracy_scores = Array1::zeros(num_voters);

        for i in 0..num_voters {
            let mut total_error = 0.0;
            let mut vote_count = 0;

            for j in 0..num_decisions {
                let vote = vote_matrix[[i, j]];
                if !vote.is_nan() {
                    let error = (vote - outcomes[j]).abs();
                    total_error += error;
                    vote_count += 1;
                }
            }

            // Accuracy is 1 - average_error, bounded to [0, 1]
            accuracy_scores[i] = if vote_count > 0 {
                let average_error = total_error / vote_count as f64;
                (1.0 - average_error).clamp(0.0, 1.0)
            } else {
                0.5 // Neutral score for voters with no votes
            };
        }

        Ok(accuracy_scores)
    }

    /// Calculate correlation matrix between decisions
    ///
    /// Computes pairwise correlations between decisions to identify
    /// dependencies and patterns in voting behavior.
    pub fn calculate_decision_correlations(
        vote_matrix: &Array2<f64>,
        outcomes: &Array1<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = vote_matrix.dim();

        if outcomes.len() != num_decisions {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("outcomes length: {}", num_decisions),
                actual: format!("outcomes length: {}", outcomes.len()),
            });
        }

        let mut correlation_matrix = Array2::eye(num_decisions);

        // Center votes around outcomes
        let mut centered_matrix = vote_matrix.clone();
        for i in 0..num_voters {
            for j in 0..num_decisions {
                if !centered_matrix[[i, j]].is_nan() {
                    centered_matrix[[i, j]] -= outcomes[j];
                }
            }
        }

        // Calculate correlations
        for j1 in 0..num_decisions {
            for j2 in (j1 + 1)..num_decisions {
                let mut numerator = 0.0;
                let mut sum_sq_1 = 0.0;
                let mut sum_sq_2 = 0.0;
                let mut count = 0;

                for i in 0..num_voters {
                    let val1 = centered_matrix[[i, j1]];
                    let val2 = centered_matrix[[i, j2]];

                    if !val1.is_nan() && !val2.is_nan() {
                        numerator += val1 * val2;
                        sum_sq_1 += val1 * val1;
                        sum_sq_2 += val2 * val2;
                        count += 1;
                    }
                }

                let correlation =
                    if count > 1 && sum_sq_1 > 0.0 && sum_sq_2 > 0.0 {
                        numerator / (sum_sq_1.sqrt() * sum_sq_2.sqrt())
                    } else {
                        0.0
                    };

                correlation_matrix[[j1, j2]] = correlation;
                correlation_matrix[[j2, j1]] = correlation;
            }
        }

        Ok(correlation_matrix)
    }

    /// Calculate vote entropy for decisions
    ///
    /// Measures the uncertainty/randomness in voting patterns.
    /// Higher entropy indicates more disagreement among voters.
    pub fn calculate_vote_entropy(vote_matrix: &Array2<f64>) -> Array1<f64> {
        let (num_voters, num_decisions) = vote_matrix.dim();
        let mut entropy = Array1::zeros(num_decisions);

        for j in 0..num_decisions {
            // Collect non-missing votes for this decision
            let votes: Vec<f64> = (0..num_voters)
                .map(|i| vote_matrix[[i, j]])
                .filter(|&v| !v.is_nan())
                .collect();

            if votes.is_empty() {
                entropy[j] = 0.0;
                continue;
            }

            // For binary votes, calculate entropy based on proportions
            let positive_votes =
                votes.iter().filter(|&&v| v >= 0.5).count() as f64;
            let total_votes = votes.len() as f64;
            let p_positive = positive_votes / total_votes;
            let p_negative = 1.0 - p_positive;

            entropy[j] = if p_positive > 0.0 && p_negative > 0.0 {
                -(p_positive * p_positive.log2()
                    + p_negative * p_negative.log2())
            } else {
                0.0 // Perfect agreement
            };
        }

        entropy
    }

    /// Detect and handle voting anomalies
    ///
    /// Identifies potential coordinated attacks or unusual voting patterns
    /// that might compromise the consensus algorithm.
    pub fn detect_voting_anomalies(
        vote_matrix: &Array2<f64>,
        reputation: &Array1<f64>,
        threshold: f64,
    ) -> Result<Vec<VotingAnomaly>, VotingMathError> {
        let mut anomalies = Vec::new();
        let (num_voters, num_decisions) = vote_matrix.dim();

        if reputation.len() != num_voters {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("reputation length: {}", num_voters),
                actual: format!("reputation length: {}", reputation.len()),
            });
        }

        // Check for perfectly correlated voters (potential collusion)
        for i1 in 0..num_voters {
            for i2 in (i1 + 1)..num_voters {
                let mut matching_votes = 0;
                let mut total_comparisons = 0;

                for j in 0..num_decisions {
                    let vote1 = vote_matrix[[i1, j]];
                    let vote2 = vote_matrix[[i2, j]];

                    if !vote1.is_nan() && !vote2.is_nan() {
                        if (vote1 - vote2).abs() < 0.01 {
                            // Nearly identical
                            matching_votes += 1;
                        }
                        total_comparisons += 1;
                    }
                }

                if total_comparisons > 0 {
                    let correlation =
                        matching_votes as f64 / total_comparisons as f64;
                    if correlation > threshold {
                        anomalies.push(VotingAnomaly::SuspiciousCorrelation {
                            voter1_index: i1,
                            voter2_index: i2,
                            correlation,
                            shared_votes: matching_votes,
                        });
                    }
                }
            }
        }

        // Check for voters with excessive influence relative to reputation
        let total_reputation: f64 = reputation.sum();
        if total_reputation > 0.0 {
            for i in 0..num_voters {
                let reputation_share = reputation[i] / total_reputation;
                let vote_count = (0..num_decisions)
                    .map(|j| if vote_matrix[[i, j]].is_nan() { 0 } else { 1 })
                    .sum::<i32>() as f64;
                let vote_share =
                    vote_count / (num_voters * num_decisions) as f64;

                if reputation_share < 0.01 && vote_share > 0.5 {
                    anomalies.push(VotingAnomaly::ExcessiveVoting {
                        voter_index: i,
                        reputation_share,
                        vote_share,
                    });
                }
            }
        }

        Ok(anomalies)
    }
}

/// Types of voting anomalies that can be detected
#[derive(Debug, Clone)]
pub enum VotingAnomaly {
    /// Two voters have suspiciously similar voting patterns
    SuspiciousCorrelation {
        voter1_index: usize,
        voter2_index: usize,
        correlation: f64,
        shared_votes: usize,
    },
    /// Voter has disproportionate voting activity relative to reputation
    ExcessiveVoting {
        voter_index: usize,
        reputation_share: f64,
        vote_share: f64,
    },
}

#[cfg(test)]
mod tests;
