//! Bitcoin Hivemind Voting Mathematics

pub mod consensus;
pub mod constants;
pub mod matrix;

use crate::state::slots::SlotId;

use ndarray::{Array1, Array2};
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct SparseVoteMatrix {
    entries: HashMap<(usize, usize), f64>,
    voter_indices: HashMap<crate::types::Address, usize>,
    decision_indices: HashMap<SlotId, usize>,
    index_to_voter: HashMap<usize, crate::types::Address>,
    index_to_decision: HashMap<usize, SlotId>,
    num_voters: usize,
    num_decisions: usize,
}

impl SparseVoteMatrix {
    pub fn new(
        voters: Vec<crate::types::Address>,
        decisions: Vec<SlotId>,
    ) -> Self {
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

    pub fn set_vote(
        &mut self,
        voter_address: crate::types::Address,
        decision_id: SlotId,
        value: f64,
    ) -> Result<(), VotingMathError> {
        let voter_idx =
            *self.voter_indices.get(&voter_address).ok_or_else(|| {
                VotingMathError::InvalidReputation {
                    reason: format!(
                        "Voter {:?} not found in matrix",
                        voter_address
                    ),
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

    pub fn get_vote(
        &self,
        voter_address: crate::types::Address,
        decision_id: SlotId,
    ) -> Option<f64> {
        let voter_idx = *self.voter_indices.get(&voter_address)?;
        let decision_idx = *self.decision_indices.get(&decision_id)?;
        self.entries.get(&(voter_idx, decision_idx)).copied()
    }

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

    pub fn dimensions(&self) -> (usize, usize) {
        (self.num_voters, self.num_decisions)
    }

    pub fn num_votes(&self) -> usize {
        self.entries.len()
    }

    pub fn density(&self) -> f64 {
        if self.num_voters == 0 || self.num_decisions == 0 {
            return 0.0;
        }
        self.entries.len() as f64
            / (self.num_voters * self.num_decisions) as f64
    }

    pub fn get_voter_votes(
        &self,
        voter_address: crate::types::Address,
    ) -> HashMap<SlotId, f64> {
        let mut votes = HashMap::new();

        if let Some(&voter_idx) = self.voter_indices.get(&voter_address) {
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

    pub fn get_decision_votes(
        &self,
        decision_id: SlotId,
    ) -> HashMap<crate::types::Address, f64> {
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

    pub fn get_voters(&self) -> Vec<crate::types::Address> {
        (0..self.num_voters)
            .filter_map(|i| self.index_to_voter.get(&i).copied())
            .collect()
    }

    pub fn get_decisions(&self) -> Vec<SlotId> {
        (0..self.num_decisions)
            .filter_map(|j| self.index_to_decision.get(&j).copied())
            .collect()
    }
}

/// Voting weights combining reputation and Votecoin holdings.
/// Final Weight = Base Reputation × Votecoin Holdings Proportion
#[derive(Debug, Clone)]
pub struct ReputationVector {
    reputations: HashMap<crate::types::Address, f64>,
    total_weight: Option<f64>,
}

impl ReputationVector {
    pub fn new() -> Self {
        Self {
            reputations: HashMap::new(),
            total_weight: None,
        }
    }

    pub fn set_reputation(
        &mut self,
        voter_address: crate::types::Address,
        voting_weight: f64,
    ) {
        self.reputations
            .insert(voter_address, voting_weight.clamp(0.0, 1.0));
        self.total_weight = None;
    }

    pub fn get_reputation(&self, voter_address: crate::types::Address) -> f64 {
        self.reputations.get(&voter_address).copied().unwrap_or(0.0)
    }

    pub fn total_weight(&mut self) -> f64 {
        if let Some(weight) = self.total_weight {
            return weight;
        }

        let weight: f64 = self.reputations.values().sum();
        self.total_weight = Some(weight);
        weight
    }

    pub fn normalize(&mut self) {
        let total = self.total_weight();
        if total > 0.0 {
            for reputation in self.reputations.values_mut() {
                *reputation /= total;
            }
            self.total_weight = Some(1.0);
        }
    }

    pub fn to_array(&self, voters: &[crate::types::Address]) -> Array1<f64> {
        Array1::from_iter(
            voters.iter().map(|&voter_id| self.get_reputation(voter_id)),
        )
    }

    pub fn len(&self) -> usize {
        self.reputations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reputations.is_empty()
    }

    pub fn from_voter_reputations(
        voter_reputations: &HashMap<
            crate::types::Address,
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

pub struct VoteAggregator;

impl VoteAggregator {
    pub fn simple_majority(votes: &HashMap<crate::types::Address, f64>) -> f64 {
        if votes.is_empty() {
            return 0.5;
        }

        let total: f64 = votes.values().sum();
        let average = total / votes.len() as f64;

        if average >= 0.5 { 1.0 } else { 0.0 }
    }

    pub fn weighted_average(
        votes: &HashMap<crate::types::Address, f64>,
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

    pub fn median_vote(votes: &HashMap<crate::types::Address, f64>) -> f64 {
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

    pub fn calculate_confidence(
        votes: &HashMap<crate::types::Address, f64>,
        outcome: f64,
    ) -> f64 {
        if votes.is_empty() {
            return 0.0;
        }

        let mut agreement_count = 0;
        let total_votes = votes.len();

        for &vote in votes.values() {
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

pub struct MatrixUtils;

impl MatrixUtils {
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

    pub fn calculate_voter_activity(
        matrix: &SparseVoteMatrix,
    ) -> HashMap<crate::types::Address, f64> {
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

    pub fn find_outlier_voters(
        matrix: &SparseVoteMatrix,
        threshold: f64,
    ) -> Result<Vec<crate::types::Address>, VotingMathError> {
        let decisions = matrix.get_decisions();
        if decisions.is_empty() {
            return Ok(Vec::new());
        }

        let mut outliers = Vec::new();

        for voter_id in matrix.get_voters() {
            let mut deviations = Vec::new();

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

            let mean: f64 =
                deviations.iter().sum::<f64>() / deviations.len() as f64;
            let variance: f64 =
                deviations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / deviations.len() as f64;
            let std_dev = variance.sqrt();

            if mean > threshold * std_dev {
                outliers.push(voter_id);
            }
        }

        Ok(outliers)
    }
}

#[derive(Debug, Clone)]
pub struct DetailedConsensusResult {
    /// Decision outcomes. `None` indicates unanimous abstention (no votes cast).
    pub outcomes: HashMap<SlotId, Option<f64>>,
    pub first_loading: Vec<f64>,
    pub explained_variance: f64,
    pub certainty: f64,
    pub updated_reputations: HashMap<crate::types::Address, f64>,
    pub outliers: Vec<crate::types::Address>,
}

/// Calculate consensus using SVD-based PCA and current reputation weights.
/// Outcomes are then used to update reputation in the next period.
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
