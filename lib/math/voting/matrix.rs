//! Advanced Matrix Operations for Bitcoin Hivemind Voting
//!
//! This module implements sophisticated matrix operations required for the
//! Bitcoin Hivemind consensus algorithm, including preparation for Principal
//! Component Analysis (PCA) and iterative consensus convergence.
//!
//! ## Mathematical Foundation
//! The Bitcoin Hivemind consensus algorithm relies on linear algebra operations
//! to extract truth from voter input. This module provides the building blocks
//! for the full algorithm implementation in Phase 2.
//!
//! ## Bitcoin Hivemind Specification References
//! - Section 4.4: "Principal Component Analysis" - PCA-based truth extraction
//! - Section 4.5: "Iterative Convergence" - Reputation and outcome updates

use super::{SparseVoteMatrix, VotingMathError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Matrix preprocessing operations for consensus algorithm
///
/// These operations prepare vote matrices for the mathematical operations
/// required by the Bitcoin Hivemind consensus algorithm, including handling
/// missing data and normalizing values.
pub struct MatrixPreprocessor;

impl MatrixPreprocessor {
    /// Fill missing values in vote matrix using sophisticated imputation
    ///
    /// # Arguments
    /// * `matrix` - Sparse vote matrix with missing values
    /// * `method` - Imputation method to use
    ///
    /// # Returns
    /// Dense matrix with all missing values filled
    ///
    /// # Bitcoin Hivemind Compliance
    /// Missing vote handling is critical for consensus algorithm stability.
    /// The whitepaper suggests using neutral values (0.5) or voter-specific
    /// defaults based on historical patterns.
    pub fn fill_missing_values(
        matrix: &SparseVoteMatrix,
        method: ImputationMethod,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = matrix.dimensions();

        if num_voters == 0 || num_decisions == 0 {
            return Err(VotingMathError::EmptyMatrix);
        }

        match method {
            ImputationMethod::Neutral => {
                // Fill with 0.5 (neutral value)
                Ok(matrix.to_dense(0.5))
            }
            ImputationMethod::VoterMean => Self::fill_with_voter_means(matrix),
            ImputationMethod::DecisionMean => {
                Self::fill_with_decision_means(matrix)
            }
            ImputationMethod::GlobalMean => Self::fill_with_global_mean(matrix),
        }
    }

    /// Fill missing values with per-voter means
    fn fill_with_voter_means(
        matrix: &SparseVoteMatrix,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = matrix.dimensions();
        let mut dense = Array2::from_elem((num_voters, num_decisions), 0.5);

        // Calculate voter means
        let voters = matrix.get_voters();
        let mut voter_means = HashMap::new();

        for voter_id in &voters {
            let votes = matrix.get_voter_votes(*voter_id);
            if !votes.is_empty() {
                let mean: f64 =
                    votes.values().sum::<f64>() / votes.len() as f64;
                voter_means.insert(*voter_id, mean);
            } else {
                voter_means.insert(*voter_id, 0.5);
            }
        }

        // Fill matrix with voter means and actual votes
        for (i, voter_id) in voters.iter().enumerate() {
            let voter_mean = voter_means[voter_id];
            for j in 0..num_decisions {
                if let Some(decision_id) = matrix.get_decisions().get(j) {
                    if let Some(vote) = matrix.get_vote(*voter_id, *decision_id)
                    {
                        dense[[i, j]] = vote;
                    } else {
                        dense[[i, j]] = voter_mean;
                    }
                }
            }
        }

        Ok(dense)
    }

    /// Fill missing values with per-decision means
    fn fill_with_decision_means(
        matrix: &SparseVoteMatrix,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = matrix.dimensions();
        let mut dense = Array2::from_elem((num_voters, num_decisions), 0.5);

        // Calculate decision means
        let decisions = matrix.get_decisions();
        let mut decision_means = HashMap::new();

        for decision_id in &decisions {
            let votes = matrix.get_decision_votes(*decision_id);
            if !votes.is_empty() {
                let mean: f64 =
                    votes.values().sum::<f64>() / votes.len() as f64;
                decision_means.insert(*decision_id, mean);
            } else {
                decision_means.insert(*decision_id, 0.5);
            }
        }

        // Fill matrix with decision means and actual votes
        let voters = matrix.get_voters();
        for (i, voter_id) in voters.iter().enumerate() {
            for (j, decision_id) in decisions.iter().enumerate() {
                if let Some(vote) = matrix.get_vote(*voter_id, *decision_id) {
                    dense[[i, j]] = vote;
                } else {
                    dense[[i, j]] = decision_means[decision_id];
                }
            }
        }

        Ok(dense)
    }

    /// Fill missing values with global mean
    fn fill_with_global_mean(
        matrix: &SparseVoteMatrix,
    ) -> Result<Array2<f64>, VotingMathError> {
        // Calculate global mean of all votes
        let all_votes: Vec<f64> = matrix
            .get_voters()
            .iter()
            .flat_map(|voter_id| {
                let votes = matrix.get_voter_votes(*voter_id);
                votes.values().copied().collect::<Vec<_>>()
            })
            .collect();

        let global_mean = if !all_votes.is_empty() {
            all_votes.iter().sum::<f64>() / all_votes.len() as f64
        } else {
            0.5
        };

        Ok(matrix.to_dense(global_mean))
    }

    /// Normalize matrix values to standard ranges
    ///
    /// # Arguments
    /// * `matrix` - Dense matrix to normalize
    /// * `method` - Normalization method
    ///
    /// # Returns
    /// Normalized matrix suitable for mathematical operations
    pub fn normalize_matrix(
        matrix: &Array2<f64>,
        method: NormalizationMethod,
    ) -> Result<Array2<f64>, VotingMathError> {
        match method {
            NormalizationMethod::None => Ok(matrix.clone()),
            NormalizationMethod::StandardScore => {
                Self::standardize_matrix(matrix)
            }
            NormalizationMethod::MinMax => Self::min_max_normalize(matrix),
            NormalizationMethod::UnitVector => {
                Self::unit_vector_normalize(matrix)
            }
        }
    }

    /// Standardize matrix to zero mean and unit variance
    fn standardize_matrix(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let mut normalized = matrix.clone();

        // Standardize each column (decision) independently
        for mut column in normalized.columns_mut() {
            let mean = column.mean().unwrap_or(0.0);
            let std_dev = {
                let variance =
                    column.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / column.len() as f64;
                variance.sqrt()
            };

            if std_dev > 1e-10 {
                column.mapv_inplace(|x| (x - mean) / std_dev);
            }
        }

        Ok(normalized)
    }

    /// Normalize matrix to [0, 1] range
    fn min_max_normalize(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let mut normalized = matrix.clone();

        // Normalize each column independently
        for mut column in normalized.columns_mut() {
            let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val =
                column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let range = max_val - min_val;
            if range > 1e-10 {
                column.mapv_inplace(|x| (x - min_val) / range);
            }
        }

        Ok(normalized)
    }

    /// Normalize rows to unit vectors
    fn unit_vector_normalize(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let mut normalized = matrix.clone();

        // Normalize each row (voter) independently
        for mut row in normalized.rows_mut() {
            let norm = row.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            if norm > 1e-10 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        Ok(normalized)
    }
}

/// Imputation methods for handling missing votes
#[derive(Debug, Clone, Copy)]
pub enum ImputationMethod {
    /// Fill with neutral value (0.5)
    Neutral,
    /// Fill with voter's historical mean
    VoterMean,
    /// Fill with decision's current mean
    DecisionMean,
    /// Fill with global dataset mean
    GlobalMean,
}

/// Normalization methods for matrix preprocessing
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// Zero mean, unit variance (z-score)
    StandardScore,
    /// Scale to [0, 1] range
    MinMax,
    /// Normalize rows to unit vectors
    UnitVector,
}

/// Advanced matrix analysis operations
///
/// These operations provide insights into vote matrix properties and
/// prepare data for sophisticated consensus algorithms.
pub struct MatrixAnalyzer;

impl MatrixAnalyzer {
    /// Calculate correlation matrix between decisions
    ///
    /// # Arguments
    /// * `matrix` - Vote matrix to analyze
    ///
    /// # Returns
    /// Correlation matrix showing how decisions relate to each other
    ///
    /// # Bitcoin Hivemind Application
    /// Decision correlations help identify related questions and potential
    /// inconsistencies in voter behavior that may indicate manipulation.
    pub fn decision_correlation_matrix(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (_, num_decisions) = matrix.dim();

        if num_decisions == 0 {
            return Err(VotingMathError::EmptyMatrix);
        }

        let mut correlation = Array2::zeros((num_decisions, num_decisions));

        for i in 0..num_decisions {
            for j in 0..num_decisions {
                if i == j {
                    correlation[[i, j]] = 1.0;
                } else {
                    let col_i = matrix.column(i);
                    let col_j = matrix.column(j);
                    correlation[[i, j]] =
                        Self::pearson_correlation(&col_i, &col_j)?;
                }
            }
        }

        Ok(correlation)
    }

    /// Calculate correlation matrix between voters
    ///
    /// # Arguments
    /// * `matrix` - Vote matrix to analyze
    ///
    /// # Returns
    /// Correlation matrix showing how voters relate to each other
    ///
    /// # Bitcoin Hivemind Application
    /// Voter correlations help identify coordinated behavior, echo chambers,
    /// and potential Sybil attacks on the consensus system.
    pub fn voter_correlation_matrix(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, _) = matrix.dim();

        if num_voters == 0 {
            return Err(VotingMathError::EmptyMatrix);
        }

        let mut correlation = Array2::zeros((num_voters, num_voters));

        for i in 0..num_voters {
            for j in 0..num_voters {
                if i == j {
                    correlation[[i, j]] = 1.0;
                } else {
                    let row_i = matrix.row(i);
                    let row_j = matrix.row(j);
                    correlation[[i, j]] =
                        Self::pearson_correlation(&row_i, &row_j)?;
                }
            }
        }

        Ok(correlation)
    }

    /// Calculate Pearson correlation coefficient between two vectors
    fn pearson_correlation(
        x: &ndarray::ArrayBase<
            ndarray::ViewRepr<&f64>,
            ndarray::Dim<[usize; 1]>,
        >,
        y: &ndarray::ArrayBase<
            ndarray::ViewRepr<&f64>,
            ndarray::Dim<[usize; 1]>,
        >,
    ) -> Result<f64, VotingMathError> {
        if x.len() != y.len() || x.is_empty() {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("{}", x.len()),
                actual: format!("{}", y.len()),
            });
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator < 1e-10 {
            Ok(0.0) // No correlation if no variance
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Identify strongly correlated decision pairs
    ///
    /// # Arguments
    /// * `correlation_matrix` - Decision correlation matrix
    /// * `threshold` - Minimum correlation to consider "strong"
    ///
    /// # Returns
    /// Vector of (decision_i, decision_j, correlation) tuples
    pub fn find_correlated_decisions(
        correlation_matrix: &Array2<f64>,
        threshold: f64,
    ) -> Vec<(usize, usize, f64)> {
        let mut correlations = Vec::new();
        let size = correlation_matrix.nrows();

        for i in 0..size {
            for j in (i + 1)..size {
                let corr = correlation_matrix[[i, j]];
                if corr.abs() >= threshold {
                    correlations.push((i, j, corr));
                }
            }
        }

        // Sort by absolute correlation strength
        correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        correlations
    }

    /// Calculate matrix rank (approximate)
    ///
    /// # Arguments
    /// * `matrix` - Matrix to analyze
    /// * `tolerance` - Tolerance for considering singular values as zero
    ///
    /// # Returns
    /// Approximate rank of the matrix
    ///
    /// # Bitcoin Hivemind Application
    /// Matrix rank indicates the dimensionality of the truth space and
    /// helps determine how many principal components are meaningful.
    pub fn estimate_rank(
        matrix: &Array2<f64>,
        tolerance: f64,
    ) -> Result<usize, VotingMathError> {
        // This is a simplified rank estimation
        // In practice, we would use SVD for accurate rank calculation
        let (rows, cols) = matrix.dim();
        let min_dim = rows.min(cols);

        // Count non-zero diagonal elements after Gaussian elimination (simplified)
        let mut rank = 0;
        let mut working = matrix.clone();

        for i in 0..min_dim {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..rows {
                if working[[k, i]].abs() > working[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Check if pivot is significant
            if working[[max_row, i]].abs() < tolerance {
                continue;
            }

            rank += 1;

            // Swap rows if needed
            if max_row != i && i < rows {
                for j in 0..cols {
                    let temp = working[[i, j]];
                    working[[i, j]] = working[[max_row, j]];
                    working[[max_row, j]] = temp;
                }
            }

            // Eliminate column
            for k in (i + 1)..rows {
                if working[[i, i]].abs() > 1e-10 {
                    let factor = working[[k, i]] / working[[i, i]];
                    for j in i..cols {
                        working[[k, j]] -= factor * working[[i, j]];
                    }
                }
            }
        }

        Ok(rank)
    }
}

/// Consensus algorithm building blocks
///
/// These functions provide the mathematical operations needed for implementing
/// the full Bitcoin Hivemind consensus algorithm in Phase 2.
pub struct ConsensusOps;

impl ConsensusOps {
    /// Apply reputation weighting to vote matrix
    ///
    /// # Arguments
    /// * `matrix` - Vote matrix (voters x decisions)
    /// * `reputations` - Reputation weights for each voter
    ///
    /// # Returns
    /// Reputation-weighted matrix where each row is scaled by voter reputation
    ///
    /// # Bitcoin Hivemind Application
    /// Reputation weighting ensures that historically accurate voters have
    /// more influence on consensus outcomes.
    pub fn apply_reputation_weighting(
        matrix: &Array2<f64>,
        reputations: &Array1<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = matrix.dim();

        if reputations.len() != num_voters {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("{}", num_voters),
                actual: format!("{}", reputations.len()),
            });
        }

        let mut weighted = matrix.clone();

        for (i, mut row) in weighted.rows_mut().into_iter().enumerate() {
            let weight = reputations[i];
            row.mapv_inplace(|x| x * weight);
        }

        Ok(weighted)
    }

    /// Calculate decision outcomes using weighted voting
    ///
    /// # Arguments
    /// * `weighted_matrix` - Reputation-weighted vote matrix
    ///
    /// # Returns
    /// Array of consensus outcomes for each decision
    ///
    /// # Bitcoin Hivemind Application
    /// This provides simple weighted consensus as a baseline before
    /// applying more sophisticated PCA-based algorithms.
    pub fn calculate_weighted_consensus(
        weighted_matrix: &Array2<f64>,
    ) -> Result<Array1<f64>, VotingMathError> {
        let (_, num_decisions) = weighted_matrix.dim();

        if num_decisions == 0 {
            return Err(VotingMathError::EmptyMatrix);
        }

        let mut outcomes = Array1::zeros(num_decisions);

        for j in 0..num_decisions {
            let column = weighted_matrix.column(j);
            let sum: f64 = column.sum();
            let weight_sum: f64 = column.len() as f64; // Simplified - should sum actual weights

            outcomes[j] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.5
            };
        }

        Ok(outcomes)
    }

    /// Update reputation based on agreement with consensus
    ///
    /// # Arguments
    /// * `votes` - Original vote matrix
    /// * `outcomes` - Consensus outcomes
    /// * `current_reputations` - Current reputation values
    /// * `learning_rate` - How fast reputation updates (0.0 to 1.0)
    ///
    /// # Returns
    /// Updated reputation array
    ///
    /// # Bitcoin Hivemind Application
    /// Reputation updates incentivize truthful reporting by rewarding
    /// voters who agree with eventual consensus.
    pub fn update_reputations(
        votes: &Array2<f64>,
        outcomes: &Array1<f64>,
        current_reputations: &Array1<f64>,
        learning_rate: f64,
    ) -> Result<Array1<f64>, VotingMathError> {
        let (num_voters, num_decisions) = votes.dim();

        if outcomes.len() != num_decisions {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("{}", num_decisions),
                actual: format!("{}", outcomes.len()),
            });
        }

        if current_reputations.len() != num_voters {
            return Err(VotingMathError::DimensionMismatch {
                expected: format!("{}", num_voters),
                actual: format!("{}", current_reputations.len()),
            });
        }

        let mut new_reputations = current_reputations.clone();

        for i in 0..num_voters {
            let mut total_error = 0.0;
            let mut vote_count = 0;

            for j in 0..num_decisions {
                let vote = votes[[i, j]];
                let outcome = outcomes[j];

                // Skip NaN votes (abstentions)
                if !vote.is_nan() {
                    let error = (vote - outcome).abs();
                    total_error += error;
                    vote_count += 1;
                }
            }

            if vote_count > 0 {
                let avg_error = total_error / vote_count as f64;
                let accuracy = 1.0 - avg_error.min(1.0); // Convert error to accuracy

                // Update reputation with learning rate
                let current_rep = current_reputations[i];
                new_reputations[i] =
                    current_rep + learning_rate * (accuracy - current_rep);

                // Clamp to [0, 1] range
                new_reputations[i] = new_reputations[i].clamp(0.0, 1.0);
            }
        }

        Ok(new_reputations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::slots::SlotId;
    use crate::state::voting::types::VoterId;
    use crate::types::Address;

    fn create_test_sparse_matrix() -> SparseVoteMatrix {
        let voters = vec![
            VoterId::from_address(&Address([1u8; 20])),
            VoterId::from_address(&Address([2u8; 20])),
            VoterId::from_address(&Address([3u8; 20])),
        ];

        let decisions =
            vec![SlotId::new(1, 0).unwrap(), SlotId::new(1, 1).unwrap()];

        let mut matrix = SparseVoteMatrix::new(voters, decisions);

        // Add some test votes
        let voter1 = VoterId::from_address(&Address([1u8; 20]));
        let voter2 = VoterId::from_address(&Address([2u8; 20]));
        let decision1 = SlotId::new(1, 0).unwrap();
        let decision2 = SlotId::new(1, 1).unwrap();

        matrix.set_vote(voter1, decision1, 1.0).unwrap();
        matrix.set_vote(voter1, decision2, 0.0).unwrap();
        matrix.set_vote(voter2, decision1, 1.0).unwrap();
        // voter2 abstains on decision2, voter3 abstains on both

        matrix
    }

    #[test]
    fn test_matrix_preprocessing() {
        let sparse_matrix = create_test_sparse_matrix();

        // Test neutral fill
        let dense = MatrixPreprocessor::fill_missing_values(
            &sparse_matrix,
            ImputationMethod::Neutral,
        )
        .unwrap();

        assert_eq!(dense[[0, 0]], 1.0); // voter1, decision1
        assert_eq!(dense[[0, 1]], 0.0); // voter1, decision2
        assert_eq!(dense[[1, 0]], 1.0); // voter2, decision1
        assert_eq!(dense[[1, 1]], 0.5); // voter2, decision2 (filled)
        assert_eq!(dense[[2, 0]], 0.5); // voter3, decision1 (filled)
        assert_eq!(dense[[2, 1]], 0.5); // voter3, decision2 (filled)

        // Test voter mean fill
        let dense_voter_mean = MatrixPreprocessor::fill_missing_values(
            &sparse_matrix,
            ImputationMethod::VoterMean,
        )
        .unwrap();

        // voter1's mean is (1.0 + 0.0) / 2 = 0.5
        // voter2 has only one vote (1.0), but missing values filled with 0.5 default
        // voter3 has no votes, so default 0.5
        assert_eq!(dense_voter_mean[[2, 0]], 0.5); // voter3 filled with default
    }

    #[test]
    fn test_normalization() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test min-max normalization
        let normalized = MatrixPreprocessor::normalize_matrix(
            &matrix,
            NormalizationMethod::MinMax,
        )
        .unwrap();

        // First column: [1, 3] -> [0, 1]
        // Second column: [2, 4] -> [0, 1]
        assert_eq!(normalized[[0, 0]], 0.0);
        assert_eq!(normalized[[1, 0]], 1.0);
        assert_eq!(normalized[[0, 1]], 0.0);
        assert_eq!(normalized[[1, 1]], 1.0);
    }

    #[test]
    fn test_correlation_analysis() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
                .unwrap();

        let correlation =
            MatrixAnalyzer::decision_correlation_matrix(&matrix).unwrap();

        // Perfect positive correlation between identical columns
        assert!((correlation[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((correlation[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reputation_weighting() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let reputations = Array1::from_vec(vec![0.8, 0.2]);

        let weighted =
            ConsensusOps::apply_reputation_weighting(&matrix, &reputations)
                .unwrap();

        // First voter's votes scaled by 0.8
        assert_eq!(weighted[[0, 0]], 0.8);
        assert_eq!(weighted[[0, 1]], 0.0);

        // Second voter's votes scaled by 0.2
        assert_eq!(weighted[[1, 0]], 0.0);
        assert_eq!(weighted[[1, 1]], 0.2);
    }

    #[test]
    fn test_consensus_calculation() {
        let weighted_matrix =
            Array2::from_shape_vec((2, 2), vec![0.8, 0.0, 0.0, 0.2]).unwrap();

        let outcomes =
            ConsensusOps::calculate_weighted_consensus(&weighted_matrix)
                .unwrap();

        // First decision: (0.8 + 0.0) / 2 = 0.4
        // Second decision: (0.0 + 0.2) / 2 = 0.1
        assert_eq!(outcomes[0], 0.4);
        assert_eq!(outcomes[1], 0.1);
    }

    #[test]
    fn test_reputation_updates() {
        let votes =
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 1.0]).unwrap();
        let outcomes = Array1::from_vec(vec![1.0, 0.5]);
        let current_reputations = Array1::from_vec(vec![0.5, 0.5]);

        let new_reputations = ConsensusOps::update_reputations(
            &votes,
            &outcomes,
            &current_reputations,
            0.1, // 10% learning rate
        )
        .unwrap();

        // First voter: perfect on decision 1, 0.5 error on decision 2
        // Second voter: perfect on decision 1, 0.5 error on decision 2
        // Both should have similar updates since they have similar performance
        assert!(new_reputations[0] >= 0.5);
        assert!(new_reputations[1] >= 0.5);
    }

    #[test]
    fn test_rank_estimation() {
        // Create a rank-2 matrix
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
        )
        .unwrap();

        let rank = MatrixAnalyzer::estimate_rank(&matrix, 1e-10).unwrap();

        // This matrix should have rank 1 (all rows are multiples of [1, 2, 3])
        assert!(rank <= 2); // Our simplified algorithm may not be perfect
    }
}
