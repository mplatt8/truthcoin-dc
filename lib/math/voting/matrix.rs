use super::constants::{
    BITCOIN_HIVEMIND_NEUTRAL_VALUE, SVD_NUMERICAL_TOLERANCE,
};
use super::{SparseVoteMatrix, VotingMathError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct MatrixPreprocessor;

impl MatrixPreprocessor {
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
                Ok(matrix.to_dense(BITCOIN_HIVEMIND_NEUTRAL_VALUE))
            }
            ImputationMethod::VoterMean => Self::fill_with_voter_means(matrix),
            ImputationMethod::DecisionMean => {
                Self::fill_with_decision_means(matrix)
            }
            ImputationMethod::GlobalMean => Self::fill_with_global_mean(matrix),
        }
    }

    fn fill_with_voter_means(
        matrix: &SparseVoteMatrix,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = matrix.dimensions();
        let mut dense = Array2::from_elem(
            (num_voters, num_decisions),
            BITCOIN_HIVEMIND_NEUTRAL_VALUE,
        );

        let voters = matrix.get_voters();
        let mut voter_means = HashMap::new();

        for voter_id in &voters {
            let votes = matrix.get_voter_votes(*voter_id);
            if !votes.is_empty() {
                let mean: f64 =
                    votes.values().sum::<f64>() / votes.len() as f64;
                voter_means.insert(*voter_id, mean);
            } else {
                voter_means.insert(*voter_id, BITCOIN_HIVEMIND_NEUTRAL_VALUE);
            }
        }

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

    fn fill_with_decision_means(
        matrix: &SparseVoteMatrix,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, num_decisions) = matrix.dimensions();
        let mut dense = Array2::from_elem(
            (num_voters, num_decisions),
            BITCOIN_HIVEMIND_NEUTRAL_VALUE,
        );

        let decisions = matrix.get_decisions();
        let mut decision_means = HashMap::new();

        for decision_id in &decisions {
            let votes = matrix.get_decision_votes(*decision_id);
            if !votes.is_empty() {
                let mean: f64 =
                    votes.values().sum::<f64>() / votes.len() as f64;
                decision_means.insert(*decision_id, mean);
            } else {
                decision_means
                    .insert(*decision_id, BITCOIN_HIVEMIND_NEUTRAL_VALUE);
            }
        }

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

    fn fill_with_global_mean(
        matrix: &SparseVoteMatrix,
    ) -> Result<Array2<f64>, VotingMathError> {
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
            BITCOIN_HIVEMIND_NEUTRAL_VALUE
        };

        Ok(matrix.to_dense(global_mean))
    }

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

    fn standardize_matrix(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let mut normalized = matrix.clone();

        for mut column in normalized.columns_mut() {
            let mean = column.mean().unwrap_or(0.0);
            let std_dev = {
                let variance =
                    column.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / column.len() as f64;
                variance.sqrt()
            };

            if std_dev > SVD_NUMERICAL_TOLERANCE {
                column.mapv_inplace(|x| (x - mean) / std_dev);
            }
        }

        Ok(normalized)
    }

    fn min_max_normalize(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let mut normalized = matrix.clone();

        for mut column in normalized.columns_mut() {
            let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val =
                column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let range = max_val - min_val;
            if range > SVD_NUMERICAL_TOLERANCE {
                column.mapv_inplace(|x| (x - min_val) / range);
            }
        }

        Ok(normalized)
    }

    fn unit_vector_normalize(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let mut normalized = matrix.clone();

        for mut row in normalized.rows_mut() {
            let norm = row.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            if norm > SVD_NUMERICAL_TOLERANCE {
                row.mapv_inplace(|x| x / norm);
            }
        }

        Ok(normalized)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImputationMethod {
    Neutral,
    VoterMean,
    DecisionMean,
    GlobalMean,
}

#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    None,
    StandardScore,
    MinMax,
    UnitVector,
}

pub struct MatrixAnalyzer;

impl MatrixAnalyzer {
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
        if denominator < SVD_NUMERICAL_TOLERANCE {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

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

        correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        correlations
    }

    /// Estimate matrix rank using Gaussian elimination.
    /// Indicates dimensionality of truth space for PCA.
    pub fn estimate_rank(
        matrix: &Array2<f64>,
        tolerance: f64,
    ) -> Result<usize, VotingMathError> {
        let (rows, cols) = matrix.dim();
        let min_dim = rows.min(cols);

        let mut rank = 0;
        let mut working = matrix.clone();

        for i in 0..min_dim {
            let mut max_row = i;
            for k in (i + 1)..rows {
                if working[[k, i]].abs() > working[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            if working[[max_row, i]].abs() < tolerance {
                continue;
            }

            rank += 1;

            if max_row != i && i < rows {
                for j in 0..cols {
                    let temp = working[[i, j]];
                    working[[i, j]] = working[[max_row, j]];
                    working[[max_row, j]] = temp;
                }
            }

            for k in (i + 1)..rows {
                if working[[i, i]].abs() > SVD_NUMERICAL_TOLERANCE {
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

pub struct ConsensusOps;

impl ConsensusOps {
    pub fn apply_reputation_weighting(
        matrix: &Array2<f64>,
        reputations: &Array1<f64>,
    ) -> Result<Array2<f64>, VotingMathError> {
        let (num_voters, _num_decisions) = matrix.dim();

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
            let weight_sum: f64 = column.len() as f64;

            outcomes[j] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.5
            };
        }

        Ok(outcomes)
    }

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

                if !vote.is_nan() {
                    let error = (vote - outcome).abs();
                    total_error += error;
                    vote_count += 1;
                }
            }

            if vote_count > 0 {
                let avg_error = total_error / vote_count as f64;
                let accuracy = 1.0 - avg_error.min(1.0);

                let current_rep = current_reputations[i];
                new_reputations[i] =
                    current_rep + learning_rate * (accuracy - current_rep);

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
    use crate::types::Address;

    fn create_test_sparse_matrix() -> SparseVoteMatrix {
        let voters =
            vec![Address([1u8; 20]), Address([2u8; 20]), Address([3u8; 20])];

        let decisions =
            vec![SlotId::new(1, 0).unwrap(), SlotId::new(1, 1).unwrap()];

        let mut matrix = SparseVoteMatrix::new(voters, decisions);

        // Add some test votes
        let voter1 = Address([1u8; 20]);
        let voter2 = Address([2u8; 20]);
        let decision1 = SlotId::new(1, 0).unwrap();
        let decision2 = SlotId::new(1, 1).unwrap();

        matrix.set_vote(voter1, decision1, 1.0).unwrap();
        matrix.set_vote(voter1, decision2, 0.0).unwrap();
        matrix.set_vote(voter2, decision1, 1.0).unwrap();

        matrix
    }

    #[test]
    fn test_matrix_preprocessing() {
        let sparse_matrix = create_test_sparse_matrix();

        let dense = MatrixPreprocessor::fill_missing_values(
            &sparse_matrix,
            ImputationMethod::Neutral,
        )
        .unwrap();

        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 1]], 0.0);
        assert_eq!(dense[[1, 0]], 1.0);
        assert_eq!(dense[[1, 1]], 0.5);
        assert_eq!(dense[[2, 0]], 0.5);
        assert_eq!(dense[[2, 1]], 0.5);

        let dense_voter_mean = MatrixPreprocessor::fill_missing_values(
            &sparse_matrix,
            ImputationMethod::VoterMean,
        )
        .unwrap();

        assert_eq!(dense_voter_mean[[2, 0]], 0.5);
    }

    #[test]
    fn test_normalization() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let normalized = MatrixPreprocessor::normalize_matrix(
            &matrix,
            NormalizationMethod::MinMax,
        )
        .unwrap();

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

        assert_eq!(weighted[[0, 0]], 0.8);
        assert_eq!(weighted[[0, 1]], 0.0);

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
            0.1,
        )
        .unwrap();

        assert!(new_reputations[0] >= 0.5);
        assert!(new_reputations[1] >= 0.5);
    }

    #[test]
    fn test_rank_estimation() {
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
        )
        .unwrap();

        let rank = MatrixAnalyzer::estimate_rank(&matrix, 1e-10).unwrap();

        assert!(rank <= 2);
    }
}
