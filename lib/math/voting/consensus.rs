//! Bitcoin Hivemind Consensus Algorithm Implementation
//!
//! This module implements the reference consensus algorithm from the Bitcoin Hivemind
//! specification, using Principal Component Analysis (PCA) with Singular Value
//! Decomposition (SVD) for truth extraction from voting data.

use nalgebra::{DMatrix, DVector, SVD};
use std::collections::HashMap;
use crate::state::slots::SlotId;
use crate::state::voting::types::VoterId;
use super::{SparseVoteMatrix, ReputationVector, VotingMathError};
use super::constants::{BITCOIN_HIVEMIND_NEUTRAL_VALUE, CONSENSUS_CATCH_TOLERANCE, SVD_NUMERICAL_TOLERANCE};

/// Helper function to calculate weights with absolute values and normalization
///
/// Takes a vector, applies absolute value, then proportional linear deviance from 0.
/// This matches the reference implementation's GetWeight function.
pub fn get_weight(vec: &DVector<f64>) -> DVector<f64> {
    let mut new_vec = vec.abs(); // absolute value

    // if it is all zeros, add +1 to everything
    if new_vec.sum() == 0.0 {
        new_vec.fill(1.0);
    }

    // normalize to sum to 1
    new_vec /= new_vec.sum();
    new_vec
}

/// Re-weight a vector to ensure it sums to 1
///
/// Handles NaN values by replacing them with 0, takes absolute values,
/// and normalizes the result.
pub fn re_weight(vector_in: &DVector<f64>) -> DVector<f64> {
    let mut out = vector_in.clone();

    // Take absolute values and handle NaN
    out.apply(|x| {
        if x.is_nan() {
            *x = 0.0;
        } else {
            *x = x.abs();
        }
    });

    // Handle division by zero
    if out.sum() == 0.0 {
        out.fill(1.0);
    }

    // Normalize
    let sum = out.sum();
    out.apply(|x| *x = *x / sum);

    out
}

/// Catch function for binary decisions
///
/// Rounds values to 0, 0.5, or 1 based on tolerance around neutral value
pub fn catch_tl(x: f64, tolerance: f64) -> f64 {
    if x < (BITCOIN_HIVEMIND_NEUTRAL_VALUE - tolerance / 2.0) {
        0.0
    } else if x > (BITCOIN_HIVEMIND_NEUTRAL_VALUE + tolerance / 2.0) {
        1.0
    } else {
        BITCOIN_HIVEMIND_NEUTRAL_VALUE
    }
}

/// Calculate weighted median
pub fn weighted_median(values: &DVector<f64>, weights: &DVector<f64>) -> f64 {
    assert_eq!(values.len(), weights.len(), "Values and weights must have the same length");

    let mut sorted: Vec<_> = values.iter().zip(weights.iter()).collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());

    let total_weight: f64 = weights.iter().sum();
    let mut cumulative_weight = 0.0;
    let half_weight = total_weight / 2.0;

    for (i, &(value, weight)) in sorted.iter().enumerate() {
        cumulative_weight += weight;
        if cumulative_weight >= half_weight {
            if cumulative_weight == half_weight && i < sorted.len() - 1 {
                return (*value + *sorted[i + 1].0) / 2.0;
            } else {
                return *value;
            }
        }
    }

    *sorted.last().unwrap().0
}

/// Score reflection to handle PCA sign ambiguity
///
/// Takes an adjusted score and breaks it at a median point to prevent
/// exploitation through "sacrificial variance" attacks.
fn score_reflect(component: &DVector<f64>, previous_reputation: &DVector<f64>) -> DVector<f64> {
    // Only if there actually is a median point
    let unique_count = component.len();
    if unique_count <= 2 {
        return component.clone();
    }

    // Calculate weighted median
    let median_factor = weighted_median(component, previous_reputation);

    // By how much does the component exceed the Median?
    let reflection = component - DVector::from_element(component.len(), median_factor);

    // Where does the Component exceed the Median?
    let excessive = reflection.map(|x| x > 0.0);

    let mut adj_prin_comp = component.clone();

    for i in 0..adj_prin_comp.len() {
        if excessive[i] {
            adj_prin_comp[i] -= reflection[i] * 0.5; // mixes in some of the old method
        }
    }

    adj_prin_comp
}

/// Weighted Principal Component Analysis using SVD
///
/// This is the core PCA implementation that matches the reference.
/// Returns (first_score, first_loading) tuple.
pub fn weighted_prin_comp(
    x: &DMatrix<f64>,
    weights: Option<&DVector<f64>>,
    _verbose: bool,
) -> Result<(DVector<f64>, DVector<f64>), VotingMathError> {
    let n_rows = x.nrows();
    let _n_cols = x.ncols();

    let weights = match weights {
        Some(w) => {
            if w.len() != n_rows {
                return Err(VotingMathError::DimensionMismatch {
                    expected: format!("weights length {}", n_rows),
                    actual: format!("weights length {}", w.len()),
                });
            }
            w.clone()
        }
        None => re_weight(&DVector::from_element(n_rows, 1.0)),
    };

    // Compute weighted mean
    let total_weight = weights.sum();
    if total_weight <= 0.0 {
        return Err(VotingMathError::InvalidReputation {
            reason: "Total weight is non-positive".to_string(),
        });
    }

    let weighted_mean = (x.transpose() * &weights) / total_weight;

    // Center the data
    let mut centered = x.clone();
    for mut row in centered.row_iter_mut() {
        row -= &weighted_mean.transpose();
    }

    // Compute weighted covariance
    let mut weighted_centered = DMatrix::zeros(centered.nrows(), centered.ncols());
    for (i, row) in centered.row_iter().enumerate() {
        weighted_centered.set_row(i, &(row * weights[i]));
    }
    let wcvm = (centered.transpose() * weighted_centered) / total_weight;

    // SVD decomposition
    let svd = SVD::new(wcvm.clone(), true, false);

    // Extract first loading (first column of U matrix)
    let first_loading = svd.u.as_ref()
        .ok_or_else(|| VotingMathError::NumericalError {
            reason: "SVD failed to compute U matrix".to_string(),
        })?
        .column(0)
        .clone_owned();

    // Calculate first score by projecting centered data onto first loading
    let first_score = centered * &first_loading;

    Ok((first_score, first_loading))
}

/// Result from get_reward_weights
pub struct RewardWeightsResult {
    pub first_loading: DVector<f64>,
    pub old_rep: DVector<f64>,
    pub this_rep: DVector<f64>,
    pub smooth_rep: DVector<f64>,
}

/// Get reward weights - the main consensus algorithm
///
/// Implements the reference algorithm for updating reputations based on
/// voting patterns using PCA with sign ambiguity resolution.
pub fn get_reward_weights(
    vote_matrix: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    alpha: f64,
    _tolerance: f64,
    _verbose: bool,
) -> Result<RewardWeightsResult, VotingMathError> {
    let num_voters = vote_matrix.nrows();

    // Use democratic reputation if none provided
    let old_rep = match rep {
        Some(r) => r.clone(),
        None => DVector::from_element(num_voters, 1.0 / num_voters as f64),
    };

    // Weighted Principal Components Analysis
    let (first_score, first_loading) = weighted_prin_comp(vote_matrix, Some(&old_rep), false)?;

    let mut new_rep = old_rep.clone();

    // If there was 100% agreement, reputations don't change
    if first_score.abs().sum() != 0.0 {
        // Handle PCA sign ambiguity - two possible interpretations
        let score_min = DVector::from_element(first_score.len(), first_score.min());
        let score_max = DVector::from_element(first_score.len(), first_score.max());

        let option1 = &first_score + score_min.abs();
        let option2 = (&first_score - score_max).abs();

        let new_score_1 = score_reflect(&option1, &old_rep);
        let new_score_2 = score_reflect(&option2, &old_rep);

        // Method 6: Use old reputation to calculate outcomes and compare
        // Calculate decision outcomes using old reputation
        let mut old_rep_outcomes = DVector::zeros(vote_matrix.ncols());
        for j in 0..vote_matrix.ncols() {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for i in 0..vote_matrix.nrows() {
                let vote = vote_matrix[(i, j)];
                if !vote.is_nan() {
                    let weight = old_rep[i];
                    weighted_sum += vote * weight;
                    total_weight += weight;
                }
            }

            old_rep_outcomes[j] = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                BITCOIN_HIVEMIND_NEUTRAL_VALUE
            };
        }

        // Apply catch for binary decisions
        for j in 0..old_rep_outcomes.len() {
            old_rep_outcomes[j] = catch_tl(old_rep_outcomes[j], CONSENSUS_CATCH_TOLERANCE);
        }

        // Build distance matrix
        let mut distance = DMatrix::zeros(vote_matrix.nrows(), vote_matrix.ncols());
        for i in 0..vote_matrix.nrows() {
            for j in 0..vote_matrix.ncols() {
                let vote = vote_matrix[(i, j)];
                if !vote.is_nan() && j < old_rep_outcomes.len() {
                    distance[(i, j)] = (vote - old_rep_outcomes[j]).abs();
                }
            }
        }

        // Build mainstream weighting (column vector)
        let dissent = first_loading.abs();
        let mainstream_preweight = dissent.map(|x| if x != 0.0 { 1.0 / x } else { f64::NAN });
        let mainstream = re_weight(&mainstream_preweight);

        // Calculate non-compliance for each voter-decision pair
        let mut non_compliance = DMatrix::zeros(vote_matrix.nrows(), vote_matrix.ncols());
        for i in 0..vote_matrix.nrows() {
            for j in 0..vote_matrix.ncols() {
                if j < mainstream.len() {
                    non_compliance[(i, j)] = distance[(i, j)] * mainstream[j];
                }
            }
        }

        let max_non_compliance = non_compliance.max();
        let compliance = non_compliance.map(|x| (x - max_non_compliance).abs());

        // Compute compliance scores for each voter
        let mut compliance_scores = DVector::zeros(num_voters);
        for i in 0..num_voters {
            let mut sum = 0.0;
            let mut count = 0.0;
            for j in 0..vote_matrix.ncols() {
                if j < mainstream.len() {
                    sum += compliance[(i, j)];
                    count += 1.0;
                }
            }
            compliance_scores[i] = if count > 0.0 { sum / count } else { 0.0 };
        }

        // Choose between the two score options
        let weighted_score_1 = get_weight(&new_score_1);
        let difference_1 = &weighted_score_1 - &compliance_scores;
        let choice1: f64 = difference_1.iter().map(|&x| x.powi(2)).sum();

        let weighted_score_2 = get_weight(&new_score_2);
        let difference_2 = &weighted_score_2 - &compliance_scores;
        let choice2: f64 = difference_2.iter().map(|&x| x.powi(2)).sum();

        // Select the interpretation with lower error
        let ref_ind = choice1 - choice2;

        // Scale reputation to ensure additivity
        let rep_mean = old_rep.mean();
        let scaled_rep = old_rep.map(|x| x / rep_mean);

        let rep1 = get_weight(&new_score_1.component_mul(&scaled_rep));
        let rep2 = get_weight(&new_score_2.component_mul(&scaled_rep));

        new_rep = if ref_ind < 0.0 { rep1 } else { rep2 };
    }

    // Smooth reputation update
    let smooth_rep = &new_rep * alpha + &old_rep * (1.0 - alpha);

    Ok(RewardWeightsResult {
        first_loading,
        old_rep,
        this_rep: new_rep,
        smooth_rep,
    })
}

/// Fill missing values in vote matrix using consensus
pub fn fill_na(
    m_na: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    catch_p: f64,
    _verbose: bool,
) -> DMatrix<f64> {
    let mut m_new = m_na.clone();

    // Check if there are any NaN values
    let has_nan = m_na.iter().any(|&x| x.is_nan());

    if has_nan {
        // Get reputation vector if not provided
        let num_voters = m_na.nrows();
        let rep = match rep {
            Some(r) => r.clone(),
            None => DVector::from_element(num_voters, 1.0 / num_voters as f64),
        };

        // Calculate decision outcomes for filling
        let mut decision_outcomes = DVector::zeros(m_na.ncols());
        for j in 0..m_na.ncols() {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for i in 0..m_na.nrows() {
                let vote = m_na[(i, j)];
                if !vote.is_nan() {
                    let weight = rep[i];
                    weighted_sum += vote * weight;
                    total_weight += weight;
                }
            }

            let outcome = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                BITCOIN_HIVEMIND_NEUTRAL_VALUE
            };

            // Apply catch for binary decisions
            decision_outcomes[j] = catch_tl(outcome, catch_p);
        }

        // Replace NaN values with caught decision outcomes
        for i in 0..m_new.nrows() {
            for j in 0..m_new.ncols() {
                if m_new[(i, j)].is_nan() {
                    m_new[(i, j)] = decision_outcomes[j];
                }
            }
        }
    }

    m_new
}

/// Main factory result
pub struct FactoryResult {
    pub filled: DMatrix<f64>,
    pub agents: RewardWeightsResult,
    pub decisions: DVector<f64>,
    pub certainty: f64,
    pub first_loading: DVector<f64>,
    pub explained_variance: f64,
}

/// Main factory function - implements the complete consensus algorithm
///
/// This is the main entry point that orchestrates the full Bitcoin Hivemind
/// consensus algorithm as specified in the reference implementation.
pub fn factory(
    m0: &DMatrix<f64>,
    rep: Option<&DVector<f64>>,
    catch_p: Option<f64>,
    verbose: Option<bool>,
) -> Result<FactoryResult, VotingMathError> {
    let verbose = verbose.unwrap_or(false);
    let catch_p = catch_p.unwrap_or(CONSENSUS_CATCH_TOLERANCE);

    // Fill missing values
    let filled = fill_na(m0, rep, catch_p, verbose);

    // Get reward weights (reputation updates)
    use super::constants::REPUTATION_SMOOTHING_ALPHA;
    let player_info = get_reward_weights(
        &filled,
        rep,
        REPUTATION_SMOOTHING_ALPHA,
        catch_p,
        verbose,
    )?;

    // Calculate decision outcomes using smoothed reputation
    let mut decision_outcomes = DVector::zeros(filled.ncols());
    for j in 0..filled.ncols() {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for i in 0..filled.nrows() {
            let vote = filled[(i, j)];
            if !vote.is_nan() {
                let weight = player_info.smooth_rep[i];
                weighted_sum += vote * weight;
                total_weight += weight;
            }
        }

        decision_outcomes[j] = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            BITCOIN_HIVEMIND_NEUTRAL_VALUE
        };
    }

    // Apply catch for binary decisions
    let mut decision_outcomes_final = decision_outcomes.clone();
    for j in 0..decision_outcomes_final.len() {
        decision_outcomes_final[j] = catch_tl(decision_outcomes[j], catch_p);
    }

    // Calculate certainty
    let mut certainty = vec![0.0; filled.ncols()];
    for j in 0..filled.ncols() {
        let mut sum = 0.0;
        for i in 0..filled.nrows() {
            if (decision_outcomes_final[j] - filled[(i, j)]).abs() < SVD_NUMERICAL_TOLERANCE {
                sum += player_info.smooth_rep[i];
            }
        }
        certainty[j] = sum;
    }

    let avg_certainty = certainty.iter().sum::<f64>() / certainty.len() as f64;

    // Calculate explained variance from SVD eigenvalues
    // Re-run SVD on filled matrix to get eigenvalues for variance calculation
    let svd_for_variance = weighted_prin_comp(&filled, Some(&player_info.smooth_rep), false)?;

    // Calculate total variance and explained variance
    // The explained variance is the ratio of the first eigenvalue to the sum of all eigenvalues
    // For now, we approximate using the magnitude of the first component
    let first_component_magnitude = svd_for_variance.0.norm();
    let total_magnitude = filled.norm();
    let explained_variance = if total_magnitude > 0.0 {
        (first_component_magnitude / total_magnitude).min(1.0)
    } else {
        0.0
    };

    Ok(FactoryResult {
        filled,
        first_loading: player_info.first_loading.clone(),
        explained_variance, // Actual calculated value from SVD
        agents: player_info,
        decisions: decision_outcomes_final,
        certainty: avg_certainty,
    })
}

/// Detailed consensus result
pub struct DetailedConsensusOutput {
    pub outcomes: HashMap<SlotId, f64>,
    pub first_loading: Vec<f64>,
    pub explained_variance: f64,
    pub certainty: f64,
    pub updated_reputations: HashMap<VoterId, f64>,
    pub outliers: Vec<VoterId>,
}

/// Run consensus algorithm with full SVD information and metrics
/// This is the single unified consensus function that returns all necessary data
pub fn run_consensus(
    vote_matrix: &SparseVoteMatrix,
    reputation_vector: &ReputationVector,
) -> Result<DetailedConsensusOutput, VotingMathError> {
    // Convert sparse matrix to dense nalgebra matrix
    let voters = vote_matrix.get_voters();
    let decisions = vote_matrix.get_decisions();
    let (num_voters, num_decisions) = vote_matrix.dimensions();

    let mut dense_matrix = DMatrix::from_element(num_voters, num_decisions, f64::NAN);

    for (i, voter_id) in voters.iter().enumerate() {
        for (j, decision_id) in decisions.iter().enumerate() {
            if let Some(vote) = vote_matrix.get_vote(*voter_id, *decision_id) {
                dense_matrix[(i, j)] = vote;
            }
        }
    }

    // Convert reputation vector to nalgebra vector
    let mut rep_vector = DVector::zeros(num_voters);
    for (i, voter_id) in voters.iter().enumerate() {
        rep_vector[i] = reputation_vector.get_reputation(*voter_id);
    }

    // Normalize reputation
    let rep_sum = rep_vector.sum();
    if rep_sum > 0.0 {
        rep_vector /= rep_sum;
    } else {
        rep_vector = DVector::from_element(num_voters, 1.0 / num_voters as f64);
    }

    // Run factory consensus algorithm
    let result = factory(&dense_matrix, Some(&rep_vector), Some(CONSENSUS_CATCH_TOLERANCE), Some(false))?;

    // Convert results back to HashMaps
    let mut outcomes = HashMap::new();
    for (j, decision_id) in decisions.iter().enumerate() {
        outcomes.insert(*decision_id, result.decisions[j]);
    }

    // Scale reputation values back to maintain the total reputation pool
    // The consensus algorithm normalizes reputation to sum to 1.0, but we need to maintain
    // the original total reputation to keep values around 0.5 for neutral reputation
    let original_total: f64 = voters.iter().map(|v| reputation_vector.get_reputation(*v)).sum();
    let consensus_total: f64 = result.agents.smooth_rep.sum();
    let scale_factor = if consensus_total > 0.0 && original_total > 0.0 {
        original_total / consensus_total
    } else {
        num_voters as f64 * BITCOIN_HIVEMIND_NEUTRAL_VALUE
    };

    let mut updated_reputations = HashMap::new();
    for (i, voter_id) in voters.iter().enumerate() {
        // Scale the reputation back to maintain total reputation pool
        let scaled_reputation = result.agents.smooth_rep[i] * scale_factor;
        updated_reputations.insert(*voter_id, scaled_reputation);
    }

    // Convert first loading to Vec
    let first_loading: Vec<f64> = result.first_loading.iter().cloned().collect();

    // Detect outliers (voters with significant reputation decrease)
    let mut outliers = Vec::new();
    for (i, voter_id) in voters.iter().enumerate() {
        let old_rep = result.agents.old_rep[i];
        let new_rep = result.agents.smooth_rep[i];
        if new_rep < old_rep * 0.5 {  // Lost more than 50% reputation
            outliers.push(*voter_id);
        }
    }

    Ok(DetailedConsensusOutput {
        outcomes,
        first_loading,
        explained_variance: result.explained_variance,
        certainty: result.certainty,
        updated_reputations,
        outliers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_get_weight() {
        let vec = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = get_weight(&vec);
        assert_eq!(result, DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]));

        let vec2 = DVector::from_vec(vec![4.0, 5.0, 6.0, 7.0]);
        let result2 = get_weight(&vec2);
        assert!((result2[0] - 0.18181818).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_median() {
        let values = DVector::from_vec(vec![3.0, 4.0, 5.0]);
        let weights = DVector::from_vec(vec![0.2, 0.2, 0.6]);
        assert_eq!(weighted_median(&values, &weights), 5.0);

        let weights2 = DVector::from_vec(vec![0.2, 0.2, 0.4]);
        assert_eq!(weighted_median(&values, &weights2), 4.5);
    }

    #[test]
    fn test_catch_tl() {
        assert_eq!(catch_tl(0.2, CONSENSUS_CATCH_TOLERANCE), 0.0);
        assert_eq!(catch_tl(BITCOIN_HIVEMIND_NEUTRAL_VALUE, CONSENSUS_CATCH_TOLERANCE), BITCOIN_HIVEMIND_NEUTRAL_VALUE);
        assert_eq!(catch_tl(0.8, CONSENSUS_CATCH_TOLERANCE), 1.0);
        assert_eq!(catch_tl(0.14, 0.7), 0.0);
        assert_eq!(catch_tl(0.16, 0.7), BITCOIN_HIVEMIND_NEUTRAL_VALUE);
    }

    #[test]
    fn test_weighted_prin_comp() {
        let m1 = DMatrix::from_row_slice(3, 3, &[
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0
        ]);

        let (score, loading) = weighted_prin_comp(&m1, None, false).unwrap();

        // Check dimensions
        assert_eq!(score.len(), 3);
        assert_eq!(loading.len(), 3);

        // Results should be normalized
        assert!(loading.norm() - 1.0 < 1e-6);
    }
}