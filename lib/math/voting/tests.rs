//! Comprehensive tests for Bitcoin Hivemind voting mathematics
//!
//! This module tests the mathematical foundations of the voting system,
//! including sparse matrix operations, reputation calculations, and
//! vote aggregation algorithms.

use super::*;
use crate::state::slots::SlotId;
use crate::state::voting::types::VoterId;
use crate::types::Address;
use approx::assert_relative_eq;
use std::collections::HashMap;

// Helper functions for creating test data
fn create_test_voter_ids(count: usize) -> Vec<VoterId> {
    (0..count)
        .map(|i| {
            let mut addr_bytes = [0u8; 20];
            addr_bytes[0] = i as u8;
            VoterId::from_address(&Address(addr_bytes))
        })
        .collect()
}

fn create_test_decision_ids(count: usize) -> Vec<SlotId> {
    (0..count)
        .map(|i| SlotId::new(1, i as u32).unwrap())
        .collect()
}

/// Test sparse vote matrix creation and basic operations
#[test]
fn test_sparse_vote_matrix_creation() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(4);

    let matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Test dimensions
    assert_eq!(matrix.dimensions(), (3, 4));

    // Test initial state
    assert_eq!(matrix.num_votes(), 0);
    assert_eq!(matrix.density(), 0.0);

    // Test voter and decision retrieval
    let retrieved_voters = matrix.get_voters();
    let retrieved_decisions = matrix.get_decisions();
    assert_eq!(retrieved_voters.len(), 3);
    assert_eq!(retrieved_decisions.len(), 4);

    // Verify all voters and decisions are present (order may differ)
    for voter in &voters {
        assert!(retrieved_voters.contains(voter));
    }
    for decision in &decisions {
        assert!(retrieved_decisions.contains(decision));
    }
}

/// Test vote setting and retrieval in sparse matrix
#[test]
fn test_sparse_vote_matrix_vote_operations() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(2);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Test setting votes
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[2], decisions[1], 0.75).unwrap();

    // Test retrieval
    assert_eq!(matrix.get_vote(voters[0], decisions[0]), Some(1.0));
    assert_eq!(matrix.get_vote(voters[1], decisions[0]), Some(0.0));
    assert_eq!(matrix.get_vote(voters[2], decisions[1]), Some(0.75));

    // Test non-existent votes
    assert_eq!(matrix.get_vote(voters[0], decisions[1]), None);
    assert_eq!(matrix.get_vote(voters[1], decisions[1]), None);
    assert_eq!(matrix.get_vote(voters[2], decisions[0]), None);

    // Test matrix statistics
    assert_eq!(matrix.num_votes(), 3);
    assert_eq!(matrix.density(), 3.0 / 6.0); // 3 votes out of 6 possible positions
}

/// Test vote matrix error conditions
#[test]
fn test_sparse_vote_matrix_errors() {
    let voters = create_test_voter_ids(2);
    let decisions = create_test_decision_ids(2);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Test invalid voter
    let invalid_voter = VoterId::from_address(&Address([99u8; 20]));
    let result = matrix.set_vote(invalid_voter, decisions[0], 1.0);
    assert!(result.is_err());
    if let Err(VotingMathError::InvalidReputation { reason }) = result {
        assert!(reason.contains("Voter"));
        assert!(reason.contains("not found"));
    } else {
        panic!("Expected InvalidReputation error for voter");
    }

    // Test invalid decision
    let invalid_decision = SlotId::new(2, 0).unwrap();
    let result = matrix.set_vote(voters[0], invalid_decision, 1.0);
    assert!(result.is_err());
    if let Err(VotingMathError::InvalidReputation { reason }) = result {
        assert!(reason.contains("Decision"));
        assert!(reason.contains("not found"));
    } else {
        panic!("Expected InvalidReputation error for decision");
    }
}

/// Test dense matrix conversion
#[test]
fn test_sparse_to_dense_conversion() {
    let voters = create_test_voter_ids(2);
    let decisions = create_test_decision_ids(3);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Set some votes
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[2], 0.25).unwrap();

    // Convert to dense with fill value
    let dense = matrix.to_dense(0.5);

    // Check dimensions
    assert_eq!(dense.dim(), (2, 3));

    // Check values (note: we don't know the exact indices without looking at internal maps)
    // Instead, we verify that our set values appear somewhere and fill values appear elsewhere
    let flat: Vec<f64> = dense.iter().copied().collect();
    assert!(flat.contains(&1.0));
    assert!(flat.contains(&0.25));
    assert_eq!(flat.iter().filter(|&&x| x == 0.5).count(), 4); // 4 positions should have fill value
}

/// Test getting votes for specific voters and decisions
#[test]
fn test_sparse_matrix_queries() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(3);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Create a pattern of votes
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[0], decisions[1], 0.8).unwrap();
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[2], decisions[2], 0.6).unwrap();

    // Test getting voter votes
    let voter0_votes = matrix.get_voter_votes(voters[0]);
    assert_eq!(voter0_votes.len(), 2);
    assert_eq!(voter0_votes.get(&decisions[0]), Some(&1.0));
    assert_eq!(voter0_votes.get(&decisions[1]), Some(&0.8));

    let voter1_votes = matrix.get_voter_votes(voters[1]);
    assert_eq!(voter1_votes.len(), 1);
    assert_eq!(voter1_votes.get(&decisions[0]), Some(&0.0));

    let voter2_votes = matrix.get_voter_votes(voters[2]);
    assert_eq!(voter2_votes.len(), 1);
    assert_eq!(voter2_votes.get(&decisions[2]), Some(&0.6));

    // Test getting decision votes
    let decision0_votes = matrix.get_decision_votes(decisions[0]);
    assert_eq!(decision0_votes.len(), 2);
    assert_eq!(decision0_votes.get(&voters[0]), Some(&1.0));
    assert_eq!(decision0_votes.get(&voters[1]), Some(&0.0));

    let decision1_votes = matrix.get_decision_votes(decisions[1]);
    assert_eq!(decision1_votes.len(), 1);
    assert_eq!(decision1_votes.get(&voters[0]), Some(&0.8));

    let decision2_votes = matrix.get_decision_votes(decisions[2]);
    assert_eq!(decision2_votes.len(), 1);
    assert_eq!(decision2_votes.get(&voters[2]), Some(&0.6));
}

/// Test reputation vector operations
#[test]
fn test_reputation_vector_basic_operations() {
    let mut reputation = ReputationVector::new();
    let voters = create_test_voter_ids(3);

    // Test initial state
    assert!(reputation.is_empty());
    assert_eq!(reputation.len(), 0);

    // Test setting reputations
    reputation.set_reputation(voters[0], 0.8);
    reputation.set_reputation(voters[1], 0.6);
    reputation.set_reputation(voters[2], 0.4);

    assert!(!reputation.is_empty());
    assert_eq!(reputation.len(), 3);

    // Test getting reputations
    assert_eq!(reputation.get_reputation(voters[0]), 0.8);
    assert_eq!(reputation.get_reputation(voters[1]), 0.6);
    assert_eq!(reputation.get_reputation(voters[2]), 0.4);

    // Test default reputation for unknown voter
    let unknown_voter = VoterId::from_address(&Address([99u8; 20]));
    assert_eq!(reputation.get_reputation(unknown_voter), 0.5);

    // Test total weight
    assert_eq!(reputation.total_weight(), 1.8);
}

/// Test reputation vector boundary conditions and clamping
#[test]
fn test_reputation_vector_boundaries() {
    let mut reputation = ReputationVector::new();
    let voter = create_test_voter_ids(1)[0];

    // Test values outside [0, 1] range get clamped
    reputation.set_reputation(voter, -0.5);
    assert_eq!(reputation.get_reputation(voter), 0.0);

    reputation.set_reputation(voter, 1.5);
    assert_eq!(reputation.get_reputation(voter), 1.0);

    reputation.set_reputation(voter, 0.0);
    assert_eq!(reputation.get_reputation(voter), 0.0);

    reputation.set_reputation(voter, 1.0);
    assert_eq!(reputation.get_reputation(voter), 1.0);
}

/// Test reputation vector normalization
#[test]
fn test_reputation_vector_normalization() {
    let mut reputation = ReputationVector::new();
    let voters = create_test_voter_ids(3);

    // Set non-normalized reputations
    reputation.set_reputation(voters[0], 0.4);
    reputation.set_reputation(voters[1], 0.6);
    reputation.set_reputation(voters[2], 1.0);

    // Total should be 2.0
    assert_eq!(reputation.total_weight(), 2.0);

    // Normalize
    reputation.normalize();

    // Total should now be 1.0
    assert_relative_eq!(reputation.total_weight(), 1.0, epsilon = 1e-10);

    // Individual values should be proportionally scaled
    assert_relative_eq!(
        reputation.get_reputation(voters[0]),
        0.2,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        reputation.get_reputation(voters[1]),
        0.3,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        reputation.get_reputation(voters[2]),
        0.5,
        epsilon = 1e-10
    );
}

/// Test reputation vector to array conversion
#[test]
fn test_reputation_vector_to_array() {
    let mut reputation = ReputationVector::new();
    let voters = create_test_voter_ids(4);

    // Set some reputations (not all voters)
    reputation.set_reputation(voters[0], 0.8);
    reputation.set_reputation(voters[2], 0.6);

    // Convert to array in specific order
    let ordered_voters = vec![voters[1], voters[0], voters[3], voters[2]];
    let array = reputation.to_array(&ordered_voters);

    // Should get default value for voters without reputation
    assert_eq!(array[0], 0.5); // voters[1] - default
    assert_eq!(array[1], 0.8); // voters[0] - set value
    assert_eq!(array[2], 0.5); // voters[3] - default
    assert_eq!(array[3], 0.6); // voters[2] - set value
}

/// Test simple majority vote aggregation
#[test]
fn test_simple_majority_aggregation() {
    let voters = create_test_voter_ids(5);
    let mut votes = HashMap::new();

    // Test clear majority (3 yes, 2 no)
    votes.insert(voters[0], 1.0);
    votes.insert(voters[1], 1.0);
    votes.insert(voters[2], 1.0);
    votes.insert(voters[3], 0.0);
    votes.insert(voters[4], 0.0);

    let result = VoteAggregator::simple_majority(&votes);
    assert_eq!(result, 1.0);

    // Test clear minority (2 yes, 3 no)
    votes.clear();
    votes.insert(voters[0], 1.0);
    votes.insert(voters[1], 1.0);
    votes.insert(voters[2], 0.0);
    votes.insert(voters[3], 0.0);
    votes.insert(voters[4], 0.0);

    let result = VoteAggregator::simple_majority(&votes);
    assert_eq!(result, 0.0);

    // Test tie (2 yes, 2 no) - should round to 0 since average is 0.5
    votes.clear();
    votes.insert(voters[0], 1.0);
    votes.insert(voters[1], 1.0);
    votes.insert(voters[2], 0.0);
    votes.insert(voters[3], 0.0);

    let result = VoteAggregator::simple_majority(&votes);
    assert_eq!(result, 1.0); // 0.5 rounds up to 1.0

    // Test empty votes
    votes.clear();
    let result = VoteAggregator::simple_majority(&votes);
    assert_eq!(result, 0.5);
}

/// Test weighted average vote aggregation
#[test]
fn test_weighted_average_aggregation() {
    let voters = create_test_voter_ids(3);
    let mut votes = HashMap::new();
    let mut reputation = ReputationVector::new();

    // Set up votes and reputations
    votes.insert(voters[0], 1.0);
    votes.insert(voters[1], 0.0);
    votes.insert(voters[2], 1.0);

    reputation.set_reputation(voters[0], 0.8); // High reputation votes yes
    reputation.set_reputation(voters[1], 0.2); // Low reputation votes no
    reputation.set_reputation(voters[2], 0.4); // Medium reputation votes yes

    let result = VoteAggregator::weighted_average(&votes, &reputation);

    // Expected: (1.0 * 0.8 + 0.0 * 0.2 + 1.0 * 0.4) / (0.8 + 0.2 + 0.4) = 1.2 / 1.4 ≈ 0.857
    assert_relative_eq!(result, 1.2 / 1.4, epsilon = 1e-10);

    // Test with empty votes
    votes.clear();
    let result = VoteAggregator::weighted_average(&votes, &reputation);
    assert_eq!(result, 0.5);

    // Test with zero total weight
    votes.insert(voters[0], 1.0);
    let empty_reputation = ReputationVector::new();
    let result = VoteAggregator::weighted_average(&votes, &empty_reputation);
    assert_eq!(result, 0.5); // Should default to 0.5 when total weight is 0
}

/// Test median vote calculation
#[test]
fn test_median_vote_calculation() {
    let voters = create_test_voter_ids(5);
    let mut votes = HashMap::new();

    // Test odd number of votes
    votes.insert(voters[0], 0.1);
    votes.insert(voters[1], 0.4);
    votes.insert(voters[2], 0.7);
    votes.insert(voters[3], 0.8);
    votes.insert(voters[4], 0.9);

    let result = VoteAggregator::median_vote(&votes);
    assert_eq!(result, 0.7); // Middle value

    // Test even number of votes
    votes.remove(&voters[4]);
    let result = VoteAggregator::median_vote(&votes);
    assert_eq!(result, (0.4 + 0.7) / 2.0); // Average of two middle values

    // Test single vote
    votes.clear();
    votes.insert(voters[0], 0.6);
    let result = VoteAggregator::median_vote(&votes);
    assert_eq!(result, 0.6);

    // Test empty votes
    votes.clear();
    let result = VoteAggregator::median_vote(&votes);
    assert_eq!(result, 0.5);
}

/// Test confidence calculation
#[test]
fn test_confidence_calculation() {
    let voters = create_test_voter_ids(4);
    let mut votes = HashMap::new();

    // Test unanimous agreement with binary outcome 1.0
    votes.insert(voters[0], 1.0);
    votes.insert(voters[1], 0.8);
    votes.insert(voters[2], 0.9);
    votes.insert(voters[3], 0.7);

    let confidence = VoteAggregator::calculate_confidence(&votes, 1.0);
    assert_eq!(confidence, 1.0); // All votes agree with outcome >= 0.5

    // Test unanimous disagreement with binary outcome 0.0
    let confidence = VoteAggregator::calculate_confidence(&votes, 0.0);
    assert_eq!(confidence, 0.0); // No votes agree with outcome < 0.5

    // Test mixed agreement
    votes.clear();
    votes.insert(voters[0], 1.0); // Agrees with outcome 1.0
    votes.insert(voters[1], 0.8); // Agrees with outcome 1.0
    votes.insert(voters[2], 0.2); // Disagrees with outcome 1.0
    votes.insert(voters[3], 0.1); // Disagrees with outcome 1.0

    let confidence = VoteAggregator::calculate_confidence(&votes, 1.0);
    assert_eq!(confidence, 0.5); // 2 out of 4 agree

    // Test with outcome 0.0
    let confidence = VoteAggregator::calculate_confidence(&votes, 0.0);
    assert_eq!(confidence, 0.5); // 2 out of 4 agree (the low votes)

    // Test empty votes
    votes.clear();
    let confidence = VoteAggregator::calculate_confidence(&votes, 0.8);
    assert_eq!(confidence, 0.0);
}

/// Test matrix utility functions
#[test]
fn test_matrix_utils_participation_rates() {
    let voters = create_test_voter_ids(4);
    let decisions = create_test_decision_ids(3);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Create voting pattern:
    // Decision 0: 3 voters participate
    // Decision 1: 2 voters participate
    // Decision 2: 1 voter participates
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 0.0).unwrap();
    matrix.set_vote(voters[2], decisions[0], 1.0).unwrap();

    matrix.set_vote(voters[0], decisions[1], 0.5).unwrap();
    matrix.set_vote(voters[3], decisions[1], 0.8).unwrap();

    matrix.set_vote(voters[1], decisions[2], 0.3).unwrap();

    let participation_rates =
        MatrixUtils::calculate_participation_rates(&matrix);

    assert_eq!(participation_rates.len(), 3);
    assert_eq!(participation_rates.get(&decisions[0]), Some(&0.75)); // 3/4
    assert_eq!(participation_rates.get(&decisions[1]), Some(&0.5)); // 2/4
    assert_eq!(participation_rates.get(&decisions[2]), Some(&0.25)); // 1/4
}

/// Test matrix utility voter activity calculation
#[test]
fn test_matrix_utils_voter_activity() {
    let voters = create_test_voter_ids(3);
    let decisions = create_test_decision_ids(4);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Create voting pattern:
    // Voter 0: votes on all 4 decisions
    // Voter 1: votes on 2 decisions
    // Voter 2: votes on 1 decision
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[0], decisions[1], 0.0).unwrap();
    matrix.set_vote(voters[0], decisions[2], 0.5).unwrap();
    matrix.set_vote(voters[0], decisions[3], 0.8).unwrap();

    matrix.set_vote(voters[1], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[2], 0.2).unwrap();

    matrix.set_vote(voters[2], decisions[1], 0.9).unwrap();

    let activity_rates = MatrixUtils::calculate_voter_activity(&matrix);

    assert_eq!(activity_rates.len(), 3);
    assert_eq!(activity_rates.get(&voters[0]), Some(&1.0)); // 4/4
    assert_eq!(activity_rates.get(&voters[1]), Some(&0.5)); // 2/4
    assert_eq!(activity_rates.get(&voters[2]), Some(&0.25)); // 1/4
}

/// Test outlier voter detection
#[test]
fn test_outlier_voter_detection() {
    let voters = create_test_voter_ids(4);
    let decisions = create_test_decision_ids(3);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Create pattern where voters[3] is an outlier
    // Most voters vote 1.0, outlier votes 0.0
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[2], decisions[0], 1.0).unwrap();
    matrix.set_vote(voters[3], decisions[0], 0.0).unwrap(); // Outlier

    matrix.set_vote(voters[0], decisions[1], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[1], 1.0).unwrap();
    matrix.set_vote(voters[2], decisions[1], 1.0).unwrap();
    matrix.set_vote(voters[3], decisions[1], 0.0).unwrap(); // Outlier

    matrix.set_vote(voters[0], decisions[2], 1.0).unwrap();
    matrix.set_vote(voters[1], decisions[2], 1.0).unwrap();
    matrix.set_vote(voters[2], decisions[2], 1.0).unwrap();
    matrix.set_vote(voters[3], decisions[2], 0.0).unwrap(); // Outlier

    let outliers = MatrixUtils::find_outlier_voters(&matrix, 1.0).unwrap();

    // voters[3] should be detected as an outlier
    assert_eq!(outliers.len(), 1);
    assert!(outliers.contains(&voters[3]));
}

/// Test outlier detection with insufficient data
#[test]
fn test_outlier_detection_edge_cases() {
    let voters = create_test_voter_ids(2);
    let decisions = create_test_decision_ids(1);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Only one decision with one vote - should not detect outliers
    matrix.set_vote(voters[0], decisions[0], 1.0).unwrap();

    let outliers = MatrixUtils::find_outlier_voters(&matrix, 1.0).unwrap();
    assert!(outliers.is_empty());

    // Empty matrix
    let empty_matrix = SparseVoteMatrix::new(Vec::new(), Vec::new());
    let outliers =
        MatrixUtils::find_outlier_voters(&empty_matrix, 1.0).unwrap();
    assert!(outliers.is_empty());
}

/// Test large sparse matrix operations for performance validation
#[test]
fn test_large_sparse_matrix_operations() {
    let num_voters = 100;
    let num_decisions = 50;

    let voters = create_test_voter_ids(num_voters);
    let decisions = create_test_decision_ids(num_decisions);
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Fill matrix with sparse pattern (20% density)
    let mut vote_count = 0;
    for (i, voter) in voters.iter().enumerate() {
        for (j, decision) in decisions.iter().enumerate() {
            if (i + j) % 5 == 0 {
                // 20% density
                let vote_value = if (i + j) % 2 == 0 { 1.0 } else { 0.0 };
                matrix.set_vote(*voter, *decision, vote_value).unwrap();
                vote_count += 1;
            }
        }
    }

    // Verify matrix properties
    assert_eq!(matrix.dimensions(), (num_voters, num_decisions));
    assert_eq!(matrix.num_votes(), vote_count);
    assert_relative_eq!(matrix.density(), 0.2, epsilon = 0.01);

    // Test dense conversion performance
    let dense = matrix.to_dense(0.5);
    assert_eq!(dense.dim(), (num_voters, num_decisions));

    // Test participation rate calculation
    let participation_rates =
        MatrixUtils::calculate_participation_rates(&matrix);
    assert_eq!(participation_rates.len(), num_decisions);

    // Test activity calculation
    let activity_rates = MatrixUtils::calculate_voter_activity(&matrix);
    assert_eq!(activity_rates.len(), num_voters);
}

/// Test reputation vector with large datasets
#[test]
fn test_large_reputation_vector() {
    let num_voters = 1000;
    let voters = create_test_voter_ids(num_voters);
    let mut reputation = ReputationVector::new();

    // Set reputations for all voters
    for (i, voter) in voters.iter().enumerate() {
        let rep_value = (i as f64) / (num_voters as f64); // 0.0 to ~1.0
        reputation.set_reputation(*voter, rep_value);
    }

    assert_eq!(reputation.len(), num_voters);

    // Test normalization with large dataset
    let original_total = reputation.total_weight();
    reputation.normalize();
    assert_relative_eq!(reputation.total_weight(), 1.0, epsilon = 1e-10);

    // Test array conversion
    let array = reputation.to_array(&voters);
    assert_eq!(array.len(), num_voters);

    // Verify sum is approximately 1.0
    let sum: f64 = array.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
}

/// Test mathematical edge cases and numerical stability
#[test]
fn test_numerical_stability() {
    let voters = create_test_voter_ids(3);
    let mut reputation = ReputationVector::new();

    // Test very small reputation values
    reputation.set_reputation(voters[0], 1e-10);
    reputation.set_reputation(voters[1], 1e-10);
    reputation.set_reputation(voters[2], 1e-10);

    let total_weight = reputation.total_weight();
    assert!(total_weight > 0.0);

    // Test normalization with very small values
    reputation.normalize();
    assert_relative_eq!(reputation.total_weight(), 1.0, epsilon = 1e-9);

    // Each should be approximately 1/3
    for voter in &voters {
        let rep = reputation.get_reputation(*voter);
        assert_relative_eq!(rep, 1.0 / 3.0, epsilon = 1e-9);
    }
}

/// Test vote aggregation with extreme values
#[test]
fn test_aggregation_extreme_values() {
    let voters = create_test_voter_ids(4);
    let mut votes = HashMap::new();
    let mut reputation = ReputationVector::new();

    // Test with extreme reputation weights
    reputation.set_reputation(voters[0], 1e-10); // Extremely low
    reputation.set_reputation(voters[1], 0.999999); // Extremely high
    reputation.set_reputation(voters[2], 0.5);
    reputation.set_reputation(voters[3], 0.0); // Zero weight

    votes.insert(voters[0], 1.0);
    votes.insert(voters[1], 0.0);
    votes.insert(voters[2], 1.0);
    votes.insert(voters[3], 0.0);

    let result = VoteAggregator::weighted_average(&votes, &reputation);

    // Result should be heavily influenced by voters[1] due to extreme weight
    assert!(result < 0.5); // Should be closer to 0.0 than 1.0

    // Test with all zero weights
    let mut zero_reputation = ReputationVector::new();
    for voter in &voters {
        zero_reputation.set_reputation(*voter, 0.0);
    }

    let result = VoteAggregator::weighted_average(&votes, &zero_reputation);
    assert_eq!(result, 0.5); // Should default to 0.5
}

/// Integration test combining all mathematical components
#[test]
fn test_mathematical_integration() {
    let num_voters = 10;
    let num_decisions = 5;

    let voters = create_test_voter_ids(num_voters);
    let decisions = create_test_decision_ids(num_decisions);

    // Create vote matrix
    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

    // Create reputation vector
    let mut reputation = ReputationVector::new();

    // Populate with realistic data
    for (i, voter) in voters.iter().enumerate() {
        // Set varied reputations
        let rep = 0.3 + (i as f64 * 0.07); // Range from 0.3 to ~0.9
        reputation.set_reputation(*voter, rep);

        // Each voter votes on random subset of decisions
        for (j, decision) in decisions.iter().enumerate() {
            if (i + j) % 3 != 0 {
                // ~67% participation
                let vote_value = if (i * j) % 2 == 0 { 1.0 } else { 0.0 };
                matrix.set_vote(*voter, *decision, vote_value).unwrap();
            }
        }
    }

    // Test integrated analysis
    for decision in &decisions {
        let decision_votes = matrix.get_decision_votes(*decision);

        if !decision_votes.is_empty() {
            // Calculate different aggregation methods
            let simple_majority =
                VoteAggregator::simple_majority(&decision_votes);
            let weighted_avg =
                VoteAggregator::weighted_average(&decision_votes, &reputation);
            let median = VoteAggregator::median_vote(&decision_votes);

            // All should be valid values
            assert!(simple_majority >= 0.0 && simple_majority <= 1.0);
            assert!(weighted_avg >= 0.0 && weighted_avg <= 1.0);
            assert!(median >= 0.0 && median <= 1.0);

            // Calculate confidence for each method
            let confidence_simple = VoteAggregator::calculate_confidence(
                &decision_votes,
                simple_majority,
            );
            let confidence_weighted = VoteAggregator::calculate_confidence(
                &decision_votes,
                weighted_avg,
            );
            let confidence_median =
                VoteAggregator::calculate_confidence(&decision_votes, median);

            assert!(confidence_simple >= 0.0 && confidence_simple <= 1.0);
            assert!(confidence_weighted >= 0.0 && confidence_weighted <= 1.0);
            assert!(confidence_median >= 0.0 && confidence_median <= 1.0);
        }
    }

    // Test matrix-level analytics
    let participation_rates =
        MatrixUtils::calculate_participation_rates(&matrix);
    let activity_rates = MatrixUtils::calculate_voter_activity(&matrix);

    // All rates should be in valid range
    for rate in participation_rates.values() {
        assert!(*rate >= 0.0 && *rate <= 1.0);
    }
    for rate in activity_rates.values() {
        assert!(*rate >= 0.0 && *rate <= 1.0);
    }

    // Test outlier detection
    let outliers = MatrixUtils::find_outlier_voters(&matrix, 2.0).unwrap();
    // With threshold of 2.0, should find few or no outliers in random data
    assert!(outliers.len() <= voters.len() / 2);
}
