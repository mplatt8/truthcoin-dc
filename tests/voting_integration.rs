//! End-to-end integration tests for Bitcoin Hivemind Phase 1 voting system
//!
//! These tests verify that all voting components work together correctly,
//! including integration with the existing slot system, transaction
//! validation, and state management.

use truthcoin_dc::{
    state::{
        State, Error as StateError,
        voting::{
            VotingSystem,
            types::{VotingPeriod, VotingPeriodId, VotingPeriodStatus, Vote, VoteValue, VoterId},
        },
        slots::SlotId,
    },
    types::{Address, transaction::{Transaction, TransactionData, Output, OutputContent}},
    validation::validate_transaction,
};
use sneed::Env;
use tempfile::TempDir;
use std::collections::HashMap;

/// Integration test environment
struct IntegrationTestEnv {
    _temp_dir: TempDir,
    state: State,
}

impl IntegrationTestEnv {
    fn new() -> Result<Self, StateError> {
        let temp_dir = TempDir::new().map_err(|e| StateError::Database {
            msg: format!("Failed to create temp dir: {}", e),
        })?;

        let env = Env::new(&temp_dir, 1024 * 1024 * 100).map_err(|e| StateError::Database {
            msg: format!("Failed to create environment: {}", e),
        })?;

        let mut rwtxn = env.rw_txn().map_err(|e| StateError::Database {
            msg: format!("Failed to create transaction: {}", e),
        })?;

        let state = State::new(&env, &mut rwtxn)?;
        rwtxn.commit().map_err(|e| StateError::Database {
            msg: format!("Failed to commit transaction: {}", e),
        })?;

        Ok(Self {
            _temp_dir: temp_dir,
            state,
        })
    }
}

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
        .map(|i| SlotId::new(1, i as u16).unwrap())
        .collect()
}

fn create_test_tx_hash(seed: u8) -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0] = seed;
    hash
}

/// Test voting system integration with state management
#[test]
fn test_voting_system_state_integration() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period_id = VotingPeriodId::new(1);
    let decisions = create_test_decision_ids(3);
    let voters = create_test_voter_ids(2);

    // Test voting period creation through state
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().create_voting_period(
            &mut rwtxn,
            period_id,
            1000,
            2000,
            decisions.clone(),
            100,
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Verify period exists
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let period = test_env.state.voting_system().databases()
            .get_voting_period(&rotxn, period_id).unwrap().unwrap();
        assert_eq!(period.status, VotingPeriodStatus::Pending);
        rotxn.commit().unwrap();
    }

    // Test period activation
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        test_env.state.voting_system().activate_voting_period(
            &mut rwtxn,
            period_id,
            1000,
        ).unwrap();
        rwtxn.commit().unwrap();
    }

    // Test vote casting
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn,
            voters[0],
            period_id,
            decisions[0],
            VoteValue::Binary(true),
            1500,
            200,
            create_test_tx_hash(1),
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn,
            voters[1],
            period_id,
            decisions[1],
            VoteValue::Scalar(0.75),
            1600,
            201,
            create_test_tx_hash(2),
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Verify votes were stored
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let votes = test_env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();
        assert_eq!(votes.len(), 2);
        assert_eq!(votes.get(&(voters[0], decisions[0])), Some(&VoteValue::Binary(true)));
        assert_eq!(votes.get(&(voters[1], decisions[1])), Some(&VoteValue::Scalar(0.75)));
        rotxn.commit().unwrap();
    }
}

/// Test voting period lifecycle management
#[test]
fn test_voting_period_lifecycle_integration() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period_id = VotingPeriodId::new(1);
    let decisions = create_test_decision_ids(2);
    let voters = create_test_voter_ids(3);

    // Create period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        test_env.state.voting_system().create_voting_period(
            &mut rwtxn,
            period_id,
            1000,
            2000,
            decisions.clone(),
            100,
        ).unwrap();
        rwtxn.commit().unwrap();
    }

    // Test automatic period status updates
    {
        let rotxn = test_env.state.ro_txn().unwrap();

        // Check what periods need updates at different times
        let updates_before = test_env.state.voting_system()
            .get_periods_needing_update(&rotxn, 500).unwrap(); // Before start
        assert!(updates_before.is_empty());

        let updates_at_start = test_env.state.voting_system()
            .get_periods_needing_update(&rotxn, 1000).unwrap(); // At start
        assert_eq!(updates_at_start.len(), 1);
        assert_eq!(updates_at_start[0], (period_id, VotingPeriodStatus::Active));

        rotxn.commit().unwrap();
    }

    // Activate period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();
        rwtxn.commit().unwrap();
    }

    // Cast some votes during active period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period_id, decisions[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(1)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[1], period_id, decisions[0],
            VoteValue::Binary(false), 1600, 201, create_test_tx_hash(2)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[2], period_id, decisions[1],
            VoteValue::Scalar(0.6), 1700, 202, create_test_tx_hash(3)
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Test period closure
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let updates_at_end = test_env.state.voting_system()
            .get_periods_needing_update(&rotxn, 2000).unwrap(); // At end
        assert_eq!(updates_at_end.len(), 1);
        assert_eq!(updates_at_end[0], (period_id, VotingPeriodStatus::Closed));
        rotxn.commit().unwrap();
    }

    // Close period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        test_env.state.voting_system().close_voting_period(&mut rwtxn, period_id, 2000).unwrap();
        rwtxn.commit().unwrap();
    }

    // Test decision resolution
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        let outcomes = test_env.state.voting_system().resolve_period_decisions(
            &mut rwtxn,
            period_id,
            2500,
            300,
        ).unwrap();

        assert_eq!(outcomes.len(), 2); // Should resolve both decisions
        rwtxn.commit().unwrap();
    }

    // Verify period is resolved
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let period = test_env.state.voting_system().databases()
            .get_voting_period(&rotxn, period_id).unwrap().unwrap();
        assert_eq!(period.status, VotingPeriodStatus::Resolved);

        let outcomes = test_env.state.voting_system().get_period_outcomes(&rotxn, period_id).unwrap();
        assert_eq!(outcomes.len(), 2);
        rotxn.commit().unwrap();
    }
}

/// Test voter reputation management over multiple periods
#[test]
fn test_multi_period_reputation_management() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period1 = VotingPeriodId::new(1);
    let period2 = VotingPeriodId::new(2);
    let decisions = create_test_decision_ids(2);
    let voters = create_test_voter_ids(3);

    // Initialize voter reputations
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        for voter in &voters {
            test_env.state.voting_system().initialize_voter_reputation(
                &mut rwtxn,
                *voter,
                0.5, // Start with neutral reputation
                1000,
                period1,
            ).unwrap();
        }

        rwtxn.commit().unwrap();
    }

    // Create and run first voting period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().create_voting_period(
            &mut rwtxn, period1, 1000, 2000, decisions.clone(), 100
        ).unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period1, 1000).unwrap();

        // Simulate votes where voter[0] and voter[1] are accurate, voter[2] is not
        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period1, decisions[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(1)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[1], period1, decisions[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(2)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[2], period1, decisions[0],
            VoteValue::Binary(false), 1500, 200, create_test_tx_hash(3)
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Update reputations based on period 1 performance
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Voters 0 and 1 were correct (voted with majority)
        test_env.state.voting_system().update_voter_reputation(
            &mut rwtxn, voters[0], true, 2500, period2
        ).unwrap();

        test_env.state.voting_system().update_voter_reputation(
            &mut rwtxn, voters[1], true, 2500, period2
        ).unwrap();

        // Voter 2 was incorrect
        test_env.state.voting_system().update_voter_reputation(
            &mut rwtxn, voters[2], false, 2500, period2
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Verify reputation changes
    {
        let rotxn = test_env.state.ro_txn().unwrap();

        let rep0 = test_env.state.voting_system().databases()
            .get_voter_reputation(&rotxn, voters[0]).unwrap().unwrap();
        let rep1 = test_env.state.voting_system().databases()
            .get_voter_reputation(&rotxn, voters[1]).unwrap().unwrap();
        let rep2 = test_env.state.voting_system().databases()
            .get_voter_reputation(&rotxn, voters[2]).unwrap().unwrap();

        // Correct voters should have increased reputation
        assert!(rep0.reputation > 0.5);
        assert!(rep1.reputation > 0.5);

        // Incorrect voter should have decreased reputation
        assert!(rep2.reputation < 0.5);

        rotxn.commit().unwrap();
    }

    // Create second period and test weighted voting
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().create_voting_period(
            &mut rwtxn, period2, 3000, 4000, decisions.clone(), 200
        ).unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period2, 3000).unwrap();

        // Cast votes in second period
        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period2, decisions[0],
            VoteValue::Binary(true), 3500, 300, create_test_tx_hash(4)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[1], period2, decisions[0],
            VoteValue::Binary(true), 3500, 300, create_test_tx_hash(5)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[2], period2, decisions[0],
            VoteValue::Binary(false), 3500, 300, create_test_tx_hash(6)
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Test reputation-weighted outcomes
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let reputation_weights = test_env.state.voting_system()
            .get_reputation_weights(&rotxn, period2).unwrap();

        // High reputation voters should have more weight
        assert!(reputation_weights.get(&voters[0]).unwrap() > &0.5);
        assert!(reputation_weights.get(&voters[1]).unwrap() > &0.5);
        assert!(reputation_weights.get(&voters[2]).unwrap() < &0.5);

        rotxn.commit().unwrap();
    }
}

/// Test voting system integration with existing slot system
#[test]
fn test_slot_system_integration() {
    let test_env = IntegrationTestEnv::new().unwrap();

    // First, we need to create some slots that can be used for decisions
    let slot_ids = create_test_decision_ids(3);

    // Create voting period using existing slot IDs
    let period_id = VotingPeriodId::new(1);
    let voters = create_test_voter_ids(2);

    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Create voting period with slot decisions
        test_env.state.voting_system().create_voting_period(
            &mut rwtxn,
            period_id,
            1000,
            2000,
            slot_ids.clone(),
            100,
        ).unwrap();

        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();

        rwtxn.commit().unwrap();
    }

    // Test voting on slot-based decisions
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Vote on different slots
        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period_id, slot_ids[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(1)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[1], period_id, slot_ids[1],
            VoteValue::Scalar(0.8), 1500, 200, create_test_tx_hash(2)
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Verify votes are correctly associated with slots
    {
        let rotxn = test_env.state.ro_txn().unwrap();

        let vote1 = test_env.state.voting_system().databases()
            .get_vote(&rotxn, period_id, voters[0], slot_ids[0]).unwrap().unwrap();
        assert_eq!(vote1.decision_id, slot_ids[0]);
        assert_eq!(vote1.value, VoteValue::Binary(true));

        let vote2 = test_env.state.voting_system().databases()
            .get_vote(&rotxn, period_id, voters[1], slot_ids[1]).unwrap().unwrap();
        assert_eq!(vote2.decision_id, slot_ids[1]);
        assert_eq!(vote2.value, VoteValue::Scalar(0.8));

        rotxn.commit().unwrap();
    }
}

/// Test voting system statistics and analytics integration
#[test]
fn test_voting_analytics_integration() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period_id = VotingPeriodId::new(1);
    let decisions = create_test_decision_ids(4);
    let voters = create_test_voter_ids(5);

    // Setup voting period with comprehensive data
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Create period
        test_env.state.voting_system().create_voting_period(
            &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
        ).unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();

        // Initialize voter reputations
        for (i, voter) in voters.iter().enumerate() {
            let reputation = 0.2 + (i as f64 * 0.15); // 0.2 to 0.8
            test_env.state.voting_system().initialize_voter_reputation(
                &mut rwtxn, *voter, reputation, 1000, period_id
            ).unwrap();
        }

        // Create diverse voting pattern
        let votes = vec![
            (voters[0], decisions[0], VoteValue::Binary(true)),
            (voters[0], decisions[1], VoteValue::Binary(false)),
            (voters[0], decisions[2], VoteValue::Scalar(0.7)),

            (voters[1], decisions[0], VoteValue::Binary(true)),
            (voters[1], decisions[3], VoteValue::Abstain),

            (voters[2], decisions[0], VoteValue::Binary(false)),
            (voters[2], decisions[1], VoteValue::Binary(true)),
            (voters[2], decisions[2], VoteValue::Scalar(0.3)),
            (voters[2], decisions[3], VoteValue::Scalar(0.9)),

            (voters[3], decisions[1], VoteValue::Binary(true)),

            // voters[4] doesn't vote at all
        ];

        for (i, (voter, decision, value)) in votes.into_iter().enumerate() {
            test_env.state.voting_system().cast_vote(
                &mut rwtxn, voter, period_id, decision, value,
                1500 + i as u64, 200 + i as u64, create_test_tx_hash(i as u8)
            ).unwrap();
        }

        rwtxn.commit().unwrap();
    }

    // Test participation statistics
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let (total_voters, total_votes, participation_rate) = test_env.state.voting_system()
            .get_participation_stats(&rotxn, period_id).unwrap();

        assert_eq!(total_voters, 4); // Only 4 voters participated
        assert_eq!(total_votes, 10); // Total individual votes cast

        // Participation rate = 10 votes / (4 active voters * 4 decisions) = 10/16 = 0.625
        assert!((participation_rate - 0.625).abs() < 0.001);

        rotxn.commit().unwrap();
    }

    // Test vote matrix generation
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let vote_matrix = test_env.state.voting_system().get_vote_matrix(&rotxn, period_id).unwrap();

        // Should contain only non-abstain votes
        assert_eq!(vote_matrix.len(), 9); // 10 total - 1 abstain = 9

        // Verify specific votes are in matrix
        assert_eq!(vote_matrix.get(&(voters[0], decisions[0])), Some(&1.0)); // Binary true
        assert_eq!(vote_matrix.get(&(voters[2], decisions[0])), Some(&0.0)); // Binary false
        assert_eq!(vote_matrix.get(&(voters[0], decisions[2])), Some(&0.7)); // Scalar
        assert!(vote_matrix.get(&(voters[1], decisions[3])).is_none()); // Abstain not in matrix

        rotxn.commit().unwrap();
    }

    // Test comprehensive period statistics
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let stats = test_env.state.voting_system().calculate_period_statistics(
            &rotxn, period_id, 2500
        ).unwrap();

        assert_eq!(stats.period_id, period_id);
        assert_eq!(stats.total_voters, 4);
        assert_eq!(stats.total_votes, 10);
        assert_eq!(stats.total_decisions, 4);
        assert!((stats.avg_participation_rate - 0.625).abs() < 0.001);

        // Total reputation weight should be sum of active voters' reputations
        let expected_weight = 0.2 + 0.35 + 0.5 + 0.65; // First 4 voters
        assert!((stats.total_reputation_weight - expected_weight).abs() < 0.001);

        rotxn.commit().unwrap();
    }

    // Test system-wide statistics
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let (total_periods, total_votes, total_voters, avg_reputation) = test_env.state.voting_system()
            .get_system_stats(&rotxn).unwrap();

        assert_eq!(total_periods, 1);
        assert_eq!(total_votes, 10);
        assert_eq!(total_voters, 5); // All voters who have reputation

        // Average reputation of all 5 voters
        let expected_avg = (0.2 + 0.35 + 0.5 + 0.65 + 0.8) / 5.0;
        assert!((avg_reputation - expected_avg).abs() < 0.001);

        rotxn.commit().unwrap();
    }
}

/// Test database consistency validation across voting components
#[test]
fn test_cross_component_consistency() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period_id = VotingPeriodId::new(1);
    let decisions = create_test_decision_ids(2);
    let voters = create_test_voter_ids(3);

    // Create complete voting scenario
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Create period
        test_env.state.voting_system().create_voting_period(
            &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
        ).unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();

        // Initialize reputations
        for (i, voter) in voters.iter().enumerate() {
            test_env.state.voting_system().initialize_voter_reputation(
                &mut rwtxn, *voter, 0.3 + (i as f64 * 0.2), 1000, period_id
            ).unwrap();
        }

        // Cast votes
        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period_id, decisions[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(1)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[1], period_id, decisions[1],
            VoteValue::Scalar(0.8), 1500, 200, create_test_tx_hash(2)
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Close period and create outcomes
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().close_voting_period(&mut rwtxn, period_id, 2000).unwrap();

        let outcomes = test_env.state.voting_system().resolve_period_decisions(
            &mut rwtxn, period_id, 2500, 300
        ).unwrap();

        assert_eq!(outcomes.len(), 2);
        rwtxn.commit().unwrap();
    }

    // Validate consistency across all components
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let consistency_issues = test_env.state.voting_system().validate_consistency(&rotxn).unwrap();
        assert!(consistency_issues.is_empty(), "Found consistency issues: {:?}", consistency_issues);
        rotxn.commit().unwrap();
    }

    // Verify data integrity across different queries
    {
        let rotxn = test_env.state.ro_txn().unwrap();

        // Check that all votes have corresponding voter reputations
        let votes = test_env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();
        for (voter_id, _) in votes.keys() {
            let reputation = test_env.state.voting_system().databases()
                .get_voter_reputation(&rotxn, *voter_id).unwrap();
            assert!(reputation.is_some(), "Missing reputation for voter {:?}", voter_id);
        }

        // Check that all outcomes correspond to decisions in the period
        let period = test_env.state.voting_system().databases()
            .get_voting_period(&rotxn, period_id).unwrap().unwrap();
        let outcomes = test_env.state.voting_system().get_period_outcomes(&rotxn, period_id).unwrap();

        for outcome_decision in outcomes.keys() {
            assert!(period.decision_slots.contains(outcome_decision),
                "Outcome for decision {:?} not in period decisions", outcome_decision);
        }

        rotxn.commit().unwrap();
    }
}

/// Test concurrent voting operations (simulated)
#[test]
fn test_concurrent_voting_simulation() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period_id = VotingPeriodId::new(1);
    let decisions = create_test_decision_ids(5);
    let voters = create_test_voter_ids(10);

    // Setup period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        test_env.state.voting_system().create_voting_period(
            &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
        ).unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();
        rwtxn.commit().unwrap();
    }

    // Simulate concurrent vote submissions (sequential execution for testing)
    for (i, voter) in voters.iter().enumerate() {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Each voter votes on a random subset of decisions
        for (j, decision) in decisions.iter().enumerate() {
            if (i + j) % 3 == 0 { // Some voting pattern
                let vote_value = match (i + j) % 3 {
                    0 => VoteValue::Binary(true),
                    1 => VoteValue::Binary(false),
                    _ => VoteValue::Scalar((i + j) as f64 / 20.0),
                };

                test_env.state.voting_system().cast_vote(
                    &mut rwtxn,
                    *voter,
                    period_id,
                    *decision,
                    vote_value,
                    1500 + (i * 10 + j) as u64,
                    200 + (i * 10 + j) as u64,
                    create_test_tx_hash((i * 10 + j) as u8),
                ).unwrap();
            }
        }

        rwtxn.commit().unwrap();
    }

    // Verify all votes were stored correctly
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let all_votes = test_env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();

        // Count expected votes
        let mut expected_count = 0;
        for i in 0..voters.len() {
            for j in 0..decisions.len() {
                if (i + j) % 3 == 0 {
                    expected_count += 1;
                }
            }
        }

        assert_eq!(all_votes.len(), expected_count);
        rotxn.commit().unwrap();
    }

    // Test concurrent read operations
    for _ in 0..5 {
        let rotxn = test_env.state.ro_txn().unwrap();

        let _votes = test_env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();
        let _matrix = test_env.state.voting_system().get_vote_matrix(&rotxn, period_id).unwrap();
        let _stats = test_env.state.voting_system().get_participation_stats(&rotxn, period_id).unwrap();

        rotxn.commit().unwrap();
    }
}

/// Test error handling and recovery in integrated environment
#[test]
fn test_integrated_error_handling() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let period_id = VotingPeriodId::new(1);
    let decisions = create_test_decision_ids(2);
    let voters = create_test_voter_ids(2);

    // Create period
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();
        test_env.state.voting_system().create_voting_period(
            &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
        ).unwrap();
        test_env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();
        rwtxn.commit().unwrap();
    }

    // Test transaction rollback on error
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        // Cast valid vote
        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period_id, decisions[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(1)
        ).unwrap();

        // Try to cast duplicate vote (should fail)
        let result = test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period_id, decisions[0],
            VoteValue::Binary(false), 1600, 201, create_test_tx_hash(2)
        );
        assert!(result.is_err());

        // Rollback transaction
        rwtxn.abort();
    }

    // Verify no votes were stored due to rollback
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let votes = test_env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();
        assert_eq!(votes.len(), 0);
        rotxn.commit().unwrap();
    }

    // Test successful recovery after error
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[0], period_id, decisions[0],
            VoteValue::Binary(true), 1500, 200, create_test_tx_hash(1)
        ).unwrap();

        test_env.state.voting_system().cast_vote(
            &mut rwtxn, voters[1], period_id, decisions[1],
            VoteValue::Scalar(0.7), 1600, 201, create_test_tx_hash(2)
        ).unwrap();

        rwtxn.commit().unwrap();
    }

    // Verify successful votes were stored
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let votes = test_env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();
        assert_eq!(votes.len(), 2);
        rotxn.commit().unwrap();
    }
}

/// Test large-scale voting scenario
#[test]
fn test_large_scale_voting_scenario() {
    let test_env = IntegrationTestEnv::new().unwrap();

    let num_periods = 3;
    let num_voters = 50;
    let num_decisions_per_period = 10;

    let voters = create_test_voter_ids(num_voters);

    // Initialize all voter reputations
    {
        let mut rwtxn = test_env.state.rw_txn().unwrap();

        for (i, voter) in voters.iter().enumerate() {
            let initial_reputation = 0.3 + (i as f64 / num_voters as f64 * 0.4); // 0.3 to 0.7
            test_env.state.voting_system().initialize_voter_reputation(
                &mut rwtxn, *voter, initial_reputation, 1000, VotingPeriodId::new(1)
            ).unwrap();
        }

        rwtxn.commit().unwrap();
    }

    // Run multiple voting periods
    for period_num in 1..=num_periods {
        let period_id = VotingPeriodId::new(period_num);
        let decisions = create_test_decision_ids(num_decisions_per_period);
        let start_time = 1000 + (period_num - 1) * 10000;
        let end_time = start_time + 5000;

        // Create and activate period
        {
            let mut rwtxn = test_env.state.rw_txn().unwrap();

            test_env.state.voting_system().create_voting_period(
                &mut rwtxn, period_id, start_time, end_time, decisions.clone(), 100 + period_num * 50
            ).unwrap();

            test_env.state.voting_system().activate_voting_period(
                &mut rwtxn, period_id, start_time
            ).unwrap();

            rwtxn.commit().unwrap();
        }

        // Cast votes with realistic participation (70% of voters vote on 60% of decisions)
        {
            let mut rwtxn = test_env.state.rw_txn().unwrap();
            let mut vote_counter = 0u8;

            for (voter_idx, voter) in voters.iter().enumerate() {
                if voter_idx as f64 / num_voters as f64 < 0.7 { // 70% participation
                    for (decision_idx, decision) in decisions.iter().enumerate() {
                        if decision_idx as f64 / num_decisions_per_period as f64 < 0.6 { // 60% decision coverage
                            let vote_value = match (voter_idx + decision_idx + period_num) % 4 {
                                0 => VoteValue::Binary(true),
                                1 => VoteValue::Binary(false),
                                2 => VoteValue::Scalar((voter_idx + decision_idx) as f64 / 100.0),
                                _ => VoteValue::Abstain,
                            };

                            test_env.state.voting_system().cast_vote(
                                &mut rwtxn,
                                *voter,
                                period_id,
                                *decision,
                                vote_value,
                                start_time + 1000 + vote_counter as u64,
                                200 + period_num * 100 + vote_counter as u64,
                                create_test_tx_hash(vote_counter),
                            ).unwrap();

                            vote_counter = vote_counter.wrapping_add(1);
                        }
                    }
                }
            }

            rwtxn.commit().unwrap();
        }

        // Close period and resolve decisions
        {
            let mut rwtxn = test_env.state.rw_txn().unwrap();

            test_env.state.voting_system().close_voting_period(
                &mut rwtxn, period_id, end_time
            ).unwrap();

            let outcomes = test_env.state.voting_system().resolve_period_decisions(
                &mut rwtxn, period_id, end_time + 1000, 300 + period_num * 100
            ).unwrap();

            assert_eq!(outcomes.len(), num_decisions_per_period);
            rwtxn.commit().unwrap();
        }
    }

    // Verify system-wide statistics
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let (total_periods, total_votes, total_voters, avg_reputation) = test_env.state.voting_system()
            .get_system_stats(&rotxn).unwrap();

        assert_eq!(total_periods as usize, num_periods);
        assert_eq!(total_voters as usize, num_voters);
        assert!(total_votes > 0);
        assert!(avg_reputation > 0.0 && avg_reputation <= 1.0);

        rotxn.commit().unwrap();
    }

    // Validate final consistency
    {
        let rotxn = test_env.state.ro_txn().unwrap();
        let consistency_issues = test_env.state.voting_system().validate_consistency(&rotxn).unwrap();
        assert!(consistency_issues.is_empty(), "Found consistency issues in large-scale test: {:?}", consistency_issues);
        rotxn.commit().unwrap();
    }
}