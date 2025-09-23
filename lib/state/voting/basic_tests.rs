//! Basic tests for voting data structures without full State integration
//!
//! These tests validate the core voting functionality independently of
//! the State integration, ensuring the foundation is solid.

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use crate::state::slots::SlotId;
    use crate::types::Address;

    // Helper functions for creating test data
    fn create_test_voter_id(index: u8) -> VoterId {
        let mut addr_bytes = [0u8; 20];
        addr_bytes[0] = index;
        VoterId::from_address(&Address(addr_bytes))
    }

    fn create_test_decision_id(index: u32) -> SlotId {
        SlotId::new(1, index).unwrap()
    }

    #[test]
    fn test_voting_period_creation() {
        let period_id = VotingPeriodId::new(1);
        let start_time = 1000u64;
        let end_time = 2000u64;
        let decision_slots =
            vec![create_test_decision_id(1), create_test_decision_id(2)];
        let created_height = 100u32;

        let period = VotingPeriod::new(
            period_id,
            start_time,
            end_time,
            decision_slots.clone(),
            VotingPeriodStatus::Created,
            created_height,
        );

        assert_eq!(period.id, period_id);
        assert_eq!(period.start_time, start_time);
        assert_eq!(period.end_time, end_time);
        assert_eq!(period.decision_slots, decision_slots);
        assert_eq!(period.status, VotingPeriodStatus::Created);
        assert_eq!(period.created_height, created_height);

        assert!(!period.is_active(1500));
        assert!(period.is_voting_open(1500));
    }

    #[test]
    fn test_vote_creation() {
        let voter_id = create_test_voter_id(1);
        let decision_id = create_test_decision_id(1);
        let voting_period = VotingPeriodId::new(1);
        let vote_value = VoteValue::Binary(true);

        let vote = Vote::new(
            voter_id,
            decision_id,
            voting_period,
            vote_value.clone(),
            1234u64,
        );

        assert_eq!(vote.voter_id, voter_id);
        assert_eq!(vote.decision_id, decision_id);
        assert_eq!(vote.voting_period, voting_period);
        assert_eq!(vote.vote_value, vote_value);
        assert_eq!(vote.timestamp, 1234u64);
    }

    #[test]
    fn test_vote_value_types() {
        let binary_true = VoteValue::Binary(true);
        let binary_false = VoteValue::Binary(false);
        let scaled = VoteValue::Scaled(0.75);

        assert_eq!(binary_true.to_f64(), 1.0);
        assert_eq!(binary_false.to_f64(), 0.0);
        assert_eq!(scaled.to_f64(), 0.75);

        assert!(binary_true.is_binary());
        assert!(binary_false.is_binary());
        assert!(!scaled.is_binary());
    }

    #[test]
    fn test_voter_reputation() {
        let voter_id = create_test_voter_id(1);
        let period_id = VotingPeriodId::new(1);

        let mut reputation = VoterReputation::new(voter_id, 0.5);
        assert_eq!(reputation.voter_id, voter_id);
        assert_eq!(reputation.reputation, 0.5);
        assert_eq!(reputation.vote_count, 0);
        assert_eq!(reputation.accuracy_score, 0.0);

        reputation.update_reputation(period_id, 0.75, 0.8);
        assert_eq!(reputation.reputation, 0.75);
        assert_eq!(reputation.vote_count, 1);
        assert_eq!(reputation.accuracy_score, 0.8);
    }

    #[test]
    fn test_decision_outcome() {
        let decision_id = create_test_decision_id(1);
        let period_id = VotingPeriodId::new(1);

        let outcome = DecisionOutcome::new(
            decision_id,
            period_id,
            OutcomeValue::Binary(true),
            0.85,
            123u32,
        );

        assert_eq!(outcome.decision_id, decision_id);
        assert_eq!(outcome.voting_period, period_id);
        assert_eq!(outcome.outcome_value, OutcomeValue::Binary(true));
        assert_eq!(outcome.confidence, 0.85);
        assert_eq!(outcome.resolution_height, 123u32);
    }

    #[test]
    fn test_voting_period_stats() {
        let period_id = VotingPeriodId::new(1);

        let stats = VotingPeriodStats::new(
            period_id, 100,  // total_voters
            75,   // active_voters
            50,   // total_decisions
            45,   // decisions_with_votes
            2250, // total_votes
        );

        assert_eq!(stats.period_id, period_id);
        assert_eq!(stats.total_voters, 100);
        assert_eq!(stats.active_voters, 75);
        assert_eq!(stats.total_decisions, 50);
        assert_eq!(stats.decisions_with_votes, 45);
        assert_eq!(stats.total_votes, 2250);

        assert_eq!(stats.participation_rate(), 0.75);
        assert_eq!(stats.decision_coverage_rate(), 0.9);
        assert_eq!(stats.average_votes_per_decision(), 45.0);
    }

    #[test]
    fn test_vote_id_generation() {
        let voter_id = create_test_voter_id(42);
        let decision_id = create_test_decision_id(123);

        let vote_id_1 = VoteId::new(&voter_id, &decision_id);
        let vote_id_2 = VoteId::new(&voter_id, &decision_id);
        let vote_id_3 = VoteId::new(&create_test_voter_id(43), &decision_id);

        // Same voter + decision should produce same ID
        assert_eq!(vote_id_1, vote_id_2);

        // Different voter should produce different ID
        assert_ne!(vote_id_1, vote_id_3);
    }

    #[test]
    fn test_voting_period_status_transitions() {
        let period_id = VotingPeriodId::new(1);
        let decision_slots = vec![create_test_decision_id(1)];

        let mut period = VotingPeriod::new(
            period_id,
            1000,
            2000,
            decision_slots,
            VotingPeriodStatus::Created,
            100,
        );

        // Test status transitions
        assert_eq!(period.status, VotingPeriodStatus::Created);

        // Simulate activation
        period.status = VotingPeriodStatus::Active;
        assert_eq!(period.status, VotingPeriodStatus::Active);

        // Simulate closure
        period.status = VotingPeriodStatus::Closed;
        assert_eq!(period.status, VotingPeriodStatus::Closed);
    }
}
