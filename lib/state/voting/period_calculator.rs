//! Bitcoin Hivemind Period Calculator and State Machine
//!
//! # Single Source of Truth for ALL Period-Related Operations
//!
//! This module is the **SOLE AUTHORITY** for all period-related operations:
//! - Calculating voting period status based on timestamps and consensus state
//! - Validating state transitions
//! - Determining period boundaries from period indices
//! - Querying decision slots for each period
//! - Checking if votes can be accepted
//! - Providing complete VotingPeriod structs with correct status
//!
//! No other module should duplicate this logic to avoid redundancy and ensure consistency.
//!
//! ## Bitcoin Hivemind Principles
//! - Slots encode their claim period in their ID (10 bits)
//! - Slots claimed in period N are voted on in period N+1
//! - Period boundaries are deterministically calculated from the period index
//! - Decision slots for a period are found by querying all slots with matching period_index()
//!
//! ## State Transition Diagram
//! ```text
//! Pending --> Active --> Closed --> Resolved
//!    ^         |          |          |
//!    |         v          v          v
//!    +-------- Invalid Transitions ----+
//! ```
//!
//! This eliminates the need to store voting period entities in the database,
//! ensuring consistency and reducing redundancy.

use crate::state::{
    Error,
    slots::{SlotId, SlotConfig, Dbs as SlotsDbs},
    voting::types::{VotingPeriod, VotingPeriodId, VotingPeriodStatus},
};
use sneed::RoTxn;
use std::collections::HashMap;

/// Calculate voting period boundaries from period index
///
/// # Arguments
/// * `period_index` - The voting period index (not the slot claim period)
/// * `config` - Slot configuration for testing vs production mode
///
/// # Returns
/// Tuple of (start_timestamp, end_timestamp) in seconds
///
/// # Bitcoin Hivemind Compliance
/// Period boundaries must be deterministic and consensus-critical.
/// All nodes must calculate the same boundaries for the same period index.
pub fn calculate_period_boundaries(
    period_index: u32,
    config: &SlotConfig,
) -> (u64, u64) {
    if config.testing_mode {
        // In testing mode, use block-based periods
        let start_block = period_index * config.testing_blocks_per_period;
        let end_block = (period_index + 1) * config.testing_blocks_per_period;

        // Use block numbers as proxy for timestamps in testing
        (start_block as u64, end_block as u64)
    } else {
        // In production, use quarter-based periods (3 months each)
        use crate::math::voting::constants::PRODUCTION_PERIOD_DURATION_SECONDS;
        let start = period_index as u64 * PRODUCTION_PERIOD_DURATION_SECONDS;
        let end = start + PRODUCTION_PERIOD_DURATION_SECONDS;
        (start, end)
    }
}

/// Calculate voting period status from timestamps
///
/// **SINGLE SOURCE OF TRUTH for period status calculation**
///
/// This is the ONLY function in the codebase that should determine period status
/// based on timestamps. All other code must call this function or use VotingPeriod
/// structs with status already calculated by this function.
///
/// # Arguments
/// * `start_timestamp` - Period start time
/// * `end_timestamp` - Period end time
/// * `current_timestamp` - Current L1 timestamp
/// * `has_outcomes` - Whether consensus outcomes have been calculated
///
/// # Returns
/// Current status of the voting period
///
/// # Bitcoin Hivemind Compliance
/// Status transitions are deterministic based on timestamps:
/// - Pending: before start_timestamp
/// - Active: between start and end timestamps
/// - Closed: after end_timestamp, before outcomes calculated
/// - Resolved: after outcomes calculated
///
/// # Architectural Note
/// This function replaced redundant logic that previously existed in:
/// - state_machine::determine_next_state() [REMOVED]
/// - state_machine::can_accept_votes() [SIMPLIFIED to check status only]
pub fn calculate_period_status(
    start_timestamp: u64,
    end_timestamp: u64,
    current_timestamp: u64,
    has_outcomes: bool,
) -> VotingPeriodStatus {
    if has_outcomes {
        VotingPeriodStatus::Resolved
    } else if current_timestamp >= end_timestamp {
        VotingPeriodStatus::Closed
    } else if current_timestamp >= start_timestamp {
        VotingPeriodStatus::Active
    } else {
        VotingPeriodStatus::Pending
    }
}

/// Get all decision slots for a voting period by querying the slot database
///
/// # Arguments
/// * `rotxn` - Read-only transaction
/// * `voting_period_id` - The voting period to query
/// * `slots_db` - Reference to slots database
///
/// # Returns
/// Vector of SlotIds that should be voted on in this period
///
/// # Bitcoin Hivemind Principle
/// Slots claimed in period N are voted on in period N+1.
/// This function finds all slots where slot.period_index() == voting_period_id - 1
pub fn get_decision_slots_for_period(
    rotxn: &RoTxn,
    voting_period_id: VotingPeriodId,
    slots_db: &SlotsDbs,
) -> Result<Vec<SlotId>, Error> {
    // Voting period N contains decisions from slots claimed in period N-1
    let voting_period = voting_period_id.as_u32();

    if voting_period == 0 {
        // Period 0 has no slots (nothing was claimed in period -1)
        return Ok(Vec::new());
    }

    // Inverse of SlotId::voting_period(): voting_period = claim_period + 1
    // Therefore: claim_period = voting_period - 1
    let claim_period = voting_period - 1;

    // Query all claimed slots and filter by period_index
    let all_slots = slots_db.get_all_claimed_slots(rotxn)?;

    let mut decision_slots = Vec::new();
    for slot in all_slots {
        if slot.slot_id.period_index() == claim_period {
            decision_slots.push(slot.slot_id);
        }
    }

    Ok(decision_slots)
}

/// Calculate a complete VotingPeriod struct on-demand
///
/// # Arguments
/// * `rotxn` - Read-only transaction
/// * `period_id` - Period to calculate
/// * `current_timestamp` - Current L1 timestamp for status calculation
/// * `config` - Slot configuration
/// * `slots_db` - Slots database reference
/// * `has_outcomes` - Whether consensus outcomes exist for this period
///
/// # Returns
/// Calculated VotingPeriod struct with all fields derived from fundamental data
///
/// # Bitcoin Hivemind Compliance
/// This is the single source of truth for period information. No period data
/// is stored in the database - everything is calculated from:
/// - Period index (deterministic boundaries)
/// - Slot database (decision slots)
/// - Current timestamp (status)
/// - Consensus database (resolved status)
pub fn calculate_voting_period(
    rotxn: &RoTxn,
    period_id: VotingPeriodId,
    current_timestamp: u64,
    config: &SlotConfig,
    slots_db: &SlotsDbs,
    has_outcomes: bool,
) -> Result<VotingPeriod, Error> {
    let (start_timestamp, end_timestamp) = calculate_period_boundaries(
        period_id.as_u32(),
        config,
    );

    let decision_slots = get_decision_slots_for_period(
        rotxn,
        period_id,
        slots_db,
    )?;

    // Calculate and set the correct period status based on current timestamp and consensus state
    // This ensures the period always has the correct status without requiring callers to update it

    // created_at_height is not available from calculated data
    // Use 0 as a sentinel value for calculated periods
    let created_at_height = 0;

    // Calculate the actual status based on timestamps and consensus state
    let status = calculate_period_status(
        start_timestamp,
        end_timestamp,
        current_timestamp,
        has_outcomes,
    );

    // Create the period with default status (will be overridden)
    let mut period = VotingPeriod::new(
        period_id,
        start_timestamp,
        end_timestamp,
        decision_slots,
        created_at_height,
    );

    // Set the correct calculated status
    period.status = status;

    Ok(period)
}

/// Get all voting periods that currently exist (have decision slots)
///
/// # Arguments
/// * `rotxn` - Read-only transaction
/// * `slots_db` - Slots database reference
/// * `config` - Slot configuration
/// * `current_timestamp` - Current timestamp for status calculation
/// * `voting_db` - Voting database reference for checking outcomes
///
/// # Returns
/// Map of VotingPeriodId to calculated VotingPeriod for all periods with slots
///
/// # Bitcoin Hivemind Compliance
/// This scans all claimed slots and determines which voting periods have content.
pub fn get_all_active_periods(
    rotxn: &RoTxn,
    slots_db: &SlotsDbs,
    config: &SlotConfig,
    current_timestamp: u64,
    voting_db: &crate::state::voting::database::VotingDatabases,
) -> Result<HashMap<VotingPeriodId, VotingPeriod>, Error> {
    let all_slots = slots_db.get_all_claimed_slots(rotxn)?;
    let mut period_map: HashMap<u32, Vec<SlotId>> = HashMap::new();

    // Group slots by their voting period (using SlotId::voting_period())
    for slot in all_slots {
        let voting_period = slot.slot_id.voting_period();
        period_map.entry(voting_period).or_insert_with(Vec::new).push(slot.slot_id);
    }

    // Build VotingPeriod for each period that has slots
    let mut result = HashMap::new();
    for (period_index, decision_slots) in period_map {
        let period_id = VotingPeriodId::new(period_index);
        let has_outcomes = voting_db.has_consensus(rotxn, period_id)?;

        let (start_timestamp, end_timestamp) = calculate_period_boundaries(
            period_index,
            config,
        );

        let status = calculate_period_status(
            start_timestamp,
            end_timestamp,
            current_timestamp,
            has_outcomes,
        );

        let period = VotingPeriod {
            id: period_id,
            start_timestamp,
            end_timestamp,
            status,
            decision_slots,
            created_at_height: 0,
        };

        result.insert(period_id, period);
    }

    Ok(result)
}

// ================================================================================
// State Transition Validation
// ================================================================================

/// Validate and execute state transition
///
/// # Arguments
/// * `period` - Current voting period
/// * `new_status` - Desired new status
/// * `current_timestamp` - Current L1 timestamp
///
/// # Returns
/// Ok(()) if transition is valid
///
/// # Errors
/// Returns error if transition is invalid
///
/// # Bitcoin Hivemind Compliance
/// Section 4.1: "Voting Periods" - Deterministic period lifecycle
/// Section 4.2: "Consensus Algorithm" - State transitions tied to consensus
pub fn validate_transition(
    period: &VotingPeriod,
    new_status: VotingPeriodStatus,
    current_timestamp: u64,
) -> Result<(), Error> {
    match (period.status, new_status) {
        // Valid transitions
        (VotingPeriodStatus::Pending, VotingPeriodStatus::Active) => {
            validate_pending_to_active(period, current_timestamp)
        }
        (VotingPeriodStatus::Active, VotingPeriodStatus::Closed) => {
            validate_active_to_closed(period, current_timestamp)
        }
        (VotingPeriodStatus::Closed, VotingPeriodStatus::Resolved) => {
            validate_closed_to_resolved(period)
        }

        // Same state (no-op)
        (status, new) if status == new => Ok(()),

        // Invalid transitions
        _ => Err(Error::InvalidTransaction {
            reason: format!(
                "Invalid voting period state transition: {:?} -> {:?}",
                period.status, new_status
            ),
        }),
    }
}

/// Validate Pending -> Active transition
///
/// # Bitcoin Hivemind Compliance
/// Period must reach its start_timestamp before activation
fn validate_pending_to_active(
    period: &VotingPeriod,
    current_timestamp: u64,
) -> Result<(), Error> {
    if current_timestamp < period.start_timestamp {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Cannot activate period before start time (current: {}, start: {})",
                current_timestamp, period.start_timestamp
            ),
        });
    }

    Ok(())
}

/// Validate Active -> Closed transition
///
/// # Bitcoin Hivemind Compliance
/// Period must reach its end_timestamp before closing
fn validate_active_to_closed(
    period: &VotingPeriod,
    current_timestamp: u64,
) -> Result<(), Error> {
    if current_timestamp < period.end_timestamp {
        return Err(Error::InvalidTransaction {
            reason: format!(
                "Cannot close period before end time (current: {}, end: {})",
                current_timestamp, period.end_timestamp
            ),
        });
    }

    Ok(())
}

/// Validate Closed -> Resolved transition
///
/// # Bitcoin Hivemind Compliance
/// Period can only be resolved after consensus calculation
/// (This is validated by checking if consensus outcomes exist)
fn validate_closed_to_resolved(_period: &VotingPeriod) -> Result<(), Error> {
    // No additional validation needed here - consensus existence
    // is checked by the calling code
    Ok(())
}

// ================================================================================
// State-Based Permission Checks
// ================================================================================

/// Check if a period can accept votes
///
/// # Arguments
/// * `period` - Voting period to check (with status already calculated by calculate_period_status)
///
/// # Returns
/// true if votes can be accepted
///
/// # Bitcoin Hivemind Compliance
/// Votes can only be cast during the Active state.
///
/// # Design Note
/// This method relies on the period's status having been correctly calculated by
/// calculate_period_status(). It does NOT recalculate status based on timestamps
/// to maintain Single Source of Truth.
pub fn can_accept_votes(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Active
}

/// Check if a period can be closed
///
/// # Arguments
/// * `period` - Voting period to check (with status already calculated by calculate_period_status)
///
/// # Returns
/// true if period can be closed
///
/// # Design Note
/// This checks if transition from Active -> Closed is valid.
/// The period's status should already reflect whether end_timestamp has been reached.
pub fn can_close(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Active
}

/// Check if a period can be resolved
///
/// # Arguments
/// * `period` - Voting period to check
///
/// # Returns
/// true if period can be resolved
pub fn can_resolve(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Closed
}

/// Check if a period is in a terminal state
///
/// # Arguments
/// * `period` - Voting period to check
///
/// # Returns
/// true if period is in Resolved state (terminal)
pub fn is_terminal(period: &VotingPeriod) -> bool {
    period.status == VotingPeriodStatus::Resolved
}

// ================================================================================
// Tests
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::VotingPeriodId;

    fn create_test_period(status: VotingPeriodStatus) -> VotingPeriod {
        let mut period = VotingPeriod::new(
            VotingPeriodId::new(1),
            1000,
            2000,
            vec![],
            0,
        );
        period.status = status;
        period
    }

    #[test]
    fn test_valid_transitions() {
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Active,
            1000
        )
        .is_ok());

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Closed,
            2000
        )
        .is_ok());

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Resolved,
            2000
        )
        .is_ok());
    }

    #[test]
    fn test_invalid_transitions() {
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Closed,
            1000
        )
        .is_err());

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Resolved,
            2000
        )
        .is_err());
    }

    #[test]
    fn test_early_activation() {
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Active,
            999 // Before start time
        )
        .is_err());
    }

    #[test]
    fn test_early_close() {
        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(validate_transition(
            &period,
            VotingPeriodStatus::Closed,
            1999 // Before end time
        )
        .is_err());
    }

    #[test]
    fn test_can_accept_votes() {
        // Test that Active periods can accept votes
        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(can_accept_votes(&period));

        // Test that non-Active periods cannot accept votes
        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!can_accept_votes(&period));

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(!can_accept_votes(&period));

        let period = create_test_period(VotingPeriodStatus::Resolved);
        assert!(!can_accept_votes(&period));
    }

    #[test]
    fn test_can_close() {
        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(can_close(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!can_close(&period));

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(!can_close(&period));
    }

    #[test]
    fn test_can_resolve() {
        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(can_resolve(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!can_resolve(&period));

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(!can_resolve(&period));

        let period = create_test_period(VotingPeriodStatus::Resolved);
        assert!(!can_resolve(&period));
    }

    #[test]
    fn test_is_terminal() {
        let period = create_test_period(VotingPeriodStatus::Resolved);
        assert!(is_terminal(&period));

        let period = create_test_period(VotingPeriodStatus::Pending);
        assert!(!is_terminal(&period));

        let period = create_test_period(VotingPeriodStatus::Active);
        assert!(!is_terminal(&period));

        let period = create_test_period(VotingPeriodStatus::Closed);
        assert!(!is_terminal(&period));
    }
}
