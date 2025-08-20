//! Common validation utilities for Bitcoin Hivemind sidechain
//! 
//! This module provides shared validation logic to eliminate code duplication
//! across RPC, state management, and transaction processing layers.
//! All implementations follow the Bitcoin Hivemind whitepaper specifications.

use hex;
use crate::state::slots::{SlotId, Decision};
use crate::state::Error;
use crate::types::{FilledTransaction, Address};
use sneed::RoTxn;

/// Trait for slot validation database interface
/// 
/// Abstracts the slots database operations needed for validation,
/// allowing the validation logic to work with different database
/// implementations while maintaining Bitcoin Hivemind compliance.
pub trait SlotValidationInterface {
    /// Validate that a slot can be claimed according to Hivemind rules
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error>;

    /// Try to get current blockchain height for validation context
    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error>;
}

/// Slot ID validation utilities
/// 
/// Provides common slot ID parsing and validation logic used across
/// RPC methods and transaction validation. Follows Hivemind whitepaper
/// slot identification specification.
pub struct SlotValidator;

impl SlotValidator {
    /// Parse slot ID from hex string with comprehensive validation
    /// 
    /// # Arguments
    /// * `slot_id_hex` - Hex-encoded slot ID string
    /// 
    /// # Returns
    /// * `Ok(SlotId)` - Valid slot ID
    /// * `Err(Error)` - Invalid format or slot ID
    /// 
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on slot addressing and period indexing
    pub fn parse_slot_id_from_hex(slot_id_hex: &str) -> Result<SlotId, Error> {
        // Parse slot ID from hex
        let slot_id_bytes = hex::decode(slot_id_hex)
            .map_err(|_| Error::InvalidSlotId {
                reason: "Invalid slot ID hex format".to_string(),
            })?;

        if slot_id_bytes.len() != 3 {
            return Err(Error::InvalidSlotId {
                reason: "Slot ID must be exactly 3 bytes".to_string(),
            });
        }

        let slot_id_array: [u8; 3] = slot_id_bytes.try_into().unwrap();
        SlotId::from_bytes(slot_id_array)
    }

    /// Validate slot ID bytes match computed slot ID
    /// 
    /// Ensures slot ID bytes consistency as specified in Hivemind whitepaper
    /// for slot claim validation.
    pub fn validate_slot_id_consistency(slot_id: &SlotId, slot_id_bytes: [u8; 3]) -> Result<(), Error> {
        if slot_id.as_bytes() != slot_id_bytes {
            return Err(Error::InvalidSlotId {
                reason: "Slot ID bytes don't match computed slot ID".to_string(),
            });
        }
        Ok(())
    }

    /// Validate decision slot claim structure
    /// 
    /// Common validation logic for decision slot claims, ensuring
    /// market maker authorization and decision structure validity.
    pub fn validate_decision_structure(
        market_maker_address_bytes: [u8; 20],
        slot_id_bytes: [u8; 3],
        is_standard: bool,
        is_scaled: bool,
        question: &str,
        min: Option<u16>,
        max: Option<u16>,
    ) -> Result<Decision, Error> {
        Decision::new(
            market_maker_address_bytes,
            slot_id_bytes,
            is_standard,
            is_scaled,
            question.to_string(),
            min,
            max,
        )
    }

    /// Comprehensive decision slot claim validation
    /// 
    /// This is the single source of truth for validating decision slot claims
    /// across the entire system. Consolidates all validation logic including:
    /// - Transaction structure validation
    /// - Slot ID consistency checks  
    /// - Market maker authorization
    /// - Decision structure validation
    /// - Slot availability and timing validation
    /// 
    /// # Arguments
    /// * `slots_db` - Slots database interface for claim validation
    /// * `rotxn` - Read-only transaction for database access
    /// * `tx` - Filled transaction containing slot claim data
    /// * `override_height` - Optional height override for validation context
    /// 
    /// # Returns
    /// * `Ok(())` - Valid slot claim meeting all requirements
    /// * `Err(Error)` - Invalid claim with detailed error information
    /// 
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on:
    /// - Decision slot allocation and timing
    /// - Market maker authorization requirements  
    /// - Slot claim validation procedures
    pub fn validate_complete_decision_slot_claim<T>(
        slots_db: &T,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error>
    where
        T: SlotValidationInterface,
    {
        // 1. Validate transaction contains slot claim data
        let claim = tx.claim_decision_slot()
            .ok_or_else(|| Error::InvalidSlotId {
                reason: "Not a decision slot claim transaction".to_string(),
            })?;

        // 2. Parse and validate slot ID from claim bytes
        let slot_id = SlotId::from_bytes(claim.slot_id_bytes)?;

        // 3. Validate slot ID bytes consistency 
        Self::validate_slot_id_consistency(&slot_id, claim.slot_id_bytes)?;

        // 4. Validate market maker authorization (all UTXOs from same maker)
        let market_maker_address = MarketValidator::validate_market_maker_authorization(tx)?;
        let market_maker_address_bytes = market_maker_address.0;

        // 5. Validate decision structure according to Hivemind specification
        let decision = Self::validate_decision_structure(
            market_maker_address_bytes,
            claim.slot_id_bytes,
            claim.is_standard,
            claim.is_scaled,
            &claim.question,
            claim.min,
            claim.max,
        )?;

        // 6. Get current timestamp for temporal validation
        let current_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();
        
        // 7. Get current height for validation context
        let current_height = override_height.or_else(|| slots_db.try_get_height(rotxn).ok().flatten());
        
        // 8. Perform comprehensive slot claim validation
        slots_db.validate_slot_claim(
            rotxn,
            slot_id,
            &decision,
            current_ts,
            current_height
        ).map_err(|e| match e {
            // Convert slot-specific errors to InvalidSlotId for API consistency
            Error::SlotNotAvailable { slot_id: _, reason } => Error::InvalidSlotId { reason },
            Error::SlotAlreadyClaimed { slot_id: _ } => Error::InvalidSlotId { 
                reason: "Slot already claimed".to_string() 
            },
            other => other,
        })?;

        Ok(())
    }
}

/// Market validation utilities
/// 
/// Provides common market validation logic used in transaction processing
/// and state management. Follows Bitcoin Hivemind market specification.
pub struct MarketValidator;

impl MarketValidator {
    /// Validate market maker authorization from transaction UTXOs
    /// 
    /// Ensures all UTXOs belong to the same market maker to prevent collusion,
    /// as specified in the Hivemind whitepaper market maker requirements.
    pub fn validate_market_maker_authorization(tx: &FilledTransaction) -> Result<Address, Error> {
        // Validate that we have at least one input spending a market maker's funds
        if tx.inputs().is_empty() {
            return Err(Error::InvalidSlotId {
                reason: "Transaction must have at least one input".to_string(),
            });
        }

        // Extract market maker address from the first UTXO
        let first_utxo = tx.spent_utxos.first().ok_or_else(|| Error::InvalidSlotId {
            reason: "No spent UTXOs found".to_string(),
        })?;

        let market_maker_address = first_utxo.address;

        // Validate ALL UTXOs belong to the same market maker (prevent collusion)
        for (i, spent_utxo) in tx.spent_utxos.iter().enumerate() {
            if spent_utxo.address != market_maker_address {
                return Err(Error::InvalidSlotId {
                    reason: format!(
                        "All UTXOs must belong to the same market maker. UTXO {} has address {}, expected {}",
                        i, spent_utxo.address, market_maker_address
                    ),
                });
            }
        }

        Ok(market_maker_address)
    }
}

/// Period calculation utilities
/// 
/// Consolidates timestamp and block height to period conversions used
/// across the application. Follows Bitcoin Hivemind temporal specification.
pub struct PeriodCalculator;

impl PeriodCalculator {
    /// Convert block height to testing period for development/testing mode
    /// 
    /// # Arguments
    /// * `block_height` - Current L2 block height
    /// * `testing_blocks_per_period` - Blocks per period in testing mode
    /// 
    /// # Returns
    /// Period index for the given block height
    /// 
    /// # Specification Reference  
    /// Bitcoin Hivemind whitepaper section on temporal mechanics and voting periods
    pub fn block_height_to_testing_period(block_height: u32, testing_blocks_per_period: u32) -> u32 {
        block_height / testing_blocks_per_period
    }

    /// Convert L1 timestamp to production period (quarter-based)
    /// 
    /// # Arguments
    /// * `timestamp` - Unix timestamp from L1 Bitcoin blockchain
    /// 
    /// # Returns
    /// Period index based on quarterly cycles
    /// 
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper section on production temporal mechanics
    pub fn timestamp_to_period(timestamp: u64) -> u32 {
        // Implementation follows Hivemind specification for quarterly periods
        // January 1, 2009 00:00:00 UTC as Bitcoin genesis reference
        const BITCOIN_GENESIS_TIMESTAMP: u64 = 1231006505;
        const SECONDS_PER_QUARTER: u64 = 3600 * 24 * 91; // ~91 days per quarter
        
        if timestamp < BITCOIN_GENESIS_TIMESTAMP {
            return 0;
        }
        
        let elapsed_seconds = timestamp - BITCOIN_GENESIS_TIMESTAMP;
        (elapsed_seconds / SECONDS_PER_QUARTER) as u32
    }

    /// Get human-readable period name for timestamp
    /// 
    /// Provides quarter-year formatting for period display
    pub fn period_to_name(period_index: u32) -> String {
        let year = 2009 + (period_index / 4);
        let quarter = (period_index % 4) + 1;
        format!("Q{} {}", quarter, year)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_id_hex_parsing() {
        // Valid 3-byte slot ID
        let valid_hex = "000102";
        let slot_id = SlotValidator::parse_slot_id_from_hex(valid_hex);
        assert!(slot_id.is_ok());

        // Invalid hex format
        let invalid_hex = "invalid";
        let result = SlotValidator::parse_slot_id_from_hex(invalid_hex);
        assert!(result.is_err());

        // Wrong length
        let wrong_length = "0001";
        let result = SlotValidator::parse_slot_id_from_hex(wrong_length);
        assert!(result.is_err());
    }

    #[test]
    fn test_period_calculations() {
        // Test block height to testing period
        assert_eq!(PeriodCalculator::block_height_to_testing_period(100, 50), 2);
        assert_eq!(PeriodCalculator::block_height_to_testing_period(0, 50), 0);

        // Test period name formatting
        assert_eq!(PeriodCalculator::period_to_name(0), "Q1 2009");
        assert_eq!(PeriodCalculator::period_to_name(4), "Q1 2010");
        assert_eq!(PeriodCalculator::period_to_name(7), "Q4 2010");
    }

    #[test]
    fn test_timestamp_to_period() {
        // Test genesis timestamp
        const GENESIS: u64 = 1231006505;
        assert_eq!(PeriodCalculator::timestamp_to_period(GENESIS), 0);
        
        // Test before genesis
        assert_eq!(PeriodCalculator::timestamp_to_period(0), 0);
        
        // Test period calculation
        const SECONDS_PER_QUARTER: u64 = 3600 * 24 * 91;
        let one_quarter_later = GENESIS + SECONDS_PER_QUARTER;
        assert_eq!(PeriodCalculator::timestamp_to_period(one_quarter_later), 1);
    }

    /// Mock implementation for testing slot validation interface
    struct MockSlotValidator {
        should_pass: bool,
        height: Option<u32>,
    }

    impl SlotValidationInterface for MockSlotValidator {
        fn validate_slot_claim(
            &self,
            _rotxn: &RoTxn,
            _slot_id: SlotId,
            _decision: &Decision,
            _current_ts: u64,
            _current_height: Option<u32>,
        ) -> Result<(), Error> {
            if self.should_pass {
                Ok(())
            } else {
                Err(Error::SlotNotAvailable {
                    slot_id: SlotId::from_bytes([0, 0, 1]).unwrap(),
                    reason: "Test failure".to_string(),
                })
            }
        }

        fn try_get_height(&self, _rotxn: &RoTxn) -> Result<Option<u32>, Error> {
            Ok(self.height)
        }
    }

    #[test]
    fn test_slot_id_consistency_validation() {
        let slot_id_bytes = [0, 0, 1];
        let slot_id = SlotId::from_bytes(slot_id_bytes).unwrap();
        
        // Test valid consistency
        assert!(SlotValidator::validate_slot_id_consistency(&slot_id, slot_id_bytes).is_ok());
        
        // Test invalid consistency
        let wrong_bytes = [0, 0, 2];
        assert!(SlotValidator::validate_slot_id_consistency(&slot_id, wrong_bytes).is_err());
    }

    #[test]
    fn test_consolidated_validation_interface() {
        // Test successful validation path
        let mock_validator = MockSlotValidator {
            should_pass: true,
            height: Some(100),
        };
        
        // We can't easily test the full validation without extensive setup,
        // but we can verify the trait interface works correctly
        assert_eq!(mock_validator.try_get_height(&unsafe { std::mem::zeroed() }).unwrap(), Some(100));
        
        // Test failure case
        let failing_validator = MockSlotValidator {
            should_pass: false,
            height: Some(100),
        };
        
        let slot_id = SlotId::from_bytes([0, 0, 1]).unwrap();
        let decision = Decision::new(
            [0; 20],
            [0, 0, 1],
            true,
            false,
            "Test question".to_string(),
            None,
            None,
        ).unwrap();
        
        assert!(failing_validator.validate_slot_claim(
            &unsafe { std::mem::zeroed() },
            slot_id,
            &decision,
            0,
            Some(100)
        ).is_err());
    }
}