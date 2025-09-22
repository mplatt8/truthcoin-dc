use crate::state::Error;
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, Env, RwTxn, RoTxn};
use std::collections::BTreeSet;
use crate::validation::{PeriodCalculator, SlotValidationInterface};

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Deserialize,
    Serialize,
)]
pub struct SlotId([u8; 3]);

// Compile-time constants for efficient bit operations and bounds checking
const MAX_PERIOD_INDEX: u32 = (1 << 10) - 1; // 1023
const MAX_SLOT_INDEX: u32 = (1 << 14) - 1;   // 16383
const STANDARD_SLOT_MAX: u32 = 499;
const NONSTANDARD_SLOT_MIN: u32 = 500;

// Bit manipulation constants for fast slot ID encoding/decoding
const PERIOD_SHIFT: u32 = 14;
const SLOT_MASK: u32 = MAX_SLOT_INDEX;

impl SlotId {
    pub fn new(period: u32, index: u32) -> Result<Self, Error> {
        if period > MAX_PERIOD_INDEX {
            return Err(Error::InvalidSlotId {
                reason: format!(
                    "Period {} exceeds maximum {}",
                    period, MAX_PERIOD_INDEX
                ),
            });
        }
        if index > MAX_SLOT_INDEX {
            return Err(Error::InvalidSlotId {
                reason: format!(
                    "Slot index {} exceeds maximum {}",
                    index, MAX_SLOT_INDEX
                ),
            });
        }

        // Optimized encode: period (10 bits) || slot_index (14 bits)
        let combined = (period << PERIOD_SHIFT) | index;
        let bytes = [
            (combined >> 16) as u8,
            (combined >> 8) as u8,
            combined as u8,
        ];

        Ok(SlotId(bytes))
    }

    #[inline(always)]
    pub const fn period_index(self) -> u32 {
        // Fast bit manipulation using const operations
        let combined = ((self.0[0] as u32) << 16) | ((self.0[1] as u32) << 8) | (self.0[2] as u32);
        combined >> PERIOD_SHIFT
    }

    #[inline(always)]
    pub const fn slot_index(self) -> u32 {
        // Fast bit manipulation using const operations
        let combined = ((self.0[0] as u32) << 16) | ((self.0[1] as u32) << 8) | (self.0[2] as u32);
        combined & SLOT_MASK
    }

    pub fn as_bytes(self) -> [u8; 3] {
        self.0
    }

    pub fn from_bytes(bytes: [u8; 3]) -> Result<Self, Error> {
        // Optimized bit manipulation without intermediate big-endian conversion
        let combined = ((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | (bytes[2] as u32);
        let period = combined >> PERIOD_SHIFT;
        let index = combined & SLOT_MASK;

        if period > MAX_PERIOD_INDEX {
            return Err(Error::InvalidSlotId {
                reason: format!(
                    "Period {} exceeds maximum {}",
                    period, MAX_PERIOD_INDEX
                ),
            });
        }
        if index > MAX_SLOT_INDEX {
            return Err(Error::InvalidSlotId {
                reason: format!(
                    "Slot index {} exceeds maximum {}",
                    index, MAX_SLOT_INDEX
                ),
            });
        }

        Ok(SlotId(bytes))
    }

    /// Parse slot ID from hex string.
    /// This is the single source of truth for SlotId parsing from hex strings.
    pub fn from_hex(slot_id_hex: &str) -> Result<Self, Error> {
        // Fast path validation for common case (6 hex chars = 3 bytes)
        if slot_id_hex.len() != 6 {
            return Err(Error::InvalidSlotId {
                reason: "Slot ID hex must be exactly 6 characters (3 bytes)".to_string(),
            });
        }
        
        // Manual hex parsing can be faster than hex::decode for small fixed sizes
        let mut slot_id_bytes = [0u8; 3];
        for (i, chunk) in slot_id_hex.as_bytes().chunks_exact(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| Error::InvalidSlotId {
                    reason: "Invalid slot ID hex format".to_string(),
                })?;
            
            slot_id_bytes[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| Error::InvalidSlotId {
                    reason: "Invalid slot ID hex format".to_string(),
                })?;
        }

        Self::from_bytes(slot_id_bytes)
    }

    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
pub struct Decision {
    pub id: [u8; 32],
    pub market_maker_pubkey_hash: [u8; 20],
    pub slot_id_bytes: [u8; 3],
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question: String,
    pub min: Option<u16>,
    pub max: Option<u16>,
}

impl Decision {
    pub fn new(
        market_maker_pubkey_hash: [u8; 20],
        slot_id_bytes: [u8; 3],
        is_standard: bool,
        is_scaled: bool,
        question: String,
        min: Option<u16>,
        max: Option<u16>,
    ) -> Result<Self, Error> {
        if question.as_bytes().len() > 1000 {
            return Err(Error::InvalidSlotId {
                reason: "Question must be 1000 bytes or less".to_string(),
            });
        }

        match (min, max, is_scaled) {
            (Some(min_val), Some(max_val), true) => {
                if min_val >= max_val {
                    return Err(Error::InvalidRange);
                }
            }
            (None, None, false) => {}
            _ => return Err(Error::InconsistentDecisionType),
        }

        let mut decision = Decision {
            id: [0; 32],
            market_maker_pubkey_hash,
            slot_id_bytes,
            is_standard,
            is_scaled,
            question,
            min,
            max,
        };

        decision.id = decision.compute_id_hash();

        Ok(decision)
    }

    fn compute_id_hash(&self) -> [u8; 32] {
        use crate::types::hashes;

        let hash_data = (
            &self.market_maker_pubkey_hash,
            &self.slot_id_bytes,
            self.is_standard,
            self.is_scaled,
            &self.question,
            self.min,
            self.max,
        );

        hashes::hash(&hash_data)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
pub struct Slot {
    pub slot_id: SlotId,
    pub decision: Option<Decision>,
}

// Compile-time slot configuration constants
const FUTURE_PERIODS: u32 = 20;
const SLOTS_DECLINING_RATE: u64 = 25; // Slots decrease by 25 per period  
const INITIAL_SLOTS_PER_PERIOD: u64 = 500;

#[derive(Clone, Debug)]
pub struct SlotConfig {
    pub testing_mode: bool,
    pub testing_blocks_per_period: u32,
}

impl Default for SlotConfig {
    fn default() -> Self {
        Self {
            testing_mode: true,
            testing_blocks_per_period: 1,
        }
    }
}

impl SlotConfig {
    pub fn production() -> Self {
        Self {
            testing_mode: false,
            testing_blocks_per_period: 1,
        }
    }

    pub fn testing(blocks_per_period: u32) -> Self {
        if blocks_per_period == 0 {
            panic!("blocks_per_period must be > 0");
        }
        Self {
            testing_mode: true,
            testing_blocks_per_period: blocks_per_period,
        }
    }
}

// Use PeriodCalculator::timestamp_to_period for consistency
#[inline]
fn timestamp_to_quarter_index(ts_secs: u64) -> Result<u32, Error> {
    Ok(PeriodCalculator::timestamp_to_period(ts_secs))
}

pub fn quarter_to_string(quarter_idx: u32, config: &SlotConfig) -> String {
    if config.testing_mode {
        format!("Testing Period {}", quarter_idx)
    } else {
        let year = quarter_idx / 4;
        let quarter = quarter_idx % 4;
        let quarter_name = match quarter {
            0 => "Q1",
            1 => "Q2",
            2 => "Q3",
            3 => "Q4",
            _ => unreachable!(),
        };
        format!("{} {}", quarter_name, year)
    }
}

fn get_current_period(
    timestamp: u64,
    block_height: Option<u32>,
    config: &SlotConfig,
) -> Result<u32, Error> {
    if config.testing_mode {
        let height = block_height.unwrap_or(0);
        Ok(PeriodCalculator::block_height_to_testing_period(height, config.testing_blocks_per_period))
    } else {
        Ok(PeriodCalculator::timestamp_to_period(timestamp))
    }
}

#[derive(Clone)]
pub struct Dbs {
    period_slots: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<BTreeSet<Slot>>>,
    claimed_slot_ids: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<BTreeSet<SlotId>>>,
    config: SlotConfig,
}

impl Dbs {
    pub const NUM_DBS: u32 = 2;

    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Self::new_with_config(env, rwtxn, SlotConfig::default())
    }

    pub fn new_with_config(
        env: &Env,
        rwtxn: &mut RwTxn<'_>,
        config: SlotConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            period_slots: DatabaseUnique::create(env, rwtxn, "period_slots")?,
            claimed_slot_ids: DatabaseUnique::create(env, rwtxn, "claimed_slot_ids")?,
            config,
        })
    }

    pub fn get_current_period(
        &self,
        ts_secs: u64,
        block_height: Option<u32>,
    ) -> Result<u32, Error> {
        get_current_period(ts_secs, block_height, &self.config)
    }

    fn get_active_window(
        &self,
        ts_secs: u64,
        block_height: Option<u32>,
    ) -> Result<(u32, u32), Error> {
        let current = self.get_current_period(ts_secs, block_height)?;
        Ok((current, current + FUTURE_PERIODS - 1))
    }

    // Optimized declining pattern: 500→475→450→...→25 over 20 future periods
    #[inline]
    const fn calculate_available_slots(
        &self,
        period: u32,
        current_period: u32,
    ) -> u64 {
        // Fast path for historical periods (ossified or in voting)
        if period < current_period {
            return 0;
        }

        let offset = period.saturating_sub(current_period);
        if offset >= FUTURE_PERIODS {
            return 0;
        }

        // Optimized calculation using const arithmetic
        INITIAL_SLOTS_PER_PERIOD.saturating_sub((offset as u64) * SLOTS_DECLINING_RATE)
    }

    pub fn mint_genesis(
        &self,
        rwtxn: &mut RwTxn<'_>,
        ts_secs: u64,
        block_height: u32,
    ) -> Result<(), Error> {
        let current_period =
            self.get_current_period(ts_secs, Some(block_height))?;
        self.mint_periods_up_to(rwtxn, current_period + FUTURE_PERIODS - 1)?;
        Ok(())
    }

    pub fn mint_up_to(
        &self,
        rwtxn: &mut RwTxn<'_>,
        ts_secs: u64,
        block_height: u32,
    ) -> Result<(), Error> {
        let current_period =
            self.get_current_period(ts_secs, Some(block_height))?;
        let target_period = current_period + FUTURE_PERIODS - 1;
        self.mint_periods_up_to(rwtxn, target_period)?;
        Ok(())
    }

    fn mint_periods_up_to(
        &self,
        rwtxn: &mut RwTxn<'_>,
        target_period: u32,
    ) -> Result<(), Error> {
        let mut highest_existing = 0u32;
        {
            let mut iter = self.period_slots.iter(rwtxn)?;
            while let Some((period, _)) = iter.next()? {
                if period > highest_existing {
                    highest_existing = period;
                }
            }
        }

        for period in (highest_existing + 1)..=target_period {
            if self.period_slots.try_get(rwtxn, &period)?.is_none() {
                let empty_period: BTreeSet<Slot> = BTreeSet::new();
                self.period_slots.put(rwtxn, &period, &empty_period)?;
                
                // Initialize claimed_slot_ids for this period as well
                let empty_claimed: BTreeSet<SlotId> = BTreeSet::new();
                self.claimed_slot_ids.put(rwtxn, &period, &empty_claimed)?;
            }
        }

        Ok(())
    }

    pub fn total_for(
        &self,
        _rotxn: &sneed::RoTxn,
        period: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<u64, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        Ok(self.calculate_available_slots(period, current_period))
    }

    pub fn get_active_periods(
        &self,
        _rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<(u32, u64)>, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        let mut periods = Vec::new();

        for period in current_period..=(current_period + FUTURE_PERIODS - 1) {
            let slots = self.calculate_available_slots(period, current_period);
            if slots > 0 {
                periods.push((period, slots));
            }
        }

        Ok(periods)
    }

    pub fn timestamp_to_quarter(ts_secs: u64) -> Result<u32, Error> {
        timestamp_to_quarter_index(ts_secs)
    }

    pub fn is_testing_mode(&self) -> bool {
        self.config.testing_mode
    }

    pub fn get_testing_blocks_per_period(&self) -> u32 {
        self.config.testing_blocks_per_period
    }

    pub fn block_height_to_testing_period(&self, block_height: u32) -> u32 {
        PeriodCalculator::block_height_to_testing_period(block_height, self.config.testing_blocks_per_period)
    }

    pub fn get_config(&self) -> &SlotConfig {
        &self.config
    }

    pub fn quarter_to_string(&self, quarter_idx: u32) -> String {
        quarter_to_string(quarter_idx, &self.config)
    }

    pub fn is_period_ossified(
        &self,
        slot_period: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        let current_period = self
            .get_current_period(current_ts, current_height)
            .unwrap_or(0);
        slot_period < current_period.saturating_sub(4)
    }

    pub fn is_period_in_voting(
        &self,
        slot_period: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        let current_period = self
            .get_current_period(current_ts, current_height)
            .unwrap_or(0);
        slot_period < current_period
            && slot_period >= current_period.saturating_sub(4)
    }

    pub fn is_slot_ossified(
        &self,
        slot_id: SlotId,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        // A slot is ossified when its voting period has ended
        // (more than 4 periods old)
        let period = slot_id.period_index();
        let current_period = self
            .get_current_period(current_ts, current_height)
            .unwrap_or(0);
        period < current_period.saturating_sub(4)
    }

    pub fn is_slot_in_voting(
        &self,
        slot_id: SlotId,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        self.is_period_in_voting(
            slot_id.period_index(),
            current_ts,
            current_height,
        )
    }

    pub fn get_ossified_slots(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<Slot>, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        let ossified_cutoff = current_period.saturating_sub(4);
        
        let mut ossified_slots = Vec::new();
        
        let mut iter = self.period_slots.iter(rotxn)?;
        while let Some((period, slots)) = iter.next()? {
            if period < ossified_cutoff {
                // These slots are ossified (voting has ended)
                ossified_slots.extend(slots.iter().cloned());
            }
        }
        
        Ok(ossified_slots)
    }

    /// Validate that a slot can be claimed without actually claiming it
    /// This is the single source of truth for slot claim validation
    pub fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        let period_index = slot_id.period_index();
        let slot_index = slot_id.slot_index();
        let current_period =
            self.get_current_period(current_ts, current_height)?;

        if self.is_slot_ossified(slot_id, current_ts, current_height) {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Slot period {} is ossified (voting has ended)",
                    period_index
                ),
            });
        }

        if self.is_slot_in_voting(slot_id, current_ts, current_height) {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Period {} is in voting period - no new slots can be claimed",
                    period_index
                ),
            });
        }

        if period_index > current_period + FUTURE_PERIODS - 1 {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Slot period {} exceeds maximum allowed period {} (current + 20)",
                    period_index,
                    current_period + FUTURE_PERIODS - 1
                ),
            });
        }

        if decision.is_standard {
            if slot_index > STANDARD_SLOT_MAX {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Standard slots must have index <= {}, got {}",
                        STANDARD_SLOT_MAX, slot_index
                    ),
                });
            }

            let (start_period, end_period) =
                self.get_active_window(current_ts, current_height)?;
            if period_index < start_period || period_index > end_period {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Period {} is not in active window for new slots ({}-{})",
                        period_index, start_period, end_period
                    ),
                });
            }

            let total_slots = self.total_for(
                rotxn,
                period_index,
                current_ts,
                current_height,
            )?;
            if slot_index as u64 >= total_slots {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Standard slot index {} exceeds available slots {} for period {}",
                        slot_index, total_slots, period_index
                    ),
                });
            }
        } else {
            if slot_index < NONSTANDARD_SLOT_MIN {
                return Err(Error::SlotNotAvailable {
                    slot_id,
                    reason: format!(
                        "Non-standard slots must have index >= {}, got {}",
                        NONSTANDARD_SLOT_MIN, slot_index
                    ),
                });
            }
        }

        // Check if slot is already claimed using O(log n) BTreeSet lookup
        // This is a significant performance improvement over the previous O(n) linear search
        let claimed_slots = self
            .claimed_slot_ids
            .try_get(rotxn, &period_index)?
            .unwrap_or_default();

        if claimed_slots.contains(&slot_id) {
            return Err(Error::SlotAlreadyClaimed { slot_id });
        }

        Ok(())
    }

    pub fn claim_slot(
        &self,
        rwtxn: &mut RwTxn<'_>,
        slot_id: SlotId,
        decision: Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        // Use the single source of truth for validation
        self.validate_slot_claim(rwtxn, slot_id, &decision, current_ts, current_height)?;

        // Now perform the actual claim using optimized BTreeSet operations
        let period_index = slot_id.period_index();
        
        // Update period_slots with BTreeSet::insert (O(log n))
        let mut period_slots = self
            .period_slots
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        let new_slot = Slot {
            slot_id,
            decision: Some(decision),
        };
        period_slots.insert(new_slot);
        self.period_slots.put(rwtxn, &period_index, &period_slots)?;

        // Update claimed_slot_ids for O(log n) lookups
        let mut claimed_slots = self
            .claimed_slot_ids
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();
        claimed_slots.insert(slot_id);
        self.claimed_slot_ids.put(rwtxn, &period_index, &claimed_slots)?;

        Ok(())
    }

    pub fn get_slot(
        &self,
        rotxn: &sneed::RoTxn,
        slot_id: SlotId,
    ) -> Result<Option<Slot>, Error> {
        let period = slot_id.period_index();
        if let Some(period_slots) = self.period_slots.try_get(rotxn, &period)? {
            // Use BTreeSet's efficient search instead of linear iteration
            // Create a dummy slot for binary search
            let search_slot = Slot {
                slot_id,
                decision: None,
            };
            
            // BTreeSet search is O(log n) vs O(n) linear search
            if let Some(found_slot) = period_slots.get(&search_slot) {
                return Ok(Some(found_slot.clone()));
            }
            
            // If exact match not found, try to find slot with matching ID but different decision
            // This handles the case where the slot exists but has a decision
            for slot in period_slots.range(search_slot..) {
                if slot.slot_id == slot_id {
                    return Ok(Some(slot.clone()));
                }
                // Early termination since BTreeSet is ordered
                if slot.slot_id > slot_id {
                    break;
                }
            }
        }
        Ok(None)
    }

    pub fn get_available_slots_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<SlotId>, Error> {
        let total_slots =
            self.total_for(rotxn, period_index, current_ts, current_height)?;
        
        let max_slot_index =
            std::cmp::min(total_slots, (STANDARD_SLOT_MAX + 1) as u64);
        
        // Pre-allocate with known capacity to avoid reallocations
        let mut available_slots = Vec::with_capacity(max_slot_index as usize);
        
        // Get claimed slots for this period once (O(log n) database lookup)
        let claimed_slots = self
            .claimed_slot_ids
            .try_get(rotxn, &period_index)?
            .unwrap_or_default();
        
        // Efficient availability check using the claimed_slots index
        for slot_index in 0..max_slot_index {
            let slot_id = SlotId::new(period_index, slot_index as u32)?;
            
            // O(log k) lookup where k = claimed slots in period, much faster than O(n) get_slot call
            if !claimed_slots.contains(&slot_id) {
                available_slots.push(slot_id);
            }
        }

        Ok(available_slots)
    }

    pub fn get_claimed_slots_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<Vec<Slot>, Error> {
        let mut claimed_slots = Vec::new();

        if let Some(period_slots) =
            self.period_slots.try_get(rotxn, &period_index)?
        {
            for slot in &period_slots {
                if matches!(slot.decision, Some(_)) {
                    claimed_slots.push(slot.clone());
                }
            }
        }

        Ok(claimed_slots)
    }

    pub fn get_claimed_slot_count_in_period(
        &self,
        rotxn: &sneed::RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        // Use the claimed_slots index for O(1) count lookup instead of O(n) iteration
        let claimed_slots = self
            .claimed_slot_ids
            .try_get(rotxn, &period_index)?
            .unwrap_or_default();
        
        Ok(claimed_slots.len() as u64)
    }

    pub fn get_voting_periods(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<Vec<(u32, u64, u64)>, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        let mut voting_periods = Vec::new();

        let voting_start = current_period.saturating_sub(4);
        let voting_end = current_period;

        for period in voting_start..voting_end {
            let count = self.get_claimed_slot_count_in_period(rotxn, period)?;
            if count > 0 {
                let total_slots = 500u64;
                voting_periods.push((period, count, total_slots));
            }
        }

        voting_periods.sort_by_key(|(period, _, _)| *period);
        Ok(voting_periods)
    }

    pub fn get_period_summary(
        &self,
        rotxn: &sneed::RoTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(Vec<(u32, u64)>, Vec<(u32, u64)>), Error> {
        let active_periods =
            self.get_active_periods(rotxn, current_ts, current_height)?;
        let voting_periods_full =
            self.get_voting_periods(rotxn, current_ts, current_height)?;
        let voting_periods = voting_periods_full
            .into_iter()
            .map(|(period, claimed, _total)| (period, claimed))
            .collect();

        Ok((active_periods, voting_periods))
    }

    pub fn revert_claim_slot(
        &self,
        rwtxn: &mut RwTxn<'_>,
        slot_id: SlotId,
    ) -> Result<(), Error> {
        let period_index = slot_id.period_index();

        let mut period_slots = self
            .period_slots
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        // Find and remove the slot using BTreeSet operations
        let target_slot = period_slots
            .iter()
            .find(|slot| slot.slot_id == slot_id)
            .cloned();

        if let Some(slot_to_remove) = target_slot {
            period_slots.remove(&slot_to_remove);
            
            // Also remove from claimed_slot_ids
            let mut claimed_slots = self
                .claimed_slot_ids
                .try_get(rwtxn, &period_index)?
                .unwrap_or_default();
            claimed_slots.remove(&slot_id);
            
            // Update both databases
            if period_slots.is_empty() {
                self.period_slots.delete(rwtxn, &period_index)?;
            } else {
                self.period_slots.put(rwtxn, &period_index, &period_slots)?;
            }
            
            if claimed_slots.is_empty() {
                self.claimed_slot_ids.delete(rwtxn, &period_index)?;
            } else {
                self.claimed_slot_ids.put(rwtxn, &period_index, &claimed_slots)?;
            }
        } else {
            tracing::debug!(
                "Attempted to revert slot {:?} that wasn't found",
                slot_id
            );
        }

        Ok(())
    }
}

impl SlotValidationInterface for Dbs {
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        // Delegate to the optimized validation method
        self.validate_slot_claim(rotxn, slot_id, decision, current_ts, current_height)
    }

    fn try_get_height(&self, _rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        // Slots database doesn't store height information
        // This will be provided by the caller in most cases
        Ok(None)
    }
}