use crate::state::Error;
use chrono::{Datelike, TimeZone, Utc};
use fallible_iterator::FallibleIterator;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, Env, RwTxn};

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

const MAX_PERIOD_INDEX: u32 = (1 << 10) - 1;
const MAX_SLOT_INDEX: u32 = (1 << 14) - 1;
const STANDARD_SLOT_MAX: u32 = 499;
const NONSTANDARD_SLOT_MIN: u32 = 500;

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

        // Encode: period (10 bits) || slot_index (14 bits)
        let combined = (period << 14) | index;
        let bytes = [
            ((combined >> 16) & 0xFF) as u8,
            ((combined >> 8) & 0xFF) as u8,
            (combined & 0xFF) as u8,
        ];

        Ok(SlotId(bytes))
    }

    pub fn period_index(self) -> u32 {
        let combined = u32::from_be_bytes([0, self.0[0], self.0[1], self.0[2]]);
        combined >> 14
    }

    pub fn slot_index(self) -> u32 {
        let combined = u32::from_be_bytes([0, self.0[0], self.0[1], self.0[2]]);
        combined & MAX_SLOT_INDEX
    }

    pub fn as_bytes(self) -> [u8; 3] {
        self.0
    }

    pub fn from_bytes(bytes: [u8; 3]) -> Result<Self, Error> {
        let combined = u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]);
        let period = combined >> 14;
        let index = combined & MAX_SLOT_INDEX;

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

    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Slot {
    pub slot_id: SlotId,
    pub decision: Option<Decision>,
}

const FUTURE_PERIODS: u32 = 20;

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

// Convert timestamp to quarter: year*4 + (month0/3)
fn timestamp_to_quarter_index(ts_secs: u64) -> Result<u32, Error> {
    if ts_secs > i64::MAX as u64 {
        return Err(Error::InvalidTimestamp);
    }

    let dt = Utc
        .timestamp_opt(ts_secs as i64, 0)
        .single()
        .ok_or(Error::InvalidTimestamp)?;

    let year = dt.year();
    if year < 0 || year > (u32::MAX / 4) as i32 {
        return Err(Error::TimestampOutOfRange);
    }

    let quarter = dt.month0() / 3;
    Ok((year as u32) * 4 + quarter)
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
        Ok(height / config.testing_blocks_per_period)
    } else {
        timestamp_to_quarter_index(timestamp)
    }
}

#[derive(Clone)]
pub struct Dbs {
    period_slots: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<Vec<Slot>>>,
    config: SlotConfig,
}

impl Dbs {
    pub const NUM_DBS: u32 = 1;

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
            config,
        })
    }

    fn get_current_period(
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

    // Declining pattern: 500→475→450→...→25 over 20 future periods
    fn calculate_available_slots(
        &self,
        period: u32,
        current_period: u32,
    ) -> u64 {
        if period < current_period.saturating_sub(4) {
            return 0;
        }

        if period < current_period && period >= current_period.saturating_sub(4)
        {
            return 0;
        }

        if period < current_period {
            return 0;
        }

        let offset = period - current_period;
        if offset >= FUTURE_PERIODS {
            return 0;
        }

        500u64.saturating_sub(offset as u64 * 25)
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
                let empty_period: Vec<Slot> = Vec::new();
                self.period_slots.put(rwtxn, &period, &empty_period)?;
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
        block_height / self.config.testing_blocks_per_period
    }

    pub fn get_config(&self) -> &SlotConfig {
        &self.config
    }

    pub fn quarter_to_string(&self, quarter_idx: u32) -> String {
        quarter_to_string(quarter_idx, &self.config)
    }

    pub fn is_period_too_old(
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

    pub fn is_slot_too_old(
        &self,
        slot_id: SlotId,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> bool {
        self.is_period_too_old(
            slot_id.period_index(),
            current_ts,
            current_height,
        )
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

    pub fn purge_old_slots(
        &self,
        rwtxn: &mut sneed::RwTxn,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<usize, Error> {
        let current_period =
            self.get_current_period(current_ts, current_height)?;
        let cutoff_period = current_period.saturating_sub(4);
        let voting_start = current_period.saturating_sub(4);
        let voting_end = current_period;

        let mut periods_to_purge = Vec::new();
        let mut total_slots_purged = 0;

        {
            let mut iter = self.period_slots.iter(rwtxn)?;
            while let Some((period, slots)) = iter.next()? {
                let should_purge = if period < cutoff_period {
                    true
                } else if period >= voting_start && period < voting_end {
                    slots.is_empty()
                } else {
                    false
                };

                if should_purge {
                    total_slots_purged += slots.len();
                    periods_to_purge.push(period);
                }
            }
        }

        for period in periods_to_purge {
            self.period_slots.delete(rwtxn, &period)?;
        }

        Ok(total_slots_purged)
    }

    pub fn claim_slot(
        &self,
        rwtxn: &mut RwTxn<'_>,
        slot_id: SlotId,
        decision: Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        let period_index = slot_id.period_index();
        let slot_index = slot_id.slot_index();
        let current_period =
            self.get_current_period(current_ts, current_height)?;

        if self.is_slot_too_old(slot_id, current_ts, current_height) {
            return Err(Error::SlotNotAvailable {
                slot_id,
                reason: format!(
                    "Slot period {} is too old and should be purged",
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
                rwtxn,
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

        let mut period_slots = self
            .period_slots
            .try_get(rwtxn, &period_index)?
            .unwrap_or_default();

        for slot in &period_slots {
            if slot.slot_id == slot_id {
                return Err(Error::SlotAlreadyClaimed { slot_id });
            }
        }

        let new_slot = Slot {
            slot_id,
            decision: Some(decision),
        };
        period_slots.push(new_slot);
        self.period_slots.put(rwtxn, &period_index, &period_slots)?;

        Ok(())
    }

    pub fn get_slot(
        &self,
        rotxn: &sneed::RoTxn,
        slot_id: SlotId,
    ) -> Result<Option<Slot>, Error> {
        let period = slot_id.period_index();
        if let Some(period_slots) = self.period_slots.try_get(rotxn, &period)? {
            for slot in period_slots {
                if slot.slot_id == slot_id {
                    return Ok(Some(slot));
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
        let mut available_slots = Vec::new();

        let max_slot_index =
            std::cmp::min(total_slots, (STANDARD_SLOT_MAX + 1) as u64);
        for slot_index in 0..max_slot_index {
            let slot_id = SlotId::new(period_index, slot_index as u32)?;

            match self.get_slot(rotxn, slot_id)? {
                None => {
                    available_slots.push(slot_id);
                }
                Some(slot) => match slot.decision {
                    Some(_) => {}
                    None => {
                        available_slots.push(slot_id);
                    }
                },
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
            for slot in period_slots {
                if matches!(slot.decision, Some(_)) {
                    claimed_slots.push(slot);
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
        let mut count = 0;

        if let Some(period_slots) =
            self.period_slots.try_get(rotxn, &period_index)?
        {
            for slot in period_slots {
                if matches!(slot.decision, Some(_)) {
                    count += 1;
                }
            }
        }

        Ok(count)
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

        let original_len = period_slots.len();
        period_slots.retain(|slot| slot.slot_id != slot_id);

        if period_slots.len() == original_len {
            tracing::debug!(
                "Attempted to revert slot {:?} that wasn't found",
                slot_id
            );
            return Ok(());
        }

        if period_slots.is_empty() {
            self.period_slots.delete(rwtxn, &period_index)?;
        } else {
            self.period_slots.put(rwtxn, &period_index, &period_slots)?;
        }

        Ok(())
    }
}
