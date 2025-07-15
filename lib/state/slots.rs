// lib/state/slots.rs
use chrono::{DateTime, Utc, TimeZone, Datelike};
use heed::types::SerdeBincode;
use sneed::{Env, RwTxn, DatabaseUnique, UnitKey};
use fallible_iterator::FallibleIterator;
use crate::state::Error;

/// Slots added to *each* tracked quarter every real‐world quarter
const SLOTS_PER_PERIOD: u64 = 25;
/// How many quarters ahead we track (including the current one)
const FUTURE_PERIODS: u32 = 20;

/// Testing mode configuration
const TESTING_MODE: bool = true; // Set to false for production
/// In testing mode, mint slots every N blocks instead of every quarter
const TESTING_BLOCKS_PER_PERIOD: u32 = 1; // Change this to adjust frequency

/// Convert a Unix timestamp (secs) to a monotonic quarter index:
/// e.g. year*4 + (month0/3): Jan–Mar→0, Apr–Jun→1, Jul–Sep→2, Oct–Dec→3
fn quarter_index(ts_secs: u64) -> u32 {
    let dt: DateTime<Utc> = Utc.timestamp_opt(ts_secs as i64, 0).unwrap();
    let year = dt.year() as u32;
    let quarter = dt.month0() / 3; // 0–3
    year * 4 + quarter
}

/// Convert quarter index back to human readable format
pub fn quarter_to_string(quarter_idx: u32) -> String {
    if TESTING_MODE {
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

/// Convert block height to testing period index (for testing mode)
fn block_height_to_period(block_height: u32) -> u32 {
    block_height / TESTING_BLOCKS_PER_PERIOD
}

/// Get current period index based on mode
fn get_current_period(timestamp_or_height: u64, block_height: Option<u32>) -> u32 {
    if TESTING_MODE {
        block_height_to_period(block_height.expect("Block height required in testing mode"))
    } else {
        quarter_index(timestamp_or_height)
    }
}

#[derive(Clone)]
pub struct Dbs {
    /// period_index → total slots
    total: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<u64>>,
    /// last processed period
    last: DatabaseUnique<UnitKey, SerdeBincode<u32>>,
    /// track active window bounds: (start_period, end_period)
    active_window: DatabaseUnique<UnitKey, SerdeBincode<(u32, u32)>>,
    /// track when periods were expired: period → expiry_period
    expired_periods: DatabaseUnique<SerdeBincode<u32>, SerdeBincode<u32>>,
}

impl Dbs {
    pub fn new(env: &Env, rwtxn: &mut RwTxn<'_>) -> Result<Self, Error> {
        Ok(Self {
            total: DatabaseUnique::create(env, rwtxn, "slots_total")?,
            last: DatabaseUnique::create(env, rwtxn, "slots_last")?,
            active_window: DatabaseUnique::create(env, rwtxn, "slots_active_window")?,
            expired_periods: DatabaseUnique::create(env, rwtxn, "slots_expired_periods")?,
        })
    }

    pub fn mint_genesis(&self, rwtxn: &mut RwTxn<'_>, ts_secs: u64, block_height: u32) -> Result<(), Error> {
        // check if genesis has already been minted
        if self.last.try_get(rwtxn, &())?.is_some() {
            return Err(Error::GenesisAlreadyInitialized);
        }
        
        let start_period = get_current_period(ts_secs, Some(block_height));
        
        // Create declining pattern: 500, 475, 450, ..., 25 (total: 5250 slots)
        for offset in 0..FUTURE_PERIODS {
            let period = start_period + offset;
            let slots = 500 - (offset as u64 * 25); // 500, 475, 450, ..., 25
            self.total.put(rwtxn, &period, &slots)?;
        }

        // Set active window bounds
        let end_period = start_period + FUTURE_PERIODS - 1;
        self.active_window.put(rwtxn, &(), &(start_period, end_period))?;

        // record mint through start_period
        self.last.put(rwtxn, &(), &start_period)?;
        Ok(())
    }

    /// Add slots to all existing periods and advance the active window
    pub fn mint_up_to(&self, rwtxn: &mut RwTxn<'_>, ts_secs: u64, block_height: u32) -> Result<(), Error> {
        let curr_period = get_current_period(ts_secs, Some(block_height));
        let last_period = self.last.try_get(rwtxn, &())?.unwrap_or(curr_period);

        for p in (last_period + 1)..=curr_period {
            // FIRST: Shift the active window and mark expired periods
            self.shift_active_window(rwtxn, p)?;
            
            // SECOND: Add slots only to the NEW active periods (not expired ones)
            self.add_slots_to_active_periods(rwtxn, SLOTS_PER_PERIOD)?;

            // THIRD: Clean up old expired periods
            self.cleanup_old_periods(rwtxn, p)?;
        }

        // update last‐processed period
        self.last.put(rwtxn, &(), &curr_period)?;
        Ok(())
    }

    fn add_slots_to_active_periods(&self, rwtxn: &mut RwTxn<'_>, slots_to_add: u64) -> Result<(), Error> {
        // Get the CURRENT active window (after shifting)
        let (start, end) = self.active_window.try_get(rwtxn, &())?.unwrap_or((0, FUTURE_PERIODS - 1));
        
        // Add slots to ALL active periods EXCEPT the newest one (which was just created with base amount)
        for period in start..end { // Note: end is exclusive now
            if let Some(current_slots) = self.total.try_get(rwtxn, &period)? {
                self.total.put(rwtxn, &period, &(current_slots + slots_to_add))?;
            }
        }
        Ok(())
    }
    
    fn shift_active_window(&self, rwtxn: &mut RwTxn<'_>, new_period: u32) -> Result<(), Error> {
        let (current_start, _) = self.active_window.try_get(rwtxn, &())?.unwrap_or((0, FUTURE_PERIODS - 1));
        
        let new_end_period = new_period + FUTURE_PERIODS - 1;
        
        // Create the new end period with 25 slots (this is the initial amount, not additional)
        self.total.put(rwtxn, &new_end_period, &SLOTS_PER_PERIOD)?;
        
        // Mark the oldest active period as expired (will be cleaned up after 4 periods)
        let oldest_active = current_start;
        self.expired_periods.put(rwtxn, &oldest_active, &new_period)?; // expires at new_period
        
        // Update active window bounds
        self.active_window.put(rwtxn, &(), &(new_period, new_end_period))?;
        Ok(())
    }
    
    fn cleanup_old_periods(&self, rwtxn: &mut RwTxn<'_>, current_period: u32) -> Result<(), Error> {
        // Find periods that expired more than 4 periods ago
        let cutoff_period = current_period.saturating_sub(4);
        
        let to_delete = {
            let mut to_delete = Vec::new();
            let mut iter = self.expired_periods.iter(rwtxn)?;
            
            while let Some((period, expired_at)) = iter.next()? {
                if expired_at <= cutoff_period {
                    to_delete.push(period);
                }
            }
            to_delete
        }; // Iterator is dropped here
        
        for period in to_delete {
            self.total.delete(rwtxn, &period)?;
            self.expired_periods.delete(rwtxn, &period)?;
        }
        Ok(())
    }

    /// How many slots have been minted for quarter `q` so far?
    pub fn total_for(&self, rotxn: &sneed::RoTxn, q: u32) -> Result<u64, Error> {
        Ok(self.total.try_get(rotxn, &q)?.unwrap_or(0))
    }

    /// Get only active periods (within the current 20-period window)
    pub fn get_active_periods(&self, rotxn: &sneed::RoTxn) -> Result<Vec<(u32, u64)>, Error> {
        let (start, end) = self.active_window.try_get(rotxn, &())?.unwrap_or((0, FUTURE_PERIODS - 1));
        
        let mut periods = Vec::new();
        for period in start..=end {
            if let Some(slots) = self.total.try_get(rotxn, &period)? {
                periods.push((period, slots));
            }
        }
        Ok(periods)
    }

    /// Convert timestamp to quarter index (public utility function)
    pub fn timestamp_to_quarter(ts_secs: u64) -> u32 {
        quarter_index(ts_secs)
    }

    /// Check if we're in testing mode
    pub fn is_testing_mode() -> bool {
        TESTING_MODE
    }

    /// Get the current testing period interval (blocks per period)
    pub fn get_testing_blocks_per_period() -> u32 {
        TESTING_BLOCKS_PER_PERIOD
    }

    /// Convert block height to testing period (public utility function)
    pub fn block_height_to_testing_period(block_height: u32) -> u32 {
        block_height_to_period(block_height)
    }

    /// Get all quarters that have slots minted (active + expired, for admin/debug)
    pub fn get_all_quarters_with_slots(&self, rotxn: &sneed::RoTxn) -> Result<Vec<(u32, u64)>, Error> {
        let mut quarters = Vec::new();
        let mut iter = self.total.iter(rotxn)?;
        while let Some((quarter, slots)) = iter.next()? {
            if slots > 0 {
                quarters.push((quarter, slots));
            }
        }
        Ok(quarters)
    }

    /// Get expired periods (those marked for cleanup but not yet purged)
    pub fn get_expired_periods(&self, rotxn: &sneed::RoTxn) -> Result<Vec<(u32, u32, u64)>, Error> {
        let mut expired = Vec::new();
        let mut iter = self.expired_periods.iter(rotxn)?;
        while let Some((period, expired_at)) = iter.next()? {
            // Get the slots for this expired period
            let slots = self.total.try_get(rotxn, &period)?.unwrap_or(0);
            expired.push((period, expired_at, slots));
        }
        Ok(expired)
    }

    /// Get slots for a range of quarters
    pub fn get_slots_for_range(&self, rotxn: &sneed::RoTxn, start_quarter: u32, end_quarter: u32) -> Result<Vec<(u32, u64)>, Error> {
        let mut quarters = Vec::new();
        for q in start_quarter..=end_quarter {
            let slots = self.total_for(rotxn, q)?;
            if slots > 0 {
                quarters.push((q, slots));
            }
        }
        Ok(quarters)
    }

    /// Get slots for quarters around a timestamp (±range quarters)
    pub fn get_slots_around_time(&self, rotxn: &sneed::RoTxn, ts_secs: u64, range: u32) -> Result<Vec<(u32, u64)>, Error> {
        let center_quarter = quarter_index(ts_secs);
        let start_quarter = center_quarter.saturating_sub(range);
        let end_quarter = center_quarter + range;
        self.get_slots_for_range(rotxn, start_quarter, end_quarter)
    }
}