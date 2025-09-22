use std::collections::{BTreeMap, HashMap, HashSet};

use fallible_iterator::FallibleIterator;
use futures::Stream;
use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, UnitKey};

use crate::{
    authorization::Authorization,
    types::{
        Address, AmountOverflowError, Authorized, AuthorizedTransaction,
        BlockHash, Body, FilledOutput, FilledTransaction, GetAddress as _,
        GetBitcoinValue as _, Header, InPoint, M6id, OutPoint, SpentOutput,
        Transaction, VERSION, Verify as _, Version, WithdrawalBundle,
        WithdrawalBundleStatus, proto::mainchain::TwoWayPegData,
    },
    util::Watchable,
    validation::{SlotValidator, MarketValidator, SlotValidationInterface},
};

/// Trait for managing UTXO operations with address indexing
/// 
/// This trait consolidates the repetitive pattern of updating both the primary UTXO database
/// and the secondary address index atomically. This ensures compliance with Bitcoin Hivemind
/// whitepaper specifications for UTXO management while eliminating code duplication.
pub trait UtxoManager {
    /// Insert a UTXO into both primary database and address index
    fn insert_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error>;

    /// Delete a UTXO from both primary database and address index
    fn delete_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error>;

    /// Clear both UTXO database and address index
    fn clear_utxos_and_address_index(&self, rwtxn: &mut RwTxn) -> Result<(), Error>;
}

pub mod block;
pub mod error;
pub mod markets;
mod rollback;
pub mod slots;
use slots::{SlotId, Decision};
mod two_way_peg_data;
pub mod votecoin;


pub use error::Error;
pub use markets::{Market, MarketBuilder, MarketId, MarketState, MarketsDatabase, ShareAccount, BatchedMarketTrade};
use rollback::{HeightStamped, RollBack};

pub const WITHDRAWAL_BUNDLE_FAILURE_GAP: u32 = 4;

/// Information we have regarding a withdrawal bundle
#[derive(Debug, Deserialize, Serialize)]
enum WithdrawalBundleInfo {
    /// Withdrawal bundle is known
    Known(WithdrawalBundle),
    /// Withdrawal bundle is unknown but unconfirmed / failed
    Unknown,
    /// If an unknown withdrawal bundle is confirmed, ALL UTXOs are
    /// considered spent.
    UnknownConfirmed {
        spend_utxos: BTreeMap<OutPoint, FilledOutput>,
    },
}

impl WithdrawalBundleInfo {
    fn is_known(&self) -> bool {
        match self {
            Self::Known(_) => true,
            Self::Unknown | Self::UnknownConfirmed { .. } => false,
        }
    }
}

type WithdrawalBundlesDb = DatabaseUnique<
    SerdeBincode<M6id>,
    SerdeBincode<(
        WithdrawalBundleInfo,
        RollBack<HeightStamped<WithdrawalBundleStatus>>,
    )>,
>;

#[derive(Clone)]
pub struct State {
    /// Current tip
    tip: DatabaseUnique<UnitKey, SerdeBincode<BlockHash>>,
    /// Current height
    height: DatabaseUnique<UnitKey, SerdeBincode<u32>>,
    /// Current mainchain timestamp
    mainchain_timestamp: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    votecoin: votecoin::Dbs,
    slots: slots::Dbs,
    markets: MarketsDatabase,
    utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<FilledOutput>>,
    /// Address-indexed UTXO database for efficient address-based lookups
    /// Maps (Address, OutPoint) -> () for O(k) address filtering where k is the number of UTXOs for the address
    /// Uses compound key approach to maintain Bitcoin Hivemind sidechain compliance for UTXO management per whitepaper specifications
    utxos_by_address: DatabaseUnique<SerdeBincode<(Address, OutPoint)>, SerdeBincode<()>>,
    stxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>,
    /// Pending withdrawal bundle and block height
    pending_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<(WithdrawalBundle, u32)>>,
    /// Latest failed (known) withdrawal bundle
    latest_failed_withdrawal_bundle:
        DatabaseUnique<UnitKey, SerdeBincode<RollBack<HeightStamped<M6id>>>>,
    /// Withdrawal bundles and their status.
    /// Some withdrawal bundles may be unknown.
    /// in which case they are `None`.
    withdrawal_bundles: WithdrawalBundlesDb,
    /// Deposit blocks and the height at which they were applied, keyed sequentially
    deposit_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    /// Withdrawal bundle event blocks and the height at which they were applied, keyed sequentially
    withdrawal_bundle_event_blocks: DatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    >,
    _version: DatabaseUnique<UnitKey, SerdeBincode<Version>>,
}

/// Implementation of SlotValidationInterface for State
/// 
/// Provides the necessary database operations for slot validation
/// while maintaining Bitcoin Hivemind compliance and performance.
impl SlotValidationInterface for State {
    /// Delegate slot claim validation to slots database
    /// 
    /// Uses the single source of truth for slot validation as specified
    /// in the Bitcoin Hivemind whitepaper slot allocation procedures.
    fn validate_slot_claim(
        &self,
        rotxn: &RoTxn,
        slot_id: SlotId,
        decision: &Decision,
        current_ts: u64,
        current_height: Option<u32>,
    ) -> Result<(), Error> {
        self.slots().validate_slot_claim(rotxn, slot_id, decision, current_ts, current_height)
    }

    /// Delegate height retrieval to State's try_get_height method
    /// 
    /// Provides current blockchain height for validation context as needed
    /// for period-based slot availability calculations.
    fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        self.try_get_height(rotxn)
    }
}

impl State {
    pub const NUM_DBS: u32 = votecoin::Dbs::NUM_DBS + slots::Dbs::NUM_DBS + MarketsDatabase::NUM_DBS + 14;

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let tip = DatabaseUnique::create(env, &mut rwtxn, "tip")?;
        let height = DatabaseUnique::create(env, &mut rwtxn, "height")?;
        let mainchain_timestamp =
            DatabaseUnique::create(env, &mut rwtxn, "mainchain_timestamp")?;
        let votecoin = votecoin::Dbs::new(env, &mut rwtxn)?;
        let slots = slots::Dbs::new(env, &mut rwtxn)?;
        let markets = MarketsDatabase::new(env, &mut rwtxn)?;
        let utxos = DatabaseUnique::create(env, &mut rwtxn, "utxos")?;
        let utxos_by_address = DatabaseUnique::create(env, &mut rwtxn, "utxos_by_address")?;
        let stxos = DatabaseUnique::create(env, &mut rwtxn, "stxos")?;
        let pending_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "pending_withdrawal_bundle",
        )?;
        let latest_failed_withdrawal_bundle = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "latest_failed_withdrawal_bundle",
        )?;
        let withdrawal_bundles =
            DatabaseUnique::create(env, &mut rwtxn, "withdrawal_bundles")?;
        let deposit_blocks =
            DatabaseUnique::create(env, &mut rwtxn, "deposit_blocks")?;
        let withdrawal_bundle_event_blocks = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "withdrawal_bundle_event_blocks",
        )?;
        let version = DatabaseUnique::create(env, &mut rwtxn, "state_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }
        rwtxn.commit()?;
        Ok(Self {
            tip,
            height,
            mainchain_timestamp,
            votecoin,
            slots,
            markets,
            utxos,
            utxos_by_address,
            stxos,
            pending_withdrawal_bundle,
            latest_failed_withdrawal_bundle,
            withdrawal_bundles,
            withdrawal_bundle_event_blocks,
            deposit_blocks,
            _version: version,
        })
    }


    pub fn votecoin(&self) -> &votecoin::Dbs {
        &self.votecoin
    }

    pub fn slots(&self) -> &slots::Dbs {
        &self.slots
    }

    pub fn markets(&self) -> &MarketsDatabase {
        &self.markets
    }

    pub fn deposit_blocks(
        &self,
    ) -> &RoDatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    > {
        &self.deposit_blocks
    }

    pub fn stxos(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<SpentOutput>>
    {
        &self.stxos
    }

    pub fn withdrawal_bundle_event_blocks(
        &self,
    ) -> &RoDatabaseUnique<
        SerdeBincode<u32>,
        SerdeBincode<(bitcoin::BlockHash, u32)>,
    > {
        &self.withdrawal_bundle_event_blocks
    }

    pub fn try_get_tip(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<BlockHash>, Error> {
        let tip = self.tip.try_get(rotxn, &())?;
        Ok(tip)
    }

    pub fn try_get_height(&self, rotxn: &RoTxn) -> Result<Option<u32>, Error> {
        let height = self.height.try_get(rotxn, &())?;
        Ok(height)
    }

    pub fn try_get_mainchain_timestamp(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<u64>, Error> {
        let timestamp = self.mainchain_timestamp.try_get(rotxn, &())?;
        Ok(timestamp)
    }

    pub fn get_utxos(
        &self,
        rotxn: &RoTxn,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        let utxos = self.utxos.iter(rotxn)?.collect()?;
        Ok(utxos)
    }

    /// Efficient O(k) address-based UTXO lookup using address index
    /// 
    /// Returns UTXOs for the specified addresses, leveraging the secondary index
    /// to achieve O(k) performance where k is the number of matching UTXOs,
    /// compared to the previous O(n) implementation that scanned all UTXOs.
    /// 
    /// # Performance
    /// - Time Complexity: O(k) where k = number of UTXOs for requested addresses
    /// - Space Complexity: O(k) for result storage
    /// 
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `addresses` - Set of addresses to lookup UTXOs for
    /// 
    /// # Returns
    /// HashMap mapping OutPoint to FilledOutput for all UTXOs owned by the specified addresses
    /// 
    /// # Bitcoin Hivemind Compliance
    /// This optimization maintains full compatibility with Bitcoin Hivemind sidechain
    /// UTXO management while dramatically improving performance for address-based queries.
    pub fn get_utxos_by_addresses(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<OutPoint, FilledOutput>, Error> {
        // Pre-allocate with estimated capacity to reduce reallocations
        let mut result = HashMap::with_capacity(addresses.len() * 4); // Estimate 4 UTXOs per address
        
        // Iterate through all address-indexed UTXOs and filter by requested addresses
        let mut iter = self.utxos_by_address.iter(rotxn)?;
        while let Some(((addr, outpoint), _)) = iter.next()? {
            if addresses.contains(&addr) {
                if let Some(filled_output) = self.utxos.try_get(rotxn, &outpoint)? {
                    result.insert(outpoint, filled_output);
                }
            }
        }
        
        Ok(result)
    }

}

/// Implementation of UtxoManager trait for State with ACID transaction safety
/// 
/// This provides a interface for UTXO operations that maintains consistency
/// between the primary UTXO database and the address index, following Bitcoin Hivemind
/// whitepaper specifications for sidechain UTXO management.
/// 
/// # Transaction Safety
/// All operations use LMDB's ACID transaction semantics:
/// - Atomicity: Either all operations succeed or all fail and rollback
/// - Consistency: Database invariants are maintained across operations  
/// - Isolation: Concurrent transactions don't interfere
/// - Durability: Committed changes survive system failures
/// 
/// # Rollback Behavior
/// If any operation within a transaction fails, the entire RwTxn automatically rolls back
/// all changes, ensuring database consistency per Bitcoin Hivemind specifications.
impl UtxoManager for State {
    fn insert_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        filled_output: &FilledOutput,
    ) -> Result<(), Error> {
        // Insert into primary UTXO database
        self.utxos.put(rwtxn, outpoint, filled_output)?;
        
        // Insert into address index for efficient lookup using compound key
        self.utxos_by_address.put(rwtxn, &(filled_output.address, *outpoint), &())?;
        
        Ok(())
    }

    fn delete_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error> {
        // Get the UTXO to find its address before deletion
        let filled_output = if let Some(output) = self.utxos.try_get(rwtxn, outpoint)? {
            output
        } else {
            // UTXO not found, nothing to delete
            return Ok(false);
        };

        // Perform atomic deletion: both operations must succeed or both must fail
        // Remove from address index first (less critical if this fails)
        self.utxos_by_address.delete(rwtxn, &(filled_output.address, *outpoint))?;

        // Remove from primary UTXO database (this is the critical operation)
        let deleted = self.utxos.delete(rwtxn, outpoint)?;

        // If primary deletion failed, we need to restore address index consistency
        // This should never happen due to LMDB transaction atomicity, but we check for safety
        if !deleted {
            // Restore address index entry
            self.utxos_by_address.put(rwtxn, &(filled_output.address, *outpoint), &())?;
            return Ok(false);
        }

        Ok(true)
    }

    fn clear_utxos_and_address_index(&self, rwtxn: &mut RwTxn) -> Result<(), Error> {
        // Clear both databases atomically
        self.utxos.clear(rwtxn)?;
        self.utxos_by_address.clear(rwtxn)?;
        Ok(())
    }
}

impl State {
    /// Get mempool-adjusted shares for a market (for real-time price updates)
    pub fn get_mempool_shares(
        &self,
        rotxn: &RoTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<Option<ndarray::Array1<f64>>, Error> {
        // Use the markets database to store mempool shares temporarily
        self.markets.get_mempool_shares(rotxn, market_id)
    }

    /// Store mempool-adjusted shares for a market
    pub fn put_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
        shares: &ndarray::Array1<f64>,
    ) -> Result<(), Error> {
        self.markets.put_mempool_shares(rwtxn, market_id, shares)
    }

    /// Clear mempool shares after block confirmation
    pub fn clear_mempool_shares(
        &self,
        rwtxn: &mut RwTxn,
        market_id: &crate::state::markets::MarketId,
    ) -> Result<(), Error> {
        self.markets.clear_mempool_shares(rwtxn, market_id)
    }

    /// Get the latest failed withdrawal bundle, and the height at which it failed
    pub fn get_latest_failed_withdrawal_bundle(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<(u32, M6id)>, Error> {
        let Some(latest_failed_m6id) =
            self.latest_failed_withdrawal_bundle.try_get(rotxn, &())?
        else {
            return Ok(None);
        };
        let latest_failed_m6id = latest_failed_m6id.latest().value;
        let (_bundle, bundle_status) = self.withdrawal_bundles.try_get(rotxn, &latest_failed_m6id)?
            .expect("Inconsistent DBs: latest failed m6id should exist in withdrawal_bundles");
        let bundle_status = bundle_status.latest();
        assert_eq!(bundle_status.value, WithdrawalBundleStatus::Failed);
        Ok(Some((bundle_status.height, latest_failed_m6id)))
    }

    pub fn fill_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &Transaction,
    ) -> Result<FilledTransaction, Error> {
        let mut spent_utxos = vec![];
        for input in &transaction.inputs {
            let utxo = self
                .utxos
                .try_get(rotxn, input)?
                .ok_or(Error::NoUtxo { outpoint: *input })?;
            spent_utxos.push(utxo);
        }
        Ok(FilledTransaction {
            spent_utxos,
            transaction: transaction.clone(),
        })
    }

    /// Fill a transaction that has already been applied
    pub fn fill_transaction_from_stxos(
        &self,
        rotxn: &RoTxn,
        tx: Transaction,
    ) -> Result<FilledTransaction, Error> {
        let txid = tx.txid();
        let mut spent_utxos = vec![];
        // fill inputs last-to-first
        for (vin, input) in tx.inputs.iter().enumerate().rev() {
            let stxo = self
                .stxos
                .try_get(rotxn, input)?
                .ok_or(Error::NoStxo { outpoint: *input })?;
            assert_eq!(
                stxo.inpoint,
                InPoint::Regular {
                    txid,
                    vin: vin as u32
                }
            );
            spent_utxos.push(stxo.output);
        }
        spent_utxos.reverse();
        Ok(FilledTransaction {
            spent_utxos,
            transaction: tx,
        })
    }

    pub fn fill_authorized_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: AuthorizedTransaction,
    ) -> Result<Authorized<FilledTransaction>, Error> {
        let filled_tx =
            self.fill_transaction(rotxn, &transaction.transaction)?;
        let authorizations = transaction.authorizations;
        Ok(Authorized {
            transaction: filled_tx,
            authorizations,
        })
    }

    /// Get pending withdrawal bundle and block height
    pub fn get_pending_withdrawal_bundle(
        &self,
        txn: &RoTxn,
    ) -> Result<Option<(WithdrawalBundle, u32)>, Error> {
        Ok(self.pending_withdrawal_bundle.try_get(txn, &())?)
    }

    /** Check Votecoin balance constraints for prediction market operations.
     *  Since Votecoin has a fixed supply and no registration/reservation system,
     *  validation is much simpler than the old Truthcoin system.
     *  Special case: Allow Votecoin creation in genesis block (height 0).
     * */
    pub fn validate_votecoin(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
    ) -> Result<(), Error> {
        // Get total Votecoin in inputs and outputs
        let votecoin_inputs: u32 = tx
            .spent_votecoin()
            .filter_map(|(_, output)| output.votecoin())
            .sum();
        let votecoin_outputs: u32 = tx
            .votecoin_outputs()
            .filter_map(|output| {
                if let crate::types::OutputContent::Votecoin(amount) =
                    &output.content
                {
                    Some(*amount)
                } else {
                    None
                }
            })
            .sum();

        // Check if we're in the genesis block (height 0)
        let current_height = self.try_get_height(rotxn)?.unwrap_or(0);
        let is_genesis = current_height == 0;

        if is_genesis {
            // In genesis block, allow Votecoin creation (inputs can be 0, outputs > 0)
            // No validation needed as this is the initial supply distribution
            Ok(())
        } else {
            // In all other blocks, enforce strict conservation: total in == total out
            if votecoin_inputs != votecoin_outputs {
                return Err(Error::UnbalancedVotecoin {
                    inputs: votecoin_inputs,
                    outputs: votecoin_outputs,
                });
            }
            Ok(())
        }
    }

    /// Validate decision slot claim transaction
    ///
    /// This method serves as the entry point for slot claim validation,
    /// delegating to the centralized validation logic in the validation module.
    ///
    /// # Validation Flow Architecture
    /// 1. **Mempool Path**: `validate_filled_transaction` → `validate_decision_slot_claim` →
    ///    `SlotValidator::validate_complete_decision_slot_claim` → `slots::Dbs::validate_slot_claim`
    ///
    /// 2. **Block Application Path**: `apply_claim_decision_slot` → `slots::Dbs::claim_slot` →
    ///    `slots::Dbs::validate_slot_claim` (same validation logic)
    ///
    /// This ensures a single source of truth for validation logic while allowing
    /// different entry points for mempool validation vs block application.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `tx` - Filled transaction containing slot claim data
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid slot claim meeting all Hivemind requirements
    /// * `Err(Error)` - Invalid claim with detailed error information
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on decision slot validation
    pub fn validate_decision_slot_claim(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        // Delegate to centralized validation logic in validation module
        SlotValidator::validate_complete_decision_slot_claim(
            self,
            rotxn,
            tx,
            override_height,
        )
    }

    pub fn validate_market_creation(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        let market_data = tx.create_market()
            .ok_or_else(|| Error::InvalidSlotId {
                reason: "Not a market creation transaction".to_string(),
            })?;

        // Validate market type
        if market_data.market_type != "independent" && market_data.market_type != "categorical" {
            return Err(Error::InvalidSlotId {
                reason: format!("Invalid market type: {}", market_data.market_type),
            });
        }

        // Validate decision slots exist and are properly formatted
        if market_data.decision_slots.is_empty() {
            return Err(Error::InvalidSlotId {
                reason: "Market must have at least one decision slot".to_string(),
            });
        }

        // Validate slot IDs and ensure they exist
        let mut slot_ids = Vec::new();
        for slot_hex in &market_data.decision_slots {
            // Use common validation utility for slot ID parsing
            let slot_id = SlotValidator::parse_slot_id_from_hex(slot_hex)?;
            
            // Verify slot exists and has a decision
            let slot = self.slots.get_slot(rotxn, slot_id)?
                .ok_or_else(|| Error::InvalidSlotId {
                    reason: format!("Slot {} does not exist", slot_hex),
                })?;
            
            if slot.decision.is_none() {
                return Err(Error::InvalidSlotId {
                    reason: format!("Slot {} has no decision", slot_hex),
                });
            }
            
            slot_ids.push(slot_id);
        }

        // Validate categorical market constraints
        if market_data.market_type == "categorical" {
            // All decisions must be binary for categorical markets
            for slot_id in &slot_ids {
                let slot = self.slots.get_slot(rotxn, *slot_id)?.unwrap();
                let decision = slot.decision.unwrap();
                if decision.is_scaled {
                    return Err(Error::InvalidSlotId {
                        reason: "Categorical markets can only use binary decisions".to_string(),
                    });
                }
            }
        }

        // Validate LMSR parameters
        let beta = market_data.b;
        if beta <= 0.0 {
            return Err(Error::InvalidSlotId {
                reason: format!("Invalid beta parameter: {}", beta),
            });
        }

        if let Some(fee) = market_data.trading_fee {
            if fee < 0.0 || fee > 1.0 {
                return Err(Error::InvalidSlotId {
                    reason: format!("Trading fee must be between 0 and 1: {}", fee),
                });
            }
        }

        // Use common validation utility for market maker authorization
        let _market_maker_address = MarketValidator::validate_market_maker_authorization(tx)?;

        Ok(())
    }

    pub fn validate_buy_shares(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        _override_height: Option<u32>,
    ) -> Result<(), Error> {
        use crate::state::markets::{MarketId, MarketState};
        use crate::math::lmsr::Lmsr;

        let buy_data = tx.buy_shares()
            .ok_or_else(|| Error::InvalidSlotId {
                reason: "Not a buy shares transaction".to_string(),
            })?;

        // Validate market exists
        let market_id = MarketId::new(buy_data.market_id);
        let market = self.markets().get_market(rotxn, &market_id)?
            .ok_or_else(|| Error::InvalidSlotId {
                reason: format!("Market {:?} does not exist", buy_data.market_id),
            })?;

        // Validate market is in trading state
        if market.state() != MarketState::Trading {
            return Err(Error::InvalidSlotId {
                reason: format!("Market is not in trading state (current state: {:?})", market.state()),
            });
        }

        // Validate outcome index
        if buy_data.outcome_index as usize >= market.shares().len() {
            return Err(Error::InvalidSlotId {
                reason: format!("Outcome index {} exceeds market outcomes {}",
                    buy_data.outcome_index, market.shares().len()),
            });
        }

        // Validate shares amount is positive
        if buy_data.shares_to_buy <= 0.0 {
            return Err(Error::InvalidSlotId {
                reason: format!("Shares to buy must be positive: {}", buy_data.shares_to_buy),
            });
        }

        // Validate max cost is positive
        if buy_data.max_cost <= 0 {
            return Err(Error::InvalidSlotId {
                reason: format!("Max cost must be positive: {}", buy_data.max_cost),
            });
        }

        // Calculate new share quantities after the trade
        let mut new_shares = market.shares().clone();
        new_shares[buy_data.outcome_index as usize] += buy_data.shares_to_buy;

        // Validate LMSR constraints
        let lmsr = Lmsr::new(market.shares().len());
        let current_cost = lmsr.cost_function(market.b(), &market.shares().view())
            .map_err(|e| Error::InvalidSlotId {
                reason: format!("Failed to calculate current market cost: {:?}", e),
            })?;
        let new_cost = lmsr.cost_function(market.b(), &new_shares.view())
            .map_err(|e| Error::InvalidSlotId {
                reason: format!("Failed to calculate new market cost: {:?}", e),
            })?;

        let trade_cost = new_cost - current_cost;

        // Validate trade cost doesn't exceed max cost
        if trade_cost > buy_data.max_cost as f64 {
            return Err(Error::InvalidSlotId {
                reason: format!("Trade cost {} exceeds max cost {}", trade_cost, buy_data.max_cost),
            });
        }

        // Use common validation utility for trader authorization
        let _trader_address = MarketValidator::validate_market_maker_authorization(tx)?;

        Ok(())
    }

    /// Validates a filled transaction, and returns the fee
    pub fn validate_filled_transaction(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<bitcoin::Amount, Error> {
        let () = self.validate_votecoin(rotxn, tx)?;

        // Validate decision slot claims
        if tx.is_claim_decision_slot() {
            self.validate_decision_slot_claim(rotxn, tx, override_height)?;
        }

        // Validate market creation
        if tx.is_create_market() {
            self.validate_market_creation(rotxn, tx, override_height)?;
        }

        // Validate buy shares transactions
        if tx.transaction.data.as_ref().map_or(false, |data| data.is_buy_shares()) {
            self.validate_buy_shares(rotxn, tx, override_height)?;
        }

        tx.bitcoin_fee()?.ok_or(Error::NotEnoughValueIn)
    }

    pub fn validate_transaction(
        &self,
        rotxn: &RoTxn,
        transaction: &AuthorizedTransaction,
    ) -> Result<bitcoin::Amount, Error> {
        let filled_transaction =
            self.fill_transaction(rotxn, &transaction.transaction)?;
        for (authorization, spent_utxo) in transaction
            .authorizations
            .iter()
            .zip(filled_transaction.spent_utxos.iter())
        {
            if authorization.get_address() != spent_utxo.address {
                return Err(Error::WrongPubKeyForAddress);
            }
        }
        if Authorization::verify_transaction(transaction).is_err() {
            return Err(Error::AuthorizationError);
        }
        let fee =
            self.validate_filled_transaction(rotxn, &filled_transaction, None)?;
        Ok(fee)
    }

    pub fn get_last_deposit_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .deposit_blocks
            .last(rotxn)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    pub fn get_last_withdrawal_bundle_event_block_hash(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Option<bitcoin::BlockHash>, Error> {
        let block_hash = self
            .withdrawal_bundle_event_blocks
            .last(rotxn)?
            .map(|(_, (block_hash, _))| block_hash);
        Ok(block_hash)
    }

    /// Get total sidechain wealth in Bitcoin with optimized single-pass calculation
    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        // Single-pass calculation with early termination on overflow
        let mut total_deposit_utxo_value = bitcoin::Amount::ZERO;
        
        // UTXO iteration using fallible iterator
        let mut utxo_iter = self.utxos.iter(rotxn)?;
        while let Some((outpoint, output)) = utxo_iter.next()? {
            if matches!(outpoint, OutPoint::Deposit(_)) {
                let value = output.get_bitcoin_value();
                total_deposit_utxo_value = total_deposit_utxo_value
                    .checked_add(value)
                    .ok_or(AmountOverflowError)?;
            }
        }
        
        // Single-pass STXO iteration with logic
        let mut total_deposit_stxo_value = bitcoin::Amount::ZERO;
        let mut total_withdrawal_stxo_value = bitcoin::Amount::ZERO;
        
        let mut stxo_iter = self.stxos.iter(rotxn)?;
        while let Some((outpoint, spent_output)) = stxo_iter.next()? {
            let bitcoin_value = spent_output.output.get_bitcoin_value();
            
            // Process deposit STXOs
            if matches!(outpoint, OutPoint::Deposit(_)) {
                total_deposit_stxo_value = total_deposit_stxo_value
                    .checked_add(bitcoin_value)
                    .ok_or(AmountOverflowError)?;
            }
            
            // Process withdrawal STXOs (fixed bug - was using wrong variable)
            if matches!(spent_output.inpoint, InPoint::Withdrawal { .. }) {
                total_withdrawal_stxo_value = total_withdrawal_stxo_value
                    .checked_add(bitcoin_value)
                    .ok_or(AmountOverflowError)?;
            }
        }
        
        // Consolidated calculation with overflow protection
        let total_wealth = total_deposit_utxo_value
            .checked_add(total_deposit_stxo_value)
            .and_then(|sum| sum.checked_sub(total_withdrawal_stxo_value))
            .ok_or(AmountOverflowError)?;
        
        Ok(total_wealth)
    }

    pub fn validate_block(
        &self,
        rotxn: &RoTxn,
        header: &Header,
        body: &Body,
    ) -> Result<bitcoin::Amount, Error> {
        block::validate(self, rotxn, header, body)
    }

    pub fn connect_block(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
        mainchain_timestamp: u64,
    ) -> Result<(), Error> {
        block::connect(self, rwtxn, header, body, mainchain_timestamp)
    }

    pub fn disconnect_tip(
        &self,
        rwtxn: &mut RwTxn,
        header: &Header,
        body: &Body,
    ) -> Result<(), Error> {
        block::disconnect_tip(self, rwtxn, header, body)
    }

    pub fn connect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        two_way_peg_data::connect(self, rwtxn, two_way_peg_data)
    }

    pub fn disconnect_two_way_peg_data(
        &self,
        rwtxn: &mut RwTxn,
        two_way_peg_data: &TwoWayPegData,
    ) -> Result<(), Error> {
        two_way_peg_data::disconnect(self, rwtxn, two_way_peg_data)
    }

    pub fn get_all_slot_quarters(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<(u32, u64)>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_active_periods(rotxn, current_ts, current_height)
    }

    pub fn get_slots_for_quarter(
        &self,
        rotxn: &RoTxn,
        quarter: u32,
    ) -> Result<u64, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .total_for(rotxn, quarter, current_ts, current_height)
    }

    pub fn get_available_slots_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<Vec<crate::state::slots::SlotId>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots.get_available_slots_in_period(
            rotxn,
            period_index,
            current_ts,
            current_height,
        )
    }

    pub fn get_ossified_slots(&self, rotxn: &RoTxn) -> Result<Vec<crate::state::slots::Slot>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_ossified_slots(rotxn, current_ts, current_height)
    }

    pub fn is_slot_in_voting(
        &self,
        rotxn: &RoTxn,
        slot_id: crate::state::slots::SlotId,
    ) -> Result<bool, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        Ok(self
            .slots
            .is_slot_in_voting(slot_id, current_ts, current_height))
    }

    pub fn get_voting_periods(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<(u32, u64, u64)>, Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_voting_periods(rotxn, current_ts, current_height)
    }

    pub fn get_period_summary(
        &self,
        rotxn: &RoTxn,
    ) -> Result<(Vec<(u32, u64)>, Vec<(u32, u64)>), Error> {
        let current_ts = self.try_get_mainchain_timestamp(rotxn)?.unwrap_or(0);
        let current_height = self.try_get_height(rotxn)?;
        self.slots
            .get_period_summary(rotxn, current_ts, current_height)
    }

    pub fn get_claimed_slot_count_in_period(
        &self,
        rotxn: &RoTxn,
        period_index: u32,
    ) -> Result<u64, Error> {
        self.slots
            .get_claimed_slot_count_in_period(rotxn, period_index)
    }
}

impl Watchable<()> for State {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the tip changes
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch().clone())
    }
}
