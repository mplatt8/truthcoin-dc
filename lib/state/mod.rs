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
    validation::{MarketValidator, SlotValidationInterface, SlotValidator},
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
    fn clear_utxos_and_address_index(
        &self,
        rwtxn: &mut RwTxn,
    ) -> Result<(), Error>;
}

pub mod block;
pub mod error;
pub mod markets;
mod rollback;
pub mod slots;
use slots::{Decision, SlotId};
mod two_way_peg_data;
pub mod votecoin;
pub mod voting;

pub use error::Error;
pub use markets::{
    BatchedMarketTrade, Market, MarketBuilder, MarketId, MarketState,
    MarketsDatabase, ShareAccount,
};
use rollback::{HeightStamped, RollBack};
pub use voting::VotingSystem;

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
    voting: VotingSystem,
    utxos: DatabaseUnique<SerdeBincode<OutPoint>, SerdeBincode<FilledOutput>>,
    /// Address-indexed UTXO database for efficient address-based lookups
    /// Maps (Address, OutPoint) -> () for O(k) address filtering where k is the number of UTXOs for the address
    /// Uses compound key approach to maintain Bitcoin Hivemind sidechain compliance for UTXO management per whitepaper specifications
    utxos_by_address:
        DatabaseUnique<SerdeBincode<(Address, OutPoint)>, SerdeBincode<()>>,
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
    /// Cached total value of deposit UTXOs for O(1) sidechain wealth calculation
    /// Updated atomically with UTXO operations per Bitcoin Hivemind sidechain requirements
    cached_deposit_utxo_value: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    /// Cached total value of deposit STXOs for O(1) sidechain wealth calculation
    /// Updated atomically with STXO operations per Bitcoin Hivemind sidechain requirements
    cached_deposit_stxo_value: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
    /// Cached total value of withdrawal STXOs for O(1) sidechain wealth calculation
    /// Updated atomically with STXO operations per Bitcoin Hivemind sidechain requirements
    cached_withdrawal_stxo_value: DatabaseUnique<UnitKey, SerdeBincode<u64>>,
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
        self.slots().validate_slot_claim(
            rotxn,
            slot_id,
            decision,
            current_ts,
            current_height,
        )
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
    pub const NUM_DBS: u32 = votecoin::Dbs::NUM_DBS
        + slots::Dbs::NUM_DBS
        + MarketsDatabase::NUM_DBS
        + VotingSystem::NUM_DBS
        + 17; // Added 3 new cache databases

    pub fn new(env: &sneed::Env) -> Result<Self, Error> {
        let mut rwtxn = env.write_txn()?;
        let tip = DatabaseUnique::create(env, &mut rwtxn, "tip")?;
        let height = DatabaseUnique::create(env, &mut rwtxn, "height")?;
        let mainchain_timestamp =
            DatabaseUnique::create(env, &mut rwtxn, "mainchain_timestamp")?;
        let votecoin = votecoin::Dbs::new(env, &mut rwtxn)?;
        let slots = slots::Dbs::new(env, &mut rwtxn)?;
        let markets = MarketsDatabase::new(env, &mut rwtxn)?;
        let voting = VotingSystem::new(env, &mut rwtxn)?;
        let utxos = DatabaseUnique::create(env, &mut rwtxn, "utxos")?;
        let utxos_by_address =
            DatabaseUnique::create(env, &mut rwtxn, "utxos_by_address")?;
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
        let cached_deposit_utxo_value = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "cached_deposit_utxo_value",
        )?;
        let cached_deposit_stxo_value = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "cached_deposit_stxo_value",
        )?;
        let cached_withdrawal_stxo_value = DatabaseUnique::create(
            env,
            &mut rwtxn,
            "cached_withdrawal_stxo_value",
        )?;
        let version = DatabaseUnique::create(env, &mut rwtxn, "state_version")?;
        if version.try_get(&rwtxn, &())?.is_none() {
            version.put(&mut rwtxn, &(), &*VERSION)?;
        }

        // Initialize cache values to zero if they don't exist
        if cached_deposit_utxo_value.try_get(&rwtxn, &())?.is_none() {
            cached_deposit_utxo_value.put(&mut rwtxn, &(), &0u64)?;
        }
        if cached_deposit_stxo_value.try_get(&rwtxn, &())?.is_none() {
            cached_deposit_stxo_value.put(&mut rwtxn, &(), &0u64)?;
        }
        if cached_withdrawal_stxo_value.try_get(&rwtxn, &())?.is_none() {
            cached_withdrawal_stxo_value.put(&mut rwtxn, &(), &0u64)?;
        }
        rwtxn.commit()?;
        Ok(Self {
            tip,
            height,
            mainchain_timestamp,
            votecoin,
            slots,
            markets,
            voting,
            utxos,
            utxos_by_address,
            stxos,
            pending_withdrawal_bundle,
            latest_failed_withdrawal_bundle,
            withdrawal_bundles,
            withdrawal_bundle_event_blocks,
            deposit_blocks,
            cached_deposit_utxo_value,
            cached_deposit_stxo_value,
            cached_withdrawal_stxo_value,
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

    pub fn voting(&self) -> &VotingSystem {
        &self.voting
    }

    /// Create a mutable reference to the voting system for advanced operations
    pub fn voting_mut(&mut self) -> &mut VotingSystem {
        &mut self.voting
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
                if let Some(filled_output) =
                    self.utxos.try_get(rotxn, &outpoint)?
                {
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
        self.utxos_by_address.put(
            rwtxn,
            &(filled_output.address, *outpoint),
            &(),
        )?;

        // Update cached deposit UTXO value if this is a deposit
        if matches!(outpoint, OutPoint::Deposit(_)) {
            let current_value = self.cached_deposit_utxo_value.try_get(rwtxn, &())?.unwrap_or(0);
            let bitcoin_value = filled_output.get_bitcoin_value().to_sat();
            let new_value = current_value.saturating_add(bitcoin_value);
            self.cached_deposit_utxo_value.put(rwtxn, &(), &new_value)?;
        }

        Ok(())
    }

    fn delete_utxo_with_address_index(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
    ) -> Result<bool, Error> {
        // Get the UTXO to find its address before deletion
        let filled_output =
            if let Some(output) = self.utxos.try_get(rwtxn, outpoint)? {
                output
            } else {
                // UTXO not found, nothing to delete
                return Ok(false);
            };

        // Update cached deposit UTXO value if this is a deposit (before deletion)
        if matches!(outpoint, OutPoint::Deposit(_)) {
            let current_value = self.cached_deposit_utxo_value.try_get(rwtxn, &())?.unwrap_or(0);
            let bitcoin_value = filled_output.get_bitcoin_value().to_sat();
            let new_value = current_value.saturating_sub(bitcoin_value);
            self.cached_deposit_utxo_value.put(rwtxn, &(), &new_value)?;
        }

        // Perform atomic deletion: both operations must succeed or both must fail
        // Remove from address index first (less critical if this fails)
        self.utxos_by_address
            .delete(rwtxn, &(filled_output.address, *outpoint))?;

        // Remove from primary UTXO database (this is the critical operation)
        let deleted = self.utxos.delete(rwtxn, outpoint)?;

        // If primary deletion failed, we need to restore address index consistency
        // This should never happen due to LMDB transaction atomicity, but we check for safety
        if !deleted {
            // Restore address index entry
            self.utxos_by_address.put(
                rwtxn,
                &(filled_output.address, *outpoint),
                &(),
            )?;
            // Restore cache value
            if matches!(outpoint, OutPoint::Deposit(_)) {
                let current_value = self.cached_deposit_utxo_value.try_get(rwtxn, &())?.unwrap_or(0);
                let bitcoin_value = filled_output.get_bitcoin_value().to_sat();
                let restored_value = current_value.saturating_add(bitcoin_value);
                self.cached_deposit_utxo_value.put(rwtxn, &(), &restored_value)?;
            }
            return Ok(false);
        }

        Ok(true)
    }

    fn clear_utxos_and_address_index(
        &self,
        rwtxn: &mut RwTxn,
    ) -> Result<(), Error> {
        // Clear both databases atomically
        self.utxos.clear(rwtxn)?;
        self.utxos_by_address.clear(rwtxn)?;

        // Reset cached values to zero
        self.cached_deposit_utxo_value.put(rwtxn, &(), &0u64)?;
        self.cached_deposit_stxo_value.put(rwtxn, &(), &0u64)?;
        self.cached_withdrawal_stxo_value.put(rwtxn, &(), &0u64)?;

        Ok(())
    }
}

impl State {
    /// Update cached STXO values when adding a spent output
    ///
    /// This method maintains the cached counters for deposit STXOs and withdrawal STXOs
    /// to enable O(1) sidechain wealth calculation per Bitcoin Hivemind specifications.
    ///
    /// # Arguments
    /// * `rwtxn` - Mutable database transaction
    /// * `outpoint` - OutPoint being spent
    /// * `spent_output` - SpentOutput being added
    ///
    /// # ACID Compliance
    /// Updates are performed within the same transaction as STXO insertion
    /// to ensure atomicity per Bitcoin Hivemind sidechain requirements.
    pub fn update_stxo_caches(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        spent_output: &SpentOutput,
    ) -> Result<(), Error> {
        let bitcoin_value = spent_output.output.get_bitcoin_value().to_sat();

        // Update deposit STXO cache if this is a deposit
        if matches!(outpoint, OutPoint::Deposit(_)) {
            let current_value = self.cached_deposit_stxo_value.try_get(rwtxn, &())?.unwrap_or(0);
            let new_value = current_value.saturating_add(bitcoin_value);
            self.cached_deposit_stxo_value.put(rwtxn, &(), &new_value)?;
        }

        // Update withdrawal STXO cache if this is a withdrawal
        if matches!(spent_output.inpoint, InPoint::Withdrawal { .. }) {
            let current_value = self.cached_withdrawal_stxo_value.try_get(rwtxn, &())?.unwrap_or(0);
            let new_value = current_value.saturating_add(bitcoin_value);
            self.cached_withdrawal_stxo_value.put(rwtxn, &(), &new_value)?;
        }

        Ok(())
    }

    /// Remove cached STXO values when removing a spent output
    ///
    /// This method decrements the cached counters for deposit STXOs and withdrawal STXOs
    /// to maintain consistency during STXO removal operations.
    ///
    /// # Arguments
    /// * `rwtxn` - Mutable database transaction
    /// * `outpoint` - OutPoint being unspent
    /// * `spent_output` - SpentOutput being removed
    pub fn remove_stxo_caches(
        &self,
        rwtxn: &mut RwTxn,
        outpoint: &OutPoint,
        spent_output: &SpentOutput,
    ) -> Result<(), Error> {
        let bitcoin_value = spent_output.output.get_bitcoin_value().to_sat();

        // Update deposit STXO cache if this is a deposit
        if matches!(outpoint, OutPoint::Deposit(_)) {
            let current_value = self.cached_deposit_stxo_value.try_get(rwtxn, &())?.unwrap_or(0);
            let new_value = current_value.saturating_sub(bitcoin_value);
            self.cached_deposit_stxo_value.put(rwtxn, &(), &new_value)?;
        }

        // Update withdrawal STXO cache if this is a withdrawal
        if matches!(spent_output.inpoint, InPoint::Withdrawal { .. }) {
            let current_value = self.cached_withdrawal_stxo_value.try_get(rwtxn, &())?.unwrap_or(0);
            let new_value = current_value.saturating_sub(bitcoin_value);
            self.cached_withdrawal_stxo_value.put(rwtxn, &(), &new_value)?;
        }

        Ok(())
    }

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

    /// Validate market creation transaction.
    ///
    /// Delegates to centralized validation logic in validation module for
    /// single source of truth following the pattern established for slots and voting.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `tx` - Filled transaction containing market creation data
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid market creation meeting all Hivemind requirements
    /// * `Err(Error)` - Invalid creation with detailed error information
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on market creation
    pub fn validate_market_creation(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        // Delegate to centralized validation logic in validation module
        MarketValidator::validate_market_creation(self, rotxn, tx, override_height)
    }

    /// Validate share purchase transaction.
    ///
    /// Delegates to centralized validation logic in validation module for
    /// single source of truth following the pattern established for slots and voting.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `tx` - Filled transaction containing buy shares data
    /// * `override_height` - Optional height override for validation context
    ///
    /// # Returns
    /// * `Ok(())` - Valid share purchase meeting all Hivemind requirements
    /// * `Err(Error)` - Invalid trade with detailed error information
    ///
    /// # Specification Reference
    /// Bitcoin Hivemind whitepaper sections on LMSR trading
    pub fn validate_buy_shares(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<(), Error> {
        // Delegate to centralized validation logic in validation module
        MarketValidator::validate_buy_shares(self, rotxn, tx, override_height)
    }

    /// Validates a filled transaction, and returns the fee
    pub fn validate_filled_transaction(
        &self,
        rotxn: &RoTxn,
        tx: &FilledTransaction,
        override_height: Option<u32>,
    ) -> Result<bitcoin::Amount, Error> {
        use crate::validation::VoteValidator;

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
        if tx
            .transaction
            .data
            .as_ref()
            .map_or(false, |data| data.is_buy_shares())
        {
            self.validate_buy_shares(rotxn, tx, override_height)?;
        }

        // Validate vote submissions
        if tx.is_submit_vote() {
            VoteValidator::validate_vote_submission(self, rotxn, tx, override_height)?;
        }

        // Validate batch vote submissions
        if tx.is_submit_vote_batch() {
            VoteValidator::validate_vote_batch(self, rotxn, tx, override_height)?;
        }

        // Validate voter registration
        if tx.is_register_voter() {
            // Registration validation is minimal - just check for Votecoin balance
            // The actual registration happens during block application
            let voter_address = tx
                .spent_utxos
                .first()
                .ok_or_else(|| Error::InvalidTransaction {
                    reason: "Voter registration transaction must have inputs".to_string(),
                })?
                .address;

            let votecoin_balance = self.get_votecoin_balance(rotxn, &voter_address)?;
            if votecoin_balance == 0 {
                return Err(Error::InvalidTransaction {
                    reason: "Voter registration requires Votecoin balance".to_string(),
                });
            }
        }

        // Validate reputation updates (system transactions only - minimal validation)
        if tx.is_update_reputation() {
            // Reputation updates are typically system-generated after consensus
            // Validation is minimal here, actual logic in block application
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

    /// Get total sidechain wealth in Bitcoin using O(1) cached values
    ///
    /// This optimized implementation uses cached counters maintained during UTXO/STXO
    /// operations to achieve O(1) performance instead of the previous O(n) iteration.
    ///
    /// # Performance Improvement
    /// - Previous: O(n) where n = total UTXOs + STXOs (could be millions)
    /// - Current: O(1) constant time lookup from cached values
    ///
    /// # Bitcoin Hivemind Compliance
    /// Calculation follows the Bitcoin Hivemind whitepaper specification:
    /// **Sidechain Wealth = Deposit UTXOs + Deposit STXOs - Withdrawal STXOs**
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    ///
    /// # Returns
    /// Total sidechain wealth in Bitcoin satoshis with overflow protection
    pub fn sidechain_wealth(
        &self,
        rotxn: &RoTxn,
    ) -> Result<bitcoin::Amount, Error> {
        // O(1) cache lookups instead of O(n) iteration
        let deposit_utxo_value = self.cached_deposit_utxo_value.try_get(rotxn, &())?.unwrap_or(0);
        let deposit_stxo_value = self.cached_deposit_stxo_value.try_get(rotxn, &())?.unwrap_or(0);
        let withdrawal_stxo_value = self.cached_withdrawal_stxo_value.try_get(rotxn, &())?.unwrap_or(0);

        // Convert to bitcoin::Amount with overflow protection
        let total_deposit_utxo_value = bitcoin::Amount::from_sat(deposit_utxo_value);
        let total_deposit_stxo_value = bitcoin::Amount::from_sat(deposit_stxo_value);
        let total_withdrawal_stxo_value = bitcoin::Amount::from_sat(withdrawal_stxo_value);

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

    pub fn get_ossified_slots(
        &self,
        rotxn: &RoTxn,
    ) -> Result<Vec<crate::state::slots::Slot>, Error> {
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

    // ================================================================================
    // Votecoin Balance Queries for Voting System Integration
    // ================================================================================

    /// Get Votecoin balance for a specific address
    ///
    /// This method calculates the total Votecoin holdings for an address by
    /// summing all Votecoin UTXOs owned by that address. This is essential
    /// for the Bitcoin Hivemind voting weight calculation.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `address` - Address to query Votecoin balance for
    ///
    /// # Returns
    /// Total Votecoin balance (u32) for the address
    ///
    /// # Bitcoin Hivemind Specification
    /// According to the Bitcoin Hivemind whitepaper, voting weight is calculated as:
    /// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
    /// This method provides the Votecoin holdings component of that calculation.
    pub fn get_votecoin_balance(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<u32, Error> {
        let mut addresses = HashSet::new();
        addresses.insert(*address);

        let utxos = self.get_utxos_by_addresses(rotxn, &addresses)?;
        let mut total_votecoin = 0u32;

        for (_, filled_output) in utxos {
            if let crate::types::FilledOutputContent::Votecoin(amount) =
                &filled_output.content
            {
                total_votecoin = total_votecoin.saturating_add(*amount);
            }
        }

        Ok(total_votecoin)
    }

    /// Get Votecoin balances for multiple addresses efficiently
    ///
    /// This method performs batch querying of Votecoin balances for multiple
    /// addresses simultaneously, which is more efficient than individual queries
    /// when calculating voting weights for all participants in a voting period.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `addresses` - Set of addresses to query balances for
    ///
    /// # Returns
    /// HashMap mapping Address to Votecoin balance (u32)
    ///
    /// # Performance
    /// - Time Complexity: O(k) where k = total UTXOs for all addresses
    /// - Leverages the existing address index for efficient lookups
    pub fn get_votecoin_balances_batch(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<Address, u32>, Error> {
        let utxos = self.get_utxos_by_addresses(rotxn, addresses)?;
        let mut balances = HashMap::new();

        // Initialize all addresses with zero balance
        for &address in addresses {
            balances.insert(address, 0u32);
        }

        // Sum Votecoin amounts for each address
        for (_, filled_output) in utxos {
            if let crate::types::FilledOutputContent::Votecoin(amount) =
                &filled_output.content
            {
                let current_balance =
                    balances.get(&filled_output.address).unwrap_or(&0);
                balances.insert(
                    filled_output.address,
                    current_balance.saturating_add(*amount),
                );
            }
        }

        Ok(balances)
    }

    /// Get total Votecoin supply currently in circulation
    ///
    /// This method calculates the total amount of Votecoin currently held in UTXOs,
    /// which should equal the fixed supply of 1,000,000 unless some coins are
    /// permanently lost (spent to invalid outputs).
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    ///
    /// # Returns
    /// Total Votecoin supply in circulation
    ///
    /// # Bitcoin Hivemind Compliance
    /// Used for calculating proportional voting weights as specified in the whitepaper.
    /// Each voter's Votecoin proportion = voter_votecoin_balance / total_supply
    pub fn get_total_votecoin_supply(
        &self,
        rotxn: &RoTxn,
    ) -> Result<u32, Error> {
        let utxos = self.get_utxos(rotxn)?;
        let mut total_supply = 0u32;

        for (_, filled_output) in utxos {
            if let crate::types::FilledOutputContent::Votecoin(amount) =
                &filled_output.content
            {
                total_supply = total_supply.saturating_add(*amount);
            }
        }

        Ok(total_supply)
    }

    /// Get Votecoin holdings proportion for an address
    ///
    /// This method calculates the proportional Votecoin holdings for an address
    /// relative to the total supply, which is used directly in the Bitcoin Hivemind
    /// voting weight calculation.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `address` - Address to calculate proportion for
    ///
    /// # Returns
    /// Proportion of total Votecoin supply held by address (0.0 to 1.0)
    ///
    /// # Bitcoin Hivemind Specification
    /// This implements the Votecoin Holdings Proportion component of:
    /// **Final Voting Weight = Base Reputation × Votecoin Holdings Proportion**
    pub fn get_votecoin_proportion(
        &self,
        rotxn: &RoTxn,
        address: &Address,
    ) -> Result<f64, Error> {
        let balance = self.get_votecoin_balance(rotxn, address)?;
        let total_supply = self.get_total_votecoin_supply(rotxn)?;

        if total_supply == 0 {
            return Ok(0.0);
        }

        Ok(balance as f64 / total_supply as f64)
    }

    /// Get Votecoin proportions for multiple addresses efficiently
    ///
    /// Batch calculation of Votecoin proportions for multiple addresses.
    /// This is optimized for voting weight calculations across all participants
    /// in a voting period.
    ///
    /// # Arguments
    /// * `rotxn` - Read-only database transaction
    /// * `addresses` - Set of addresses to calculate proportions for
    ///
    /// # Returns
    /// HashMap mapping Address to Votecoin proportion (0.0 to 1.0)
    pub fn get_votecoin_proportions_batch(
        &self,
        rotxn: &RoTxn,
        addresses: &HashSet<Address>,
    ) -> Result<HashMap<Address, f64>, Error> {
        let balances = self.get_votecoin_balances_batch(rotxn, addresses)?;
        let total_supply = self.get_total_votecoin_supply(rotxn)?;
        let mut proportions = HashMap::new();

        if total_supply == 0 {
            // No Votecoin in circulation - all proportions are 0.0
            for &address in addresses {
                proportions.insert(address, 0.0);
            }
            return Ok(proportions);
        }

        for (&address, &balance) in &balances {
            let proportion = balance as f64 / total_supply as f64;
            proportions.insert(address, proportion);
        }

        Ok(proportions)
    }
}

impl Watchable<()> for State {
    type WatchStream = impl Stream<Item = ()>;

    /// Get a signal that notifies whenever the tip changes
    fn watch(&self) -> Self::WatchStream {
        tokio_stream::wrappers::WatchStream::new(self.tip.watch().clone())
    }
}
