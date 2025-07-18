//! Functions and types related to Votecoin

use std::net::{SocketAddrV4, SocketAddrV6};

use heed::types::SerdeBincode;
use serde::{Deserialize, Serialize};
use sneed::{DatabaseUnique, RoDatabaseUnique, RoTxn, RwTxn, db, env};

use crate::{
    state::{
        error::Error,
        rollback::{RollBack, TxidStamped},
    },
    types::{EncryptionPubKey, FilledTransaction, Hash, Txid, VerifyingKey},
};

/// Update actions for Votecoin network data
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum VotecoinUpdate<T> {
    /// Delete the value
    Delete,
    /// Keep the existing value unchanged
    Retain,
    /// Set a new value
    Set(T),
}

/// Votecoin network data updates
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VotecoinDataUpdates {
    /// Update to commitment
    pub commitment: VotecoinUpdate<Hash>,
    /// Update to IPv4 socket address
    pub socket_addr_v4: VotecoinUpdate<SocketAddrV4>,
    /// Update to IPv6 socket address
    pub socket_addr_v6: VotecoinUpdate<SocketAddrV6>,
    /// Update to encryption public key
    pub encryption_pubkey: VotecoinUpdate<EncryptionPubKey>,
    /// Update to signing public key
    pub signing_pubkey: VotecoinUpdate<VerifyingKey>,
}

/// Votecoin network data (no supply tracking - Votecoin has fixed supply)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VotecoinNetworkData {
    /// Commitment to arbitrary data
    pub commitment: Option<Hash>,
    /// Optional ipv4 addr
    pub socket_addr_v4: Option<SocketAddrV4>,
    /// Optional ipv6 addr
    pub socket_addr_v6: Option<SocketAddrV6>,
    /// Optional pubkey used for encryption
    pub encryption_pubkey: Option<EncryptionPubKey>,
    /// Optional pubkey used for signing messages
    pub signing_pubkey: Option<VerifyingKey>,
}

/// Votecoin ID for identifying Votecoin network data
#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct VotecoinId(pub Hash);

/// Representation of Votecoin network data that supports rollbacks.
/// Votecoin has a fixed supply of 1,000,000 units, so no supply tracking is needed.
/// This only tracks network properties for Votecoin transactions.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VotecoinData {
    /// Commitment to arbitrary data
    pub(in crate::state) commitment: RollBack<TxidStamped<Option<Hash>>>,
    /// Optional ipv4 addr
    pub(in crate::state) socket_addr_v4:
        RollBack<TxidStamped<Option<SocketAddrV4>>>,
    /// Optional ipv6 addr
    pub(in crate::state) socket_addr_v6:
        RollBack<TxidStamped<Option<SocketAddrV6>>>,
    /// Optional pubkey used for encryption
    pub(in crate::state) encryption_pubkey:
        RollBack<TxidStamped<Option<EncryptionPubKey>>>,
    /// Optional pubkey used for signing messages
    pub(in crate::state) signing_pubkey:
        RollBack<TxidStamped<Option<VerifyingKey>>>,
}

impl VotecoinData {
    // Initialize from network data (no supply tracking needed for Votecoin)
    pub(in crate::state) fn init(
        network_data: VotecoinNetworkData,
        txid: Txid,
        height: u32,
    ) -> Self {
        Self {
            commitment: RollBack::<TxidStamped<_>>::new(
                network_data.commitment,
                txid,
                height,
            ),
            socket_addr_v4: RollBack::<TxidStamped<_>>::new(
                network_data.socket_addr_v4,
                txid,
                height,
            ),
            socket_addr_v6: RollBack::<TxidStamped<_>>::new(
                network_data.socket_addr_v6,
                txid,
                height,
            ),
            encryption_pubkey: RollBack::<TxidStamped<_>>::new(
                network_data.encryption_pubkey,
                txid,
                height,
            ),
            signing_pubkey: RollBack::<TxidStamped<_>>::new(
                network_data.signing_pubkey,
                txid,
                height,
            ),
        }
    }

    // Apply network data updates for Votecoin
    pub(in crate::state) fn apply_updates(
        &mut self,
        updates: VotecoinDataUpdates,
        txid: Txid,
        height: u32,
    ) {
        let Self {
            commitment,
            socket_addr_v4,
            socket_addr_v6,
            encryption_pubkey,
            signing_pubkey,
        } = self;

        // apply an update to a single data field
        fn apply_field_update<T>(
            data_field: &mut RollBack<TxidStamped<Option<T>>>,
            update: VotecoinUpdate<T>,
            txid: Txid,
            height: u32,
        ) {
            match update {
                VotecoinUpdate::Delete => data_field.push(None, txid, height),
                VotecoinUpdate::Retain => (),
                VotecoinUpdate::Set(value) => {
                    data_field.push(Some(value), txid, height)
                }
            }
        }
        apply_field_update(commitment, updates.commitment, txid, height);
        apply_field_update(
            socket_addr_v4,
            updates.socket_addr_v4,
            txid,
            height,
        );
        apply_field_update(
            socket_addr_v6,
            updates.socket_addr_v6,
            txid,
            height,
        );
        apply_field_update(
            encryption_pubkey,
            updates.encryption_pubkey,
            txid,
            height,
        );
        apply_field_update(
            signing_pubkey,
            updates.signing_pubkey,
            txid,
            height,
        );
    }

    // Revert network data updates for Votecoin
    pub(in crate::state) fn revert_updates(
        &mut self,
        updates: VotecoinDataUpdates,
        txid: Txid,
        height: u32,
    ) {
        // apply an update to a single data field
        fn revert_field_update<T>(
            data_field: &mut RollBack<TxidStamped<Option<T>>>,
            update: VotecoinUpdate<T>,
            txid: Txid,
            height: u32,
        ) where
            T: std::fmt::Debug + Eq,
        {
            match update {
                VotecoinUpdate::Delete => {
                    let popped = data_field.pop();
                    assert!(popped.is_some());
                    let popped = popped.unwrap();
                    assert!(popped.data.is_none());
                    assert_eq!(popped.txid, txid);
                    assert_eq!(popped.height, height)
                }
                VotecoinUpdate::Retain => (),
                VotecoinUpdate::Set(value) => {
                    let popped = data_field.pop();
                    assert!(popped.is_some());
                    let popped = popped.unwrap();
                    assert!(popped.data.is_some());
                    assert_eq!(popped.data.unwrap(), value);
                    assert_eq!(popped.txid, txid);
                    assert_eq!(popped.height, height)
                }
            }
        }

        let Self {
            commitment,
            socket_addr_v4,
            socket_addr_v6,
            encryption_pubkey,
            signing_pubkey,
        } = self;
        revert_field_update(
            signing_pubkey,
            updates.signing_pubkey,
            txid,
            height,
        );
        revert_field_update(
            encryption_pubkey,
            updates.encryption_pubkey,
            txid,
            height,
        );
        revert_field_update(
            socket_addr_v6,
            updates.socket_addr_v6,
            txid,
            height,
        );
        revert_field_update(
            socket_addr_v4,
            updates.socket_addr_v4,
            txid,
            height,
        );
        revert_field_update(commitment, updates.commitment, txid, height);
    }

    /** Returns the Votecoin network data as it was, at the specified block height.
     *  If a value was updated several times in the block, returns the
     *  last value seen in the block.
     *  Returns `None` if the data did not exist at the specified block
     *  height. */
    pub fn at_block_height(&self, height: u32) -> Option<VotecoinNetworkData> {
        Some(VotecoinNetworkData {
            commitment: self.commitment.at_block_height(height)?.data,
            socket_addr_v4: self.socket_addr_v4.at_block_height(height)?.data,
            socket_addr_v6: self.socket_addr_v6.at_block_height(height)?.data,
            encryption_pubkey: self
                .encryption_pubkey
                .at_block_height(height)?
                .data,
            signing_pubkey: self.signing_pubkey.at_block_height(height)?.data,
        })
    }

    /// Get the current Votecoin network data
    pub fn current(&self) -> VotecoinNetworkData {
        VotecoinNetworkData {
            commitment: self.commitment.latest().data,
            socket_addr_v4: self.socket_addr_v4.latest().data,
            socket_addr_v6: self.socket_addr_v6.latest().data,
            encryption_pubkey: self.encryption_pubkey.latest().data,
            signing_pubkey: self.signing_pubkey.latest().data,
        }
    }
}

/// Votecoin sequence ID
#[derive(
    utoipa::ToSchema,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SeqId(pub u32);

/// Votecoin databases
/// Note: No registration or reservation functionality since Votecoin has fixed supply
#[derive(Clone)]
pub struct Dbs {
    /// Associates Votecoin IDs with Votecoin network data
    votecoin:
        DatabaseUnique<SerdeBincode<VotecoinId>, SerdeBincode<VotecoinData>>,
    /// Associates Votecoin sequence numbers with Votecoin IDs
    seq_to_votecoin:
        DatabaseUnique<SerdeBincode<SeqId>, SerdeBincode<VotecoinId>>,
}

impl Dbs {
    pub const NUM_DBS: u32 = 2;

    /// Create / Open DBs. Does not commit the RwTxn.
    pub(in crate::state) fn new(
        env: &sneed::Env,
        rwtxn: &mut RwTxn,
    ) -> Result<Self, env::error::CreateDb> {
        let votecoin = DatabaseUnique::create(env, rwtxn, "votecoin")?;
        let seq_to_votecoin =
            DatabaseUnique::create(env, rwtxn, "votecoin_seq_to_id")?;
        Ok(Self {
            seq_to_votecoin,
            votecoin,
        })
    }

    pub fn votecoin(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<VotecoinId>, SerdeBincode<VotecoinData>>
    {
        &self.votecoin
    }

    pub fn seq_to_votecoin(
        &self,
    ) -> &RoDatabaseUnique<SerdeBincode<SeqId>, SerdeBincode<VotecoinId>> {
        &self.seq_to_votecoin
    }

    /// Return the Votecoin network data, if it exists
    pub fn try_get_votecoin(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
    ) -> Result<Option<VotecoinData>, db::error::TryGet> {
        self.votecoin.try_get(rotxn, votecoin_id)
    }

    /// Return the Votecoin network data. Returns an error if it does not exist.
    pub fn get_votecoin(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
    ) -> Result<VotecoinData, Error> {
        self.try_get_votecoin(rotxn, votecoin_id)?.ok_or(
            Error::UnbalancedVotecoin {
                inputs: 0,
                outputs: 0,
            },
        )
    }

    /// Resolve Votecoin network data at the specified block height, if it exists.
    pub fn try_get_votecoin_data_at_block_height(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
        height: u32,
    ) -> Result<Option<VotecoinNetworkData>, db::error::TryGet> {
        let res = self
            .votecoin
            .try_get(rotxn, votecoin_id)?
            .and_then(|votecoin_data| votecoin_data.at_block_height(height));
        Ok(res)
    }

    /// Resolve current Votecoin network data, if it exists
    pub fn try_get_current_votecoin_data(
        &self,
        rotxn: &RoTxn,
        votecoin_id: &VotecoinId,
    ) -> Result<Option<VotecoinNetworkData>, Error> {
        let res = self
            .votecoin
            .try_get(rotxn, votecoin_id)?
            .map(|votecoin_data| votecoin_data.current());
        Ok(res)
    }

    /// Apply Votecoin network data updates (no supply changes since it's fixed)
    pub(in crate::state) fn apply_updates(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        votecoin_updates: VotecoinDataUpdates,
        height: u32,
    ) -> Result<(), Error> {
        // Get a Votecoin from the transaction to identify which network data to update
        let votecoin_amount = filled_tx
            .spent_votecoin()
            .next()
            .and_then(|(_, output)| output.votecoin())
            .ok_or(Error::UnbalancedVotecoin {
                inputs: 0,
                outputs: 0,
            })?;

        // Use the Votecoin amount to create a deterministic ID for network data
        let votecoin_id =
            VotecoinId(crate::types::hashes::hash(&votecoin_amount));

        let mut votecoin_data = self
            .votecoin
            .try_get(rwtxn, &votecoin_id)?
            .unwrap_or_else(|| {
                VotecoinData::init(
                    VotecoinNetworkData {
                        commitment: None,
                        socket_addr_v4: None,
                        socket_addr_v6: None,
                        encryption_pubkey: None,
                        signing_pubkey: None,
                    },
                    filled_tx.txid(),
                    height,
                )
            });

        votecoin_data.apply_updates(votecoin_updates, filled_tx.txid(), height);
        self.votecoin.put(rwtxn, &votecoin_id, &votecoin_data)?;
        Ok(())
    }

    /// Revert Votecoin network data updates
    pub(in crate::state) fn revert_updates(
        &self,
        rwtxn: &mut RwTxn,
        filled_tx: &FilledTransaction,
        votecoin_updates: VotecoinDataUpdates,
        height: u32,
    ) -> Result<(), Error> {
        // Get a Votecoin from the transaction to identify which network data to revert
        let votecoin_amount = filled_tx
            .spent_votecoin()
            .next()
            .and_then(|(_, output)| output.votecoin())
            .ok_or(Error::UnbalancedVotecoin {
                inputs: 0,
                outputs: 0,
            })?;

        // Use the Votecoin amount to create a deterministic ID for network data
        let votecoin_id =
            VotecoinId(crate::types::hashes::hash(&votecoin_amount));

        let mut votecoin_data = self
            .votecoin
            .try_get(rwtxn, &votecoin_id)?
            .ok_or(Error::UnbalancedVotecoin {
                inputs: 0,
                outputs: 0,
            })?;

        votecoin_data.revert_updates(
            votecoin_updates,
            filled_tx.txid(),
            height,
        );
        self.votecoin.put(rwtxn, &votecoin_id, &votecoin_data)?;
        Ok(())
    }
}
