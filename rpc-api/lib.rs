//! RPC API

use std::net::SocketAddr;

use fraction::Fraction;
use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use l2l_openapi::open_api;

use serde::{Deserialize, Serialize};
use truthcoin_dc::{
    authorization::{Dst, Signature},
    net::{Peer, PeerConnectionStatus},
    state::AmmPoolState,
    types::{
        Address, AssetId, Authorization, BitcoinOutputContent, Block,
        BlockHash, Body, EncryptionPubKey, FilledOutputContent, Header,
        MerkleRoot, OutPoint, Output, OutputContent, PointedOutput,
        Transaction, TxData, TxIn, Txid, VerifyingKey, WithdrawalBundle,
        WithdrawalOutputContent, schema as truthcoin_schema,
    },
    wallet::Balance,
};
use utoipa::ToSchema;

mod schema;
#[cfg(test)]
mod test;

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct TxInfo {
    pub confirmations: Option<u32>,
    pub fee_sats: u64,
    pub txin: Option<TxIn>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotInfo {
    pub period: u32,
    pub slots: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotDetails {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub content: SlotContentInfo,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub enum SlotContentInfo {
    Empty,
    Decision(DecisionInfo),
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionInfo {
    pub id: String,
    pub market_maker_pubkey_hash: String,
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question: String,
    pub min: Option<u16>,
    pub max: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct ClaimedSlotSummary {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub market_maker_pubkey_hash: String,
    pub is_standard: bool,
    pub is_scaled: bool,
    pub question_preview: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VotingPeriodInfo {
    pub period: u32,
    pub claimed_slots: u64,
    pub total_slots: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct AvailableSlotId {
    pub period_index: u32,
    pub slot_index: u32,
    pub slot_id_hex: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SlotStatus {
    pub is_testing_mode: bool,
    pub blocks_per_period: u32,
    pub current_period: u32,
    pub current_period_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OssifiedSlotInfo {
    pub slot_id_hex: String,
    pub period_index: u32,
    pub slot_index: u32,
    pub decision: Option<DecisionInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketInfo {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcomes: Vec<String>,
    pub current_prices: Vec<f64>,
    pub expires_at: Option<u64>,
    pub volume: f64,
    pub state: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketOutcome {
    pub name: String,
    pub current_price: f64,
    pub probability: f64,
    pub volume: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketDetails {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcomes: Vec<MarketOutcome>,
    pub market_maker: String,
    pub expiry: Option<u64>,
    pub liquidity: f64,
    pub trading_state: String,
    pub beta: f64,
    pub trading_fee: f64,
    pub tags: Vec<String>,
    pub created_at_height: u64,
    pub treasury: f64,
    pub decision_slots: Vec<String>,
}

#[open_api(ref_schemas[
    truthcoin_schema::BitcoinAddr, truthcoin_schema::BitcoinBlockHash,
    truthcoin_schema::BitcoinTransaction, truthcoin_schema::BitcoinOutPoint,
    truthcoin_schema::SocketAddr, Address, AssetId, Authorization,
    BitcoinOutputContent, BlockHash, Body, EncryptionPubKey,
    FilledOutputContent, Header, MarketDetails, MarketInfo, MarketOutcome,
    MerkleRoot, OutPoint, Output, OutputContent,
    PeerConnectionStatus, Signature, SlotInfo, SlotStatus, Transaction, TxData, Txid, TxIn,
    VotingPeriodInfo, WithdrawalOutputContent, VerifyingKey,
])]
#[rpc(client, server)]
pub trait Rpc {
    /// Burn an AMM position
    #[method(name = "amm_burn")]
    async fn amm_burn(
        &self,
        asset0: AssetId,
        asset1: AssetId,
        lp_token_amount: u64,
    ) -> RpcResult<Txid>;

    /// Mint an AMM position
    #[method(name = "amm_mint")]
    async fn amm_mint(
        &self,
        asset0: AssetId,
        asset1: AssetId,
        amount0: u64,
        amount1: u64,
    ) -> RpcResult<Txid>;

    /// Returns the amount of `asset_receive` to receive
    #[method(name = "amm_swap")]
    async fn amm_swap(
        &self,
        asset_spend: AssetId,
        asset_receive: AssetId,
        amount_spend: u64,
    ) -> RpcResult<u64>;

    /// Balance in sats
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "bitcoin_balance")]
    async fn bitcoin_balance(&self) -> RpcResult<Balance>;

    /// Deposit to address
    #[open_api_method(output_schema(PartialSchema = "schema::BitcoinTxid"))]
    #[method(name = "create_deposit")]
    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid>;

    /// Connect to a peer
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "connect_peer")]
    async fn connect_peer(
        &self,
        #[open_api_method_arg(schema(
            ToSchema = "truthcoin_schema::SocketAddr"
        ))]
        addr: SocketAddr,
    ) -> RpcResult<()>;

    /// Decrypt a message with the specified encryption key corresponding to
    /// the specified encryption pubkey.
    /// Returns a decrypted hex string.
    #[method(name = "decrypt_msg")]
    async fn decrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        ciphertext: String,
    ) -> RpcResult<String>;

    /// Encrypt a message to the specified encryption pubkey
    /// Returns the ciphertext as a hex string.
    #[method(name = "encrypt_msg")]
    async fn encrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        msg: String,
    ) -> RpcResult<String>;

    /// Format a deposit address
    #[method(name = "format_deposit_address")]
    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String>;

    /// Generate a mnemonic seed phrase
    #[method(name = "generate_mnemonic")]
    async fn generate_mnemonic(&self) -> RpcResult<String>;

    /// Get the state of the specified AMM pool
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_amm_pool_state")]
    async fn get_amm_pool_state(
        &self,
        asset0: AssetId,
        asset1: AssetId,
    ) -> RpcResult<AmmPoolState>;

    /// Get the current price for the specified pair
    #[open_api_method(output_schema(
        PartialSchema = "schema::Optional<schema::Fraction>"
    ))]
    #[method(name = "get_amm_price")]
    async fn get_amm_price(
        &self,
        base: AssetId,
        quote: AssetId,
    ) -> RpcResult<Option<Fraction>>;

    /// Get block data
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_block")]
    async fn get_block(&self, block_hash: BlockHash) -> RpcResult<Block>;

    /// Get mainchain blocks that commit to a specified block hash
    #[open_api_method(output_schema(
        PartialSchema = "truthcoin_schema::BitcoinBlockHash"
    ))]
    #[method(name = "get_bmm_inclusions")]
    async fn get_bmm_inclusions(
        &self,
        block_hash: truthcoin_dc::types::BlockHash,
    ) -> RpcResult<Vec<bitcoin::BlockHash>>;

    /// Get the best mainchain block hash known by Thunder
    #[open_api_method(output_schema(
        PartialSchema = "schema::Optional<truthcoin_schema::BitcoinBlockHash>"
    ))]
    #[method(name = "get_best_mainchain_block_hash")]
    async fn get_best_mainchain_block_hash(
        &self,
    ) -> RpcResult<Option<bitcoin::BlockHash>>;

    /// Get the best sidechain block hash known by Truthcoin
    #[open_api_method(output_schema(
        PartialSchema = "schema::Optional<BlockHash>"
    ))]
    #[method(name = "get_best_sidechain_block_hash")]
    async fn get_best_sidechain_block_hash(
        &self,
    ) -> RpcResult<Option<BlockHash>>;

    /// Get a new address
    #[method(name = "get_new_address")]
    async fn get_new_address(&self) -> RpcResult<Address>;

    /// Get new encryption key
    #[method(name = "get_new_encryption_key")]
    async fn get_new_encryption_key(&self) -> RpcResult<EncryptionPubKey>;

    /// Get new verifying/signing key
    #[method(name = "get_new_verifying_key")]
    async fn get_new_verifying_key(&self) -> RpcResult<VerifyingKey>;

    /// Get transaction by txid
    #[method(name = "get_transaction")]
    async fn get_transaction(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<Transaction>>;

    /// Get information about a transaction in the current chain
    #[method(name = "get_transaction_info")]
    async fn get_transaction_info(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<TxInfo>>;

    /// Get wallet addresses, sorted by base58 encoding
    #[method(name = "get_wallet_addresses")]
    async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>>;

    /// Get wallet UTXOs
    #[method(name = "get_wallet_utxos")]
    async fn get_wallet_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

    /// Get the current block count
    #[method(name = "getblockcount")]
    async fn getblockcount(&self) -> RpcResult<u32>;

    /// Get the height of the latest failed withdrawal bundle
    #[method(name = "latest_failed_withdrawal_bundle_height")]
    async fn latest_failed_withdrawal_bundle_height(
        &self,
    ) -> RpcResult<Option<u32>>;

    /// List peers
    #[method(name = "list_peers")]
    async fn list_peers(&self) -> RpcResult<Vec<Peer>>;

    /// List all UTXOs
    #[open_api_method(output_schema(
        ToSchema = "Vec<PointedOutput<FilledOutputContent>>"
    ))]
    #[method(name = "list_utxos")]
    async fn list_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

    /// Attempt to mine a sidechain block
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "mine")]
    async fn mine(&self, fee: Option<u64>) -> RpcResult<()>;

    /*
    #[method(name = "my_unconfirmed_stxos")]
    async fn my_unconfirmed_stxos(&self) -> RpcResult<Vec<InPoint>>;
    */

    /// List unconfirmed owned UTXOs
    #[method(name = "my_unconfirmed_utxos")]
    async fn my_unconfirmed_utxos(&self) -> RpcResult<Vec<PointedOutput>>;

    /// List owned UTXOs
    #[method(name = "my_utxos")]
    async fn my_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>>;

    /// Get pending withdrawal bundle
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "pending_withdrawal_bundle")]
    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>>;

    /// Get OpenRPC schema
    #[open_api_method(output_schema(ToSchema = "schema::OpenApi"))]
    #[method(name = "openapi_schema")]
    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi>;

    /// Remove a tx from the mempool
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "remove_from_mempool")]
    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()>;

    /// Set the wallet seed from a mnemonic seed phrase
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "set_seed_from_mnemonic")]
    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()>;

    /// Get total sidechain wealth in sats
    #[method(name = "sidechain_wealth")]
    async fn sidechain_wealth_sats(&self) -> RpcResult<u64>;

    /// Sign an arbitrary message with the specified verifying key
    #[method(name = "sign_arbitrary_msg")]
    async fn sign_arbitrary_msg(
        &self,
        verifying_key: VerifyingKey,
        msg: String,
    ) -> RpcResult<Signature>;

    /// Sign an arbitrary message with the secret key for the specified address
    #[method(name = "sign_arbitrary_msg_as_addr")]
    async fn sign_arbitrary_msg_as_addr(
        &self,
        address: Address,
        msg: String,
    ) -> RpcResult<Authorization>;

    /// Stop the node
    #[method(name = "stop")]
    async fn stop(&self);

    /// Transfer funds to the specified address
    #[method(name = "transfer")]
    async fn transfer(
        &self,
        dest: Address,
        value: u64,
        fee: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid>;

    /// Transfer votecoin to the specified address
    #[method(name = "transfer_votecoin")]
    async fn transfer_votecoin(
        &self,
        dest: Address,
        amount: u32,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid>;

    /// Verify a signature on a message against the specified verifying key.
    /// Returns `true` if the signature is valid
    #[method(name = "verify_signature")]
    async fn verify_signature(
        &self,
        signature: Signature,
        verifying_key: VerifyingKey,
        dst: Dst,
        msg: String,
    ) -> RpcResult<bool>;

    /// Initiate a withdrawal to the specified mainchain address
    #[method(name = "withdraw")]
    async fn withdraw(
        &self,
        #[open_api_method_arg(schema(
            PartialSchema = "truthcoin_schema::BitcoinAddr"
        ))]
        mainchain_address: bitcoin::Address<
            bitcoin::address::NetworkUnchecked,
        >,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid>;

    /// Get all available slots by quarter/period
    #[method(name = "slots_list_all")]
    async fn slots_list_all(&self) -> RpcResult<Vec<SlotInfo>>;

    /// Get slots for a specific quarter/period
    #[method(name = "slots_get_quarter")]
    async fn slots_get_quarter(&self, quarter: u32) -> RpcResult<u64>;

    /// Get slot system status and configuration
    #[method(name = "slots_status")]
    async fn slots_status(&self) -> RpcResult<SlotStatus>;

    /// Convert timestamp to quarter/period index
    #[method(name = "timestamp_to_quarter")]
    async fn timestamp_to_quarter(&self, timestamp: u64) -> RpcResult<u32>;

    /// Convert quarter/period index to human readable string
    #[method(name = "quarter_to_string")]
    async fn quarter_to_string(&self, quarter: u32) -> RpcResult<String>;

    /// Convert block height to testing period (testing mode only)
    #[method(name = "block_height_to_testing_period")]
    async fn block_height_to_testing_period(
        &self,
        block_height: u32,
    ) -> RpcResult<u32>;

    /// Claim a decision slot with a new decision/question
    #[method(name = "claim_decision_slot")]
    async fn claim_decision_slot(
        &self,
        period_index: u32,
        slot_index: u32,
        is_standard: bool,
        is_scaled: bool,
        question: String,
        min: Option<u16>,
        max: Option<u16>,
        fee_sats: u64,
    ) -> RpcResult<Txid>;

    /// Get all available (unclaimed) slot IDs in a specific period
    #[method(name = "get_available_slots_in_period")]
    async fn get_available_slots_in_period(
        &self,
        period_index: u32,
    ) -> RpcResult<Vec<AvailableSlotId>>;

    /// Get a specific slot by its ID
    #[method(name = "get_slot_by_id")]
    async fn get_slot_by_id(
        &self,
        slot_id_hex: String,
    ) -> RpcResult<Option<SlotDetails>>;

    /// Get all claimed slots for a specific period
    #[method(name = "get_claimed_slots_in_period")]
    async fn get_claimed_slots_in_period(
        &self,
        period_index: u32,
    ) -> RpcResult<Vec<ClaimedSlotSummary>>;

    /// Get periods currently in voting phase with claimed/total slot counts
    #[method(name = "get_voting_periods")]
    async fn get_voting_periods(&self) -> RpcResult<Vec<VotingPeriodInfo>>;

    /// Check if a slot is in voting period
    #[method(name = "is_slot_in_voting")]
    async fn is_slot_in_voting(&self, slot_id_hex: String) -> RpcResult<bool>;

    /// Get ossified slots (slots whose voting period has ended)
    #[method(name = "get_ossified_slots")]
    async fn get_ossified_slots(&self) -> RpcResult<Vec<OssifiedSlotInfo>>;

    /// Create a new prediction market
    #[method(name = "create_market")]
    async fn create_market(
        &self,
        title: String,
        description: String,
        decision_slots: Vec<String>, // Hex-encoded slot IDs
        market_type: String, // "independent" or "categorical"
        has_residual: Option<bool>, // Only for categorical markets
        b: Option<f64>, // LMSR beta parameter
        trading_fee: Option<f64>, // Trading fee percentage
        tags: Option<Vec<String>>, // Categorization tags
        fee_sats: u64, // Transaction fee
    ) -> RpcResult<String>; // Returns market ID

    /// Create a new multidimensional prediction market with mixed dimension types
    #[method(name = "create_market_dimensional")]
    async fn create_market_dimensional(
        &self,
        title: String,
        description: String,
        dimensions: String, // JSON-like dimension specification: "[slot1,[slot2,slot3],slot4]"
        b: Option<f64>, // LMSR beta parameter
        trading_fee: Option<f64>, // Trading fee percentage
        tags: Option<Vec<String>>, // Categorization tags
        fee_sats: u64, // Transaction fee
    ) -> RpcResult<String>; // Returns market ID

    /// List all markets in Trading state
    #[open_api_method(output_schema(ToSchema = "Vec<MarketInfo>"))]
    #[method(name = "list_markets")]
    async fn list_markets(&self) -> RpcResult<Vec<MarketInfo>>;

    /// View detailed information for a specific market
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "view_market")]
    async fn view_market(&self, market_id: String) -> RpcResult<Option<MarketDetails>>;
}
