//! RPC API

use std::net::SocketAddr;

use jsonrpsee::{core::RpcResult, proc_macros::rpc};
use l2l_openapi::open_api;

use serde::{Deserialize, Serialize};
use truthcoin_dc::{
    authorization::{Dst, Signature},
    net::{Peer, PeerConnectionStatus},
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
pub struct MarketOutcome {
    pub name: String,
    pub current_price: f64,
    pub probability: f64,
    pub volume: f64,
    pub index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketData {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcomes: Vec<MarketOutcome>,
    pub state: String,
    pub market_maker: String,
    pub expires_at: Option<u64>,
    pub beta: f64,
    pub trading_fee: f64,
    pub tags: Vec<String>,
    pub created_at_height: u64,
    pub treasury: f64,
    pub total_volume: f64,
    pub liquidity: f64,
    pub decision_slots: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MarketSummary {
    pub market_id: String,
    pub title: String,
    pub description: String,
    pub outcome_count: usize,
    pub state: String,
    pub volume: f64,
    pub created_at_height: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SharePosition {
    pub market_id: String,
    pub outcome_index: usize,
    pub outcome_name: String,
    pub shares_held: f64,
    pub avg_purchase_price: f64,
    pub current_price: f64,
    pub current_value: f64,
    pub unrealized_pnl: f64,
    pub cost_basis: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct UserHoldings {
    pub address: String,
    pub positions: Vec<SharePosition>,
    pub total_value: f64,
    pub total_cost_basis: f64,
    pub total_unrealized_pnl: f64,
    pub active_markets: usize,
    pub last_updated_height: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CreateMarketRequest {
    pub title: String,
    pub description: String,
    pub market_type: String,
    pub decision_slots: Vec<String>,
    pub dimensions: Option<String>,
    pub has_residual: Option<bool>,
    pub beta: Option<f64>,
    pub trading_fee: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub initial_liquidity: Option<u64>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct CalculateInitialLiquidityRequest {
    pub beta: f64,
    pub market_type: String,
    pub decision_slots: Option<Vec<String>>,
    pub num_outcomes: Option<usize>,
    pub dimensions: Option<String>,
    pub has_residual: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct InitialLiquidityCalculation {
    pub beta: f64,
    pub num_outcomes: usize,
    pub initial_liquidity_sats: u64,
    pub min_treasury_sats: u64,
    pub market_config: String,
    pub outcome_breakdown: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct RegisterVoterRequest {
    pub reputation_bond_sats: Option<u64>,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SubmitVoteRequest {
    pub decision_id: String,
    pub vote_value: f64,
    pub period_id: u32,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoteBatchItem {
    pub decision_id: String,
    pub vote_value: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct SubmitVoteBatchRequest {
    pub votes: Vec<VoteBatchItem>,
    pub period_id: u32,
    pub fee_sats: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterInfo {
    pub address: String,
    pub reputation: f64,
    pub total_votes: u64,
    pub periods_active: u32,
    pub accuracy_score: f64,
    pub registered_at_height: u64,
    pub is_active: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoteInfo {
    pub voter_address: String,
    pub decision_id: String,
    pub vote_value: f64,
    pub period_id: u32,
    pub block_height: u64,
    pub txid: String,
    pub is_batch_vote: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VotingPeriodDetails {
    pub period_id: u32,
    pub start_time: u64,
    pub end_time: u64,
    pub status: String,
    pub decision_slots: Vec<String>,
    pub created_at_height: u64,
    pub total_voters: u64,
    pub active_voters: u64,
    pub total_votes: u64,
    pub consensus_reached: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct VoterParticipation {
    pub address: String,
    pub period_id: u32,
    pub votes_cast: u32,
    pub decisions_available: u32,
    pub participation_rate: f64,
    pub participated_in_consensus: bool,
}

#[open_api(ref_schemas[
    truthcoin_schema::BitcoinAddr, truthcoin_schema::BitcoinBlockHash,
    truthcoin_schema::BitcoinTransaction, truthcoin_schema::BitcoinOutPoint,
    truthcoin_schema::SocketAddr, Address, AssetId, Authorization,
    BitcoinOutputContent, BlockHash, Body, CalculateInitialLiquidityRequest,
    CreateMarketRequest, EncryptionPubKey, FilledOutputContent, Header,
    InitialLiquidityCalculation, MarketData, MarketOutcome, MarketSummary,
    MerkleRoot, OutPoint, Output, OutputContent,
    PeerConnectionStatus, RegisterVoterRequest, SharePosition, Signature, SlotInfo, SlotStatus,
    SubmitVoteRequest, SubmitVoteBatchRequest, Transaction, TxData, Txid, TxIn, UserHoldings,
    VoteBatchItem, VoteInfo, VoterInfo, VoterParticipation,
    VotingPeriodDetails, VotingPeriodInfo, WithdrawalOutputContent, VerifyingKey,
])]
#[rpc(client, server)]
pub trait Rpc {
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "bitcoin_balance")]
    async fn bitcoin_balance(&self) -> RpcResult<Balance>;

    #[open_api_method(output_schema(PartialSchema = "schema::BitcoinTxid"))]
    #[method(name = "create_deposit")]
    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid>;

    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "connect_peer")]
    async fn connect_peer(
        &self,
        #[open_api_method_arg(schema(
            ToSchema = "truthcoin_schema::SocketAddr"
        ))]
        addr: SocketAddr,
    ) -> RpcResult<()>;

    #[method(name = "decrypt_msg")]
    async fn decrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        ciphertext: String,
    ) -> RpcResult<String>;

    #[method(name = "encrypt_msg")]
    async fn encrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        msg: String,
    ) -> RpcResult<String>;

    #[method(name = "format_deposit_address")]
    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String>;

    #[method(name = "generate_mnemonic")]
    async fn generate_mnemonic(&self) -> RpcResult<String>;

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

    /// Create a new prediction market supporting all market types
    /// according to Bitcoin Hivemind whitepaper Section 3.1 - Market Creation
    #[method(name = "create_market")]
    async fn create_market(
        &self,
        request: CreateMarketRequest,
    ) -> RpcResult<String>; // Returns market ID

    /// List all markets in Trading state with lightweight data
    #[open_api_method(output_schema(ToSchema = "Vec<MarketSummary>"))]
    #[method(name = "list_markets")]
    async fn list_markets(&self) -> RpcResult<Vec<MarketSummary>>;

    /// View detailed information for a specific market
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "view_market")]
    async fn view_market(
        &self,
        market_id: String,
    ) -> RpcResult<Option<MarketData>>;

    /// Calculate initial liquidity required for market creation based on beta parameter
    /// Uses formula: Initial Liquidity = β × ln(Number of States in the Market)
    /// Helpful for previewing costs and GUI parameter selection
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "calculate_initial_liquidity")]
    async fn calculate_initial_liquidity(
        &self,
        request: CalculateInitialLiquidityRequest,
    ) -> RpcResult<InitialLiquidityCalculation>;

    /// Get share positions for a specific user address
    /// Returns all positions across all markets according to Hivemind Section 4.3
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_user_share_positions")]
    async fn get_user_share_positions(
        &self,
        address: Address,
    ) -> RpcResult<UserHoldings>;

    /// Get share positions for a specific market and user
    #[open_api_method(output_schema(ToSchema = "Vec<SharePosition>"))]
    #[method(name = "get_market_share_positions")]
    async fn get_market_share_positions(
        &self,
        address: Address,
        market_id: String,
    ) -> RpcResult<Vec<SharePosition>>;

    /// Buy shares in a prediction market using LMSR pricing
    /// Implements optimal pricing according to Hivemind Section 2.3 - LMSR Mechanics
    #[method(name = "buy_shares")]
    async fn buy_shares(
        &self,
        market_id: String,
        outcome_index: usize,
        shares_amount: f64,
        max_cost: u64, // Maximum cost in sats (slippage protection)
        fee_sats: u64, // Transaction fee
    ) -> RpcResult<String>; // Returns transaction ID

    /// Calculate share purchase cost using current market snapshot
    /// Provides accurate pricing without executing the trade
    #[method(name = "calculate_share_cost")]
    async fn calculate_share_cost(
        &self,
        market_id: String,
        outcome_index: usize,
        shares_amount: f64,
    ) -> RpcResult<u64>; // Returns cost in sats

    /// Redeem shares in a resolved prediction market
    #[method(name = "redeem_shares")]
    async fn redeem_shares(
        &self,
        market_id: String,
        outcome_index: usize,
        shares_amount: f64,
        fee_sats: u64, // Transaction fee
    ) -> RpcResult<String>; // Returns transaction ID

    /// Register as a voter in the Bitcoin Hivemind voting system
    /// This is a one-time registration that establishes voter identity and reputation bond
    #[open_api_method(output_schema(ToSchema = "String"))]
    #[method(name = "register_voter")]
    async fn register_voter(
        &self,
        request: RegisterVoterRequest,
    ) -> RpcResult<String>; // Returns transaction ID

    /// Submit a single vote for a decision in the current voting period
    /// Vote value should be 0.0-1.0 for binary decisions, scaled appropriately for scalar decisions
    #[open_api_method(output_schema(ToSchema = "String"))]
    #[method(name = "submit_vote")]
    async fn submit_vote(
        &self,
        request: SubmitVoteRequest,
    ) -> RpcResult<String>; // Returns transaction ID

    /// Submit multiple votes efficiently in a single transaction
    /// Useful for voters who want to vote on many decisions at once
    #[open_api_method(output_schema(ToSchema = "String"))]
    #[method(name = "submit_vote_batch")]
    async fn submit_vote_batch(
        &self,
        request: SubmitVoteBatchRequest,
    ) -> RpcResult<String>; // Returns transaction ID

    /// Get information about a specific voter including reputation and voting history
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_voter_info")]
    async fn get_voter_info(
        &self,
        address: Address,
    ) -> RpcResult<Option<VoterInfo>>;

    /// Get detailed information about a specific voting period
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_voting_period_details")]
    async fn get_voting_period_details(
        &self,
        period_id: u32,
    ) -> RpcResult<Option<VotingPeriodDetails>>;

    /// Get all votes cast by a specific voter in a voting period
    #[open_api_method(output_schema(ToSchema = "Vec<VoteInfo>"))]
    #[method(name = "get_voter_votes")]
    async fn get_voter_votes(
        &self,
        address: Address,
        period_id: Option<u32>, // If None, returns votes from all periods
    ) -> RpcResult<Vec<VoteInfo>>;

    /// Get all votes cast for a specific decision slot
    #[open_api_method(output_schema(ToSchema = "Vec<VoteInfo>"))]
    #[method(name = "get_decision_votes")]
    async fn get_decision_votes(
        &self,
        slot_id: String,
    ) -> RpcResult<Vec<VoteInfo>>;

    /// Get voter participation summary for a specific period
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_voter_participation")]
    async fn get_voter_participation(
        &self,
        address: Address,
        period_id: u32,
    ) -> RpcResult<Option<VoterParticipation>>;

    /// List all registered voters with their current reputation
    #[open_api_method(output_schema(ToSchema = "Vec<VoterInfo>"))]
    #[method(name = "list_voters")]
    async fn list_voters(&self) -> RpcResult<Vec<VoterInfo>>;

    /// Check if an address is registered as a voter
    #[open_api_method(output_schema(ToSchema = "bool"))]
    #[method(name = "is_registered_voter")]
    async fn is_registered_voter(&self, address: Address) -> RpcResult<bool>;

    /// Get the current voting power (Votecoin balance) for an address
    /// Voting power determines weight in consensus calculations
    #[open_api_method(output_schema(ToSchema = "u32"))]
    #[method(name = "get_voting_power")]
    async fn get_voting_power(&self, address: Address) -> RpcResult<u32>; // Returns Votecoin balance

    /// Get voting statistics for the current active voting period
    /// Returns aggregated data about participation and vote counts
    #[open_api_method(output_schema(ToSchema))]
    #[method(name = "get_current_voting_stats")]
    async fn get_current_voting_stats(
        &self,
    ) -> RpcResult<Option<VotingPeriodDetails>>;
}
