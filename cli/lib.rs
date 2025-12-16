use std::{
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use clap::{ArgAction, Parser, Subcommand};
use http::HeaderMap;
use jsonrpsee::{core::client::ClientT, http_client::HttpClientBuilder};
use serde::Serialize;
use tracing_subscriber::layer::SubscriberExt as _;
use truthcoin_dc::{
    authorization::{Dst, Signature},
    types::{
        Address, BlockHash, EncryptionPubKey, THIS_SIDECHAIN, Txid,
        VerifyingKey,
    },
};
use truthcoin_dc_app_rpc_api::RpcClient;
use url::{Host, Url};

/// Parse comma-separated input into filtered string vector
pub fn parse_comma_separated(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Validate that a list contains at least one item
pub fn validate_non_empty_list(
    list: &[String],
    name: &str,
) -> Result<(), String> {
    if list.is_empty() {
        Err(format!("Error: At least one {} is required", name))
    } else {
        Ok(())
    }
}

/// Extract decision slots from dimensions specification
/// Dimensions format: "[slot1,[slot2,slot3],slot4]" or "[slot1,slot2]"
pub fn extract_slots_from_dimensions(
    dimensions: &str,
) -> Result<Vec<String>, String> {
    use regex::Regex;

    // Remove outer brackets and whitespace
    let cleaned = dimensions
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');

    // Extract all slot IDs using regex pattern
    let slot_pattern = Regex::new(r"[a-fA-F0-9]{6}")
        .map_err(|e| format!("Regex error: {}", e))?;
    let slots: Vec<String> = slot_pattern
        .find_iter(cleaned)
        .map(|m| m.as_str().to_string())
        .collect();

    if slots.is_empty() {
        return Err(
            "Error: No valid decision slots found in dimensions specification"
                .to_string(),
        );
    }

    Ok(slots)
}

/// Format transaction success messages consistently
pub fn format_tx_success(
    operation: &str,
    details: Option<&str>,
    txid: &str,
) -> String {
    match details {
        Some(details) => {
            format!("{} successful ({}): {}", operation, details, txid)
        }
        None => format!("{} successful: {}", operation, txid),
    }
}

/// Truncate strings for display with ellipsis
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

/// Format slot display consistently
pub fn format_slot_info(slot_id: &str, period: u32, index: u32) -> String {
    format!("Slot {} (Period {}, Index {})", slot_id, period, index)
}

pub fn json_response<T>(data: &T) -> anyhow::Result<String>
where
    T: Serialize,
{
    Ok(serde_json::to_string_pretty(data)?)
}

#[derive(Clone, Debug, Subcommand)]
#[command(arg_required_else_help(true))]
pub enum Command {
    // === SYSTEM COMMANDS ===
    /// Check node status and connection
    #[command(name = "status", alias = "stat", alias = "s")]
    Status,

    /// Stop the node  
    #[command(name = "stop", alias = "shutdown")]
    Stop,

    /// Attempt to mine a sidechain block
    #[command(name = "mine", alias = "m")]
    Mine {
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Show OpenAPI schema
    #[command(name = "openapi-schema", alias = "schema", alias = "api-docs")]
    OpenApiSchema,

    // === WALLET COMMANDS ===
    /// Get wallet balance in sats
    #[command(name = "balance", alias = "bal", alias = "b")]
    Balance,

    /// Get a new address
    #[command(
        name = "get-new-address",
        alias = "address",
        alias = "addr",
        alias = "new-addr"
    )]
    GetNewAddress,

    /// Get wallet addresses  
    #[command(
        name = "get-wallet-addresses",
        alias = "addresses",
        alias = "addrs"
    )]
    GetWalletAddresses,

    /// List owned UTXOs
    #[command(name = "my-utxos", alias = "utxos", alias = "my-coins")]
    MyUtxos,

    /// List unconfirmed owned UTXOs  
    #[command(name = "my-unconfirmed-utxos")]
    MyUnconfirmedUtxos,

    /// Get wallet UTXOs
    #[command(name = "get-wallet-utxos")]
    GetWalletUtxos,

    /// List all UTXOs (admin)
    #[command(name = "list-utxos")]
    ListUtxos,

    /// Transfer funds to address
    #[command(name = "transfer", alias = "send", alias = "tx")]
    Transfer {
        dest: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Transfer votecoin to address  
    #[command(
        name = "transfer-votecoin",
        alias = "send-votecoin",
        alias = "send-vc",
        alias = "transfer-vc"
    )]
    TransferVotecoin {
        dest: Address,
        #[arg(long)]
        amount: u32,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Initiate withdrawal to mainchain
    #[command(name = "withdraw", alias = "wd", alias = "exit")]
    Withdraw {
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        #[arg(long)]
        amount_sats: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
        #[arg(long, default_value = "1000")]
        mainchain_fee_sats: u64,
    },

    /// Deposit to address
    #[command(name = "create-deposit", alias = "deposit", alias = "dep")]
    CreateDeposit {
        address: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Format a deposit address
    #[command(name = "format-deposit-address")]
    FormatDepositAddress { address: Address },

    /// Generate mnemonic seed phrase
    #[command(
        name = "generate-mnemonic",
        alias = "gen-mnemonic",
        alias = "mnemonic"
    )]
    GenerateMnemonic,

    /// Set wallet seed from mnemonic
    #[command(
        name = "set-seed-from-mnemonic",
        alias = "import-seed",
        alias = "restore-seed"
    )]
    SetSeedFromMnemonic { mnemonic: String },

    /// Get total sidechain wealth
    #[command(
        name = "sidechain-wealth",
        alias = "wealth",
        alias = "total-wealth"
    )]
    SidechainWealth,

    // === BLOCKCHAIN COMMANDS ===
    /// Get current block count
    #[command(name = "get-block-count", alias = "blockcount", alias = "height")]
    GetBlockCount,

    /// Get block data
    #[command(name = "get-block", alias = "block")]
    GetBlock { block_hash: BlockHash },

    /// Get best mainchain block hash
    #[command(name = "get-best-mainchain-block-hash")]
    GetBestMainchainBlockHash,

    /// Get best sidechain block hash  
    #[command(name = "get-best-sidechain-block-hash")]
    GetBestSidechainBlockHash,

    /// Get mainchain BMM inclusions
    #[command(name = "get-bmm-inclusions")]
    GetBmmInclusions {
        block_hash: truthcoin_dc::types::BlockHash,
    },

    /// Get transaction by txid
    #[command(
        name = "get-transaction",
        alias = "get-tx",
        alias = "transaction"
    )]
    GetTransaction { txid: Txid },

    /// Get transaction info
    #[command(
        name = "get-transaction-info",
        alias = "txinfo",
        alias = "tx-info"
    )]
    GetTransactionInfo { txid: Txid },

    /// Get pending withdrawal bundle
    #[command(name = "pending-withdrawal-bundle")]
    PendingWithdrawalBundle,

    /// Get latest failed withdrawal bundle height
    #[command(name = "latest-failed-withdrawal-bundle-height")]
    LatestFailedWithdrawalBundleHeight,

    /// Remove transaction from mempool
    #[command(name = "remove-from-mempool")]
    RemoveFromMempool { txid: Txid },

    // === NETWORK COMMANDS ===
    /// Connect to a peer
    #[command(
        name = "connect-peer",
        alias = "connect",
        alias = "peer",
        alias = "add-peer"
    )]
    ConnectPeer { addr: SocketAddr },

    /// List connected peers
    #[command(
        name = "list-peers",
        alias = "peers",
        alias = "connections",
        alias = "network"
    )]
    ListPeers,

    // === CRYPTOGRAPHY COMMANDS ===
    /// Get new encryption key
    #[command(
        name = "get-new-encryption-key",
        alias = "new-encryption-key",
        alias = "new-enc-key"
    )]
    GetNewEncryptionKey,

    /// Get new verifying key
    #[command(
        name = "get-new-verifying-key",
        alias = "new-verifying-key",
        alias = "new-verify-key"
    )]
    GetNewVerifyingKey,

    /// Encrypt message
    #[command(name = "encrypt-msg", alias = "encrypt")]
    EncryptMsg {
        #[arg(long)]
        encryption_pubkey: EncryptionPubKey,
        #[arg(long)]
        msg: String,
    },

    /// Decrypt message
    #[command(name = "decrypt-msg", alias = "decrypt")]
    DecryptMsg {
        #[arg(long)]
        encryption_pubkey: EncryptionPubKey,
        #[arg(long)]
        msg: String,
        /// Decode as UTF-8
        #[arg(long)]
        utf8: bool,
    },

    /// Sign arbitrary message with verifying key
    #[command(name = "sign-arbitrary-msg", alias = "sign", alias = "sign-msg")]
    SignArbitraryMsg {
        #[arg(long)]
        verifying_key: VerifyingKey,
        #[arg(long)]
        msg: String,
    },

    /// Sign arbitrary message as address
    #[command(
        name = "sign-arbitrary-msg-as-addr",
        alias = "sign-as-addr",
        alias = "sign-addr"
    )]
    SignArbitraryMsgAsAddr {
        #[arg(long)]
        address: Address,
        #[arg(long)]
        msg: String,
    },

    /// Verify signature
    #[command(
        name = "verify-signature",
        alias = "verify",
        alias = "verify-sig"
    )]
    VerifySignature {
        #[arg(long)]
        signature: Signature,
        #[arg(long)]
        verifying_key: VerifyingKey,
        #[arg(long)]
        dst: Dst,
        #[arg(long)]
        msg: String,
    },

    // === SLOT COMMANDS ===
    /// Show slot system status
    #[command(name = "slots-status")]
    SlotsStatus,

    /// List all available slots
    #[command(name = "slots-list-all", alias = "slots", alias = "list-slots")]
    SlotsListAll,

    /// Get slots for specific period
    #[command(name = "slots-get-quarter")]
    SlotsGetQuarter { quarter: u32 },

    /// Get available slots in period
    #[command(name = "get-available-slots")]
    GetAvailableSlots {
        #[arg(long)]
        period_index: u32,
    },

    /// Get slot by ID
    #[command(name = "get-slot-by-id", alias = "slot", alias = "get-slot")]
    GetSlotById {
        #[arg(long)]
        slot_id_hex: String,
    },

    /// Get claimed slots in period
    #[command(name = "get-claimed-slots")]
    GetClaimedSlots {
        #[arg(long)]
        period_index: u32,
    },

    /// Check if slot is in voting
    #[command(name = "is-slot-in-voting")]
    IsSlotInVoting {
        #[arg(long)]
        slot_id_hex: String,
    },

    /// Get voting periods
    #[command(
        name = "get-voting-periods",
        alias = "voting-periods",
        alias = "voting"
    )]
    GetVotingPeriods,

    /// Get ossified slots
    #[command(
        name = "get-ossified-slots",
        alias = "ossified-slots",
        alias = "ossified"
    )]
    GetOssifiedSlots,

    /// Claim decision slot
    #[command(
        name = "claim-decision-slot",
        alias = "claim-slot",
        alias = "claim"
    )]
    ClaimDecisionSlot {
        #[arg(long)]
        period_index: u32,
        #[arg(long)]
        slot_index: u32,
        #[arg(long, action = ArgAction::Set)]
        is_standard: bool,
        #[arg(long, action = ArgAction::Set)]
        is_scaled: bool,
        #[arg(long)]
        question: String,
        #[arg(long)]
        min: Option<u16>,
        #[arg(long)]
        max: Option<u16>,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    // === MARKET COMMANDS ===
    /// Create prediction market using dimensions specification
    #[command(name = "create-market", alias = "cm", alias = "market")]
    CreateMarket {
        /// Market title
        #[arg(long)]
        title: String,
        /// Market description
        #[arg(long)]
        description: String,
        /// Dimensions specification using bracket notation:
        /// - "[050065]" = Single binary market
        /// - "[050065,050066]" = Two independent binary dimensions
        /// - "[[050065,050066,050067]]" = One categorical dimension (mutually exclusive)
        /// - "[050065,[050066,050067],050068]" = Mixed: independent + categorical + independent
        #[arg(long)]
        dimensions: String,
        /// LMSR beta parameter (liquidity sensitivity)
        #[arg(long, default_value = "7.0")]
        beta: f64,
        /// Trading fee percentage
        #[arg(long, default_value = "0.005")]
        trading_fee: f64,
        /// Market tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Transaction fee
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Create yes/no prediction market (simplified)
    #[command(name = "create-yes-no-market", alias = "yn", alias = "yes-no")]
    CreateYesNoMarket {
        /// Market question/title
        #[arg(long)]
        question: String,
        /// Decision slot IDs (comma-separated)  
        #[arg(long)]
        decision_slots: String,
        /// Market tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// LMSR beta parameter (liquidity sensitivity)
        #[arg(long, default_value = "7.0")]
        beta: f64,
        /// Transaction fee
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// List all markets
    #[command(name = "list-markets", alias = "markets", alias = "ls-markets")]
    ListMarkets,

    /// View market details
    #[command(name = "view-market", alias = "show-market", alias = "info")]
    ViewMarket { market_id: String },

    /// Buy shares
    #[command(name = "buy-shares", alias = "buy")]
    BuyShares {
        #[arg(long)]
        market_id: String,
        #[arg(long)]
        outcome_index: usize,
        #[arg(long)]
        shares_amount: f64,
        #[arg(long)]
        max_cost: u64,
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Calculate share cost
    #[command(
        name = "calculate-share-cost",
        alias = "cost",
        alias = "calc-cost"
    )]
    CalculateShareCost {
        #[arg(long)]
        market_id: String,
        #[arg(long)]
        outcome_index: usize,
        #[arg(long)]
        shares_amount: f64,
    },

    /// Calculate initial liquidity required for market creation
    #[command(
        name = "calculate-initial-liquidity",
        alias = "calc-liquidity",
        alias = "liquidity"
    )]
    CalculateInitialLiquidity {
        /// LMSR beta parameter
        #[arg(long)]
        beta: f64,
        /// Market type: independent, categorical, dimensional
        #[arg(long, default_value = "independent")]
        market_type: String,
        /// Number of outcomes (for preview mode)
        #[arg(long)]
        num_outcomes: Option<usize>,
        /// Decision slot IDs (comma-separated)
        #[arg(long)]
        decision_slots: Option<String>,
        /// Has residual outcome (for categorical markets)
        #[arg(long)]
        has_residual: Option<bool>,
        /// Dimensional specification (for dimensional markets)
        #[arg(long)]
        dimensions: Option<String>,
    },

    /// Get user share positions
    #[command(
        name = "get-user-share-positions",
        alias = "positions",
        alias = "portfolio"
    )]
    GetUserSharePositions {
        #[arg(long)]
        address: Option<Address>,
    },

    /// Get market share positions
    #[command(name = "get-market-share-positions")]
    GetMarketSharePositions {
        #[arg(long)]
        address: Option<Address>,
        #[arg(long)]
        market_id: String,
    },

    // === VOTING COMMANDS ===
    /// Register as a voter in the Bitcoin Hivemind voting system
    #[command(
        name = "register-voter",
        alias = "reg-voter",
        alias = "voter-register"
    )]
    RegisterVoter {
        /// Reputation bond amount in sats (optional)
        #[arg(long)]
        reputation_bond_sats: Option<u64>,
        /// Transaction fee in sats
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Submit a single vote for a decision in the current voting period
    #[command(name = "submit-vote", alias = "vote", alias = "cast-vote")]
    SubmitVote {
        /// Decision ID (slot ID) to vote on
        #[arg(long)]
        decision_id: String,
        /// Vote value (0.0-1.0 for binary, scaled for scalar decisions)
        #[arg(long)]
        vote_value: f64,
        /// Transaction fee in sats
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Submit multiple votes efficiently in a single transaction
    #[command(
        name = "submit-vote-batch",
        alias = "vote-batch",
        alias = "batch-vote"
    )]
    SubmitVoteBatch {
        /// Votes in format "decision_id:value,decision_id:value,..."
        #[arg(long)]
        votes: String,
        /// Transaction fee in sats
        #[arg(long, default_value = "1000")]
        fee_sats: u64,
    },

    /// Get information about a specific voter
    #[command(name = "get-voter-info", alias = "voter-info", alias = "voter")]
    GetVoterInfo {
        /// Voter address
        #[arg(long)]
        address: Address,
    },

    /// Get detailed information about a specific voting period
    #[command(
        name = "get-voting-period-details",
        alias = "period-details",
        alias = "voting-details"
    )]
    GetVotingPeriodDetails {
        /// Period ID
        #[arg(long)]
        period_id: u32,
    },

    /// Get all votes cast by a specific voter
    #[command(
        name = "get-voter-votes",
        alias = "voter-votes",
        alias = "my-votes"
    )]
    GetVoterVotes {
        /// Voter address
        #[arg(long)]
        address: Address,
        /// Optional period ID (if not specified, returns votes from all periods)
        #[arg(long)]
        period_id: Option<u32>,
    },

    /// Get all votes cast for a specific decision slot
    #[command(
        name = "get-decision-votes",
        alias = "decision-votes",
        alias = "slot-votes"
    )]
    GetDecisionVotes {
        /// Decision slot ID
        #[arg(long)]
        slot_id: String,
    },

    /// Get voter participation summary for a specific period
    #[command(
        name = "get-voter-participation",
        alias = "participation",
        alias = "voter-participation"
    )]
    GetVoterParticipation {
        /// Voter address
        #[arg(long)]
        address: Address,
        /// Period ID
        #[arg(long)]
        period_id: u32,
    },

    /// List all registered voters with their current reputation
    #[command(name = "list-voters", alias = "voters", alias = "all-voters")]
    ListVoters,

    /// Check if an address is registered as a voter
    #[command(
        name = "is-registered-voter",
        alias = "check-voter",
        alias = "is-voter"
    )]
    IsRegisteredVoter {
        /// Address to check
        #[arg(long)]
        address: Address,
    },

    /// Get the Votecoin balance for an address
    #[command(
        name = "get-votecoin-balance",
        alias = "votecoin-balance",
        alias = "vc-balance"
    )]
    GetVotecoinBalance {
        /// Address to check
        #[arg(long)]
        address: Address,
    },

    /// Get voting statistics for the current active voting period
    #[command(
        name = "get-current-voting-stats",
        alias = "voting-stats",
        alias = "current-voting"
    )]
    GetCurrentVotingStats,

    /// Get the consensus results for a voting period including SVD analysis
    #[command(
        name = "get-voting-consensus-results",
        alias = "consensus-results",
        alias = "svd-results"
    )]
    GetVotingConsensusResults {
        /// Period ID
        #[arg(long)]
        period_id: u32,
    },

    /// Get the reputation value for a specific voter
    #[command(
        name = "get-voter-reputation",
        alias = "reputation",
        alias = "voter-rep"
    )]
    GetVoterReputation {
        /// Voter ID (address as string)
        #[arg(long)]
        voter_id: String,
    },

    /// Get the status of a voting period
    #[command(
        name = "get-voting-period-status",
        alias = "period-status",
        alias = "voting-status"
    )]
    GetVotingPeriodStatus {
        /// Period ID
        #[arg(long)]
        period_id: u32,
    },

    /// Get redistribution summary for a specific voting period
    #[command(
        name = "get-redistribution-summary",
        alias = "redistribution",
        alias = "redist-summary"
    )]
    GetRedistributionSummary {
        /// Period ID
        #[arg(long)]
        period_id: u32,
    },
}

const DEFAULT_RPC_HOST: Host = Host::Ipv4(Ipv4Addr::LOCALHOST);

const DEFAULT_RPC_PORT: u16 = 6000 + THIS_SIDECHAIN as u16;

const DEFAULT_TIMEOUT_SECS: u64 = 60;

#[derive(Clone, Debug, Parser)]
#[command(
    author,
    version,
    about = "Truthcoin Drivechain CLI - Bitcoin Hivemind prediction market client",
    long_about = "
Truthcoin DC CLI - Command-line interface for Bitcoin Hivemind prediction markets

COMMAND GROUPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖥️  SYSTEM:
    status (stat, s)         Check node status and connection
    stop (shutdown)          Shutdown the node  
    mine (m)                 Mine a sidechain block
    openapi-schema (schema)  Show OpenAPI documentation

💰 WALLET:
    balance (bal, b)                    Check wallet balance
    get-new-address (addr, address)     Generate new address
    transfer (send, tx)                 Send funds to address
    transfer-votecoin (send-vc)         Send votecoin to address
    withdraw (wd, exit)                 Withdraw to Bitcoin mainchain
    create-deposit (deposit, dep)       Create deposit transaction
    my-utxos (utxos, my-coins)          List your unspent outputs
    sidechain-wealth (wealth)           Show total sidechain value

📊 MARKETS:
    create-market (cm, market)          Create new prediction market with beta parameter
    create-yes-no-market (yn, yes-no)   Quick yes/no market creation with beta parameter
    list-markets (markets)              List all active markets
    view-market (show-market, info)     View detailed market info
    buy-shares (buy)                    Purchase market shares
    calculate-share-cost (cost)         Calculate share purchase cost
    calculate-initial-liquidity (calc-liquidity) Calculate required initial liquidity
    get-user-share-positions (positions) View your market positions

🎰 SLOTS:
    slots-list-all (slots)              List all available slots
    claim-decision-slot (claim)         Claim a decision slot
    get-slot-by-id (slot)              View slot details by ID
    get-voting-periods (voting)         Show voting periods
    get-ossified-slots (ossified)       Show finalized slots

🔗 BLOCKCHAIN:
    get-block-count (height, blockcount) Current blockchain height
    get-block (block)                   Get block information
    get-transaction (get-tx, transaction) Get transaction details
    connect-peer (connect, peer)        Connect to network peer
    list-peers (peers, network)         Show connected peers

🔐 CRYPTO:
    generate-mnemonic (mnemonic)        Generate seed phrase
    encrypt-msg (encrypt)               Encrypt message
    decrypt-msg (decrypt)               Decrypt message
    sign-arbitrary-msg (sign)           Sign message with key
    verify-signature (verify)           Verify message signature

🗳️  VOTING:
    register-voter (reg-voter)          Register as a voter
    submit-vote (vote)                  Submit a single vote
    submit-vote-batch (vote-batch)      Submit multiple votes at once
    list-voters (voters)                List all registered voters
    get-voter-info (voter-info)         View voter details
    get-votecoin-balance (vc-balance)   Check votecoin balance
    get-voting-periods (voting)         Show voting periods
    get-current-voting-stats (voting-stats) Current voting statistics
    get-voting-consensus-results (svd-results) View SVD consensus results
    get-redistribution-summary (redistribution) VoteCoin redistribution info

QUICK START:
    truthcoin_dc_app_cli status         # Check if node is running
    truthcoin_dc_app_cli balance        # Check your wallet balance  
    truthcoin_dc_app_cli markets        # Browse active markets
    truthcoin_dc_app_cli yn --question 'Will it rain tomorrow?' --decision-slots abc123 --beta 7.0

For command details: truthcoin_dc_app_cli <command> --help
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
    /// Host used for requests to the RPC server
    #[arg(default_value_t = DEFAULT_RPC_HOST, long, value_parser = Host::parse)]
    pub rpc_host: Host,
    /// Port used for requests to the RPC server
    #[arg(default_value_t = DEFAULT_RPC_PORT, long)]
    pub rpc_port: u16,
    /// Timeout for RPC requests in seconds.
    #[arg(default_value_t = DEFAULT_TIMEOUT_SECS, long = "timeout")]
    timeout_secs: u64,
    #[arg(short, long, help = "Enable verbose HTTP output")]
    pub verbose: bool,
}

impl Cli {
    pub fn new(
        command: Command,
        rpc_host: Option<Host>,
        rpc_port: Option<u16>,
        timeout_secs: Option<u64>,
        verbose: Option<bool>,
    ) -> Self {
        Self {
            command,
            rpc_host: rpc_host.unwrap_or(DEFAULT_RPC_HOST),
            rpc_port: rpc_port.unwrap_or(DEFAULT_RPC_PORT),
            timeout_secs: timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            verbose: verbose.unwrap_or(false),
        }
    }

    fn rpc_url(&self) -> url::Url {
        Url::parse(&format!("http://{}:{}", self.rpc_host, self.rpc_port))
            .unwrap()
    }
}

async fn handle_command<RpcClient>(
    rpc_client: &RpcClient,
    command: Command,
) -> anyhow::Result<String>
where
    RpcClient: ClientT + Sync,
{
    Ok(match command {
        // === SYSTEM COMMANDS ===
        Command::Status => {
            let blockcount = rpc_client.getblockcount().await?;
            let peers = rpc_client.list_peers().await?;
            let balance = rpc_client.bitcoin_balance().await?;
            format!(
                "Node Status: ✓ Online\n\
                Block Count: {}\n\
                Connected Peers: {}\n\
                Wallet Balance: {} sats",
                blockcount,
                peers.len(),
                balance.total
            )
        }
        Command::Stop => {
            let () = rpc_client.stop().await?;
            "Node stopping...".to_string()
        }
        Command::Mine { fee_sats } => {
            let () = rpc_client.mine(Some(fee_sats)).await?;
            format!("Mining block with fee {} sats", fee_sats)
        }
        Command::OpenApiSchema => {
            let openapi =
                <truthcoin_dc_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi(
                );
            openapi.to_pretty_json()?
        }

        // === WALLET COMMANDS ===
        Command::Balance => {
            let balance = rpc_client.bitcoin_balance().await?;
            json_response(&balance)?
        }
        Command::GetNewAddress => {
            let address = rpc_client.get_new_address().await?;
            format!("{address}")
        }
        Command::GetWalletAddresses => {
            let addresses = rpc_client.get_wallet_addresses().await?;
            json_response(&addresses)?
        }
        Command::MyUtxos => {
            let utxos = rpc_client.my_utxos().await?;
            json_response(&utxos)?
        }
        Command::MyUnconfirmedUtxos => {
            let utxos = rpc_client.my_unconfirmed_utxos().await?;
            json_response(&utxos)?
        }
        Command::GetWalletUtxos => {
            let utxos = rpc_client.get_wallet_utxos().await?;
            json_response(&utxos)?
        }
        Command::ListUtxos => {
            let utxos = rpc_client.list_utxos().await?;
            json_response(&utxos)?
        }
        Command::Transfer {
            dest,
            value_sats,
            fee_sats,
        } => {
            let txid = rpc_client
                .transfer(dest, value_sats, fee_sats, None)
                .await?;
            format_tx_success("Transfer", None, &txid.to_string())
        }
        Command::TransferVotecoin {
            dest,
            amount,
            fee_sats,
        } => {
            let txid = rpc_client
                .votecoin_transfer(dest, amount, fee_sats, None)
                .await?;
            format_tx_success("Votecoin transfer", None, &txid.to_string())
        }
        Command::Withdraw {
            mainchain_address,
            amount_sats,
            fee_sats,
            mainchain_fee_sats,
        } => {
            let txid = rpc_client
                .withdraw(
                    mainchain_address,
                    amount_sats,
                    fee_sats,
                    mainchain_fee_sats,
                )
                .await?;
            format_tx_success("Withdrawal initiated", None, &txid.to_string())
        }
        Command::CreateDeposit {
            address,
            value_sats,
            fee_sats,
        } => {
            let txid = rpc_client
                .create_deposit(address, value_sats, fee_sats)
                .await?;
            format_tx_success("Deposit created", None, &txid.to_string())
        }
        Command::FormatDepositAddress { address } => {
            rpc_client.format_deposit_address(address).await?
        }
        Command::GenerateMnemonic => rpc_client.generate_mnemonic().await?,
        Command::SetSeedFromMnemonic { mnemonic } => {
            let () = rpc_client.set_seed_from_mnemonic(mnemonic).await?;
            "Wallet seed imported successfully".to_string()
        }
        Command::SidechainWealth => {
            let wealth = rpc_client.sidechain_wealth_sats().await?;
            format!("{wealth} sats")
        }

        // === BLOCKCHAIN COMMANDS ===
        Command::GetBlockCount => {
            let blockcount = rpc_client.getblockcount().await?;
            format!("{blockcount}")
        }
        Command::GetBlock { block_hash } => {
            let block = rpc_client.get_block(block_hash).await?;
            json_response(&block)?
        }
        Command::GetBestMainchainBlockHash => {
            let block_hash = rpc_client.get_best_mainchain_block_hash().await?;
            json_response(&block_hash)?
        }
        Command::GetBestSidechainBlockHash => {
            let block_hash = rpc_client.get_best_sidechain_block_hash().await?;
            json_response(&block_hash)?
        }
        Command::GetBmmInclusions { block_hash } => {
            let bmm_inclusions =
                rpc_client.get_bmm_inclusions(block_hash).await?;
            json_response(&bmm_inclusions)?
        }
        Command::GetTransaction { txid } => {
            let tx = rpc_client.get_transaction(txid).await?;
            json_response(&tx)?
        }
        Command::GetTransactionInfo { txid } => {
            let tx_info = rpc_client.get_transaction_info(txid).await?;
            json_response(&tx_info)?
        }
        Command::PendingWithdrawalBundle => {
            let withdrawal_bundle =
                rpc_client.pending_withdrawal_bundle().await?;
            json_response(&withdrawal_bundle)?
        }
        Command::LatestFailedWithdrawalBundleHeight => {
            let height =
                rpc_client.latest_failed_withdrawal_bundle_height().await?;
            json_response(&height)?
        }
        Command::RemoveFromMempool { txid } => {
            let () = rpc_client.remove_from_mempool(txid).await?;
            format!("Transaction {} removed from mempool", txid)
        }

        // === NETWORK COMMANDS ===
        Command::ConnectPeer { addr } => {
            let () = rpc_client.connect_peer(addr).await?;
            format!("Connected to peer: {}", addr)
        }
        Command::ListPeers => {
            let peers = rpc_client.list_peers().await?;
            json_response(&peers)?
        }

        // === CRYPTOGRAPHY COMMANDS ===
        Command::GetNewEncryptionKey => {
            let epk = rpc_client.get_new_encryption_key().await?;
            format!("{epk}")
        }
        Command::GetNewVerifyingKey => {
            let vk = rpc_client.get_new_verifying_key().await?;
            format!("{vk}")
        }
        Command::EncryptMsg {
            encryption_pubkey,
            msg,
        } => rpc_client.encrypt_msg(encryption_pubkey, msg).await?,
        Command::DecryptMsg {
            encryption_pubkey,
            msg,
            utf8,
        } => {
            let msg_hex =
                rpc_client.decrypt_msg(encryption_pubkey, msg).await?;
            if utf8 {
                let msg_bytes: Vec<u8> = hex::decode(msg_hex)?;
                String::from_utf8(msg_bytes)?
            } else {
                msg_hex
            }
        }
        Command::SignArbitraryMsg { verifying_key, msg } => {
            let signature =
                rpc_client.sign_arbitrary_msg(verifying_key, msg).await?;
            format!("{signature}")
        }
        Command::SignArbitraryMsgAsAddr { address, msg } => {
            let authorization =
                rpc_client.sign_arbitrary_msg_as_addr(address, msg).await?;
            serde_json::to_string_pretty(&authorization)?
        }
        Command::VerifySignature {
            signature,
            verifying_key,
            dst,
            msg,
        } => {
            let res = rpc_client
                .verify_signature(signature, verifying_key, dst, msg)
                .await?;
            format!("{res}")
        }

        // === SLOT COMMANDS ===
        Command::SlotsStatus => {
            let status = rpc_client.slot_status().await?;
            let mut output = String::new();
            output.push_str("Slot System Status:\n");
            output.push_str("==================\n\n");

            if status.is_testing_mode {
                output.push_str("Mode: TESTING\n");
                output.push_str(&format!(
                    "Blocks per period: {}\n",
                    status.blocks_per_period
                ));
                output.push_str("Slots are minted every N blocks instead of every quarter.\n");
            } else {
                output.push_str("Mode: PRODUCTION\n");
                output.push_str("Slots are minted every calendar quarter based on Bitcoin timestamps.\n");
            }

            output.push_str(&format!(
                "\nCurrent period: {} ({})\n",
                status.current_period_name, status.current_period
            ));
            output
        }
        Command::SlotsListAll => {
            let slots = rpc_client.slot_list(None).await?;
            let mut output = String::new();
            output.push_str("All Slots by Period:\n");
            output.push_str("====================\n\n");
            if slots.is_empty() {
                output.push_str("No slots minted yet.\n");
            } else {
                // Group by period
                let mut period_counts: std::collections::BTreeMap<u32, u64> =
                    std::collections::BTreeMap::new();
                for slot in &slots {
                    *period_counts.entry(slot.period_index).or_insert(0) += 1;
                }
                for (period, count) in period_counts {
                    output.push_str(&format!(
                        "Period {}: {} slots\n",
                        period, count
                    ));
                }
            }
            output
        }
        Command::SlotsGetQuarter { quarter } => {
            use truthcoin_dc_app_rpc_api::SlotFilter;
            let filter = SlotFilter {
                period: Some(quarter),
                status: None,
            };
            let slots = rpc_client.slot_list(Some(filter)).await?;
            format!("Period {}: {} slots", quarter, slots.len())
        }
        Command::GetAvailableSlots { period_index } => {
            use truthcoin_dc_app_rpc_api::{SlotFilter, SlotState};
            let filter = SlotFilter {
                period: Some(period_index),
                status: Some(SlotState::Available),
            };
            let available_slots = rpc_client.slot_list(Some(filter)).await?;
            if available_slots.is_empty() {
                format!("No available slots in period {}", period_index)
            } else {
                let mut result = format!(
                    "Available slots in period {} ({} total):\n",
                    period_index,
                    available_slots.len()
                );
                for slot in available_slots {
                    result.push_str(&format!(
                        "  Slot {}: {}\n",
                        slot.slot_index, slot.slot_id_hex
                    ));
                }
                result
            }
        }
        Command::GetSlotById { slot_id_hex } => {
            let slot = rpc_client.slot_get(slot_id_hex.clone()).await?;
            match slot {
                Some(slot_details) => {
                    let mut result = format!(
                        "Slot {} (Period {}, Index {}):\n",
                        slot_details.slot_id_hex,
                        slot_details.period_index,
                        slot_details.slot_index
                    );

                    match slot_details.content {
                        truthcoin_dc_app_rpc_api::SlotContentInfo::Empty => {
                            result.push_str(
                                "  Status: EMPTY (available for claiming)\n",
                            );
                        }
                        truthcoin_dc_app_rpc_api::SlotContentInfo::Decision(
                            decision,
                        ) => {
                            result.push_str(&format!(
                                "  Decision ID: {}\n",
                                decision.id
                            ));
                            result.push_str(&format!(
                                "  Market Maker: {}\n",
                                decision.market_maker_pubkey_hash
                            ));
                            result.push_str(&format!(
                                "  Type: {} | {}\n",
                                if decision.is_standard {
                                    "Standard"
                                } else {
                                    "Non-Standard"
                                },
                                if decision.is_scaled {
                                    "Scaled"
                                } else {
                                    "Binary"
                                }
                            ));
                            result.push_str(&format!(
                                "  Question: {}\n",
                                decision.question
                            ));
                            if let (Some(min), Some(max)) =
                                (decision.min, decision.max)
                            {
                                result.push_str(&format!(
                                    "  Range: {} to {}\n",
                                    min, max
                                ));
                            }
                        }
                    }
                    result
                }
                None => format!("Slot {} not found", slot_id_hex),
            }
        }
        Command::GetClaimedSlots { period_index } => {
            use truthcoin_dc_app_rpc_api::{SlotFilter, SlotState};
            let filter = SlotFilter {
                period: Some(period_index),
                status: Some(SlotState::Claimed),
            };
            let claimed_slots = rpc_client.slot_list(Some(filter)).await?;
            if claimed_slots.is_empty() {
                format!("No claimed slots found in period {}", period_index)
            } else {
                let mut result = format!(
                    "Claimed slots in period {} ({} total):\n",
                    period_index,
                    claimed_slots.len()
                );
                for slot in claimed_slots {
                    let question_preview =
                        slot.decision.as_ref().map_or("N/A".to_string(), |d| {
                            if d.question.len() > 30 {
                                format!("{}...", &d.question[..30])
                            } else {
                                d.question.clone()
                            }
                        });
                    let is_standard =
                        slot.decision.as_ref().map_or(false, |d| d.is_standard);
                    let is_scaled =
                        slot.decision.as_ref().map_or(false, |d| d.is_scaled);
                    let market_maker =
                        slot.decision.as_ref().map_or("N/A".to_string(), |d| {
                            if d.market_maker_pubkey_hash.len() >= 8 {
                                d.market_maker_pubkey_hash[..8].to_string()
                            } else {
                                d.market_maker_pubkey_hash.clone()
                            }
                        });
                    result.push_str(&format!(
                        "  Slot {} ({}): {} | {} | Market Maker: {} | \"{}\"\n",
                        slot.slot_index,
                        slot.slot_id_hex,
                        if is_standard {
                            "Standard"
                        } else {
                            "Non-Standard"
                        },
                        if is_scaled { "Scaled" } else { "Binary" },
                        market_maker,
                        question_preview
                    ));
                }
                result
            }
        }
        Command::IsSlotInVoting { slot_id_hex } => {
            use truthcoin_dc_app_rpc_api::SlotState;
            let slot = rpc_client.slot_get(slot_id_hex.clone()).await?;
            match slot {
                Some(s) => {
                    let is_voting = matches!(
                        s.content,
                        truthcoin_dc_app_rpc_api::SlotContentInfo::Decision(_)
                    ) && matches!(
                        rpc_client
                            .slot_list(Some(
                                truthcoin_dc_app_rpc_api::SlotFilter {
                                    period: Some(s.period_index),
                                    status: Some(SlotState::Voting),
                                }
                            ))
                            .await?
                            .iter()
                            .find(|sl| sl.slot_id_hex == slot_id_hex),
                        Some(_)
                    );
                    if is_voting {
                        format!("Slot {} is in voting period", slot_id_hex)
                    } else {
                        format!("Slot {} is NOT in voting period", slot_id_hex)
                    }
                }
                None => format!("Slot {} not found", slot_id_hex),
            }
        }
        Command::GetVotingPeriods => {
            use truthcoin_dc_app_rpc_api::{SlotFilter, SlotState};
            // Get all slots in voting state
            let filter = SlotFilter {
                period: None,
                status: Some(SlotState::Voting),
            };
            let voting_slots = rpc_client.slot_list(Some(filter)).await?;
            if voting_slots.is_empty() {
                "No periods currently in voting phase".to_string()
            } else {
                // Group by period and count
                let mut period_counts: std::collections::BTreeMap<
                    u32,
                    (u64, u64),
                > = std::collections::BTreeMap::new();
                for slot in &voting_slots {
                    let entry = period_counts
                        .entry(slot.period_index)
                        .or_insert((0, 0));
                    entry.0 += 1; // total slots in voting
                    if slot.decision.is_some() {
                        entry.1 += 1; // claimed slots
                    }
                }
                let mut result = format!(
                    "Voting Periods ({} total):\n",
                    period_counts.len()
                );
                result.push_str("================================\n");
                for (period, (total, claimed)) in period_counts {
                    result.push_str(&format!(
                        "Period {}: {}/{} slots claimed ({:.1}%)\n",
                        period,
                        claimed,
                        total,
                        if total > 0 {
                            (claimed as f64 / total as f64) * 100.0
                        } else {
                            0.0
                        }
                    ));
                }
                result
            }
        }
        Command::GetOssifiedSlots => {
            use truthcoin_dc_app_rpc_api::{SlotFilter, SlotState};
            let filter = SlotFilter {
                period: None,
                status: Some(SlotState::Ossified),
            };
            let ossified_slots = rpc_client.slot_list(Some(filter)).await?;
            if ossified_slots.is_empty() {
                "No ossified slots found".to_string()
            } else {
                let mut result = format!(
                    "Ossified Slots ({} total):\n",
                    ossified_slots.len()
                );
                result.push_str("==========================\n");
                for slot in ossified_slots {
                    result.push_str(&format!(
                        "Slot {} (Period {}, Index {}): ",
                        slot.slot_id_hex, slot.period_index, slot.slot_index
                    ));
                    if let Some(decision) = &slot.decision {
                        let question_preview = if decision.question.len() > 50 {
                            format!("{}...", &decision.question[..50])
                        } else {
                            decision.question.clone()
                        };
                        result.push_str(&format!(
                            "{} - {}\n",
                            if decision.is_standard {
                                "Standard"
                            } else {
                                "Non-standard"
                            },
                            question_preview
                        ));
                    } else {
                        result.push_str("Empty slot\n");
                    }
                }
                result
            }
        }
        Command::ClaimDecisionSlot {
            period_index,
            slot_index,
            is_standard,
            is_scaled,
            question,
            min,
            max,
            fee_sats,
        } => {
            let txid = rpc_client
                .slot_claim(
                    period_index,
                    slot_index,
                    is_standard,
                    is_scaled,
                    question,
                    min,
                    max,
                    fee_sats,
                )
                .await?;
            format!(
                "Decision slot claimed successfully. Transaction ID: {}",
                txid
            )
        }

        // === MARKET COMMANDS ===
        Command::CreateMarket {
            title,
            description,
            dimensions,
            beta,
            trading_fee,
            tags,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::CreateMarketRequest;

            // Extract decision slots from dimensions specification
            let slots = match extract_slots_from_dimensions(&dimensions) {
                Ok(slots) => slots,
                Err(err) => return Ok(err),
            };

            // Parse tags if provided
            let parsed_tags = tags.map(|t| parse_comma_separated(&t));

            // All markets now use dimensional specification
            let request = CreateMarketRequest {
                title: title.clone(),
                description,
                market_type: "dimensional".to_string(),
                decision_slots: slots,
                dimensions: Some(dimensions),
                has_residual: None, // Determined by dimensional specification
                beta: Some(beta),
                trading_fee: Some(trading_fee),
                tags: parsed_tags,
                initial_liquidity: None, // Liquidity is calculated automatically
                fee_sats,
            };

            let txid = rpc_client.market_create(request).await?;
            format!(
                "Market '{}' submitted successfully.\nTransaction ID: {}\nUse 'list-markets' after confirmation to get the market ID.",
                title, txid
            )
        }
        Command::CreateYesNoMarket {
            question,
            decision_slots,
            tags,
            beta,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::CreateMarketRequest;

            // Parse decision slots from comma-separated string
            let slots = parse_comma_separated(&decision_slots);

            if let Err(err) =
                validate_non_empty_list(&slots, "decision slot ID")
            {
                return Ok(err);
            }

            // Parse tags if provided
            let parsed_tags = tags.map(|t| parse_comma_separated(&t));

            let request = CreateMarketRequest {
                title: question.clone(),
                description: format!("Yes/No market: {}", question),
                market_type: "independent".to_string(),
                decision_slots: slots,
                dimensions: None,
                has_residual: None,
                beta: Some(beta),
                trading_fee: Some(0.005),
                tags: parsed_tags,
                initial_liquidity: None, // Liquidity is calculated automatically
                fee_sats,
            };

            let txid = rpc_client.market_create(request).await?;
            format!(
                "Yes/No market '{}' submitted successfully.\nTransaction ID: {}\nUse 'list-markets' after confirmation to get the market ID.",
                question, txid
            )
        }
        Command::ListMarkets => {
            let markets = rpc_client.market_list().await?;
            if markets.is_empty() {
                "No markets in Trading state found.".to_string()
            } else {
                let mut output = String::new();
                output.push_str("Markets in Trading State:\n");
                output.push_str("┌──────────────────┬──────────────────────────┬─────────────────────────┬──────────────┬────────────┐\n");
                output.push_str("│ Market ID        │ Title                    │ Outcomes                │ Volume       │ State      │\n");
                output.push_str("├──────────────────┼──────────────────────────┼─────────────────────────┼──────────────┼────────────┤\n");

                for market in &markets {
                    let short_id = market.market_id.clone();

                    // Truncate title to fit column
                    let short_title = if market.title.len() > 22 {
                        format!("{}...", &market.title[..19])
                    } else {
                        market.title.clone()
                    };

                    // Format outcome count for display
                    let outcomes_display =
                        format!("{} outcomes", market.outcome_count);
                    let short_outcomes = if outcomes_display.len() > 21 {
                        format!("{}...", &outcomes_display[..18])
                    } else {
                        outcomes_display
                    };

                    // Show volume
                    let volume_display = format!("{:.1}", market.volume);
                    let short_volume = if volume_display.len() > 12 {
                        format!("{}...", &volume_display[..9])
                    } else {
                        volume_display
                    };

                    output.push_str(&format!(
                        "│ {:16} │ {:24} │ {:23} │ {:12} │ {:10} │\n",
                        short_id,
                        short_title,
                        short_outcomes,
                        short_volume,
                        market.state
                    ));
                }

                output.push_str("└──────────────────┴──────────────────────────┴─────────────────────────┴──────────────┴────────────┘\n");
                output.push_str(&format!("\nTotal markets: {}", markets.len()));
                output
            }
        }
        Command::ViewMarket { market_id } => {
            let market_details =
                rpc_client.market_get(market_id.clone()).await?;
            match market_details {
                Some(details) => {
                    let mut output = String::new();
                    output.push_str(&format!(
                        "Market Details: {}\n\n",
                        details.market_id
                    ));

                    output.push_str(&format!("Title: {}\n", details.title));
                    output.push_str(&format!(
                        "Description: {}\n",
                        details.description
                    ));
                    output.push_str(&format!(
                        "Market Maker: {}\n",
                        details.market_maker
                    ));
                    output.push_str(&format!("State: {}\n", details.state));

                    if let Some(expiry) = details.expires_at {
                        output
                            .push_str(&format!("Expires: Block {}\n", expiry));
                    } else {
                        output.push_str("Expires: No expiry set\n");
                    }

                    output.push_str(&format!(
                        "Beta Parameter: {:.2}\n",
                        details.beta
                    ));
                    output.push_str(&format!(
                        "Trading Fee: {:.1}%\n",
                        details.trading_fee * 100.0
                    ));
                    output.push_str(&format!(
                        "Created at Height: {}\n",
                        details.created_at_height
                    ));

                    if !details.tags.is_empty() {
                        output.push_str(&format!(
                            "Tags: {}\n",
                            details.tags.join(", ")
                        ));
                    }

                    output.push_str("\nOutcomes:\n");
                    let mut total_volume = 0.0;
                    for (i, outcome) in details.outcomes.iter().enumerate() {
                        output.push_str(&format!(
                            "  {}. {:20} Price: {:6.3}  Probability: {:5.1}%  Volume: {:.0} sats\n",
                            i + 1,
                            outcome.name,
                            outcome.current_price,
                            outcome.probability * 100.0,
                            outcome.volume
                        ));
                        total_volume += outcome.volume;
                    }

                    output.push_str(&format!(
                        "\nTotal Volume: {:.0} sats\n",
                        total_volume
                    ));
                    output.push_str(&format!(
                        "Total Liquidity: {:.0} sats\n",
                        details.liquidity
                    ));
                    output.push_str(&format!(
                        "Treasury: {:.0} sats\n",
                        details.treasury
                    ));

                    if !details.decision_slots.is_empty() {
                        output.push_str("\nDecision Slots:\n");
                        for slot_id in &details.decision_slots {
                            let short_slot = if slot_id.len() > 16 {
                                format!(
                                    "{}...{}",
                                    &slot_id[..8],
                                    &slot_id[slot_id.len() - 8..]
                                )
                            } else {
                                slot_id.clone()
                            };
                            output.push_str(&format!("  - {}\n", short_slot));
                        }
                    }

                    // Display resolution info for ossified markets
                    if let Some(resolution) = &details.resolution {
                        output.push_str(
                            "\n════════════════════════════════════════\n",
                        );
                        output.push_str("  MARKET RESOLUTION\n");
                        output.push_str(
                            "════════════════════════════════════════\n",
                        );
                        output.push_str(&format!("{}\n\n", resolution.summary));

                        if !resolution.winning_outcomes.is_empty() {
                            output.push_str("Winning Outcomes:\n");
                            for outcome in &resolution.winning_outcomes {
                                output.push_str(&format!(
                                    "  • {} (payout: {:.1}%)\n",
                                    outcome.outcome_name,
                                    outcome.final_price * 100.0
                                ));
                            }
                        }
                    }

                    output
                }
                None => format!("Market {} not found", market_id),
            }
        }
        Command::BuyShares {
            market_id,
            outcome_index,
            shares_amount,
            max_cost,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::MarketBuyRequest;
            let request = MarketBuyRequest {
                market_id: market_id.clone(),
                outcome_index,
                shares_amount,
                max_cost: Some(max_cost),
                fee_sats: Some(fee_sats),
                dry_run: Some(false),
            };
            let result = rpc_client.market_buy(request).await?;
            format!(
                "Successfully submitted buy shares transaction!\n\
                Market: {}\n\
                Outcome Index: {}\n\
                Shares: {:.4}\n\
                Cost: {} sats\n\
                Transaction ID: {}",
                market_id,
                outcome_index,
                shares_amount,
                result.cost_sats,
                result.txid.unwrap_or_default()
            )
        }
        Command::CalculateShareCost {
            market_id,
            outcome_index,
            shares_amount,
        } => {
            use truthcoin_dc_app_rpc_api::MarketBuyRequest;
            let request = MarketBuyRequest {
                market_id: market_id.clone(),
                outcome_index,
                shares_amount,
                max_cost: None,
                fee_sats: Some(0),
                dry_run: Some(true),
            };
            let result = rpc_client.market_buy(request).await?;

            format!(
                "Share Purchase Cost Calculation:\n\
                Market: {}\n\
                Outcome Index: {}\n\
                Shares Amount: {:.4}\n\
                Estimated Cost: {} sats ({:.8} BTC)",
                market_id,
                outcome_index,
                shares_amount,
                result.cost_sats,
                result.cost_sats as f64 / 100_000_000.0
            )
        }
        Command::GetUserSharePositions { address } => {
            // If address provided, query that address. Otherwise query all wallet addresses.
            let holdings = if let Some(addr) = address {
                rpc_client.market_positions(addr, None).await?
            } else {
                // Get all wallet addresses and aggregate positions
                let wallet_addresses =
                    rpc_client.get_wallet_addresses().await?;
                let mut all_positions = Vec::new();
                let mut total_value = 0.0;
                let mut total_cost_basis = 0.0;
                let mut active_markets = std::collections::HashSet::new();

                for addr in wallet_addresses {
                    let holdings =
                        rpc_client.market_positions(addr, None).await?;
                    for pos in holdings.positions {
                        active_markets.insert(pos.market_id.clone());
                        total_value += pos.current_value;
                        total_cost_basis += pos.cost_basis;
                        all_positions.push(pos);
                    }
                }

                truthcoin_dc_app_rpc_api::UserHoldings {
                    address: "All Wallet Addresses".to_string(),
                    positions: all_positions,
                    total_value,
                    total_cost_basis,
                    total_unrealized_pnl: total_value - total_cost_basis,
                    active_markets: active_markets.len(),
                    last_updated_height: 0,
                }
            };

            if holdings.positions.is_empty() {
                "No share positions found.".to_string()
            } else {
                let mut output = String::new();
                output.push_str(&format!(
                    "Share Holdings for {}\n\n",
                    holdings.address
                ));
                output.push_str(&format!("Portfolio Summary:\n"));
                output.push_str(&format!(
                    "  Total Value: {:.2} sats\n",
                    holdings.total_value
                ));
                output.push_str(&format!(
                    "  Total Cost Basis: {:.2} sats\n",
                    holdings.total_cost_basis
                ));
                output.push_str(&format!(
                    "  Unrealized P&L: {:.2} sats\n",
                    holdings.total_unrealized_pnl
                ));
                output.push_str(&format!(
                    "  Active Markets: {}\n\n",
                    holdings.active_markets
                ));

                output.push_str("Individual Positions:\n");
                output.push_str("┌──────────────────┬────────────┬──────────────┬───────────────┬────────────────┬─────────────────┐\n");
                output.push_str("│ Market ID        │ Outcome    │ Shares       │ Avg Price     │ Current Value  │ P&L             │\n");
                output.push_str("├──────────────────┼────────────┼──────────────┼───────────────┼────────────────┼─────────────────┤\n");

                for pos in &holdings.positions {
                    let short_market_id = &pos.market_id[..8]; // Show first 8 chars
                    let short_outcome = if pos.outcome_name.len() > 10 {
                        format!("{}...", &pos.outcome_name[..7])
                    } else {
                        pos.outcome_name.clone()
                    };

                    output.push_str(&format!(
                        "│ {:16} │ {:10} │ {:12.4} │ {:13.6} │ {:14.2} │ {:15.2} │\n",
                        short_market_id,
                        short_outcome,
                        pos.shares_held,
                        pos.avg_purchase_price,
                        pos.current_value,
                        pos.unrealized_pnl
                    ));
                }

                output.push_str("└──────────────────┴────────────┴──────────────┴───────────────┴────────────────┴─────────────────┘\n");
                output
            }
        }
        Command::GetMarketSharePositions { address, market_id } => {
            // Use provided address or get default wallet address
            let addr = if let Some(addr) = address {
                addr
            } else {
                rpc_client.get_new_address().await?
            };

            let holdings = rpc_client
                .market_positions(addr, Some(market_id.clone()))
                .await?;
            let positions = holdings.positions;

            if positions.is_empty() {
                format!(
                    "No positions found for market {} and address {}",
                    market_id, addr
                )
            } else {
                let mut output = String::new();
                output.push_str(&format!(
                    "Share Positions for Market {}\n",
                    market_id
                ));
                output.push_str(&format!("User: {}\n\n", addr));

                output.push_str("┌────────────┬────────────────────┬──────────────┬───────────────┬────────────────┬─────────────────┐\n");
                output.push_str("│ Outcome #  │ Outcome Name       │ Shares       │ Avg Price     │ Current Value  │ P&L             │\n");
                output.push_str("├────────────┼────────────────────┼──────────────┼───────────────┼────────────────┼─────────────────┤\n");

                for pos in &positions {
                    let short_outcome = if pos.outcome_name.len() > 18 {
                        format!("{}...", &pos.outcome_name[..15])
                    } else {
                        pos.outcome_name.clone()
                    };

                    output.push_str(&format!(
                        "│ {:10} │ {:18} │ {:12.4} │ {:13.6} │ {:14.2} │ {:15.2} │\n",
                        pos.outcome_index,
                        short_outcome,
                        pos.shares_held,
                        pos.avg_purchase_price,
                        pos.current_value,
                        pos.unrealized_pnl
                    ));
                }

                output.push_str("└────────────┴────────────────────┴──────────────┴───────────────┴────────────────┴─────────────────┘\n");
                output
            }
        }

        Command::CalculateInitialLiquidity {
            beta,
            market_type,
            num_outcomes,
            decision_slots,
            has_residual,
            dimensions,
        } => {
            use truthcoin_dc_app_rpc_api::CalculateInitialLiquidityRequest;

            // Parse decision slots if provided
            let parsed_slots =
                decision_slots.map(|s| parse_comma_separated(&s));

            let request = CalculateInitialLiquidityRequest {
                beta,
                market_type,
                decision_slots: parsed_slots,
                num_outcomes,
                dimensions,
                has_residual,
            };

            let result =
                rpc_client.calculate_initial_liquidity(request).await?;

            format!(
                "Initial Liquidity Calculation:\n\
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
                Beta Parameter:       {:.2}\n\
                Number of Outcomes:   {}\n\
                Market Configuration: {}\n\
                \n\
                Calculation Details:\n\
                {}\n\
                \n\
                Required Initial Liquidity: {} sats\n\
                Minimum Treasury:          {} sats\n\
                \n\
                Formula Used: Initial Liquidity = β × ln(Number of States)\n\
                             = {:.2} × ln({}) = {:.6} ≈ {} sats\n\
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                result.beta,
                result.num_outcomes,
                result.market_config,
                result.outcome_breakdown,
                result.initial_liquidity_sats,
                result.min_treasury_sats,
                result.beta,
                result.num_outcomes,
                result.beta * (result.num_outcomes as f64).ln(),
                result.initial_liquidity_sats
            )
        }

        // === VOTING COMMANDS ===
        Command::RegisterVoter {
            reputation_bond_sats,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::RegisterVoterRequest;

            let request = RegisterVoterRequest {
                reputation_bond_sats,
                fee_sats,
            };

            let txid = rpc_client.vote_register(request).await?;
            format_tx_success("Voter registration", None, &txid)
        }

        Command::SubmitVote {
            decision_id,
            vote_value,
            fee_sats,
        } => {
            use truthcoin_dc_app_rpc_api::VoteBatchItem;

            let vote_items = vec![VoteBatchItem {
                decision_id: decision_id.clone(),
                vote_value,
            }];

            let txid = rpc_client.vote_submit(vote_items, fee_sats).await?;
            format!(
                "Vote submitted successfully!\n\
                Decision: {}\n\
                Vote Value: {:.4}\n\
                Transaction ID: {}",
                decision_id, vote_value, txid
            )
        }

        Command::SubmitVoteBatch { votes, fee_sats } => {
            use truthcoin_dc_app_rpc_api::VoteBatchItem;

            // Parse votes from "decision_id:value,decision_id:value,..." format
            let vote_items: Result<Vec<VoteBatchItem>, String> = votes
                .split(',')
                .map(|pair| {
                    let parts: Vec<&str> = pair.trim().split(':').collect();
                    if parts.len() != 2 {
                        return Err(format!(
                            "Invalid vote format '{}'. Expected 'decision_id:value'",
                            pair
                        ));
                    }
                    let vote_value: f64 = parts[1].parse().map_err(|_| {
                        format!("Invalid vote value '{}'. Must be a number.", parts[1])
                    })?;
                    Ok(VoteBatchItem {
                        decision_id: parts[0].to_string(),
                        vote_value,
                    })
                })
                .collect();

            let vote_items = match vote_items {
                Ok(items) => items,
                Err(err) => return Ok(err),
            };

            let vote_count = vote_items.len();

            let txid = rpc_client.vote_submit(vote_items, fee_sats).await?;
            format!(
                "Vote batch submitted successfully!\n\
                Votes: {} decisions\n\
                Transaction ID: {}",
                vote_count, txid
            )
        }

        Command::GetVoterInfo { address } => {
            let voter_info = rpc_client.vote_voter(address).await?;
            match voter_info {
                Some(info) => {
                    let mut output = String::new();
                    output.push_str("Voter Information:\n");
                    output.push_str("==================\n\n");
                    output.push_str(&format!("Address: {}\n", info.address));
                    output.push_str(&format!(
                        "Reputation: {:.6}\n",
                        info.reputation
                    ));
                    output.push_str(&format!(
                        "Total Votes: {}\n",
                        info.total_votes
                    ));
                    output.push_str(&format!(
                        "Periods Active: {}\n",
                        info.periods_active
                    ));
                    output.push_str(&format!(
                        "Accuracy Score: {:.2}%\n",
                        info.accuracy_score * 100.0
                    ));
                    output.push_str(&format!(
                        "Registered at Height: {}\n",
                        info.registered_at_height
                    ));
                    output.push_str(&format!(
                        "Status: {}\n",
                        if info.is_active { "Active" } else { "Inactive" }
                    ));
                    output
                }
                None => format!("Voter {} not found", address),
            }
        }

        Command::GetVotingPeriodDetails { period_id } => {
            let details = rpc_client.vote_period(Some(period_id)).await?;
            match details {
                Some(info) => {
                    let mut output = String::new();
                    output.push_str(&format!(
                        "Voting Period {} Details:\n",
                        period_id
                    ));
                    output.push_str("=============================\n\n");
                    output.push_str(&format!("Status: {}\n", info.status));
                    output.push_str(&format!(
                        "Start Height: {}\n",
                        info.start_height
                    ));
                    output.push_str(&format!(
                        "End Height: {}\n",
                        info.end_height
                    ));
                    output.push_str(&format!(
                        "Total Voters: {}\n",
                        info.stats.total_voters
                    ));
                    output.push_str(&format!(
                        "Active Voters: {}\n",
                        info.stats.active_voters
                    ));
                    output.push_str(&format!(
                        "Total Votes: {}\n",
                        info.stats.total_votes
                    ));
                    output.push_str(&format!(
                        "Consensus Reached: {}\n",
                        if info.consensus.is_some() {
                            "Yes"
                        } else {
                            "No"
                        }
                    ));
                    if !info.decisions.is_empty() {
                        output.push_str(&format!(
                            "\nDecisions ({}):\n",
                            info.decisions.len()
                        ));
                        for decision in &info.decisions {
                            output.push_str(&format!(
                                "  - {} ({})\n",
                                decision.slot_id_hex, decision.question
                            ));
                        }
                    }
                    output
                }
                None => format!("Voting period {} not found", period_id),
            }
        }

        Command::GetVoterVotes { address, period_id } => {
            use truthcoin_dc_app_rpc_api::VoteFilter;
            let filter = VoteFilter {
                voter: Some(address),
                decision_id: None,
                period_id,
            };
            let votes = rpc_client.vote_list(filter).await?;
            if votes.is_empty() {
                match period_id {
                    Some(pid) => {
                        format!(
                            "No votes found for {} in period {}",
                            address, pid
                        )
                    }
                    None => format!("No votes found for {}", address),
                }
            } else {
                let mut output = String::new();
                output.push_str(&format!("Votes by {}:\n", address));
                output.push_str("=====================================\n\n");
                output.push_str("┌────────────────────┬──────────────┬────────────┬────────────┬──────────┐\n");
                output.push_str("│ Decision ID        │ Vote Value   │ Period     │ Height     │ Batch    │\n");
                output.push_str("├────────────────────┼──────────────┼────────────┼────────────┼──────────┤\n");

                for vote in &votes {
                    let short_decision = if vote.decision_id.len() > 18 {
                        format!("{}...", &vote.decision_id[..15])
                    } else {
                        vote.decision_id.clone()
                    };

                    output.push_str(&format!(
                        "│ {:18} │ {:12.4} │ {:10} │ {:10} │ {:8} │\n",
                        short_decision,
                        vote.vote_value,
                        vote.period_id,
                        vote.block_height,
                        if vote.is_batch_vote { "Yes" } else { "No" }
                    ));
                }

                output.push_str("└────────────────────┴──────────────┴────────────┴────────────┴──────────┘\n");
                output.push_str(&format!("\nTotal votes: {}", votes.len()));
                output
            }
        }

        Command::GetDecisionVotes { slot_id } => {
            use truthcoin_dc_app_rpc_api::VoteFilter;
            let filter = VoteFilter {
                voter: None,
                decision_id: Some(slot_id.clone()),
                period_id: None,
            };
            let votes = rpc_client.vote_list(filter).await?;
            if votes.is_empty() {
                format!("No votes found for decision {}", slot_id)
            } else {
                let mut output = String::new();
                output.push_str(&format!("Votes for Decision {}:\n", slot_id));
                output.push_str("=====================================\n\n");
                output.push_str("┌────────────────────────────────┬──────────────┬────────────┬────────────┐\n");
                output.push_str("│ Voter Address                  │ Vote Value   │ Period     │ Height     │\n");
                output.push_str("├────────────────────────────────┼──────────────┼────────────┼────────────┤\n");

                for vote in &votes {
                    let short_addr = if vote.voter_address.len() > 30 {
                        format!("{}...", &vote.voter_address[..27])
                    } else {
                        vote.voter_address.clone()
                    };

                    output.push_str(&format!(
                        "│ {:30} │ {:12.4} │ {:10} │ {:10} │\n",
                        short_addr,
                        vote.vote_value,
                        vote.period_id,
                        vote.block_height
                    ));
                }

                output.push_str("└────────────────────────────────┴──────────────┴────────────┴────────────┘\n");
                output.push_str(&format!("\nTotal votes: {}", votes.len()));
                output
            }
        }

        Command::GetVoterParticipation { address, period_id } => {
            let voter_info = rpc_client.vote_voter(address).await?;
            match voter_info {
                Some(info) => {
                    let mut output = String::new();
                    output.push_str("Voter Participation Summary:\n");
                    output.push_str("============================\n\n");
                    output.push_str(&format!("Address: {}\n", info.address));
                    output.push_str(&format!("Period: {}\n", period_id));
                    if let Some(participation) =
                        &info.current_period_participation
                    {
                        output.push_str(&format!(
                            "Votes Cast: {}\n",
                            participation.votes_cast
                        ));
                        output.push_str(&format!(
                            "Decisions Available: {}\n",
                            participation.decisions_available
                        ));
                        output.push_str(&format!(
                            "Participation Rate: {:.1}%\n",
                            participation.participation_rate * 100.0
                        ));
                    } else {
                        output.push_str(
                            "No current period participation data available.\n",
                        );
                    }
                    output
                }
                None => format!(
                    "No participation record found for {} in period {}",
                    address, period_id
                ),
            }
        }

        Command::ListVoters => {
            let voters = rpc_client.vote_voters().await?;
            if voters.is_empty() {
                "No registered voters found.".to_string()
            } else {
                let mut output = String::new();
                output.push_str("Registered Voters:\n");
                output.push_str("==================\n\n");
                output.push_str("┌────────────────────────────────┬────────────┬────────────┬────────────┬──────────┐\n");
                output.push_str("│ Address                        │ Reputation │ Votes      │ Periods    │ Active   │\n");
                output.push_str("├────────────────────────────────┼────────────┼────────────┼────────────┼──────────┤\n");

                for voter in &voters {
                    let short_addr = if voter.address.len() > 30 {
                        format!("{}...", &voter.address[..27])
                    } else {
                        voter.address.clone()
                    };

                    output.push_str(&format!(
                        "│ {:30} │ {:10.4} │ {:10} │ {:10} │ {:8} │\n",
                        short_addr,
                        voter.reputation,
                        voter.total_votes,
                        voter.periods_active,
                        if voter.is_active { "Yes" } else { "No" }
                    ));
                }

                output.push_str("└────────────────────────────────┴────────────┴────────────┴────────────┴──────────┘\n");
                output.push_str(&format!("\nTotal voters: {}", voters.len()));
                output
            }
        }

        Command::IsRegisteredVoter { address } => {
            let voter_info = rpc_client.vote_voter(address).await?;
            if voter_info.is_some() {
                format!("Address {} is a registered voter", address)
            } else {
                format!("Address {} is NOT a registered voter", address)
            }
        }

        Command::GetVotecoinBalance { address } => {
            let balance = rpc_client.votecoin_balance(address).await?;
            format!("Votecoin balance for {}: {} VTC", address, balance)
        }

        Command::GetCurrentVotingStats => {
            let stats = rpc_client.vote_period(None).await?;
            match stats {
                Some(info) => {
                    let mut output = String::new();
                    output.push_str("Current Voting Period Statistics:\n");
                    output.push_str("==================================\n\n");
                    output
                        .push_str(&format!("Period ID: {}\n", info.period_id));
                    output.push_str(&format!("Status: {}\n", info.status));
                    output.push_str(&format!(
                        "Total Voters: {}\n",
                        info.stats.total_voters
                    ));
                    output.push_str(&format!(
                        "Active Voters: {}\n",
                        info.stats.active_voters
                    ));
                    output.push_str(&format!(
                        "Total Votes Cast: {}\n",
                        info.stats.total_votes
                    ));
                    output.push_str(&format!(
                        "Decisions: {}\n",
                        info.decisions.len()
                    ));
                    output.push_str(&format!(
                        "Consensus Reached: {}\n",
                        if info.consensus.is_some() {
                            "Yes"
                        } else {
                            "No"
                        }
                    ));
                    output
                }
                None => "No active voting period found.".to_string(),
            }
        }

        Command::GetVotingConsensusResults { period_id } => {
            let period_info = rpc_client.vote_period(Some(period_id)).await?;
            match period_info {
                Some(info) => {
                    let mut output = String::new();
                    output.push_str(&format!(
                        "Voting Consensus Results for Period {}:\n",
                        period_id
                    ));
                    output.push_str(
                        "==========================================\n\n",
                    );
                    output.push_str(&format!("Status: {}\n", info.status));

                    if let Some(consensus) = &info.consensus {
                        output.push_str(&format!(
                            "Algorithm Version: {}\n",
                            consensus.algorithm_version
                        ));
                        output.push_str(&format!(
                            "Vote Matrix: {} voters × {} decisions\n",
                            consensus.vote_matrix_dimensions.0,
                            consensus.vote_matrix_dimensions.1
                        ));
                        output.push_str(&format!(
                            "Explained Variance: {:.2}%\n",
                            consensus.explained_variance * 100.0
                        ));
                        output.push_str(&format!(
                            "Certainty: {:.4}\n",
                            consensus.certainty
                        ));

                        if !consensus.first_loading.is_empty() {
                            output.push_str("\nFirst Principal Component:\n");
                            for (i, val) in
                                consensus.first_loading.iter().enumerate()
                            {
                                output.push_str(&format!(
                                    "  Decision {}: {:.4}\n",
                                    i, val
                                ));
                            }
                        }

                        if !consensus.outcomes.is_empty() {
                            output.push_str("\nDecision Outcomes:\n");
                            for (decision_id, outcome) in &consensus.outcomes {
                                output.push_str(&format!(
                                    "  {}: {:.4}\n",
                                    decision_id, outcome
                                ));
                            }
                        }

                        if !consensus.reputation_updates.is_empty() {
                            output.push_str("\nReputation Updates:\n");
                            for (voter, update) in &consensus.reputation_updates
                            {
                                let short_voter = if voter.len() > 20 {
                                    format!("{}...", &voter[..17])
                                } else {
                                    voter.clone()
                                };
                                output.push_str(&format!(
                                    "  {}: {:.4} → {:.4} (compliance: {:.2}%)\n",
                                    short_voter,
                                    update.old_reputation,
                                    update.new_reputation,
                                    update.compliance_score * 100.0
                                ));
                            }
                        }

                        if !consensus.outliers.is_empty() {
                            output.push_str("\nOutliers Detected:\n");
                            for outlier in &consensus.outliers {
                                output.push_str(&format!("  - {}\n", outlier));
                            }
                        }
                    } else {
                        output.push_str("No consensus data available yet.\n");
                    }

                    output
                }
                None => format!("Voting period {} not found", period_id),
            }
        }

        Command::GetVoterReputation { voter_id } => {
            // Parse voter_id as Address
            let address: Address = voter_id
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid address format"))?;
            let voter_info = rpc_client.vote_voter(address).await?;
            match voter_info {
                Some(info) => format!(
                    "Reputation for {}: {:.6}",
                    voter_id, info.reputation
                ),
                None => format!("Voter {} not found", voter_id),
            }
        }

        Command::GetVotingPeriodStatus { period_id } => {
            let period_info = rpc_client.vote_period(Some(period_id)).await?;
            match period_info {
                Some(info) => format!(
                    "Voting period {} status: {}",
                    period_id, info.status
                ),
                None => format!("Voting period {} not found", period_id),
            }
        }

        Command::GetRedistributionSummary { period_id } => {
            let period_info = rpc_client.vote_period(Some(period_id)).await?;
            match period_info.and_then(|p| p.redistribution) {
                Some(info) => {
                    let mut output = String::new();
                    output.push_str(&format!(
                        "VoteCoin Redistribution Summary for Period {}:\n",
                        period_id
                    ));
                    output.push_str(
                        "===============================================\n\n",
                    );
                    output.push_str(&format!(
                        "Total Redistributed: {} sats\n",
                        info.total_redistributed
                    ));
                    output.push_str(&format!(
                        "Winners: {}\n",
                        info.winners_count
                    ));
                    output
                        .push_str(&format!("Losers: {}\n", info.losers_count));
                    output.push_str(&format!(
                        "Unchanged: {}\n",
                        info.unchanged_count
                    ));
                    output.push_str(&format!(
                        "Conservation Check: {} (should be 0)\n",
                        info.conservation_check
                    ));
                    output.push_str(&format!(
                        "Block Height: {}\n",
                        info.block_height
                    ));
                    output.push_str(&format!(
                        "Applied: {}\n",
                        if info.is_applied { "Yes" } else { "No" }
                    ));

                    if !info.slots_affected.is_empty() {
                        output.push_str(&format!(
                            "\nSlots Affected ({}):\n",
                            info.slots_affected.len()
                        ));
                        for slot in &info.slots_affected {
                            output.push_str(&format!("  - {}\n", slot));
                        }
                    }
                    output
                }
                None => {
                    format!(
                        "No redistribution summary found for period {}",
                        period_id
                    )
                }
            }
        }
    })
}

fn set_tracing_subscriber() -> anyhow::Result<()> {
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .with_file(true)
        .with_line_number(true);

    let subscriber = tracing_subscriber::registry().with(stdout_layer);
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

impl Cli {
    pub async fn run(self) -> anyhow::Result<String> {
        if self.verbose {
            set_tracing_subscriber()?;
        }

        let request_id = uuid::Uuid::new_v4().as_simple().to_string();
        tracing::info!(%request_id);
        let builder = HttpClientBuilder::default()
            .request_timeout(Duration::from_secs(self.timeout_secs))
            .set_max_logging_length(1024)
            .set_headers(HeaderMap::from_iter([(
                http::header::HeaderName::from_static("x-request-id"),
                http::header::HeaderValue::from_str(&request_id)?,
            )]));
        let client = builder.build(self.rpc_url())?;

        let result = handle_command(&client, self.command).await?;
        Ok(result)
    }
}
