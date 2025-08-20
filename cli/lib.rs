use std::{
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use clap::{ArgAction, Parser, Subcommand};
use http::HeaderMap;
use jsonrpsee::{core::client::ClientT, http_client::HttpClientBuilder};
use tracing_subscriber::layer::SubscriberExt as _;
use truthcoin_dc::{
    authorization::{Dst, Signature},
    types::{
        Address, AssetId, BlockHash, EncryptionPubKey, THIS_SIDECHAIN, Txid,
        VerifyingKey,
    },
};
use truthcoin_dc_app_rpc_api::RpcClient;
use url::{Host, Url};

#[derive(Clone, Debug, Subcommand)]
#[command(arg_required_else_help(true))]
pub enum Command {
    /// Burn an AMM position
    AmmBurn {
        #[arg(long)]
        asset0: AssetId,
        #[arg(long)]
        asset1: AssetId,
        #[arg(long)]
        lp_token_amount: u64,
    },
    /// Mint an AMM position
    AmmMint {
        #[arg(long)]
        asset0: AssetId,
        #[arg(long)]
        asset1: AssetId,
        #[arg(long)]
        amount0: u64,
        #[arg(long)]
        amount1: u64,
    },
    /// Returns the amount of `asset_receive` to receive
    AmmSwap {
        #[arg(long)]
        asset_spend: AssetId,
        #[arg(long)]
        asset_receive: AssetId,
        #[arg(long)]
        amount_spend: u64,
    },
    /// Get Bitcoin balance in sats
    BitcoinBalance,
    /// Connect to a peer
    ConnectPeer {
        addr: SocketAddr,
    },
    /// Deposit to address
    CreateDeposit {
        address: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long)]
        fee_sats: u64,
    },
    /// Decrypt a message with the specified encryption key corresponding to
    /// the specified encryption pubkey
    DecryptMsg {
        #[arg(long)]
        encryption_pubkey: EncryptionPubKey,
        #[arg(long)]
        msg: String,
        /// If set, decode as UTF-8
        #[arg(long)]
        utf8: bool,
    },
    /// Encrypt a message to the specified encryption pubkey.
    /// Returns the ciphertext as a hex string.
    EncryptMsg {
        #[arg(long)]
        encryption_pubkey: EncryptionPubKey,
        #[arg(long)]
        msg: String,
    },
    /// Format a deposit address
    FormatDepositAddress {
        address: Address,
    },
    /// Generate a mnemonic seed phrase
    GenerateMnemonic,
    /// Get the state of the specified AMM pool
    GetAmmPoolState {
        #[arg(long)]
        asset0: AssetId,
        #[arg(long)]
        asset1: AssetId,
    },
    /// Get the current price for the specified pair
    GetAmmPrice {
        #[arg(long)]
        base: AssetId,
        #[arg(long)]
        quote: AssetId,
    },
    /// Get the best mainchain block hash
    GetBestMainchainBlockHash,
    /// Get the best sidechain block hash
    GetBestSidechainBlockHash,
    /// Get block data
    GetBlock {
        block_hash: BlockHash,
    },
    /// Get the current block count
    GetBlockcount,
    /// Get mainchain blocks that commit to a specified block hash
    GetBmmInclusions {
        block_hash: truthcoin_dc::types::BlockHash,
    },
    /// Get a new address
    GetNewAddress,
    /// Get a new encryption pubkey
    GetNewEncryptionKey,
    /// Get a new verifying key
    GetNewVerifyingKey,
    /// Get wallet addresses, sorted by base58 encoding
    /// Get transaction by txid
    GetTransaction {
        txid: Txid,
    },
    /// Get information about a transaction in the current chain
    GetTransactionInfo {
        txid: Txid,
    },
    GetWalletAddresses,
    /// Get wallet UTXOs
    GetWalletUtxos,
    /// Get the height of the latest failed withdrawal bundle
    LatestFailedWithdrawalBundleHeight,
    /// List peers
    ListPeers,
    /// List all UTXOs
    ListUtxos,
    /// Attempt to mine a sidechain block
    Mine {
        #[arg(long)]
        fee_sats: Option<u64>,
    },
    /// List unconfirmed owned UTXOs
    MyUnconfirmedUtxos,
    /// List owned UTXOs
    MyUtxos,
    /// Show OpenAPI schema
    #[command(name = "openapi-schema")]
    OpenApiSchema,
    /// Get pending withdrawal bundle
    PendingWithdrawalBundle,
    /// Remove a tx from the mempool
    RemoveFromMempool {
        txid: Txid,
    },
    /// Set the wallet seed from a mnemonic seed phrase
    SetSeedFromMnemonic {
        mnemonic: String,
    },
    /// Get total sidechain wealth
    SidechainWealth,
    /// Sign an arbitrary message with the specified verifying key
    SignArbitraryMsg {
        #[arg(long)]
        verifying_key: VerifyingKey,
        #[arg(long)]
        msg: String,
    },
    /// Sign an arbitrary message with the secret key for the specified address
    SignArbitraryMsgAsAddr {
        #[arg(long)]
        address: Address,
        #[arg(long)]
        msg: String,
    },
    /// Stop the node
    Stop,
    /// Transfer funds to the specified address
    Transfer {
        dest: Address,
        #[arg(long)]
        value_sats: u64,
        #[arg(long)]
        fee_sats: u64,
    },
    /// Transfer votecoin to the specified address
    TransferVotecoin {
        dest: Address,
        #[arg(long)]
        amount: u32,
        #[arg(long)]
        fee_sats: u64,
    },
    /// Verify a signature on a message against the specified verifying key.
    /// Returns `true` if the signature is valid
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
    /// Initiate a withdrawal to the specified mainchain address
    Withdraw {
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        #[arg(long)]
        amount_sats: u64,
        #[arg(long)]
        fee_sats: u64,
        #[arg(long)]
        mainchain_fee_sats: u64,
    },
    /// Get all available slots by period
    SlotsListAll,
    /// Get slots for a specific period
    SlotsGetQuarter {
        quarter: u32,
    },
    /// Show slot system status
    SlotsStatus,
    /// Convert timestamp to period
    SlotsConvertTimestamp {
        timestamp: u64,
    },
    /// Claim a decision slot
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
        #[arg(long)]
        fee_sats: u64,
    },
    /// Create a prediction market
    CreateMarket {
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        dimensions: String, // JSON-like dimension specification: "[slot1,[slot2,slot3],slot4]"
        #[arg(long)]
        b: Option<f64>, // LMSR beta parameter
        #[arg(long)]
        trading_fee: Option<f64>, // Trading fee percentage
        #[arg(long)]
        tags: Option<String>, // Comma-separated tags
        #[arg(long)]
        fee_sats: u64, // Transaction fee
    },
    /// Get available slots in a period
    GetAvailableSlots {
        #[arg(long)]
        period_index: u32,
    },
    /// Get slot by ID
    GetSlotById {
        #[arg(long)]
        slot_id_hex: String,
    },
    /// Get claimed slots in a period
    GetClaimedSlots {
        #[arg(long)]
        period_index: u32,
    },
    /// Check if slot is in voting period
    IsSlotInVoting {
        #[arg(long)]
        slot_id_hex: String,
    },
    /// Get periods currently in voting phase
    GetVotingPeriods,
    /// Get ossified slots (slots whose voting period has ended)
    GetOssifiedSlots,
    /// List all markets in Trading state
    ListMarkets,
    /// View detailed information for a specific market
    ViewMarket {
        /// Market ID as hex-encoded hash
        market_id: String,
    },
}

const DEFAULT_RPC_HOST: Host = Host::Ipv4(Ipv4Addr::LOCALHOST);

const DEFAULT_RPC_PORT: u16 = 6000 + THIS_SIDECHAIN as u16;

const DEFAULT_TIMEOUT_SECS: u64 = 60;

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
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
/// Handle a command, returning CLI output
async fn handle_command<RpcClient>(
    rpc_client: &RpcClient,
    command: Command,
) -> anyhow::Result<String>
where
    RpcClient: ClientT + Sync,
{
    Ok(match command {
        Command::AmmBurn {
            asset0,
            asset1,
            lp_token_amount,
        } => {
            let txid =
                rpc_client.amm_burn(asset0, asset1, lp_token_amount).await?;
            format!("{txid}")
        }
        Command::AmmMint {
            asset0,
            asset1,
            amount0,
            amount1,
        } => {
            let txid = rpc_client
                .amm_mint(asset0, asset1, amount0, amount1)
                .await?;
            format!("{txid}")
        }
        Command::AmmSwap {
            asset_spend,
            asset_receive,
            amount_spend,
        } => {
            let amount = rpc_client
                .amm_swap(asset_spend, asset_receive, amount_spend)
                .await?;
            format!("{amount}")
        }
        Command::BitcoinBalance => {
            let balance = rpc_client.bitcoin_balance().await?;
            serde_json::to_string_pretty(&balance)?
        }
        Command::ConnectPeer { addr } => {
            let () = rpc_client.connect_peer(addr).await?;
            String::default()
        }
        Command::CreateDeposit {
            address,
            value_sats,
            fee_sats,
        } => {
            let txid = rpc_client
                .create_deposit(address, value_sats, fee_sats)
                .await?;
            format!("{txid}")
        }
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
        Command::EncryptMsg {
            encryption_pubkey,
            msg,
        } => rpc_client.encrypt_msg(encryption_pubkey, msg).await?,
        Command::FormatDepositAddress { address } => {
            rpc_client.format_deposit_address(address).await?
        }
        Command::GenerateMnemonic => rpc_client.generate_mnemonic().await?,
        Command::GetAmmPoolState { asset0, asset1 } => {
            let state = rpc_client.get_amm_pool_state(asset0, asset1).await?;
            serde_json::to_string_pretty(&state)?
        }
        Command::GetAmmPrice { base, quote } => {
            let price = rpc_client.get_amm_price(base, quote).await?;
            serde_json::to_string_pretty(&price)?
        }
        Command::GetBlock { block_hash } => {
            let block = rpc_client.get_block(block_hash).await?;
            serde_json::to_string_pretty(&block)?
        }
        Command::GetBlockcount => {
            let blockcount = rpc_client.getblockcount().await?;
            format!("{blockcount}")
        }
        Command::GetBestMainchainBlockHash => {
            let block_hash = rpc_client.get_best_mainchain_block_hash().await?;
            serde_json::to_string_pretty(&block_hash)?
        }
        Command::GetBestSidechainBlockHash => {
            let block_hash = rpc_client.get_best_sidechain_block_hash().await?;
            serde_json::to_string_pretty(&block_hash)?
        }
        Command::GetBmmInclusions { block_hash } => {
            let bmm_inclusions =
                rpc_client.get_bmm_inclusions(block_hash).await?;
            serde_json::to_string_pretty(&bmm_inclusions)?
        }
        Command::GetNewAddress => {
            let address = rpc_client.get_new_address().await?;
            format!("{address}")
        }
        Command::GetNewEncryptionKey => {
            let epk = rpc_client.get_new_encryption_key().await?;
            format!("{epk}")
        }
        Command::GetNewVerifyingKey => {
            let vk = rpc_client.get_new_verifying_key().await?;
            format!("{vk}")
        }
        Command::GetTransaction { txid } => {
            let tx = rpc_client.get_transaction(txid).await?;
            serde_json::to_string_pretty(&tx)?
        }
        Command::GetTransactionInfo { txid } => {
            let tx_info = rpc_client.get_transaction_info(txid).await?;
            serde_json::to_string_pretty(&tx_info)?
        }
        Command::GetWalletAddresses => {
            let addresses = rpc_client.get_wallet_addresses().await?;
            serde_json::to_string_pretty(&addresses)?
        }
        Command::GetWalletUtxos => {
            let utxos = rpc_client.get_wallet_utxos().await?;
            serde_json::to_string_pretty(&utxos)?
        }
        Command::LatestFailedWithdrawalBundleHeight => {
            let height =
                rpc_client.latest_failed_withdrawal_bundle_height().await?;
            serde_json::to_string_pretty(&height)?
        }
        Command::ListPeers => {
            let peers = rpc_client.list_peers().await?;
            serde_json::to_string_pretty(&peers)?
        }
        Command::ListUtxos => {
            let utxos = rpc_client.list_utxos().await?;
            serde_json::to_string_pretty(&utxos)?
        }
        Command::Mine { fee_sats } => {
            let () = rpc_client.mine(fee_sats).await?;
            String::default()
        }
        Command::MyUnconfirmedUtxos => {
            let utxos = rpc_client.my_unconfirmed_utxos().await?;
            serde_json::to_string_pretty(&utxos)?
        }
        Command::MyUtxos => {
            let utxos = rpc_client.my_utxos().await?;
            serde_json::to_string_pretty(&utxos)?
        }
        Command::OpenApiSchema => {
            let openapi =
                <truthcoin_dc_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi(
                );
            openapi.to_pretty_json()?
        }
        Command::PendingWithdrawalBundle => {
            let withdrawal_bundle =
                rpc_client.pending_withdrawal_bundle().await?;
            serde_json::to_string_pretty(&withdrawal_bundle)?
        }
        Command::RemoveFromMempool { txid } => {
            let () = rpc_client.remove_from_mempool(txid).await?;
            String::default()
        }
        Command::SetSeedFromMnemonic { mnemonic } => {
            let () = rpc_client.set_seed_from_mnemonic(mnemonic).await?;
            String::default()
        }
        Command::SidechainWealth => {
            let wealth = rpc_client.sidechain_wealth_sats().await?;
            format!("{wealth}")
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
        Command::Stop => {
            let () = rpc_client.stop().await?;
            String::default()
        }
        Command::Transfer {
            dest,
            value_sats,
            fee_sats,
        } => {
            let txid = rpc_client
                .transfer(dest, value_sats, fee_sats, None)
                .await?;
            format!("{txid}")
        }
        Command::TransferVotecoin {
            dest,
            amount,
            fee_sats,
        } => {
            let txid = rpc_client
                .transfer_votecoin(dest, amount, fee_sats, None)
                .await?;
            format!("{txid}")
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
            format!("{txid}")
        }
        Command::SlotsListAll => {
            let slots = rpc_client.slots_list_all().await?;
            let mut output = String::new();
            output.push_str("Available Slots by Period:\n");
            output.push_str("========================\n\n");
            if slots.is_empty() {
                output.push_str("No slots minted yet.\n");
            } else {
                for slot_info in slots {
                    let period_name =
                        rpc_client.quarter_to_string(slot_info.period).await?;
                    output.push_str(&format!(
                        "{}: {} slots\n",
                        period_name, slot_info.slots
                    ));
                }
            }
            output
        }
        Command::SlotsGetQuarter { quarter } => {
            let slot_count = rpc_client.slots_get_quarter(quarter).await?;
            let period_name = rpc_client.quarter_to_string(quarter).await?;
            format!("{}: {} slots", period_name, slot_count)
        }
        Command::SlotsStatus => {
            let status = rpc_client.slots_status().await?;
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
        Command::SlotsConvertTimestamp { timestamp } => {
            let quarter = rpc_client.timestamp_to_quarter(timestamp).await?;
            let period_name = rpc_client.quarter_to_string(quarter).await?;
            format!(
                "Timestamp {} converts to: {} (Period {})",
                timestamp, period_name, quarter
            )
        }
        Command::GetAvailableSlots { period_index } => {
            let available_slots = rpc_client
                .get_available_slots_in_period(period_index)
                .await?;
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
                .claim_decision_slot(
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
        Command::CreateMarket {
            title,
            description,
            dimensions,
            b,
            trading_fee,
            tags,
            fee_sats,
        } => {
            let tags_vec = tags.map(|t| 
                t.split(',')
                .map(|s| s.trim().to_string())
                .collect()
            );

            let result = rpc_client
                .create_market_dimensional(
                    title,
                    description,
                    dimensions,
                    b,
                    trading_fee,
                    tags_vec,
                    fee_sats,
                )
                .await?;
            format!("Market created successfully: {}", result)
        }
        Command::GetSlotById { slot_id_hex } => {
            let slot = rpc_client.get_slot_by_id(slot_id_hex.clone()).await?;
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
            let claimed_slots =
                rpc_client.get_claimed_slots_in_period(period_index).await?;
            if claimed_slots.is_empty() {
                format!("No claimed slots found in period {}", period_index)
            } else {
                let mut result = format!(
                    "Claimed slots in period {} ({} total):\n",
                    period_index,
                    claimed_slots.len()
                );
                for slot in claimed_slots {
                    result.push_str(&format!(
                        "  Slot {} ({}): {} | {} | Market Maker: {} | \"{}\"\n",
                        slot.slot_index,
                        slot.slot_id_hex,
                        if slot.is_standard {
                            "Standard"
                        } else {
                            "Non-Standard"
                        },
                        if slot.is_scaled { "Scaled" } else { "Binary" },
                        &slot.market_maker_pubkey_hash[..8], // Show first 8 chars of hash
                        slot.question_preview
                    ));
                }
                result
            }
        }
        Command::IsSlotInVoting { slot_id_hex } => {
            let is_voting =
                rpc_client.is_slot_in_voting(slot_id_hex.clone()).await?;
            if is_voting {
                format!("Slot {} is in voting period", slot_id_hex)
            } else {
                format!("Slot {} is NOT in voting period", slot_id_hex)
            }
        }
        Command::GetVotingPeriods => {
            let voting_periods = rpc_client.get_voting_periods().await?;
            if voting_periods.is_empty() {
                "No periods currently in voting phase".to_string()
            } else {
                let mut result = format!(
                    "Voting Periods ({} total):\n",
                    voting_periods.len()
                );
                result.push_str("================================\n");
                for period_info in voting_periods {
                    result.push_str(&format!(
                        "Period {}: {}/{} slots claimed ({:.1}%)\n",
                        period_info.period,
                        period_info.claimed_slots,
                        period_info.total_slots,
                        (period_info.claimed_slots as f64
                            / period_info.total_slots as f64)
                            * 100.0
                    ));
                }
                result
            }
        }
        Command::GetOssifiedSlots => {
            let ossified_slots = rpc_client.get_ossified_slots().await?;
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
                    if let Some(decision) = slot.decision {
                        let question_preview = if decision.question.len() > 50 {
                            format!("{}...", &decision.question[..50])
                        } else {
                            decision.question
                        };
                        result.push_str(&format!(
                            "{} - {}\n",
                            if decision.is_standard { "Standard" } else { "Non-standard" },
                            question_preview
                        ));
                    } else {
                        result.push_str("Empty slot\n");
                    }
                }
                result
            }
        }
        Command::ListMarkets => {
            let markets = rpc_client.list_markets().await?;
            if markets.is_empty() {
                "No markets in Trading state found.".to_string()
            } else {
                let mut output = String::new();
                output.push_str("Markets in Trading State:\n");
                output.push_str("┌──────────────────┬──────────────────────────┬─────────────────────────┬──────────────┬────────────┐\n");
                output.push_str("│ Market ID        │ Title                    │ Outcomes                │ Prices       │ State      │\n");
                output.push_str("├──────────────────┼──────────────────────────┼─────────────────────────┼──────────────┼────────────┤\n");
                
                for market in &markets {
                    // Truncate market ID to first 8 characters
                    // With 6-byte MarketId (12 hex chars), no truncation needed
                    let short_id = market.market_id.clone();
                    
                    // Truncate title to fit column
                    let short_title = if market.title.len() > 22 {
                        format!("{}...", &market.title[..19])
                    } else {
                        market.title.clone()
                    };
                    
                    // Format outcomes (limit to 2-3 for display)
                    let outcomes_display = if market.outcomes.len() <= 2 {
                        market.outcomes.join(", ")
                    } else {
                        format!("{}, {}...", market.outcomes[0], market.outcomes[1])
                    };
                    let short_outcomes = if outcomes_display.len() > 21 {
                        format!("{}...", &outcomes_display[..18])
                    } else {
                        outcomes_display
                    };
                    
                    // Format prices
                    let prices_display = if market.current_prices.len() <= 3 {
                        market.current_prices.iter()
                            .map(|p| format!("{:.3}", p))
                            .collect::<Vec<_>>()
                            .join(", ")
                    } else {
                        format!("{:.3}, {:.3}...", market.current_prices[0], market.current_prices[1])
                    };
                    let short_prices = if prices_display.len() > 12 {
                        format!("{}...", &prices_display[..9])
                    } else {
                        prices_display
                    };
                    
                    output.push_str(&format!(
                        "│ {:16} │ {:24} │ {:23} │ {:12} │ {:10} │\n",
                        short_id, short_title, short_outcomes, short_prices, market.state
                    ));
                }
                
                output.push_str("└──────────────────┴──────────────────────────┴─────────────────────────┴──────────────┴────────────┘\n");
                output.push_str(&format!("\nTotal markets: {}", markets.len()));
                output
            }
        }
        Command::ViewMarket { market_id } => {
            let market_details = rpc_client.view_market(market_id.clone()).await?;
            match market_details {
                Some(details) => {
                    let mut output = String::new();
                    output.push_str(&format!("Market Details: {}\n\n", details.market_id));
                    
                    output.push_str(&format!("Title: {}\n", details.title));
                    output.push_str(&format!("Description: {}\n", details.description));
                    output.push_str(&format!("Market Maker: {}\n", details.market_maker));
                    output.push_str(&format!("State: {}\n", details.trading_state));
                    
                    if let Some(expiry) = details.expiry {
                        output.push_str(&format!("Expires: Block {}\n", expiry));
                    } else {
                        output.push_str("Expires: No expiry set\n");
                    }
                    
                    output.push_str(&format!("Beta Parameter: {:.2}\n", details.beta));
                    output.push_str(&format!("Trading Fee: {:.1}%\n", details.trading_fee * 100.0));
                    output.push_str(&format!("Created at Height: {}\n", details.created_at_height));
                    
                    if !details.tags.is_empty() {
                        output.push_str(&format!("Tags: {}\n", details.tags.join(", ")));
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
                    
                    output.push_str(&format!("\nTotal Volume: {:.0} sats\n", total_volume));
                    output.push_str(&format!("Total Liquidity: {:.0} sats\n", details.liquidity));
                    output.push_str(&format!("Treasury: {:.0} sats\n", details.treasury));
                    
                    if !details.decision_slots.is_empty() {
                        output.push_str("\nDecision Slots:\n");
                        for slot_id in &details.decision_slots {
                            let short_slot = if slot_id.len() > 16 {
                                format!("{}...{}", &slot_id[..8], &slot_id[slot_id.len()-8..])
                            } else {
                                slot_id.clone()
                            };
                            output.push_str(&format!("  - {}\n", short_slot));
                        }
                    }
                    
                    output
                }
                None => format!("Market {} not found", market_id),
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
