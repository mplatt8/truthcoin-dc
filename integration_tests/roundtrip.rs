//! Test an unknown withdrawal event

use std::collections::HashMap;

use bip300301_enforcer_integration_tests::{
    integration_test::{
        activate_sidechain, deposit, fund_enforcer, propose_sidechain,
    },
    setup::{
        Mode, Network, PostSetup as EnforcerPostSetup, Sidechain as _,
        setup as setup_enforcer,
    },
    util::{AbortOnDrop, AsyncTrial},
};
use futures::{
    FutureExt as _, StreamExt as _, channel::mpsc, future::BoxFuture,
};
use tokio::time::sleep;
use tracing::Instrument as _;
use truthcoin_dc::{
    authorization::{Dst, Signature},
    types::{Address, FilledOutputContent, OutputContent, PointedOutput, GetAddress as _, Txid},
};
use truthcoin_dc_app_rpc_api::RpcClient as _;

use crate::{
    setup::{Init, PostSetup},
    util::BinPaths,
};


#[derive(Debug)]
struct TruthcoinNodes {
    /// Sidechain process that will be issuing a Truthcoin
    issuer: PostSetup,
    /// Sidechain process that will be voting
    voter_0: PostSetup,
    /// Sidechain process that will be voting
    voter_1: PostSetup,
    /// Sidechain process that will be voting
    voter_2: PostSetup,
    /// Sidechain process that will be voting
    voter_3: PostSetup,
    /// Sidechain process that will be voting
    voter_4: PostSetup,
    /// Sidechain process that will be voting
    voter_5: PostSetup,
    /// Sidechain process that will be voting
    voter_6: PostSetup,
}

impl TruthcoinNodes {
    async fn setup(
        bin_paths: &BinPaths,
        res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
        enforcer_post_setup: &EnforcerPostSetup,
    ) -> anyhow::Result<Self> {
        // Initialize a single node
        let setup_single = |suffix: &str| {
            PostSetup::setup(
                Init {
                    truthcoin_app: bin_paths.truthcoin.clone(),
                    data_dir_suffix: Some(suffix.to_owned()),
                },
                enforcer_post_setup,
                res_tx.clone(),
            )
        };
        let res = Self {
            issuer: setup_single("issuer").await?,
            voter_0: setup_single("voter_0").await?,
            voter_1: setup_single("voter_1").await?,
            voter_2: setup_single("voter_2").await?,
            voter_3: setup_single("voter_3").await?,
            voter_4: setup_single("voter_4").await?,
            voter_5: setup_single("voter_5").await?,
            voter_6: setup_single("voter_6").await?,
        };
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_0_addr = %res.voter_0.net_addr(),
            "Connecting issuer to voter 0");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_0.net_addr().into())
            .await?;
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_1_addr = %res.voter_1.net_addr(),
            "Connecting issuer to voter 1");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_1.net_addr().into())
            .await?;
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_2_addr = %res.voter_2.net_addr(),
            "Connecting issuer to voter 2");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_2.net_addr().into())
            .await?;
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_3_addr = %res.voter_3.net_addr(),
            "Connecting issuer to voter 3");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_3.net_addr().into())
            .await?;
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_4_addr = %res.voter_4.net_addr(),
            "Connecting issuer to voter 4");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_4.net_addr().into())
            .await?;
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_5_addr = %res.voter_5.net_addr(),
            "Connecting issuer to voter 5");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_5.net_addr().into())
            .await?;
        tracing::debug!(
            issuer_addr = %res.issuer.net_addr(),
            voter_6_addr = %res.voter_6.net_addr(),
            "Connecting issuer to voter 6");
        let () = res
            .issuer
            .rpc_client
            .connect_peer(res.voter_6.net_addr().into())
            .await?;
        Ok(res)
    }
}

const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(21_000_000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);

/// Initial setup for the test
async fn setup(
    bin_paths: &BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<(EnforcerPostSetup, TruthcoinNodes)> {
    let mut enforcer_post_setup = setup_enforcer(
        &bin_paths.others,
        Network::Regtest,
        Mode::Mempool,
        res_tx.clone(),
    )
    .await?;
    let () = propose_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Proposed sidechain successfully");
    let () = activate_sidechain::<PostSetup>(&mut enforcer_post_setup).await?;
    tracing::info!("Activated sidechain successfully");
    let () = fund_enforcer::<PostSetup>(&mut enforcer_post_setup).await?;
    let mut truthcoin_nodes =
        TruthcoinNodes::setup(bin_paths, res_tx, &enforcer_post_setup).await?;
    let issuer_deposit_address =
        truthcoin_nodes.issuer.get_deposit_address().await?;
    let () = deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.issuer,
        &issuer_deposit_address,
        DEPOSIT_AMOUNT,
        DEPOSIT_FEE,
    )
    .await?;
    tracing::info!("Deposited to sidechain successfully");
    Ok((enforcer_post_setup, truthcoin_nodes))
}

const VOTE_CALL_MSG: &str = "test vote call";
const VOTE_YES_MSG: &str = "test vote call YES";
const VOTE_NO_MSG: &str = "test vote call NO";
/// Total initial Votecoin supply
const INITIAL_VOTECOIN_SUPPLY: u32 = 1000000;
/// Votecoin allocated to voter 0
const VOTER_ALLOCATION: u32 = 142857;

async fn roundtrip_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, mut truthcoin_nodes) =
        setup(&bin_paths, res_tx.clone()).await?;

    tracing::info!("Generating issuer verifying key");
    let issuer_vk = truthcoin_nodes
        .issuer
        .rpc_client
        .get_new_verifying_key()
        .await?;

    tracing::info!(
        "Creating genesis block with initial Votecoin supply of {} units",
        INITIAL_VOTECOIN_SUPPLY
    );
    // Mine a genesis block that will create the initial Votecoin supply
    // This block should contain a coinbase output with all Votecoin supply to the issuer
    let _issuer_addr =
        truthcoin_nodes.issuer.rpc_client.get_new_address().await?;

    // Mine the genesis block - this should automatically create the Votecoin supply
    // The node should be configured to mint the initial supply in the first block
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    // Verify the initial supply was created correctly
    let utxos = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let total_votecoin: u32 = utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    anyhow::ensure!(
        total_votecoin == INITIAL_VOTECOIN_SUPPLY,
        "Expected initial Votecoin supply of {}, found {}",
        INITIAL_VOTECOIN_SUPPLY,
        total_votecoin
    );
    tracing::info!(
        "Verified initial Votecoin supply: {} units",
        total_votecoin
    );

    tracing::info!("Setting up voter addresses");
    let voters = [
        &truthcoin_nodes.voter_0,
        &truthcoin_nodes.voter_1,
        &truthcoin_nodes.voter_2,
        &truthcoin_nodes.voter_3,
        &truthcoin_nodes.voter_4,
        &truthcoin_nodes.voter_5,
        &truthcoin_nodes.voter_6,
    ];

    let mut voter_addrs = Vec::new();
    for voter in &voters {
        voter_addrs.push(voter.rpc_client.get_new_address().await?);
    }
    let [voter_addr_0, voter_addr_1, voter_addr_2, voter_addr_3, voter_addr_4, voter_addr_5, voter_addr_6]: [Address; 7] =
        voter_addrs.try_into().unwrap();

    tracing::info!("Distributing Votecoin to voters");

    // Detailed debug for first transfer only
    let voter_addresses = [voter_addr_0, voter_addr_1, voter_addr_2, voter_addr_3, voter_addr_4, voter_addr_5, voter_addr_6];

    for (i, &voter_addr) in voter_addresses.iter().enumerate() {
        tracing::info!("=== DEBUG: Transfer {} - Sending {} Votecoin to voter_{} ({}) ===",
            i + 1, VOTER_ALLOCATION, i, voter_addr);

        // Only do detailed debugging for the first transfer
        if i == 0 {
            // Check issuer's UTXOs before transfer
            let utxos_before = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
            let votecoin_before: u32 = utxos_before
                .iter()
                .filter_map(|utxo| utxo.output.content.votecoin())
                .sum();
            tracing::info!("DEBUG: Issuer has {} Votecoin before transfer", votecoin_before);

            // Create the transfer transaction
            let txid: Txid = truthcoin_nodes
                .issuer
                .rpc_client
                .transfer_votecoin(voter_addr, VOTER_ALLOCATION, 0, None)
                .await?;
            tracing::info!("DEBUG: Created transaction {}", txid);

            // Check if transaction is in mempool
            let tx_in_mempool = truthcoin_nodes.issuer.rpc_client.get_transaction(txid).await?;
            tracing::info!("DEBUG: Transaction in mempool: {:?}", tx_in_mempool.is_some());
            if let Some(tx) = &tx_in_mempool {
                tracing::info!("DEBUG: Transaction has {} outputs", tx.outputs.len());
                for (j, output) in tx.outputs.iter().enumerate() {
                    match &output.content {
                        OutputContent::Bitcoin(v) => tracing::info!("  Output {}: {} sats to {}", j, v.0.to_sat(), output.address),
                        OutputContent::Votecoin(vc) => tracing::info!("  Output {}: {} Votecoin to {}", j, vc, output.address),
                        OutputContent::Withdrawal(w) => tracing::info!("  Output {}: Withdrawal {} sats to {}", j, w.value.to_sat(), output.address),
                    }
                }
            }

            // Mine the transaction into a block
            tracing::info!("DEBUG: Mining block to include transaction...");
            let height_before = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
            truthcoin_nodes
                .issuer
                .bmm_single(&mut enforcer_post_setup)
                .await?;
            let height_after = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
            tracing::info!("DEBUG: Mined block, height {} -> {}", height_before, height_after);

            // Check if transaction is now confirmed
            let tx_info = truthcoin_nodes.issuer.rpc_client.get_transaction_info(txid).await?;
            tracing::info!("DEBUG: Transaction info: {:?}", tx_info);

            // Check issuer's UTXOs after mining
            let utxos_after = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
            let votecoin_after: u32 = utxos_after
                .iter()
                .filter_map(|utxo| utxo.output.content.votecoin())
                .sum();
            tracing::info!("DEBUG: Issuer has {} Votecoin after mining (expected: {})",
                votecoin_after, votecoin_before.saturating_sub(VOTER_ALLOCATION));

            // Check ALL UTXOs on the blockchain state (not just wallet)
            tracing::info!("DEBUG: Checking blockchain state for all votecoin UTXOs...");
            let all_utxos_issuer = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
            let mut votecoin_by_address: HashMap<Address, u32> = HashMap::new();
            for utxo in &all_utxos_issuer {
                if let Some(vc) = utxo.output.content.votecoin() {
                    *votecoin_by_address.entry(utxo.output.address).or_default() += vc;
                }
            }
            tracing::info!("DEBUG: Blockchain state (from issuer view) has {} addresses with votecoin:", votecoin_by_address.len());
            for (addr, vc) in &votecoin_by_address {
                tracing::info!("  {}: {} Votecoin", addr, vc);
            }

            // Check voter_0's wallet
            let voter_0_utxos = truthcoin_nodes.voter_0.rpc_client.get_wallet_utxos().await?;
            let voter_0_votecoin: u32 = voter_0_utxos
                .iter()
                .filter_map(|utxo| utxo.output.content.votecoin())
                .sum();
            tracing::info!("DEBUG: voter_0 wallet has {} Votecoin (expected: {})", voter_0_votecoin, VOTER_ALLOCATION);
        } else {
            // For other transfers, just create transaction and mine the block
            let txid: Txid = truthcoin_nodes
                .issuer
                .rpc_client
                .transfer_votecoin(voter_addr, VOTER_ALLOCATION, 0, None)
                .await?;
            tracing::info!("DEBUG: Created transaction {}", txid);

            truthcoin_nodes
                .issuer
                .bmm_single(&mut enforcer_post_setup)
                .await?;
        }

        // Wait for network sync
        tracing::info!("DEBUG: Waiting 2s for network sync...");
        sleep(std::time::Duration::from_secs(2)).await;
    }

    tracing::info!("Signing vote call message");
    let vote_call_msg_sig: Signature = truthcoin_nodes
        .issuer
        .rpc_client
        .sign_arbitrary_msg(issuer_vk, VOTE_CALL_MSG.to_owned())
        .await?;

    tracing::info!("Verifying vote call message signature");
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3, &truthcoin_nodes.voter_4, &truthcoin_nodes.voter_5, &truthcoin_nodes.voter_6] {
        anyhow::ensure!(
            voter
                .rpc_client
                .verify_signature(
                    vote_call_msg_sig,
                    issuer_vk,
                    Dst::Arbitrary,
                    VOTE_CALL_MSG.to_owned()
                )
                .await?
        )
    }

    tracing::info!("Taking snapshot of Votecoin holders");

    // First, verify votecoin balances using RPC from each voter's perspective
    tracing::info!("=== Verifying Votecoin Distribution via RPCs ===");

    for (i, (&voter_addr, voter)) in voter_addresses.iter().zip(&voters).enumerate() {
        // Check voter's balance from their own perspective
        let balance_rpc = voter
            .rpc_client
            .get_votecoin_balance(voter_addr)
            .await?;
        tracing::info!("voter_{} RPC balance (from voter_{}): {} Votecoin", i, i, balance_rpc);

        // Check voter's balance from issuer's perspective
        let balance_issuer = truthcoin_nodes
            .issuer
            .rpc_client
            .get_votecoin_balance(voter_addr)
            .await?;
        tracing::info!("voter_{} RPC balance (from issuer): {} Votecoin", i, balance_issuer);

        // Verify both perspectives agree
        anyhow::ensure!(
            balance_rpc == VOTER_ALLOCATION,
            "voter_{} self-reported balance {} != expected {}",
            i,
            balance_rpc,
            VOTER_ALLOCATION
        );
        anyhow::ensure!(
            balance_issuer == VOTER_ALLOCATION,
            "voter_{} issuer-view balance {} != expected {}",
            i,
            balance_issuer,
            VOTER_ALLOCATION
        );
    }

    // Check all UTXOs across all nodes to see where Votecoin actually is
    tracing::info!("=== Dumping all UTXOs across all nodes ===");

    // Helper function to get and log UTXOs for a node
    let get_utxo_info = |name: &str, utxos: &[PointedOutput<FilledOutputContent>]| -> u32 {
        let total_votecoin: u32 = utxos
            .iter()
            .filter_map(|utxo| utxo.output.content.votecoin())
            .sum();
        tracing::info!("{}: {} UTXOs, {} total Votecoin", name, utxos.len(), total_votecoin);
        for utxo in utxos {
            if let Some(votecoin_amount) = utxo.output.content.votecoin() {
                tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
            }
        }
        total_votecoin
    };

    // Issuer UTXOs
    let issuer_utxos = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let issuer_total_votecoin = get_utxo_info("issuer", &issuer_utxos);

    // Voter UTXOs
    let voter_totals: Vec<u32> = {
        let mut totals = Vec::new();
        for (i, voter) in voters.iter().enumerate() {
            let utxos = voter.rpc_client.get_wallet_utxos().await?;
            totals.push(get_utxo_info(&format!("voter_{}", i), &utxos));
        }
        totals
    };

    tracing::info!("✓ Votecoin transfers working correctly!");
    let expected = std::iter::repeat(VOTER_ALLOCATION).take(7)
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!("Expected: {}, issuer remainder", expected);

    let actual = voter_totals.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!("Actual: {}, issuer={}", actual, issuer_total_votecoin);

    // Now build the snapshot using list_utxos from issuer
    let vote_weights: HashMap<Address, u32> = {
        let mut weights = HashMap::new();
        let utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
        tracing::info!("Building snapshot from issuer's UTXO set ({} UTXOs)", utxos.len());
        for utxo in utxos {
            if let Some(votecoin_amount) = utxo.output.content.votecoin() {
                tracing::debug!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
                *weights.entry(utxo.output.address).or_default() +=
                    votecoin_amount;
            }
        }
        tracing::info!("Snapshot contains {} addresses with votecoin", weights.len());
        for (addr, weight) in &weights {
            tracing::info!("  {}: {} Votecoin", addr, weight);
        }
        weights
    };
    anyhow::ensure!(vote_weights.len() >= 7, "Expected at least 7 voters in snapshot, found {}", vote_weights.len());

    tracing::info!("Signing votes");
    let vote_auth_0 = truthcoin_nodes
        .voter_0
        .rpc_client
        .sign_arbitrary_msg_as_addr(voter_addr_0, VOTE_YES_MSG.to_owned())
        .await?;
    let vote_auth_1 = truthcoin_nodes
        .voter_1
        .rpc_client
        .sign_arbitrary_msg_as_addr(voter_addr_1, VOTE_NO_MSG.to_owned())
        .await?;

    tracing::info!("Verifying votes");
    let (total_yes, total_no) = {
        let (mut total_yes, mut total_no) = (0, 0);
        let mut vote_weights = vote_weights;
        for vote_auth in [vote_auth_0, vote_auth_1] {
            let voter_addr = vote_auth.get_address();
            if truthcoin_nodes
                .issuer
                .rpc_client
                .verify_signature(
                    vote_auth.signature,
                    vote_auth.verifying_key,
                    Dst::Arbitrary,
                    VOTE_YES_MSG.to_owned(),
                )
                .await?
            {
                if let Some(weight) = vote_weights.remove(&voter_addr) {
                    total_yes += weight;
                }
            } else if truthcoin_nodes
                .issuer
                .rpc_client
                .verify_signature(
                    vote_auth.signature,
                    vote_auth.verifying_key,
                    Dst::Arbitrary,
                    VOTE_NO_MSG.to_owned(),
                )
                .await?
            {
                if let Some(weight) = vote_weights.remove(&voter_addr) {
                    total_no += weight;
                }
            }
        }
        (total_yes, total_no)
    };
    anyhow::ensure!(total_yes == VOTER_ALLOCATION);
    anyhow::ensure!(total_no == VOTER_ALLOCATION);

    tracing::info!(
        "Vote test completed successfully - Votecoin voting system working"
    );

    // Verify all nodes are at the same block height
    tracing::info!("=== Verifying Block Height Synchronization ===");
    let issuer_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;

    let mut voter_heights = Vec::new();
    for (i, voter) in voters.iter().enumerate() {
        let height = voter.rpc_client.getblockcount().await?;
        voter_heights.push(height);
        tracing::info!("  - voter_{}: {}", i, height);
    }

    // Check if all voter heights match the issuer height
    for (i, &height) in voter_heights.iter().enumerate() {
        anyhow::ensure!(
            issuer_height == height,
            "Block height mismatch: issuer={}, voter_{}={}",
            issuer_height, i, height
        );
    }

    tracing::info!("✓ All nodes synchronized at block height: {}", issuer_height);
    tracing::info!("  - issuer: {}", issuer_height);

    // ============================================================================
    // PHASE 1.5: Fund Voters with Bitcoin for Transaction Fees
    // ============================================================================
    tracing::info!("=== PHASE 1.5: Depositing Bitcoin to Voters ===");

    // Define smaller deposit amounts for voters (they just need to pay fees)
    const VOTER_DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(5_000_000); // 5M sats = 0.05 BTC
    const VOTER_DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(500_000); // 0.5M sats

    // Deposit to voter_0
    tracing::info!("Depositing {} sats to voter_0", VOTER_DEPOSIT_AMOUNT.to_sat());
    let voter_0_deposit_address = truthcoin_nodes.voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_0,
        &voter_0_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    tracing::info!("✓ Deposited to voter_0");
    sleep(std::time::Duration::from_secs(1)).await;

    // Deposit to voter_1
    tracing::info!("Depositing {} sats to voter_1", VOTER_DEPOSIT_AMOUNT.to_sat());
    let voter_1_deposit_address = truthcoin_nodes.voter_1.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_1,
        &voter_1_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    tracing::info!("✓ Deposited to voter_1");
    sleep(std::time::Duration::from_secs(1)).await;

    // Deposit to voter_2
    tracing::info!("Depositing {} sats to voter_2", VOTER_DEPOSIT_AMOUNT.to_sat());
    let voter_2_deposit_address = truthcoin_nodes.voter_2.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_2,
        &voter_2_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    tracing::info!("✓ Deposited to voter_2");
    sleep(std::time::Duration::from_secs(1)).await;

    // Deposit to voter_3
    tracing::info!("Depositing {} sats to voter_3", VOTER_DEPOSIT_AMOUNT.to_sat());
    let voter_3_deposit_address = truthcoin_nodes.voter_3.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_3,
        &voter_3_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    tracing::info!("✓ Deposited to voter_3");
    sleep(std::time::Duration::from_secs(1)).await;

    // Verify deposits were received
    let voter_0_balance = truthcoin_nodes.voter_0.rpc_client.bitcoin_balance().await?;
    tracing::info!("voter_0 balance: {} sats", voter_0_balance.total);
    anyhow::ensure!(voter_0_balance.total > bitcoin::Amount::ZERO, "voter_0 should have positive balance after deposit");

    let voter_1_balance = truthcoin_nodes.voter_1.rpc_client.bitcoin_balance().await?;
    tracing::info!("voter_1 balance: {} sats", voter_1_balance.total);
    anyhow::ensure!(voter_1_balance.total > bitcoin::Amount::ZERO, "voter_1 should have positive balance after deposit");

    let voter_2_balance = truthcoin_nodes.voter_2.rpc_client.bitcoin_balance().await?;
    tracing::info!("voter_2 balance: {} sats", voter_2_balance.total);
    anyhow::ensure!(voter_2_balance.total > bitcoin::Amount::ZERO, "voter_2 should have positive balance after deposit");

    let voter_3_balance = truthcoin_nodes.voter_3.rpc_client.bitcoin_balance().await?;
    tracing::info!("voter_3 balance: {} sats", voter_3_balance.total);
    anyhow::ensure!(voter_3_balance.total > bitcoin::Amount::ZERO, "voter_3 should have positive balance after deposit");

    tracing::info!("✓ All voters funded with Bitcoin for transaction fees");

    // ============================================================================
    // PHASE 2: Decision Slot Creation
    // ============================================================================
    tracing::info!("=== PHASE 2: Decision Slot Claiming ===");

    // Check current block height
    let current_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    tracing::info!("Current block height: {}", current_height);

    // In testing mode with 10 blocks per period:
    // - Period 0: blocks 0-9
    // - Period 1: blocks 10-19
    // - Period 2: blocks 20-29
    // - Period 3: blocks 30-39 (target for claiming)
    // Slots in period 3 are claimable now (from the current period)

    // Four different voters claim four different decision slots in period 3
    tracing::info!("Having four voters claim decision slots in period 3...");

    let slot_claims = [
        (voter_addr_0, 0, "Decision 1: Will Bitcoin reach $100k in 2025?"),
        (voter_addr_1, 1, "Decision 2: Will Ethereum merge to PoS succeed?"),
        (voter_addr_2, 2, "Decision 3: Will there be 1M BTC addresses by 2026?"),
        (voter_addr_3, 3, "Decision 4: Will Lightning Network capacity exceed 5000 BTC?"),
    ];

    let mut claim_txids = Vec::new();
    for (i, (voter_addr, slot_index, question)) in slot_claims.iter().enumerate() {
        tracing::info!("voter_{} claiming slot {} in period 3: \"{}\"", i, slot_index, question);

        let voter_node = match i {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            _ => unreachable!(),
        };

        let txid = voter_node
            .rpc_client
            .claim_decision_slot(
                3, // period_index
                *slot_index, // slot_index
                true, // is_standard
                false, // is_scaled (binary decision)
                question.to_string(),
                None, // min (not scaled)
                None, // max (not scaled)
                1000, // fee_sats
            )
            .await?;

        tracing::info!("voter_{} claim transaction: {}", i, txid);
        claim_txids.push(txid);
    }

    // Wait for transaction propagation to issuer's mempool
    tracing::info!("Waiting for transaction propagation...");
    sleep(std::time::Duration::from_millis(500)).await;

    // Mine a block to confirm all four slot claims
    tracing::info!("Mining block to confirm all four slot claims...");
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    // Wait for network sync - first ensure all voters receive the block via P2P
    sleep(std::time::Duration::from_secs(2)).await;

    tracing::info!("Waiting for block propagation to all voters...");
    let issuer_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    tracing::info!("Issuer block height: {}", issuer_height);

    // Wait for all voters to receive the block (check block height matches issuer)
    for (i, voter) in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3].iter().enumerate() {
        let mut block_received = false;
        for attempt in 0..20 {
            let voter_height = voter.rpc_client.getblockcount().await?;
            if voter_height >= issuer_height {
                tracing::info!("voter_{} received block: height={}", i, voter_height);
                block_received = true;
                break;
            }
            tracing::debug!("voter_{} waiting for block (height={}, expected={}), attempt {}/20",
                i, voter_height, issuer_height, attempt + 1);
            sleep(std::time::Duration::from_millis(500)).await;
        }
        anyhow::ensure!(block_received, "voter_{} did not receive block after 10 seconds", i);
    }

    // Diagnostic: Check what UTXOs exist for voter_0
    tracing::info!("=== DIAGNOSTIC: Checking voter_0 UTXOs ===");

    // Get wallet UTXOs (what the wallet thinks it owns)
    let wallet_utxos = truthcoin_nodes.voter_0.rpc_client.get_wallet_utxos().await?;
    tracing::info!("voter_0 wallet UTXOs count: {}", wallet_utxos.len());
    for (idx, utxo) in wallet_utxos.iter().enumerate() {
        match &utxo.output.content {
            FilledOutputContent::Bitcoin(bitcoin_content) => {
                tracing::info!("  Wallet UTXO {}: {} sats at address {}, outpoint: {:?}",
                    idx, bitcoin_content.0.to_sat(), utxo.output.address, utxo.outpoint);
            }
            FilledOutputContent::Votecoin(vc) => {
                tracing::info!("  Wallet UTXO {}: {} Votecoin at address {}", idx, vc, utxo.output.address);
            }
            FilledOutputContent::BitcoinWithdrawal { .. } => {
                tracing::info!("  Wallet UTXO {}: Withdrawal", idx);
            }
        }
    }

    // Get chain state UTXOs (all UTXOs from blockchain perspective)
    let chain_utxos = truthcoin_nodes.voter_0.rpc_client.list_utxos().await?;
    tracing::info!("voter_0 chain state total UTXOs: {}", chain_utxos.len());

    // Filter to just Bitcoin UTXOs owned by voter_0's addresses
    let voter_0_addresses: Vec<_> = wallet_utxos.iter().map(|u| u.output.address).collect();
    let voter_0_chain_bitcoin_utxos: Vec<_> = chain_utxos.iter()
        .filter(|utxo| {
            matches!(utxo.output.content, FilledOutputContent::Bitcoin(_))
                && voter_0_addresses.contains(&utxo.output.address)
        })
        .collect();

    tracing::info!("voter_0 Bitcoin UTXOs in chain state owned by voter_0: {}", voter_0_chain_bitcoin_utxos.len());
    for (idx, utxo) in voter_0_chain_bitcoin_utxos.iter().enumerate() {
        if let FilledOutputContent::Bitcoin(bitcoin_content) = &utxo.output.content {
            tracing::info!("  Chain UTXO {}: {} sats at address {}, outpoint: {:?}",
                idx, bitcoin_content.0.to_sat(), utxo.output.address, utxo.outpoint);
        }
    }

    let balance = truthcoin_nodes.voter_0.rpc_client.bitcoin_balance().await?;
    tracing::info!("voter_0 balance: total={} sats, available={} sats", balance.total, balance.available);

    // Check if voter_0 has no Bitcoin UTXOs - if so, deposit more
    if voter_0_chain_bitcoin_utxos.is_empty() {
        tracing::warn!("WARNING: voter_0 has NO Bitcoin UTXOs in chain state!");
        tracing::warn!("This is unexpected - the change UTXO from claim_decision_slot should exist.");
        tracing::warn!("Possible causes:");
        tracing::warn!("  1. The transaction didn't create a change output");
        tracing::warn!("  2. The UTXO was incorrectly marked as spent");
        tracing::warn!("  3. There's a bug in UTXO tracking");

        // Try mining a few more blocks to see if it helps
        tracing::info!("Mining 3 more blocks to ensure chain state is fully synced...");
        for i in 0..3 {
            truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
            tracing::info!("  Mined block {}/3", i + 1);
            sleep(std::time::Duration::from_secs(1)).await;
        }

        // Check again after mining
        let wallet_utxos_after_mining = truthcoin_nodes.voter_0.rpc_client.get_wallet_utxos().await?;
        let chain_utxos_after_mining = truthcoin_nodes.voter_0.rpc_client.list_utxos().await?;
        let voter_0_addresses_after: Vec<_> = wallet_utxos_after_mining.iter().map(|u| u.output.address).collect();
        let voter_0_chain_bitcoin_utxos_after: Vec<_> = chain_utxos_after_mining.iter()
            .filter(|utxo| {
                matches!(utxo.output.content, FilledOutputContent::Bitcoin(_))
                    && voter_0_addresses_after.contains(&utxo.output.address)
            })
            .collect();

        tracing::info!("After mining 3 blocks: voter_0 Bitcoin UTXOs in chain = {}", voter_0_chain_bitcoin_utxos_after.len());

        if voter_0_chain_bitcoin_utxos_after.is_empty() {
            tracing::warn!("Still no Bitcoin UTXOs after mining - depositing fresh funds...");

            // Deposit more Bitcoin to ALL voters so they can create markets
            tracing::info!("Depositing additional {} sats to voter_0", VOTER_DEPOSIT_AMOUNT.to_sat());
            let voter_0_deposit_addr = truthcoin_nodes.voter_0.get_deposit_address().await?;
            deposit(
                &mut enforcer_post_setup,
                &mut truthcoin_nodes.voter_0,
                &voter_0_deposit_addr,
                VOTER_DEPOSIT_AMOUNT,
                VOTER_DEPOSIT_FEE,
            ).await?;
            tracing::info!("✓ Additional deposit to voter_0 complete");

            tracing::info!("Depositing additional {} sats to voter_1", VOTER_DEPOSIT_AMOUNT.to_sat());
            let voter_1_deposit_addr = truthcoin_nodes.voter_1.get_deposit_address().await?;
            deposit(
                &mut enforcer_post_setup,
                &mut truthcoin_nodes.voter_1,
                &voter_1_deposit_addr,
                VOTER_DEPOSIT_AMOUNT,
                VOTER_DEPOSIT_FEE,
            ).await?;
            tracing::info!("✓ Additional deposit to voter_1 complete");

            tracing::info!("Depositing additional {} sats to voter_2", VOTER_DEPOSIT_AMOUNT.to_sat());
            let voter_2_deposit_addr = truthcoin_nodes.voter_2.get_deposit_address().await?;
            deposit(
                &mut enforcer_post_setup,
                &mut truthcoin_nodes.voter_2,
                &voter_2_deposit_addr,
                VOTER_DEPOSIT_AMOUNT,
                VOTER_DEPOSIT_FEE,
            ).await?;
            tracing::info!("✓ Additional deposit to voter_2 complete");

            tracing::info!("Depositing additional {} sats to voter_3", VOTER_DEPOSIT_AMOUNT.to_sat());
            let voter_3_deposit_addr = truthcoin_nodes.voter_3.get_deposit_address().await?;
            deposit(
                &mut enforcer_post_setup,
                &mut truthcoin_nodes.voter_3,
                &voter_3_deposit_addr,
                VOTER_DEPOSIT_AMOUNT,
                VOTER_DEPOSIT_FEE,
            ).await?;
            tracing::info!("✓ Additional deposit to voter_3 complete");
        }
    } else {
        tracing::info!("voter_0 has {} Bitcoin UTXOs in chain state - wallet may just need more time to sync",
            voter_0_chain_bitcoin_utxos.len());
    }

    // Verify all four slots were claimed successfully
    tracing::info!("Verifying claimed slots in period 3...");
    let claimed_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .get_claimed_slots_in_period(3)
        .await?;

    tracing::info!("Found {} claimed slots in period 3", claimed_slots.len());
    anyhow::ensure!(
        claimed_slots.len() == 4,
        "Expected 4 claimed slots in period 3, found {}",
        claimed_slots.len()
    );

    for (i, slot) in claimed_slots.iter().enumerate() {
        tracing::info!("Claimed slot {}: {} (Period {}, Index {})",
            i + 1,
            slot.slot_id_hex,
            slot.period_index,
            slot.slot_index
        );
        tracing::info!("  Question preview: {}", slot.question_preview);
        tracing::info!("  Market maker: {}", slot.market_maker_pubkey_hash);
        tracing::info!("  Type: {} | {}",
            if slot.is_standard { "Standard" } else { "Non-Standard" },
            if slot.is_scaled { "Scaled" } else { "Binary" }
        );
    }

    tracing::info!("✓ Successfully claimed and verified 4 decision slots in period 3");

    // ============================================================================
    // PHASE 3: Market Creation
    // ============================================================================
    tracing::info!("=== PHASE 3: Creating Binary Markets ===");

    // Extract the slot IDs from claimed slots for market creation
    let market_slot_ids: Vec<String> = claimed_slots
        .iter()
        .map(|slot| slot.slot_id_hex.clone())
        .collect();

    tracing::info!(
        "Creating binary markets on the four claimed decision slots..."
    );

    // Each voter creates a binary market on their respective decision slot
    let market_configs = [
        (
            &truthcoin_nodes.voter_0,
            0,
            "Will Bitcoin reach $100k in 2025?",
            "Binary market tracking BTC price prediction",
        ),
        (
            &truthcoin_nodes.voter_1,
            1,
            "Will Ethereum merge to PoS succeed?",
            "Prediction market for Ethereum 2.0 success",
        ),
        (
            &truthcoin_nodes.voter_2,
            2,
            "Will there be 1M BTC addresses by 2026?",
            "Tracking Bitcoin adoption milestone",
        ),
        (
            &truthcoin_nodes.voter_3,
            3,
            "Will Lightning Network capacity exceed 5000 BTC?",
            "Lightning Network growth prediction",
        ),
    ];

    let mut market_txids = Vec::new();
    for (i, (voter_node, slot_idx, title, description)) in
        market_configs.iter().enumerate()
    {
        use truthcoin_dc_app_rpc_api::CreateMarketRequest;

        let slot_id = &market_slot_ids[*slot_idx];
        tracing::info!(
            "voter_{} creating binary market: \"{}\" on slot {}",
            i,
            title,
            slot_id
        );

        let request = CreateMarketRequest {
            title: title.to_string(),
            description: description.to_string(),
            market_type: "independent".to_string(),
            decision_slots: vec![slot_id.clone()],
            dimensions: None,
            has_residual: None,
            beta: Some(100.0),
            trading_fee: Some(0.005),
            tags: Some(vec!["integration-test".to_string()]),
            initial_liquidity: None,
            fee_sats: 1000,
        };

        let market_id = voter_node.rpc_client.create_market(request).await?;
        tracing::info!(
            "voter_{} created market with ID: {}",
            i,
            market_id
        );
        market_txids.push(market_id);
    }

    // Wait for transaction propagation
    tracing::info!("Waiting for market creation transactions to propagate...");
    sleep(std::time::Duration::from_millis(500)).await;

    // Mine a block to confirm all four market creations
    tracing::info!("Mining block to confirm market creations...");
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    // Wait for network sync
    sleep(std::time::Duration::from_secs(2)).await;

    // Verify all markets were created successfully using list_markets RPC
    tracing::info!("Verifying markets using list_markets RPC...");
    let markets = truthcoin_nodes.issuer.rpc_client.list_markets().await?;

    tracing::info!("Found {} markets in Trading state", markets.len());
    anyhow::ensure!(
        markets.len() == 4,
        "Expected 4 markets, found {}",
        markets.len()
    );

    for (i, market) in markets.iter().enumerate() {
        tracing::info!(
            "Market {}: ID={}, Title=\"{}\", Outcomes={}, State={}",
            i + 1,
            market.market_id,
            market.title,
            market.outcome_count,
            market.state
        );

        // Verify market is in Trading state
        anyhow::ensure!(
            market.state == "Trading",
            "Market {} should be in Trading state, found: {}",
            market.market_id,
            market.state
        );

        // Verify binary markets have 2-3 outcomes (2 outcomes + unresolvable state = 3)
        anyhow::ensure!(
            market.outcome_count == 2 || market.outcome_count == 3,
            "Binary market {} should have 2-3 outcomes (including unresolvable), found {}",
            market.market_id,
            market.outcome_count
        );
    }

    tracing::info!("✓ Successfully created and verified 4 binary markets");

    // Cleanup
    {
        drop(truthcoin_nodes);
        tracing::info!(
            "Removing {}",
            enforcer_post_setup.out_dir.path().display()
        );
        drop(enforcer_post_setup.tasks);
        // Wait for tasks to die
        sleep(std::time::Duration::from_secs(1)).await;
        enforcer_post_setup.out_dir.cleanup()?;
    }
    Ok(())
}

async fn roundtrip(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = roundtrip_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn roundtrip_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("roundtrip", roundtrip(bin_paths).boxed())
}
