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
    types::{Address, OutputContent, GetAddress as _, Txid},
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
const VOTER_ALLOCATION_0: u32 = 60;
/// Votecoin allocated to voter 1
const VOTER_ALLOCATION_1: u32 = 40;
/// Votecoin allocated to voter 2
const VOTER_ALLOCATION_2: u32 = 80;
/// Votecoin allocated to voter 3
const VOTER_ALLOCATION_3: u32 = 100;
/// Votecoin allocated to voter 4
const VOTER_ALLOCATION_4: u32 = 120;
/// Votecoin allocated to voter 5
const VOTER_ALLOCATION_5: u32 = 140;
/// Votecoin allocated to voter 6
const VOTER_ALLOCATION_6: u32 = 160;

async fn roundtrip_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let (mut enforcer_post_setup, truthcoin_nodes) =
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
    let voter_addr_0 =
        truthcoin_nodes.voter_0.rpc_client.get_new_address().await?;
    let voter_addr_1 =
        truthcoin_nodes.voter_1.rpc_client.get_new_address().await?;
    let voter_addr_2 =
        truthcoin_nodes.voter_2.rpc_client.get_new_address().await?;
    let voter_addr_3 =
        truthcoin_nodes.voter_3.rpc_client.get_new_address().await?;
    let voter_addr_4 =
        truthcoin_nodes.voter_4.rpc_client.get_new_address().await?;
    let voter_addr_5 =
        truthcoin_nodes.voter_5.rpc_client.get_new_address().await?;
    let voter_addr_6 =
        truthcoin_nodes.voter_6.rpc_client.get_new_address().await?;

    tracing::info!("Distributing Votecoin to voters");

    // === TRANSFER 1: voter_0 ===
    tracing::info!("=== DEBUG: Transfer 1 - Sending {} Votecoin to voter_0 ({}) ===", VOTER_ALLOCATION_0, voter_addr_0);

    // Check issuer's UTXOs before transfer
    let utxos_before = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let votecoin_before: u32 = utxos_before
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("DEBUG: Issuer has {} Votecoin before transfer", votecoin_before);

    // Create the transfer transaction
    let txid_1: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_0, VOTER_ALLOCATION_0, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_1);

    // Check if transaction is in mempool
    let tx_in_mempool = truthcoin_nodes.issuer.rpc_client.get_transaction(txid_1).await?;
    tracing::info!("DEBUG: Transaction in mempool: {:?}", tx_in_mempool.is_some());
    if let Some(tx) = &tx_in_mempool {
        tracing::info!("DEBUG: Transaction has {} outputs", tx.outputs.len());
        for (i, output) in tx.outputs.iter().enumerate() {
            match &output.content {
                OutputContent::Bitcoin(v) => tracing::info!("  Output {}: {} sats to {}", i, v.0.to_sat(), output.address),
                OutputContent::Votecoin(vc) => tracing::info!("  Output {}: {} Votecoin to {}", i, vc, output.address),
                OutputContent::Withdrawal(w) => tracing::info!("  Output {}: Withdrawal {} sats to {}", i, w.value.to_sat(), output.address),
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
    let tx_info = truthcoin_nodes.issuer.rpc_client.get_transaction_info(txid_1).await?;
    tracing::info!("DEBUG: Transaction info: {:?}", tx_info);

    // Check issuer's UTXOs after mining
    let utxos_after = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let votecoin_after: u32 = utxos_after
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("DEBUG: Issuer has {} Votecoin after mining (expected: {})",
        votecoin_after, votecoin_before - VOTER_ALLOCATION_0);

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
    tracing::info!("DEBUG: voter_0 wallet has {} Votecoin (expected: {})", voter_0_votecoin, VOTER_ALLOCATION_0);

    // Wait for network sync
    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

    // === TRANSFER 2: voter_1 ===
    tracing::info!("=== DEBUG: Transfer 2 - Sending {} Votecoin to voter_1 ({}) ===", VOTER_ALLOCATION_1, voter_addr_1);

    let txid_2: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_1, VOTER_ALLOCATION_1, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_2);

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

    // === TRANSFER 3: voter_2 ===
    tracing::info!("=== DEBUG: Transfer 3 - Sending {} Votecoin to voter_2 ({}) ===", VOTER_ALLOCATION_2, voter_addr_2);

    let txid_3: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_2, VOTER_ALLOCATION_2, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_3);

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

    // === TRANSFER 4: voter_3 ===
    tracing::info!("=== DEBUG: Transfer 4 - Sending {} Votecoin to voter_3 ({}) ===", VOTER_ALLOCATION_3, voter_addr_3);

    let txid_4: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_3, VOTER_ALLOCATION_3, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_4);

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

    // === TRANSFER 5: voter_4 ===
    tracing::info!("=== DEBUG: Transfer 5 - Sending {} Votecoin to voter_4 ({}) ===", VOTER_ALLOCATION_4, voter_addr_4);

    let txid_5: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_4, VOTER_ALLOCATION_4, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_5);

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

    // === TRANSFER 6: voter_5 ===
    tracing::info!("=== DEBUG: Transfer 6 - Sending {} Votecoin to voter_5 ({}) ===", VOTER_ALLOCATION_5, voter_addr_5);

    let txid_6: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_5, VOTER_ALLOCATION_5, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_6);

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

    // === TRANSFER 7: voter_6 ===
    tracing::info!("=== DEBUG: Transfer 7 - Sending {} Votecoin to voter_6 ({}) ===", VOTER_ALLOCATION_6, voter_addr_6);

    let txid_7: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_6, VOTER_ALLOCATION_6, 0, None)
        .await?;
    tracing::info!("DEBUG: Created transaction {}", txid_7);

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("DEBUG: Waiting 2s for network sync...");
    sleep(std::time::Duration::from_secs(2)).await;

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

    // Check voter_0's balance from voter_0's perspective
    let voter_0_balance_rpc = truthcoin_nodes
        .voter_0
        .rpc_client
        .get_votecoin_balance(voter_addr_0)
        .await?;
    tracing::info!("voter_0 RPC balance (from voter_0): {} Votecoin", voter_0_balance_rpc);

    // Check voter_0's balance from issuer's perspective
    let voter_0_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_0)
        .await?;
    tracing::info!("voter_0 RPC balance (from issuer): {} Votecoin", voter_0_balance_issuer);

    // Check voter_1's balance from voter_1's perspective
    let voter_1_balance_rpc = truthcoin_nodes
        .voter_1
        .rpc_client
        .get_votecoin_balance(voter_addr_1)
        .await?;
    tracing::info!("voter_1 RPC balance (from voter_1): {} Votecoin", voter_1_balance_rpc);

    // Check voter_1's balance from issuer's perspective
    let voter_1_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_1)
        .await?;
    tracing::info!("voter_1 RPC balance (from issuer): {} Votecoin", voter_1_balance_issuer);

    // Check voter_2's balance from voter_2's perspective
    let voter_2_balance_rpc = truthcoin_nodes
        .voter_2
        .rpc_client
        .get_votecoin_balance(voter_addr_2)
        .await?;
    tracing::info!("voter_2 RPC balance (from voter_2): {} Votecoin", voter_2_balance_rpc);

    // Check voter_2's balance from issuer's perspective
    let voter_2_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_2)
        .await?;
    tracing::info!("voter_2 RPC balance (from issuer): {} Votecoin", voter_2_balance_issuer);

    // Check voter_3's balance from voter_3's perspective
    let voter_3_balance_rpc = truthcoin_nodes
        .voter_3
        .rpc_client
        .get_votecoin_balance(voter_addr_3)
        .await?;
    tracing::info!("voter_3 RPC balance (from voter_3): {} Votecoin", voter_3_balance_rpc);

    // Check voter_3's balance from issuer's perspective
    let voter_3_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_3)
        .await?;
    tracing::info!("voter_3 RPC balance (from issuer): {} Votecoin", voter_3_balance_issuer);

    // Check voter_4's balance from voter_4's perspective
    let voter_4_balance_rpc = truthcoin_nodes
        .voter_4
        .rpc_client
        .get_votecoin_balance(voter_addr_4)
        .await?;
    tracing::info!("voter_4 RPC balance (from voter_4): {} Votecoin", voter_4_balance_rpc);

    // Check voter_4's balance from issuer's perspective
    let voter_4_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_4)
        .await?;
    tracing::info!("voter_4 RPC balance (from issuer): {} Votecoin", voter_4_balance_issuer);

    // Check voter_5's balance from voter_5's perspective
    let voter_5_balance_rpc = truthcoin_nodes
        .voter_5
        .rpc_client
        .get_votecoin_balance(voter_addr_5)
        .await?;
    tracing::info!("voter_5 RPC balance (from voter_5): {} Votecoin", voter_5_balance_rpc);

    // Check voter_5's balance from issuer's perspective
    let voter_5_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_5)
        .await?;
    tracing::info!("voter_5 RPC balance (from issuer): {} Votecoin", voter_5_balance_issuer);

    // Check voter_6's balance from voter_6's perspective
    let voter_6_balance_rpc = truthcoin_nodes
        .voter_6
        .rpc_client
        .get_votecoin_balance(voter_addr_6)
        .await?;
    tracing::info!("voter_6 RPC balance (from voter_6): {} Votecoin", voter_6_balance_rpc);

    // Check voter_6's balance from issuer's perspective
    let voter_6_balance_issuer = truthcoin_nodes
        .issuer
        .rpc_client
        .get_votecoin_balance(voter_addr_6)
        .await?;
    tracing::info!("voter_6 RPC balance (from issuer): {} Votecoin", voter_6_balance_issuer);

    // Verify all perspectives agree
    anyhow::ensure!(
        voter_0_balance_rpc == VOTER_ALLOCATION_0,
        "voter_0 self-reported balance {} != expected {}",
        voter_0_balance_rpc,
        VOTER_ALLOCATION_0
    );
    anyhow::ensure!(
        voter_0_balance_issuer == VOTER_ALLOCATION_0,
        "voter_0 issuer-view balance {} != expected {}",
        voter_0_balance_issuer,
        VOTER_ALLOCATION_0
    );
    anyhow::ensure!(
        voter_1_balance_rpc == VOTER_ALLOCATION_1,
        "voter_1 self-reported balance {} != expected {}",
        voter_1_balance_rpc,
        VOTER_ALLOCATION_1
    );
    anyhow::ensure!(
        voter_1_balance_issuer == VOTER_ALLOCATION_1,
        "voter_1 issuer-view balance {} != expected {}",
        voter_1_balance_issuer,
        VOTER_ALLOCATION_1
    );
    anyhow::ensure!(
        voter_2_balance_rpc == VOTER_ALLOCATION_2,
        "voter_2 self-reported balance {} != expected {}",
        voter_2_balance_rpc,
        VOTER_ALLOCATION_2
    );
    anyhow::ensure!(
        voter_2_balance_issuer == VOTER_ALLOCATION_2,
        "voter_2 issuer-view balance {} != expected {}",
        voter_2_balance_issuer,
        VOTER_ALLOCATION_2
    );
    anyhow::ensure!(
        voter_3_balance_rpc == VOTER_ALLOCATION_3,
        "voter_3 self-reported balance {} != expected {}",
        voter_3_balance_rpc,
        VOTER_ALLOCATION_3
    );
    anyhow::ensure!(
        voter_3_balance_issuer == VOTER_ALLOCATION_3,
        "voter_3 issuer-view balance {} != expected {}",
        voter_3_balance_issuer,
        VOTER_ALLOCATION_3
    );
    anyhow::ensure!(
        voter_4_balance_rpc == VOTER_ALLOCATION_4,
        "voter_4 self-reported balance {} != expected {}",
        voter_4_balance_rpc,
        VOTER_ALLOCATION_4
    );
    anyhow::ensure!(
        voter_4_balance_issuer == VOTER_ALLOCATION_4,
        "voter_4 issuer-view balance {} != expected {}",
        voter_4_balance_issuer,
        VOTER_ALLOCATION_4
    );
    anyhow::ensure!(
        voter_5_balance_rpc == VOTER_ALLOCATION_5,
        "voter_5 self-reported balance {} != expected {}",
        voter_5_balance_rpc,
        VOTER_ALLOCATION_5
    );
    anyhow::ensure!(
        voter_5_balance_issuer == VOTER_ALLOCATION_5,
        "voter_5 issuer-view balance {} != expected {}",
        voter_5_balance_issuer,
        VOTER_ALLOCATION_5
    );
    anyhow::ensure!(
        voter_6_balance_rpc == VOTER_ALLOCATION_6,
        "voter_6 self-reported balance {} != expected {}",
        voter_6_balance_rpc,
        VOTER_ALLOCATION_6
    );
    anyhow::ensure!(
        voter_6_balance_issuer == VOTER_ALLOCATION_6,
        "voter_6 issuer-view balance {} != expected {}",
        voter_6_balance_issuer,
        VOTER_ALLOCATION_6
    );

    // Check all UTXOs across all nodes to see where Votecoin actually is
    tracing::info!("=== Dumping all UTXOs across all nodes ===");

    // Issuer UTXOs
    let issuer_utxos = truthcoin_nodes.issuer.rpc_client.get_wallet_utxos().await?;
    let issuer_total_votecoin: u32 = issuer_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("issuer: {} UTXOs, {} total Votecoin", issuer_utxos.len(), issuer_total_votecoin);
    for utxo in &issuer_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_0 UTXOs
    let voter_0_utxos = truthcoin_nodes.voter_0.rpc_client.get_wallet_utxos().await?;
    let voter_0_total_votecoin: u32 = voter_0_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_0: {} UTXOs, {} total Votecoin", voter_0_utxos.len(), voter_0_total_votecoin);
    for utxo in &voter_0_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_1 UTXOs
    let voter_1_utxos = truthcoin_nodes.voter_1.rpc_client.get_wallet_utxos().await?;
    let voter_1_total_votecoin: u32 = voter_1_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_1: {} UTXOs, {} total Votecoin", voter_1_utxos.len(), voter_1_total_votecoin);
    for utxo in &voter_1_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_2 UTXOs
    let voter_2_utxos = truthcoin_nodes.voter_2.rpc_client.get_wallet_utxos().await?;
    let voter_2_total_votecoin: u32 = voter_2_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_2: {} UTXOs, {} total Votecoin", voter_2_utxos.len(), voter_2_total_votecoin);
    for utxo in &voter_2_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_3 UTXOs
    let voter_3_utxos = truthcoin_nodes.voter_3.rpc_client.get_wallet_utxos().await?;
    let voter_3_total_votecoin: u32 = voter_3_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_3: {} UTXOs, {} total Votecoin", voter_3_utxos.len(), voter_3_total_votecoin);
    for utxo in &voter_3_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_4 UTXOs
    let voter_4_utxos = truthcoin_nodes.voter_4.rpc_client.get_wallet_utxos().await?;
    let voter_4_total_votecoin: u32 = voter_4_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_4: {} UTXOs, {} total Votecoin", voter_4_utxos.len(), voter_4_total_votecoin);
    for utxo in &voter_4_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_5 UTXOs
    let voter_5_utxos = truthcoin_nodes.voter_5.rpc_client.get_wallet_utxos().await?;
    let voter_5_total_votecoin: u32 = voter_5_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_5: {} UTXOs, {} total Votecoin", voter_5_utxos.len(), voter_5_total_votecoin);
    for utxo in &voter_5_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    // voter_6 UTXOs
    let voter_6_utxos = truthcoin_nodes.voter_6.rpc_client.get_wallet_utxos().await?;
    let voter_6_total_votecoin: u32 = voter_6_utxos
        .iter()
        .filter_map(|utxo| utxo.output.content.votecoin())
        .sum();
    tracing::info!("voter_6: {} UTXOs, {} total Votecoin", voter_6_utxos.len(), voter_6_total_votecoin);
    for utxo in &voter_6_utxos {
        if let Some(votecoin_amount) = utxo.output.content.votecoin() {
            tracing::info!("  UTXO: address={}, votecoin={}", utxo.output.address, votecoin_amount);
        }
    }

    tracing::info!("✓ Votecoin transfers working correctly!");
    tracing::info!("Expected: voter_0={}, voter_1={}, voter_2={}, voter_3={}, voter_4={}, voter_5={}, voter_6={}, issuer remainder", VOTER_ALLOCATION_0, VOTER_ALLOCATION_1, VOTER_ALLOCATION_2, VOTER_ALLOCATION_3, VOTER_ALLOCATION_4, VOTER_ALLOCATION_5, VOTER_ALLOCATION_6);
    tracing::info!("Actual: voter_0={}, voter_1={}, voter_2={}, voter_3={}, voter_4={}, voter_5={}, voter_6={}, issuer={}", voter_0_total_votecoin, voter_1_total_votecoin, voter_2_total_votecoin, voter_3_total_votecoin, voter_4_total_votecoin, voter_5_total_votecoin, voter_6_total_votecoin, issuer_total_votecoin);

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
    anyhow::ensure!(total_yes == VOTER_ALLOCATION_0);
    anyhow::ensure!(total_no == VOTER_ALLOCATION_1);

    tracing::info!(
        "Vote test completed successfully - Votecoin voting system working"
    );

    // Verify all nodes are at the same block height
    tracing::info!("=== Verifying Block Height Synchronization ===");
    let issuer_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    let voter_0_height = truthcoin_nodes.voter_0.rpc_client.getblockcount().await?;
    let voter_1_height = truthcoin_nodes.voter_1.rpc_client.getblockcount().await?;
    let voter_2_height = truthcoin_nodes.voter_2.rpc_client.getblockcount().await?;
    let voter_3_height = truthcoin_nodes.voter_3.rpc_client.getblockcount().await?;
    let voter_4_height = truthcoin_nodes.voter_4.rpc_client.getblockcount().await?;
    let voter_5_height = truthcoin_nodes.voter_5.rpc_client.getblockcount().await?;
    let voter_6_height = truthcoin_nodes.voter_6.rpc_client.getblockcount().await?;

    anyhow::ensure!(
        issuer_height == voter_0_height &&
        issuer_height == voter_1_height &&
        issuer_height == voter_2_height &&
        issuer_height == voter_3_height &&
        issuer_height == voter_4_height &&
        issuer_height == voter_5_height &&
        issuer_height == voter_6_height,
        "Block height mismatch: issuer={}, voter_0={}, voter_1={}, voter_2={}, voter_3={}, voter_4={}, voter_5={}, voter_6={}",
        issuer_height, voter_0_height, voter_1_height, voter_2_height, voter_3_height, voter_4_height, voter_5_height, voter_6_height
    );

    tracing::info!("✓ All nodes synchronized at block height: {}", issuer_height);
    tracing::info!("  - issuer: {}", issuer_height);
    tracing::info!("  - voter_0: {}", voter_0_height);
    tracing::info!("  - voter_1: {}", voter_1_height);
    tracing::info!("  - voter_2: {}", voter_2_height);
    tracing::info!("  - voter_3: {}", voter_3_height);
    tracing::info!("  - voter_4: {}", voter_4_height);
    tracing::info!("  - voter_5: {}", voter_5_height);
    tracing::info!("  - voter_6: {}", voter_6_height);

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
