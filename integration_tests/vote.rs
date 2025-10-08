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
    types::{Address, GetAddress as _, Txid},
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
        Ok(res)
    }
}

/// Seven-node setup for Phase 1 of the Hivemind voting integration test
#[derive(Debug)]
struct SevenNodeSetup {
    voter_0: PostSetup,
    voter_1: PostSetup,
    voter_2: PostSetup,
    voter_3: PostSetup,
    voter_4: PostSetup,
    voter_5: PostSetup,
    voter_6: PostSetup,
}

impl SevenNodeSetup {
    /// Initialize all seven nodes
    async fn setup(
        bin_paths: &BinPaths,
        res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
        enforcer_post_setup: &EnforcerPostSetup,
    ) -> anyhow::Result<Self> {
        tracing::info!("=== Phase 1.1: Seven-Node Network Setup ===");

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

        tracing::info!("Initializing 7 nodes...");
        let res = Self {
            voter_0: setup_single("voter_0").await?,
            voter_1: setup_single("voter_1").await?,
            voter_2: setup_single("voter_2").await?,
            voter_3: setup_single("voter_3").await?,
            voter_4: setup_single("voter_4").await?,
            voter_5: setup_single("voter_5").await?,
            voter_6: setup_single("voter_6").await?,
        };
        tracing::info!("All 7 nodes initialized successfully");

        Ok(res)
    }

    /// Get all nodes as a vector for iteration
    fn all_nodes(&self) -> Vec<&PostSetup> {
        vec![
            &self.voter_0,
            &self.voter_1,
            &self.voter_2,
            &self.voter_3,
            &self.voter_4,
            &self.voter_5,
            &self.voter_6,
        ]
    }

    /// Get mutable references to all nodes
    fn all_nodes_mut(&mut self) -> Vec<&mut PostSetup> {
        vec![
            &mut self.voter_0,
            &mut self.voter_1,
            &mut self.voter_2,
            &mut self.voter_3,
            &mut self.voter_4,
            &mut self.voter_5,
            &mut self.voter_6,
        ]
    }

    /// Setup mesh network topology - connect all nodes to each other
    /// This creates 21 peer connections total (7 nodes * 6 connections each / 2)
    async fn setup_mesh_network(&self) -> anyhow::Result<()> {
        tracing::info!("=== Setting up mesh network topology ===");
        tracing::info!("Creating full mesh: each node connected to all others (21 connections total)");

        let nodes = self.all_nodes();
        let mut connection_count = 0;

        // Connect each node to every other node
        for (i, node_i) in nodes.iter().enumerate() {
            for (j, node_j) in nodes.iter().enumerate() {
                if i < j {  // Only connect once per pair to avoid duplicates
                    tracing::debug!(
                        "Connecting voter_{} ({}) to voter_{} ({})",
                        i,
                        node_i.net_addr(),
                        j,
                        node_j.net_addr()
                    );
                    node_i
                        .rpc_client
                        .connect_peer(node_j.net_addr().into())
                        .await?;
                    connection_count += 1;
                }
            }
        }

        tracing::info!("Mesh network established: {} connections created", connection_count);
        Ok(())
    }

    /// Verify peer connectivity for all nodes
    async fn verify_peer_connectivity(&self) -> anyhow::Result<()> {
        tracing::info!("=== Verifying peer connectivity ===");

        let nodes = self.all_nodes();
        for (i, node) in nodes.iter().enumerate() {
            let peers = node.rpc_client.list_peers().await?;
            let peer_count = peers.len();

            tracing::info!("voter_{}: {} peers connected", i, peer_count);

            // Each node should be connected to 6 other nodes
            anyhow::ensure!(
                peer_count == 6,
                "voter_{} has {} peers, expected 6",
                i,
                peer_count
            );
        }

        tracing::info!("Peer connectivity verified: all nodes have 6 peers");
        Ok(())
    }

    /// Distribute initial funding (Bitcoin deposits) to all nodes
    async fn distribute_initial_funding(
        &mut self,
        enforcer_post_setup: &mut EnforcerPostSetup,
        amount_per_node: bitcoin::Amount,
        fee: bitcoin::Amount,
    ) -> anyhow::Result<()> {
        tracing::info!("=== Phase 1.2: Initial Funding Distribution ===");
        tracing::info!(
            "Funding each of 7 nodes with {} satoshis",
            amount_per_node.to_sat()
        );

        for (i, node) in self.all_nodes_mut().iter_mut().enumerate() {
            let deposit_address = node.get_deposit_address().await?;
            tracing::debug!("Depositing to voter_{} at {}", i, deposit_address);

            deposit(
                enforcer_post_setup,
                *node,
                &deposit_address,
                amount_per_node,
                fee,
            )
            .await?;

            tracing::info!("voter_{}: funded with {} sats", i, amount_per_node.to_sat());
        }

        tracing::info!("All 7 nodes funded successfully");
        Ok(())
    }
}

const DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(21_000_000);
const DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);

// Phase 1 constants
const FUNDING_PER_NODE: bitcoin::Amount = bitcoin::Amount::from_sat(50_000_000); // 50M sats each
const FUNDING_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(1_000_000);       // 1M sats fee

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

/// Seven-node network setup with funding and synchronization via Initial Block Download
async fn seven_nodes_test_task(
    bin_paths: BinPaths,
    res_tx: mpsc::UnboundedSender<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    tracing::info!("SEVEN NODE NETWORK TEST");

    // Setup enforcer
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

    // Phase 1.1: Initialize ONLY voter_0 first
    tracing::info!("Initializing voter_0...");
    let mut voter_0 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_0".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;

    // Phase 1.2: Fund ONLY voter_0 with Bitcoin deposits
    let deposit_address = voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut voter_0,
        &deposit_address,
        FUNDING_PER_NODE,
        FUNDING_FEE,
    )
    .await?;

    // Phase 1.3: Mine blocks on voter_0 to create the canonical blockchain
    voter_0.bmm(&mut enforcer_post_setup, 5).await?;

    // Verify voter_0 has blocks
    let voter_0_blocks = voter_0.rpc_client.getblockcount().await?;
    tracing::info!("✓ voter_0 established canonical chain: {} blocks", voter_0_blocks);
    anyhow::ensure!(
        voter_0_blocks > 0,
        "voter_0 should have at least 1 block, but has {}",
        voter_0_blocks
    );

    // Phase 1.4: Initialize the remaining 6 nodes (they start with empty blockchains)
    tracing::info!("Initializing remaining 6 nodes...");
    let voter_1 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_1".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;
    let voter_2 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_2".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;
    let voter_3 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_3".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;
    let voter_4 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_4".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;
    let voter_5 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_5".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;
    let voter_6 = PostSetup::setup(
        Init {
            truthcoin_app: bin_paths.truthcoin.clone(),
            data_dir_suffix: Some("voter_6".to_owned()),
        },
        &enforcer_post_setup,
        res_tx.clone(),
    ).await?;
    tracing::info!("✓ All 7 nodes initialized");

    // Phase 1.5: Connect nodes in hub-and-ring topology
    tracing::info!("Connecting network (hub-and-ring topology: 12 edges)...");
    let nodes = vec![&voter_0, &voter_1, &voter_2, &voter_3, &voter_4, &voter_5, &voter_6];

    // First: Connect all nodes to voter_0 (6 connections)
    for (i, node) in nodes.iter().enumerate().skip(1) {
        node.rpc_client
            .connect_peer(voter_0.net_addr().into())
            .await?;
    }

    // Second: Connect nodes in a ring (6 more connections)
    // voter_1 -> voter_2 -> voter_3 -> voter_4 -> voter_5 -> voter_6 -> voter_1
    for i in 1..=6 {
        let next_i = if i == 6 { 1 } else { i + 1 };
        nodes[i].rpc_client
            .connect_peer(nodes[next_i].net_addr().into())
            .await?;
    }
    tracing::info!("✓ Network connected");

    // Phase 1.6: Wait for Initial Block Download (IBD) to complete
    tracing::info!("Waiting for IBD sync (10s)...");
    sleep(std::time::Duration::from_secs(10)).await;

    // Phase 1.7: Verify all nodes synced to same height
    tracing::info!("Verifying synchronization...");
    for (i, node) in nodes.iter().enumerate() {
        let block_count = node.rpc_client.getblockcount().await?;
        anyhow::ensure!(
            block_count == voter_0_blocks,
            "voter_{} has {} blocks, expected {}",
            i,
            block_count,
            voter_0_blocks
        );
    }

    tracing::info!("✓ All 7 nodes synced to {} blocks", voter_0_blocks);
    tracing::info!("SEVEN NODE TEST: COMPLETE ✓");

    // Cleanup
    {
        drop(voter_0);
        drop(voter_1);
        drop(voter_2);
        drop(voter_3);
        drop(voter_4);
        drop(voter_5);
        drop(voter_6);
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

async fn vote_task(
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
    let utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
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

    tracing::info!("Distributing Votecoin to voters");
    // Transfer some Votecoin from issuer to voters
    let _: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_0, VOTER_ALLOCATION_0, 0, None)
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    let _: Txid = truthcoin_nodes
        .issuer
        .rpc_client
        .transfer_votecoin(voter_addr_1, VOTER_ALLOCATION_1, 0, None)
        .await?;
    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    tracing::info!("Signing vote call message");
    let vote_call_msg_sig: Signature = truthcoin_nodes
        .issuer
        .rpc_client
        .sign_arbitrary_msg(issuer_vk, VOTE_CALL_MSG.to_owned())
        .await?;

    tracing::info!("Verifying vote call message signature");
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1] {
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
    let vote_weights: HashMap<Address, u32> = {
        let mut weights = HashMap::new();
        let utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
        for utxo in utxos {
            if let Some(votecoin_amount) = utxo.output.content.votecoin() {
                *weights.entry(utxo.output.address).or_default() +=
                    votecoin_amount;
            }
        }
        weights
    };
    anyhow::ensure!(vote_weights.len() >= 2);

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

async fn seven_nodes_test(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = seven_nodes_test_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn seven_nodes_test_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("seven_nodes_test", seven_nodes_test(bin_paths).boxed())
}

async fn vote(bin_paths: BinPaths) -> anyhow::Result<()> {
    let (res_tx, mut res_rx) = mpsc::unbounded();
    let _test_task: AbortOnDrop<()> = tokio::task::spawn({
        let res_tx = res_tx.clone();
        async move {
            let res = vote_task(bin_paths, res_tx.clone()).await;
            let _send_err: Result<(), _> = res_tx.unbounded_send(res);
        }
        .in_current_span()
    })
    .into();
    res_rx.next().await.ok_or_else(|| {
        anyhow::anyhow!("Unexpected end of test task result stream")
    })?
}

pub fn vote_trial(
    bin_paths: BinPaths,
) -> AsyncTrial<BoxFuture<'static, anyhow::Result<()>>> {
    AsyncTrial::new("vote", vote(bin_paths).boxed())
}
