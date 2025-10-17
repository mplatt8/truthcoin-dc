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
    types::{Address, FilledOutputContent, GetAddress as _},
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
        for voter in [&res.voter_0, &res.voter_1, &res.voter_2, &res.voter_3, &res.voter_4, &res.voter_5, &res.voter_6] {
            res.issuer.rpc_client.connect_peer(voter.net_addr().into()).await?;
        }
        tracing::debug!("Connected 8 nodes in P2P network");
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

    let issuer_vk = truthcoin_nodes
        .issuer
        .rpc_client
        .get_new_verifying_key()
        .await?;

    let _issuer_addr =
        truthcoin_nodes.issuer.rpc_client.get_new_address().await?;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

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

    let voter_addresses = [voter_addr_0, voter_addr_1, voter_addr_2, voter_addr_3, voter_addr_4, voter_addr_5, voter_addr_6];

    for &voter_addr in &voter_addresses {
        truthcoin_nodes
            .issuer
            .rpc_client
            .transfer_votecoin(voter_addr, VOTER_ALLOCATION, 0, None)
            .await?;

        truthcoin_nodes
            .issuer
            .bmm_single(&mut enforcer_post_setup)
            .await?;

        sleep(std::time::Duration::from_secs(2)).await;
    }

    let vote_call_msg_sig: Signature = truthcoin_nodes
        .issuer
        .rpc_client
        .sign_arbitrary_msg(issuer_vk, VOTE_CALL_MSG.to_owned())
        .await?;

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

    for (i, (&voter_addr, voter)) in voter_addresses.iter().zip(&voters).enumerate() {
        let balance = voter.rpc_client.get_votecoin_balance(voter_addr).await?;
        anyhow::ensure!(
            balance == VOTER_ALLOCATION,
            "voter_{} balance {} != expected {}",
            i,
            balance,
            VOTER_ALLOCATION
        );
    }

    let vote_weights: HashMap<Address, u32> = {
        let mut weights = HashMap::new();
        let utxos = truthcoin_nodes.issuer.rpc_client.list_utxos().await?;
        for utxo in utxos {
            if let Some(votecoin_amount) = utxo.output.content.votecoin() {
                *weights.entry(utxo.output.address).or_default() += votecoin_amount;
            }
        }
        weights
    };
    anyhow::ensure!(vote_weights.len() >= 7, "Expected at least 7 voters in snapshot, found {}", vote_weights.len());

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

    let issuer_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    for (i, voter) in voters.iter().enumerate() {
        let height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(
            issuer_height == height,
            "Block height mismatch: issuer={}, voter_{}={}",
            issuer_height, i, height
        );
    }

    tracing::info!("✓ Phase 1: Votecoin distribution and voting verified");

    const VOTER_DEPOSIT_AMOUNT: bitcoin::Amount = bitcoin::Amount::from_sat(5_000_000);
    const VOTER_DEPOSIT_FEE: bitcoin::Amount = bitcoin::Amount::from_sat(500_000);

    let voter_0_deposit_address = truthcoin_nodes.voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_0,
        &voter_0_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let voter_1_deposit_address = truthcoin_nodes.voter_1.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_1,
        &voter_1_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let voter_2_deposit_address = truthcoin_nodes.voter_2.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_2,
        &voter_2_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let voter_3_deposit_address = truthcoin_nodes.voter_3.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_3,
        &voter_3_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        let balance = voter.rpc_client.bitcoin_balance().await?;
        anyhow::ensure!(balance.total > bitcoin::Amount::ZERO, "voter should have positive balance after deposit");
    }

    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    let slot_claims = [
        (voter_addr_0, 0, "Decision 1: Will Bitcoin reach $100k in 2025?"),
        (voter_addr_1, 1, "Decision 2: Will Ethereum merge to PoS succeed?"),
        (voter_addr_2, 2, "Decision 3: Will there be 1M BTC addresses by 2026?"),
        (voter_addr_3, 3, "Decision 4: Will Lightning Network capacity exceed 5000 BTC?"),
    ];

    for (i, (_voter_addr, slot_index, question)) in slot_claims.iter().enumerate() {
        let voter_node = match i {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            _ => unreachable!(),
        };

        voter_node
            .rpc_client
            .claim_decision_slot(
                3,
                *slot_index,
                true,
                false,
                question.to_string(),
                None,
                None,
                1000,
            )
            .await?;
    }

    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;

    let issuer_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    for (i, voter) in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3].iter().enumerate() {
        let mut block_received = false;
        for attempt in 0..20 {
            let voter_height = voter.rpc_client.getblockcount().await?;
            if voter_height >= issuer_height {
                block_received = true;
                break;
            }
            sleep(std::time::Duration::from_millis(500)).await;
        }
        anyhow::ensure!(block_received, "voter_{} did not receive block after 10 seconds", i);
    }

    // Manual wallet refresh for multi-node test environment (8 nodes on one machine)
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }

    // Verify voter_0 has Bitcoin UTXOs after wallet refresh
    let wallet_utxos = truthcoin_nodes.voter_0.rpc_client.get_wallet_utxos().await?;
    let chain_utxos = truthcoin_nodes.voter_0.rpc_client.list_utxos().await?;
    let voter_0_addresses: Vec<_> = wallet_utxos.iter().map(|u| u.output.address).collect();
    let voter_0_chain_bitcoin_utxos: Vec<_> = chain_utxos.iter()
        .filter(|utxo| {
            matches!(utxo.output.content, FilledOutputContent::Bitcoin(_))
                && voter_0_addresses.contains(&utxo.output.address)
        })
        .collect();

    anyhow::ensure!(
        !voter_0_chain_bitcoin_utxos.is_empty(),
        "voter_0 has no Bitcoin UTXOs after refresh_wallet"
    );

    let claimed_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .get_claimed_slots_in_period(3)
        .await?;

    anyhow::ensure!(
        claimed_slots.len() == 4,
        "Expected 4 claimed slots in period 3, found {}",
        claimed_slots.len()
    );

    tracing::info!("✓ Phase 2: Claimed 4 decision slots:");
    for (i, slot) in claimed_slots.iter().enumerate() {
        tracing::info!("  {}. {} ({})",
            i + 1,
            slot.question_preview,
            if slot.is_scaled { "scaled" } else { "binary" }
        );
    }

    let market_slot_ids: Vec<String> = claimed_slots
        .iter()
        .map(|slot| slot.slot_id_hex.clone())
        .collect();
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
        market_txids.push(market_id);
    }

    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;

    let markets = truthcoin_nodes.issuer.rpc_client.list_markets().await?;
    anyhow::ensure!(
        markets.len() == 4,
        "Expected 4 markets, found {}",
        markets.len()
    );

    for market in &markets {
        anyhow::ensure!(
            market.state == "Trading",
            "Market {} should be in Trading state, found: {}",
            market.market_id,
            market.state
        );
        anyhow::ensure!(
            market.outcome_count == 2 || market.outcome_count == 3,
            "Binary market {} should have 2-3 outcomes, found {}",
            market.market_id,
            market.outcome_count
        );
    }

    tracing::info!("✓ Phase 3: Created 4 binary markets:");
    for (i, market) in markets.iter().enumerate() {
        tracing::info!("  {}. {} ({})",
            i + 1,
            market.title,
            market.state
        );
    }

    // Phase 4: Trading across 7 blocks
    tracing::info!("Starting Phase 4: Buy shares trading (5 blocks)");

    let market_ids: Vec<String> = markets.iter().map(|m| m.market_id.clone()).collect();

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

    tracing::info!("Block 1: Small initial trades on all markets");
    for (i, market_id) in market_ids.iter().enumerate() {
        let voter = match i {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            _ => unreachable!(),
        };

        let cost = voter.rpc_client.calculate_share_cost(
            market_id.clone(),
            0,
            5.0,
        ).await?;

        voter.rpc_client.buy_shares(
            market_id.clone(),
            0,
            5.0,
            cost + 10000,
            1000,
        ).await?;
    }
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    tracing::info!("Block 2: Balanced trading on market 0");
    truthcoin_nodes.voter_1.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let cost = truthcoin_nodes.voter_1.rpc_client.calculate_share_cost(
        market_ids[0].clone(),
        1,
        5.0,
    ).await?;
    truthcoin_nodes.voter_1.rpc_client.buy_shares(
        market_ids[0].clone(),
        1,
        5.0,
        cost + 10000,
        1000,
    ).await?;
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    tracing::info!("Block 3: Large trade creating skewed market 1");
    truthcoin_nodes.voter_2.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let cost = truthcoin_nodes.voter_2.rpc_client.calculate_share_cost(
        market_ids[1].clone(),
        0,
        50.0,
    ).await?;
    truthcoin_nodes.voter_2.rpc_client.buy_shares(
        market_ids[1].clone(),
        0,
        50.0,
        cost + 50000,
        1000,
    ).await?;
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    tracing::info!("Replenishing voter_2, voter_3, and voters 5-7 funds early");
    let voter_2_deposit_address = truthcoin_nodes.voter_2.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_2,
        &voter_2_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_3_deposit_address = truthcoin_nodes.voter_3.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_3,
        &voter_3_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_millis(500)).await;

    // Fund voters 5-7 for vote submission in Phase 6
    let voter_4_deposit_address = truthcoin_nodes.voter_4.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_4,
        &voter_4_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_5_deposit_address = truthcoin_nodes.voter_5.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_5,
        &voter_5_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let voter_6_deposit_address = truthcoin_nodes.voter_6.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_6,
        &voter_6_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    tracing::info!("Block 4: Multiple small trades on market 2");
    truthcoin_nodes.voter_3.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    for outcome in [0, 1] {
        let cost = truthcoin_nodes.voter_3.rpc_client.calculate_share_cost(
            market_ids[2].clone(),
            outcome,
            2.0,
        ).await?;
        truthcoin_nodes.voter_3.rpc_client.buy_shares(
            market_ids[2].clone(),
            outcome,
            2.0,
            cost + 5000,
            1000,
        ).await?;
    }
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    tracing::info!("Replenishing voter_0 funds");
    let voter_0_deposit_address = truthcoin_nodes.voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_0,
        &voter_0_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    tracing::info!("Block 5: Additional YES position on market 1");
    truthcoin_nodes.voter_3.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let cost = truthcoin_nodes.voter_3.rpc_client.calculate_share_cost(
        market_ids[1].clone(),
        0,
        20.0,
    ).await?;
    truthcoin_nodes.voter_3.rpc_client.buy_shares(
        market_ids[1].clone(),
        0,
        20.0,
        cost + 30000,
        1000,
    ).await?;
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    tracing::info!("Block 6: Contrarian NO position on heavily skewed market 1");
    truthcoin_nodes.voter_0.rpc_client.refresh_wallet().await?;
    sleep(std::time::Duration::from_millis(500)).await;

    let cost = truthcoin_nodes.voter_0.rpc_client.calculate_share_cost(
        market_ids[1].clone(),
        1,
        15.0,
    ).await?;
    truthcoin_nodes.voter_0.rpc_client.buy_shares(
        market_ids[1].clone(),
        1,
        15.0,
        cost + 20000,
        1000,
    ).await?;
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    tracing::info!("Block 7: Initial trading on market 3");
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    for outcome in [0, 1] {
        let voter = if outcome == 0 { &truthcoin_nodes.voter_0 } else { &truthcoin_nodes.voter_1 };
        let cost = voter.rpc_client.calculate_share_cost(
            market_ids[3].clone(),
            outcome,
            10.0,
        ).await?;
        voter.rpc_client.buy_shares(
            market_ids[3].clone(),
            outcome,
            10.0,
            cost + 15000,
            1000,
        ).await?;
    }
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(1)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_millis(500)).await;

    let final_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        let voter_height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(
            voter_height == final_height,
            "Node out of sync: expected {}, got {}",
            final_height,
            voter_height
        );
    }

    let final_markets = truthcoin_nodes.issuer.rpc_client.list_markets().await?;
    anyhow::ensure!(
        final_markets.len() == 4,
        "Expected 4 markets after trading, found {}",
        final_markets.len()
    );

    tracing::info!("✓ Phase 4: Completed 7 blocks of trading");
    tracing::info!("Final market states:");
    for (i, market) in final_markets.iter().enumerate() {
        tracing::info!("  Market {}: volume = {:.2}, state: {}",
            i + 1,
            market.volume,
            market.state
        );
    }

    tracing::info!("Verifying share ownership persistence:");
    for (voter_idx, voter_addr) in [voter_addr_0, voter_addr_1, voter_addr_2, voter_addr_3].iter().enumerate() {
        let holdings = truthcoin_nodes.issuer.rpc_client.get_user_share_positions(*voter_addr).await?;
        if holdings.active_markets > 0 {
            tracing::info!("  Voter {} has {} active positions across {} markets",
                voter_idx,
                holdings.positions.len(),
                holdings.active_markets
            );
        }
    }

    for market in &final_markets {
        let market_detail = truthcoin_nodes.issuer.rpc_client.view_market(market.market_id.clone()).await?;
        if let Some(market_data) = market_detail {
            anyhow::ensure!(
                market_data.treasury > 0.0,
                "Market {} treasury should be positive after trades, found {}",
                market.market_id,
                market_data.treasury
            );
            tracing::info!("  Market {}: treasury = {:.2} sats",
                market.market_id,
                market_data.treasury
            );
        } else {
            anyhow::bail!("Market {} not found", market.market_id);
        }
    }

    tracing::info!("✓ All market states persisted correctly across 7 blocks of trading");

    // Phase 5: Mine to voting period and verify state transitions
    tracing::info!("Starting Phase 5: Mining to voting period");

    let current_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    tracing::info!("Current block height: {}", current_height);

    // Period boundaries: heights 20-29 = period 3 (trading), heights 30-39 = period 4 (voting)
    let voting_period_start_height = 31u32;
    let blocks_to_mine = if current_height < voting_period_start_height {
        voting_period_start_height - current_height
    } else {
        0
    };

    tracing::info!(
        "Need to mine {} blocks to reach height {} (period 4 voting starts)",
        blocks_to_mine,
        voting_period_start_height
    );

    for i in 0..blocks_to_mine {
        tracing::info!("Mining block {} of {}", i + 1, blocks_to_mine);
        truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
        sleep(std::time::Duration::from_millis(500)).await;
    }

    sleep(std::time::Duration::from_secs(2)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(3)).await;

    let final_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    anyhow::ensure!(
        final_height >= voting_period_start_height,
        "Failed to reach voting period: at height {}, expected >= {}",
        final_height,
        voting_period_start_height
    );
    tracing::info!("✓ Reached voting period at height {} (period 4)", final_height);

    for (i, voter) in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3].iter().enumerate() {
        let voter_height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(
            voter_height == final_height,
            "Voter {} out of sync: expected {}, got {}",
            i,
            final_height,
            voter_height
        );
    }

    tracing::info!("Checking decision slot states (period 3):");
    let slots_at_voting = truthcoin_nodes.issuer.rpc_client.get_claimed_slots_in_period(3).await?;
    anyhow::ensure!(
        slots_at_voting.len() == 4,
        "Expected 4 decision slots from period 3, found {}",
        slots_at_voting.len()
    );

    for (i, slot) in slots_at_voting.iter().enumerate() {
        tracing::info!(
            "  Slot {}: {} (period=3, slot_index={})",
            i + 1,
            slot.question_preview,
            slot.slot_index
        );
    }

    for slot in &slots_at_voting {
        let is_voting = truthcoin_nodes.issuer.rpc_client.is_slot_in_voting(slot.slot_id_hex.clone()).await?;
        tracing::info!(
            "  Slot {} in voting: {}",
            slot.slot_id_hex,
            is_voting
        );
        anyhow::ensure!(
            is_voting,
            "Slot {} (period 3) should be in voting at height {} (period 4), but is_slot_in_voting returned false",
            slot.slot_id_hex,
            final_height
        );
    }

    tracing::info!("✓ All 4 decision slots from period 3 are confirmed in voting period");

    let markets_at_voting = truthcoin_nodes.issuer.rpc_client.list_markets().await?;
    anyhow::ensure!(
        markets_at_voting.len() == 4,
        "Expected 4 markets to exist at voting period, found {}",
        markets_at_voting.len()
    );

    tracing::info!("✓ Market states at voting period (height {}, period 4):", final_height);
    for (i, market) in markets_at_voting.iter().enumerate() {
        let market_detail = truthcoin_nodes.issuer.rpc_client.view_market(market.market_id.clone()).await?;

        if let Some(detail) = market_detail {
            tracing::info!(
                "  Market {}: {} - State: {}, Decision slots: {:?}",
                i + 1,
                market.title,
                market.state,
                detail.decision_slots
            );
        } else {
            tracing::info!(
                "  Market {}: {} - State: {} (no detail available)",
                i + 1,
                market.title,
                market.state
            );
        }

        anyhow::ensure!(
            market.state == "Voting",
            "Market {} should be in Voting state, found: {}",
            market.market_id,
            market.state
        );
    }

    tracing::info!("✓ Phase 5: All markets correctly transitioned to Voting state");

    for market in &markets_at_voting {
        let market_detail = truthcoin_nodes.issuer.rpc_client.view_market(market.market_id.clone()).await?;
        anyhow::ensure!(
            market_detail.is_some(),
            "Market {} details should be accessible in voting period",
            market.market_id
        );

        if let Some(detail) = market_detail {
            tracing::info!(
                "  Market {}: treasury={:.2} sats, volume={:.2}",
                market.market_id,
                detail.treasury,
                market.volume
            );
        }
    }

    // Phase 6: Vote submission according to Bitcoin Hivemind whitepaper Figure 5
    tracing::info!("Starting Phase 6: Vote submission (Bitcoin Hivemind whitepaper example)");
    tracing::info!("Implementing vote matrix from Figure 5, page 18:");
    tracing::info!("  7 voters × 4 binary decisions");
    tracing::info!("  Vote values: D1=[1,1,1,1,1,1,1], D2=[0.5,0.5,1,0.5,0.5,0.5,0.5], D3=[0,0,0,0,0,0,0], D4=[0,0,0,0,0,0,0]");

    // Get decision slot IDs for voting
    let decision_slot_ids: Vec<String> = slots_at_voting
        .iter()
        .map(|slot| slot.slot_id_hex.clone())
        .collect();

    anyhow::ensure!(
        decision_slot_ids.len() == 4,
        "Expected 4 decision slots for voting, found {}",
        decision_slot_ids.len()
    );

    tracing::info!("Decision slots for voting:");
    for (i, slot_id) in decision_slot_ids.iter().enumerate() {
        tracing::info!("  D{}: {}", i + 1, slot_id);
    }

    // Whitepaper vote matrix (Figure 5, left example - 7 voters, 4 decisions)
    // Voter 1: [1.0, 0.5, 0.0, 0.0]
    // Voter 2: [1.0, 0.5, 0.0, 0.0]
    // Voter 3: [1.0, 1.0, 0.0, 0.0]  <- dissenter on D2
    // Voter 4: [1.0, 0.5, 0.0, 0.0]
    // Voter 5: [1.0, 0.5, 0.0, 0.0]
    // Voter 6: [1.0, 0.5, 0.0, 0.0]
    // Voter 7: [1.0, 0.5, 0.0, 0.0]
    let vote_matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 0.5, 0.0, 0.0], // Voter 0 (1 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 1 (2 in whitepaper)
        vec![1.0, 1.0, 0.0, 0.0], // Voter 2 (3 in whitepaper) - dissenter
        vec![1.0, 0.5, 0.0, 0.0], // Voter 3 (4 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 4 (5 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 5 (6 in whitepaper)
        vec![1.0, 0.5, 0.0, 0.0], // Voter 6 (7 in whitepaper)
    ];

    // Submit votes for all 7 voters using batch submission
    use truthcoin_dc_app_rpc_api::{SubmitVoteBatchRequest, VoteBatchItem};

    let current_period = 4u32; // We're in period 4 (voting on period 3 decisions)

    tracing::info!("Submitting votes for all 7 voters...");
    for (voter_idx, votes) in vote_matrix.iter().enumerate() {
        let voter = match voter_idx {
            0 => &truthcoin_nodes.voter_0,
            1 => &truthcoin_nodes.voter_1,
            2 => &truthcoin_nodes.voter_2,
            3 => &truthcoin_nodes.voter_3,
            4 => &truthcoin_nodes.voter_4,
            5 => &truthcoin_nodes.voter_5,
            6 => &truthcoin_nodes.voter_6,
            _ => unreachable!(),
        };

        // Build vote batch for this voter
        let mut vote_items = Vec::new();
        for (decision_idx, &vote_value) in votes.iter().enumerate() {
            vote_items.push(VoteBatchItem {
                decision_id: decision_slot_ids[decision_idx].clone(),
                vote_value,
            });
        }

        let batch_request = SubmitVoteBatchRequest {
            votes: vote_items,
            period_id: current_period,
            fee_sats: 1000,
        };

        tracing::info!(
            "  Voter {} submitting batch: D1={}, D2={}, D3={}, D4={}",
            voter_idx + 1,
            votes[0],
            votes[1],
            votes[2],
            votes[3]
        );

        let vote_txid = voter.rpc_client.submit_vote_batch(batch_request).await?;
        tracing::debug!("    Vote batch txid: {}", vote_txid);
    }

    tracing::info!("✓ All 7 voters submitted vote batches");

    // Mine a block to confirm all votes
    sleep(std::time::Duration::from_millis(500)).await;
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(2)).await;

    // Sync all nodes
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3, &truthcoin_nodes.voter_4, &truthcoin_nodes.voter_5, &truthcoin_nodes.voter_6] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

    let vote_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    tracing::info!("✓ Votes confirmed at height {}", vote_height);

    // Verify votes were recorded correctly
    tracing::info!("Verifying vote submissions via RPC...");

    for (decision_idx, decision_id) in decision_slot_ids.iter().enumerate() {
        let votes = truthcoin_nodes
            .issuer
            .rpc_client
            .get_decision_votes(decision_id.clone())
            .await?;

        tracing::info!(
            "  Decision {} (D{}): {} votes recorded",
            decision_id,
            decision_idx + 1,
            votes.len()
        );

        anyhow::ensure!(
            votes.len() == 7,
            "Expected 7 votes for decision {}, found {}",
            decision_id,
            votes.len()
        );

        // Verify vote values match the whitepaper matrix
        // Note: votes are returned in arbitrary order (likely by voter ID hash),
        // so we need to match by voter address rather than assuming order
        for vote in &votes {
            // Find which voter this is by matching address
            let voter_idx = voter_addresses
                .iter()
                .position(|addr| addr.to_string() == vote.voter_address)
                .ok_or_else(|| anyhow::anyhow!(
                    "Unknown voter address: {}",
                    vote.voter_address
                ))?;

            let expected_value = vote_matrix[voter_idx][decision_idx];
            anyhow::ensure!(
                (vote.vote_value - expected_value).abs() < 0.01,
                "Vote mismatch: Voter {} on D{} expected {}, got {}",
                voter_idx + 1,
                decision_idx + 1,
                expected_value,
                vote.vote_value
            );
        }
    }

    tracing::info!("✓ All votes verified against whitepaper matrix");

    // Verify voter participation
    for (voter_idx, voter_addr) in voter_addresses.iter().enumerate() {
        let voter_votes = truthcoin_nodes
            .issuer
            .rpc_client
            .get_voter_votes(*voter_addr, Some(current_period))
            .await?;

        tracing::info!(
            "  Voter {} cast {} votes in period {}",
            voter_idx + 1,
            voter_votes.len(),
            current_period
        );

        anyhow::ensure!(
            voter_votes.len() == 4,
            "Expected 4 votes from voter {}, found {}",
            voter_idx + 1,
            voter_votes.len()
        );
    }

    tracing::info!("✓ Phase 6: Vote submission completed successfully");
    tracing::info!("  - 7 voters each voted on 4 binary decisions");
    tracing::info!("  - Total 28 votes recorded and verified");
    tracing::info!("  - Vote matrix matches Bitcoin Hivemind whitepaper Figure 5");

    // Print complete vote matrix as submitted and verified
    tracing::info!("\n=== FINAL VOTE MATRIX (Bitcoin Hivemind Figure 5) ===");
    tracing::info!("       D1    D2    D3    D4");
    tracing::info!("     ╔═════╦═════╦═════╦═════╗");
    for (voter_idx, votes) in vote_matrix.iter().enumerate() {
        tracing::info!("V{} → ║ {:>3} ║ {:>3} ║ {:>3} ║ {:>3} ║{}",
            voter_idx + 1,
            votes[0],
            votes[1],
            votes[2],
            votes[3],
            if voter_idx == 2 { " (dissenter on D2)" } else { "" }
        );
    }
    tracing::info!("     ╚═════╩═════╩═════╩═════╝");
    tracing::info!("Decision IDs:");
    for (i, decision_id) in decision_slot_ids.iter().enumerate() {
        tracing::info!("  D{}: {}", i + 1, decision_id);
    }
    tracing::info!("Voter addresses:");
    for (i, voter_addr) in voter_addresses.iter().enumerate() {
        tracing::info!("  V{}: {}", i + 1, voter_addr);
    }
    tracing::info!("════════════════════════════════════════════════════\n");

    tracing::info!("\n✓ All 6 phases completed successfully:");
    tracing::info!("  Phase 1: Votecoin distribution and voting");
    tracing::info!("  Phase 2: Decision slot claims");
    tracing::info!("  Phase 3: Market creation");
    tracing::info!("  Phase 4: 7 blocks of trading");
    tracing::info!("  Phase 5: Transition to voting period");
    tracing::info!("  Phase 6: Vote submission per whitepaper");

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
