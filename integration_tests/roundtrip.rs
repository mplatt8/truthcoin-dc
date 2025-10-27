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
    state::voting::types::{VotingPeriodId, VoterId},
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

    for (&voter_addr, voter) in voter_addresses.iter().zip(&voters) {
        let balance = voter.rpc_client.get_votecoin_balance(voter_addr).await?;
        anyhow::ensure!(balance == VOTER_ALLOCATION);
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
    for voter in voters.iter() {
        let height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(issuer_height == height);
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
        anyhow::ensure!(balance.total > bitcoin::Amount::ZERO);
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
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3].iter() {
        let mut block_received = false;
        for _ in 0..20 {
            let voter_height = voter.rpc_client.getblockcount().await?;
            if voter_height >= issuer_height {
                block_received = true;
                break;
            }
            sleep(std::time::Duration::from_millis(500)).await;
        }
        anyhow::ensure!(block_received);
    }

    // Manual wallet refresh for multi-node test environment (8 nodes on one machine)
    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }

    let wallet_utxos = truthcoin_nodes.voter_0.rpc_client.get_wallet_utxos().await?;
    let chain_utxos = truthcoin_nodes.voter_0.rpc_client.list_utxos().await?;
    let voter_0_addresses: Vec<_> = wallet_utxos.iter().map(|u| u.output.address).collect();
    let voter_0_chain_bitcoin_utxos: Vec<_> = chain_utxos.iter()
        .filter(|utxo| {
            matches!(utxo.output.content, FilledOutputContent::Bitcoin(_))
                && voter_0_addresses.contains(&utxo.output.address)
        })
        .collect();

    anyhow::ensure!(!voter_0_chain_bitcoin_utxos.is_empty());

    let claimed_slots = truthcoin_nodes
        .issuer
        .rpc_client
        .get_claimed_slots_in_period(3)
        .await?;

    anyhow::ensure!(claimed_slots.len() == 4);

    tracing::info!("✓ Phase 2: Claimed 4 decision slots");
    for slot in claimed_slots.iter() {
        tracing::info!("  - {}", slot.question_preview);
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

    for (voter_node, slot_idx, title, description) in market_configs.iter() {
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

        voter_node.rpc_client.create_market(request).await?;
    }

    sleep(std::time::Duration::from_millis(500)).await;

    truthcoin_nodes
        .issuer
        .bmm_single(&mut enforcer_post_setup)
        .await?;

    sleep(std::time::Duration::from_secs(2)).await;

    let markets = truthcoin_nodes.issuer.rpc_client.list_markets().await?;
    anyhow::ensure!(markets.len() == 4);

    for market in &markets {
        anyhow::ensure!(market.state == "Trading");
        anyhow::ensure!(market.outcome_count == 2 || market.outcome_count == 3);
    }

    tracing::info!("✓ Phase 3: Created 4 binary markets");
    for market in markets.iter() {
        tracing::info!("  - {}", market.title);
    }

    let market_ids: Vec<String> = markets.iter().map(|m| m.market_id.clone()).collect();

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

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

    let voter_0_deposit_address = truthcoin_nodes.voter_0.get_deposit_address().await?;
    deposit(
        &mut enforcer_post_setup,
        &mut truthcoin_nodes.voter_0,
        &voter_0_deposit_address,
        VOTER_DEPOSIT_AMOUNT,
        VOTER_DEPOSIT_FEE,
    ).await?;
    sleep(std::time::Duration::from_secs(1)).await;

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
        anyhow::ensure!(voter_height == final_height);
    }

    let final_markets = truthcoin_nodes.issuer.rpc_client.list_markets().await?;
    anyhow::ensure!(final_markets.len() == 4);

    for market in &final_markets {
        let market_detail = truthcoin_nodes.issuer.rpc_client.view_market(market.market_id.clone()).await?;
        if let Some(market_data) = market_detail {
            anyhow::ensure!(market_data.treasury > 0.0);
        } else {
            anyhow::bail!("Market not found");
        }
    }

    tracing::info!("✓ Phase 4: Completed 7 blocks of trading");

    let current_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;

    let voting_period_start_height = 31u32;
    let blocks_to_mine = if current_height < voting_period_start_height {
        voting_period_start_height - current_height
    } else {
        0
    };

    for _ in 0..blocks_to_mine {
        truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
        sleep(std::time::Duration::from_millis(500)).await;
    }

    sleep(std::time::Duration::from_secs(2)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(3)).await;

    let final_height = truthcoin_nodes.issuer.rpc_client.getblockcount().await?;
    anyhow::ensure!(final_height >= voting_period_start_height);

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3].iter() {
        let voter_height = voter.rpc_client.getblockcount().await?;
        anyhow::ensure!(voter_height == final_height);
    }

    let slots_at_voting = truthcoin_nodes.issuer.rpc_client.get_claimed_slots_in_period(3).await?;
    anyhow::ensure!(slots_at_voting.len() == 4);

    for slot in &slots_at_voting {
        let is_voting = truthcoin_nodes.issuer.rpc_client.is_slot_in_voting(slot.slot_id_hex.clone()).await?;
        anyhow::ensure!(is_voting);
    }

    let markets_at_voting = truthcoin_nodes.issuer.rpc_client.list_markets().await?;
    anyhow::ensure!(markets_at_voting.len() == 4);

    for market in markets_at_voting.iter() {
        anyhow::ensure!(market.state == "Voting");
    }

    for market in &markets_at_voting {
        let market_detail = truthcoin_nodes.issuer.rpc_client.view_market(market.market_id.clone()).await?;
        anyhow::ensure!(market_detail.is_some());
    }

    tracing::info!("✓ Phase 5: All markets transitioned to Voting state");

    let decision_slot_ids: Vec<String> = slots_at_voting
        .iter()
        .map(|slot| slot.slot_id_hex.clone())
        .collect();

    anyhow::ensure!(decision_slot_ids.len() == 4);

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

    let voting_period_id = 3u32;

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

        let mut vote_items = Vec::new();
        for (decision_idx, &vote_value) in votes.iter().enumerate() {
            vote_items.push(VoteBatchItem {
                decision_id: decision_slot_ids[decision_idx].clone(),
                vote_value,
            });
        }

        let batch_request = SubmitVoteBatchRequest {
            votes: vote_items,
            period_id: voting_period_id,
            fee_sats: 1000,
        };

        voter.rpc_client.submit_vote_batch(batch_request).await?;
    }

    sleep(std::time::Duration::from_millis(500)).await;
    truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
    sleep(std::time::Duration::from_secs(2)).await;

    for voter in [&truthcoin_nodes.voter_0, &truthcoin_nodes.voter_1, &truthcoin_nodes.voter_2, &truthcoin_nodes.voter_3, &truthcoin_nodes.voter_4, &truthcoin_nodes.voter_5, &truthcoin_nodes.voter_6] {
        voter.rpc_client.refresh_wallet().await?;
    }
    sleep(std::time::Duration::from_secs(1)).await;

    for decision_id in decision_slot_ids.iter() {
        let votes = truthcoin_nodes
            .issuer
            .rpc_client
            .get_decision_votes(decision_id.clone())
            .await?;

        anyhow::ensure!(votes.len() == 7);

        for vote in &votes {
            let voter_idx = voter_addresses
                .iter()
                .position(|addr| addr.to_string() == vote.voter_address)
                .ok_or_else(|| anyhow::anyhow!("Unknown voter address"))?;

            let decision_idx = decision_slot_ids
                .iter()
                .position(|id| id == decision_id)
                .unwrap();

            let expected_value = vote_matrix[voter_idx][decision_idx];
            anyhow::ensure!((vote.vote_value - expected_value).abs() < 0.01);
        }
    }

    for voter_addr in voter_addresses.iter() {
        let voter_votes = truthcoin_nodes
            .issuer
            .rpc_client
            .get_voter_votes(*voter_addr, Some(voting_period_id))
            .await?;

        anyhow::ensure!(voter_votes.len() == 4);
    }

    tracing::info!("\n=== Vote Matrix (Bitcoin Hivemind Figure 5) ===");
    tracing::info!("       D1    D2    D3    D4");
    tracing::info!("     ╔═════╦═════╦═════╦═════╗");
    for (voter_idx, votes) in vote_matrix.iter().enumerate() {
        tracing::info!("V{} → ║ {:>3} ║ {:>3} ║ {:>3} ║ {:>3} ║{}",
            voter_idx + 1,
            votes[0],
            votes[1],
            votes[2],
            votes[3],
            if voter_idx == 2 { " (dissenter)" } else { "" }
        );
    }
    tracing::info!("     ╚═════╩═════╩═════╩═════╝\n");

    tracing::info!("✓ Phase 6: Vote submission completed");

    let period_id = voting_period_id;

    let blocks_to_mine = 10;

    for _ in 1..=blocks_to_mine {
        truthcoin_nodes.issuer.bmm_single(&mut enforcer_post_setup).await?;
        sleep(std::time::Duration::from_millis(100)).await;
    }

    tracing::info!("✓ Phase 7: Voting period closed");

    let period_details = truthcoin_nodes.issuer.rpc_client
        .get_voting_period_details(period_id)
        .await?;

    if period_details.is_none() {
        return Err(anyhow::anyhow!("Voting period does not exist"));
    }

    let consensus_results = match truthcoin_nodes.issuer.rpc_client
        .get_voting_consensus_results(period_id)
        .await
    {
        Ok(results) => results,
        Err(e) => {
            let period_status = truthcoin_nodes.issuer.rpc_client
                .get_voting_period_status(period_id)
                .await
                .unwrap_or_else(|_| "Unknown".to_string());

            return Err(anyhow::anyhow!(
                "Failed to resolve consensus. Error: {}. Period status: {}",
                e, period_status
            ));
        }
    };

    anyhow::ensure!(
        consensus_results.status == "Resolved" || consensus_results.status == "resolved"
    );

    anyhow::ensure!(consensus_results.explained_variance != 0.85);
    anyhow::ensure!(consensus_results.explained_variance != 0.95);
    anyhow::ensure!(
        consensus_results.explained_variance > 0.0 && consensus_results.explained_variance <= 1.0
    );

    let mut reputation_updates_count = 0;
    let mut reputation_changes = vec![];

    for (i, voter_addr) in voter_addresses.iter().enumerate() {
        let voter_id = VoterId::from_address(voter_addr);
        let voter_id_hex = hex::encode(voter_id.as_bytes());

        anyhow::ensure!(voter_id == VoterId::from_address(voter_addr));

        if let Some(rep_update) = consensus_results.reputation_updates.get(&voter_id_hex) {
            reputation_updates_count += 1;
            let delta = rep_update.new_reputation - rep_update.old_reputation;
            reputation_changes.push((i, rep_update.old_reputation, rep_update.new_reputation, delta));

            let current_rep = truthcoin_nodes.issuer.rpc_client
                .get_voter_reputation(voter_id_hex.clone())
                .await?;

            anyhow::ensure!((current_rep - rep_update.new_reputation).abs() < 0.0001);
        }
    }

    anyhow::ensure!(reputation_updates_count == 7);

    tracing::info!("\n=== Consensus Results ===");
    tracing::info!("Period: {} ({})", period_id, consensus_results.status);
    tracing::info!("\nDecision Outcomes:");
    tracing::info!("  D1: {:.2}", consensus_results.outcomes.get(&decision_slot_ids[0]).unwrap_or(&0.0));
    tracing::info!("  D2: {:.2}", consensus_results.outcomes.get(&decision_slot_ids[1]).unwrap_or(&0.0));
    tracing::info!("  D3: {:.2}", consensus_results.outcomes.get(&decision_slot_ids[2]).unwrap_or(&0.0));
    tracing::info!("  D4: {:.2}", consensus_results.outcomes.get(&decision_slot_ids[3]).unwrap_or(&0.0));
    tracing::info!("\nSVD Analysis:");
    tracing::info!("  Explained Variance: {:.4}", consensus_results.explained_variance);
    tracing::info!("  Certainty Score: {:.4}", consensus_results.certainty);
    tracing::info!("\nReputation Updates:");
    for (voter_idx, old_rep, new_rep, delta) in &reputation_changes {
        tracing::info!("  V{}: {:.4} → {:.4} (Δ={:+.4})",
            voter_idx + 1, old_rep, new_rep, delta
        );
    }
    if !consensus_results.outliers.is_empty() {
        tracing::info!("\nOutliers:");
        for outlier in &consensus_results.outliers {
            tracing::info!("  - {}", outlier);
        }
    }
    tracing::info!("");

    let expected_outcomes = vec![1.0, 0.5, 0.0, 0.0];

    for (decision_id, expected) in decision_slot_ids.iter().zip(expected_outcomes.iter()) {
        let actual = consensus_results.outcomes.get(decision_id).unwrap_or(&-1.0);
        let tolerance = 0.01;
        anyhow::ensure!((actual - expected).abs() < tolerance);
    }

    tracing::info!("✓ Phase 8: Consensus resolution completed\n");

    tracing::info!("=== Test Summary ===");
    tracing::info!("All phases completed successfully:");
    tracing::info!("  1. Votecoin distribution");
    tracing::info!("  2. Decision slot claims");
    tracing::info!("  3. Market creation");
    tracing::info!("  4. Trading activity");
    tracing::info!("  5. Voting period transition");
    tracing::info!("  6. Vote submission");
    tracing::info!("  7. Period closure");
    tracing::info!("  8. Consensus resolution\n");

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
