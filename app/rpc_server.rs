use std::net::SocketAddr;

use bip39;
use bitcoin::Amount;
use jsonrpsee::{
    core::{RpcResult, async_trait},
    server::{RpcServiceBuilder, Server},
    types::ErrorObject,
};

use tower_http::{
    request_id::{
        MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
    },
    trace::{DefaultOnFailure, DefaultOnResponse, TraceLayer},
};
use truthcoin_dc::{
    authorization::{self, Dst, Signature},
    net::Peer,
    node::Node,
    types::{
        Address, Authorization, Block, BlockHash, EncryptionPubKey,
        FilledOutputContent, PointedOutput, Transaction, Txid, VerifyingKey,
        WithdrawalBundle,
    },
    validation::SlotValidator,
    wallet::Balance,
};
use truthcoin_dc_app_rpc_api::{
    AvailableSlotId, RegisterVoterRequest, RpcServer, SubmitVoteBatchRequest,
    SubmitVoteRequest, TxInfo, VoteInfo, VoterInfo, VoterParticipation,
    VotingPeriodDetails,
};

use crate::app::App;

fn custom_err_msg(err_msg: impl Into<String>) -> ErrorObject<'static> {
    ErrorObject::owned(-1, err_msg.into(), Option::<()>::None)
}

fn custom_err<Error>(error: Error) -> ErrorObject<'static>
where
    anyhow::Error: From<Error>,
{
    let error = anyhow::Error::from(error);
    custom_err_msg(format!("{error:#}"))
}

fn parse_market_id(
    market_id: &str,
) -> RpcResult<truthcoin_dc::state::MarketId> {
    let market_id_bytes = hex::decode(market_id)
        .map_err(|_| custom_err_msg("Invalid market ID hex format"))?;

    if market_id_bytes.len() != 6 {
        return Err(custom_err_msg("Market ID must be exactly 6 bytes"));
    }

    let mut id_array = [0u8; 6];
    id_array.copy_from_slice(&market_id_bytes);
    Ok(truthcoin_dc::state::MarketId::new(id_array))
}

pub struct RpcServerImpl {
    app: App,
}

impl RpcServerImpl {
    #[inline(always)]
    fn node(&self) -> &Node {
        &self.app.node
    }

    #[inline]
    fn block_height_to_testing_period(
        block_height: u32,
        testing_blocks_per_period: u32,
    ) -> u32 {
        if testing_blocks_per_period == 0 {
            0
        } else {
            (block_height / testing_blocks_per_period) + 1
        }
    }

    #[inline]
    fn timestamp_to_period(timestamp: u64) -> u32 {
        const BITCOIN_GENESIS_TIMESTAMP: u64 = 1231006505;
        const SECONDS_PER_QUARTER: u64 = 3 * 30 * 24 * 60 * 60;

        if timestamp < BITCOIN_GENESIS_TIMESTAMP {
            return 0;
        }
        let elapsed_seconds = timestamp - BITCOIN_GENESIS_TIMESTAMP;
        (elapsed_seconds / SECONDS_PER_QUARTER) as u32
    }

    #[inline]
    fn period_to_name(period: u32) -> String {
        if period == 0 {
            return "Genesis".to_string();
        }

        let year = 2009 + (period - 1) / 4;
        let quarter = ((period - 1) % 4) + 1;

        format!("Q{} Y{}", quarter, year)
    }
}

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn get_block(&self, block_hash: BlockHash) -> RpcResult<Block> {
        let block = self.node().get_block(block_hash).map_err(custom_err)?;
        Ok(block)
    }

    async fn get_best_sidechain_block_hash(
        &self,
    ) -> RpcResult<Option<BlockHash>> {
        self.node().try_get_tip().map_err(custom_err)
    }

    async fn get_best_mainchain_block_hash(
        &self,
    ) -> RpcResult<Option<bitcoin::BlockHash>> {
        let Some(sidechain_hash) =
            self.node().try_get_tip().map_err(custom_err)?
        else {
            return Ok(None);
        };
        let block_hash = self
            .node()
            .get_best_main_verification(sidechain_hash)
            .map_err(custom_err)?;
        Ok(Some(block_hash))
    }

    async fn get_bmm_inclusions(
        &self,
        block_hash: truthcoin_dc::types::BlockHash,
    ) -> RpcResult<Vec<bitcoin::BlockHash>> {
        self.app
            .node
            .get_bmm_inclusions(block_hash)
            .map_err(custom_err)
    }

    async fn get_new_address(&self) -> RpcResult<Address> {
        self.app.wallet.get_new_address().map_err(custom_err)
    }

    async fn get_new_encryption_key(&self) -> RpcResult<EncryptionPubKey> {
        self.app.wallet.get_new_encryption_key().map_err(custom_err)
    }

    async fn get_new_verifying_key(&self) -> RpcResult<VerifyingKey> {
        self.app.wallet.get_new_verifying_key().map_err(custom_err)
    }

    async fn get_transaction(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<Transaction>> {
        self.node().try_get_transaction(txid).map_err(custom_err)
    }

    async fn get_transaction_info(
        &self,
        txid: Txid,
    ) -> RpcResult<Option<TxInfo>> {
        let Some((filled_tx, txin)) = self
            .app
            .node
            .try_get_filled_transaction(txid)
            .map_err(custom_err)?
        else {
            return Ok(None);
        };
        let confirmations = match txin {
            Some(txin) => {
                let tip_height = self
                    .app
                    .node
                    .try_get_tip_height()
                    .map_err(custom_err)?
                    .expect("Height should exist for tip");
                let height = self
                    .app
                    .node
                    .get_height(txin.block_hash)
                    .map_err(custom_err)?;
                Some(tip_height - height)
            }
            None => None,
        };
        let fee_sats = filled_tx
            .transaction
            .bitcoin_fee()
            .map_err(custom_err)?
            .unwrap()
            .to_sat();
        let res = TxInfo {
            confirmations,
            fee_sats,
            txin,
        };
        Ok(Some(res))
    }

    async fn get_wallet_addresses(&self) -> RpcResult<Vec<Address>> {
        let addrs = self.app.wallet.get_addresses().map_err(custom_err)?;
        let mut res: Vec<_> = addrs.into_iter().collect();
        res.sort_by_key(|addr| addr.as_base58());
        Ok(res)
    }

    async fn get_wallet_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self.app.wallet.get_utxos().map_err(custom_err)?;
        let utxos = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn getblockcount(&self) -> RpcResult<u32> {
        let height = self.node().try_get_tip_height().map_err(custom_err)?;
        let height = height.unwrap_or(0);
        Ok(height + 1)
    }

    async fn latest_failed_withdrawal_bundle_height(
        &self,
    ) -> RpcResult<Option<u32>> {
        let height = self
            .app
            .node
            .get_latest_failed_bundle_height()
            .map_err(custom_err)?;
        Ok(height)
    }

    async fn list_peers(&self) -> RpcResult<Vec<Peer>> {
        let peers = self.node().get_active_peers();
        Ok(peers)
    }

    async fn list_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self.node().get_all_utxos().map_err(custom_err)?;
        let res = utxos
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(res)
    }

    async fn mine(&self, fee: Option<u64>) -> RpcResult<()> {
        let fee = fee.map(bitcoin::Amount::from_sat);
        self.app
            .local_pool
            .spawn_pinned({
                let app = self.app.clone();
                move || async move { app.mine(fee).await.map_err(custom_err) }
            })
            .await
            .unwrap()
    }

    async fn my_unconfirmed_utxos(&self) -> RpcResult<Vec<PointedOutput>> {
        let addresses = self.app.wallet.get_addresses().map_err(custom_err)?;
        let utxos = self
            .app
            .node
            .get_unconfirmed_utxos_by_addresses(&addresses)
            .map_err(custom_err)?
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn my_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self
            .app
            .wallet
            .get_utxos()
            .map_err(custom_err)?
            .into_iter()
            .map(|(outpoint, output)| PointedOutput { outpoint, output })
            .collect();
        Ok(utxos)
    }

    async fn openapi_schema(&self) -> RpcResult<utoipa::openapi::OpenApi> {
        let res =
            <truthcoin_dc_app_rpc_api::RpcDoc as utoipa::OpenApi>::openapi();
        Ok(res)
    }

    async fn pending_withdrawal_bundle(
        &self,
    ) -> RpcResult<Option<WithdrawalBundle>> {
        self.app
            .node
            .get_pending_withdrawal_bundle()
            .map_err(custom_err)
    }

    async fn remove_from_mempool(&self, txid: Txid) -> RpcResult<()> {
        self.node().remove_from_mempool(txid).map_err(custom_err)
    }

    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()> {
        self.app
            .wallet
            .set_seed_from_mnemonic(mnemonic.as_str())
            .map_err(custom_err)
    }

    async fn sidechain_wealth_sats(&self) -> RpcResult<u64> {
        let sidechain_wealth =
            self.node().get_sidechain_wealth().map_err(custom_err)?;
        Ok(sidechain_wealth.to_sat())
    }

    async fn sign_arbitrary_msg(
        &self,
        verifying_key: VerifyingKey,
        msg: String,
    ) -> RpcResult<Signature> {
        self.app
            .wallet
            .sign_arbitrary_msg(&verifying_key, &msg)
            .map_err(custom_err)
    }

    async fn sign_arbitrary_msg_as_addr(
        &self,
        address: Address,
        msg: String,
    ) -> RpcResult<Authorization> {
        self.app
            .wallet
            .sign_arbitrary_msg_as_addr(&address, &msg)
            .map_err(custom_err)
    }

    async fn stop(&self) {
        std::process::exit(0);
    }

    async fn transfer(
        &self,
        dest: Address,
        value_sats: u64,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid> {
        let memo = match memo {
            None => None,
            Some(memo) => {
                let hex = hex::decode(memo).map_err(custom_err)?;
                Some(hex)
            }
        };
        let tx = self
            .app
            .wallet
            .create_transfer(
                dest,
                Amount::from_sat(value_sats),
                Amount::from_sat(fee_sats),
                memo,
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn transfer_votecoin(
        &self,
        dest: Address,
        amount: u32,
        fee_sats: u64,
        memo: Option<String>,
    ) -> RpcResult<Txid> {
        let memo = match memo {
            None => None,
            Some(memo) => {
                let hex = hex::decode(memo).map_err(custom_err)?;
                Some(hex)
            }
        };
        let tx = self
            .app
            .wallet
            .create_votecoin_transfer(
                dest,
                amount,
                Amount::from_sat(fee_sats),
                memo,
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn verify_signature(
        &self,
        signature: Signature,
        verifying_key: VerifyingKey,
        dst: Dst,
        msg: String,
    ) -> RpcResult<bool> {
        let res = authorization::verify(
            signature,
            &verifying_key,
            dst,
            msg.as_bytes(),
        );
        Ok(res)
    }

    async fn withdraw(
        &self,
        mainchain_address: bitcoin::Address<bitcoin::address::NetworkUnchecked>,
        amount_sats: u64,
        fee_sats: u64,
        mainchain_fee_sats: u64,
    ) -> RpcResult<Txid> {
        let tx = self
            .app
            .wallet
            .create_withdrawal(
                mainchain_address,
                Amount::from_sat(amount_sats),
                Amount::from_sat(mainchain_fee_sats),
                Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn slots_list_all(
        &self,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::SlotInfo>> {
        let slots = self.node().get_all_slot_quarters().map_err(custom_err)?;
        let result = slots
            .into_iter()
            .map(|(period, slots)| truthcoin_dc_app_rpc_api::SlotInfo {
                period,
                slots,
            })
            .collect();
        Ok(result)
    }

    async fn slots_get_quarter(&self, quarter: u32) -> RpcResult<u64> {
        self.app
            .node
            .get_slots_for_quarter(quarter)
            .map_err(custom_err)
    }

    async fn slots_status(
        &self,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::SlotStatus> {
        let is_testing_mode = self.node().is_slots_testing_mode();
        let blocks_per_period = if is_testing_mode {
            self.node().get_slots_testing_config()
        } else {
            0
        };

        let current_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let current_period = if is_testing_mode {
            let tip_height = self
                .app
                .node
                .try_get_tip_height()
                .map_err(custom_err)?
                .unwrap_or(0);
            let testing_blocks_per_period =
                self.node().get_slots_testing_config();
            Self::block_height_to_testing_period(
                tip_height,
                testing_blocks_per_period,
            )
        } else {
            Self::timestamp_to_period(current_timestamp)
        };

        let current_period_name = Self::period_to_name(current_period);

        Ok(truthcoin_dc_app_rpc_api::SlotStatus {
            is_testing_mode,
            blocks_per_period,
            current_period,
            current_period_name,
        })
    }

    async fn timestamp_to_quarter(&self, timestamp: u64) -> RpcResult<u32> {
        Ok(Self::timestamp_to_period(timestamp))
    }

    async fn quarter_to_string(&self, quarter: u32) -> RpcResult<String> {
        Ok(Self::period_to_name(quarter))
    }

    async fn block_height_to_testing_period(
        &self,
        block_height: u32,
    ) -> RpcResult<u32> {
        let testing_blocks_per_period = self.node().get_slots_testing_config();
        Ok(Self::block_height_to_testing_period(
            block_height,
            testing_blocks_per_period,
        ))
    }

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
    ) -> RpcResult<Txid> {
        use truthcoin_dc::state::slots::SlotId;

        if question.as_bytes().len() >= 1000 {
            return Err(custom_err_msg(
                "Question must be less than 1000 bytes",
            ));
        }

        let slot_id =
            SlotId::new(period_index, slot_index).map_err(custom_err)?;
        let slot_id_bytes = slot_id.as_bytes();
        let fee = Amount::from_sat(fee_sats);
        let tx = self
            .app
            .wallet
            .claim_decision_slot(
                slot_id_bytes,
                is_standard,
                is_scaled,
                question,
                min,
                max,
                fee,
            )
            .map_err(custom_err)?;

        let txid = tx.txid();

        // Sign and send the transaction
        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(txid)
    }

    async fn get_available_slots_in_period(
        &self,
        period_index: u32,
    ) -> RpcResult<Vec<AvailableSlotId>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let period_id = VotingPeriodId(period_index);
        let available_slots = self
            .app
            .node
            .get_available_slots_in_period(period_id)
            .map_err(custom_err)?;

        let result = available_slots
            .into_iter()
            .map(|slot_id| AvailableSlotId {
                period_index: slot_id.period_index(),
                slot_index: slot_id.slot_index(),
                slot_id_hex: slot_id.to_hex(),
            })
            .collect();

        Ok(result)
    }

    async fn get_slot_by_id(
        &self,
        slot_id_hex: String,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::SlotDetails>> {
        let slot_id = SlotValidator::parse_slot_id_from_hex(&slot_id_hex)
            .map_err(custom_err)?;
        let slot_opt = self.node().get_slot(slot_id).map_err(custom_err)?;

        let result = slot_opt.map(|slot| {
            let content = match slot.decision {
                None => truthcoin_dc_app_rpc_api::SlotContentInfo::Empty,
                Some(decision) => {
                    truthcoin_dc_app_rpc_api::SlotContentInfo::Decision(
                        truthcoin_dc_app_rpc_api::DecisionInfo {
                            id: hex::encode(decision.id),
                            market_maker_pubkey_hash: hex::encode(
                                decision.market_maker_pubkey_hash,
                            ),
                            is_standard: decision.is_standard,
                            is_scaled: decision.is_scaled,
                            question: decision.question,
                            min: decision.min,
                            max: decision.max,
                        },
                    )
                }
            };

            truthcoin_dc_app_rpc_api::SlotDetails {
                slot_id_hex: slot_id.to_hex(),
                period_index: slot_id.period_index(),
                slot_index: slot_id.slot_index(),
                content,
            }
        });

        Ok(result)
    }

    async fn get_claimed_slots_in_period(
        &self,
        period_index: u32,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::ClaimedSlotSummary>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let period_id = VotingPeriodId(period_index);
        let claimed_slots_data = self
            .app
            .node
            .get_claimed_slots_in_period(period_id)
            .map_err(custom_err)?;

        let claimed_slots = claimed_slots_data
            .into_iter()
            .filter_map(|slot| {
                if let Some(decision) = slot.decision {
                    let question_preview = if decision.question.len() > 100 {
                        format!("{}...", &decision.question[..100])
                    } else {
                        decision.question.clone()
                    };

                    Some(truthcoin_dc_app_rpc_api::ClaimedSlotSummary {
                        slot_id_hex: slot.slot_id.to_hex(),
                        period_index: slot.slot_id.period_index(),
                        slot_index: slot.slot_id.slot_index(),
                        market_maker_pubkey_hash: hex::encode(
                            decision.market_maker_pubkey_hash,
                        ),
                        is_standard: decision.is_standard,
                        is_scaled: decision.is_scaled,
                        question_preview,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(claimed_slots)
    }

    async fn get_voting_periods(
        &self,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::VotingPeriodInfo>> {
        let voting_periods =
            self.node().get_voting_periods().map_err(custom_err)?;

        let result = voting_periods
            .into_iter()
            .map(|(period, claimed_slots, total_slots)| {
                truthcoin_dc_app_rpc_api::VotingPeriodInfo {
                    period,
                    claimed_slots,
                    total_slots,
                }
            })
            .collect();

        Ok(result)
    }

    async fn is_slot_in_voting(&self, slot_id_hex: String) -> RpcResult<bool> {
        use truthcoin_dc::validation::SlotValidator;

        let slot_id = SlotValidator::parse_slot_id_from_hex(&slot_id_hex)
            .map_err(custom_err)?;
        let is_voting = self
            .app
            .node
            .is_slot_in_voting(slot_id)
            .map_err(custom_err)?;

        Ok(is_voting)
    }

    async fn get_ossified_slots(
        &self,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::OssifiedSlotInfo>> {
        let ossified_slots =
            self.node().get_ossified_slots().map_err(custom_err)?;

        let result = ossified_slots
            .into_iter()
            .map(|slot| {
                let decision = slot.decision.map(|decision| {
                    truthcoin_dc_app_rpc_api::DecisionInfo {
                        id: hex::encode(decision.id),
                        market_maker_pubkey_hash: hex::encode(
                            decision.market_maker_pubkey_hash,
                        ),
                        is_standard: decision.is_standard,
                        is_scaled: decision.is_scaled,
                        question: decision.question,
                        min: decision.min,
                        max: decision.max,
                    }
                });

                truthcoin_dc_app_rpc_api::OssifiedSlotInfo {
                    slot_id_hex: slot.slot_id.to_hex(),
                    period_index: slot.slot_id.period_index(),
                    slot_index: slot.slot_id.slot_index(),
                    decision,
                }
            })
            .collect();

        Ok(result)
    }

    async fn list_markets(
        &self,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::MarketSummary>> {
        // Get markets with computed states from SlotStateHistory (single source of truth)
        let markets_with_states = self
            .app
            .node
            .get_all_markets_with_states()
            .map_err(custom_err)?;

        let market_summaries = markets_with_states
            .into_iter()
            .map(|(market, computed_state)| {
                let market_id_hex = hex::encode(market.id.as_bytes());

                truthcoin_dc_app_rpc_api::MarketSummary {
                    market_id: market_id_hex,
                    title: market.title.clone(),
                    description: if market.description.len() > 100 {
                        format!("{}...", &market.description[..97])
                    } else {
                        market.description.clone()
                    },
                    outcome_count: market.get_outcome_count(),
                    state: format!("{:?}", computed_state),
                    volume: market.treasury(),
                    created_at_height: market.created_at_height,
                }
            })
            .collect();

        Ok(market_summaries)
    }

    async fn view_market(
        &self,
        market_id: String,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::MarketData>> {
        let market_id_struct = parse_market_id(&market_id)?;

        // Get market with computed state from SlotStateHistory (single source of truth)
        let (market, computed_state) = match self
            .app
            .node
            .get_market_by_id_with_state(&market_id_struct)
            .map_err(custom_err)?
        {
            Some((market, computed_state)) => (market, computed_state),
            None => return Ok(None),
        };

        let decisions = self
            .app
            .node
            .get_market_decisions(&market)
            .map_err(custom_err)?;

        let mut outcomes = Vec::new();
        let valid_state_combos = market.get_valid_state_combos();

        let prices = if let Some(mempool_shares) = self
            .app
            .node
            .get_mempool_shares(&market_id_struct)
            .map_err(custom_err)?
        {
            let all_prices = market.calculate_prices(&mempool_shares);
            let valid_prices: Vec<f64> = valid_state_combos
                .iter()
                .map(|(state_idx, _)| all_prices[*state_idx])
                .collect();

            let valid_sum: f64 = valid_prices.iter().sum();
            if valid_sum > 0.0 {
                valid_prices.iter().map(|p| p / valid_sum).collect()
            } else {
                vec![1.0 / valid_prices.len() as f64; valid_prices.len()]
            }
        } else {
            market.calculate_prices_for_display()
        };

        let total_volume = market.total_volume_sats as f64;

        for (i, (state_idx, _combo)) in valid_state_combos.iter().enumerate() {
            let name = match market
                .describe_outcome_by_state(*state_idx, &decisions)
            {
                Ok(description) => description,
                Err(_) => format!("Outcome {}", state_idx),
            };

            let current_price = prices[i];
            let probability = current_price;
            let volume = if i < market.outcome_volumes_sats.len() {
                market.outcome_volumes_sats[i] as f64
            } else {
                0.0
            };

            outcomes.push(truthcoin_dc_app_rpc_api::MarketOutcome {
                name,
                current_price,
                probability,
                volume,
                index: i,
            });
        }

        let decision_slots: Vec<String> = market
            .decision_slots
            .iter()
            .map(|slot_id| slot_id.to_hex())
            .collect();

        // computed_state already obtained above from SlotStateHistory

        let market_data = truthcoin_dc_app_rpc_api::MarketData {
            market_id,
            title: market.title.clone(),
            description: market.description.clone(),
            outcomes,
            state: format!("{:?}", computed_state),
            market_maker: market.creator_address.to_string(),
            expires_at: market.expires_at_height,
            beta: market.b(),
            trading_fee: market.trading_fee(),
            tags: market.tags.clone(),
            created_at_height: market.created_at_height,
            treasury: market.treasury(),
            total_volume,
            liquidity: market.treasury(),
            decision_slots,
        };

        Ok(Some(market_data))
    }

    async fn redeem_shares(
        &self,
        market_id: String,
        outcome_index: usize,
        shares_amount: f64,
        fee_sats: u64,
    ) -> RpcResult<String> {
        let market_id_struct = parse_market_id(&market_id)?;

        let tx = self
            .app
            .wallet
            .redeem_shares(
                market_id_struct,
                outcome_index,
                shares_amount,
                bitcoin::Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;

        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{}", txid))
    }

    async fn create_market(
        &self,
        request: truthcoin_dc_app_rpc_api::CreateMarketRequest,
    ) -> RpcResult<String> {
        let tx = self
            .app
            .wallet
            .create_market(
                request.title,
                request.description,
                request.market_type,
                if request.decision_slots.is_empty() {
                    None
                } else {
                    Some(request.decision_slots)
                },
                request.dimensions,
                request.has_residual,
                request.beta,
                request.trading_fee,
                request.tags,
                request.initial_liquidity,
                bitcoin::Amount::from_sat(request.fee_sats),
            )
            .map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        let market_id_bytes = &txid.as_slice()[..6];
        let market_id_hex = hex::encode(market_id_bytes);

        Ok(market_id_hex)
    }

    async fn calculate_initial_liquidity(
        &self,
        request: truthcoin_dc_app_rpc_api::CalculateInitialLiquidityRequest,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::InitialLiquidityCalculation> {
        let beta = request.beta;

        if beta <= 0.0 {
            return Err(custom_err_msg(format!(
                "Beta parameter must be positive, got: {}",
                beta
            )));
        }
        let (num_outcomes, market_config, outcome_breakdown) = if let Some(
            num,
        ) =
            request.num_outcomes
        {
            if num < 2 {
                return Err(custom_err_msg(
                    "Number of outcomes must be at least 2".to_string(),
                ));
            }
            (
                num,
                format!("Preview: {} outcomes", num),
                format!("{} outcomes specified", num),
            )
        } else if let Some(ref decision_slots) = request.decision_slots {
            match request.market_type.as_str() {
                "independent" => {
                    let num = 2_usize.pow(decision_slots.len() as u32);
                    (
                        num,
                        format!(
                            "Independent: {} binary decisions",
                            decision_slots.len()
                        ),
                        format!(
                            "{} binary decisions = 2^{} = {} total outcome combinations",
                            decision_slots.len(),
                            decision_slots.len(),
                            num
                        ),
                    )
                }
                "categorical" => {
                    let base_outcomes = decision_slots.len();
                    let has_residual = request.has_residual.unwrap_or(false);
                    let total = if has_residual {
                        base_outcomes + 1
                    } else {
                        base_outcomes
                    };
                    (
                        total,
                        format!(
                            "Categorical: {} categories{}",
                            base_outcomes,
                            if has_residual { " + residual" } else { "" }
                        ),
                        format!(
                            "{} categories{} = {} total outcomes",
                            base_outcomes,
                            if has_residual { " + 1 residual" } else { "" },
                            total
                        ),
                    )
                }
                "dimensional" => {
                    if let Some(ref _dims) = request.dimensions {
                        let estimated_outcomes = decision_slots.len() * 2;
                        (
                            estimated_outcomes,
                            format!(
                                "Dimensional: {} dimensions",
                                decision_slots.len()
                            ),
                            format!(
                                "Estimated {} outcomes from {} dimensions (actual calculation requires dimension parsing)",
                                estimated_outcomes,
                                decision_slots.len()
                            ),
                        )
                    } else {
                        return Err(custom_err_msg(
                            "Dimensional markets require dimension specification".to_string()
                        ));
                    }
                }
                _ => {
                    return Err(custom_err_msg(format!(
                        "Unknown market type: {}",
                        request.market_type
                    )));
                }
            }
        } else {
            return Err(custom_err_msg(
                "Either decision_slots or num_outcomes must be provided"
                    .to_string(),
            ));
        };

        let initial_liquidity_f64 = beta * (num_outcomes as f64).ln();
        let initial_liquidity_sats = initial_liquidity_f64.ceil() as u64;

        Ok(truthcoin_dc_app_rpc_api::InitialLiquidityCalculation {
            beta,
            num_outcomes,
            initial_liquidity_sats,
            min_treasury_sats: initial_liquidity_sats,
            market_config,
            outcome_breakdown,
        })
    }

    async fn get_user_share_positions(
        &self,
        address: Address,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::UserHoldings> {
        let node = &self.node();

        let positions_data = node
            .get_user_share_positions(&address)
            .map_err(custom_err)?;

        let mut positions = Vec::new();
        let mut total_value = 0.0;
        let mut total_cost_basis = 0.0;
        let mut active_markets = std::collections::HashSet::new();
        let mut last_updated_height = 0u64;

        let unique_market_ids: Vec<truthcoin_dc::state::MarketId> =
            positions_data
                .iter()
                .map(|(market_id, _, _)| market_id.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

        let markets_map = node
            .get_markets_batch(&unique_market_ids)
            .map_err(custom_err)?;

        let mut prices_cache = std::collections::HashMap::new();
        for (market_id, market) in &markets_map {
            prices_cache.insert(market_id.clone(), market.get_current_prices());
        }

        for (market_id, outcome_index, position_data) in positions_data {
            if let (Some(market), Some(current_prices)) =
                (markets_map.get(&market_id), prices_cache.get(&market_id))
            {
                let outcome_price = current_prices
                    .get(outcome_index as usize)
                    .copied()
                    .unwrap_or(0.0);
                let shares_held = position_data;
                let current_value = shares_held * outcome_price;

                let outcome_name = if let Some(combo) =
                    market.state_combos.get(outcome_index as usize)
                {
                    format!("Outcome {}: {:?}", outcome_index, combo)
                } else {
                    format!("Outcome {}", outcome_index)
                };

                positions.push(truthcoin_dc_app_rpc_api::SharePosition {
                    market_id: market_id.to_string(),
                    outcome_index: outcome_index as usize,
                    outcome_name,
                    shares_held: shares_held,
                    avg_purchase_price: outcome_price,
                    current_price: outcome_price,
                    current_value,
                    unrealized_pnl: 0.0,
                    cost_basis: current_value,
                });

                total_value += current_value;
                total_cost_basis += current_value;
                active_markets.insert(market_id);
                last_updated_height = last_updated_height.max(1);
            }
        }

        let total_unrealized_pnl = total_value - total_cost_basis;

        Ok(truthcoin_dc_app_rpc_api::UserHoldings {
            address: address.to_string(),
            positions,
            total_value,
            total_cost_basis,
            total_unrealized_pnl,
            active_markets: active_markets.len(),
            last_updated_height,
        })
    }

    async fn get_market_share_positions(
        &self,
        address: Address,
        market_id: String,
    ) -> RpcResult<Vec<truthcoin_dc_app_rpc_api::SharePosition>> {
        let market_id_struct = parse_market_id(&market_id)?;
        let node = &self.node();

        let positions_data = node
            .get_market_user_positions(&address, &market_id_struct)
            .map_err(custom_err)?;

        let mut positions = Vec::new();

        if let Ok(Some(market)) = node.get_market_by_id(&market_id_struct) {
            let current_prices = market.get_current_prices();

            for (outcome_index, position_data) in positions_data {
                let outcome_price = current_prices
                    .get(outcome_index as usize)
                    .copied()
                    .unwrap_or(0.0);
                let shares_held = position_data;
                let current_value = shares_held * outcome_price;

                let outcome_name = if let Some(combo) =
                    market.state_combos.get(outcome_index as usize)
                {
                    format!("Outcome {}: {:?}", outcome_index, combo)
                } else {
                    format!("Outcome {}", outcome_index)
                };

                positions.push(truthcoin_dc_app_rpc_api::SharePosition {
                    market_id: market_id.clone(),
                    outcome_index: outcome_index as usize,
                    outcome_name,
                    shares_held: shares_held,
                    avg_purchase_price: outcome_price,
                    current_price: outcome_price,
                    current_value,
                    unrealized_pnl: 0.0,
                    cost_basis: current_value,
                });
            }
        }

        Ok(positions)
    }

    async fn calculate_share_cost(
        &self,
        market_id: String,
        outcome_index: usize,
        shares_amount: f64,
    ) -> RpcResult<u64> {
        let market_id_struct = parse_market_id(&market_id)?;
        let node = &self.node();

        if let Ok(Some(market)) = node.get_market_by_id(&market_id_struct) {
            let cost_sats = market
                .calculate_buy_cost(outcome_index as u32, shares_amount)
                .map_err(custom_err)?;
            Ok(cost_sats)
        } else {
            Err(custom_err_msg("Market not found"))
        }
    }

    async fn buy_shares(
        &self,
        market_id: String,
        outcome_index: usize,
        shares_amount: f64,
        max_cost: u64,
        fee_sats: u64,
    ) -> RpcResult<String> {
        let calculated_cost = self
            .calculate_share_cost(
                market_id.clone(),
                outcome_index,
                shares_amount,
            )
            .await?;

        if calculated_cost > max_cost {
            return Err(custom_err_msg(format!(
                "Share cost {} exceeds maximum cost {} (slippage protection)",
                calculated_cost, max_cost
            )));
        }

        let market_id_struct = parse_market_id(&market_id)?;

        let tx = self
            .app
            .wallet
            .buy_shares(
                market_id_struct,
                outcome_index,
                shares_amount,
                max_cost,
                bitcoin::Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)?;

        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{}", txid))
    }

    async fn bitcoin_balance(&self) -> RpcResult<Balance> {
        self.app.wallet.get_bitcoin_balance().map_err(custom_err)
    }

    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid> {
        let tx = self
            .app
            .wallet
            .create_transfer(
                address,
                bitcoin::Amount::from_sat(value_sats),
                bitcoin::Amount::from_sat(fee_sats),
                None,
            )
            .map_err(custom_err)?;

        let txid = tx.txid();
        self.app.sign_and_send(tx).map_err(custom_err)?;

        let bitcoin_txid = bitcoin::Txid::from_raw_hash(
            bitcoin::hashes::Hash::from_byte_array(txid.0),
        );
        Ok(bitcoin_txid)
    }

    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        self.node().connect_peer(addr).map_err(custom_err)
    }

    async fn decrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        ciphertext: String,
    ) -> RpcResult<String> {
        let ciphertext_bytes = hex::decode(&ciphertext).map_err(|e| {
            ErrorObject::owned(
                -32602,
                "Invalid hex string",
                Some(e.to_string()),
            )
        })?;

        let decrypted_bytes = self
            .app
            .wallet
            .decrypt_msg(&encryption_pubkey, &ciphertext_bytes)
            .map_err(custom_err)?;

        Ok(hex::encode(decrypted_bytes))
    }

    async fn encrypt_msg(
        &self,
        _encryption_pubkey: EncryptionPubKey,
        _msg: String,
    ) -> RpcResult<String> {
        Err(ErrorObject::owned(
            -32601,
            "Encryption not implemented",
            Some("Use external encryption tools"),
        ))
    }

    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String> {
        Ok(format!("{}", address))
    }

    async fn generate_mnemonic(&self) -> RpcResult<String> {
        let mnemonic = bip39::Mnemonic::new(
            bip39::MnemonicType::Words12,
            bip39::Language::English,
        );
        Ok(mnemonic.to_string())
    }

    async fn register_voter(
        &self,
        request: RegisterVoterRequest,
    ) -> RpcResult<String> {
        let fee = bitcoin::Amount::from_sat(request.fee_sats);

        let tx = self.app.wallet.register_voter(fee).map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{}", txid))
    }

    async fn submit_vote(
        &self,
        request: SubmitVoteRequest,
    ) -> RpcResult<String> {
        use truthcoin_dc::validation::SlotValidator;

        // Parse decision ID from hex string
        let slot_id =
            SlotValidator::parse_slot_id_from_hex(&request.decision_id)
                .map_err(|e| {
                    custom_err_msg(format!("Invalid decision ID: {}", e))
                })?;

        let fee = bitcoin::Amount::from_sat(request.fee_sats);

        let tx = self
            .app
            .wallet
            .submit_vote(
                slot_id.as_bytes(),
                request.vote_value,
                request.period_id,
                fee,
            )
            .map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{}", txid))
    }

    async fn submit_vote_batch(
        &self,
        request: SubmitVoteBatchRequest,
    ) -> RpcResult<String> {
        use truthcoin_dc::types::VoteBatchItem;
        use truthcoin_dc::validation::SlotValidator;

        if request.votes.is_empty() {
            return Err(custom_err_msg("Batch cannot be empty"));
        }

        // Parse and convert all vote items
        let mut batch_items = Vec::new();
        for vote in request.votes {
            let slot_id =
                SlotValidator::parse_slot_id_from_hex(&vote.decision_id)
                    .map_err(|e| {
                        custom_err_msg(format!("Invalid decision ID: {}", e))
                    })?;

            batch_items.push(VoteBatchItem {
                slot_id_bytes: slot_id.as_bytes(),
                vote_value: vote.vote_value,
            });
        }

        let fee = bitcoin::Amount::from_sat(request.fee_sats);

        let tx = self
            .app
            .wallet
            .submit_vote_batch(batch_items, request.period_id, fee)
            .map_err(custom_err)?;

        let txid = tx.txid();

        self.app.sign_and_send(tx).map_err(custom_err)?;

        Ok(format!("{}", txid))
    }

    async fn get_voter_info(
        &self,
        address: Address,
    ) -> RpcResult<Option<VoterInfo>> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Get reputation
        let reputation_opt = self
            .app
            .node
            .voting_state()
            .databases()
            .get_voter_reputation(&rotxn, address)
            .map_err(custom_err)?;

        let Some(reputation) = reputation_opt else {
            return Ok(None);
        };

        // Get vote count
        let votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_by_voter(&rotxn, address)
            .map_err(custom_err)?;

        Ok(Some(VoterInfo {
            address: address.to_string(),
            reputation: reputation.reputation,
            total_votes: votes.len() as u64,
            periods_active: reputation.total_decisions as u32,
            accuracy_score: reputation.get_accuracy_rate(),
            registered_at_height: 0,
            is_active: reputation.total_decisions > 0,
        }))
    }

    async fn get_voting_period_details(
        &self,
        period_id: u32,
    ) -> RpcResult<Option<VotingPeriodDetails>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let rotxn = self.node().read_txn().map_err(custom_err)?;

        let voting_period_id = VotingPeriodId::new(period_id);

        let current_timestamp =
            self.node().get_last_block_timestamp().map_err(custom_err)?;

        let config = self.node().get_slot_config();
        let slots_db = self.node().get_slots_db();

        let has_consensus = self
            .app
            .node
            .voting_state()
            .databases()
            .has_consensus(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let period = match truthcoin_dc::state::voting::period_calculator::calculate_voting_period(
            &rotxn,
            voting_period_id,
            current_timestamp,
            config,
            slots_db,
            has_consensus,
        ) {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        let (total_voters, total_votes, _) = self
            .app
            .node
            .voting_state()
            .get_participation_stats(&rotxn, voting_period_id, config, slots_db)
            .map_err(custom_err)?;

        let decision_slots: Vec<String> = period
            .decision_slots
            .iter()
            .map(|slot_id| slot_id.to_hex())
            .collect();

        let votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_for_period(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let active_voters: std::collections::HashSet<_> =
            votes.keys().map(|k| k.voter_address).collect();

        let consensus_reached = period.status == truthcoin_dc::state::voting::types::VotingPeriodStatus::Resolved
            || period.status == truthcoin_dc::state::voting::types::VotingPeriodStatus::Closed;

        Ok(Some(VotingPeriodDetails {
            period_id,
            start_time: period.start_timestamp,
            end_time: period.end_timestamp,
            status: format!("{:?}", period.status),
            decision_slots,
            created_at_height: 0, // Deprecated field, periods are calculated not stored
            total_voters,
            active_voters: active_voters.len() as u64,
            total_votes,
            consensus_reached,
        }))
    }

    async fn get_voter_votes(
        &self,
        address: Address,
        period_id: Option<u32>,
    ) -> RpcResult<Vec<VoteInfo>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Get all votes by this voter
        let all_votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_by_voter(&rotxn, address)
            .map_err(custom_err)?;

        let mut vote_infos = Vec::new();

        for (vote_key, vote_entry) in all_votes {
            if let Some(pid) = period_id {
                if vote_key.period_id != VotingPeriodId::new(pid) {
                    continue;
                }
            }

            vote_infos.push(VoteInfo {
                voter_address: address.to_string(),
                decision_id: vote_key.decision_id.to_hex(),
                vote_value: vote_entry.to_f64(),
                period_id: vote_key.period_id.as_u32(),
                block_height: vote_entry.block_height,
                txid: String::from(""),
                is_batch_vote: false,
            });
        }

        Ok(vote_infos)
    }

    async fn get_decision_votes(
        &self,
        decision_id: String,
    ) -> RpcResult<Vec<VoteInfo>> {
        use truthcoin_dc::validation::SlotValidator;

        let rotxn = self.node().read_txn().map_err(custom_err)?;

        let slot_id = SlotValidator::parse_slot_id_from_hex(&decision_id)
            .map_err(|e| {
                custom_err_msg(format!("Invalid decision ID: {}", e))
            })?;

        let votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_for_decision(&rotxn, slot_id)
            .map_err(custom_err)?;

        let mut vote_infos = Vec::new();

        for (vote_key, vote_entry) in votes {
            vote_infos.push(VoteInfo {
                voter_address: vote_key.voter_address.to_string(),
                decision_id: decision_id.clone(),
                vote_value: vote_entry.to_f64(),
                period_id: vote_key.period_id.as_u32(),
                block_height: vote_entry.block_height,
                txid: String::from(""),
                is_batch_vote: false,
            });
        }

        Ok(vote_infos)
    }

    async fn get_voter_participation(
        &self,
        address: Address,
        period_id: u32,
    ) -> RpcResult<Option<VoterParticipation>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let rotxn = self.node().read_txn().map_err(custom_err)?;
        let voting_period_id = VotingPeriodId::new(period_id);

        let current_timestamp =
            self.node().get_last_block_timestamp().map_err(custom_err)?;

        let config = self.node().get_slot_config();
        let slots_db = self.node().get_slots_db();

        let has_consensus = self
            .app
            .node
            .voting_state()
            .databases()
            .has_consensus(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let period = match truthcoin_dc::state::voting::period_calculator::calculate_voting_period(
            &rotxn,
            voting_period_id,
            current_timestamp,
            config,
            slots_db,
            has_consensus,
        ) {
            Ok(p) => p,
            Err(_) => return Ok(None),
        };

        let all_votes = self
            .app
            .node
            .voting_state()
            .databases()
            .get_votes_by_voter(&rotxn, address)
            .map_err(custom_err)?;

        let period_votes: Vec<_> = all_votes
            .into_iter()
            .filter(|(key, _)| key.period_id == voting_period_id)
            .collect();

        let votes_cast = period_votes.len() as u32;
        let decisions_available = period.decision_slots.len() as u32;

        let participation_rate = if decisions_available > 0 {
            votes_cast as f64 / decisions_available as f64
        } else {
            0.0
        };

        let reputation_opt = self
            .app
            .node
            .voting_state()
            .databases()
            .get_voter_reputation(&rotxn, address)
            .map_err(custom_err)?;

        let participated_in_consensus = reputation_opt
            .map(|rep| rep.last_period == voting_period_id)
            .unwrap_or(false);

        Ok(Some(VoterParticipation {
            address: address.to_string(),
            period_id,
            votes_cast,
            decisions_available,
            participation_rate,
            participated_in_consensus,
        }))
    }

    async fn list_voters(&self) -> RpcResult<Vec<VoterInfo>> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Get all voters
        let all_voters = self
            .app
            .node
            .voting_state()
            .databases()
            .get_all_voters(&rotxn)
            .map_err(custom_err)?;

        let mut voter_infos = Vec::new();

        for voter_address in all_voters {
            // Get reputation
            let reputation_opt = self
                .app
                .node
                .voting_state()
                .databases()
                .get_voter_reputation(&rotxn, voter_address)
                .map_err(custom_err)?;

            let Some(reputation) = reputation_opt else {
                continue;
            };

            // Get vote count
            let votes = self
                .app
                .node
                .voting_state()
                .databases()
                .get_votes_by_voter(&rotxn, voter_address)
                .map_err(custom_err)?;

            voter_infos.push(VoterInfo {
                address: voter_address.to_string(),
                reputation: reputation.reputation,
                total_votes: votes.len() as u64,
                periods_active: reputation.total_decisions as u32,
                accuracy_score: reputation.get_accuracy_rate(),
                registered_at_height: 0,
                is_active: reputation.total_decisions > 0,
            });
        }

        Ok(voter_infos)
    }

    async fn is_registered_voter(&self, address: Address) -> RpcResult<bool> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Check if voter has reputation (indicates registration)
        let reputation_opt = self
            .app
            .node
            .voting_state()
            .databases()
            .get_voter_reputation(&rotxn, address)
            .map_err(custom_err)?;

        Ok(reputation_opt.is_some())
    }

    async fn get_votecoin_balance(&self, address: Address) -> RpcResult<u32> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        // Get Votecoin balance directly
        let votecoin_balance = self
            .app
            .node
            .get_votecoin_balance_for(&rotxn, &address)
            .map_err(custom_err)?;

        Ok(votecoin_balance)
    }

    async fn get_current_voting_stats(
        &self,
    ) -> RpcResult<Option<VotingPeriodDetails>> {
        let period_id = {
            let rotxn = self.node().read_txn().map_err(custom_err)?;

            let current_timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    custom_err_msg(format!("System time error: {}", e))
                })?
                .as_secs();

            let config = self.node().get_slot_config();
            let slots_db = self.node().get_slots_db();

            let active_period_opt = self
                .app
                .node
                .voting_state()
                .get_active_period(&rotxn, current_timestamp, config, slots_db)
                .map_err(custom_err)?;

            let Some(period) = active_period_opt else {
                return Ok(None);
            };

            period.id.as_u32()
        };

        self.get_voting_period_details(period_id).await
    }

    async fn refresh_wallet(&self) -> RpcResult<()> {
        self.app.update().map_err(custom_err)
    }

    async fn get_voting_consensus_results(
        &self,
        period_id: u32,
    ) -> RpcResult<truthcoin_dc_app_rpc_api::VotingConsensusResults> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let voting_period_id = VotingPeriodId(period_id);

        let outcomes = self
            .node()
            .resolve_voting_period(voting_period_id)
            .map_err(|e| {
                tracing::error!(
                    "Failed to resolve voting period {}: {:?}",
                    period_id,
                    e
                );
                custom_err(e)
            })?;

        if outcomes.is_empty() {
            return Err(custom_err_msg(format!(
                "No votes found for period {}",
                period_id
            )));
        }

        let rotxn = self.node().read_txn().map_err(custom_err)?;
        let outcomes_map = self
            .node()
            .voting_state()
            .databases()
            .get_consensus_outcomes_for_period(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let mut outcomes = std::collections::HashMap::new();
        for (slot_id, outcome) in outcomes_map {
            outcomes.insert(slot_id.to_hex(), outcome);
        }

        let votes = self
            .node()
            .voting_state()
            .databases()
            .get_votes_for_period(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let mut unique_voters = std::collections::HashSet::new();
        let mut unique_decisions = std::collections::HashSet::new();
        for (key, _) in &votes {
            unique_voters.insert(key.voter_address);
            unique_decisions.insert(key.decision_id);
        }

        let period_stats = self
            .node()
            .voting_state()
            .databases()
            .get_period_stats(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let mut reputation_updates = std::collections::HashMap::new();

        let (first_loading, explained_variance, certainty) =
            if let Some(ref stats) = period_stats {
                if let Some(ref rep_changes) = stats.reputation_changes {
                    for (voter_id, (old_rep, new_rep)) in rep_changes {
                        if let Some(reputation) = self
                            .node()
                            .voting_state()
                            .databases()
                            .get_voter_reputation(&rotxn, *voter_id)
                            .map_err(custom_err)?
                        {
                            let rep_update =
                                truthcoin_dc_app_rpc_api::ReputationUpdate {
                                    old_reputation: *old_rep,
                                    new_reputation: *new_rep,
                                    votecoin_proportion: reputation
                                        .votecoin_proportion,
                                    compliance_score: 0.0,
                                };
                            reputation_updates
                                .insert(voter_id.to_string(), rep_update);
                        }
                    }
                }

                (
                    stats
                        .first_loading
                        .clone()
                        .unwrap_or_else(|| vec![0.0; unique_voters.len()]),
                    stats.explained_variance.unwrap_or(0.0),
                    stats.certainty.unwrap_or(0.0),
                )
            } else {
                (vec![0.0; unique_voters.len()], 0.0, 0.0)
            };

        if reputation_updates.is_empty() {
            for voter_id in &unique_voters {
                if let Some(reputation) = self
                    .node()
                    .voting_state()
                    .databases()
                    .get_voter_reputation(&rotxn, *voter_id)
                    .map_err(custom_err)?
                {
                    let rep_update =
                        truthcoin_dc_app_rpc_api::ReputationUpdate {
                            old_reputation: reputation.reputation,
                            new_reputation: reputation.reputation,
                            votecoin_proportion: reputation.votecoin_proportion,
                            compliance_score: 0.0,
                        };
                    reputation_updates.insert(voter_id.to_string(), rep_update);
                }
            }
        }

        Ok(truthcoin_dc_app_rpc_api::VotingConsensusResults {
            period_id,
            status: "Resolved".to_string(),
            outcomes,
            first_loading,
            explained_variance,
            certainty,
            reputation_updates,
            outliers: Vec::new(),
            vote_matrix_dimensions: (
                unique_voters.len(),
                unique_decisions.len(),
            ),
            algorithm_version: "SVD-PCA-v1.0".to_string(),
        })
    }

    async fn get_voter_reputation(
        &self,
        voter_address: String,
    ) -> RpcResult<f64> {
        let rotxn = self.node().read_txn().map_err(custom_err)?;

        let address: truthcoin_dc::types::Address =
            voter_address.parse().map_err(|e| {
                custom_err(anyhow::anyhow!("Invalid voter address: {}", e))
            })?;

        let reputation = self
            .node()
            .voting_state()
            .databases()
            .get_voter_reputation(&rotxn, address)
            .map_err(custom_err)?
            .ok_or_else(|| custom_err(anyhow::anyhow!("Voter not found")))?;

        Ok(reputation.reputation)
    }

    async fn get_voting_period_status(
        &self,
        period_id: u32,
    ) -> RpcResult<String> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let rotxn = self.node().read_txn().map_err(custom_err)?;
        let voting_period_id = VotingPeriodId(period_id);

        if self
            .node()
            .voting_state()
            .databases()
            .get_consensus_outcomes_for_period(&rotxn, voting_period_id)
            .map_err(custom_err)?
            .len()
            > 0
        {
            return Ok("Resolved".to_string());
        }

        let votes = self
            .node()
            .voting_state()
            .databases()
            .get_votes_for_period(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        if !votes.is_empty() {
            return Ok("Active".to_string());
        }

        let slots = self
            .node()
            .get_claimed_slots_in_period(voting_period_id)
            .map_err(custom_err)?;

        if !slots.is_empty() {
            return Ok("Pending".to_string());
        }

        Ok("NotFound".to_string())
    }

    async fn get_redistribution_summary(
        &self,
        period_id: u32,
    ) -> RpcResult<Option<truthcoin_dc_app_rpc_api::RedistributionInfo>> {
        use truthcoin_dc::state::voting::types::VotingPeriodId;

        let rotxn = self.node().read_txn().map_err(custom_err)?;
        let voting_period_id = VotingPeriodId(period_id);

        // Get the redistribution data for this period
        let period_redist = self
            .node()
            .voting_state()
            .databases()
            .get_pending_redistribution(&rotxn, voting_period_id)
            .map_err(custom_err)?;

        let Some(redist) = period_redist else {
            return Ok(None);
        };

        // Convert slot IDs to hex strings
        let slots_affected: Vec<String> = redist
            .slots_pending_redistribution
            .iter()
            .map(|slot_id| slot_id.to_hex())
            .collect();

        // Build the response
        let info = truthcoin_dc_app_rpc_api::RedistributionInfo {
            period_id: period_id,
            total_redistributed: redist
                .redistribution_summary
                .total_redistributed,
            winners_count: redist.redistribution_summary.winners_count,
            losers_count: redist.redistribution_summary.losers_count,
            unchanged_count: redist.redistribution_summary.unchanged_count,
            conservation_check: redist
                .redistribution_summary
                .conservation_check,
            block_height: redist.calculated_at_height,
            is_applied: redist.applied,
            slots_affected,
        };

        Ok(Some(info))
    }
}

#[derive(Clone, Debug)]
struct RequestIdMaker;

impl MakeRequestId for RequestIdMaker {
    fn make_request_id<B>(
        &mut self,
        _: &http::Request<B>,
    ) -> Option<RequestId> {
        use uuid::Uuid;
        let id = Uuid::new_v4();
        let id = id.as_simple();
        let id = format!("req_{id}");

        let Ok(header_value) = http::HeaderValue::from_str(&id) else {
            return None;
        };

        Some(RequestId::new(header_value))
    }
}

pub async fn run_server(
    app: App,
    rpc_url: url::Url,
) -> anyhow::Result<SocketAddr> {
    const REQUEST_ID_HEADER: &str = "x-request-id";

    let tracer = tower::ServiceBuilder::new()
        .layer(SetRequestIdLayer::new(
            http::HeaderName::from_static(REQUEST_ID_HEADER),
            RequestIdMaker,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(move |request: &http::Request<_>| {
                    let request_id = request
                        .headers()
                        .get(http::HeaderName::from_static(REQUEST_ID_HEADER))
                        .and_then(|h| h.to_str().ok())
                        .filter(|s| !s.is_empty());

                    tracing::span!(
                        tracing::Level::DEBUG,
                        "request",
                        method = %request.method(),
                        uri = %request.uri(),
                        request_id,
                    )
                })
                .on_request(())
                .on_eos(())
                .on_response(
                    DefaultOnResponse::new().level(tracing::Level::INFO),
                )
                .on_failure(
                    DefaultOnFailure::new().level(tracing::Level::ERROR),
                ),
        )
        .layer(PropagateRequestIdLayer::new(http::HeaderName::from_static(
            REQUEST_ID_HEADER,
        )))
        .into_inner();

    let http_middleware = tower::ServiceBuilder::new().layer(tracer);
    let rpc_middleware = RpcServiceBuilder::new().rpc_logger(1024);

    let server = Server::builder()
        .set_http_middleware(http_middleware)
        .set_rpc_middleware(rpc_middleware)
        .build(rpc_url.socket_addrs(|| None)?.as_slice())
        .await?;

    let addr = server.local_addr()?;
    let handle = server.start(RpcServerImpl { app }.into_rpc());

    tokio::spawn(handle.stopped());

    Ok(addr)
}
