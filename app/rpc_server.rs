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
    types::{
        Address, Authorization, Block, BlockHash, EncryptionPubKey,
        FilledOutputContent, PointedOutput, Transaction, Txid, VerifyingKey,
        WithdrawalBundle,
    },
    validation::{PeriodCalculator, SlotValidator},
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

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn get_block(&self, block_hash: BlockHash) -> RpcResult<Block> {
        let block = self
            .app
            .node
            .get_block(block_hash)
            .expect("This error should have been handled properly.");
        Ok(block)
    }

    async fn get_best_sidechain_block_hash(
        &self,
    ) -> RpcResult<Option<BlockHash>> {
        self.app.node.try_get_tip().map_err(custom_err)
    }

    async fn get_best_mainchain_block_hash(
        &self,
    ) -> RpcResult<Option<bitcoin::BlockHash>> {
        let Some(sidechain_hash) =
            self.app.node.try_get_tip().map_err(custom_err)?
        else {
            return Ok(None);
        };
        let block_hash = self
            .app
            .node
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
        self.app.node.try_get_transaction(txid).map_err(custom_err)
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
        let height = self.app.node.try_get_tip_height().map_err(custom_err)?;
        let block_count = height.map_or(0, |height| height + 1);
        Ok(block_count)
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
        let peers = self.app.node.get_active_peers();
        Ok(peers)
    }

    async fn list_utxos(
        &self,
    ) -> RpcResult<Vec<PointedOutput<FilledOutputContent>>> {
        let utxos = self.app.node.get_all_utxos().map_err(custom_err)?;
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
        self.app.node.remove_from_mempool(txid).map_err(custom_err)
    }

    async fn set_seed_from_mnemonic(&self, mnemonic: String) -> RpcResult<()> {
        self.app
            .wallet
            .set_seed_from_mnemonic(mnemonic.as_str())
            .map_err(custom_err)
    }

    async fn sidechain_wealth_sats(&self) -> RpcResult<u64> {
        let sidechain_wealth =
            self.app.node.get_sidechain_wealth().map_err(custom_err)?;
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
        let slots =
            self.app.node.get_all_slot_quarters().map_err(custom_err)?;
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
        let is_testing_mode = self.app.node.is_slots_testing_mode();
        let blocks_per_period = if is_testing_mode {
            self.app.node.get_slots_testing_config()
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
                self.app.node.get_slots_testing_config();
            PeriodCalculator::block_height_to_testing_period(
                tip_height,
                testing_blocks_per_period,
            )
        } else {
            PeriodCalculator::timestamp_to_period(current_timestamp)
        };

        let current_period_name =
            PeriodCalculator::period_to_name(current_period);

        Ok(truthcoin_dc_app_rpc_api::SlotStatus {
            is_testing_mode,
            blocks_per_period,
            current_period,
            current_period_name,
        })
    }

    async fn timestamp_to_quarter(&self, timestamp: u64) -> RpcResult<u32> {
        Ok(PeriodCalculator::timestamp_to_period(timestamp))
    }

    async fn quarter_to_string(&self, quarter: u32) -> RpcResult<String> {
        Ok(PeriodCalculator::period_to_name(quarter))
    }

    async fn block_height_to_testing_period(
        &self,
        block_height: u32,
    ) -> RpcResult<u32> {
        let testing_blocks_per_period =
            self.app.node.get_slots_testing_config();
        Ok(PeriodCalculator::block_height_to_testing_period(
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
        let mut tx = Transaction::default();
        let fee = Amount::from_sat(fee_sats);
        self.app
            .wallet
            .claim_decision_slot(
                &mut tx,
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
        let available_slots = self
            .app
            .node
            .get_available_slots_in_period(period_index)
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
        let slot_opt = self.app.node.get_slot(slot_id).map_err(custom_err)?;

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
        let claimed_slots_data = self
            .app
            .node
            .get_claimed_slots_in_period(period_index)
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
            self.app.node.get_voting_periods().map_err(custom_err)?;

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
            self.app.node.get_ossified_slots().map_err(custom_err)?;

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
        let markets = self
            .app
            .node
            .get_markets_by_state(truthcoin_dc::state::MarketState::Trading)
            .map_err(custom_err)?;

        let market_summaries = markets
            .into_iter()
            .map(|market| {
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
                    state: format!("{:?}", market.state()),
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

        let market = match self
            .app
            .node
            .get_market_by_id(&market_id_struct)
            .map_err(custom_err)?
        {
            Some(market) => market,
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

        let market_data = truthcoin_dc_app_rpc_api::MarketData {
            market_id,
            title: market.title.clone(),
            description: market.description.clone(),
            outcomes,
            state: format!("{:?}", market.state()),
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
                    if let Some(ref dims) = request.dimensions {
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
        let node = &self.app.node;

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
        let node = &self.app.node;

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
        let node = &self.app.node;

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
        self.app.node.connect_peer(addr).map_err(custom_err)
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
        // For now, return an error indicating this feature is not implemented
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
        _request: RegisterVoterRequest,
    ) -> RpcResult<String> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn submit_vote(
        &self,
        _request: SubmitVoteRequest,
    ) -> RpcResult<String> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn submit_vote_batch(
        &self,
        _request: SubmitVoteBatchRequest,
    ) -> RpcResult<String> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_voter_info(
        &self,
        _address: Address,
    ) -> RpcResult<Option<VoterInfo>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_voting_period_details(
        &self,
        _period_id: u32,
    ) -> RpcResult<Option<VotingPeriodDetails>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_voter_votes(
        &self,
        _address: Address,
        _period_id: Option<u32>,
    ) -> RpcResult<Vec<VoteInfo>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_decision_votes(
        &self,
        _decision_id: String,
    ) -> RpcResult<Vec<VoteInfo>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_voter_participation(
        &self,
        _address: Address,
        _period_id: u32,
    ) -> RpcResult<Option<VoterParticipation>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn list_voters(&self) -> RpcResult<Vec<VoterInfo>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn is_registered_voter(&self, _address: Address) -> RpcResult<bool> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_voting_power(&self, _address: Address) -> RpcResult<u32> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
    }

    async fn get_current_voting_stats(
        &self,
    ) -> RpcResult<Option<VotingPeriodDetails>> {
        Err(ErrorObject::owned(
            -32601,
            "Not implemented",
            Some("Voting system not yet implemented"),
        ))
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
        // the 'simple' format renders the UUID with no dashes, which
        // makes for easier copy/pasting.
        let id = Uuid::new_v4();
        let id = id.as_simple();
        let id = format!("req_{id}"); // prefix all IDs with "req_", to make them easier to identify

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

    // Ordering here matters! Order here is from official docs on request IDs tracings
    // https://docs.rs/tower-http/latest/tower_http/request_id/index.html#using-trace
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
                        request_id , // this is needed for the record call below to work
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

    // In this example we don't care about doing shutdown so let's it run forever.
    // You may use the `ServerHandle` to shut it down or manage it yourself.
    tokio::spawn(handle.stopped());

    Ok(addr)
}
