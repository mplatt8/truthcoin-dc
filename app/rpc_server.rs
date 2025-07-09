use std::{borrow::Cow, cmp::Ordering, net::SocketAddr};

use bitcoin::Amount;
use fraction::Fraction;
use jsonrpsee::{
    core::{RpcResult, async_trait},
    server::{RpcServiceBuilder, Server},
    types::ErrorObject,
};

use truthcoin_dc::{
    authorization::{self, Dst, Signature},
    net::Peer,
    state::{self, AmmPair, AmmPoolState},
    types::{
        Address, AssetId, Authorization, Block,
        BlockHash, EncryptionPubKey,
        FilledOutputContent, PointedOutput, Transaction, Txid, VerifyingKey,
        WithdrawalBundle, keys::Ecies,
    },
    wallet::Balance,
};
use truthcoin_dc_app_rpc_api::{RpcServer, TxInfo};
use tower_http::{
    request_id::{
        MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
    },
    trace::{DefaultOnFailure, DefaultOnResponse, TraceLayer},
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

pub struct RpcServerImpl {
    app: App,
}

#[async_trait]
impl RpcServer for RpcServerImpl {
    async fn amm_burn(
        &self,
        asset0: AssetId,
        asset1: AssetId,
        lp_token_amount: u64,
    ) -> RpcResult<Txid> {
        let amm_pair = AmmPair::new(asset0, asset1);
        let amm_pool_state = self.get_amm_pool_state(asset0, asset1).await?;
        let next_amm_pool_state =
            amm_pool_state.burn(lp_token_amount).map_err(custom_err)?;
        let amount0 = amm_pool_state.reserve0 - next_amm_pool_state.reserve0;
        let amount1 = amm_pool_state.reserve1 - next_amm_pool_state.reserve1;
        let mut tx = Transaction::default();
        let () = self
            .app
            .wallet
            .amm_burn(
                &mut tx,
                amm_pair.asset0(),
                amm_pair.asset1(),
                amount0,
                amount1,
                lp_token_amount,
            )
            .map_err(custom_err)?;
        let txid = tx.txid();
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn amm_mint(
        &self,
        asset0: AssetId,
        asset1: AssetId,
        amount0: u64,
        amount1: u64,
    ) -> RpcResult<Txid> {
        let amm_pool_state = self.get_amm_pool_state(asset0, asset1).await?;
        let next_amm_pool_state =
            amm_pool_state.mint(amount0, amount1).map_err(custom_err)?;
        let lp_token_mint = next_amm_pool_state.outstanding_lp_tokens
            - amm_pool_state.outstanding_lp_tokens;
        let mut tx = Transaction::default();
        let () = self
            .app
            .wallet
            .amm_mint(&mut tx, asset0, asset1, amount0, amount1, lp_token_mint)
            .map_err(custom_err)?;
        let txid = tx.txid();
        let () = self.app.sign_and_send(tx).map_err(custom_err)?;
        Ok(txid)
    }

    async fn amm_swap(
        &self,
        asset_spend: AssetId,
        asset_receive: AssetId,
        amount_spend: u64,
    ) -> RpcResult<u64> {
        let pair = match asset_spend.cmp(&asset_receive) {
            Ordering::Less => (asset_spend, asset_receive),
            Ordering::Equal => {
                let err = state::error::Amm::InvalidSwap;
                return Err(custom_err(err));
            }
            Ordering::Greater => (asset_receive, asset_spend),
        };
        let amm_pool_state = self.get_amm_pool_state(pair.0, pair.1).await?;
        let amount_receive = (if asset_spend < asset_receive {
            amm_pool_state.swap_asset0_for_asset1(amount_spend).map(
                |new_amm_pool_state| {
                    new_amm_pool_state.reserve1 - amm_pool_state.reserve1
                },
            )
        } else {
            amm_pool_state.swap_asset1_for_asset0(amount_spend).map(
                |new_amm_pool_state| {
                    new_amm_pool_state.reserve0 - amm_pool_state.reserve0
                },
            )
        })
        .map_err(custom_err)?;
        let mut tx = Transaction::default();
        let () = self
            .app
            .wallet
            .amm_swap(
                &mut tx,
                asset_spend,
                asset_receive,
                amount_spend,
                amount_receive,
            )
            .map_err(custom_err)?;
        let authorized_tx =
            self.app.wallet.authorize(tx).map_err(custom_err)?;
        self.app
            .node
            .submit_transaction(authorized_tx)
            .map_err(custom_err)?;
        Ok(amount_receive)
    }

    async fn bitcoin_balance(&self) -> RpcResult<Balance> {
        self.app.wallet.get_bitcoin_balance().map_err(custom_err)
    }

    async fn connect_peer(&self, addr: SocketAddr) -> RpcResult<()> {
        self.app.node.connect_peer(addr).map_err(custom_err)
    }

    async fn create_deposit(
        &self,
        address: Address,
        value_sats: u64,
        fee_sats: u64,
    ) -> RpcResult<bitcoin::Txid> {
        let app = self.app.clone();
        tokio::task::spawn_blocking(move || {
            app.deposit(
                address,
                bitcoin::Amount::from_sat(value_sats),
                bitcoin::Amount::from_sat(fee_sats),
            )
            .map_err(custom_err)
        })
        .await
        .unwrap()
    }

    async fn decrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        msg: String,
    ) -> RpcResult<String> {
        let ciphertext = hex::decode(msg).map_err(custom_err)?;
        self.app
            .wallet
            .decrypt_msg(&encryption_pubkey, &ciphertext)
            .map(hex::encode)
            .map_err(custom_err)
    }

    async fn encrypt_msg(
        &self,
        encryption_pubkey: EncryptionPubKey,
        msg: String,
    ) -> RpcResult<String> {
        Ecies::new(encryption_pubkey.0)
            .encrypt(msg.as_bytes())
            .map(hex::encode)
            .map_err(|err| custom_err(anyhow::anyhow!("{err:?}")))
    }

    async fn format_deposit_address(
        &self,
        address: Address,
    ) -> RpcResult<String> {
        let deposit_address = address.format_for_deposit();
        Ok(deposit_address)
    }

    async fn generate_mnemonic(&self) -> RpcResult<String> {
        let mnemonic = bip39::Mnemonic::new(
            bip39::MnemonicType::Words12,
            bip39::Language::English,
        );
        Ok(mnemonic.to_string())
    }

    async fn get_amm_pool_state(
        &self,
        asset0: AssetId,
        asset1: AssetId,
    ) -> RpcResult<AmmPoolState> {
        let amm_pair = AmmPair::new(asset0, asset1);
        self.app
            .node
            .get_amm_pool_state(amm_pair)
            .map_err(custom_err)
    }

    async fn get_amm_price(
        &self,
        base: AssetId,
        quote: AssetId,
    ) -> RpcResult<Option<Fraction>> {
        self.app
            .node
            .try_get_amm_price(base, quote)
            .map_err(custom_err)
    }

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
            // No sidechain tip, so no best mainchain block hash.
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
            .get_latest_failed_withdrawal_bundle_height()
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
