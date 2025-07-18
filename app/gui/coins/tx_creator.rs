use std::str::FromStr;

use eframe::egui::{self, InnerResponse, Response, TextBuffer};

use truthcoin_dc::{
    state::AmmPair,
    types::{AssetId, Transaction, Txid},
};

use crate::{
    app::App,
    gui::util::{InnerResponseExt, borsh_deserialize_hex},
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DexBurn {
    asset0: String,
    asset1: String,
    amount_lp_tokens: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DexMint {
    asset0: String,
    asset1: String,
    amount0: String,
    amount1: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DexSwap {
    asset_spend: String,
    asset_receive: String,
    amount_spend: String,
    amount_receive: String,
}

#[derive(
    Clone, Debug, Default, strum::Display, strum::EnumIter, Eq, PartialEq,
)]
pub enum TxType {
    #[default]
    Regular,
    #[strum(to_string = "DEX (Burn Position)")]
    DexBurn(DexBurn),
    #[strum(to_string = "DEX (Mint Position)")]
    DexMint(DexMint),
    #[strum(to_string = "DEX (Swap)")]
    DexSwap(DexSwap),
}

#[derive(Debug, Default)]
pub struct TxCreator {
    pub bitcoin_value_in: bitcoin::Amount,
    pub bitcoin_value_out: bitcoin::Amount,
    pub tx_type: TxType,
    // if the base tx has changed, need to recompute final tx
    base_txid: Txid,
    final_tx: Option<anyhow::Result<Transaction>>,
}

fn show_monospace_single_line_input(
    ui: &mut egui::Ui,
    text_buffer: &mut dyn TextBuffer,
    descriptor: &str,
) -> InnerResponse<Response> {
    ui.horizontal(|ui| {
        ui.monospace(format!("{descriptor}:       "))
            | ui.add(egui::TextEdit::singleline(text_buffer))
    })
}

fn show_monospace_single_line_inputs<'iter, I>(
    ui: &mut egui::Ui,
    iter: I,
) -> Option<Response>
where
    I: IntoIterator<Item = (&'iter mut dyn TextBuffer, &'iter str)>,
{
    iter.into_iter()
        .map(|(text_buffer, descriptor)| {
            show_monospace_single_line_input(ui, text_buffer, descriptor).join()
        })
        .reduce(|resp0, resp1| resp0 | resp1)
}

impl TxCreator {
    fn set_dex_burn(
        app: &App,
        mut tx: Transaction,
        dex_burn: &DexBurn,
    ) -> anyhow::Result<Transaction> {
        let asset0: AssetId = borsh_deserialize_hex(&dex_burn.asset0)
            .map_err(|err| anyhow::anyhow!("Failed to parse asset 0: {err}"))?;
        let asset1: AssetId = borsh_deserialize_hex(&dex_burn.asset1)
            .map_err(|err| anyhow::anyhow!("Failed to parse asset 1: {err}"))?;
        let amount_lp_tokens = u64::from_str(&dex_burn.amount_lp_tokens)
            .map_err(|err| {
                anyhow::anyhow!("Failed to parse LP token amount: {err}")
            })?;
        let amm_pair = AmmPair::new(asset0, asset1);
        let (amount0, amount1);
        {
            let amm_pool_state = app
                .node
                .get_amm_pool_state(amm_pair)
                .map_err(anyhow::Error::new)?;
            let next_amm_pool_state = amm_pool_state
                .burn(amount_lp_tokens)
                .map_err(anyhow::Error::new)?;
            amount0 = amm_pool_state.reserve0 - next_amm_pool_state.reserve0;
            amount1 = amm_pool_state.reserve1 - next_amm_pool_state.reserve1;
        };
        let () = app.wallet.amm_burn(
            &mut tx,
            amm_pair.asset0(),
            amm_pair.asset1(),
            amount0,
            amount1,
            amount_lp_tokens,
        )?;
        Ok(tx)
    }

    fn set_dex_mint(
        app: &App,
        mut tx: Transaction,
        dex_mint: &DexMint,
    ) -> anyhow::Result<Transaction> {
        let asset0: AssetId = borsh_deserialize_hex(&dex_mint.asset0)
            .map_err(|err| anyhow::anyhow!("Failed to parse asset 0: {err}"))?;
        let asset1: AssetId = borsh_deserialize_hex(&dex_mint.asset1)
            .map_err(|err| anyhow::anyhow!("Failed to parse asset 1: {err}"))?;
        let amount0 = u64::from_str(&dex_mint.amount0).map_err(|err| {
            anyhow::anyhow!("Failed to parse amount (asset 0): {err}")
        })?;
        let amount1 = u64::from_str(&dex_mint.amount1).map_err(|err| {
            anyhow::anyhow!("Failed to parse amount (asset 1): {err}")
        })?;
        let lp_token_mint = {
            let amm_pair = AmmPair::new(asset0, asset1);
            let amm_pool_state = app
                .node
                .get_amm_pool_state(amm_pair)
                .map_err(anyhow::Error::new)?;
            let next_amm_pool_state = amm_pool_state
                .mint(amount0, amount1)
                .map_err(anyhow::Error::new)?;
            next_amm_pool_state.outstanding_lp_tokens
                - amm_pool_state.outstanding_lp_tokens
        };
        let () = app.wallet.amm_mint(
            &mut tx,
            asset0,
            asset1,
            amount0,
            amount1,
            lp_token_mint,
        )?;
        Ok(tx)
    }

    fn set_dex_swap(
        app: &App,
        mut tx: Transaction,
        dex_swap: &DexSwap,
    ) -> anyhow::Result<Transaction> {
        let asset_spend: AssetId = borsh_deserialize_hex(&dex_swap.asset_spend)
            .map_err(|err| {
                anyhow::anyhow!("Failed to parse spend asset: {err}")
            })?;
        let asset_receive: AssetId =
            borsh_deserialize_hex(&dex_swap.asset_receive).map_err(|err| {
                anyhow::anyhow!("Failed to parse receive asset: {err}")
            })?;
        let amount_spend =
            u64::from_str(&dex_swap.amount_spend).map_err(|err| {
                anyhow::anyhow!("Failed to parse spend amount: {err}")
            })?;
        let amount_receive =
            u64::from_str(&dex_swap.amount_receive).map_err(|err| {
                anyhow::anyhow!("Failed to parse receive amount: {err}")
            })?;
        let () = app.wallet.amm_swap(
            &mut tx,
            asset_spend,
            asset_receive,
            amount_spend,
            amount_receive,
        )?;
        Ok(tx)
    }

    // set tx data for the current transaction
    fn set_tx_data(
        &self,
        app: &App,
        tx: Transaction,
    ) -> anyhow::Result<Transaction> {
        match &self.tx_type {
            TxType::Regular => Ok(tx),
            TxType::DexBurn(dex_burn) => Self::set_dex_burn(app, tx, dex_burn),
            TxType::DexMint(dex_mint) => Self::set_dex_mint(app, tx, dex_mint),
            TxType::DexSwap(dex_swap) => Self::set_dex_swap(app, tx, dex_swap),
        }
    }

    fn show_dex_burn(
        ui: &mut egui::Ui,
        dex_burn: &mut DexBurn,
    ) -> Option<Response> {
        show_monospace_single_line_inputs(
            ui,
            [
                (&mut dex_burn.asset0 as &mut dyn TextBuffer, "Asset 0"),
                (&mut dex_burn.asset1, "Asset 1"),
                (&mut dex_burn.amount_lp_tokens, "LP Token Amount"),
            ],
        )
    }

    fn show_dex_mint(
        ui: &mut egui::Ui,
        dex_mint: &mut DexMint,
    ) -> Option<Response> {
        show_monospace_single_line_inputs(
            ui,
            [
                (&mut dex_mint.asset0 as &mut dyn TextBuffer, "Asset 0"),
                (&mut dex_mint.asset1, "Asset 1"),
                (&mut dex_mint.amount0, "Amount (Asset 0)"),
                (&mut dex_mint.amount1, "Amount (Asset 1)"),
            ],
        )
    }

    fn show_dex_swap(
        ui: &mut egui::Ui,
        dex_swap: &mut DexSwap,
    ) -> Option<Response> {
        show_monospace_single_line_inputs(
            ui,
            [
                (
                    &mut dex_swap.asset_spend as &mut dyn TextBuffer,
                    "Spend Asset",
                ),
                (&mut dex_swap.asset_receive, "Receive Asset"),
                (&mut dex_swap.amount_spend, "Spend Amount"),
                (&mut dex_swap.amount_receive, "Receive Amount"),
            ],
        )
    }

    pub fn show(
        &mut self,
        app: Option<&App>,
        ui: &mut egui::Ui,
        base_tx: &mut Transaction,
    ) -> anyhow::Result<()> {
        let Some(app) = app else { return Ok(()) };
        let tx_type_dropdown = ui.horizontal(|ui| {
            let combobox = egui::ComboBox::from_id_salt("tx_type")
                .selected_text(format!("{}", self.tx_type))
                .show_ui(ui, |ui| {
                    use strum::IntoEnumIterator;
                    TxType::iter()
                        .map(|tx_type| {
                            let text = tx_type.to_string();
                            ui.selectable_value(
                                &mut self.tx_type,
                                tx_type,
                                text,
                            )
                        })
                        .reduce(|resp0, resp1| resp0 | resp1)
                        .unwrap()
                });
            combobox.join() | ui.heading("Transaction")
        });
        let tx_data_ui = match &mut self.tx_type {
            TxType::Regular => None,
            TxType::DexBurn(dex_burn) => Self::show_dex_burn(ui, dex_burn),
            TxType::DexMint(dex_mint) => Self::show_dex_mint(ui, dex_mint),
            TxType::DexSwap(dex_swap) => Self::show_dex_swap(ui, dex_swap),
        };
        let tx_data_changed = tx_data_ui.is_some_and(|resp| resp.changed());
        // if base txid has changed, store the new txid
        let base_txid = base_tx.txid();
        let base_txid_changed = base_txid != self.base_txid;
        if base_txid_changed {
            self.base_txid = base_txid;
        }
        // (re)compute final tx if:
        // * the tx type, tx data, or base txid has changed
        // * final tx not yet set
        let refresh_final_tx = tx_type_dropdown.join().changed()
            || tx_data_changed
            || base_txid_changed
            || self.final_tx.is_none();
        if refresh_final_tx {
            self.final_tx = Some(self.set_tx_data(app, base_tx.clone()));
        }
        let final_tx = match &self.final_tx {
            None => panic!("impossible! final tx should have been set"),
            Some(Ok(final_tx)) => final_tx,
            Some(Err(wallet_err)) => {
                ui.monospace(format!("{wallet_err}"));
                return Ok(());
            }
        };
        let txid = &format!("{}", final_tx.txid())[0..8];
        ui.monospace(format!("txid: {txid}"));
        if self.bitcoin_value_in >= self.bitcoin_value_out {
            let fee = self.bitcoin_value_in - self.bitcoin_value_out;
            ui.monospace(format!("fee(sats):  {}", fee.to_sat()));
            if ui.button("sign and send").clicked() {
                let () = app.sign_and_send(final_tx.clone())?;
                *base_tx = Transaction::default();
                self.final_tx = None;
            }
        } else {
            ui.label("Not Enough Value In");
        }
        Ok(())
    }
}
