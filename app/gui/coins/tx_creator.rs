use eframe::egui::{self, InnerResponse, Response, TextBuffer};

use truthcoin_dc::types::{Transaction, Txid};

use crate::{app::App, gui::util::InnerResponseExt};

#[derive(
    Clone, Debug, Default, strum::Display, strum::EnumIter, Eq, PartialEq,
)]
pub enum TxType {
    #[default]
    Regular,
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
    // set tx data for the current transaction
    fn set_tx_data(
        &self,
        _app: &App,
        tx: Transaction,
    ) -> anyhow::Result<Transaction> {
        match &self.tx_type {
            TxType::Regular => Ok(tx),
        }
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
        };
        let tx_data_changed =
            tx_data_ui.is_some_and(|resp: Response| resp.changed());
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
