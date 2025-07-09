use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use bitcoin::amount::CheckedSum as _;
use borsh::BorshSerialize;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use utoipa::{PartialSchema, ToSchema};

use crate::{
    authorization::Authorization,
    types::{
        AmountOverflowError, GetAddress,
        GetBitcoinValue,
        address::Address,
        hashes::{
            self, AssetId, M6id, MerkleRoot,
            Txid,
        },
        serde_hexstr_human_readable,
    },
};

mod output;
pub use output::{
    AssetContent as AssetOutputContent, AssetOutput,
    BitcoinContent as BitcoinOutputContent, BitcoinOutput,
    Content as OutputContent, FilledContent as FilledOutputContent,
    FilledOutput, Output, Pointed as PointedOutput, SpentOutput,
    WithdrawalContent as WithdrawalOutputContent,
};

fn borsh_serialize_bitcoin_outpoint<W>(
    block_hash: &bitcoin::OutPoint,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bitcoin::OutPoint { txid, vout } = block_hash;
    let txid_bytes: &[u8; 32] = txid.as_ref();
    borsh::BorshSerialize::serialize(&(txid_bytes, vout), writer)
}

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    ToSchema,
)]
pub enum OutPoint {
    // Created by transactions.
    Regular {
        txid: Txid,
        vout: u32,
    },
    // Created by block bodies.
    Coinbase {
        merkle_root: MerkleRoot,
        vout: u32,
    },
    // Created by mainchain deposits.
    #[schema(value_type = crate::types::schema::BitcoinOutPoint)]
    Deposit(
        #[borsh(serialize_with = "borsh_serialize_bitcoin_outpoint")]
        bitcoin::OutPoint,
    ),
}

impl std::fmt::Display for OutPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular { txid, vout } => write!(f, "regular {txid} {vout}"),
            Self::Coinbase { merkle_root, vout } => {
                write!(f, "coinbase {merkle_root} {vout}")
            }
            Self::Deposit(bitcoin::OutPoint { txid, vout }) => {
                write!(f, "deposit {txid} {vout}")
            }
        }
    }
}

/// Reference to a tx input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum InPoint {
    /// Transaction input
    Regular {
        txid: Txid,
        // index of the spend in the inputs to spend_tx
        vin: u32,
    },
    // Created by mainchain withdrawals
    Withdrawal {
        m6id: M6id,
    },
}

pub type TxInputs = Vec<OutPoint>;

pub type TxOutputs = Vec<Output>;



#[allow(clippy::enum_variant_names)]
#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(as = TxData)]
pub enum TransactionData {
    /// Burn an AMM position
    AmmBurn {
        /// Amount of the ordered Bitcoin to receive
        amount0: u64,
        /// Amount of the ordered Votecoin to receive
        amount1: u64,
        /// Amount of the LP token to burn
        lp_token_burn: u64,
    },
    /// Mint an AMM position
    AmmMint {
        /// Amount of the ordered Bitcoin to receive
        amount0: u64,
        /// Amount of the ordered Votecoin to receive
        amount1: u64,
        /// Amount of the LP token to receive
        lp_token_mint: u64,
    },
    /// AMM swap
    AmmSwap {
        /// Amount to spend
        amount_spent: u64,
        /// Amount to receive
        amount_receive: u64,
        /// Pair asset to swap for
        pair_asset: AssetId,
    },

}

pub type TxData = TransactionData;

impl TxData {
    /// `true` if the tx data corresponds to an AMM burn
    pub fn is_amm_burn(&self) -> bool {
        matches!(self, Self::AmmBurn { .. })
    }

    /// `true` if the tx data corresponds to an AMM mint
    pub fn is_amm_mint(&self) -> bool {
        matches!(self, Self::AmmMint { .. })
    }

    /// `true` if the tx data corresponds to an AMM swap
    pub fn is_amm_swap(&self) -> bool {
        matches!(self, Self::AmmSwap { .. })
    }
}

/// Struct describing an AMM burn
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmmBurn {
    pub asset0: AssetId,
    pub asset1: AssetId,
    /// Amount of asset 0 received
    pub amount0: u64,
    /// Amount of asset 1 received
    pub amount1: u64,
    /// Amount of LP token burned
    pub lp_token_burn: u64,
}

/// Struct describing an AMM mint
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmmMint {
    pub asset0: AssetId,
    pub asset1: AssetId,
    /// Amount of asset 0 deposited
    pub amount0: u64,
    /// Amount of asset 1 deposited
    pub amount1: u64,
    /// Amount of LP token received
    pub lp_token_mint: u64,
}

/// Struct describing an AMM swap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmmSwap {
    pub asset_spend: AssetId,
    pub asset_receive: AssetId,
    /// Amount of spend asset spent
    pub amount_spend: u64,
    //// Amount of receive asset received
    pub amount_receive: u64,
}



#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
pub struct Transaction {
    #[schema(schema_with = TxInputs::schema)]
    pub inputs: TxInputs,
    #[schema(schema_with = TxOutputs::schema)]
    pub outputs: TxOutputs,
    #[serde(with = "serde_hexstr_human_readable")]
    #[schema(value_type = String)]
    pub memo: Vec<u8>,
    pub data: Option<TransactionData>,
}

impl Transaction {
    pub fn new(inputs: TxInputs, outputs: TxOutputs) -> Self {
        Self {
            inputs,
            outputs,
            memo: Vec::new(),
            data: None,
        }
    }

    /// Return an iterator over asset outputs with index
    pub fn indexed_asset_outputs(
        &self,
    ) -> impl Iterator<Item = (usize, AssetOutput)> + '_ {
        self.outputs.iter().enumerate().filter_map(|(idx, output)| {
            let asset_output: AssetOutput =
                Option::<AssetOutput>::from(output.clone())?;
            Some((idx, asset_output))
        })
    }

    /// Return an iterator over Votecoin outputs
    pub fn votecoin_outputs(&self) -> impl Iterator<Item = &Output> {
        self.outputs.iter().filter(|output| output.is_votecoin())
    }

    /// `true` if the tx data corresponds to an AMM burn
    pub fn is_amm_burn(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_amm_burn(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to an AMM mint
    pub fn is_amm_mint(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_amm_mint(),
            None => false,
        }
    }

    /// `true` if the tx data corresponds to an AMM swap
    pub fn is_amm_swap(&self) -> bool {
        match &self.data {
            Some(tx_data) => tx_data.is_amm_swap(),
            None => false,
        }
    }



    /// `true` if the tx data corresponds to a regular tx
    pub fn is_regular(&self) -> bool {
        self.data.is_none()
    }

    pub fn txid(&self) -> Txid {
        hashes::hash(self).into()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<FilledOutput>,
}

impl FilledTransaction {
    // Return an iterator over Votecoin outputs
    pub fn votecoin_outputs(&self) -> impl Iterator<Item = &Output> {
        self.transaction.votecoin_outputs()
    }

    /// Accessor for tx data
    pub fn data(&self) -> &Option<TxData> {
        &self.transaction.data
    }

    /// Accessor for tx inputs
    pub fn inputs(&self) -> &TxInputs {
        &self.transaction.inputs
    }

    /// `true` if the tx data corresponds to an AMM burn
    pub fn is_amm_burn(&self) -> bool {
        self.transaction.is_amm_burn()
    }

    /// `true` if the tx data corresponds to an AMM mint
    pub fn is_amm_mint(&self) -> bool {
        self.transaction.is_amm_mint()
    }

    /// `true` if the tx data corresponds to an AMM swap
    pub fn is_amm_swap(&self) -> bool {
        self.transaction.is_amm_swap()
    }



    /// `true` if the tx data corresponds to a regular tx
    pub fn is_regular(&self) -> bool {
        self.transaction.is_regular()
    }

    /// Accessor for tx outputs
    pub fn outputs(&self) -> &TxOutputs {
        &self.transaction.outputs
    }

     /** If the tx is an AMM burn, returns the LP token's
     *  corresponding [`AmmBurn`]. */
    pub fn amm_burn(&self) -> Option<AmmBurn> {
        match self.transaction.data {
            Some(TransactionData::AmmBurn {
                amount0,
                amount1,
                lp_token_burn,
            }) => {
                let unique_spent_lp_tokens = self.unique_spent_lp_tokens();
                let (asset0, asset1, _) = unique_spent_lp_tokens.first()?;
                Some(AmmBurn {
                    asset0: *asset0,
                    asset1: *asset1,
                    amount0,
                    amount1,
                    lp_token_burn,
                })
            }
            _ => None,
        }
    }

    /// If the tx is an AMM mint, returns the corresponding [`AmmMint`].
    pub fn amm_mint(&self) -> Option<AmmMint> {
        match self.transaction.data {
            Some(TransactionData::AmmMint {
                amount0,
                amount1,
                lp_token_mint,
            }) => match self.unique_spent_assets().get(0..=1) {
                Some([(first_asset, _), (second_asset, _)]) => {
                    let mut assets = [first_asset, second_asset];
                    assets.sort();
                    let [asset0, asset1] = assets;
                    Some(AmmMint {
                        asset0: *asset0,
                        asset1: *asset1,
                        amount0,
                        amount1,
                        lp_token_mint,
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// If the tx is an AMM swap, returns the corresponding [`AmmSwap`].
    pub fn amm_swap(&self) -> Option<AmmSwap> {
        match self.transaction.data {
            Some(TransactionData::AmmSwap {
                amount_spent,
                amount_receive,
                pair_asset,
            }) => {
                let (spent_asset, _) = *self.unique_spent_assets().first()?;
                Some(AmmSwap {
                    asset_spend: spent_asset,
                    asset_receive: pair_asset,
                    amount_spend: amount_spent,
                    amount_receive,
                })
            }
            _ => None,
        }
    }



    /// Accessor for txid
    pub fn txid(&self) -> Txid {
        self.transaction.txid()
    }

    /// Return an iterator over spent outpoints/outputs
    pub fn spent_inputs(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.inputs().iter().zip(self.spent_utxos.iter())
    }

    /// Returns the total Bitcoin value spent
    pub fn spent_bitcoin_value(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.spent_utxos
            .iter()
            .map(GetBitcoinValue::get_bitcoin_value)
            .checked_sum()
            .ok_or(AmountOverflowError)
    }

    /// Returns the total Bitcoin value in the outputs
    pub fn bitcoin_value_out(
        &self,
    ) -> Result<bitcoin::Amount, AmountOverflowError> {
        self.outputs()
            .iter()
            .map(GetBitcoinValue::get_bitcoin_value)
            .checked_sum()
            .ok_or(AmountOverflowError)
    }

    /// Returns the difference between the value spent and value out, if it is
    /// non-negative.
    pub fn bitcoin_fee(
        &self,
    ) -> Result<Option<bitcoin::Amount>, AmountOverflowError> {
        let spent_value = self.spent_bitcoin_value()?;
        let value_out = self.bitcoin_value_out()?;
        if spent_value < value_out {
            Ok(None)
        } else {
            Ok(Some(spent_value - value_out))
        }
    }

    /// Return an iterator over spent Votecoin
    pub fn spent_votecoin(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_votecoin())
    }

    /** Return an iterator over spent assets (Bitcoin and Votecoin) */
    pub fn spent_assets(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs().filter(|(_, filled_output)| {
            filled_output.is_bitcoin() || filled_output.is_votecoin()
        })
    }

         /** Returns the total amount of Votecoin spent in this transaction.
      *  Since Votecoin has a fixed supply with no subtypes, this returns
      *  a simple total amount. */
     pub fn unique_spent_votecoin(&self) -> Option<u32> {
         let mut total_votecoin: u64 = 0;
         for (_, output) in self.spent_votecoin() {
             if let Some(amount) = output.votecoin() {
                 total_votecoin += amount as u64;
             }
         }
         if total_votecoin > 0 {
             // Convert back to u32, should not overflow since Votecoin uses u32
             total_votecoin.try_into().ok()
         } else {
             None
         }
     }

    /** Return a vector of pairs consisting of an [`AssetId`] and the combined
     *  input value for that asset.
     *  The vector is ordered such that assets occur in the same order
     *  as they first occur in the inputs. */
    pub fn unique_spent_assets(&self) -> Vec<(AssetId, u64)> {
        // Combined value for each asset
        let mut combined_value = HashMap::<AssetId, u64>::new();
        let spent_asset_values = || {
            self.spent_assets()
                .filter_map(|(_, output)| output.asset_value())
        };
        // Increment combined value for the asset
        spent_asset_values().for_each(|(asset, value)| {
            *combined_value.entry(asset).or_default() += value;
        });
        spent_asset_values()
            .unique_by(|(asset, _)| *asset)
            .map(|(asset, _)| (asset, combined_value[&asset]))
            .collect()
    }



    /// Return an iterator over spent AMM LP tokens
    pub fn spent_lp_tokens(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &FilledOutput)> {
        self.spent_inputs()
            .filter(|(_, filled_output)| filled_output.is_lp_token())
    }

    /** Return a vector of pairs consisting of an LP token's corresponding
     *  asset pair and the combined input amount for that LP token.
     *  The vector is ordered such that LP tokens occur in the same order
     *  as they first occur in the inputs. */
    pub fn unique_spent_lp_tokens(&self) -> Vec<(AssetId, AssetId, u64)> {
        // Combined amount for each LP token
        let mut combined_amounts = HashMap::<(AssetId, AssetId), u64>::new();
        let spent_lp_token_amounts = || {
            self.spent_lp_tokens()
                .filter_map(|(_, output)| output.lp_token_amount())
        };
        // Increment combined amount for the LP token
        spent_lp_token_amounts().for_each(|(asset0, asset1, amount)| {
            *combined_amounts.entry((asset0, asset1)).or_default() += amount;
        });
        spent_lp_token_amounts()
            .unique_by(|(asset0, asset1, _)| (*asset0, *asset1))
            .map(|(asset0, asset1, _)| {
                (asset0, asset1, combined_amounts[&(asset0, asset1)])
            })
            .collect()
    }

    /** Returns an iterator over total value for each asset that must
     *  appear in the outputs, in order.
     *  The total output value can possibly over/underflow in a transaction,
     *  so the total output values are [`Option<u64>`],
     *  where `None` indicates over/underflow. */
    fn output_asset_total_values(
        &self,
    ) -> impl Iterator<Item = (AssetId, Option<u64>)> + '_ {
        // Votecoin has a fixed supply and cannot be created or destroyed
        // No special transaction data handling needed for Votecoin
        let (mut amm_burn0, mut amm_burn1) = match self.amm_burn() {
            Some(AmmBurn {
                asset0,
                asset1,
                amount0,
                amount1,
                lp_token_burn: _,
            }) => (Some((asset0, amount0)), Some((asset1, amount1))),
            None => (None, None),
        };
        let (mut amm_mint0, mut amm_mint1) = match self.amm_mint() {
            Some(AmmMint {
                asset0,
                asset1,
                amount0,
                amount1,
                lp_token_mint: _,
            }) => (Some((asset0, amount0)), Some((asset1, amount1))),
            None => (None, None),
        };
        let (mut amm_swap_spend, mut amm_swap_receive) = match self.amm_swap() {
            Some(AmmSwap {
                asset_spend,
                asset_receive,
                amount_spend,
                amount_receive,
            }) => (
                Some((asset_spend, amount_spend)),
                Some((asset_receive, amount_receive)),
            ),
            None => (None, None),
        };

        self.unique_spent_assets()
            .into_iter()
            .map(move |(asset, total_value)| {
                let total_value = if let Some((burn_asset, burn_amount)) = amm_burn0
                    && burn_asset == asset
                {
                    amm_burn0 = None;
                    total_value.checked_add(burn_amount)
                } else if let Some((burn_asset, burn_amount)) = amm_burn1
                    && burn_asset == asset
                {
                    amm_burn1 = None;
                    total_value.checked_add(burn_amount)
                } else if let Some((mint_asset, mint_amount)) = amm_mint0
                    && mint_asset == asset
                {
                    amm_mint0 = None;
                    total_value.checked_sub(mint_amount)
                } else if let Some((mint_asset, mint_amount)) = amm_mint1
                    && mint_asset == asset
                {
                    amm_mint1 = None;
                    total_value.checked_sub(mint_amount)
                } else if let Some((swap_spend_asset, swap_spend_amount)) =
                    amm_swap_spend
                    && swap_spend_asset == asset
                {
                    amm_swap_spend = None;
                    total_value.checked_sub(swap_spend_amount)
                } else if let Some((swap_receive_asset, swap_receive_amount)) =
                    amm_swap_receive
                    && swap_receive_asset == asset
                {
                    amm_swap_receive = None;
                    total_value.checked_add(swap_receive_amount)

                } else {
                    Some(total_value)
                };
                (asset, total_value)
            })
            .filter(|(_, amount)| *amount != Some(0))
            .chain(amm_burn0.map(|(burn_asset, burn_amount)| {
                (burn_asset, Some(burn_amount))
            }))
            .chain(amm_burn1.map(|(burn_asset, burn_amount)| {
                (burn_asset, Some(burn_amount))
            }))
            .chain(amm_mint0.map(|(mint_asset, _)|
                    /* If the assets are not already accounted for,
                    * indicate an underflow */
                    (mint_asset, None)))
            .chain(amm_mint1.map(|(mint_asset, _)|
                    /* If the assets are not already accounted for,
                    * indicate an underflow */
                    (mint_asset, None)))
            .chain(amm_swap_spend.map(|(spend_asset, _)|
                    /* If the assets are not already accounted for,
                    * indicate an underflow */
                    (spend_asset, None)))
            .chain(amm_swap_receive.map(|(receive_asset, receive_amount)| {
                (receive_asset, Some(receive_amount))
            }))
    }



    /** Returns the max value of Bitcoin that can occur in the outputs.
     *  The total output value can possibly over/underflow in a transaction,
     *  so the total output values are [`Option<bitcoin::Amount>`],
     *  where `None` indicates over/underflow. */
    fn output_bitcoin_max_value(&self) -> Option<bitcoin::Amount> {
        self.output_asset_total_values()
            .find_map(|(asset_id, value)| match asset_id {
                AssetId::Bitcoin => Some(value.map(bitcoin::Amount::from_sat)),
                _ => None,
            })
            .unwrap_or(Some(bitcoin::Amount::ZERO))
    }

    /** Returns an iterator over total amount for each LP token that must
     *  appear in the outputs, in order.
     *  The total output value can possibly over/underflow,
     *  so the total output values are [`Option<u64>`],
     *  where `None` indicates over/underflow. */
    fn output_lp_token_total_amounts(
        &self,
    ) -> impl Iterator<Item = (AssetId, AssetId, Option<u64>)> + '_ {
        /* If this tx is an AMM burn, this is the corresponding asset IDs
        and token amount of the output corresponding to the newly created
        AMM LP position. */
        let mut amm_burn: Option<AmmBurn> = self.amm_burn();
        /* If this tx is an AMM mint, this is the corresponding asset IDs
        and token amount of the output corresponding to the newly created
        AMM LP position. */
        let mut amm_mint: Option<AmmMint> = self.amm_mint();
        self.unique_spent_lp_tokens()
            .into_iter()
            .map(move |(asset0, asset1, total_amount)| {
                let total_value = if let Some(AmmBurn {
                    asset0: burn_asset0,
                    asset1: burn_asset1,
                    amount0: _,
                    amount1: _,
                    lp_token_burn,
                }) = amm_burn
                    && (burn_asset0, burn_asset1) == (asset0, asset1)
                {
                    amm_burn = None;
                    total_amount.checked_sub(lp_token_burn)
                } else if let Some(AmmMint {
                    asset0: mint_asset0,
                    asset1: mint_asset1,
                    amount0: _,
                    amount1: _,
                    lp_token_mint,
                }) = amm_mint
                    && (mint_asset0, mint_asset1) == (asset0, asset1)
                {
                    amm_mint = None;
                    total_amount.checked_add(lp_token_mint)
                } else {
                    Some(total_amount)
                };
                (asset0, asset1, total_value)
            })
            .chain(amm_burn.map(|amm_burn| {
                /* If the LP tokens are not already accounted for,
                 * indicate an underflow */
                (amm_burn.asset0, amm_burn.asset1, None)
            }))
            .chain(amm_mint.map(|amm_mint| {
                (
                    amm_mint.asset0,
                    amm_mint.asset1,
                    Some(amm_mint.lp_token_mint),
                )
            }))
    }







    /// compute the filled outputs.
    /// returns None if the outputs cannot be filled because the tx is invalid
    // FIXME: Invalidate tx if any iterator is incomplete
    pub fn filled_outputs(&self) -> Option<Vec<FilledOutput>> {
        let mut output_bitcoin_max_value = self.output_bitcoin_max_value()?;
        let mut output_lp_token_total_amounts =
            self.output_lp_token_total_amounts().peekable();

        self.outputs()
            .iter()
            .map(|output| {
                let content = match output.content.clone() {
                    OutputContent::AmmLpToken(amount) => {
                        let (asset0, asset1, remaining_amount) =
                            output_lp_token_total_amounts.peek_mut()?;
                        let remaining_amount = remaining_amount.as_mut()?;
                        let filled_content = FilledOutputContent::AmmLpToken {
                            asset0: *asset0,
                            asset1: *asset1,
                            amount,
                        };
                        match amount.cmp(remaining_amount) {
                            Ordering::Greater => {
                                // Invalid tx, return `None`
                                return None;
                            }
                            Ordering::Equal => {
                                // Advance the iterator to the next LP token
                                let _ = output_lp_token_total_amounts.next()?;
                            }
                            Ordering::Less => {
                                // Decrement the remaining value for the current LP token
                                *remaining_amount -= amount;
                            }
                        }
                        filled_content
                    }
                    OutputContent::Votecoin(value) => {
                        FilledOutputContent::Votecoin(value)
                    }
                    OutputContent::Bitcoin(value) => {
                        output_bitcoin_max_value =
                            output_bitcoin_max_value.checked_sub(value.0)?;
                        FilledOutputContent::Bitcoin(value)
                    }
                    OutputContent::Withdrawal(withdrawal) => {
                        FilledOutputContent::BitcoinWithdrawal(withdrawal)
                    }
                };
                Some(FilledOutput {
                    address: output.address,
                    content,
                    memo: output.memo.clone(),
                })
            })
            .collect()
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize)]
pub struct Authorized<T> {
    pub transaction: T,
    /// Authorizations are called witnesses in Bitcoin.
    pub authorizations: Vec<Authorization>,
}

pub type AuthorizedTransaction = Authorized<Transaction>;

impl AuthorizedTransaction {
    /// Return an iterator over all addresses relevant to the transaction
    pub fn relevant_addresses(&self) -> HashSet<Address> {
        let input_addrs =
            self.authorizations.iter().map(|auth| auth.get_address());
        let output_addrs =
            self.transaction.outputs.iter().map(|output| output.address);
        input_addrs.chain(output_addrs).collect()
    }
}

impl From<Authorized<FilledTransaction>> for AuthorizedTransaction {
    fn from(tx: Authorized<FilledTransaction>) -> Self {
        Self {
            transaction: tx.transaction.transaction,
            authorizations: tx.authorizations,
        }
    }
}
