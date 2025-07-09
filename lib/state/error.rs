//! State errors

use sneed::{db::error as db, env::error as env, rwtxn::error as rwtxn};
use thiserror::Error;
use transitive::Transitive;

use crate::types::{
    AmountOverflowError, AmountUnderflowError, AssetId, BlockHash,
    Hash, M6id, MerkleRoot, OutPoint, Txid, WithdrawalBundleError,
};

/// Errors related to an AMM pool
#[derive(Debug, Error, Transitive)]
#[transitive(from(db::Delete, db::Error))]
#[transitive(from(db::Error, sneed::Error))]
#[transitive(from(db::Put, db::Error))]
#[transitive(from(db::TryGet, db::Error))]
pub enum Amm {
    #[error("AMM burn overflow")]
    BurnOverflow,
    #[error("AMM burn underflow")]
    BurnUnderflow,
    #[error(transparent)]
    Db(#[from] sneed::Error),
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    #[error("Invalid AMM burn")]
    InvalidBurn,
    #[error("Invalid AMM mint")]
    InvalidMint,
    #[error("Invalid AMM swap")]
    InvalidSwap,
    #[error("AMM LP token overflow")]
    LpTokenOverflow,
    #[error("AMM LP token underflow")]
    LpTokenUnderflow,
    #[error("missing AMM pool state for {asset0}-{asset1}")]
    MissingPoolState { asset0: AssetId, asset1: AssetId },
    #[error("AMM pool invariant")]
    PoolInvariant,
    #[error("Failed to revert AMM mint")]
    RevertMint,
    #[error("Failed to revert AMM swap")]
    RevertSwap,
    #[error("Too few Votecoin to mint an AMM position")]
    TooFewVotecoinToMint,
}





#[derive(Debug, Error)]
pub enum InvalidHeader {
    #[error("expected block hash {expected}, but computed {computed}")]
    BlockHash {
        expected: BlockHash,
        computed: BlockHash,
    },
    #[error(
        "expected previous sidechain block hash {expected:?}, but received {received:?}"
    )]
    PrevSideHash {
        expected: Option<BlockHash>,
        received: Option<BlockHash>,
    },
}

#[derive(Debug)]
pub struct FillTxOutputContents(pub Box<crate::types::FilledTransaction>);

impl std::fmt::Display for FillTxOutputContents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let txid = self.0.txid();
        write!(
            f,
            "failed to fill tx output contents ({txid}): invalid transaction"
        )?;
        if f.alternate() {
            let tx_json = serde_json::to_string(&self.0).unwrap();
            write!(f, " ({tx_json})")?;
        }
        Ok(())
    }
}

impl std::error::Error for FillTxOutputContents {}

#[derive(Debug, Error, Transitive)]
#[transitive(from(db::Clear, db::Error))]
#[transitive(from(db::Delete, db::Error))]
#[transitive(from(db::Error, sneed::Error))]
#[transitive(from(db::IterInit, db::Error))]
#[transitive(from(db::IterItem, db::Error))]
#[transitive(from(db::Last, db::Error))]
#[transitive(from(db::Put, db::Error))]
#[transitive(from(db::TryGet, db::Error))]
#[transitive(from(env::CreateDb, env::Error))]
#[transitive(from(env::Error, sneed::Error))]
#[transitive(from(env::WriteTxn, env::Error))]
#[transitive(from(rwtxn::Commit, rwtxn::Error))]
#[transitive(from(rwtxn::Error, sneed::Error))]
pub enum Error {
    #[error(transparent)]
    Amm(#[from] Amm),
    #[error(transparent)]
    AmountOverflow(#[from] AmountOverflowError),
    #[error(transparent)]
    AmountUnderflow(#[from] AmountUnderflowError),
    #[error("failed to verify authorization")]
    AuthorizationError,
    #[error("bad coinbase output content")]
    BadCoinbaseOutputContent,


    #[error("bundle too heavy {weight} > {max_weight}")]
    BundleTooHeavy { weight: u64, max_weight: u64 },
    #[error(transparent)]
    BorshSerialize(borsh::io::Error),
    #[error(transparent)]
    Db(#[from] sneed::Error),
    #[error(transparent)]
    FillTxOutputContents(#[from] FillTxOutputContents),
    #[error(
        "invalid body: expected merkle root {expected}, but computed {computed}"
    )]
    InvalidBody {
        expected: MerkleRoot,
        computed: MerkleRoot,
    },
    #[error("invalid header: {0}")]
    InvalidHeader(InvalidHeader),


    #[error("deposit block doesn't exist")]
    NoDepositBlock,
    #[error("total fees less than coinbase value")]
    NotEnoughFees,
    #[error("value in is less than value out")]
    NotEnoughValueIn,
    #[error("no tip")]
    NoTip,
    #[error("stxo {outpoint} doesn't exist")]
    NoStxo { outpoint: OutPoint },
    #[error("utxo {outpoint} doesn't exist")]
    NoUtxo { outpoint: OutPoint },
    #[error("Withdrawal bundle event block doesn't exist")]
    NoWithdrawalBundleEventBlock,

    #[error(transparent)]
    SignatureError(#[from] ed25519_dalek::SignatureError),



    #[error(
        "unbalanced Votecoin: {inputs} inputs, {outputs} outputs"
    )]
    UnbalancedVotecoin {
        inputs: u32,
        outputs: u32,
    },

    #[error("Unknown withdrawal bundle: {m6id}")]
    UnknownWithdrawalBundle { m6id: M6id },
    #[error(
        "Unknown withdrawal bundle confirmed in {event_block_hash}: {m6id}"
    )]
    UnknownWithdrawalBundleConfirmed {
        event_block_hash: bitcoin::BlockHash,
        m6id: M6id,
    },
    #[error("utxo double spent")]
    UtxoDoubleSpent,
    #[error(transparent)]
    WithdrawalBundle(#[from] WithdrawalBundleError),
    #[error("wrong public key for address")]
    WrongPubKeyForAddress,
}
