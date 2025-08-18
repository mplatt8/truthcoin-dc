# Manual Regtest Setup for Truthcoin Drivechain Testing

This guide provides the exact steps to manually run a regtest environment for testing Votecoin transactions with the Truthcoin GUI.

## Prerequisites

Build all required components:

```bash
# Build bitcoind and bitcoin-cli
cd ../bitcoin && make -j$(nproc)

# Build electrs
cd ../electrs && cargo build --release

# Build bip300301_enforcer
cd ../bip300301_enforcer && cargo build --debug

# Build truthcoin_dc_app
cd ../truthcoin-dc && cargo build --bin truthcoin_dc_app
```

## Setup

### 1. Create Data Directories

## Clean up old temp dir

```bash
rm -rf /tmp/regtest-data
```

## New temp dir

```bash
mkdir -p /tmp/regtest-data/{bitcoin,electrs,enforcer,truthcoin}
```

### 2. Start Bitcoin Core

```bash
../bitcoin/build/src/bitcoind \
  -acceptnonstdtxn \
  -chain=regtest \
  -datadir=/tmp/regtest-data/bitcoin \
  -bind=127.0.0.1:18444 \
  -rpcuser=regtest_user \
  -rpcpassword=regtest_pass \
  -rpcport=18443 \
  -rest \
  -server \
  -zmqpubsequence=tcp://127.0.0.1:28332 \
  -listenonion=0 \
  -txindex
```

### 3. Start Electrs

```bash
../electrs/target/release/electrs \
  -vv \
  --db-dir=/tmp/regtest-data/electrs \
  --daemon-dir=/tmp/regtest-data/bitcoin \
  --daemon-rpc-addr=127.0.0.1:18443 \
  --electrum-rpc-addr=127.0.0.1:50001 \
  --http-addr=127.0.0.1:3000 \
  --monitoring-addr=127.0.0.1:4224 \
  --network=regtest \
  --cookie=regtest_user:regtest_pass \
  --jsonrpc-import
```

### 4. Start BIP300301 Enforcer

```bash
../bip300301_enforcer/target/debug/bip300301_enforcer \
  --data-dir=/tmp/regtest-data/enforcer \
  --node-rpc-addr=127.0.0.1:18443 \
  --node-rpc-user=regtest_user \
  --node-rpc-pass=regtest_pass \
  --enable-wallet \
  --log-level=trace \
  --serve-grpc-addr=127.0.0.1:50051 \
  --serve-json-rpc-addr=127.0.0.1:18080 \
  --serve-rpc-addr=127.0.0.1:18081 \
  --wallet-auto-create \
  --wallet-electrum-host=127.0.0.1 \
  --wallet-electrum-port=50001 \
  --wallet-esplora-url=http://127.0.0.1:3000 \
  --wallet-skip-periodic-sync \
  --enable-mempool
```

### 5. Start Truthcoin App

## GUI

```bash
./target/debug/truthcoin_dc_app \
  --datadir=/tmp/regtest-data/truthcoin \
  --network=regtest \
  --mainchain-grpc-port=50051 \
  --net-addr=127.0.0.1:18445 \
  --rpc-port=18332 \
  --zmq-addr=127.0.0.1:28333
```

## Headless for CLI
```bash
./target/debug/truthcoin_dc_app \
  --headless \
  --datadir=/tmp/regtest-data/truthcoin \
  --network=regtest \
  --mainchain-grpc-port=50051 \
  --net-addr=127.0.0.1:18445 \
  --rpc-port=18332 \
  --zmq-addr=127.0.0.1:28333
  ```
### headless needs a l2 wallet created
```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 generate-mnemonic
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 set-seed-from-mnemonic "wise siren dizzy garden hamster depend van round banana nose tree utility"
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-new-address

```

## Activation and Funding

### 1. Fund the Enforcer

```bash
# Generate 101 blocks (enforcer receives coinbase rewards automatically)
grpcurl -plaintext -d '{"blocks": 101}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks

# Check enforcer balance
grpcurl -plaintext -d '{}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GetBalance
```

### 2. Propose and Activate Truthcoin Sidechain

```bash
# Propose sidechain (keep this command running - it's a stream)
grpcurl -plaintext -d '{
  "sidechain_id": 13,
  "declaration": {
    "v0": {
      "title": "Truthcoin",
      "description": "Truthcoin Drivechain",
      "hash_id_1": {"hex": "0000000000000000000000000000000000000000000000000000000000000000"},
      "hash_id_2": {"hex": "0000000000000000000000000000000000000000"}
    }
  }
}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.CreateSidechainProposal
```

In another terminal:

```bash
# Mine 1 block to include proposal
grpcurl -plaintext -d '{"blocks": 1, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks

# Mine 6 blocks with ACK to activate sidechain
grpcurl -plaintext -d '{"blocks": 6, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks

# Check sidechain is active
grpcurl -plaintext -d '{}' 127.0.0.1:50051 cusf.mainchain.v1.ValidatorService.GetSidechains
```

### 3. Create Deposit to Sidechain

Get a Truthcoin address from your GUI or RPC, then:

```bash
# Create deposit (replace ADDRESS with your Truthcoin address)
grpcurl -plaintext -d '{
  "sidechain_id": 13,
  "address": "Q1Qs4uBcgWxLymD1smACn8kgZEb",
  "value_sats": 100000000,
  "fee_sats": 10000
}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.CreateDepositTransaction

# Mine Bitcoin block to confirm BMM transaction
grpcurl -plaintext -d '{"blocks": 1, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks
```

### 4. Mine Truthcoin Genesis Block

```bash
# Mine genesis block to get initial Votecoin supply (1,000,000 units)
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "mine", "params": [null], "id": 1}'
```
### 5. Mine L1 block

```bash
# Mine one block on L1 to confirm sidechain state
grpcurl -plaintext -d '{"blocks": 1, "ack_all_proposals": true}' 127.0.0.1:50051 cusf.mainchain.v1.WalletService.GenerateBlocks
```


## Testing

After completing the setup:

- **Bitcoin deposits** will appear in your Truthcoin GUI
- **Votecoin balance** will show 1,000,000 units after mining genesis block
- **Send Votecoin** between addresses using the GUI
- **Mine more blocks** using the curl command above to confirm transactions

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Bitcoin Core RPC | 18443 | `user: regtest_user, pass: regtest_pass` |
| BIP300301 Enforcer gRPC | 50051 | Sidechain operations via grpcurl |
| Truthcoin App RPC | 18332 | GUI connection and mining |

## Cleanup

```bash
rm -rf /tmp/regtest-data
```

**Note**: All services automatically manage their own wallets. The enforcer receives mining rewards and handles Bitcoin operations. The Truthcoin app manages the sidechain and Votecoin.

## Available CLI RPC Commands

All commands use the format: `./target/debug/truthcoin_dc_app_cli --rpc-port 18332 <COMMAND> [OPTIONS]`

### Wallet Management
```bash
# Generate a new mnemonic seed phrase
generate-mnemonic

# Set wallet seed from mnemonic
set-seed-from-mnemonic "your twelve word mnemonic phrase here"

# Get a new address
get-new-address

# Get all wallet addresses
get-wallet-addresses

# Get a new encryption key
get-new-encryption-key

# Get a new verifying key  
get-new-verifying-key

# Get wallet UTXOs
get-wallet-utxos

# List owned UTXOs
my-utxos

# List unconfirmed owned UTXOs
my-unconfirmed-utxos
```

### Balance and Wealth
```bash
# Get Bitcoin balance in sats
bitcoin-balance

# Get total sidechain wealth
sidechain-wealth
```

### Transfers and Deposits
```bash
# Transfer Bitcoin to address
transfer DEST_ADDRESS --value-sats 1000000 --fee-sats 1000

# Transfer Votecoin to address
transfer-votecoin DEST_ADDRESS --amount 100 --fee-sats 1000

# Create deposit to address
create-deposit DEST_ADDRESS --value-sats 1000000 --fee-sats 1000

# Withdraw to mainchain address
withdraw MAINCHAIN_ADDRESS --amount-sats 1000000 --fee-sats 1000 --mainchain-fee-sats 10000

# Format deposit address
format-deposit-address ADDRESS
```

### Mining and Blocks
```bash
# Mine a sidechain block
mine [--fee-sats 1000]

# Get current block count
get-blockcount

# Get block data by hash
get-block BLOCK_HASH

# Get best mainchain block hash
get-best-mainchain-block-hash

# Get best sidechain block hash  
get-best-sidechain-block-hash

# Get BMM inclusions for block hash
get-bmm-inclusions BLOCK_HASH
```

### Transactions
```bash
# Get transaction by txid
get-transaction TXID

# Get transaction info
get-transaction-info TXID

# Remove transaction from mempool
remove-from-mempool TXID

# List all UTXOs
list-utxos
```

### AMM (Automated Market Maker)
```bash
# Get AMM pool state
get-amm-pool-state --asset0 bitcoin --asset1 votecoin

# Get AMM price
get-amm-price --base bitcoin --quote votecoin

# AMM swap
amm-swap --asset-spend bitcoin --asset-receive votecoin --amount-spend 1000000

# Mint AMM position
amm-mint --asset0 bitcoin --asset1 votecoin --amount0 1000000 --amount1 100

# Burn AMM position
amm-burn --asset0 bitcoin --asset1 votecoin --lp-token-amount 1000
```

### Slots and Decisions
```bash
# List all slots by period
slots-list-all

# Get slots for specific period  
slots-get-quarter 42

# Show slot system status
slots-status

# Convert timestamp to period
slots-convert-timestamp 1640995200

# Claim a decision slot
claim-decision-slot --period-index 42 --slot-index 100 --is-standard true --is-scaled false --question "Will Bitcoin reach $100k?" --fee-sats 1000

# Get available slots in period
get-available-slots --period-index 42

# Get slot by ID
get-slot-by-id --slot-id-hex "2a0064"

# Get claimed slots in period
get-claimed-slots --period-index 42

# Check if slot is in voting period
is-slot-in-voting --slot-id-hex "2a0064"

# Get periods currently in voting phase
get-voting-periods

# Get ossified slots (slots whose voting period has ended)
get-ossified-slots
```

### Networking
```bash
# Connect to peer
connect-peer 127.0.0.1:8333

# List connected peers
list-peers
```

### Encryption and Signing
```bash
# Encrypt message to pubkey
encrypt-msg --encryption-pubkey PUBKEY --msg "secret message"

# Decrypt message
decrypt-msg --encryption-pubkey PUBKEY --msg ENCRYPTED_HEX [--utf8]

# Sign arbitrary message with verifying key
sign-arbitrary-msg --verifying-key KEY --msg "message to sign"

# Sign arbitrary message as address
sign-arbitrary-msg-as-addr --address ADDRESS --msg "message to sign"

# Verify signature
verify-signature --signature SIG --verifying-key KEY --dst DST --msg "original message"
```

### Withdrawals
```bash
# Get pending withdrawal bundle
pending-withdrawal-bundle

# Get height of latest failed withdrawal bundle
latest-failed-withdrawal-bundle-height
```