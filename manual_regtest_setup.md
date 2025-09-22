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

```bash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 generate-mnemonic
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 set-seed-from-mnemonic "leaf ready burden satisfy fire setup crack tide sound crucial appear cup"
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-new-address
```

### 3. Create Deposit to Sidechain

Get a Truthcoin address from your GUI or RPC, then:

```bash
# Create deposit (replace ADDRESS with your Truthcoin address)
grpcurl -plaintext -d '{
  "sidechain_id": 13,
  "address": "4Mu1f9Tw95SfoBdffagH7a3JrWBf",
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
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 generate-mnemonic

# Set wallet seed from mnemonic
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 set-seed-from-mnemonic "your twelve word mnemonic phrase here"

# Get a new address (aliases: addr, address, new-addr)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-new-address

# Get all wallet addresses
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-wallet-addresses

# Get a new encryption key
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-new-encryption-key

# Get a new verifying key  
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-new-verifying-key

# Get wallet UTXOs
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-wallet-utxos

# List owned UTXOs
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 my-utxos

# List unconfirmed owned UTXOs
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 my-unconfirmed-utxos
```

### Balance and Wealth
```bash
# Get Bitcoin balance in sats (aliases: bal, b)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 balance

# Get total sidechain wealth
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 sidechain-wealth
```

### Transfers and Deposits
```bash
# Transfer Bitcoin to address (aliases: send, tx)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 transfer DEST_ADDRESS --value-sats 1000000 --fee-sats 1000

# Transfer Votecoin to address (aliases: send-votecoin, send-vc, transfer-vc)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 transfer-votecoin DEST_ADDRESS --amount 100 --fee-sats 1000

# Create deposit to address
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 create-deposit DEST_ADDRESS --value-sats 1000000 --fee-sats 1000

# Withdraw to mainchain address
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 withdraw MAINCHAIN_ADDRESS --amount-sats 1000000 --fee-sats 1000 --mainchain-fee-sats 10000

# Format deposit address
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 format-deposit-address ADDRESS
```

### Mining and Blocks
```bash
# Mine a sidechain block (alias: m)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine [--fee-sats 1000]

# Get current block count (aliases: blockcount, height)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-block-count

# Get block data by hash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-block BLOCK_HASH

# Get best mainchain block hash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-best-mainchain-block-hash

# Get best sidechain block hash  
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-best-sidechain-block-hash

# Get BMM inclusions for block hash
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-bmm-inclusions BLOCK_HASH
```

### Transactions
```bash
# Get transaction by txid
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-transaction TXID

# Get transaction info
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-transaction-info TXID

# Remove transaction from mempool
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 remove-from-mempool TXID

# List all UTXOs
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-utxos
```


### Slots and Decisions
```bash
# List all slots by period (aliases: slots, list-slots)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slots-list-all

# Get slots for specific period  
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slots-get-quarter 30

# Show slot system status
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slots-status

# Convert timestamp to period
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 slots-convert-timestamp 1640995200

# Claim a decision slot (aliases: claim-slot, claim)

# Get available slots in period
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-available-slots --period-index 30

# Get slot by ID (aliases: slot, get-slot)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-slot-by-id --slot-id-hex "2a0064"

# Get claimed slots in period
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-claimed-slots --period-index 20

# Check if slot is in voting period
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 is-slot-in-voting --slot-id-hex "2a0064"

# Get periods currently in voting phase (aliases: voting-periods, voting)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-voting-periods

# Get ossified slots (slots whose voting period has ended) (aliases: ossified-slots, ossified)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-ossified-slots

### Market Creation Examples

# Create a market using initial capital (liquidity) in satoshis
# The beta parameter will be automatically derived: β = Capital / ln(n)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 create-market \
  --title "Steelers Super Bowl 60" \
  --description "Will the Pittsburgh Steelers win Super Bowl 60?" \
  --decision-slots "050001" \
  --initial-liquidity 1000000 \
  --trading-fee 0.005 \
  --tags "sports,nfl,superbowl,steelers" \
  --fee-sats 1000

# Alternative: Create a market using the beta parameter directly
# The required capital will be derived: Capital = β × ln(n)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 create-market \
  --title "Bitcoin $100k" \
  --description "Will Bitcoin reach $100,000 by end of 2024?" \
  --decision-slots "050001" \
  --beta 7.0 \
  --fee-sats 1000
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-ossified-slots
```

### Multidimensional Markets and Prediction Trading
```bash
# Create a multidimensional prediction market using the new dimensional specification
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 create-market \
  --title "2024 Election & Economy Multidimensional Market" \
  --description "Complex market with multiple independent and categorical dimensions" \
  --dimensions "[2a0064,[2a0065,2a0066,2a0067],2a0068]" \
  --beta 14400.0 \  # Beta in satoshis (14400 ≈ 10k sats initial capital for binary market)
  --trading-fee 0.005 \
  --tags "politics,economy,multidimensional" \
  --fee-sats 15000

# List all markets currently in Trading state (alias: markets, ls-markets)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-markets

# View detailed information for a specific market (aliases: show-market, info)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market 8d2a6b4bb2f6
```

### Multidimensional Market Creation Workflow

This example creates a complex market with mixed dimension types:

```bash
# 1. First claim decision slots for your market dimensions
# Dimension 1: Independent binary decision 
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 25 --slot-index 100 --is-standard true --is-scaled false --question "Will Bitcoin reach $150k by end of 2025?" --fee-sats 1000

# Dimension 2: Categorical dimension (mutually exclusive options)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 42 --slot-index 101 --is-standard true --is-scaled false --question "Will Donald Trump win the 2024 election?" --fee-sats 1000
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 42 --slot-index 102 --is-standard true --is-scaled false --question "Will Joe Biden win the 2024 election?" --fee-sats 1000  
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 42 --slot-index 103 --is-standard true --is-scaled false --question "Will a third party candidate win the 2024 election?" --fee-sats 1000

# Dimension 3: Independent scaled decision
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 42 --slot-index 104 --is-standard true --is-scaled true --question "What will be the S&P 500 closing value on Dec 31, 2024?" --min 3000 --max 8000 --fee-sats 1000

# 2. Mine a block to confirm the slot claims
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000

# 3. Create the multidimensional market using dimensional specification
# Format: [slot1,[slot2,slot3,slot4],slot5]
# - slot1: Independent dimension (Bitcoin price)
# - [slot2,slot3,slot4]: Categorical dimension (election winner - mutually exclusive)  
# - slot5: Independent dimension (S&P 500 value)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 create-market \
  --title "2024 Bitcoin, Election & S&P 500 Multidimensional Market" \
  --description "Complex prediction market combining Bitcoin price, election outcome, and stock market performance. Each dimension can be traded independently except the election outcome which is mutually exclusive." \
  --dimensions "[2a0064,[2a0065,2a0066,2a0067],2a0068]" \
  --beta 15.0 \
  --trading-fee 0.008 \
  --tags "bitcoin,politics,election,stocks,multidimensional" \
  --fee-sats 20000

# 4. Mine another block to confirm the market creation
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000

# 5. List markets to see your new multidimensional market
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-markets

# 6. View detailed market information showing all dimensions and outcome combinations
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market MARKET_ID_HEX
```

### Dimension Specification Format

The `--dimensions` parameter uses a JSON-like format to specify market structure:

- **Single slot** (independent dimension): `slot_id`
- **Categorical dimension** (mutually exclusive): `[slot_id1,slot_id2,slot_id3]`
- **Mixed dimensions**: `[slot1,[slot2,slot3],slot4,slot5]`

**Example interpretations:**
- `[2a0064]` → 1 dimension, independent
- `[[2a0064,2a0065,2a0066]]` → 1 dimension, categorical (3 options + residual)
- `[2a0064,2a0065,2a0066]` → 3 dimensions, all independent  
- `[2a0064,[2a0065,2a0066],2a0067]` → 3 dimensions: independent, categorical, independent

This creates markets with proper **D_Functions** that enforce logical constraints (e.g., only one election winner) while allowing independent trading on other dimensions.

### Networking
```bash
# Connect to peer
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 connect-peer 127.0.0.1:8333

# List connected peers
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-peers
```

### Encryption and Signing
```bash
# Encrypt message to pubkey
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 encrypt-msg --encryption-pubkey PUBKEY --msg "secret message"

# Decrypt message
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 decrypt-msg --encryption-pubkey PUBKEY --msg ENCRYPTED_HEX [--utf8]

# Sign arbitrary message with verifying key
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 sign-arbitrary-msg --verifying-key KEY --msg "message to sign"

# Sign arbitrary message as address
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 sign-arbitrary-msg-as-addr --address ADDRESS --msg "message to sign"

# Verify signature
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 verify-signature --signature SIG --verifying-key KEY --dst DST --msg "original message"
```

### Withdrawals
```bash
# Get pending withdrawal bundle
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 pending-withdrawal-bundle

# Get height of latest failed withdrawal bundle
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 latest-failed-withdrawal-bundle-height
```

## Available JSON-RPC Methods

All RPC methods can be called using curl with the JSON-RPC 2.0 protocol format:

```bash
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "METHOD_NAME", "params": [PARAMS], "id": 1}'
```

### Multidimensional Market RPCs
```bash
# List all markets in Trading state
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "list_markets", "params": [], "id": 1}'

# View detailed market information by market ID
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "view_market", "params": ["MARKET_ID_HEX"], "id": 1}'

# Create a new multidimensional prediction market
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", 
    "method": "create_market", 
    "params": {
      "title": "Pittsburgh Sports & Politics 2026 Mega-Market",
      "description": "Multidimensional predictions on Steelers wins, AFC North winner, mayor race,
      draft pick, Bitcoin price, Corc Brum 18 y/o",
      "dimensions": "[steelers_wins,[steelers_afc,ravens_afc,bengals_afc],mayor_party,draft_pick,bitcoin_200k]",
      "b": 25.0,
      "trading_fee": 0.01,
      "tags": ["sports", "politics", "crypto", "pittsburgh", "multidimensional"],
      "fee_sats": 15000
    }, 
    "id": 1
  }'

# Alternative simple multidimensional market (all independent dimensions)
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", 
    "method": "create_market", 
    "params": {
      "title": "Independent Multi-Event Market",
      "description": "Three independent binary predictions",
      "dimensions": "[2a0064,2a0065,2a0066]",
      "b": 10.0,
      "trading_fee": 0.005,
      "tags": ["independent", "binary"],
      "fee_sats": 10000
    }, 
    "id": 1
  }'
```

### Slot RPCs
```bash
# Get available slots in a period
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_available_slots_in_period", "params": [42], "id": 1}'

# Get claimed slots in a period
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_claimed_slots_in_period", "params": [42], "id": 1}'

# Get slot by ID
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_slot_by_id", "params": ["2a0064"], "id": 1}'

# Get ossified slots (voting period ended)
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_ossified_slots", "params": [], "id": 1}'
```

### Mining and Block RPCs  
```bash
# Mine a new sidechain block
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "mine", "params": [null], "id": 1}'

# Get current block count
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "getblockcount", "params": [], "id": 1}'

# Get best sidechain block hash
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_best_sidechain_block_hash", "params": [], "id": 1}'
```

### Wallet and Balance RPCs
```bash
# Get new address
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_new_address", "params": [], "id": 1}'

# Get Bitcoin balance
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "bitcoin_balance", "params": [], "id": 1}'

# Get sidechain wealth
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "sidechain_wealth", "params": [], "id": 1}'
```

## Complete Pittsburgh Example Market (Within 256 State Limit)

Here's a realistic workflow to create a Pittsburgh sports and politics multidimensional market that stays within the **256 outcome limit**:

### Step 1: Claim Required Decision Slots

```bash
# Dimension 1: AFC North winner (categorical - mutually exclusive)

./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 11 --slot-index 102 --is-standard true --is-scaled false --question "Will the Baltimore Ravens win the AFC North in 2026?" --fee-sats 1000
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 11 --slot-index 103 --is-standard true --is-scaled false --question "Will the Cincinnati Bengals win the AFC North in 2026?" --fee-sats 1000

# Dimension 2: Pittsburgh mayor party affiliation
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 11 --slot-index 104 --is-standard true --is-scaled false --question "Will the next Pittsburgh mayor be a Republican?" --fee-sats 1000

# Dimension 3: Steelers 2026 first round draft pick position
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 11 --slot-index 105 --is-standard true --is-scaled false --question "Will the Steelers' 2026 first round pick be offense?" --fee-sats 1000

# Dimension 4: Bitcoin price prediction
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 11 --slot-index 106 --is-standard true --is-scaled false --question "Will Bitcoin price be over $200,000 by end of 2026?" --fee-sats 1000

# Mine block to confirm slot claims
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000
```

### Step 2: Create the Multidimensional Market

```bash
# Create the Pittsburgh market with proper dimensional specification
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 create-market \
  --title "Pittsburgh Sports & Politics 2026 Market" \
  --description "Multidimensional prediction market combining AFC North winner (categorical), Pittsburgh mayor party (binary), Steelers draft pick type (binary), and Bitcoin price prediction (binary)" \
  --dimensions "[[2a0065,2a0066,2a0067],2a0068,2a0069,2a006a]" \
  --beta 15.0 \
  --trading-fee 0.01 \
  --tags "steelers,afc-north,pittsburgh,politics,bitcoin,multidimensional" \
  --fee-sats 15000

# Mine block to confirm market creation
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000
```

### Market Structure (Within 256 Limit)

This creates a **4-dimensional market** with **180 possible outcome combinations**:

- **Dimension 1**: AFC North winner (5 outcomes: Steelers + Ravens + Bengals + Browns/Other + Invalid)  
- **Dimension 2**: Mayor party (3 outcomes: Republican + Democrat + Invalid)
- **Dimension 3**: Draft pick type (3 outcomes: Offense + Defense + Invalid)
- **Dimension 4**: Bitcoin $200k (3 outcomes: Yes + No + Invalid)

**Total**: 5 × 3 × 3 × 3 = **135 combinations** ✅ (Well under 256 limit)

### Alternative: Including Steelers Win Total

If you want to include the Steelers win total, you'd need to reduce it to a smaller range:

```bash
# Steelers playoff range only (9-15 wins)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 claim-decision-slot --period-index 11 --slot-index 100 --is-standard true --is-scaled true --question "How many games will the Pittsburgh Steelers win in the 2026 regular season?" --min 9 --max 15 --fee-sats 1000

# This gives: 9 outcomes (9-15 + Invalid + Null)
# Total with 5 dimensions: 9 × 5 × 3 × 3 × 3 = 1,215 outcomes ❌ (Too many)
# Total with 4 dimensions: 9 × 3 × 3 × 3 = 243 outcomes ✅ (Within limit)
```

### 256 State Limit Explanation

The Bitcoin Hivemind whitepaper specifies a maximum of 256 market states to keep:
- **Vote matrices manageable** for the consensus mechanism
- **LMSR calculations efficient** 
- **Storage requirements reasonable**

Complex markets require careful dimension design to stay within this constraint while still capturing meaningful correlations.

### Verification

```bash
# View your market details
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market MARKET_ID_HEX

# List all active markets
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-markets
```

## Share Trading

Once markets are created, you can buy and sell shares on different outcomes:

### Share Trading Quick Reference

```bash
# View market details and current prices
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market <MARKET_ID>

# Calculate cost before trading
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 calculate-share-cost \
  --market-id <MARKET_ID> --outcome-index <INDEX> --shares-amount <AMOUNT>

# Buy shares (positive amount)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id <MARKET_ID> --outcome-index <INDEX> --shares-amount <POSITIVE_AMOUNT> \
  --max-cost <MAX_SATS> --fee-sats 1000

# Sell shares (negative amount)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id <MARKET_ID> --outcome-index <INDEX> --shares-amount <NEGATIVE_AMOUNT> \
  --max-cost 0 --fee-sats 1000

# View all your share positions
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-share-positions

# Redeem winning shares after resolution
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 redeem-shares \
  --market-id <MARKET_ID> --outcome-index <WINNING_INDEX> \
  --shares-amount <AMOUNT> --fee-sats 1000
```

### Understanding Market Output

When you view a market, you'll see:
- **Beta Parameter**: The liquidity sensitivity (e.g., 910239.23)
  - Derived from initial liquidity: β = Capital / ln(n)
  - For binary markets with 1M sats: β = 1,000,000 / ln(2) ≈ 1,442,695
- **Prices**: Current price per share (0.0 to 1.0)
- **Probability**: Market's implied probability (price × 100%)
- **Treasury**: Total capital in the market maker
- **Volume**: Total sats traded per outcome

### Complete Market Trading Workflow Example

Using a real market (e.g., "Steelers Super Bowl 60" with ID 4634fd2999da):

```bash
# 1. View current market state and prices
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market 4634fd2999da
# Shows: Both outcomes at 50% probability, 1,000,000 sats liquidity

# 2. Calculate cost before buying shares (outcome 1 = Yes to Steelers winning)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 calculate-share-cost \
  --market-id 4634fd2999da \
  --outcome-index 1 \
  --shares-amount 100.0
# Returns estimated cost in sats

# 3. Buy 100 shares of "Yes" (outcome 1)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id 4634fd2999da \
  --outcome-index 1 \
  --shares-amount 100.0 \
  --max-cost 75000 \
  --fee-sats 1000

# 4. Mine a block to confirm the trade
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000

# 5. View updated market - prices have changed!
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market 4634fd2999da
# Now shows: Yes probability > 50%, No probability < 50%

# 6. Check your share positions
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-share-positions
# Shows: 100 shares of outcome 1 in market 4634fd2999da

# 7. Sell shares back to the market (negative amount = sell)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id 4634fd2999da \
  --outcome-index 1 \
  --shares-amount -50.0 \
  --max-cost 0 \
  --fee-sats 1000

# 8. Mine and check prices return toward equilibrium
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market 4634fd2999da
```

### Understanding LMSR Price Dynamics

```bash
# The more shares you buy of an outcome, the higher its price becomes
# Example: Progressive purchases showing price increase

# Initial state - check base price
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market 4634fd2999da

# Buy 10 shares - small price movement
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id 4634fd2999da --outcome-index 0 --shares-amount 10.0 \
  --max-cost 10000 --fee-sats 1000

# Buy 100 shares - larger price movement  
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id 4634fd2999da --outcome-index 0 --shares-amount 100.0 \
  --max-cost 100000 --fee-sats 1000

# Buy 1000 shares - significant price movement
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id 4634fd2999da --outcome-index 0 --shares-amount 1000.0 \
  --max-cost 1000000 --fee-sats 1000

# Check final prices - outcome 0 is now expensive!
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market 4634fd2999da
```

### Troubleshooting Common Trading Issues

```bash
# Issue: "Market not found"
# Solution: List all markets to get correct ID
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-markets

# Issue: "Insufficient funds" when buying shares
# Solution: Check your balance and mine more blocks
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-balance
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000

# Issue: "Max cost exceeded"
# Solution: Either increase --max-cost or buy fewer shares
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 calculate-share-cost \
  --market-id 4634fd2999da --outcome-index 1 --shares-amount 50.0

# Issue: Selling shares (how to do it)
# Solution: Use negative shares-amount to sell
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id 4634fd2999da --outcome-index 1 --shares-amount -10.0 \
  --max-cost 0 --fee-sats 1000
```

### Share Trading JSON-RPC

```bash
# Buy shares via JSON-RPC
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", 
    "method": "buy_shares", 
    "params": {
      "market_id": "1a2b3c4d",
      "outcome_index": 0,
      "shares_amount": 50.0,
      "max_cost": 30000,
      "fee_sats": 1000
    }, 
    "id": 1
  }'

# Get share positions
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "get_share_positions", "params": [], "id": 1}'

# Calculate share cost before buying
curl -X POST http://127.0.0.1:18332 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", 
    "method": "calculate_share_cost", 
    "params": {
      "market_id": "1a2b3c4d",
      "outcome_index": 0,
      "shares_amount": 50.0
    }, 
    "id": 1
  }'
```

## Real-time Market State Monitoring

You can monitor market state changes in real-time after each transaction:

### System Status and Monitoring

```bash
# Check overall node status (aliases: stat, s)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 status

# Monitor mempool for pending transactions
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 my-unconfirmed-utxos

# Check current block count and sync status
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-block-count

# View all transactions in mempool
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 list-utxos
```

### Market State After Transactions

After each share trade or market transaction:

```bash
# 1. Submit a share purchase transaction
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id MARKET_ID_HEX \
  --outcome-index 0 \
  --shares-amount 50.0 \
  --max-cost 25000 \
  --fee-sats 1000

# 2. Immediately check updated market state (prices reflect the new trade)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 view-market MARKET_ID_HEX

# 3. View your updated share positions
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 get-share-positions

# 4. Check if transaction is still in mempool
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 my-unconfirmed-utxos

# 5. Mine a block to confirm the transaction
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000
```

### Transaction Validation and State Updates

Every transaction in the mempool:
- **Pays a fee** (specified with `--fee-sats`) 
- **Gets logged for monitoring** when added to mempool (full state updates occur during block processing)
- **Gets validated** against current UTXO state
- **Forms chains** with previous transactions via UTXO references (OutPoint structure)

Currently uses **first-come-first-served** selection from mempool for block inclusion:

```bash
# Example: Multiple competing transactions
# Transaction A: buy 100 shares, fee = 2000 sats
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id MARKET_ID_HEX --outcome-index 0 --shares-amount 100.0 \
  --max-cost 50000 --fee-sats 2000

# Transaction B: buy 75 shares, fee = 1500 sats  
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 buy-shares \
  --market-id MARKET_ID_HEX --outcome-index 1 --shares-amount 75.0 \
  --max-cost 40000 --fee-sats 1500

# Check which transactions are in mempool
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 my-unconfirmed-utxos

# Mine block - includes transactions in mempool order (first-come-first-served)
./target/debug/truthcoin_dc_app_cli --rpc-port 18332 mine --fee-sats 1000

# Market state changes are applied during block processing, not at mempool addition
```

This market demonstrates the full power of Bitcoin Hivemind's multidimensional prediction market capabilities with proper logical constraints and efficient LMSR pricing.