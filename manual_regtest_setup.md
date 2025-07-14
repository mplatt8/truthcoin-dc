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

```bash
./target/debug/truthcoin_dc_app \
  --datadir=/tmp/regtest-data/truthcoin \
  --network=regtest \
  --mainchain-grpc-port=50051 \
  --net-addr=127.0.0.1:18445 \
  --rpc-port=18332 \
  --zmq-addr=127.0.0.1:28333
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

Get a Truthcoin address from your GUI, then:

```bash
# Create deposit (replace ADDRESS with your Truthcoin address)
grpcurl -plaintext -d '{
  "sidechain_id": 13,
  "address": "YOUR_TRUTHCOIN_ADDRESS",
  "value_sats": 21000000,
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