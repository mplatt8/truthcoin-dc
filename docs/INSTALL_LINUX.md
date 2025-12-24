# Linux Installation Guide

## Prerequisites

### 1. Install Git

**Debian/Ubuntu:**
```bash
sudo apt install git
```

**Fedora:**
```bash
sudo dnf install git
```

**Arch Linux:**
```bash
sudo pacman -S git
```

Verify installation:
```bash
git --version
```

### 2. Install grpcurl

grpcurl is required for interacting with the BIP300301 enforcer gRPC service.

```bash
# Download latest release (adjust version as needed)
curl -sSL https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz | tar -xz
sudo mv grpcurl /usr/local/bin/
```

Verify installation:
```bash
grpcurl --version
```

### 3. Install Rust, Cargo, and Rustup

Rust and Cargo are required to build truthcoin and electrs from source.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Install nightly toolchain (required):
```bash
rustup install nightly
rustup default nightly
```

Verify installation:
```bash
rustc --version
cargo --version
```

## Download and Setup Binaries

### 1. Create Directory Structure

```bash
mkdir truthcoin-binaries
cd truthcoin-binaries
```

### 2. Download Pre-built Binaries

Download pre-built binaries from [releases.drivechain.info](https://releases.drivechain.info):
- `L1-bitcoin-patched-latest-x86_64-unknown-linux-gnu.zip`
- `bip300301-enforcer-latest-x86_64-unknown-linux-gnu.zip`

### 3. Rename and Organize

```bash
mv ~/Downloads/L1-bitcoin-patched-latest-x86_64-unknown-linux-gnu ./bitcoin-patched
mv ~/Downloads/bip300301-enforcer-latest-x86_64-unknown-linux-gnu ./bip300301_enforcer
```

Rename the enforcer binary:
```bash
mv ./bip300301_enforcer/bip300301-enforcer-latest-x86_64-unknown-linux-gnu ./bip300301_enforcer/bip300301_enforcer
```

### 4. Make Binaries Executable

```bash
chmod +x ./bip300301_enforcer/bip300301_enforcer
chmod +x ./bitcoin-patched/bitcoind
chmod +x ./bitcoin-patched/bitcoin-cli
```

### 5. Build from Source

```bash
# Electrs (Blockstream fork with HTTP/REST API)
git clone https://github.com/blockstream/electrs.git
cd electrs
cargo build --release
cd ..

# Truthcoin
git clone https://github.com/LayerTwo-Labs/truthcoin-dc.git
cd truthcoin-dc
git submodule update --init --recursive
cargo build
```

---

[Back to main README](../README.md)
