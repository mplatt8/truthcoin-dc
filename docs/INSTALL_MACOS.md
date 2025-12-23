# macOS Installation Guide

## Prerequisites

### 1. Install Xcode Command Line Tools

Required for compiling code on macOS. This must be installed first.

```bash
xcode-select --install
```

Follow the prompts to complete installation.

### 2. Install Homebrew

Homebrew is a package manager for macOS used to install dependencies.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, follow the instructions in the terminal to add Homebrew to your PATH.

Verify installation:
```bash
brew --version
```

### 3. Install Git

```bash
brew install git
```

Verify installation:
```bash
git --version
```

### 4. Install grpcurl

Open a new terminal session.

grpcurl is required for interacting with the BIP300301 enforcer gRPC service.

```bash
brew install grpcurl
```

Verify installation:
```bash
grpcurl --version
```

### 5. Install Rust, Cargo, and Rustup

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
- `L1-bitcoin-patched-latest-x86_64-apple-darwin.zip`
- `bip300301-enforcer-latest-x86_64-apple-darwin.zip`

### 3. Rename and Organize

```bash
mv ~/Downloads/L1-bitcoin-patched-latest-x86_64-apple-darwin ./bitcoin-patched
mv ~/Downloads/bip300301-enforcer-latest-x86_64-apple-darwin ./bip300301_enforcer
```

Rename the enforcer binary:
```bash
mv ./bip300301_enforcer/bip300301-enforcer-latest-x86_64-apple-darwin ./bip300301_enforcer/bip300301_enforcer
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
git clone https://github.com/mplatt8/truthcoin-dc.git
cd truthcoin-dc
git submodule update --init --recursive
cargo build
```

---

[Back to main README](../README.md)
