# Windows Installation Guide

## Prerequisites

### 1. Install Git

Download and install from [git-scm.com](https://git-scm.com/download/win)

During installation, select the following recommended options:
- Use Git from the Windows Command Prompt
- Use OpenSSH
- Checkout Windows-style, commit Unix-style line endings
- Use MinTTY

Verify installation (open new terminal):
```powershell
git --version
```

### 2. Install grpcurl

grpcurl is required for interacting with the BIP300301 enforcer gRPC service.

**Option A: Using Chocolatey (recommended)**

If you have Chocolatey installed:
```powershell
choco install grpcurl
```

**Option B: Manual Installation**

1. Download `grpcurl_X.X.X_windows_x86_64.zip` from [GitHub releases](https://github.com/fullstorydev/grpcurl/releases)
2. Extract the zip file to a folder (e.g., `C:\tools\grpcurl`)
3. Add the folder to your PATH:
   - Open System Properties > Advanced > Environment Variables
   - Under System Variables, find `Path` and click Edit
   - Add the folder path containing `grpcurl.exe`

Verify installation (open new terminal):
```powershell
grpcurl --version
```

### 3. Install Rust, Cargo, and Rustup

Download and run the installer from [rustup.rs](https://rustup.rs)

Follow the on-screen instructions. You may need to install the Visual Studio C++ Build Tools if prompted.

Install nightly toolchain (required):
```powershell
rustup install nightly
rustup default nightly
```

Verify installation:
```powershell
rustc --version
cargo --version
```

### 4. Install Visual Studio Build Tools

If not already installed, download the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

During installation, select:
- "Desktop development with C++"

## Download and Setup Binaries

### 1. Create Directory Structure

```powershell
mkdir truthcoin-binaries
cd truthcoin-binaries
```

### 2. Download Pre-built Binaries

Download pre-built binaries from [releases.drivechain.info](https://releases.drivechain.info):
- `L1-bitcoin-patched-latest-x86_64-pc-windows-msvc.zip`
- `bip300301-enforcer-latest-x86_64-pc-windows-msvc.zip`

### 3. Extract and Organize

Extract the downloaded zip files and rename:

```powershell
Rename-Item -Path "L1-bitcoin-patched-latest-x86_64-pc-windows-msvc" -NewName "bitcoin-patched"
Rename-Item -Path "bip300301-enforcer-latest-x86_64-pc-windows-msvc" -NewName "bip300301_enforcer"
```

Rename the enforcer binary:
```powershell
Rename-Item -Path ".\bip300301_enforcer\bip300301-enforcer-latest-x86_64-pc-windows-msvc.exe" -NewName "bip300301_enforcer.exe"
```

### 4. Build from Source

```powershell
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

## Windows-Specific Notes

- Use PowerShell or Git Bash for running commands
- Replace `/tmp/regtest-data` with `%TEMP%\regtest-data` or a custom directory as specified in the master README
- File paths use backslashes (`\`) in Windows, but most tools accept forward slashes (`/`)

---

[Back to main README](../README.md)
