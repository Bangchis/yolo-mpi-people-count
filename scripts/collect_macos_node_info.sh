#!/usr/bin/env bash
set -euo pipefail

echo "MACOS_NODE_INFO"
date -u +"utc_time=%Y-%m-%dT%H:%M:%SZ"
echo "hostname=$(hostname)"
echo "user=$(whoami)"
echo "os=$(sw_vers -productName 2>/dev/null || uname -s) $(sw_vers -productVersion 2>/dev/null || true)"
echo "kernel=$(uname -a)"
echo "physical_cpu=$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
echo "logical_cpu=$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
echo "machine=$(uname -m)"
echo "ssh_remote_login=$(sudo systemsetup -getremotelogin 2>/dev/null || true)"
echo "mpirun=$(command -v mpirun || true)"
mpirun --version 2>/dev/null | head -n 2 || true
echo "python=$(command -v python3 || true)"
python3 --version 2>/dev/null || true
echo "ipv4_addresses:"
ifconfig | awk '/inet / && $2 != "127.0.0.1" {print "  " $2}' || true

