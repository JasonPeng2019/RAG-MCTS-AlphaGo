#!/bin/bash
# test_katago_gpu7.sh
# Quick test to verify KataGo works on GPU 7

set -euo pipefail

echo "Testing KataGo on GPU 7..."
echo "================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

"$REPO_ROOT/Go_env/bin/python" test_gtp_katago.py 2>&1 | tail -30
