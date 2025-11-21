#!/bin/bash
# test_katago_gpu7.sh
# Quick test to verify KataGo works on GPU 7

set -euo pipefail

echo "Testing KataGo on GPU 7..."
echo "================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$PARENT_ROOT/Go_env/bin/python"

"$PYTHON_BIN" test_gtp_katago.py 2>&1 | tail -30
