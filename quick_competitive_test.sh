#!/bin/bash
# Quick competitive test: 3 games to validate performance advantage
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
KATAGO_EXE="$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
MODEL_PATH="$REPO_ROOT/katago_repo/run/default_model.bin.gz"
CONFIG_PATH="$REPO_ROOT/katago_repo/run/gtp_800visits.cfg"
PYTHON_BIN="$PARENT_ROOT/Go_env/bin/python"
VENV_ACTIVATE="$PARENT_ROOT/Go_env/bin/activate"

echo "======================================================================"
echo "DataGo vs KataGo - Quick Test (3 games)"
echo "======================================================================"
echo ""

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

# Run 3 quick games
"$PYTHON_BIN" run_datago_recursive_match.py \
    --katago-executable "$KATAGO_EXE" \
    --katago-model "$MODEL_PATH" \
    --katago-config "$CONFIG_PATH" \
    --config "src/bot/config.yaml" \
    --games 3 \
    --max-moves 100 \
    2>&1 | tee quick_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Quick test complete!"
