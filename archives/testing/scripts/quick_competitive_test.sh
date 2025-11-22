#!/bin/bash
# Quick competitive test: 3 games to validate performance advantage
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$REPO_ROOT"
LOG_DIR="$SCRIPT_DIR"
KATAGO_EXE="$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
MODEL_PATH="$REPO_ROOT/katago_repo/run/default_model.bin.gz"
CONFIG_PATH="$REPO_ROOT/katago_repo/run/gtp_800visits.cfg"
PYTHON_BIN="$REPO_ROOT/Go_env/bin/python"
VENV_ACTIVATE="$REPO_ROOT/Go_env/bin/activate"

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
"$PYTHON_BIN" "$PROJECT_DIR/run_datago_recursive_match.py" \
    --katago-executable "$KATAGO_EXE" \
    --katago-model "$MODEL_PATH" \
    --katago-config "$CONFIG_PATH" \
    --config "src/bot/config.yaml" \
    --games 3 \
    --max-moves 100 \
    2>&1 | tee "$LOG_DIR/quick_test_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Quick test complete!"
