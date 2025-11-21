#!/bin/bash
# Run competitive match: DataGo vs KataGo
# DataGo uses 800 visits standard + 2000 visits deep search
# KataGo uses 800 visits constant

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
KATAGO_EXE="$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
MODEL_PATH="$REPO_ROOT/katago_repo/run/default_model.bin.gz"
CONFIG_PATH="$REPO_ROOT/katago_repo/run/gtp_800visits.cfg"
GO_ENV_DIR="$PARENT_ROOT/Go_env"
PYTHON_BIN="$GO_ENV_DIR/bin/python"
VENV_ACTIVATE="$GO_ENV_DIR/bin/activate"

echo "======================================================================"
echo "DataGo vs KataGo Competitive Match"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  DataGo:  800 visits (standard) + 2000 visits (deep search on 5% moves)"
echo "  KataGo:  800 visits (constant)"
echo "  Games:   10"
echo "  Max moves per game: 200"
echo ""
echo "Expected: DataGo should win more due to adaptive deep search"
echo "======================================================================"
echo ""

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
else
  echo "ERROR: Go_env virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

# Run the match
"$PYTHON_BIN" run_datago_recursive_match.py \
    --katago-executable "$KATAGO_EXE" \
    --katago-model "$MODEL_PATH" \
    --katago-config "$CONFIG_PATH" \
    --config "src/bot/config.yaml" \
    --games 10 \
    --max-moves 200 \
    2>&1 | tee competitive_match_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================"
echo "Match complete! Check the log file for detailed results."
echo "======================================================================"
