#!/bin/bash
# Test script for recursive deep search

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
KATAGO_EXE="$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
MODEL_PATH="$REPO_ROOT/katago_repo/run/default_model.bin.gz"
CONFIG_PATH="$REPO_ROOT/katago_repo/run/gtp_800visits.cfg"
PYTHON_BIN="$PARENT_ROOT/Go_env/bin/python"
ACTIVATE="$PARENT_ROOT/Go_env/bin/activate"

cd "$PROJECT_DIR"

# Activate environment
if [ -f "$ACTIVATE" ]; then
  # shellcheck source=/dev/null
  source "$ACTIVATE"
fi

# Run a short test game
"$PYTHON_BIN" run_datago_recursive_match.py \
    --katago-executable "$KATAGO_EXE" \
    --katago-model "$MODEL_PATH" \
    --katago-config "$CONFIG_PATH" \
    --config "src/bot/config.yaml" \
    --games 1 \
    --max-moves 40
