#!/bin/bash
# Test script for recursive deep search
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate environment
source "$REPO_ROOT/Go_env/bin/activate"

# Run a short test game
python run_datago_recursive_match.py \
    --katago-executable "$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago" \
    --katago-model "$REPO_ROOT/katago_repo/run/default_model.bin.gz" \
    --katago-config "$REPO_ROOT/katago_repo/run/gtp_800visits.cfg" \
    --config "src/bot/config.yaml" \
    --games 1 \
    --max-moves 40
