#!/bin/bash
# Extended competitive match: 10 games
set -e

echo "======================================================================"
echo "DataGo vs KataGo - Extended Match (10 games)"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  DataGo:  800 visits (standard) + 2000 visits (deep search)"
echo "  KataGo:  800 visits (constant)"
echo "  Games:   10"
echo "  Max moves: 100"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
source "$REPO_ROOT/Go_env/bin/activate"

python3 "$REPO_ROOT/run_datago_recursive_match.py" \
    --katago-executable "$REPO_ROOT/katago_repo/KataGo/cpp/build-eigen/katago" \
    --katago-model "$REPO_ROOT/katago_repo/run/default_model.bin.gz" \
    --katago-config "$REPO_ROOT/katago_repo/run/default_gtp.cfg" \
    --config "src/bot/config.yaml" \
    --games 10 \
    --max-moves 100 \
    2>&1 | tee extended_match_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================"
echo "Extended match complete!"
echo "======================================================================"
