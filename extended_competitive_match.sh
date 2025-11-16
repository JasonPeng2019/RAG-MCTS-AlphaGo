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

cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"
source /scratch2/f004ndc/AlphaGo\ Project/Go_env/bin/activate

python3 run_datago_recursive_match.py \
    --katago-executable "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago" \
    --katago-model "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz" \
    --katago-config "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg" \
    --config "src/bot/config.yaml" \
    --games 10 \
    --max-moves 100 \
    2>&1 | tee extended_match_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================"
echo "Extended match complete!"
echo "======================================================================"
