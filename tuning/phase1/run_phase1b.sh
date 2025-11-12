#!/bin/bash
# Launch script for Phase 1b storage threshold tuning
# Usage: ./run_phase1b.sh

echo "=========================================="
echo "RAG-AlphaGo Phase 1b: Storage Threshold Tuning"
echo "=========================================="
echo ""

# Configuration
PHASE1_CONFIG="./tuning_results/phase1/best_config_phase1.json"
OUTPUT_DIR="./tuning_results/phase1b"
NUM_GAMES=100
MAX_DB_SIZE=20

# Check if Phase 1 config exists
if [ ! -f "$PHASE1_CONFIG" ]; then
    echo "Error: Phase 1 config not found at $PHASE1_CONFIG"
    echo "Please run Phase 1a first: ./run_phase1.sh"
    exit 1
fi

echo "Using Phase 1 config: $PHASE1_CONFIG"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start monitoring in background
echo "Starting monitoring dashboard..."
python monitor.py \
    --mode monitor \
    --results-dir ./tuning_results \
    --phase phase1b \
    --refresh 10 &

MONITOR_PID=$!
echo "Monitor started (PID: $MONITOR_PID)"
echo ""

# Give monitor time to start
sleep 2

# Start Phase 1b tuning
echo "Starting Phase 1b tuning..."
echo "Configuration:"
echo "  Output dir: $OUTPUT_DIR"
echo "  Games per threshold: $NUM_GAMES"
echo "  Max database size: ${MAX_DB_SIZE}GB"
echo ""
echo "This will take approximately 8-10 hours."
echo ""

python phase1b_storage_threshold.py \
    --phase1-config "$PHASE1_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --num-games $NUM_GAMES \
    --max-db-size $MAX_DB_SIZE

TUNING_EXIT_CODE=$?

# Stop monitoring
echo ""
echo "Stopping monitor..."
kill $MONITOR_PID 2>/dev/null

if [ $TUNING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase 1b completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR/storage_threshold_results.json"
    echo "Plots saved to:"
    echo "  - $OUTPUT_DIR/uncertainty_distribution.png"
    echo "  - $OUTPUT_DIR/threshold_comparison.png"
    echo ""
    echo "Phase 1 (complete) is now finished!"
    echo ""
    echo "Next steps:"
    echo "1. Review results and plots"
    echo "2. Proceed to Phase 2: Deep MCTS and Recursion Control tuning"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Error: Phase 1b failed with exit code $TUNING_EXIT_CODE"
    echo "=========================================="
    echo ""
    exit $TUNING_EXIT_CODE
fi
