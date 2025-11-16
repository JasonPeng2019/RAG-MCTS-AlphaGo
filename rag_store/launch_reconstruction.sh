#!/bin/bash

# Parallel RAG Position Reconstruction Launcher
# Runs deep MCTS analysis on 5 GPUs simultaneously

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../../venv/bin/activate

# GPU assignments (based on nvidia-smi availability)
declare -a GPUS=(7 5 3 1 2)

echo "Starting parallel reconstruction on 5 GPUs..."
echo "============================================"
echo ""

for i in {1..5}; do
    GPU=${GPUS[$i-1]}
    SESSION_NAME="rag-recon-$i"
    CSV_FILE="rag_files_list_$i.csv"
    JSON_DIR="../../build/rag_data_$i"
    OUTPUT_DIR="rag_output_$i"
    
    echo "Launching reconstructor $i on GPU $GPU"
    echo "  Input: $JSON_DIR"
    echo "  Output: $OUTPUT_DIR"
    
    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    
    # Create new tmux session and run analyzer
    tmux new-session -d -s "$SESSION_NAME" bash -c "
        cd '$SCRIPT_DIR' && \
        source ../../venv/bin/activate && \
        CUDA_VISIBLE_DEVICES=$GPU python game_analyzer.py \
            --csv '$CSV_FILE' \
            --json-dir '$JSON_DIR' \
            --output-dir '$OUTPUT_DIR' \
            --max-visits 3800 \
            --katago-path '../../build/katago' \
            --config '../../katago_repo/run/analysis.cfg' \
            --model '../../katago_repo/run/kata1-b28c512nbt-s11653980416-d5514111622.bin.gz'; \
        echo ''; \
        echo '==================================='; \
        echo 'Reconstruction $i completed!'; \
        echo 'Press ENTER to close session'; \
        read
    "
    
    echo "  ✓ Started session: $SESSION_NAME"
    echo ""
    sleep 1
done

echo "============================================"
echo "✓ All 5 reconstruction sessions launched!"
echo ""
echo "Monitor with:"
echo "  tmux ls"
echo "  tmux attach -t rag-recon-1"
echo ""
echo "Check progress:"
echo "  for i in {1..5}; do tmux capture-pane -t rag-recon-\$i -p | grep 'Processing position' | tail -1; done"
