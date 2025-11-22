#!/bin/bash

# Parallel RAG Position Reconstruction Launcher
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_ACTIVATE="$REPO_ROOT/Go_env/bin/activate"
JSON_ROOT="$REPO_ROOT/build"
KATAGO_PATH="$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
KATAGO_CONFIG="$REPO_ROOT/katago_repo/run/analysis.cfg"
KATAGO_MODEL="$REPO_ROOT/katago_repo/run/default_model.bin.gz"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "ERROR: Go_env virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

if [ ! -d "$JSON_ROOT" ]; then
  echo "WARNING: JSON root $JSON_ROOT does not exist. Update JSON_ROOT in launch_reconstruction.sh to point at your rag_data splits." >&2
fi

source "$VENV_ACTIVATE"
cd "$SCRIPT_DIR"

declare -a GPUS=(7 5 3 1 2)

echo "Starting parallel reconstruction on 5 GPUs..."
echo "============================================"
echo ""

for i in {1..5}; do
    GPU=${GPUS[$i-1]}
    SESSION_NAME="rag-recon-$i"
    CSV_FILE="rag_files_list_${i}.csv"
    JSON_DIR="$JSON_ROOT/rag_data_${i}"
    OUTPUT_DIR="rag_output_${i}"

    echo "Launching reconstructor $i on GPU $GPU"
    echo "  Input: $JSON_DIR"
    echo "  Output: $OUTPUT_DIR"

    tmux kill-session -t "$SESSION_NAME" 2>/dev/null

    tmux new-session -d -s "$SESSION_NAME" bash -c "
        source '$VENV_ACTIVATE';
        cd '$SCRIPT_DIR';
        CUDA_VISIBLE_DEVICES=$GPU python game_analyzer.py \
            --csv '$CSV_FILE' \
            --json-dir '$JSON_DIR' \
            --output-dir '$OUTPUT_DIR' \
            --max-visits 10000 \
            --katago-path '$KATAGO_PATH' \
            --config '$KATAGO_CONFIG' \
            --model '$KATAGO_MODEL';
        echo '';
        echo '===================================';
        echo 'Reconstruction $i completed!';
        echo 'Press ENTER to close session';
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
