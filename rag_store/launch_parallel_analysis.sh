#!/bin/bash
# Launch parallel RAG analysis on 5 GPUs using the shared Go_env

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
GO_ENV_ACTIVATE="$WORKSPACE_ROOT/Go_env/bin/activate"

cd "$SCRIPT_DIR"

if [ ! -f "$GO_ENV_ACTIVATE" ]; then
  echo "Go_env not found at $GO_ENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$GO_ENV_ACTIVATE"

echo "Starting parallel RAG analysis on 5 GPUs with 10,000 visits per position..."
echo ""

declare -a GPUS=(1 2 3 5 7)

for i in {1..5}; do
  GPU="${GPUS[$((i-1))]}"
  SESSION_NAME="rag-analyzer-$i"
  CSV_FILE="rag_files_list_${i}.csv"
  JSON_DIR="../../build/rag_data_${i}"
  OUTPUT_DIR="./rag_output_${i}"

  tmux kill-session -t "$SESSION_NAME" 2>/dev/null

  tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd '$SCRIPT_DIR' && \
    source '$GO_ENV_ACTIVATE' && \
    CUDA_VISIBLE_DEVICES=$GPU python game_analyzer.py \
      --csv '$CSV_FILE' \
      --json-dir '$JSON_DIR' \
      --output-dir '$OUTPUT_DIR' \
      --max-visits 10000
  "
  echo \"✓ Started $SESSION_NAME on GPU $GPU\"
done

echo ""
echo "All analyzers launched!"
echo ""
echo "Monitor with: tmux ls | grep rag-analyzer"
echo "Attach via: tmux attach -t rag-analyzer-1"
echo "Stop all: for i in {1..5}; do tmux kill-session -t rag-analyzer-\$i; done"
