#!/bin/bash
# Launch parallel RAG analysis on 5 GPUs
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_ACTIVATE="$REPO_ROOT/Go_env/bin/activate"
JSON_ROOT="$REPO_ROOT/build"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "ERROR: Go_env virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

if [ ! -d "$JSON_ROOT" ]; then
  echo "WARNING: JSON root $JSON_ROOT does not exist. Update JSON_ROOT in launch_parallel_analysis.sh to point at your rag_data splits." >&2
fi

source "$VENV_ACTIVATE"
cd "$SCRIPT_DIR"

echo "Starting parallel RAG analysis on 5 GPUs with 10,000 visits per position..."
echo ""

launch_split() {
  local session="$1"
  local gpu_id="$2"
  local csv_file="$3"
  local split_id="$4"
  local json_dir="$JSON_ROOT/rag_data_${split_id}"
  local output_dir="./rag_output_${split_id}"

  tmux new -d -s "$session" bash -c "source '$VENV_ACTIVATE'; cd '$SCRIPT_DIR'; CUDA_VISIBLE_DEVICES=$gpu_id python game_analyzer.py --csv '$csv_file' --json-dir '$json_dir' --output-dir '$output_dir' --max-visits 10000"
  echo "✓ Started $session on GPU $gpu_id (split $split_id -> $csv_file)"
}

launch_split "rag-analyzer-1" 1 "rag_files_list_1.csv" 1
launch_split "rag-analyzer-2" 2 "rag_files_list_2.csv" 2
launch_split "rag-analyzer-3" 3 "rag_files_list_3.csv" 3
launch_split "rag-analyzer-4" 5 "rag_files_list_4.csv" 4
launch_split "rag-analyzer-5" 7 "rag_files_list_5.csv" 5

echo ""
echo "All analyzers launched!"
echo ""
echo "Monitor with:"
echo "  tmux ls | grep rag-analyzer"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Attach to a session:"
echo "  tmux attach -t rag-analyzer-1"
echo ""
echo "Stop all analyzers:"
echo "  for i in {1..5}; do tmux kill-session -t rag-analyzer-\$i; done"
