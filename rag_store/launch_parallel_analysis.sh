#!/bin/bash
# Launch parallel RAG analysis on 5 GPUs

# Activate virtual environment
source /scratch2/f004h1v/alphago_project/venv/bin/activate
cd /scratch2/f004h1v/alphago_project/datago/rag_store

echo "Starting parallel RAG analysis on 5 GPUs with 10,000 visits per position..."
echo ""

# GPU 1 - Split 1 (200 files)
tmux new -d -s rag-analyzer-1 bash -c "
source /scratch2/f004h1v/alphago_project/venv/bin/activate
cd /scratch2/f004h1v/alphago_project/datago/rag_store
CUDA_VISIBLE_DEVICES=1 python game_analyzer.py --csv rag_files_list_1.csv --json-dir ../../build/rag_data_1 --output-dir ./rag_output_1 --max-visits 10000
"
echo "✓ Started analyzer-1 on GPU 1 (200 files)"

# GPU 2 - Split 2 (200 files)
tmux new -d -s rag-analyzer-2 bash -c "
source /scratch2/f004h1v/alphago_project/venv/bin/activate
cd /scratch2/f004h1v/alphago_project/datago/rag_store
CUDA_VISIBLE_DEVICES=2 python game_analyzer.py --csv rag_files_list_2.csv --json-dir ../../build/rag_data_2 --output-dir ./rag_output_2 --max-visits 10000
"
echo "✓ Started analyzer-2 on GPU 2 (200 files)"

# GPU 3 - Split 3 (200 files)
tmux new -d -s rag-analyzer-3 bash -c "
source /scratch2/f004h1v/alphago_project/venv/bin/activate
cd /scratch2/f004h1v/alphago_project/datago/rag_store
CUDA_VISIBLE_DEVICES=3 python game_analyzer.py --csv rag_files_list_3.csv --json-dir ../../build/rag_data_3 --output-dir ./rag_output_3 --max-visits 10000
"
echo "✓ Started analyzer-3 on GPU 3 (200 files)"

# GPU 5 - Split 4 (200 files)
tmux new -d -s rag-analyzer-4 bash -c "
source /scratch2/f004h1v/alphago_project/venv/bin/activate
cd /scratch2/f004h1v/alphago_project/datago/rag_store
CUDA_VISIBLE_DEVICES=5 python game_analyzer.py --csv rag_files_list_4.csv --json-dir ../../build/rag_data_4 --output-dir ./rag_output_4 --max-visits 10000
"
echo "✓ Started analyzer-4 on GPU 5 (200 files)"

# GPU 7 - Split 5 (171 files)
tmux new -d -s rag-analyzer-5 bash -c "
source /scratch2/f004h1v/alphago_project/venv/bin/activate
cd /scratch2/f004h1v/alphago_project/datago/rag_store
CUDA_VISIBLE_DEVICES=7 python game_analyzer.py --csv rag_files_list_5.csv --json-dir ../../build/rag_data_5 --output-dir ./rag_output_5 --max-visits 10000
"
echo "✓ Started analyzer-5 on GPU 7 (171 files)"

echo ""
echo "All 5 analyzers launched!"
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
