# Parallel RAG Analysis on Multiple GPUs

## Data Split Summary

Your 971 RAG JSON files have been split into 5 folders:
- `rag_data_1`: 200 files
- `rag_data_2`: 200 files
- `rag_data_3`: 200 files
- `rag_data_4`: 200 files
- `rag_data_5`: 171 files

Each has a corresponding CSV: `rag_files_list_1.csv` through `rag_files_list_5.csv`

## Launch All 5 GPUs in Parallel

```bash
# First, activate Go_env
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store

# Launch all 5 analyzers in parallel on GPUs 1, 2, 3, 5, 7
for i in 1 2 3 5 7; do
    gpu_id=$i
    if [ $i -eq 1 ]; then split_id=1; fi
    if [ $i -eq 2 ]; then split_id=2; fi
    if [ $i -eq 3 ]; then split_id=3; fi
    if [ $i -eq 5 ]; then split_id=4; fi
    if [ $i -eq 7 ]; then split_id=5; fi
    
    tmux new -d -s "rag-analyzer-${split_id}" bash -c "
        source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
        cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
        CUDA_VISIBLE_DEVICES=${gpu_id} python game_analyzer.py \
          --csv rag_files_list_${split_id}.csv \
          --json-dir ../../build/rag_data_${split_id} \
          --output-dir ./rag_output_${split_id} \
          --max-visits 2400
    "
    echo "Started analyzer for split ${split_id} on GPU ${gpu_id} (session: rag-analyzer-${split_id})"
done
```

## Simpler Copy-Paste Version

```bash
# Activate Go_env first
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store

# GPU 1 - Split 1
tmux new -d -s rag-analyzer-1 bash -c "
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
CUDA_VISIBLE_DEVICES=1 python game_analyzer.py --csv rag_files_list_1.csv --json-dir ../../build/rag_data_1 --output-dir ./rag_output_1 --max-visits 2400
"

# GPU 2 - Split 2
tmux new -d -s rag-analyzer-2 bash -c "
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
CUDA_VISIBLE_DEVICES=2 python game_analyzer.py --csv rag_files_list_2.csv --json-dir ../../build/rag_data_2 --output-dir ./rag_output_2 --max-visits 2400
"

# GPU 3 - Split 3
tmux new -d -s rag-analyzer-3 bash -c "
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
CUDA_VISIBLE_DEVICES=3 python game_analyzer.py --csv rag_files_list_3.csv --json-dir ../../build/rag_data_3 --output-dir ./rag_output_3 --max-visits 2400
"

# GPU 5 - Split 4
tmux new -d -s rag-analyzer-4 bash -c "
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
CUDA_VISIBLE_DEVICES=5 python game_analyzer.py --csv rag_files_list_4.csv --json-dir ../../build/rag_data_4 --output-dir ./rag_output_4 --max-visits 2400
"

# GPU 7 - Split 5
tmux new -d -s rag-analyzer-5 bash -c "
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
CUDA_VISIBLE_DEVICES=7 python game_analyzer.py --csv rag_files_list_5.csv --json-dir ../../build/rag_data_5 --output-dir ./rag_output_5 --max-visits 2400
"

echo "All 5 analyzers started!"
```

## Monitor Progress

```bash
# List all analyzer sessions
tmux ls | grep rag-analyzer

# Attach to specific analyzer
tmux attach -t rag-analyzer-1

# Check GPU usage
watch -n 1 nvidia-smi

# Check output sizes
watch -n 10 "for i in {1..5}; do [ -f rag_output_\$i/rag_database.json ] && echo \"Split \$i: \$(ls -lh rag_output_\$i/rag_database.json | awk '{print \$5}')\"; done"

# Peek at progress without attaching
for i in {1..5}; do 
    echo "=== Analyzer $i ==="
    tmux capture-pane -t rag-analyzer-$i -p | tail -5
done
```

## Stop All Analyzers

```bash
for i in {1..5}; do
    tmux kill-session -t rag-analyzer-$i
    echo "Killed rag-analyzer-$i"
done
```

## Merge Results After Completion

Once all analyzers finish, merge the 5 output files:

```bash
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store

# Merge all JSON databases into one
python3 << 'EOF'
import json

all_positions = []
for i in range(1, 6):
    with open(f'rag_output_{i}/rag_database.json', 'r') as f:
        positions = json.load(f)
        all_positions.extend(positions)
        print(f"Loaded {len(positions)} positions from split {i}")

print(f"\nTotal: {len(all_positions)} positions")

with open('rag_output_merged/rag_database_complete.json', 'w') as f:
    json.dump(all_positions, f, indent=2)

print(f"Saved complete database to rag_output_merged/rag_database_complete.json")
EOF
```

## Estimated Time

With 2400 visits per position and 5 GPUs in parallel:
- **Splits 1-4**: ~8-10 hours each (200 games each)
- **Split 5**: ~7-8 hours (171 games)
- **Total time**: ~10 hours (vs 50 hours sequential!)

## Troubleshooting

**Check if a session crashed:**
```bash
tmux has-session -t rag-analyzer-1 && echo "Running" || echo "Not running"
```

**View errors from crashed session:**
```bash
tmux capture-pane -t rag-analyzer-1 -p | tail -50
```

**Restart specific analyzer:**
```bash
# Kill if needed
tmux kill-session -t rag-analyzer-3

# Restart (example for split 3 on GPU 3)
tmux new -d -s rag-analyzer-3 bash -c "
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store
CUDA_VISIBLE_DEVICES=3 python game_analyzer.py --csv rag_files_list_3.csv --json-dir ../../build/rag_data_3 --output-dir ./rag_output_3 --max-visits 2400
"
```
