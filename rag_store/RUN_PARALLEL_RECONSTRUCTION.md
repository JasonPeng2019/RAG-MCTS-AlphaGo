# Parallel RAG Position Reconstruction

This guide runs deep MCTS analysis on all flagged positions across 5 GPUs in parallel.

## System Setup

**Available GPUs**: 7, 5, 3, 1, 2 (sorted by available memory)

**Data Split**:
- GPU 7: rag_data_1 (200 files) → rag_output_1
- GPU 5: rag_data_2 (200 files) → rag_output_2
- GPU 3: rag_data_3 (200 files) → rag_output_3
- GPU 1: rag_data_4 (200 files) → rag_output_4
- GPU 2: rag_data_5 (171 files) → rag_output_5

**Total**: 971 RAG files containing ~12,797 flagged positions

## Configuration

- **MCTS Depth**: 10,000 visits per position (16x deeper than selfplay)
- **Rules**: Uses exact ruleset from each game (koRules, scoringRules, taxRules, etc.)
- **Output**: Incremental JSON writing (saves after each position)
- **Error Handling**: Skips positions with illegal moves (if any remain)

## Quick Launch

```bash
cd /scratch2/f004h1v/alphago_project/datago/rag_store
./launch_reconstruction.sh
```

## Monitor Progress

Check all sessions:
```bash
tmux ls | grep rag-recon
```

Attach to a specific session:
```bash
tmux attach -t rag-recon-1
```
(Press Ctrl+B then D to detach)

Check processing status:
```bash
for i in {1..5}; do 
  echo "=== Reconstructor $i ===" 
  tmux capture-pane -t rag-recon-$i -p | grep "Processing position" | tail -1
done
```

Monitor GPU utilization:
```bash
watch -n 5 nvidia-smi
```

Check output file sizes:
```bash
ls -lh rag_output_*/rag_database.json
```

## Stop All Sessions

```bash
for i in {1..5}; do tmux kill-session -t rag-recon-$i 2>/dev/null; done
```

## Expected Runtime

- ~40 seconds per position (varies by game complexity)
- ~12,797 positions total
- Estimated: **2-3 days** for complete reconstruction

## Output Format

Each `rag_output_N/rag_database.json` contains:
```json
[
  {
    "sym_hash": "...",       // Lookup key for retrieval
    "policy": [...],         // 362 floats
    "ownership": [...],      // 361 floats  
    "winrate": 0.5,
    "score_lead": 2.3,
    "move_infos": [...],     // Deep MCTS child data
    "stone_count": {...}
  }
]
```

## Merge Final Database

After all sessions complete:
```bash
python3 << 'PYEOF'
import json

all_positions = []
for i in range(1, 6):
    with open(f'rag_output_{i}/rag_database.json', 'r') as f:
        all_positions.extend(json.load(f))

with open('rag_database_complete.json', 'w') as f:
    json.dump(all_positions, f, indent=2)

print(f"✓ Merged {len(all_positions)} positions into rag_database_complete.json")
PYEOF
```
