# Running Game Analyzer on GPUs

This guide shows how to run `game_analyzer.py` to analyze flagged positions from selfplay games and generate the RAG database.

## What it does

The game analyzer:
1. Reads `rag_files_list.csv` to get list of RAG JSON files
2. Loads flagged positions from each JSON file in `build/rag_data/`
3. Runs offline MCTS analysis on each position (with configurable depth via `--max-visits`)
4. Outputs a single `rag_database.json` file with all analyzed positions

## Prerequisites

**IMPORTANT: Always activate the virtual environment first!**

```bash
# Activate Python environment (REQUIRED)
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate

# Make sure you're in the right directory
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store

# Verify sgfmill is installed
python -c "import sgfmill; print('sgfmill OK')"
```

## Running on GPU 7

```bash
# STEP 1: Activate Go_env first!
source /scratch2/f003x5w/old_RAG/Go_env/bin/activate
cd /scratch2/f003x5w/old_RAG/RAGFlow-Datago/datago/rag_store

# STEP 2: Run full analysis with 800 visits (production)
CUDA_VISIBLE_DEVICES=7 python game_analyzer.py \
  --katago-path ../../build/katago \
  --config ../../katago_repo/run/analysis.cfg \
  --model ../../katago_repo/run/kata1-b28c512nbt-s11653980416-d5514111622.bin.gz \
  --csv rag_files_list.csv \
  --json-dir ../../build/rag_data \
  --output-dir ./rag_output \
  --max-visits 800
```

## Quick test run

```bash
# Test with first 10 positions, 100 visits
CUDA_VISIBLE_DEVICES=7 python game_analyzer.py \
  --max-positions 10 \
  --max-visits 100
```

## Command line options

```
--katago-path       Path to KataGo executable (default: ../../build/katago)
--config            Path to analysis config (default: ../../katago_repo/run/analysis.cfg)
--model             Path to neural net model (default: ../../build/models/model.bin.gz)
--csv               CSV file with JSON filenames (default: rag_files_list.csv)
--json-dir          Directory with RAG JSON files (default: ../../build/rag_data)
--output-dir        Where to save output database (default: ./rag_output)
--max-visits        MCTS visits per position (default: 800)
--max-positions     Limit number of positions for testing (default: all)
```

## Running on different GPUs

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python game_analyzer.py --max-visits 800

# GPU 1
CUDA_VISIBLE_DEVICES=1 python game_analyzer.py --max-visits 800

# GPU 2
CUDA_VISIBLE_DEVICES=2 python game_analyzer.py --max-visits 800
```

## Running in background with tmux

```bash
# Start in tmux session
tmux new -s game-analyzer

# Run the analyzer
CUDA_VISIBLE_DEVICES=7 python game_analyzer.py --max-visits 800

# Detach: Ctrl+b then d

# Re-attach later
tmux attach -s game-analyzer
```

## Output

The analyzer creates:
- `./rag_output/rag_database.json` - Complete RAG database with all analyzed positions

Each position in the database contains:
- `sym_hash` - Symmetry-invariant hash (lookup key for RAG)
- `policy` - Full policy vector (362 values)
- `ownership` - Ownership map (361 values)
- `winrate` - Position evaluation
- `score_lead` - Score estimate
- `move_infos` - Detailed move analysis with visits/winrate/lcb/etc
- `child_nodes` - Child moves with values from selfplay
- `stone_count` - Black/white/total stones on board
- `komi`, `query_id`, `state_hash` - Additional metadata

## Monitoring progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check output file size as it grows
watch -n 5 "ls -lh ./rag_output/rag_database.json 2>/dev/null || echo 'Not created yet'"

# Tail the output in tmux session
tmux attach -t game-analyzer
```

## Troubleshooting

**Model not found:**
```bash
# Check if model exists
ls -lh ../../build/models/

# If missing, you may need to copy or symlink your model
```

**Config not found:**
```bash
# Check if analysis config exists
ls -lh ../../katago_repo/run/analysis.cfg

# Create one if needed based on your gtp config
```

**Out of memory:**
```bash
# Reduce max visits
python game_analyzer.py --max-visits 200
```
