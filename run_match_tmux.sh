#!/bin/bash
# Run DataGo vs KataGo match in tmux

set -euo pipefail

SESSION_NAME="datago_match_gpu7"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
GO_ENV="$PARENT_ROOT/Go_env/bin/activate"
KATAGO_EXE="$REPO_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
MODEL_PATH="$REPO_ROOT/katago_repo/run/default_model.bin.gz"
CONFIG_PATH="$REPO_ROOT/katago_repo/run/gtp_800visits.cfg"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session with working directory
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Send commands to the session
tmux send-keys -t $SESSION_NAME "source $GO_ENV" C-m
tmux send-keys -t $SESSION_NAME "python run_datago_match.py \\" C-m
tmux send-keys -t $SESSION_NAME "  --katago-executable '$KATAGO_EXE' \\" C-m
tmux send-keys -t $SESSION_NAME "  --katago-model '$MODEL_PATH' \\" C-m
tmux send-keys -t $SESSION_NAME "  --katago-config '$CONFIG_PATH' \\" C-m
tmux send-keys -t $SESSION_NAME "  --config src/bot/config.yaml \\" C-m
tmux send-keys -t $SESSION_NAME "  --games 3 \\" C-m
tmux send-keys -t $SESSION_NAME "  --max-moves 250" C-m

echo "Tmux session '$SESSION_NAME' created!"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "View output with: tmux capture-pane -t $SESSION_NAME -p"
