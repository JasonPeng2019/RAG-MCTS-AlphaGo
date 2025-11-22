#!/bin/bash
# run_simple_katago_tmux.sh
# Run simple KataGo vs KataGo test on GPU 7 in tmux

set -euo pipefail

SESSION_NAME="katago_gpu7_test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
GO_ENV="$REPO_ROOT/Go_env/bin/activate"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Set up the environment and run the test
tmux send-keys -t $SESSION_NAME "source $GO_ENV" C-m
tmux send-keys -t $SESSION_NAME "echo '=== KataGo Self-Play on GPU 7 ==='" C-m
tmux send-keys -t $SESSION_NAME "python simple_play_katago.py" C-m

echo "=========================================="
echo "Tmux session '$SESSION_NAME' created!"
echo "=========================================="
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach: Ctrl+B, then D"
echo "To kill: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Checking GPU 7 usage:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | grep "^7,"
echo "=========================================="
