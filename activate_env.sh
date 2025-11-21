#!/bin/bash
# Quick activation script for Go_env
# Usage: source activate_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
VENV_PATH="$PARENT_ROOT/Go_env"

if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH/bin/activate"
    echo "✓ Go_env activated"
    echo "Python: $(which python)"
    echo "Location: $VENV_PATH"
else
    echo "Error: Go_env not found at $VENV_PATH"
    exit 1
fi
