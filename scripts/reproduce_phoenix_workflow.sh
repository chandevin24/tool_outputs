#!/bin/sh
# Default: replay saved evaluations offline. `all --out NEW_DIR` makes 24 new calls.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_COMMAND=${1:-replay}
if [ "$#" -gt 0 ]; then shift; fi
exec .venv/bin/python scripts/phoenix_workflow.py "$TASK_COMMAND" "$@"
