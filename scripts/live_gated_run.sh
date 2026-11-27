#!/usr/bin/env bash
# Repo driver for a live LapBeast HAL run + sheaf-verify store.
# Does not overwrite the sealed T46 JSON. Does not print the seal key.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
exec python3 "$ROOT/scripts/live_gated_run.py" --fresh "$@"
