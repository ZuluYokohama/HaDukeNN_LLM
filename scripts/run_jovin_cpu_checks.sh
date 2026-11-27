#!/usr/bin/env bash
# Receipted stdlib CPU-only checks for HaDukeNN inside the jovin-harness sandbox.
# Does not use the GPU. Does not overwrite the sealed T46 JSON. Does not print the seal key.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
exec python3 "$ROOT/scripts/run_jovin_cpu_checks.py" "$@"
