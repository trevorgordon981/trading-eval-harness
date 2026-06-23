#!/bin/bash
# Resolve the harness root from this script's location (override with GG_ROOT).
GG_ROOT="${GG_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$GG_ROOT" || exit 1
# OpenAI-compatible /v1/chat/completions endpoint (override with MODEL_URL).
URL="${MODEL_URL:-http://127.0.0.1:8000/v1/chat/completions}"
# Python interpreter (override with PY); defaults to python3 on PATH.
PY="${PY:-python3}"
# Re-run the held-out trading set for the BASE model.
HELDOUT="${HELDOUT_PATH:-$GG_ROOT/batteries/heldout_example.jsonl}"
L="${LOG:-$GG_ROOT/eval_base_heldout.log}"
echo "=== held-out BASE re-run $(date) ===" > "$L"
"$PY" -u eval_finetuned.py --base-url "$URL" --ft-url "$URL" \
    --mode held-out --n 500 --heldout "$HELDOUT" \
    --timeout 180 --verbose >> "$L" 2>&1
echo "HELDOUT_RERUN_DONE $(date)" >> "$L"
