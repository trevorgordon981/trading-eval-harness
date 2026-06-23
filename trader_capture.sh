#!/bin/bash
# Resolve the harness root from this script's location (override with GG_ROOT).
GG_ROOT="${GG_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$GG_ROOT" || exit 1
# OpenAI-compatible /v1/chat/completions endpoint (override with MODEL_URL).
URL="${MODEL_URL:-http://127.0.0.1:8000/v1/chat/completions}"
# Python interpreter (override with PY); defaults to python3 on PATH.
PY="${PY:-python3}"
# Capture base answers for the (judge-heavy) trading gauntlet so they can be graded
# in-session or by a local judge.
L="${LOG:-$GG_ROOT/eval_trader_capture.log}"
echo "=== trading-gauntlet answer-capture (base) $(date) ===" > "$L"
"$PY" -u eval_finetuned.py --base-url "$URL" --ft-url "$URL" \
    --mode trading-gauntlet --timeout 180 --verbose --out "$GG_ROOT/eval_trader_answers.json" >> "$L" 2>&1
echo "TRADER_CAPTURE_DONE $(date)" >> "$L"
