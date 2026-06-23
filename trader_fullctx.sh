#!/bin/bash
# Resolve the harness root from this script's location (override with GG_ROOT).
GG_ROOT="${GG_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$GG_ROOT" || exit 1
# OpenAI-compatible /v1/chat/completions endpoint (override with MODEL_URL).
URL="${MODEL_URL:-http://127.0.0.1:8000/v1/chat/completions}"
# Python interpreter (override with PY); defaults to python3 on PATH.
PY="${PY:-python3}"
# Re-run the trading gauntlet with a FULL production context system prompt (persona +
# structure constraints + a live market brief). Supply your OWN context file as arg 1.
SYS_PROMPT="${1:?usage: trader_fullctx.sh <full_context_system.md>}"
L="${LOG:-$GG_ROOT/eval_trader_fullctx.log}"
echo "=== trading-gauntlet FULL CONTEXT ($SYS_PROMPT) $(date) ===" > "$L"
"$PY" -u eval_finetuned.py --base-url "$URL" --ft-url "$URL" \
    --mode trading-gauntlet --system "$SYS_PROMPT" \
    --timeout 180 --verbose --out "$GG_ROOT/eval_trader_fullctx.json" >> "$L" 2>&1
echo "TRADER_FULLCTX_DONE $(date)" >> "$L"
