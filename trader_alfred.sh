#!/bin/bash
# Resolve the harness root from this script's location (override with GG_ROOT).
GG_ROOT="${GG_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$GG_ROOT" || exit 1
# OpenAI-compatible /v1/chat/completions endpoint (override with MODEL_URL).
URL="${MODEL_URL:-http://127.0.0.1:8000/v1/chat/completions}"
# Python interpreter (override with PY); defaults to python3 on PATH.
PY="${PY:-python3}"
# Re-run the trading gauntlet WITH a real production system prompt. The trader battery
# items carry no system prompt, so the bare model has zero account/structure context.
# Supply your OWN system-prompt file as the first argument (not bundled here).
SYS_PROMPT="${1:?usage: trader_alfred.sh <system_prompt.md>}"
L="${LOG:-$GG_ROOT/eval_trader_sysprompt.log}"
echo "=== trading-gauntlet WITH system prompt ($SYS_PROMPT) $(date) ===" > "$L"
"$PY" -u eval_finetuned.py --base-url "$URL" --ft-url "$URL" \
    --mode trading-gauntlet --system "$SYS_PROMPT" --timeout 180 --verbose \
    --out "$GG_ROOT/eval_trader_sysprompt.json" >> "$L" 2>&1
echo "TRADER_SYSPROMPT_DONE $(date)" >> "$L"
