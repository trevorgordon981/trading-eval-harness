#!/bin/bash
# Resolve the harness root from this script's location (override with GG_ROOT).
GG_ROOT="${GG_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$GG_ROOT" || exit 1
# OpenAI-compatible /v1/chat/completions endpoint (override with MODEL_URL).
URL="${MODEL_URL:-http://127.0.0.1:8000/v1/chat/completions}"
# Python interpreter (override with PY); defaults to python3 on PATH.
PY="${PY:-python3}"
# Run the BASE trading evals as a chain: the trading gauntlet, then the held-out set.
# Base-only (ft-url == base-url -> the FT pass is auto-skipped).
LOG="${LOG:-$GG_ROOT/eval_base_trading.log}"
HELDOUT="${HELDOUT_PATH:-$GG_ROOT/batteries/heldout_example.jsonl}"

echo "=== starting TRADING evals $(date) ===" >> "$LOG"
# 1) the trading gauntlet (trader / trademath / tickers batteries)
"$PY" -u eval_finetuned.py --base-url "$URL" --ft-url "$URL" --mode trading-gauntlet --timeout 180 --verbose >> "$LOG" 2>&1
echo "=== trading-gauntlet done $(date); starting held-out ===" >> "$LOG"
# 2) the held-out trading dataset
"$PY" -u eval_finetuned.py --base-url "$URL" --ft-url "$URL" --mode held-out --n 500 \
    --heldout "$HELDOUT" --timeout 180 --verbose >> "$LOG" 2>&1
echo "TRADING_EVAL_DONE $(date)" >> "$LOG"
