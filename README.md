# trading-eval-harness

An objective evaluation/benchmark harness for an **options-trading LLM** — measures
whether a model produces good, well-calibrated options-trade *recommendations*, and
whether a fine-tuned variant actually trades better than its base.

It runs a **base vs. fine-tuned** comparison across three targets and prints a single
scorecard plus a verdict:

1. **held-out** — a held-out trading dataset. The model reads a market snapshot and
   emits a structured JSON forecast; the harness scores it field-by-field against
   ground truth (no LLM judge needed).
2. **trading-gauntlet** — small hand-written batteries (`trader`, `trademath`,
   `tickers`) that probe options reasoning, position-sizing math, affordability
   discipline, structure constraints (long-debit only), and ticker scouting.
   Objective checks are auto-scored; subjective `llm_judge` items are *flagged* for
   grading in-session or by a local judge (never auto-scored here).
3. **gordon** *(optional)* — a general-ability regression over whatever extra
   batteries you drop in `batteries/`, to confirm fine-tuning didn't degrade general
   skills. Excluded by default unless you add batteries.

Both the base and the fine-tuned models are queried over **OpenAI-compatible
`/v1/chat/completions`** endpoints — the harness just points at two URLs.

## The held-out schema (5 fields + conviction)

Each held-out row is a chat sample (`messages` with `system`/`user`/`assistant`).
The **assistant** turn is the ground truth: a JSON object with

| field          | values                                          | meaning                          |
|----------------|-------------------------------------------------|----------------------------------|
| `move`         | `QUIET` / `NORMAL` / `ELEVATED` / `EXPLOSIVE`   | expected magnitude regime        |
| `call`         | `UP` / `DOWN` / `NEUTRAL`                        | directional call                 |
| `vol_change`   | `EXPANDING` / `STABLE` / `CONTRACTING`          | implied-vol regime change        |
| `vs_iv`        | `RICH` / `FAIR` / `CHEAP`                        | option pricing vs. fair value    |
| `exp_move_pct` | number (percent)                                | expected move size               |
| `conviction`   | 1–10                                            | model's confidence in the call   |

See `batteries/heldout_example.jsonl` for a small **synthetic** example file in
exactly this format (made-up tickers/numbers — supply your own real dataset via
`--heldout` or `$HELDOUT_PATH`).

## Held-out metrics

For each model the harness reports:

- **parse_rate** — fraction of responses that yielded parseable JSON.
- **move_acc** — accuracy on the `move` magnitude category.
- **dir_acc** — **directional accuracy** on the `call` field (the headline number).
- **exp_move_mae** — mean absolute error on `exp_move_pct`.
- **vol_change_acc** / **vs_iv_acc** — accuracy on the two regime categoricals.
- Per categorical field it also prints **raw / majority-baseline / balanced-accuracy /
  lift** so skewed label distributions can't masquerade as skill.

## Conviction calibration

The model's `conviction` (1–10) is bucketed and the harness reports call-accuracy
within each bucket — does higher stated confidence actually mean a higher hit rate?

- **low_1_4**, **mid_5_7**, **high_8_10** — call-accuracy per bucket, alongside the
  majority baseline and a balanced-accuracy view so calibration is judged honestly.

## Running it

```bash
# point at two OpenAI-compatible endpoints (use the same URL for both to run base-only)
python eval_finetuned.py \
    --base-url http://127.0.0.1:8000/v1/chat/completions \
    --ft-url   http://127.0.0.1:8001/v1/chat/completions \
    --mode all --n 300
```

Modes: `held-out` | `trading-gauntlet` | `gordon` | `all`.

The optional **perplexity** sub-metric (token-level loss on the ground-truth answer)
only runs if you pass `--hf-base` and `--hf-ft` (a base model id/dir and a PEFT/LoRA
adapter dir). It lazily imports `torch` / `transformers` / `peft` and degrades
gracefully if they're missing.

### Convenience wrappers

The `*.sh` scripts chain common runs. They resolve their root from their own
location (override with `GG_ROOT`) and read the endpoint from `$MODEL_URL`:

- `trading_eval_chain.sh` — trading-gauntlet then held-out, base-only.
- `heldout_rerun.sh` — just the held-out set.
- `trader_capture.sh` — capture trading-gauntlet answers for later grading.
- `trader_alfred.sh <system_prompt.md>` — re-run the trading gauntlet **with your own
  production system prompt** supplied as an argument (the battery items carry none, so
  the bare model has no account/structure context).
- `trader_fullctx.sh <full_context_system.md>` — same, with a fuller production context
  file (persona + constraints + a market brief).

`rag_grounded_rerun.py` re-runs a `rag_grounding` battery with retrieval injected from
a document-search endpoint (`--rag-url` / `$RAG_URL`, default `localhost:9000/search`).
You supply both the battery and the corpus endpoint.

## Environment variables

| var            | used by            | default                                            |
|----------------|--------------------|----------------------------------------------------|
| `GG_ROOT`      | scripts + eval     | the harness directory                              |
| `MODEL_URL`    | shell wrappers     | `http://127.0.0.1:8000/v1/chat/completions`        |
| `HELDOUT_PATH` | eval + wrappers    | `batteries/heldout_example.jsonl`                  |
| `RAG_URL`      | `rag_grounded_*`   | `http://localhost:9000/search`                     |
| `PY`           | shell wrappers     | `python3`                                          |

## Install

```bash
pip install -r requirements.txt   # just `requests` for the core endpoint path
```

The endpoint path is otherwise standard-library only. `torch` / `transformers` /
`peft` are optional (perplexity path only).

## License

MIT — see [LICENSE](LICENSE).
