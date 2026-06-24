# Blind V5 OpenAI 1,000-Case Eval

This folder contains the most complete blind decision eval run from June 23, 2026.

It evaluates the v5 trading prompt on 1,000 generated simple decision cases using three OpenAI chat models:

- `gpt-5.5`
- `gpt-5.4`
- `gpt-5.3-chat-latest`

The run is a prompt-only synthetic decision benchmark derived from the repository's bundled `batteries/heldout_example.jsonl` archetypes. It is not a real-market backtest and does not use BotTrade or live market data.

## Files

| File | Purpose |
| --- | --- |
| `manifest.json` | Run metadata, seed, file map, and contamination rule. |
| `prompts.jsonl` | The 1,000 prompt-only cases shown to models. |
| `answer_key.jsonl` | Hidden scoring key, published only after model decisions were locked. Do not use this for future blind model phases. |
| `decisions_gpt55_v5.jsonl` | Final repaired `gpt-5.5 + v5 prompt` decisions. |
| `decisions_gpt54_v5.jsonl` | Final `gpt-5.4 + v5 prompt` decisions. |
| `decisions_gpt53_v5.jsonl` | Final repaired `gpt-5.3-chat-latest + v5 prompt` decisions. |
| `scoreboard.md` | Human-readable final leaderboard. |
| `scoreboard.json` | Machine-readable final leaderboard and quality data. |
| `repair_audit.json` | Counts for pre-repair incomplete rows and final file quality. |
| `pre_repair/` | Original incomplete decision files retained for auditability. |
| `runner/blind_decision_eval.py` | Snapshot of the runner/scorer used for this run. |

## Methodology

The workflow was intentionally split into three phases:

1. `generate`: writes `prompts.jsonl` and `answer_key.jsonl` separately.
2. `run-model`: reads only `prompts.jsonl`, calls the selected model, and writes a decisions file.
3. `score`: joins locked decisions with `answer_key.jsonl` and computes returns and baselines.

The runner phase did not read the answer key. The answer key is included here only so this completed historical run can be reproduced and audited.

Generation used, from the repository root:

```bash
python3 results/blind-v5-openai-1000-20260623/runner/blind_decision_eval.py generate \
  --source batteries/heldout_example.jsonl \
  --out-root results \
  --run-id blind-v5-openai-1000-20260623 \
  --n 1000 \
  --seed 20260623
```

Model decisions used the same prompt file and appended the v5 operating prompt via `--agent-prompt`. Representative invocation:

```bash
python3 results/blind-v5-openai-1000-20260623/runner/blind_decision_eval.py run-model \
  --prompts results/blind-v5-openai-1000-20260623/prompts.jsonl \
  --out results/blind-v5-openai-1000-20260623/decisions_gpt55_v5.jsonl \
  --model gpt-5.5 \
  --agent-prompt /path/to/robinhood-agentic-operating-prompt-v5.md \
  --max-tokens 60 \
  --concurrency 8
```

Some initial `gpt-5.5` and `gpt-5.3-chat-latest` calls exhausted the output cap and returned no parseable decision. Those rows were repaired by rerunning only the incomplete prompt-only cases at a larger output cap before scoring. The answer key was still not read during repair. See `repair_audit.json`.

Scoring rule:

- `BUY` earns `realized_return_pct`.
- `SHORT` earns `-realized_return_pct`.
- `NO TRADE` earns `0`.
- Directional accuracy is measured only on actual trades, not no-trade decisions.

## Final Board

| Rank | Policy/model | Return | Trades | Directional accuracy |
| ---: | --- | ---: | ---: | ---: |
| 1 | Perfect hindsight | +4926.17% | 998 | 100.0% |
| 2 | Source-label oracle | +3503.22% | 600 | 83.5% |
| 3 | `gpt-5.4 + v5 prompt` | +3359.84% | 576 | 83.2% |
| 4 | `gpt-5.5 + v5 prompt` | +3351.43% | 566 | 83.8% |
| 5 | `gpt-5.3-chat-latest + v5 prompt` | +3199.68% | 540 | 83.3% |
| 6 | Momentum | +2893.15% | 833 | 70.2% |
| 7 | Always Short | +546.81% | 1000 | 43.9% |
| 8 | Always No Trade | 0.00% | 0 | n/a |
| 9 | Seeded coin flip | -63.57% | 1000 | 49.3% |
| 10 | Always Buy | -546.81% | 1000 | 55.9% |

## Validity Notes

- This is a synthetic generated benchmark from five included archetypes.
- It tests simple decision discipline, not live execution quality.
- The small `gpt-5.4` edge over `gpt-5.5` on this seed should be treated as a tie unless repeated across more seeds.
- Published `answer_key.jsonl` contaminates this exact run for future blind model testing; generate a fresh run for new comparisons.
