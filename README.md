# trading-eval-harness

An objective evaluation and regression harness for an **options-trading LLM** —
measuring whether a fine-tuned model improves on held-out trading setups and
options discipline without regressing general reasoning or reliability.

It runs a **BASE vs. FINE-TUNED** (or **BASE-only**) comparison across three
strictly disjoint targets and prints a reliability-gated scorecard plus verdict:

1. **held-out** — a held-out trading JSONL set. Every attempted scenario is
   persisted as a terminal row and scored against the **full eligible attempted
   denominator** (unparsed, timed-out, empty, or refused replies count as
   incorrect in headline scores; conditional parsed-only accuracy is reported
   separately and never used for ranking).
2. **trading-gauntlet** — trading batteries from the `trading` profile in
   `battery_manifest.json` (`tickers`, `trademath`, `trader`) probing options
   math, affordability discipline, long-debit structure constraints, and ticker
   scouting. Objective checks fail closed on HTTP errors, timeouts, truncation
   (`finish_reason=length`), empty replies, explicit refusals, wrong or negated
   numeric answers, ambiguous multi-number prose, and missing structured tool
   calls. Subjective `llm_judge` items are flagged and optionally exported blind
   via `--blind-judge-dir`.
3. **gordon** — general-ability regression over strictly **disjoint non-trading
   batteries** (`general` profile in `battery_manifest.json`, shipping
   `batteries/general_reasoning.json`, plus any extra non-trading `*.json`
   dropped in `batteries/`). Never overlaps with `trading-gauntlet`, and
   withholds parity verdicts whenever valid coverage falls below the reliability
   gate (`--min-coverage`, default `0.80`).

Both BASE and FT are queried over OpenAI-compatible `/v1/chat/completions`
endpoints. The core harness is 100% Python standard library.

## Reproducible offline demo

Verify the entire pipeline locally with zero external APIs or model servers:

```bash
python3 offline_demo.py
```

`offline_demo.py` launches two deterministic ephemeral mock endpoints on
`127.0.0.1:0` and runs four stages end-to-end:

1. `eval_finetuned.py --mode all` (held-out + `trading-gauntlet` + disjoint
   `gordon`) with `--contract-dir` (hash-bound `gordon-result.v2` verification)
   and `--blind-judge-dir` (identity-blind judge packet export).
2. `calibration_gate.py --results ... --label FT` on the saved held-out output.
3. `variance.py band` bootstrap promotion noise-band check.
4. `verify_holdout_disjoint.py --battery-scan` against a synthetic clean corpus.

Pass `--out-dir ./demo_artifacts` to inspect the emitted JSONs, hash-bound
contracts, and blind judge packets.

## Running against real endpoints

```bash
# Head-to-head BASE vs. FT with hash-bound contracts and blind judge export
python3 eval_finetuned.py \
    --base-url http://127.0.0.1:8000/v1/chat/completions \
    --ft-url   http://127.0.0.1:8001/v1/chat/completions \
    --mode all --n 300 \
    --contract-dir ./results/contracts \
    --blind-judge-dir ./results/blind_judge

# Base-only mode (omit --ft-url, pass --base-only, or pass the same URL twice)
python3 eval_finetuned.py \
    --base-url http://127.0.0.1:8000/v1/chat/completions \
    --mode all --n 300
```

Modes: `held-out` | `trading-gauntlet` | `gordon` | `all`.

The optional **perplexity** sub-metric (token-level loss on ground-truth
assistant answers) runs only when `--hf-base` and `--hf-ft` are passed; it
lazily imports `torch` / `transformers` / `peft` and degrades gracefully if
absent.

## Failure coverage census schema

Every target (`held-out`, `skew`, `trading-gauntlet`, `gordon`) records a
uniform `failure_census` object alongside accuracy metrics so reliability
failures can never be hidden behind conditional denominators:

```json
{
  "attempted": 10,
  "succeeded": 8,
  "failed": 2,
  "coverage_rate": 0.8,
  "by_reason": {
    "http_error": 0,
    "timeout": 1,
    "empty": 0,
    "refusal": 0,
    "truncated": 0,
    "parse_error": 1,
    "schema_error": 0
  }
}
```

- `attempted`: total scenarios/items sent to the endpoint.
- `succeeded`: terminal items that returned an allowed terminal status
  (`stop`, or `tool_calls` for gauntlet items), non-empty, non-refused
  response — and for `held-out` / `skew`, a JSON object passing strict typed
  6-field forecast schema validation (`validate_heldout_forecast`).
- `failed`: `attempted - succeeded`.
- `coverage_rate`: `succeeded / attempted` (`0.0` when `attempted == 0`).
- `by_reason`: exact counts across `http_error`, `timeout`, `empty`, `refusal`,
  `truncated`, `parse_error`, and `schema_error`.
- Loose JSON parsing (`json_parsed`, `json_parse_rate`) is tracked separately
  as a diagnostic and never satisfies valid coverage when required forecast
  fields are missing, mistyped, or truncated (`finish_reason=length`).

Comparative improvement and parity verdicts are gated on
`coverage_rate >= --min-coverage` (default `0.80`). Runs below threshold emit
an explicit reliability warning or `VERDICT WITHHELD`.

## Held-out schema and metrics

Each held-out JSONL row is a chat sample (`messages` with
`system`/`user`/`assistant`). The `assistant` turn is the ground-truth JSON:

| field          | values                                        | meaning                        |
|----------------|-----------------------------------------------|--------------------------------|
| `move`         | `QUIET` / `NORMAL` / `ELEVATED` / `EXPLOSIVE` | expected magnitude regime      |
| `call`         | `UP` / `DOWN` / `NEUTRAL`                     | directional call               |
| `vol_change`   | `EXPANDING` / `STABLE` / `CONTRACTING`        | implied-vol regime change      |
| `vs_iv`        | `RICH` / `FAIR` / `CHEAP`                     | option pricing vs. fair value  |
| `exp_move_pct` | number (percent)                              | expected move size             |
| `conviction`   | `1`–`10`                                      | model's confidence in the call |

See `batteries/heldout_example.jsonl` for a 5-row **synthetic** example (supply
your own dataset via `--heldout` or `$HELDOUT_PATH`).

Reported metrics per model:
- **Headline all-case accuracy** (`move_acc`, `dir_acc`, `vol_change_acc`,
  `vs_iv_acc`): scored against the full eligible attempted denominator.
- **Conditional parsed-only accuracy** (`move_acc_parsed`, `dir_acc_parsed`,
  etc.): reported separately for diagnostics only, never used to rank models.
- **parse_rate / failure_census**: valid coverage and failure breakdown.
- **exp_move_mae** and **exp_move_mae_coverage**: MAE alongside coverage so
  sparse parses cannot claim an MAE win.
- **Paired 95% bootstrap CIs** (`_paired`): computed on immutable `case_id`s
  via `variance.paired_bootstrap_delta_ci`.
- **Per-field majority baseline, balanced accuracy, and lift** (`field_metrics`).

## Eval methodology modules

Each module is stdlib-only and works both standalone and (for contracts and
blind judging) directly from `eval_finetuned.py`:

### `verify_holdout_disjoint.py` — contamination gate

Checks that held-out rows and benchmark batteries are disjoint from training
corpora before trusting eval scores. Fails closed (exit `2`) on missing or
empty inputs and fails (exit `1`) on contamination.
- **Canonical normalization** (NFKC + whitespace collapse + casefold) applied
  consistently across both `--train-dir` (holdout-key mode) and
  `--battery-scan` mode.
- **Near-duplicate detection** via 3-token shingle Jaccard similarity
  (`>= 0.85`, minimum 6 tokens).
- Explicitly reports **exact**, **normalized**, **near-duplicate**, and
  **unscannable** counts along with stable scenario IDs.

### `variance.py` — reliability and promotion noise band

Measures run-to-run blank/error variance on an endpoint and computes bootstrap
95% CIs (`bootstrap_delta_ci`, `paired_bootstrap_delta_ci`,
`paired_cluster_bootstrap_delta_ci`) plus minimum detectable effect (MDE).

### `calibration_gate.py` — conviction reliability & ECE gate

Bins self-reported conviction against observed accuracy, computes ECE/MCE, and
gates high-vs-low discrimination lift (`PASS` / `FAIL` / `INSUFFICIENT`).
Accepts `.json` lists, `.jsonl` files (or JSONL fallback), and
`eval_finetuned.py` results JSON (`--results ... --label FT`). Normalizes
confidence by `--conf-scale` (`10` for 1–10, `1` for 0–1) **before** grouping
into `low (<=40%)`, `mid`, and `high (>=80%)`.

### `judge_blind.py` — identity-blind LLM judge packets

Strips `model`, `label`, and `key` from `llm_judge` / `ib_grader` records,
assigns an opaque SHA-256 `answer_uid` bound to prompt+criteria+response,
shuffles `pending_judge.jsonl`, and writes a separate `judge_uid_map.json`
sidecar. Wired directly into `eval_finetuned.py --blind-judge-dir`.

### `gauntlet_contract.py` — hash-bound (SHA-256) run contracts

Creates and verifies **hash-bound** run manifests (`gordon-run.v1`), battery
result files (`gordon-result.v2`), and completion receipts
(`gordon-run-completion.v1`) against `battery_manifest.json`. Wired directly
into `eval_finetuned.py --contract-dir` — which freezes manifest, battery
bytes, and optional system prompt **before** the first model call, records
effective per-request `model` and `prompt_sha256` digests on every result row,
and verifies at completion that no battery, manifest, system prompt, or result
changed during the run. Also runnable standalone via CLI (`list`, `create-run`,
`complete-run`, `verify-run`).

## Methodology limits & honest scope

- **Hash-bound, not cryptographically signed:** `gauntlet_contract.py` binds
  manifests, battery files, and results with SHA-256 content hashes to detect
  accidental drift or post-hoc edits. It does **not** use asymmetric signing
  keys, PKI, or hardware attestation; an adversary with write access to all
  output files can regenerate matching hashes.
- **Objective checks vs. subjective judgment:** `eval_finetuned.py` auto-scores
  deterministic checks only. Subjective trading quality in `trader.json`
  (`llm_judge` items) is never auto-scored — export blind packets with
  `--blind-judge-dir` and grade separately.
- **Contamination gate scope:** `verify_holdout_disjoint.py` catches exact
  matches, whitespace/Unicode/case variants, and high-overlap 3-token shingle
  near-duplicates. It does **not** detect semantic paraphrases or embedding
  similarity.
- **Synthetic defaults:** `batteries/heldout_example.jsonl` (5 rows) and
  `offline_demo.py` exist to prove harness mechanics offline. Real promotion
  decisions require an uncontaminated real held-out set large enough to clear
  the MDE reported by `variance.py` and the sample minimums in
  `calibration_gate.py` (`>=30` total, `>=8` per low/high group).
- **Realized P&L calibration:** `conviction_calibration.py` is read-only and
  depends on external closed-trade fills (`--fills`); without realized fills it
  reports `INSUFFICIENT DATA`.

## Convenience wrappers & environment variables

The `*.sh` scripts chain common base-only runs using `$MODEL_URL` (default
`http://127.0.0.1:8000/v1/chat/completions`), `$GG_ROOT`, `$HELDOUT_PATH`, and
`$PY`:
- `trading_eval_chain.sh` — `trading-gauntlet` then `held-out` (base-only).
- `heldout_rerun.sh` — held-out set (base-only).
- `trader_capture.sh` — capture `trading-gauntlet` answers for blind grading.
- `trader_alfred.sh <system_prompt.md>` / `trader_fullctx.sh <context.md>` —
  re-run `trading-gauntlet` with a supplied production system prompt.
- `rag_grounded_rerun.py` — re-run a retrieval battery against `$RAG_URL`.

## Testing

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 calibration_gate.py --selftest
python3 offline_demo.py
```

## License

MIT — see [LICENSE](LICENSE).
