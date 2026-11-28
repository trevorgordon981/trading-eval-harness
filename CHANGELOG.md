# Changelog

All notable changes to `trading-eval-harness` are documented in this file.

## [0.1.0] - 2026-11-27

### Added
- Stdlib-only BASE vs. fine-tuned (and base-only) evaluation harness (`eval_finetuned.py`) across three strictly disjoint targets: `held-out`, `trading-gauntlet` (`tickers`, `trademath`, `trader`), and `gordon` (`general_reasoning`).
- Full-denominator all-case scoring with strict typed 6-field forecast schema validation (`validate_heldout_forecast`), uniform `failure_census` reporting (`http_error`, `timeout`, `empty`, `refusal`, `truncated`, `parse_error`, `schema_error`), and coverage-gated verdicts (`--min-coverage`).
- Fail-closed objective checks rejecting negated numbers, ambiguous multi-number prose, truncation (`finish_reason=length`), refusals, and missing structured tool calls.
- Strict unique `case_id` enforcement, exact paired membership checks for paired bootstrap 95% CIs, and deterministic `case_ids_sha256` identity digests before any model calls.
- Hash-bound SHA-256 run contracts (`gauntlet_contract.py`) with pre-call manifest/battery/system-prompt freezing and per-row `model` + `prompt_sha256` verification.
- Identity-blind judge packet export (`judge_blind.py`), conviction calibration & ECE gate (`calibration_gate.py`), realized-fill calibration (`conviction_calibration.py`), bootstrap noise-band / MDE analysis (`variance.py`), and fail-closed exact/normalized/near-duplicate contamination gate (`verify_holdout_disjoint.py`).
- Four-stage reproducible offline demo (`offline_demo.py`), 50-test unit suite, and GitHub Actions CI.
