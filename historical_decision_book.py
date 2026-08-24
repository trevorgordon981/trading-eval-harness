#!/usr/bin/env python3
"""Build and score a real historical blind decision book.

The generated book is intentionally simple and auditable:

* prompts contain only information available at the decision date's close
* answer keys contain the hidden forward 5-trading-day return
* labels and returns are computed from historical OHLCV, not synthetic draws
* model runners read prompts only; scoring joins the answer key afterward

Data source defaults to Yahoo Finance's public chart endpoint, fetched over HTTPS
with the standard library. No API key is required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SEED = 20260624
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_HORIZON = 5
DEFAULT_N = 1000
DEFAULT_THRESHOLD = 2.0
DECISIONS = {"BUY", "SHORT", "NO TRADE"}

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "INTC", "NFLX", "JPM", "BAC", "GS", "V", "MA", "XOM", "CVX",
    "SLB", "COP", "UNH", "JNJ", "PFE", "ABBV", "LLY", "WMT", "COST",
    "HD", "NKE", "MCD", "CAT", "BA", "GE", "DE", "DIS", "CRM",
    "ORCL", "IBM", "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLI",
    "XLY", "XLP", "TLT", "GLD", "SLV", "USO",
]


def pct(x: float) -> float:
    return round(100.0 * x, 2)


def unix_day(date_str: str) -> int:
    return int(dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def yahoo_symbol(ticker: str) -> str:
    return ticker.upper()


def yahoo_url(ticker: str, start: str, end: str) -> str:
    period1 = unix_day(start)
    # Yahoo's period2 is exclusive; add one day so the end date is included.
    period2 = unix_day(end) + 24 * 60 * 60
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(ticker)}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )


def fetch_csv(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "trading-eval-harness/decision-book"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_price_json(text: str) -> List[Dict[str, Any]]:
    data = json.loads(text)
    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        return []
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    rows: List[Dict[str, Any]] = []
    for i, ts in enumerate(timestamps):
        try:
            raw_close = closes[i]
            adj_close = adj[i] if i < len(adj) and adj[i] is not None else raw_close
            if raw_close is None or adj_close is None:
                continue
            ratio = adj_close / raw_close if raw_close else 1.0
            rows.append(
                {
                    "date": dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d"),
                    "open": float(opens[i]) * ratio if i < len(opens) and opens[i] is not None else float(adj_close),
                    "high": float(highs[i]) * ratio if i < len(highs) and highs[i] is not None else float(adj_close),
                    "low": float(lows[i]) * ratio if i < len(lows) and lows[i] is not None else float(adj_close),
                    "close": float(adj_close),
                    "volume": float(volumes[i]) if i < len(volumes) and volumes[i] is not None else 0.0,
                }
            )
        except (IndexError, TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r["date"])
    return rows


def download_prices(tickers: Sequence[str], start: str, end: str, cache_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices: Dict[str, List[Dict[str, Any]]] = {}
    source_manifest: List[Dict[str, Any]] = []
    for ticker in tickers:
        url = yahoo_url(ticker, start, end)
        cache_path = cache_dir / f"{ticker.lower()}_{start}_{end}.json"
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            text = fetch_csv(url)
            cache_path.write_text(text, encoding="utf-8")
            time.sleep(0.15)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows = parse_price_json(text)
        if len(rows) < 260:
            print(f"[warn] skipping {ticker}: only {len(rows)} usable rows")
            continue
        prices[ticker] = rows
        source_manifest.append(
            {
                "ticker": ticker,
                "url": url,
                "rows": len(rows),
                "first_date": rows[0]["date"],
                "last_date": rows[-1]["date"],
                "sha256": digest,
            }
        )
    return prices, source_manifest


def returns(prices: Sequence[float]) -> List[float]:
    out = [0.0]
    for prev, cur in zip(prices, prices[1:]):
        out.append((cur / prev) - 1.0 if prev else 0.0)
    return out


def realized_vol(ret_slice: Sequence[float]) -> float:
    if len(ret_slice) < 2:
        return 0.0
    return pct(statistics.stdev(ret_slice) * math.sqrt(252))


def rel_to_sma(close: float, closes: Sequence[float]) -> float:
    avg = statistics.mean(closes)
    return pct((close / avg) - 1.0) if avg else 0.0


def range_position(close: float, lows: Sequence[float], highs: Sequence[float]) -> float:
    lo = min(lows)
    hi = max(highs)
    if hi <= lo:
        return 50.0
    return round(100.0 * (close - lo) / (hi - lo), 1)


def build_candidates(
    prices: Dict[str, List[Dict[str, Any]]],
    *,
    horizon: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    spy_by_date = {row["date"]: row for row in prices.get("SPY", [])}
    spy_rows = prices.get("SPY", [])
    spy_index = {row["date"]: idx for idx, row in enumerate(spy_rows)}
    spy_closes = [row["close"] for row in spy_rows]
    candidates: List[Dict[str, Any]] = []
    for ticker, rows in prices.items():
        closes = [row["close"] for row in rows]
        highs = [row["high"] for row in rows]
        lows = [row["low"] for row in rows]
        vols = [row["volume"] for row in rows]
        rets = returns(closes)
        for i in range(60, len(rows) - horizon):
            row = rows[i]
            if row["volume"] <= 0 or row["date"] not in spy_by_date:
                continue
            fwd_ret = pct((closes[i + horizon] / closes[i]) - 1.0)
            label = "BUY" if fwd_ret > threshold else "SHORT" if fwd_ret < -threshold else "NO TRADE"
            spy_i = spy_index.get(row["date"])
            spy_5d = spy_20d = None
            if spy_i is not None and spy_i >= 20:
                spy_5d = pct((spy_closes[spy_i] / spy_closes[spy_i - 5]) - 1.0)
                spy_20d = pct((spy_closes[spy_i] / spy_closes[spy_i - 20]) - 1.0)
            avg_vol_20 = statistics.mean(vols[i - 19 : i + 1])
            dollar_vol_20 = statistics.mean(closes[j] * vols[j] for j in range(i - 19, i + 1))
            features = {
                "ticker": ticker,
                "decision_date": row["date"],
                "close": round(row["close"], 2),
                "one_day_return_pct": pct(rets[i]),
                "five_day_return_pct": pct((closes[i] / closes[i - 5]) - 1.0),
                "twenty_day_return_pct": pct((closes[i] / closes[i - 20]) - 1.0),
                "realized_vol_20d_pct": realized_vol(rets[i - 19 : i + 1]),
                "volume_vs_20d_avg": round(row["volume"] / avg_vol_20, 2) if avg_vol_20 else None,
                "position_in_60d_range_pct": range_position(row["close"], lows[i - 59 : i + 1], highs[i - 59 : i + 1]),
                "close_vs_sma20_pct": rel_to_sma(row["close"], closes[i - 19 : i + 1]),
                "close_vs_sma50_pct": rel_to_sma(row["close"], closes[i - 49 : i + 1]),
                "avg_dollar_volume_20d_millions": round(dollar_vol_20 / 1_000_000, 1),
                "spy_five_day_return_pct": spy_5d,
                "spy_twenty_day_return_pct": spy_20d,
            }
            candidates.append(
                {
                    "ticker": ticker,
                    "date": row["date"],
                    "year": row["date"][:4],
                    "bucket": label,
                    "forward_return_pct": fwd_ret,
                    "features": features,
                    "forward_close": round(closes[i + horizon], 2),
                }
            )
    return candidates


def prompt_for_case(
    case_id: str,
    features: Dict[str, Any],
    horizon: int,
    *,
    asset_alias: str,
    reveal_identities: bool = False,
) -> str:
    if reveal_identities:
        identity = f"Decision timestamp: after market close on {features['decision_date']}. Underlying {features['ticker']}"
        market_label = "SPY"
    else:
        identity = f"Decision timestamp: anonymized historical market close. Underlying {asset_alias}"
        market_label = "broad-market ETF"
    return (
        f"Case {case_id}. {identity} closed at ${features['close']:.2f}. "
        f"Known historical features at this timestamp: 1-day return {features['one_day_return_pct']:+.2f}%, "
        f"5-day return {features['five_day_return_pct']:+.2f}%, 20-day return {features['twenty_day_return_pct']:+.2f}%, "
        f"20-day realized volatility {features['realized_vol_20d_pct']:.2f}%, "
        f"volume {features['volume_vs_20d_avg']:.2f}x its 20-day average, "
        f"price position {features['position_in_60d_range_pct']:.1f}% through its 60-day high/low range, "
        f"close vs SMA20 {features['close_vs_sma20_pct']:+.2f}%, close vs SMA50 {features['close_vs_sma50_pct']:+.2f}%, "
        f"20-day average dollar volume ${features['avg_dollar_volume_20d_millions']:.1f}M, "
        f"{market_label} 5-day return {features['spy_five_day_return_pct']:+.2f}%, "
        f"{market_label} 20-day return {features['spy_twenty_day_return_pct']:+.2f}%. "
        f"You must make a blind paper-trading decision for the next {horizon} trading days. "
        "Choose exactly one: BUY, SHORT, or NO TRADE. "
        "BUY means bullish/long exposure; SHORT means bearish/put-like exposure; NO TRADE means stand aside. "
        "Use only the timestamped information above; do not use real-world future knowledge. "
        'Return JSON only: {"decision":"BUY|SHORT|NO TRADE"}.'
    )


def sample_cases(candidates: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {b: [] for b in DECISIONS}
    for case in candidates:
        buckets[case["bucket"]].append(case)
    if any(len(v) < n // 3 for v in buckets.values()):
        sizes = {k: len(v) for k, v in buckets.items()}
        raise SystemExit(f"not enough candidates for balanced sample: {sizes}")

    target_counts = {"BUY": n // 3, "SHORT": n // 3, "NO TRADE": n - 2 * (n // 3)}
    selected: List[Dict[str, Any]] = []
    # Soft-cap per ticker/outcome to avoid one name dominating. If strict caps exhaust,
    # the fallback below fills remaining cases deterministically.
    for bucket, target in target_counts.items():
        pool = list(buckets[bucket])
        rng.shuffle(pool)
        ticker_counts: Dict[str, int] = {}
        year_counts: Dict[str, int] = {}
        chosen: List[Dict[str, Any]] = []
        for case in pool:
            if ticker_counts.get(case["ticker"], 0) >= 12:
                continue
            if year_counts.get(case["year"], 0) >= math.ceil(target / 8) + 4:
                continue
            chosen.append(case)
            ticker_counts[case["ticker"]] = ticker_counts.get(case["ticker"], 0) + 1
            year_counts[case["year"]] = year_counts.get(case["year"], 0) + 1
            if len(chosen) == target:
                break
        if len(chosen) < target:
            chosen_ids = {(c["ticker"], c["date"], c["bucket"]) for c in chosen}
            for case in pool:
                key = (case["ticker"], case["date"], case["bucket"])
                if key in chosen_ids:
                    continue
                chosen.append(case)
                chosen_ids.add(key)
                if len(chosen) == target:
                    break
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected


def write_book(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir or (out_dir / "_stooq_cache"))
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    prices, source_manifest = download_prices(tickers, args.start, args.end, cache_dir)
    candidates = build_candidates(prices, horizon=args.horizon, threshold=args.threshold)
    selected = sample_cases(candidates, args.n, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / "prompts.jsonl"
    key_path = out_dir / "answer_key.jsonl"
    prompts: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    ticker_alias = {ticker: f"Asset {idx:03d}" for idx, ticker in enumerate(sorted({c["ticker"] for c in selected}), start=1)}
    for idx, case in enumerate(selected, start=1):
        case_id = f"H{idx:04d}"
        prompts.append(
            {
                "case_id": case_id,
                "prompt": prompt_for_case(
                    case_id,
                    case["features"],
                    args.horizon,
                    asset_alias=ticker_alias[case["ticker"]],
                    reveal_identities=args.reveal_identities,
                ),
            }
        )
        keys.append(
            {
                "case_id": case_id,
                "asset_alias": ticker_alias[case["ticker"]],
                "ticker": case["ticker"],
                "decision_date": case["date"],
                "horizon_trading_days": args.horizon,
                "ground_truth_decision": case["bucket"],
                "forward_return_pct": case["forward_return_pct"],
                "forward_close": case["forward_close"],
                "features": case["features"],
            }
        )

    with prompts_path.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with key_path.open("w", encoding="utf-8") as f:
        for row in keys:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    label_counts: Dict[str, int] = {}
    ticker_counts: Dict[str, int] = {}
    year_counts: Dict[str, int] = {}
    for row in keys:
        label_counts[row["ground_truth_decision"]] = label_counts.get(row["ground_truth_decision"], 0) + 1
        ticker_counts[row["ticker"]] = ticker_counts.get(row["ticker"], 0) + 1
        year_counts[row["decision_date"][:4]] = year_counts.get(row["decision_date"][:4], 0) + 1

    manifest = {
        "name": "historical-1k",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "n": len(keys),
        "start": args.start,
        "end": args.end,
        "horizon_trading_days": args.horizon,
        "label_threshold_pct": args.threshold,
        "source": "Yahoo Finance chart API daily OHLCV with adjusted close",
        "prompts": prompts_path.name,
        "answer_key": key_path.name,
        "generation_rule": "Prompts contain only close-of-date features; answer key is hidden forward return over horizon.",
        "selection_rule": "Deterministic outcome-balanced sample across historical candidates, with soft ticker/year caps.",
        "prompt_identity_policy": (
            "Prompts reveal ticker and date." if args.reveal_identities
            else "Prompts anonymize ticker and date; answer key retains real ticker/date for audit."
        ),
        "label_counts": label_counts,
        "asset_aliases": ticker_alias,
        "ticker_counts": dict(sorted(ticker_counts.items())),
        "year_counts": dict(sorted(year_counts.items())),
        "sources": source_manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(keys)} cases to {out_dir}")
    print(f"label_counts={label_counts}")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.replace("```json", "```")
    fences = re.findall(r"```(.*?)```", text, flags=re.S)
    for cand in fences + [text]:
        for obj in reversed(re.findall(r"\{[^{}]*\}", cand, flags=re.S)):
            try:
                return json.loads(obj)
            except Exception:
                pass
    return None


def parse_decision(text: str) -> str:
    data = extract_json(text)
    if data and isinstance(data.get("decision"), str):
        value = data["decision"].strip().upper()
        if value in DECISIONS:
            return value
    upper = text.upper()
    if "NO TRADE" in upper:
        return "NO TRADE"
    if "SHORT" in upper or "PUT" in upper or "BEARISH" in upper:
        return "SHORT"
    if "BUY" in upper or "CALL" in upper or "BULLISH" in upper:
        return "BUY"
    return "NO TRADE"


def call_openai_chat(model: str, messages: List[Dict[str, str]], max_tokens: int, timeout: int) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code in {429, 500, 502, 503, 504} and attempt < 4:
                wait = exc.headers.get("Retry-After")
                delay = int(wait) if wait and wait.isdigit() else 2**attempt
                time.sleep(delay)
                continue
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {body_text}") from exc


def run_one_prompt(row: Dict[str, Any], model: str, system: str, max_tokens: int, timeout: int) -> Dict[str, Any]:
    payload = call_openai_chat(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": row["prompt"]}],
        max_tokens,
        timeout,
    )
    choice = payload["choices"][0]
    text = choice["message"].get("content") or ""
    return {
        "case_id": row["case_id"],
        "model": model,
        "decision": parse_decision(text),
        "raw_response": text,
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
    }


def run_openai(args: argparse.Namespace) -> None:
    prompts = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"decisions file exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    system = (
        "You are in a blind historical trading-decision eval. You see only the timestamped case. "
        "Do not use real-world future knowledge. Return JSON only with one key: decision, whose value is BUY, SHORT, or NO TRADE."
    )
    if args.agent_prompt:
        system += "\n\nAgent operating prompt:\n" + Path(args.agent_prompt).read_text(encoding="utf-8")

    completed = 0
    decisions: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one_prompt, row, args.model, system, args.max_tokens, args.timeout): row
            for row in prompts
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                decisions[row["case_id"]] = future.result()
            except Exception as exc:
                decisions[row["case_id"]] = {
                    "case_id": row["case_id"],
                    "model": args.model,
                    "decision": "NO TRADE",
                    "raw_response": "",
                    "finish_reason": None,
                    "usage": None,
                    "error": repr(exc),
                }
            completed += 1
            if completed % args.progress_every == 0 or completed == len(prompts):
                print(f"completed={completed}/{len(prompts)}")

    with out_path.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(decisions[row["case_id"]], sort_keys=True) + "\n")
    print(f"decisions={out_path}")


def payoff(decision: str, forward_return_pct: float) -> float:
    if decision == "BUY":
        return forward_return_pct
    if decision == "SHORT":
        return -forward_return_pct
    return 0.0


def score_rows(keys: List[Dict[str, Any]], decisions: Dict[str, str], name: str) -> Dict[str, Any]:
    total = trades = correct = wrong = neutral = 0
    for row in keys:
        decision = decisions[row["case_id"]]
        total += payoff(decision, row["forward_return_pct"])
        actual = row["ground_truth_decision"]
        if decision == "NO TRADE":
            neutral += 1
        else:
            trades += 1
            if decision == actual:
                correct += 1
            else:
                wrong += 1
    return {
        "name": name,
        "return_pct": round(total, 2),
        "trades": trades,
        "correct": correct,
        "wrong": wrong,
        "neutral": neutral,
        "directional_accuracy_pct": round(100 * correct / trades, 2) if trades else None,
    }


def baseline_decisions(keys: List[Dict[str, Any]], seed: int) -> Dict[str, Dict[str, str]]:
    rng = random.Random(seed + 1)
    return {
        "Perfect hindsight": {
            row["case_id"]: "BUY" if row["forward_return_pct"] > 0 else "SHORT" if row["forward_return_pct"] < 0 else "NO TRADE"
            for row in keys
        },
        "Label oracle": {row["case_id"]: row["ground_truth_decision"] for row in keys},
        "Always Buy": {row["case_id"]: "BUY" for row in keys},
        "Always Short": {row["case_id"]: "SHORT" for row in keys},
        "Always No Trade": {row["case_id"]: "NO TRADE" for row in keys},
        "Alternating Buy/Short": {row["case_id"]: "BUY" if i % 2 else "SHORT" for i, row in enumerate(keys, start=1)},
        "Alternating Short/Buy": {row["case_id"]: "SHORT" if i % 2 else "BUY" for i, row in enumerate(keys, start=1)},
        "Seeded coin flip": {row["case_id"]: rng.choice(["BUY", "SHORT"]) for row in keys},
        "5-day momentum": {
            row["case_id"]: "BUY" if row["features"]["five_day_return_pct"] > 1.0 else "SHORT" if row["features"]["five_day_return_pct"] < -1.0 else "NO TRADE"
            for row in keys
        },
        "5-day mean reversion": {
            row["case_id"]: "SHORT" if row["features"]["five_day_return_pct"] > 1.0 else "BUY" if row["features"]["five_day_return_pct"] < -1.0 else "NO TRADE"
            for row in keys
        },
    }


def load_decision_file(spec: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    label, path = spec.split("=", 1)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions: Dict[str, str] = {}
    quality = {"rows": len(rows), "errors": 0, "empty_raw": 0, "finish_reasons": {}, "decision_counts": {}}
    finish_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    for row in rows:
        decision = row.get("decision")
        if decision not in DECISIONS:
            raise SystemExit(f"{label}: invalid decision {decision!r} for {row.get('case_id')}")
        decisions[row["case_id"]] = decision
        quality["errors"] += 1 if row.get("error") else 0
        quality["empty_raw"] += 0 if (row.get("raw_response") or "").strip() else 1
        finish = str(row.get("finish_reason"))
        finish_counts[finish] = finish_counts.get(finish, 0) + 1
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    quality["finish_reasons"] = finish_counts
    quality["decision_counts"] = decision_counts
    return label, decisions, quality


def score(args: argparse.Namespace) -> None:
    keys = [json.loads(line) for line in Path(args.answer_key).read_text(encoding="utf-8").splitlines() if line.strip()]
    policies: List[Tuple[str, Dict[str, str]]] = list(baseline_decisions(keys, args.seed).items())
    quality: Dict[str, Any] = {}
    key_ids = {row["case_id"] for row in keys}
    for spec in args.decision or []:
        label, decisions, q = load_decision_file(spec)
        if set(decisions) != key_ids:
            raise SystemExit(f"{label}: decision IDs do not match answer key")
        policies.append((label, decisions))
        quality[label] = q

    board = [score_rows(keys, decisions, name) for name, decisions in policies]
    board.sort(key=lambda row: row["return_pct"], reverse=True)
    payload = {"answer_key": str(Path(args.answer_key).resolve()), "n": len(keys), "quality": quality, "board": board}
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = ["# Historical Decision Book Scoreboard", "", f"Cases: `{len(keys)}`", "", "## Final Board", ""]
    for idx, row in enumerate(board, start=1):
        prefix = medals.get(idx, f"{idx}.")
        sign = "+" if row["return_pct"] > 0 else ""
        acc = row["directional_accuracy_pct"]
        acc_s = "n/a" if acc is None else f"{acc:.1f}%"
        lines.append(f"{prefix} {row['name']} -> {sign}{row['return_pct']:.2f}% ({row['trades']} trades, dir acc {acc_s})")
    lines.extend(["", "## Quality Audit", ""])
    if quality:
        for label, q in quality.items():
            lines.append(f"- {label}: rows={q['rows']}, errors={q['errors']}, empty_raw={q['empty_raw']}, finish={q['finish_reasons']}, decisions={q['decision_counts']}")
    else:
        lines.append("- No model decision files supplied; baselines only.")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)
    for idx, row in enumerate(board, start=1):
        print(f"{idx:02d} {row['name']}: {row['return_pct']:+.2f}% trades={row['trades']} acc={row['directional_accuracy_pct']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Historical 1K blind decision-book generator/runner/scorer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--out-dir", default="decision_books/historical-1k-20260624")
    gen.add_argument("--cache-dir", default=None)
    gen.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    gen.add_argument("--start", default=DEFAULT_START)
    gen.add_argument("--end", default=DEFAULT_END)
    gen.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    gen.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    gen.add_argument("--n", type=int, default=DEFAULT_N)
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.add_argument("--reveal-identities", action="store_true", help="include real ticker/date in prompts instead of anonymizing them")
    gen.set_defaults(func=write_book)

    run = sub.add_parser("run-openai")
    run.add_argument("--prompts", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    run.add_argument("--agent-prompt", default=None)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=run_openai)

    sc = sub.add_parser("score")
    sc.add_argument("--answer-key", required=True)
    sc.add_argument("--decision", action="append", help="label=path to decisions jsonl")
    sc.add_argument("--out-json", required=True)
    sc.add_argument("--out-md", required=True)
    sc.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sc.set_defaults(func=score)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
