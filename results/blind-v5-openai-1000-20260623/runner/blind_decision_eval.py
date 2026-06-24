#!/usr/bin/env python3
"""Blind 100-case decision eval for the Trevor Gordon trading harness.

The workflow is intentionally split:

1. generate  -> writes prompt-only cases and a separate hidden answer key
2. run-model -> reads only prompts, writes decisions
3. score     -> joins decisions with the answer key and compares baselines

The model runner never reads the answer key.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "batteries" / "heldout_example.jsonl"
DEFAULT_OUT_ROOT = ROOT / "blind_decision_runs"
DEFAULT_SEED = 20260623
DECISIONS = {"BUY", "SHORT", "NO TRADE"}
MOVE_SCALE = {
    "QUIET": 1.8,
    "NORMAL": 4.0,
    "ELEVATED": 6.5,
    "EXPLOSIVE": 11.0,
}


def now_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.replace("```json", "```")
    fences = re.findall(r"```(.*?)```", text, flags=re.S)
    candidates = fences + [text]
    for cand in candidates:
        for obj in reversed(re.findall(r"\{[^{}]*\}", cand, flags=re.S)):
            try:
                return json.loads(obj)
            except Exception:
                pass
        i, j = cand.find("{"), cand.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(cand[i : j + 1])
            except Exception:
                pass
    return None


def source_archetypes(source: Path) -> List[Dict[str, Any]]:
    archetypes: List[Dict[str, Any]] = []
    for idx, row in enumerate(load_jsonl(source), start=1):
        messages = row["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        assistant = next(m["content"] for m in messages if m["role"] == "assistant")
        gt = extract_json(assistant) or {}
        ticker_match = re.search(r"Underlying\s+([A-Z]{2,6})", user)
        price_match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", user)
        iv_match = re.search(r"30-day IV\s+([0-9]+(?:\.[0-9]+)?)%", user)
        rv_match = re.search(r"realized vol\s+([0-9]+(?:\.[0-9]+)?)%", user)
        day_match = re.search(r"\b(up|down|flat)\s+([0-9]+(?:\.[0-9]+)?)?%", user, re.I)
        day = 0.0
        if day_match:
            if day_match.group(1).lower() == "flat":
                day = 0.0
            else:
                mag = float(day_match.group(2) or 0)
                day = mag if day_match.group(1).lower() == "up" else -mag
        archetypes.append(
            {
                "source_index": idx,
                "source_user": user,
                "ticker": ticker_match.group(1) if ticker_match else f"T{idx:03d}",
                "base_price": float(price_match.group(1)) if price_match else 100.0,
                "base_iv": float(iv_match.group(1)) if iv_match else 30.0,
                "base_rv": float(rv_match.group(1)) if rv_match else 25.0,
                "base_day_change": day,
                "move": str(gt.get("move", "NORMAL")).upper(),
                "call": str(gt.get("call", "NEUTRAL")).upper(),
                "vol_change": str(gt.get("vol_change", "STABLE")).upper(),
                "vs_iv": str(gt.get("vs_iv", "FAIR")).upper(),
                "exp_move_pct": float(gt.get("exp_move_pct", 4.0)),
                "conviction": float(gt.get("conviction", 5)),
            }
        )
    return archetypes


def prompt_for_case(case: Dict[str, Any]) -> str:
    direction_word = "flat" if abs(case["day_change_pct"]) < 0.05 else ("up" if case["day_change_pct"] > 0 else "down")
    day_piece = "flat on the day" if direction_word == "flat" else f"{direction_word} {abs(case['day_change_pct']):.1f}% on the day"
    return (
        f"Case {case['case_id']}. Underlying {case['ticker']} at ${case['price']:.2f}, {day_piece}. "
        f"30-day IV {case['iv_30d']:.0f}%, realized vol {case['realized_vol']:.0f}%. "
        f"{case['market_context']} "
        "You must make a simple paper-trading decision for this blind eval only. "
        "Choose exactly one: BUY, SHORT, or NO TRADE. "
        "BUY means bullish/long exposure; SHORT means bearish/put-like exposure; NO TRADE means stand aside. "
        "Use only the information in this case, no real-world hindsight. "
        "Return JSON only: {\"decision\":\"BUY|SHORT|NO TRADE\"}."
    )


def market_context(archetype: Dict[str, Any], rng: random.Random) -> str:
    call = archetype["call"]
    move = archetype["move"]
    if call == "UP" and move == "ELEVATED":
        options = [
            "Earnings are approaching within two weeks; sector tone is constructive but not euphoric.",
            "A known catalyst is approaching; broad tape is calm and buyers have been steady.",
            "Options are active into a scheduled event; trend is modestly constructive.",
        ]
    elif call == "UP":
        options = [
            "The stock reclaimed a key trend level; sector rotation has improved.",
            "Buyers are returning after consolidation; no major scheduled risk is imminent.",
            "Relative strength improved while realized volatility is above implied volatility.",
        ]
    elif call == "DOWN":
        options = [
            "A negative company update hit on heavy volume; sector tone is weak.",
            "The tape is risk-off and the underlying is breaking support after bad guidance.",
            "Sellers are in control after a material downside catalyst.",
        ]
    else:
        options = [
            "No clear catalyst is visible; price has been range-bound and signal quality is low.",
            "A macro event is imminent and the tape is choppy with no firm direction.",
            "Momentum is mixed and option pricing looks close to fair for the setup.",
        ]
    return rng.choice(options)


def hidden_return(archetype: Dict[str, Any], rng: random.Random) -> float:
    call = archetype["call"]
    move = archetype["move"]
    scale = MOVE_SCALE.get(move, archetype["exp_move_pct"])
    if call == "UP":
        mu = scale * 0.72
        sigma = max(1.5, scale * 0.72)
    elif call == "DOWN":
        mu = -scale * 0.82
        sigma = max(1.8, scale * 0.75)
    else:
        mu = 0.0
        sigma = max(1.4, scale * 0.55)
    ret = rng.gauss(mu, sigma)
    if rng.random() < 0.10:
        ret += rng.choice([-1, 1]) * rng.uniform(scale * 0.5, scale * 1.2)
    return round(max(-30.0, min(30.0, ret)), 2)


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    archetypes = source_archetypes(Path(args.source))
    if not archetypes:
        raise SystemExit("No source archetypes found")
    run_dir = Path(args.out_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=args.force)
    prompt_path = run_dir / "prompts.jsonl"
    key_path = run_dir / "answer_key.jsonl"
    manifest_path = run_dir / "manifest.json"
    if not args.force and (prompt_path.exists() or key_path.exists()):
        raise SystemExit(f"Run already exists: {run_dir}")

    prompts: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    n_total = args.n
    for i in range(1, n_total + 1):
        arch = archetypes[(i - 1) % len(archetypes)]
        variant_rng = random.Random(rng.randint(1, 10**12))
        case = {
            "case_id": f"C{i:04d}",
            "source_index": arch["source_index"],
            "ticker": arch["ticker"],
            "price": round(arch["base_price"] * (1 + variant_rng.gauss(0, 0.035)), 2),
            "day_change_pct": round(arch["base_day_change"] + variant_rng.gauss(0, 0.9), 2),
            "iv_30d": round(max(5.0, arch["base_iv"] + variant_rng.gauss(0, 3.5)), 1),
            "realized_vol": round(max(5.0, arch["base_rv"] + variant_rng.gauss(0, 4.0)), 1),
            "market_context": market_context(arch, variant_rng),
        }
        prompt = prompt_for_case(case)
        realized_return_pct = hidden_return(arch, variant_rng)
        ground_truth = "BUY" if realized_return_pct > 0 else "SHORT" if realized_return_pct < 0 else "NO TRADE"
        prompts.append({"case_id": case["case_id"], "prompt": prompt})
        keys.append(
            {
                "case_id": case["case_id"],
                "source_index": arch["source_index"],
                "hidden_source_call": arch["call"],
                "hidden_source_move": arch["move"],
                "realized_return_pct": realized_return_pct,
                "ground_truth_decision": ground_truth,
                "features": case,
            }
        )
    rng.shuffle(prompts)
    prompt_order = {row["case_id"]: idx for idx, row in enumerate(prompts, start=1)}
    keys.sort(key=lambda row: prompt_order[row["case_id"]])

    with prompt_path.open("w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with key_path.open("w", encoding="utf-8") as f:
        for row in keys:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "n": n_total,
        "source": str(Path(args.source).resolve()),
        "prompts": str(prompt_path),
        "answer_key": str(key_path),
        "status": "generated",
        "contamination_rule": "Runner must read prompts.jsonl only; answer_key.jsonl is for scoring after decisions are locked.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"generated_run_dir={run_dir}")
    print(f"prompts={prompt_path}")
    print(f"answer_key={key_path}")


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
    body = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
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


def run_one_prompt(
    row: Dict[str, Any],
    *,
    model: str,
    system: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["prompt"]},
    ]
    payload = call_openai_chat(model, messages, max_tokens, timeout)
    message = payload["choices"][0]["message"]
    text = message.get("content") or ""
    return {
        "case_id": row["case_id"],
        "model": model,
        "decision": parse_decision(text),
        "raw_response": text,
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "usage": payload.get("usage"),
    }


def run_model(args: argparse.Namespace) -> None:
    prompts_path = Path(args.prompts)
    out_path = Path(args.out)
    if not args.force and out_path.exists():
        raise SystemExit(f"Decisions file exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    system = (
        "You are in a blind trading-decision eval. "
        "You see only the current case and must not use real-world future knowledge or infer scenario outcomes. "
        "Return JSON only with one key: decision, whose value is BUY, SHORT, or NO TRADE."
    )
    if args.agent_prompt:
        system += "\n\nAgent operating prompt:\n" + Path(args.agent_prompt).read_text(encoding="utf-8")
    rows = load_jsonl(prompts_path)
    decisions_by_case: Dict[str, Dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one_prompt,
                row,
                model=args.model,
                system=system,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                decision = future.result()
            except Exception as exc:
                decision = {
                    "case_id": row["case_id"],
                    "model": args.model,
                    "decision": "NO TRADE",
                    "raw_response": "",
                    "finish_reason": None,
                    "usage": None,
                    "error": repr(exc),
                }
            decisions_by_case[row["case_id"]] = decision
            completed += 1
            if completed % args.progress_every == 0 or completed == len(rows):
                print(f"completed={completed}/{len(rows)}")
    decisions = [decisions_by_case[row["case_id"]] for row in rows]
    with out_path.open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"decisions={out_path}")


def payoff(decision: str, realized_return_pct: float) -> float:
    if decision == "BUY":
        return realized_return_pct
    if decision == "SHORT":
        return -realized_return_pct
    return 0.0


Policy = Callable[[Dict[str, Any], int], str]


def always_buy(_: Dict[str, Any], __: int) -> str:
    return "BUY"


def always_short(_: Dict[str, Any], __: int) -> str:
    return "SHORT"


def always_no_trade(_: Dict[str, Any], __: int) -> str:
    return "NO TRADE"


def perfect_hindsight(row: Dict[str, Any], _: int) -> str:
    ret = row["realized_return_pct"]
    return "BUY" if ret > 0 else "SHORT" if ret < 0 else "NO TRADE"


def source_label(row: Dict[str, Any], _: int) -> str:
    call = row["hidden_source_call"]
    return "BUY" if call == "UP" else "SHORT" if call == "DOWN" else "NO TRADE"


def seeded_coin(seed: int) -> Policy:
    rng = random.Random(seed)
    cache: Dict[int, str] = {}

    def decide(_: Dict[str, Any], idx: int) -> str:
        if idx not in cache:
            cache[idx] = rng.choice(["BUY", "SHORT"])
        return cache[idx]

    return decide


def momentum_rule(row: Dict[str, Any], _: int) -> str:
    day = row["features"]["day_change_pct"]
    if day > 0.4:
        return "BUY"
    if day < -0.4:
        return "SHORT"
    return "NO TRADE"


def score_rows(keys: List[Dict[str, Any]], decisions: Dict[str, str], name: str) -> Dict[str, Any]:
    total = trades = correct = wrong = neutral = 0
    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(keys, start=1):
        decision = decisions[row["case_id"]]
        ret = payoff(decision, row["realized_return_pct"])
        total += ret
        actual = row["ground_truth_decision"]
        if decision == "NO TRADE":
            neutral += 1
        else:
            trades += 1
            if decision == actual:
                correct += 1
            else:
                wrong += 1
        if len(samples) < 5:
            samples.append({"case_id": row["case_id"], "decision": decision, "return_pct": row["realized_return_pct"], "pnl": round(ret, 2)})
    return {
        "name": name,
        "return_pct": round(total, 2),
        "trades": trades,
        "correct": correct,
        "wrong": wrong,
        "neutral": neutral,
        "directional_accuracy_pct": round(100 * correct / trades, 2) if trades else None,
        "samples": samples,
    }


def score(args: argparse.Namespace) -> None:
    keys = load_jsonl(Path(args.answer_key))
    model_rows = load_jsonl(Path(args.decisions))
    model_decisions = {row["case_id"]: row["decision"] for row in model_rows}
    missing = [row["case_id"] for row in keys if row["case_id"] not in model_decisions]
    if missing:
        raise SystemExit(f"Missing decisions for {len(missing)} cases; first={missing[0]}")
    policies: List[tuple[str, Dict[str, str]]] = [
        (args.model_label, model_decisions),
    ]
    baseline_defs: List[tuple[str, Policy]] = [
        ("Perfect hindsight", perfect_hindsight),
        ("Source-label oracle", source_label),
        ("Momentum", momentum_rule),
        ("Always Buy (brick)", always_buy),
        ("Seeded coin flip", seeded_coin(args.seed + 1)),
        ("Always No Trade", always_no_trade),
        ("Always Short", always_short),
    ]
    for name, policy in baseline_defs:
        policies.append((name, {row["case_id"]: policy(row, idx) for idx, row in enumerate(keys, start=1)}))
    board = [score_rows(keys, decs, name) for name, decs in policies]
    board.sort(key=lambda row: row["return_pct"], reverse=True)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(
        json.dumps(
            {
                "answer_key": str(Path(args.answer_key).resolve()),
                "decisions": str(Path(args.decisions).resolve()),
                "n": len(keys),
                "board": board,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = [
        "# Blind Decision Eval Results",
        "",
        f"Cases: `{len(keys)}`",
        f"Decisions: `{Path(args.decisions).resolve()}`",
        "",
        "## Final Board",
        "",
    ]
    for idx, row in enumerate(board, start=1):
        prefix = medals.get(idx, f"{idx}.")
        sign = "+" if row["return_pct"] > 0 else ""
        acc = row["directional_accuracy_pct"]
        acc_s = "n/a" if acc is None else f"{acc:.1f}%"
        lines.append(f"{prefix} {row['name']} -> {sign}{row['return_pct']:.2f}% ({row['trades']} trades, dir acc {acc_s})")
    by_name = {row["name"]: row for row in board}
    short = by_name["Always Short"]
    buy = by_name["Always Buy (brick)"]
    perfect = by_name["Perfect hindsight"]
    model = by_name[args.model_label]
    coin = by_name["Seeded coin flip"]
    worse_than_short = [row for row in board if row["return_pct"] < short["return_pct"]]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"Always shorting was {short['return_pct']:.2f}%.",
            f"Always buying was {buy['return_pct']:.2f}%.",
            f"Perfect hindsight was {perfect['return_pct']:+.2f}%.",
            f"{args.model_label} was {model['return_pct']:+.2f}% versus the seeded coin flip at {coin['return_pct']:+.2f}%.",
            f"{len(worse_than_short)} policy lines did worse than blindly shorting everything.",
            "",
            "Validity: model decisions were generated from prompts only; the scorer joined the answer key afterward.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)
    print(out_json)
    for idx, row in enumerate(board, start=1):
        print(f"{idx:02d} {row['name']}: {row['return_pct']:+.2f}% trades={row['trades']} acc={row['directional_accuracy_pct']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind Trevor-style decision eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--source", default=str(DEFAULT_SOURCE))
    g.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    g.add_argument("--run-id", default="blind-" + now_slug())
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=generate)

    r = sub.add_parser("run-model")
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    r.add_argument("--agent-prompt", default=None)
    r.add_argument("--max-tokens", type=int, default=40)
    r.add_argument("--timeout", type=int, default=120)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--progress-every", type=int, default=10)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=run_model)

    s = sub.add_parser("score")
    s.add_argument("--answer-key", required=True)
    s.add_argument("--decisions", required=True)
    s.add_argument("--model-label", default="OpenAI model + v5 prompt")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--out-json", required=True)
    s.add_argument("--out-md", required=True)
    s.set_defaults(func=score)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
