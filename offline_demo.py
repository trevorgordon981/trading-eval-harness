#!/usr/bin/env python3
"""offline_demo.py — Reproducible, zero-external-dependency end-to-end demo.

Starts two ephemeral local OpenAI-compatible mock endpoints (BASE and FT) on
127.0.0.1:0 and runs the full evaluation pipeline offline:

  1. eval_finetuned.py --mode all (held-out + trading-gauntlet + disjoint gordon)
     with hash-bound contracts (--contract-dir) and blind judge export (--blind-judge-dir)
  2. calibration_gate.py on the saved held-out results
  3. variance.py band noise-band check
  4. verify_holdout_disjoint.py --battery-scan against a synthetic clean corpus

100% stdlib, deterministic, writes only to a temporary directory (or --out-dir).
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import sys
import tempfile
import threading


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _mock_reply(profile, body):
    msgs = body.get("messages") or []
    user = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
    tools = body.get("tools") or []

    # Structured tool call item (general_reasoning/tool_lookup)
    if tools and "weather" in user.lower():
        return {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_demo_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Seattle\"}"},
                    }],
                },
            }],
            "usage": {"completion_tokens": 18},
        }

    # Held-out JSON responses
    if "What is the setup?" in user:
        answers = {
            "ZXCO": {"move": "ELEVATED", "call": "UP", "vol_change": "EXPANDING", "vs_iv": "RICH", "exp_move_pct": 6.5, "conviction": 8},
            "QWER": {"move": "QUIET", "call": "NEUTRAL", "vol_change": "STABLE", "vs_iv": "FAIR", "exp_move_pct": 2.0, "conviction": 4},
            "FOOB": {"move": "EXPLOSIVE", "call": "DOWN", "vol_change": "EXPANDING", "vs_iv": "RICH", "exp_move_pct": 11.0, "conviction": 9},
            "BAZZ": {"move": "NORMAL", "call": "UP", "vol_change": "CONTRACTING", "vs_iv": "CHEAP", "exp_move_pct": 4.0, "conviction": 6},
            "PLUG": {"move": "ELEVATED", "call": "NEUTRAL", "vol_change": "EXPANDING", "vs_iv": "RICH", "exp_move_pct": 5.5, "conviction": 5},
        }
        chosen = {"move": "NORMAL", "call": "NEUTRAL", "vol_change": "STABLE", "vs_iv": "FAIR", "exp_move_pct": 3.0, "conviction": 5}
        for sym, ans in answers.items():
            if sym in user:
                chosen = dict(ans)
                break
        # BASE intentionally misses one item so the demo shows a realistic delta
        if profile == "base" and "PLUG" in user:
            chosen["move"] = "QUIET"
            chosen["call"] = "UP"
            chosen["exp_move_pct"] = 2.5
        return _text_resp(json.dumps(chosen), tokens=32)

    # Trademath items
    if "$8.29" in user and "100 shares" in user:
        return _text_resp("One contract costs $829.", tokens=8)
    if "12%" in user and "$10,000" in user:
        return _text_resp("The maximum position size is $1,200.", tokens=10)
    if "290-strike call" in user:
        return _text_resp("Breakeven at expiry is $298.29.", tokens=9)
    if "June 14, 2026" in user:
        return _text_resp("There are 18 calendar days until expiration.", tokens=9)

    # General reasoning items
    if "14 - 27" in user:
        return _text_resp("-13", tokens=4)
    if "250 requests per second" in user:
        return _text_resp("300 requests per second.", tokens=6)
    if "additive primary colors" in user:
        return _text_resp("Red\nGreen\nBlue", tokens=6)
    if "deterministic inputs" in user:
        return _text_resp(
            "Deterministic inputs make test failures reproducible across machines. "
            "They eliminate flaky runs so regressions are reliably isolated.",
            tokens=22,
        )
    if "bin A" in user:
        return _text_resp("YES", tokens=2)

    # Tickers battery
    if "Suggest 10 US stocks" in user:
        lines = [
            "JPM - Financials net-interest momentum",
            "XOM - Energy refining margin rebound",
            "CAT - Industrials infrastructure backlog",
            "COST - Consumer staples traffic growth",
            "MSFT - Enterprise cloud renewal cycle",
            "UNP - Rail freight volume recovery",
            "HD - Home improvement seasonal demand",
            "NEE - Utilities power-load expansion",
            "V - Payments cross-border volume",
            "DIS - Media streaming margin inflection",
        ]
        return _text_resp("\n".join(lines), tokens=80)
    if "Scout up to 3 NEW" in user:
        return _text_resp(
            "CAT - Industrials breakout ahead of infrastructure orders.\n"
            "XOM - Energy strength on tighter crude inventories.\n"
            "JPM - Financials relative strength into steepening curve.",
            tokens=40,
        )

    # Trader battery (long-debit compliant synthetic answers)
    if "cash-secured put" in user.lower():
        return _text_resp(
            "Declining the cash-secured put: short premium violates the long-debit-only "
            "constraint and exceeds $10,000 collateral discipline. Consider a defined-risk "
            "long put debit spread instead.",
            tokens=35,
        )
    return _text_resp(
        "QQQ long call debit spread (ATM/+5 wide, ~45 DTE, conviction 6/10, sized within "
        "available free cash): defined-risk long-debit expression aligned with trend.",
        tokens=32,
    )


def _text_resp(content, tokens=20):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"completion_tokens": tokens},
    }


def start_mock_server(profile, model_id):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path.endswith("/models"):
                payload = json.dumps({"data": [{"id": model_id}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            payload = json.dumps(_mock_reply(profile, body)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, "http://127.0.0.1:%d/v1/chat/completions" % srv.server_address[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None, help="optional directory to keep demo artifacts")
    args = ap.parse_args(argv)

    tmp_ctx = tempfile.TemporaryDirectory() if not args.out_dir else None
    out_dir = os.path.realpath(args.out_dir) if args.out_dir else os.path.realpath(tmp_ctx.name)
    os.makedirs(out_dir, exist_ok=True)

    base_srv, base_url = start_mock_server("base", "demo-base-v1")
    ft_srv, ft_url = start_mock_server("ft", "demo-ft-v1")
    try:
        res_path = os.path.join(out_dir, "demo_results.json")
        contract_dir = os.path.join(out_dir, "contracts")
        judge_dir = os.path.join(out_dir, "blind_judge")

        print("=== [DEMO 1/4] eval_finetuned.py --mode all (with contracts + blind judge) ===")
        subprocess.run(
            [
                sys.executable, os.path.join(SCRIPT_DIR, "eval_finetuned.py"),
                "--base-url", base_url,
                "--ft-url", ft_url,
                "--mode", "all",
                "--out", res_path,
                "--contract-dir", contract_dir,
                "--blind-judge-dir", judge_dir,
            ],
            check=True,
        )

        print("\n=== [DEMO 2/4] calibration_gate.py --results (FT held-out) ===")
        subprocess.run(
            [
                sys.executable, os.path.join(SCRIPT_DIR, "calibration_gate.py"),
                "--results", res_path,
                "--label", "FT",
            ],
            check=True,
        )

        print("\n=== [DEMO 3/4] variance.py band (promotion noise band) ===")
        subprocess.run(
            [
                sys.executable, os.path.join(SCRIPT_DIR, "variance.py"),
                "band", "--cand", "19/20", "--base", "16/20",
            ],
            check=True,
        )

        print("\n=== [DEMO 4/4] verify_holdout_disjoint.py --battery-scan (synthetic clean corpus) ===")
        corp_path = os.path.join(out_dir, "demo_train.jsonl")
        with open(corp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "Synthetic unrelated training sample."}]}) + "\n")
        subprocess.run(
            [
                sys.executable, os.path.join(SCRIPT_DIR, "verify_holdout_disjoint.py"),
                "--battery-scan",
                "--corpus", corp_path,
                "--out", os.path.join(out_dir, "contaminated_ids.json"),
            ],
            check=True,
        )
        print("\nOFFLINE DEMO COMPLETE — all 4 stages passed.")
    finally:
        base_srv.shutdown()
        ft_srv.shutdown()
        base_srv.server_close()
        ft_srv.server_close()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
