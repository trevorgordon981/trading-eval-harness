#!/usr/bin/env python3
"""Regression tests for eval_finetuned.py covering EV01, EV02, EV04, EV05, EV07, EV08, EV09."""

import hashlib
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import eval_finetuned as ef
import gauntlet_contract as gc
import judge_blind as jb


def _start_server(responder, model_id="mock-model"):
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
            resp = responder(body)
            if isinstance(resp, tuple):
                code, payload_obj = resp
            else:
                code, payload_obj = 200, resp
            payload = json.dumps(payload_obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, "http://127.0.0.1:%d/v1/chat/completions" % srv.server_address[1]


def _chat_resp(content, finish_reason="stop", tool_calls=None, tokens=10):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"finish_reason": finish_reason, "message": msg}],
        "usage": {"completion_tokens": tokens},
    }


def _make_heldout_file(path, n=10):
    moves = ["QUIET", "NORMAL", "ELEVATED", "EXPLOSIVE"]
    calls = ["UP", "DOWN", "NEUTRAL"]
    vols = ["EXPANDING", "STABLE", "CONTRACTING"]
    vsivs = ["RICH", "FAIR", "CHEAP"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            gt = {
                "move": moves[i % len(moves)],
                "call": calls[i % len(calls)],
                "vol_change": vols[i % len(vols)],
                "vs_iv": vsivs[i % len(vsivs)],
                "exp_move_pct": float(2 + i),
                "conviction": 8 if i % 2 == 0 else 3,
            }
            row = {
                "id": "synth_%02d" % i,
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": "CASE_%02d market snapshot" % i},
                    {"role": "assistant", "content": json.dumps(gt)},
                ],
            }
            f.write(json.dumps(row) + "\n")


class TestEV01ScoringDenominators(unittest.TestCase):
    """EV01: 1/10 parsed-correct FT MUST NEVER defeat 8/10 BASE."""

    def test_one_of_ten_ft_cannot_defeat_eight_of_ten_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            heldout = os.path.join(tmp, "heldout10.jsonl")
            _make_heldout_file(heldout, n=10)
            out_json = os.path.join(tmp, "results.json")

            gt_by_case = {}
            with open(heldout, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    r = json.loads(line)
                    gt_by_case["CASE_%02d" % idx] = json.loads(r["messages"][2]["content"])

            def base_responder(body):
                user = body["messages"][-1]["content"]
                tag = user.split()[0]
                idx = int(tag.split("_")[1])
                ans = dict(gt_by_case[tag])
                # BASE answers all 10, gets 8/10 right (misses 8 and 9)
                if idx >= 8:
                    ans["move"] = "QUIET" if ans["move"] != "QUIET" else "EXPLOSIVE"
                    ans["call"] = "DOWN" if ans["call"] != "DOWN" else "UP"
                return _chat_resp(json.dumps(ans))

            def ft_responder(body):
                user = body["messages"][-1]["content"]
                tag = user.split()[0]
                idx = int(tag.split("_")[1])
                # FT answers ONLY case 0 validly (100% conditional), fails parse on 9/10
                if idx == 0:
                    return _chat_resp(json.dumps(gt_by_case[tag]))
                return _chat_resp("not valid json output at all")

            b_srv, b_url = _start_server(base_responder, "base-m")
            f_srv, f_url = _start_server(ft_responder, "ft-m")
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    results = ef.main([
                        "--base-url", b_url,
                        "--ft-url", f_url,
                        "--mode", "held-out",
                        "--heldout", heldout,
                        "--out", out_json,
                    ])
                output = buf.getvalue()
            finally:
                b_srv.shutdown(); b_srv.server_close()
                f_srv.shutdown(); f_srv.server_close()

            ho = results["held-out"]
            # Every scenario persisted as a terminal row (10 for BASE, 10 for FT)
            self.assertEqual(len(ho["BASE"]["rows"]), 10)
            self.assertEqual(len(ho["FT"]["rows"]), 10)

            # Headline all-case accuracy uses full denominator: 80% BASE vs 10% FT
            self.assertAlmostEqual(ho["BASE"]["move_acc"], 0.8)
            self.assertAlmostEqual(ho["FT"]["move_acc"], 0.1)
            self.assertAlmostEqual(ho["FT"]["move_acc_parsed"], 1.0)
            self.assertAlmostEqual(ho["BASE"]["parse_rate"], 1.0)
            self.assertAlmostEqual(ho["FT"]["parse_rate"], 0.1)

            # Explicit failure coverage census
            fc = ho["FT"]["failure_census"]
            self.assertEqual(fc["attempted"], 10)
            self.assertEqual(fc["succeeded"], 1)
            self.assertEqual(fc["failed"], 9)
            self.assertEqual(fc["by_reason"]["parse_error"], 9)

            # Paired uncertainty computed on immutable case_ids
            self.assertIn("_paired", ho)
            self.assertEqual(ho["_paired"]["shared_n"], 10)
            self.assertAlmostEqual(ho["_paired"]["move"]["delta"], -0.7)

            # Verdict MUST NOT draw ✓FT or claim +20.0pp!
            self.assertNotIn("✓FT", output)
            self.assertNotIn("+20.0pp", output)
            self.assertIn("✗FT", output)
            self.assertIn("-70.0pp", output)
            self.assertIn("RELIABILITY", output)


class TestEV02ObjectiveChecks(unittest.TestCase):
    """EV02: Wrong sign, empty, refused, error, and fake tool replies MUST fail."""

    def test_signed_number_check_rejects_wrong_sign(self):
        chk_pos = {"type": "equals_number", "value": 3}
        chk_neg = {"type": "equals_number", "value": -3}

        self.assertFalse(ef.run_check(chk_pos, "-3")[1])
        self.assertFalse(ef.run_check(chk_pos, "The answer is -3.")[1])
        self.assertFalse(ef.run_check(chk_pos, "- 3")[1])
        self.assertFalse(ef.run_check(chk_pos, "-$3")[1])
        self.assertFalse(ef.run_check(chk_pos, "$-3")[1])
        self.assertFalse(ef.run_check(chk_pos, "negative 3")[1])
        self.assertFalse(ef.run_check(chk_pos, "\u22123")[1])
        self.assertTrue(ef.run_check(chk_pos, "+3")[1])
        self.assertTrue(ef.run_check(chk_pos, "The answer is 3.")[1])

        self.assertTrue(ef.run_check(chk_neg, "-3")[1])
        self.assertTrue(ef.run_check(chk_neg, "The result is -$3.00")[1])
        self.assertFalse(ef.run_check(chk_neg, "+3")[1])
        self.assertFalse(ef.run_check(chk_neg, "3")[1])

    def test_decimal_and_currency_precision(self):
        chk = {"type": "equals_number", "value": 298.29}
        self.assertTrue(ef.run_check(chk, "Breakeven is $298.29.")[1])
        self.assertFalse(ef.run_check(chk, "Breakeven is $298.")[1])
        self.assertFalse(ef.run_check(chk, "Breakeven is -$298.29.")[1])

        chk_cap = {"type": "equals_number", "value": 1200}
        self.assertTrue(ef.run_check(chk_cap, "Max size is $1,200.")[1])
        self.assertFalse(ef.run_check(chk_cap, "-$1,200")[1])

    def test_empty_response_fails_all_content_checks(self):
        for text in ("", "   ", "\n\t"):
            self.assertFalse(ef.run_check({"type": "non_empty"}, text)[1])
            self.assertFalse(ef.run_check({"type": "contains_none", "values": ["ARKK"]}, text)[1])
            self.assertFalse(ef.run_check({"type": "regex_none", "patterns": ["bad"]}, text)[1])
            self.assertFalse(ef.run_check({"type": "max_sentences", "n": 2}, text)[1])
            self.assertFalse(ef.run_check({"type": "equals_number", "value": 0}, text)[1])

    def test_refusal_and_prose_markers_fail_tool_intent_and_content_checks(self):
        refusal = "I'm sorry, I cannot invoke the function get_quote or emit a tool_call."
        self.assertFalse(ef.run_check({"type": "tool_intent", "name": "get_quote"}, refusal, [])[1])
        self.assertFalse(ef.run_check({"type": "contains_none", "values": ["ARKK"]}, refusal, [])[1])
        self.assertFalse(ef.run_check({"type": "non_empty"}, refusal, [])[1])
        self.assertFalse(ef.run_check({"type": "max_sentences", "n": 3}, refusal, [])[1])

        # Structured tool call with matching name passes; wrong name fails
        good_tc = [{"id": "c1", "type": "function", "function": {"name": "get_quote", "arguments": "{}"}}]
        wrong_tc = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
        self.assertTrue(ef.run_check({"type": "tool_intent", "name": "get_quote"}, "", good_tc)[1])
        self.assertFalse(ef.run_check({"type": "tool_intent", "name": "get_quote"}, "", wrong_tc)[1])
        # Legacy bool True without tool name must NOT satisfy a named tool_intent check
        self.assertFalse(ef.run_check({"type": "tool_intent", "name": "get_quote"}, "", True)[1])

    def test_http_error_and_content_filter_fail_all_objective_checks(self):
        self.assertFalse(ef.run_check({"type": "equals_number", "value": 829}, "829", err="HTTP 500")[1])
        self.assertFalse(ef.run_check({"type": "contains_none", "values": ["ARKK"]}, "clean text", err="timed out")[1])
        self.assertFalse(ef.run_check({"type": "non_empty"}, "text", finish_reason="content_filter")[1])
        self.assertFalse(ef.run_check({"type": "equals_number", "value": 829}, "829", finish_reason="length")[1])

    def test_negated_and_ambiguous_numeric_answers_never_earn_credit(self):
        """EV02 / EV08: Negated target numbers and shotgun prose must NEVER pass equals_number."""
        chk_pos3 = {"type": "equals_number", "value": 3}
        chk_neg3 = {"type": "equals_number", "value": -3}
        chk_be = {"type": "equals_number", "value": 298.29}
        chk_cap = {"type": "equals_number", "value": 1200}

        # Exact EV02 / EV08 counterexample
        self.assertFalse(ef.run_check(chk_pos3, "The correct answer is -3, not +3.")[1])
        self.assertFalse(ef.run_check(chk_pos3, "The correct answer is -3, not +3")[1])
        self.assertTrue(ef.run_check(chk_neg3, "The correct answer is -3, not +3.")[1])

        # Direct negations of target values
        for negated in (
            "not +3",
            "The answer is not +3.",
            "It isn't +3.",
            "+3 is wrong.",
            "+3 is incorrect.",
            "Answer != +3",
            "Answer \u2260 +3",
            "Neither +3 nor -3.",
            "The answer is -3 instead of +3.",
            "Rather than +3, the result is -3.",
            "Wrong: +3. Right: -3.",
            "The answer is +3 (wrong), actually -3.",
        ):
            self.assertFalse(ef.run_check(chk_pos3, negated)[1], msg=f"should reject: {negated}")

        for negated_be in (
            "Breakeven is not 298.29.",
            "It is not $298.29.",
            "The breakeven isn't $298.29",
            "298.29 is incorrect.",
            "The breakeven is $290.00, not $298.29.",
        ):
            self.assertFalse(ef.run_check(chk_be, negated_be)[1], msg=f"should reject: {negated_be}")

        # Ambiguous / shotgun multi-number prose must fail closed
        self.assertFalse(ef.run_check(chk_pos3, "It could be -3 or +3.")[1])
        self.assertFalse(ef.run_check(chk_be, "$829 $1200 $298.29 18")[1])

        # Structured JSON final-answer parsing (strict scalar field)
        self.assertTrue(ef.run_check(chk_pos3, '{"answer": 3}')[1])
        self.assertFalse(ef.run_check(chk_pos3, '{"answer": -3}')[1])
        self.assertTrue(ef.run_check(chk_be, '{"final_answer": 298.29}')[1])
        self.assertFalse(ef.run_check(chk_pos3, '{"answer": true}')[1])
        self.assertFalse(ef.run_check(chk_pos3, '{"answer": null}')[1])
        self.assertFalse(ef.run_check(chk_pos3, '{}')[1])
        self.assertFalse(ef.run_check(chk_pos3, '{"answer": "not +3"}')[1])

        # Legitimate equations and single final assertions still pass
        self.assertTrue(ef.run_check(chk_be, "290 + 8.29 = $298.29.")[1])
        self.assertTrue(ef.run_check(chk_cap, "The answer is $1,200 (not $1,000).")[1])
        self.assertTrue(ef.run_check(chk_pos3, "It's not -3, it's +3.")[1])


class TestEV04BaseOnlyMode(unittest.TestCase):
    """EV04: Same-URL base-only path must write results AND complete report() cleanly."""

    def test_same_url_base_only_heldout_and_all_complete_without_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            heldout = os.path.join(tmp, "heldout5.jsonl")
            _make_heldout_file(heldout, n=5)
            out_json = os.path.join(tmp, "base_only_results.json")

            def responder(body):
                user = body["messages"][-1]["content"]
                if body.get("tools"):
                    return _chat_resp("", finish_reason="tool_calls", tool_calls=[
                        {"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                    ])
                if "CASE_" in user:
                    return _chat_resp(json.dumps({
                        "move": "QUIET", "call": "UP", "vol_change": "EXPANDING",
                        "vs_iv": "RICH", "exp_move_pct": 2.0, "conviction": 8,
                    }))
                if "14 - 27" in user:
                    return _chat_resp("-13")
                if "250 requests" in user:
                    return _chat_resp("300")
                if "primary colors" in user:
                    return _chat_resp("Red\nGreen\nBlue")
                if "deterministic" in user:
                    return _chat_resp("Deterministic tests are reproducible.")
                if "bin A" in user:
                    return _chat_resp("YES")
                return _chat_resp("JPM - Financials\nXOM - Energy\nCAT - Industrials\nCOST - Staples\nMSFT - Tech\nUNP - Rail\nHD - Retail\nNEE - Utilities\n$829 $1200 $298.29 18")

            srv, url = _start_server(responder, "base-only-model")
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    results = ef.main([
                        "--base-url", url,
                        "--ft-url", url,  # documented same-URL base-only invocation
                        "--mode", "all",
                        "--heldout", heldout,
                        "--out", out_json,
                    ])
                output = buf.getvalue()
            finally:
                srv.shutdown(); srv.server_close()

            self.assertTrue(os.path.isfile(out_json))
            self.assertIn("BASE-ONLY — SCORECARD", output)
            self.assertIn("BASE", results["held-out"])
            self.assertNotIn("FT", results["held-out"])
            self.assertEqual(results["trading-gauntlet"]["FT"], {})
            self.assertEqual(results["gordon"]["FT"], {})

    def test_omitted_ft_url_works_as_base_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            heldout = os.path.join(tmp, "heldout2.jsonl")
            _make_heldout_file(heldout, n=2)
            out_json = os.path.join(tmp, "omitted_ft.json")
            srv, url = _start_server(lambda b: _chat_resp(json.dumps({
                "move": "QUIET", "call": "UP", "vol_change": "EXPANDING",
                "vs_iv": "RICH", "exp_move_pct": 2.0, "conviction": 7,
            })), "base-omitted")
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    ef.main([
                        "--base-url", url,
                        "--mode", "held-out",
                        "--heldout", heldout,
                        "--out", out_json,
                    ])
                output = buf.getvalue()
            finally:
                srv.shutdown(); srv.server_close()
            self.assertIn("BASE-ONLY — SCORECARD", output)


class TestEV05DisjointGeneralAndTimeouts(unittest.TestCase):
    """EV05: General regression is strictly disjoint from trading and withholds verdict on timeouts."""

    def test_trading_and_general_profiles_are_strictly_disjoint(self):
        trading = ef.get_trading_batteries()
        general = ef.get_general_batteries(trading_batteries=trading)
        self.assertTrue(len(trading) >= 3)
        self.assertTrue(len(general) >= 1)
        self.assertEqual(set(trading) & set(general), set())

    def test_all_mode_with_timeouts_withholds_verdict_and_reports_census(self):
        # Monkeypatch chat_text to simulate deterministic endpoint timeouts fast
        orig_chat_text = ef.chat_text
        orig_detect = ef.detect_model

        def fake_timeout(endpoint, messages, **kw):
            return ("", 0, None, [], "timed out")

        ef.chat_text = fake_timeout
        ef.detect_model = lambda url, timeout=10: "timeout-model"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                heldout = os.path.join(tmp, "heldout3.jsonl")
                _make_heldout_file(heldout, n=3)
                out_json = os.path.join(tmp, "timeout_results.json")
                buf = io.StringIO()
                with redirect_stdout(buf):
                    results = ef.main([
                        "--base-url", "http://127.0.0.1:1/v1/chat/completions",
                        "--ft-url", "http://127.0.0.1:2/v1/chat/completions",
                        "--mode", "all",
                        "--heldout", heldout,
                        "--out", out_json,
                    ])
                output = buf.getvalue()
        finally:
            ef.chat_text = orig_chat_text
            ef.detect_model = orig_detect

        # Trading and Gordon keys must be strictly disjoint
        t_keys = set(results["trading-gauntlet"]["BASE"].keys())
        g_keys = set(results["gordon"]["BASE"].keys())
        self.assertTrue(t_keys)
        self.assertTrue(g_keys)
        self.assertEqual(t_keys & g_keys, set())

        # Zero objective passes on timeouts (including contains_none on tickers!)
        t_agg = ef.agg_gauntlet(results["trading-gauntlet"]["BASE"])
        g_agg = ef.agg_gauntlet(results["gordon"]["BASE"])
        self.assertEqual(t_agg["hp"], 0)
        self.assertEqual(g_agg["hp"], 0)
        self.assertGreater(t_agg["failure_census"]["by_reason"]["timeout"], 0)
        self.assertGreater(g_agg["failure_census"]["by_reason"]["timeout"], 0)

        # Must NEVER claim general ability held on timeouts!
        self.assertNotIn("general ability held", output)
        self.assertIn("VERDICT WITHHELD", output)


class TestEV07ContractAndBlindJudgeIntegration(unittest.TestCase):
    """EV07: eval_finetuned wires hash-bound contracts and blind judge packets."""

    def test_contract_and_blind_judge_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_real = os.path.realpath(tmp)
            contract_dir = os.path.join(tmp_real, "contracts")
            judge_dir = os.path.join(tmp_real, "judge")
            out_json = os.path.join(tmp_real, "results.json")

            def responder(body):
                user = body["messages"][-1]["content"]
                if body.get("tools"):
                    return _chat_resp("", finish_reason="tool_calls", tool_calls=[
                        {"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                    ])
                if "14 - 27" in user:
                    return _chat_resp("-13")
                if "250 requests" in user:
                    return _chat_resp("300")
                if "primary colors" in user:
                    return _chat_resp("Red\nGreen\nBlue")
                if "deterministic" in user:
                    return _chat_resp("Deterministic tests are reproducible.")
                if "bin A" in user:
                    return _chat_resp("YES")
                return _chat_resp("JPM - Fin\nXOM - En\nCAT - Ind\nCOST - Cons\nMSFT - Tech\nUNP - Rail\nHD - Ret\nNEE - Util\n829 1200 298.29 18")

            b_srv, b_url = _start_server(responder, "base-v1")
            f_srv, f_url = _start_server(responder, "ft-v1")
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    results = ef.main([
                        "--base-url", b_url,
                        "--ft-url", f_url,
                        "--mode", "all",
                        "--out", out_json,
                        "--contract-dir", contract_dir,
                        "--blind-judge-dir", judge_dir,
                    ])
            finally:
                b_srv.shutdown(); b_srv.server_close()
                f_srv.shutdown(); f_srv.server_close()

            # Hash-bound contracts created & verified for both BASE and FT
            self.assertIn("_contracts", results)
            for side in ("BASE", "FT"):
                rec = results["_contracts"][side]
                self.assertTrue(rec["verified"])
                self.assertEqual(rec["profile"], "standard")
                gc.verify_complete(
                    Path(REPO_ROOT),
                    Path(rec["run_manifest"]),
                    Path(rec["completion_receipt"]),
                    Path(contract_dir),
                )

            # Tampering with a result file fails contract verification
            tamper_path = Path(contract_dir) / "trademath.BASE.json"
            tamper_path.write_text(tamper_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(gc.ContractError):
                gc.verify_complete(
                    Path(REPO_ROOT),
                    Path(results["_contracts"]["BASE"]["run_manifest"]),
                    Path(results["_contracts"]["BASE"]["completion_receipt"]),
                    Path(contract_dir),
                )

            # Blind judge packet exists and leaks no model/label/key
            bj = results["_blind_judge"]
            self.assertGreater(bj["count"], 0)
            with open(bj["pending_path"], encoding="utf-8") as f:
                pending_rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(pending_rows), bj["count"])
            with open(bj["map_path"], encoding="utf-8") as f:
                umap = json.load(f)
            for pr in pending_rows:
                self.assertIn("answer_uid", pr)
                self.assertIn(pr["answer_uid"], umap)
                for forbidden in ("model", "label", "key"):
                    self.assertNotIn(forbidden, pr)

            # Every gordon-result.v2 row records effective model and prompt_sha256
            gr_res = json.loads((Path(contract_dir) / "general_reasoning.BASE.json").read_text(encoding="utf-8"))
            for row in gr_res["results"]:
                self.assertEqual(row["model"], "base-v1")
                self.assertRegex(row["prompt_sha256"], r"^[0-9a-f]{64}$")

    def test_mid_run_battery_mutation_fails_contract_verification(self):
        """EV07: Mutating a same-ID battery prompt mid-run MUST fail closed (no verified receipt)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            shutil.copytree(Path(REPO_ROOT) / "batteries", root / "batteries")
            shutil.copyfile(Path(REPO_ROOT) / "battery_manifest.json", root / "battery_manifest.json")
            battery = root / "batteries" / "general_reasoning.json"
            pre_bytes = battery.read_bytes()
            pre_sha = hashlib.sha256(pre_bytes).hexdigest()

            prompts_sent = []
            mutated = False

            def responder(url, messages, **kw):
                nonlocal mutated
                prompts_sent.append(messages[-1]["content"])
                if not mutated:
                    original = json.loads(battery.read_text(encoding="utf-8"))
                    original[0]["prompt"] = "NEW_PRESERVED_ID_FIXTURE"
                    battery.write_text(json.dumps(original), encoding="utf-8")
                    mutated = True
                return "sample response", 3, "stop", [], None

            contract_dir = root / "contracts"
            with patch.object(ef, "GG_ROOT", str(root)), \
                 patch.object(ef, "BATT_DIR", str(root / "batteries")), \
                 patch.object(ef, "detect_model", return_value="fixture"), \
                 patch.object(ef, "chat_text", responder), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(gc.ContractError):
                    ef.main([
                        "--base-url", "http://127.0.0.1:4321/v1/chat/completions",
                        "--mode", "gordon",
                        "--contract-dir", str(contract_dir),
                        "--out", str(root / "output.json"),
                    ])

            post_sha = hashlib.sha256(battery.read_bytes()).hexdigest()
            self.assertNotEqual(pre_sha, post_sha)
            # Frozen pre-call bytes were bound in run.BASE.json and sent to the model
            run_manifest = json.loads((contract_dir / "run.BASE.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["batteries"][0]["sha256"], pre_sha)
            self.assertNotEqual(run_manifest["batteries"][0]["sha256"], post_sha)
            self.assertTrue(prompts_sent)
            self.assertNotIn("NEW_PRESERVED_ID_FIXTURE", prompts_sent)
            # No completion receipt may exist for the tampered run
            self.assertFalse((contract_dir / "completion.BASE.json").exists())

    def test_mid_run_heldout_mutation_fails_when_contract_dir_set(self):
        """EV07: Mutating the held-out JSONL mid-run with --contract-dir MUST fail closed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.path.realpath(tmp))
            heldout = root / "heldout3.jsonl"
            _make_heldout_file(str(heldout), n=3)
            mutated = False

            def responder(url, messages, **kw):
                nonlocal mutated
                if not mutated:
                    heldout.write_text(heldout.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    mutated = True
                return json.dumps({
                    "move": "QUIET", "call": "UP", "vol_change": "EXPANDING",
                    "vs_iv": "RICH", "exp_move_pct": 2.0, "conviction": 8,
                }), 12, "stop", [], None

            with patch.object(ef, "detect_model", return_value="fixture"), \
                 patch.object(ef, "chat_text", responder), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(gc.ContractError):
                    ef.main([
                        "--base-url", "http://127.0.0.1:4321/v1/chat/completions",
                        "--mode", "held-out",
                        "--heldout", str(heldout),
                        "--contract-dir", str(root / "contracts"),
                        "--out", str(root / "output.json"),
                    ])


class TestEV09HeldoutSchemaValidation(unittest.TestCase):
    """EV09: Empty, malformed, partial, or truncated forecast JSONs MUST fail valid coverage."""

    def test_empty_object_with_length_finish_fails_valid_coverage_gate(self):
        """Exact EV09 counterexample: 10x '{}' with finish_reason='length' -> 0% valid coverage."""
        with tempfile.TemporaryDirectory() as tmp:
            heldout = os.path.join(tmp, "heldout10.jsonl")
            _make_heldout_file(heldout, n=10)
            out_json = os.path.join(tmp, "empty_len.json")

            def empty_length(url, messages, **kw):
                return "{}", 0, "length", [], None

            with patch.object(ef, "detect_model", return_value="fixture"), \
                 patch.object(ef, "chat_text", empty_length), \
                 redirect_stdout(io.StringIO()) as buf:
                results = ef.main([
                    "--base-url", "http://127.0.0.1:4321/v1/chat/completions",
                    "--ft-url", "http://127.0.0.1:4322/v1/chat/completions",
                    "--mode", "held-out",
                    "--heldout", heldout,
                    "--out", out_json,
                ])
                output = buf.getvalue()

            ho = results["held-out"]
            for side in ("BASE", "FT"):
                s = ho[side]
                self.assertEqual(s["parse_rate"], 0.0)
                self.assertEqual(s["valid_coverage"], 0.0)
                self.assertEqual(s["valid_count"], 0)
                self.assertEqual(s["move_acc"], 0.0)
                self.assertEqual(s["dir_acc"], 0.0)
                # Loose JSON parse diagnostic tracked separately
                self.assertEqual(s["json_parsed"], 10)
                self.assertEqual(s["json_parse_rate"], 1.0)
                fc = s["failure_census"]
                self.assertEqual(fc["succeeded"], 0)
                self.assertEqual(fc["failed"], 10)
                self.assertEqual(fc["coverage_rate"], 0.0)
                self.assertEqual(fc["by_reason"]["truncated"], 10)
                for row in s["rows"]:
                    self.assertFalse(row["parsed"])
                    self.assertFalse(row["schema_valid"])
                    self.assertTrue(row["json_parsed"])
                    self.assertEqual(row["status"], "truncated")
                    self.assertIsNone(row["pred"])
            self.assertIn("VERDICT WITHHELD", output)

    def test_empty_and_malformed_objects_with_stop_fail_as_schema_error(self):
        """Empty {}, missing required fields, and invalid types all fail with schema_error."""
        bad_payloads = [
            "{}",  # empty object
            json.dumps({"move": "QUIET", "call": "UP"}),  # missing required fields
            json.dumps({  # invalid enum value
                "move": "MEGA", "call": "UP", "vol_change": "STABLE",
                "vs_iv": "FAIR", "exp_move_pct": 2.0, "conviction": 5,
            }),
            json.dumps({  # wrong type (bool) for exp_move_pct
                "move": "QUIET", "call": "UP", "vol_change": "STABLE",
                "vs_iv": "FAIR", "exp_move_pct": True, "conviction": 5,
            }),
            json.dumps({  # wrong type (string) for exp_move_pct
                "move": "QUIET", "call": "UP", "vol_change": "STABLE",
                "vs_iv": "FAIR", "exp_move_pct": "bad", "conviction": 5,
            }),
            json.dumps({  # out-of-range conviction
                "move": "QUIET", "call": "UP", "vol_change": "STABLE",
                "vs_iv": "FAIR", "exp_move_pct": 2.0, "conviction": 99,
            }),
            json.dumps({  # non-string call type
                "move": "QUIET", "call": ["UP"], "vol_change": "STABLE",
                "vs_iv": "FAIR", "exp_move_pct": 2.0, "conviction": 5,
            }),
            json.dumps({  # negative exp_move_pct
                "move": "QUIET", "call": "UP", "vol_change": "STABLE",
                "vs_iv": "FAIR", "exp_move_pct": -1.5, "conviction": 5,
            }),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            heldout = os.path.join(tmp, "heldout8.jsonl")
            _make_heldout_file(heldout, n=len(bad_payloads))
            out_json = os.path.join(tmp, "schema_fail.json")
            call_idx = {"n": 0}

            def bad_responder(url, messages, **kw):
                payload = bad_payloads[call_idx["n"] % len(bad_payloads)]
                call_idx["n"] += 1
                return payload, 10, "stop", [], None

            with patch.object(ef, "detect_model", return_value="fixture"), \
                 patch.object(ef, "chat_text", bad_responder), \
                 redirect_stdout(io.StringIO()) as buf:
                results = ef.main([
                    "--base-url", "http://127.0.0.1:4321/v1/chat/completions",
                    "--mode", "held-out",
                    "--heldout", heldout,
                    "--out", out_json,
                ])
                output = buf.getvalue()

            b = results["held-out"]["BASE"]
            self.assertEqual(b["parse_rate"], 0.0)
            self.assertEqual(b["valid_coverage"], 0.0)
            self.assertEqual(b["json_parsed"], len(bad_payloads))
            self.assertEqual(b["move_acc"], 0.0)
            self.assertEqual(b["dir_acc"], 0.0)
            self.assertEqual(b["failure_census"]["succeeded"], 0)
            self.assertEqual(b["failure_census"]["by_reason"]["schema_error"], len(bad_payloads))
            for row in b["rows"]:
                self.assertEqual(row["status"], "schema_error")
                self.assertFalse(row["parsed"])
                self.assertFalse(row["schema_valid"])
                self.assertTrue(row["schema_errors"])
                self.assertIsNone(row["pred"])
            self.assertIn("VERDICT WITHHELD", output)

    def test_valid_schema_with_length_finish_still_rejected_as_truncated(self):
        """Even if all 6 fields appear, finish_reason='length' is NOT an allowed terminal status."""
        with tempfile.TemporaryDirectory() as tmp:
            heldout = os.path.join(tmp, "heldout2.jsonl")
            _make_heldout_file(heldout, n=2)
            out_json = os.path.join(tmp, "trunc_valid.json")
            valid_json = json.dumps({
                "move": "QUIET", "call": "UP", "vol_change": "EXPANDING",
                "vs_iv": "RICH", "exp_move_pct": 2.0, "conviction": 8,
            })

            with patch.object(ef, "detect_model", return_value="fixture"), \
                 patch.object(ef, "chat_text", return_value=(valid_json, 160, "length", [], None)), \
                 redirect_stdout(io.StringIO()):
                results = ef.main([
                    "--base-url", "http://127.0.0.1:4321/v1/chat/completions",
                    "--mode", "held-out",
                    "--heldout", heldout,
                    "--out", out_json,
                ])

            b = results["held-out"]["BASE"]
            self.assertEqual(b["valid_coverage"], 0.0)
            self.assertEqual(b["move_acc"], 0.0)
            self.assertEqual(b["failure_census"]["by_reason"]["truncated"], 2)


if __name__ == "__main__":
    unittest.main()
