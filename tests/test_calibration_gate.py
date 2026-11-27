#!/usr/bin/env python3
"""Regression tests for calibration_gate.py CLI and loader (EV06)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "calibration_gate.py")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import calibration_gate as cg


def _run_cli(args):
    return subprocess.run(
        [sys.executable, SCRIPT] + list(args),
        capture_output=True,
        text=True,
    )


class TestCalibrationGateEV06(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_two_row_jsonl_file_loads_without_extra_data_error(self):
        p = os.path.join(self.tmp.name, "two_rows.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"conviction": 2, "correct": False}) + "\n")
            f.write(json.dumps({"conviction": 9, "correct": True}) + "\n")
        pairs = cg.load_pairs(p)
        self.assertEqual(pairs, [(2.0, False), (9.0, True)])

        proc = _run_cli(["--preds", p, "--json"])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("scored predictions: 2", proc.stdout)
        self.assertIn("GATE: INSUFFICIENT", proc.stdout)

    def test_jsonl_content_with_json_extension_fallback(self):
        p = os.path.join(self.tmp.name, "lines.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"conviction": 3, "pred": "UP", "gt": "DOWN"}) + "\n")
            f.write(json.dumps({"conviction": 8, "pred": "UP", "gt": "UP"}) + "\n")
        pairs = cg.load_pairs(p)
        self.assertEqual(pairs, [(3.0, False), (8.0, True)])

    def test_conf_scale_1_populates_low_and_high_groups_properly(self):
        """EV06: 0.1 / 0.9 fixture at --conf-scale 1 must populate BOTH low and high groups."""
        p = os.path.join(self.tmp.name, "prob_scale.jsonl")
        rows = []
        # 20 rows at conf=0.1 (10% acc -> 2 correct, 18 wrong)
        for i in range(20):
            rows.append({"conviction": 0.1, "correct": i < 2})
        # 20 rows at conf=0.9 (90% acc -> 18 correct, 2 wrong)
        for i in range(20):
            rows.append({"conviction": 0.9, "correct": i < 18})
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        proc = _run_cli(["--preds", p, "--conf-scale", "1", "--json"])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        # Parse JSON emitted at the end of stdout
        json_blob = proc.stdout.strip().splitlines()[-1]
        data = json.loads(json_blob)
        groups = data["reliability"]["groups"]
        self.assertEqual(groups["low"]["n"], 20)
        self.assertEqual(groups["high"]["n"], 20)
        self.assertAlmostEqual(groups["low"]["acc"], 0.10)
        self.assertAlmostEqual(groups["high"]["acc"], 0.90)
        self.assertLess(data["reliability"]["ece"], 0.01)
        self.assertEqual(data["verdict"], "PASS")

    def test_eval_finetuned_results_json_skips_unparsed_rows(self):
        p = os.path.join(self.tmp.name, "eval_results.json")
        rows = []
        for i in range(15):
            rows.append({
                "case_id": f"lo_{i}",
                "parsed": True,
                "err": None,
                "pred": {"call": "UP" if i < 3 else "DOWN", "conviction": 2},
                "gt": {"call": "UP"},
            })
        for i in range(15):
            rows.append({
                "case_id": f"hi_{i}",
                "parsed": True,
                "err": None,
                "pred": {"call": "UP" if i < 13 else "DOWN", "conviction": 9},
                "gt": {"call": "UP"},
            })
        # Add unparsed terminal row (as emitted by EV01 fix) — must be ignored cleanly
        rows.append({
            "case_id": "bad_0",
            "parsed": False,
            "err": None,
            "pred": None,
            "gt": {"call": "UP"},
        })
        payload = {"held-out": {"FT": {"rows": rows}}}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        proc = _run_cli(["--results", p, "--label", "FT", "--json"])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(data["reliability"]["n"], 30)
        self.assertEqual(data["verdict"], "PASS")

    def test_selftest_passes(self):
        proc = _run_cli(["--selftest"])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("ALL SELFTEST ASSERTIONS PASSED", proc.stdout)


if __name__ == "__main__":
    unittest.main()
