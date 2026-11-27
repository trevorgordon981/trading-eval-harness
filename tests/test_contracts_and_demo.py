#!/usr/bin/env python3
"""Integration tests for gauntlet_contract.py, variance.py, and offline_demo.py."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gauntlet_contract as gc


class TestContractsAndOfflineDemo(unittest.TestCase):
    def test_gauntlet_contract_cli_lists_all_manifest_profiles(self):
        for profile in ("trading", "general", "standard", "smoke"):
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "gauntlet_contract.py"),
                    "list",
                    "--root", REPO_ROOT,
                    "--profile", profile,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(proc.stdout.strip())

    def test_gauntlet_contract_roundtrip_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(os.path.realpath(tmp))
            root_path = Path(os.path.realpath(REPO_ROOT))
            run_path = tmp_path / "run.BASE.json"
            comp_path = tmp_path / "completion.BASE.json"
            run = gc.create_run_manifest(
                root_path,
                profile="general",
                label="BASE",
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
                model_id="test-model",
                out_path=run_path,
            )
            for entry in run["batteries"]:
                rows = [
                    {
                        "id": item_id,
                        "sec": 0.1,
                        "finish": "stop",
                        "ctok": 5,
                        "tool_calls": False,
                        "err": None,
                        "text": "ok",
                    }
                    for item_id in entry["ids"]
                ]
                gc.write_battery_result_v2(
                    run, entry["name"], rows, tmp_path / f"{entry['name']}.BASE.json"
                )
            gc.complete_run_manifest(root_path, run_path, tmp_path, comp_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "gauntlet_contract.py"),
                    "verify-run",
                    "--root", str(root_path),
                    "--run-manifest", str(run_path),
                    "--completion", str(comp_path),
                    "--results-dir", str(tmp_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertEqual(proc.stdout.strip(), run["run_id"])

    def test_variance_band_cli(self):
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "variance.py"),
                "band", "--cand", "18/20", "--base", "10/20",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("BAND delta=", proc.stdout)
        self.assertIn("verdict=IMPROVE", proc.stdout)

    def test_offline_demo_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "offline_demo.py"),
                    "--out-dir", tmp,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertIn("OFFLINE DEMO COMPLETE", proc.stdout)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "demo_results.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "contracts", "completion.BASE.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "contracts", "completion.FT.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "blind_judge", "pending_judge.jsonl")))


if __name__ == "__main__":
    unittest.main()
