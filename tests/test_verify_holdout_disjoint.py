#!/usr/bin/env python3
"""Regression tests for verify_holdout_disjoint.py fail-closed contamination gate."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "verify_holdout_disjoint.py")


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _run_gate(args, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, SCRIPT] + list(args),
        capture_output=True,
        text=True,
        env=env,
    )


class TestVerifyHoldoutDisjointBatteryScan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_path = os.path.join(self.tmp.name, "contaminated_ids.json")

    def test_missing_corpus_path_fails_closed(self):
        missing = os.path.join(self.tmp.name, "does_not_exist")
        proc = _run_gate(["--battery-scan", "--corpus", missing, "--out", self.out_path])
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
        self.assertFalse(os.path.exists(self.out_path))

    def test_missing_corpus_glob_fails_closed(self):
        pattern = os.path.join(self.tmp.name, "missing_dir", "*.jsonl")
        proc = _run_gate(["--battery-scan", "--corpus", pattern, "--out", self.out_path])
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
        self.assertFalse(os.path.exists(self.out_path))

    def test_empty_corpus_directory_fails_closed(self):
        empty_dir = os.path.join(self.tmp.name, "empty_corpus_dir")
        os.makedirs(empty_dir)
        proc = _run_gate(["--battery-scan", "--corpus", empty_dir, "--out", self.out_path])
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
        self.assertFalse(os.path.exists(self.out_path))

    def test_zero_row_corpus_file_fails_closed(self):
        corp_dir = os.path.join(self.tmp.name, "zero_row_dir")
        os.makedirs(corp_dir)
        empty_train = os.path.join(corp_dir, "train.jsonl")
        open(empty_train, "w", encoding="utf-8").close()

        for spec in (corp_dir, empty_train):
            with self.subTest(spec=spec):
                proc = _run_gate(["--battery-scan", "--corpus", spec, "--out", self.out_path])
                self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
                self.assertNotIn("VERDICT: CLEAN", proc.stdout)
                self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
                self.assertFalse(os.path.exists(self.out_path))

    def test_corpus_without_user_messages_fails_closed(self):
        corpus_file = os.path.join(self.tmp.name, "no_users.jsonl")
        _write_jsonl(
            corpus_file,
            [
                {"messages": [{"role": "assistant", "content": "only assistant"}]},
                {"messages": [{"role": "user", "content": "   "}]},
            ],
        )
        proc = _run_gate(["--battery-scan", "--corpus", corpus_file, "--out", self.out_path])
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)

    def test_mixed_valid_and_missing_corpus_fails_closed(self):
        valid_dir = os.path.join(self.tmp.name, "valid_dir")
        _write_jsonl(
            os.path.join(valid_dir, "train.jsonl"),
            [{"messages": [{"role": "user", "content": "Unrelated training prompt."}]}],
        )
        missing_dir = os.path.join(self.tmp.name, "missing_second_corpus")
        proc = _run_gate(
            [
                "--battery-scan",
                "--corpus",
                valid_dir,
                "--corpus",
                missing_dir,
                "--out",
                self.out_path,
            ]
        )
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
        self.assertFalse(os.path.exists(self.out_path))

    def test_valid_non_empty_disjoint_corpus_passes_clean(self):
        corp_dir = os.path.join(self.tmp.name, "clean_corpus")
        _write_jsonl(
            os.path.join(corp_dir, "train.jsonl"),
            [
                {"messages": [{"role": "user", "content": "Completely disjoint training question A."}]},
                {"messages": [{"role": "user", "content": "Completely disjoint training question B."}]},
            ],
        )
        proc = _run_gate(["--battery-scan", "--corpus", corp_dir, "--out", self.out_path])
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("VERDICT: CLEAN", proc.stdout)
        self.assertTrue(os.path.isfile(self.out_path))
        with open(self.out_path, encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["verdict"], "CLEAN")
        self.assertEqual(report["contaminated"], {})
        self.assertGreater(report["scanned_prompts"], 0)
        self.assertEqual(report["scanned_corpus_hashes"], 2)

    def test_valid_corpus_with_overlap_fails_contaminated(self):
        with open(os.path.join(REPO_ROOT, "batteries", "trademath.json"), encoding="utf-8") as f:
            items = json.load(f)
        leaked_prompt = items[0]["prompt"]
        # Add extra whitespace to also verify normalized-whitespace matching.
        leaked_variant = "  " + leaked_prompt.replace(" ", "   ") + "\n"

        corp_dir = os.path.join(self.tmp.name, "contaminated_corpus")
        _write_jsonl(
            os.path.join(corp_dir, "train.jsonl"),
            [
                {"messages": [{"role": "user", "content": "Clean training question."}]},
                {"messages": [{"role": "user", "content": leaked_variant}]},
            ],
        )
        proc = _run_gate(["--battery-scan", "--corpus", corp_dir, "--out", self.out_path])
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("VERDICT: CONTAMINATED", proc.stdout)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertTrue(os.path.isfile(self.out_path))
        with open(self.out_path, encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["verdict"], "CONTAMINATED")
        self.assertIn("batteries/trademath.json", report["contaminated"])
        self.assertIn(items[0]["id"], report["contaminated"]["batteries/trademath.json"])

    def test_missing_or_empty_batteries_fail_closed(self):
        corp_dir = os.path.join(self.tmp.name, "valid_corpus")
        _write_jsonl(
            os.path.join(corp_dir, "train.jsonl"),
            [{"messages": [{"role": "user", "content": "Some training prompt."}]}],
        )
        empty_root = os.path.join(self.tmp.name, "empty_gg_root")
        os.makedirs(os.path.join(empty_root, "batteries"))

        # Case 1: no battery files at all
        proc = _run_gate(
            ["--battery-scan", "--corpus", corp_dir, "--out", self.out_path],
            env_extra={"GG_ROOT": empty_root},
        )
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc.stdout)
        self.assertIn("VERDICT: INVALID_BATTERY", proc.stdout)

        # Case 2: battery file exists but has zero items
        with open(os.path.join(empty_root, "batteries", "empty.json"), "w", encoding="utf-8") as f:
            json.dump([], f)
        proc2 = _run_gate(
            ["--battery-scan", "--corpus", corp_dir, "--out", self.out_path],
            env_extra={"GG_ROOT": empty_root},
        )
        self.assertNotEqual(proc2.returncode, 0, msg=proc2.stdout + proc2.stderr)
        self.assertNotIn("VERDICT: CLEAN", proc2.stdout)
        self.assertIn("VERDICT: INVALID_BATTERY", proc2.stdout)


class TestVerifyHoldoutDisjointTrainDirMode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.holdout_keys = os.path.join(self.tmp.name, "holdout_keys.txt")
        with open(self.holdout_keys, "w", encoding="utf-8") as f:
            f.write("AAPL|2026-01-01|CALL\n")
        self.heldout = os.path.join(self.tmp.name, "heldout.jsonl")
        _write_jsonl(
            self.heldout,
            [{"messages": [{"role": "user", "content": "Heldout setup"}, {"role": "assistant", "content": "{}"}]}],
        )

    def test_missing_train_dir_fails_closed(self):
        missing = os.path.join(self.tmp.name, "missing_train")
        proc = _run_gate(
            ["--train-dir", missing, "--holdout-keys", self.holdout_keys, "--heldout", self.heldout]
        )
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
        self.assertNotIn("DISJOINT", proc.stdout)

    def test_empty_train_jsonl_fails_closed(self):
        tdir = os.path.join(self.tmp.name, "empty_train")
        os.makedirs(tdir)
        open(os.path.join(tdir, "train.jsonl"), "w", encoding="utf-8").close()
        proc = _run_gate(
            ["--train-dir", tdir, "--holdout-keys", self.holdout_keys, "--heldout", self.heldout]
        )
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("VERDICT: INVALID_CORPUS", proc.stdout)
        self.assertNotIn("DISJOINT", proc.stdout)

    def test_empty_holdout_inputs_fail_closed(self):
        tdir = os.path.join(self.tmp.name, "valid_train")
        _write_jsonl(
            os.path.join(tdir, "train.jsonl"),
            [{"messages": [{"role": "user", "content": "Train setup"}, {"role": "assistant", "content": "{}"}]}],
        )
        empty_keys = os.path.join(self.tmp.name, "empty_keys.txt")
        open(empty_keys, "w", encoding="utf-8").close()
        proc = _run_gate(
            ["--train-dir", tdir, "--holdout-keys", empty_keys, "--heldout", self.heldout]
        )
        self.assertNotEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("VERDICT: INVALID_INPUT", proc.stdout)
        self.assertNotIn("DISJOINT", proc.stdout)

        empty_heldout = os.path.join(self.tmp.name, "empty_heldout.jsonl")
        open(empty_heldout, "w", encoding="utf-8").close()
        proc2 = _run_gate(
            ["--train-dir", tdir, "--holdout-keys", self.holdout_keys, "--heldout", empty_heldout]
        )
        self.assertNotEqual(proc2.returncode, 0, msg=proc2.stdout + proc2.stderr)
        self.assertIn("VERDICT: INVALID_INPUT", proc2.stdout)
        self.assertNotIn("DISJOINT", proc2.stdout)


    def test_valid_nonempty_disjoint_holdout_passes_clean_without_eval_v4(self):
        """Clean-install replay: valid nonempty files must pass with zero eval_v4 dependency."""
        tdir = os.path.join(self.tmp.name, "valid_train_clean")
        _write_jsonl(
            os.path.join(tdir, "train.jsonl"),
            [
                {
                    "meta": {
                        "ticker": "NVDA",
                        "year": 2025,
                        "structure_type": "long_call",
                        "outcome_pnl": 1250.5,
                    },
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {
                            "role": "user",
                            "content": "[2025 -- AI | Roth] Setup: NVDA CALL $140, exp 2025-06-20 (~45 DTE at entry). You bought-to-open ~2 at $5.50, closed at $11.75 after 12 days.",
                        },
                        {"role": "assistant", "content": "{}"},
                    ],
                }
            ],
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                SCRIPT,
                "--train-dir",
                tdir,
                "--holdout-keys",
                self.holdout_keys,
                "--heldout",
                self.heldout,
            ],
            cwd=self.tmp.name,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertNotIn("ModuleNotFoundError", proc.stderr)
        self.assertIn("VERDICT: DISJOINT", proc.stdout)

    def test_valid_nonempty_trade_key_contamination_fails_closed(self):
        """Real-trade row whose canonical trade_key is in holdout_keys must exit 1 CONTAMINATED."""
        tdir = os.path.join(self.tmp.name, "contaminated_train")
        row = {
            "meta": {
                "ticker": "NVDA",
                "year": 2025,
                "structure_type": "long_call",
                "outcome_pnl": 1250.5,
            },
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": "[2025 -- AI | Roth] Setup: NVDA CALL $140, exp 2025-06-20 (~45 DTE at entry). You bought-to-open ~2 at $5.50, closed at $11.75 after 12 days.",
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
        _write_jsonl(os.path.join(tdir, "train.jsonl"), [row])
        hold_keys = os.path.join(self.tmp.name, "matching_holdout_keys.txt")
        with open(hold_keys, "w", encoding="utf-8") as f:
            f.write("cb0cd8dfd10014b3\n")
        proc = _run_gate(
            ["--train-dir", tdir, "--holdout-keys", hold_keys, "--heldout", self.heldout]
        )
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("CONTAMINATED — held-back trades present in training", proc.stdout)
        self.assertIn("VERDICT: CONTAMINATED", proc.stdout)

    def test_valid_nonempty_user_overlap_contamination_fails_closed(self):
        """Exact held-out user-message overlap in train.jsonl must exit 1 CONTAMINATED."""
        tdir = os.path.join(self.tmp.name, "user_overlap_train")
        _write_jsonl(
            os.path.join(tdir, "train.jsonl"),
            [
                {
                    "messages": [
                        {"role": "user", "content": "Heldout setup"},
                        {"role": "assistant", "content": "{}"},
                    ]
                }
            ],
        )
        proc = _run_gate(
            ["--train-dir", tdir, "--holdout-keys", self.holdout_keys, "--heldout", self.heldout]
        )
        self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
        self.assertIn("exact user-message overlaps:            1", proc.stdout)
        self.assertIn("VERDICT: CONTAMINATED", proc.stdout)

    def test_trade_key_matches_canonical_eval_v4_vectors(self):
        """Verify self-contained trade_key matches canonical eval_v4.trade_key hashes."""
        sys.path.insert(0, REPO_ROOT)
        try:
            import verify_holdout_disjoint as vhd
        finally:
            if REPO_ROOT in sys.path:
                sys.path.remove(REPO_ROOT)

        row1 = {
            "meta": {
                "ticker": "NVDA",
                "year": 2025,
                "structure_type": "long_call",
                "outcome_pnl": 1250.5,
            },
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": "[2025 -- AI | Roth] Setup: NVDA CALL $140, exp 2025-06-20 (~45 DTE at entry). You bought-to-open ~2 at $5.50, closed at $11.75 after 12 days.",
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
        row2 = {
            "meta": {
                "ticker": "AAPL",
                "year": 2025,
                "structure_type": "call_debit_spread",
                "outcome_pnl": -300,
            },
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": "Generic non-matching setup prompt text for fallback branch test",
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
        self.assertEqual(vhd.trade_key(row1), "cb0cd8dfd10014b3")
        self.assertEqual(vhd.trade_key(row2), "2ef4a28fd1dc0481")

if __name__ == "__main__":
    unittest.main()
