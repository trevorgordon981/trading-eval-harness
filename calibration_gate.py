#!/usr/bin/env python3
"""calibration_gate.py — does the MODEL's stated conviction track correctness?

WHAT IT ANSWERS  (distinct from conviction_calibration.py, which asks whether
high conviction earns real P&L on LIVE exitmgr trades)
  On the held-out eval, when the model says conviction=9 is it actually right
  more often than when it says conviction=3?  This is a reliability / ECE-style
  gate on the model's SELF-REPORTED confidence:

    * bin predictions by stated conviction
    * observed accuracy per bin  vs  the confidence that bin implies
    * ECE (expected calibration error) and MCE (max calibration error)
    * a discrimination gate: high-conviction calls must be MEANINGFULLY more
      accurate than low-conviction ones (the thing Trevor actually wants gated)
    * PASS / FAIL / INSUFFICIENT verdict

  Conviction 1..10 is treated as an implied probability-correct = conviction/10
  for ECE. (If a model emits 0..1 confidences directly, pass --conf-scale 1.)

INPUT FORMAT  (consumed from disk — no model is loaded here)
  Two accepted shapes; both may be .json (list) or .jsonl (one obj/line):

  A) eval_finetuned run_heldout results JSON — has rows[].pred / rows[].gt.
     We read pred.conviction and score correctness on --field (default "call",
     matching eval_finetuned's own conviction calibration; "move" also supported)
     via the same norm_dir / norm_move used by the eval. Point --results at it
     and (optionally) --label to pick BASE vs FT.

  B) a flat predictions file: a list/JSONL of records each with
         {"conviction": <1-10 or 0-1>, "correct": <bool>}
     OR {"conviction":.., "pred":"BULLISH", "gt":"BULLISH"}  (correctness derived)

USAGE
  # score a real eval results file once the box is free:
  python3 calibration_gate.py --results ~/gordon-gauntlet/results/<run>.json \
        --label FT --field call

  # score a flat predictions file:
  python3 calibration_gate.py --preds mypreds.jsonl

  # prove the math with synthetic fixtures (no data/model needed):
  python3 calibration_gate.py --selftest

  To produce a results file with conviction, run the normal held-out eval — its
  run_heldout already stores pred.conviction in rows[]:
      python3 eval_finetuned.py --mode held-out \
          --base-url http://127.0.0.1:8082/v1/chat/completions --n 584 \
          --out results/heldout_$(date +%Y%m%d).json

100% stdlib. Read-only. Loads NO model.
"""
import argparse
import json
import os
import sys

# reuse eval scorers when available (keeps correctness identical to the eval)
GG_ROOT = os.path.dirname(os.path.abspath(__file__))
if GG_ROOT not in sys.path:
    sys.path.insert(0, GG_ROOT)
try:
    import eval_finetuned as ef
    norm_move, norm_dir, to_float = ef.norm_move, ef.norm_dir, ef.to_float
except Exception:  # standalone fallback
    def norm_move(v):
        return None if v is None else str(v).strip().upper()

    def norm_dir(v):
        if v is None:
            return None
        s = str(v).strip().upper()
        if s in ("BULLISH", "BUY", "LONG", "UP"):
            return "UP"
        if s in ("BEARISH", "SELL", "SHORT", "DOWN"):
            return "DOWN"
        if s in ("NEUTRAL", "FLAT", "HOLD"):
            return "NEUTRAL"
        return s

    def to_float(v):
        try:
            return float(v)
        except Exception:
            return None

# ---- gate thresholds (conservative — this informs REAL-money conviction sizing)
MIN_TOTAL = 30            # min scored predictions to render a verdict
MIN_PER_GROUP = 8         # min in BOTH the high and low conviction groups
MIN_LIFT = 0.10           # high-group acc must exceed low-group acc by >= this
MAX_ECE = 0.15            # calibration error above this is flagged (soft gate)
LOW_HI, HI_LO = 4, 8      # low group = conviction<=4 ; high group = conviction>=8


def _correct_from_record(rec, field):
    """Return (conviction_float_or_None, correct_bool_or_None)."""
    conv = rec.get("conviction")
    if conv is None and isinstance(rec.get("pred"), dict):
        conv = rec["pred"].get("conviction")
    conv = to_float(conv)

    if "correct" in rec:
        return conv, bool(rec["correct"])

    # derive correctness from pred/gt
    pred, gt = rec.get("pred"), rec.get("gt")
    if isinstance(pred, dict) and isinstance(gt, dict):
        nf = norm_move if field == "move" else norm_dir
        pv, gv = nf(pred.get(field)), nf(gt.get(field))
        if gv is None:
            return conv, None
        return conv, (pv == gv)
    if pred is not None and gt is not None:  # flat pred/gt strings
        nf = norm_move if field == "move" else norm_dir
        gv = nf(gt)
        if gv is None:
            return conv, None
        return conv, (nf(pred) == gv)
    return conv, None


def load_pairs(path, field="call", label=None):
    """Return list of (conviction, correct_bool) from either input shape."""
    with open(path) as f:
        head = f.read().lstrip()
    if head.startswith("["):
        data = json.loads(head)
        records = data
    elif head.startswith("{"):
        data = json.loads(head)
        # results-JSON shape: dig for objects that carry rows[]
        recs = []

        def walk(d, lbl):
            if isinstance(d, dict):
                if "rows" in d and isinstance(d["rows"], list):
                    if label is None or lbl == label:
                        recs.extend(d["rows"])
                else:
                    for k, v in d.items():
                        walk(v, k)
        walk(data, None)
        records = recs if recs else ([data] if "conviction" in data else [])
    else:  # jsonl
        records = [json.loads(l) for l in head.splitlines() if l.strip()]

    pairs = []
    for rec in records:
        conv, correct = _correct_from_record(rec, field)
        if conv is not None and correct is not None:
            pairs.append((conv, correct))
    return pairs


# ------------------------------------------------------------- reliability math
def reliability(pairs, conf_scale=10.0):
    """Compute per-value bins, grouped bins, ECE, MCE.

    conf implied by conviction = conviction / conf_scale (default 10 -> 0..1).
    """
    n = len(pairs)
    # per stated-conviction-value reliability
    from collections import defaultdict
    byval = defaultdict(list)
    for c, ok in pairs:
        byval[c].append(ok)
    per_value = {}
    ece = mce = 0.0
    for c in sorted(byval):
        oks = byval[c]
        acc = sum(oks) / len(oks)
        conf = c / conf_scale
        gap = abs(acc - conf)
        per_value[c] = {"n": len(oks), "acc": acc, "conf": conf, "gap": gap}
        ece += (len(oks) / n) * gap
        mce = max(mce, gap)

    def _grp(pred):
        oks = [ok for c, ok in pairs if pred(c)]
        return {"n": len(oks), "acc": (sum(oks) / len(oks)) if oks else None}
    groups = {
        "low(<=%d)" % LOW_HI: _grp(lambda c: c <= LOW_HI),
        "mid": _grp(lambda c: LOW_HI < c < HI_LO),
        "high(>=%d)" % HI_LO: _grp(lambda c: c >= HI_LO),
    }
    return {"n": n, "per_value": per_value, "groups": groups,
            "ece": ece, "mce": mce}


def gate(rel):
    """PASS / FAIL / INSUFFICIENT with reasons."""
    n = rel["n"]
    lo = rel["groups"]["low(<=%d)" % LOW_HI]
    hi = rel["groups"]["high(>=%d)" % HI_LO]
    reasons = []
    if n < MIN_TOTAL:
        return "INSUFFICIENT", ["only %d scored predictions (need >=%d)" % (n, MIN_TOTAL)]
    if lo["n"] < MIN_PER_GROUP or hi["n"] < MIN_PER_GROUP:
        return "INSUFFICIENT", [
            "need >=%d in BOTH groups; have low=%d high=%d" % (
                MIN_PER_GROUP, lo["n"], hi["n"])]
    lift = hi["acc"] - lo["acc"]
    reasons.append("discrimination lift high-low = %+.1fpp (need >=%+.1fpp)"
                   % (100 * lift, 100 * MIN_LIFT))
    reasons.append("ECE = %.3f (flag if > %.2f)" % (rel["ece"], MAX_ECE))
    verdict = "PASS" if lift >= MIN_LIFT else "FAIL"
    if rel["ece"] > MAX_ECE:
        reasons.append("NOTE: ECE exceeds %.2f — confidences are mis-scaled even "
                       "if they discriminate; treat conviction as ranking not "
                       "probability." % MAX_ECE)
    return verdict, reasons


def report(pairs, conf_scale=10.0, title="CONVICTION CALIBRATION GATE"):
    rel = reliability(pairs, conf_scale)
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("scored predictions: %d" % rel["n"])
    print("\nper-conviction reliability  (conf = conviction/%g)" % conf_scale)
    print("  %-6s %5s %8s %8s %7s" % ("conv", "n", "acc", "conf", "|gap|"))
    print("  " + "-" * 40)
    for c, d in rel["per_value"].items():
        print("  %-6g %5d %7.1f%% %7.1f%% %6.3f" % (
            c, d["n"], 100 * d["acc"], 100 * d["conf"], d["gap"]))
    print("\ngrouped")
    for name, g in rel["groups"].items():
        acc = ("%.1f%%" % (100 * g["acc"])) if g["acc"] is not None else "n/a"
        print("  %-12s n=%-4d acc=%s" % (name, g["n"], acc))
    print("\nECE = %.3f   MCE = %.3f" % (rel["ece"], rel["mce"]))
    verdict, reasons = gate(rel)
    print("\nGATE: %s" % verdict)
    for r in reasons:
        print("  - %s" % r)
    print("=" * 70)
    return {"reliability": rel, "verdict": verdict, "reasons": reasons}


# ----------------------------------------------------------------------- selftest
def _synth(mapping, per=20, seed=0):
    """mapping: {conviction: P(correct)} -> deterministic pairs (no rng needed)."""
    pairs = []
    for conv, p in mapping.items():
        k = round(p * per)
        pairs += [(conv, True)] * k + [(conv, False)] * (per - k)
    return pairs


def selftest():
    print("### Fixture 1: WELL-CALIBRATED (acc == conviction/10) -> low ECE, PASS\n")
    well = _synth({2: 0.20, 4: 0.40, 6: 0.60, 8: 0.80, 10: 1.00}, per=20)
    r1 = report(well, title="SELFTEST 1 — well calibrated")
    assert r1["reliability"]["ece"] < 0.02, r1["reliability"]["ece"]
    assert r1["verdict"] == "PASS", r1

    print("\n### Fixture 2: ANTI-CALIBRATED (high conviction LESS accurate) -> FAIL\n")
    anti = _synth({2: 0.90, 4: 0.80, 8: 0.30, 10: 0.20}, per=20)
    r2 = report(anti, title="SELFTEST 2 — anti-calibrated (should FAIL)")
    assert r2["verdict"] == "FAIL", r2

    print("\n### Fixture 3: TOO FEW SAMPLES -> INSUFFICIENT\n")
    few = _synth({2: 0.5, 10: 0.5}, per=3)
    r3 = report(few, title="SELFTEST 3 — insufficient")
    assert r3["verdict"] == "INSUFFICIENT", r3

    print("\n### Fixture 4: DISCRIMINATES but MIS-SCALED (good lift, high ECE) -> PASS+flag\n")
    # accuracy rises with conviction (lift ok) but confidences way over/under implied prob
    disc = _synth({2: 0.55, 4: 0.60, 8: 0.75, 10: 0.85}, per=25)
    r4 = report(disc, title="SELFTEST 4 — discriminates but mis-scaled")
    assert r4["verdict"] == "PASS", r4
    assert r4["reliability"]["ece"] > MAX_ECE, r4["reliability"]["ece"]

    print("\nALL SELFTEST ASSERTIONS PASSED.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="eval_finetuned results JSON (rows[].pred/.gt)")
    ap.add_argument("--preds", help="flat predictions file (list/JSONL of records)")
    ap.add_argument("--label", default=None, help="pick BASE/FT when reading --results")
    ap.add_argument("--field", default="call", choices=["call", "move"],
                    help="which prediction to score correctness on")
    ap.add_argument("--conf-scale", type=float, default=10.0,
                    help="divide conviction by this to get implied prob (default 10)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    path = args.results or args.preds
    if not path:
        ap.error("pass --results, --preds, or --selftest")
    if not os.path.exists(path):
        print("ERROR: file not found: %s" % path, file=sys.stderr)
        sys.exit(2)
    pairs = load_pairs(path, field=args.field, label=args.label)
    if not pairs:
        print("ERROR: no (conviction, correct) pairs extracted from %s" % path,
              file=sys.stderr)
        sys.exit(3)
    out = report(pairs, conf_scale=args.conf_scale)
    if args.json:
        print("\n" + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
