#!/usr/bin/env python3
"""verify_holdout_disjoint.py — contamination gate: prove the eval sets are
DISJOINT from a training-data dir before trusting any v4 eval number.

Guards against the known trap: the 6/23 M3-retune held-out eval turned out
~60% train-contaminated. Run this against ANY {train,valid}.jsonl dir
(glm-v4-data-ff, gemma-v4-data, ...) BEFORE running eval_v4.py, and rerun it
if the data dir is ever regenerated.

CHECK 1 — real-P&L holdout keys (~/v4-corpus/real_trades_TESTHOLDOUT.txt):
    trade_key() (same function eval_v4.py + the data builders use) of every
    training row must NOT be in the holdout key set. Any hit = the trainer
    saw a held-back real trade -> Track R is memorization.

CHECK 2 — 584-row held-out forecasting set (~/m3-mix/heldout_trading_clean.jsonl):
    exact user-message and (user+assistant) content hashes of every held-out
    row must NOT appear among training rows. Any hit = forecasting anchor /
    calibration gate / theta-proxy setups are train-contaminated.

Exit 0 = DISJOINT / CLEAN (all checks clean). Exit 1 = CONTAMINATED.
Exit 2 = INVALID_CORPUS / INVALID_BATTERY / INVALID_INPUT (fail-closed on
missing or empty inputs). 100% stdlib, read-only, loads no model.

USAGE
  python3 verify_holdout_disjoint.py --train-dir ~/glm-v4-data-ff
  python3 verify_holdout_disjoint.py --train-dir ~/gemma-v4-data
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GG_ROOT = os.path.expanduser(os.environ.get("GG_ROOT") or SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

HOME = os.path.expanduser("~")
DEF_HOLDKEYS = os.path.join(HOME, "v4-corpus", "real_trades_TESTHOLDOUT.txt")
DEF_HELDOUT = os.path.join(HOME, "m3-mix", "heldout_trading_clean.jsonl")

# default training corpora scanned by the --battery-scan decontamination gate (fix 10)
DEF_CORPORA = [
    os.path.join(HOME, "dataset-v1-curated"),   # dir -> train.jsonl + valid.jsonl
    os.path.join(HOME, "distill-v5", "*.jsonl"),  # glob
]


def _h(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


def _norm(s):
    return " ".join((s or "").split()).strip()


def _phash(s):
    return hashlib.sha1(_norm(s).encode("utf-8", "replace")).hexdigest()


# ------------------------------------------------------------------------------
# fix 10: DECONTAMINATION GATE — hash every prompt of every battery file under
# gordon-gauntlet and fail on an exact (normalized-whitespace) match against any
# user message in the training corpus. Writes results/contaminated_ids.json so
# scoring can split clean-vs-contaminated subsets.
# ------------------------------------------------------------------------------
def _iter_battery_files():
    seen = set()
    pats = [
        os.path.join(GG_ROOT, "batteries", "*.json"),
        os.path.join(GG_ROOT, "trading", "trading_gauntlet*.json"),
        os.path.join(GG_ROOT, "trading", "historical_outcomes.json"),
        os.path.join(GG_ROOT, "trading", "major_events.json"),
        os.path.join(GG_ROOT, "trading_v2", "p*.json"),
    ]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            if ".bak-" in f or f in seen:
                continue
            seen.add(f)
            yield f


def _battery_prompts(path):
    """Yield (id, prompt_text) for every item/scenario in a battery file."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    items = d
    if isinstance(d, dict):
        items = d.get("scenarios") or d.get("items") or []
    if not isinstance(items, list):
        return
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("id")
        prompt = it.get("prompt")
        if not prompt:
            turns = it.get("turns") or []
            prompt = " ".join(x for x in turns if isinstance(x, str))
        if pid is not None and prompt and _norm(prompt):
            yield str(pid), prompt


def _resolve_corpus_files(spec):
    """Resolve a corpus dir, file path, or glob to existing regular files.

    Raises FileNotFoundError if the path/dir/glob does not exist or matches no files.
    """
    if os.path.isdir(spec):
        files = [
            os.path.join(spec, n)
            for n in ("train.jsonl", "valid.jsonl")
            if os.path.isfile(os.path.join(spec, n))
        ]
        if not files:
            raise FileNotFoundError(
                "corpus directory contains neither train.jsonl nor valid.jsonl: %s" % spec
            )
        return files
    if glob.has_magic(spec):
        files = sorted(p for p in glob.glob(spec) if os.path.isfile(p))
        if not files:
            raise FileNotFoundError("corpus glob matched 0 files: %s" % spec)
        return files
    if not os.path.exists(spec):
        raise FileNotFoundError("corpus path does not exist: %s" % spec)
    if not os.path.isfile(spec):
        raise FileNotFoundError("corpus path is not a readable file or directory: %s" % spec)
    return [spec]


def _corpus_user_hashes(spec):
    """spec = a dir (uses train.jsonl+valid.jsonl), file, or glob.

    Returns (hashes, files). Fails closed by raising FileNotFoundError or
    ValueError if the corpus path is missing, matches zero files, or yields
    zero non-empty user-message entries.
    """
    files = _resolve_corpus_files(spec)
    hashes = set()
    records = 0
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not isinstance(r, dict):
                    continue
                records += 1
                for m in r.get("messages", []):
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = _norm(m.get("content") or "")
                        if content:
                            hashes.add(_phash(content))
    if records == 0 or not hashes:
        raise ValueError(
            "corpus yielded 0 user-message entries (%d JSON record(s) across %d file[s]): %s"
            % (records, len(files), spec)
        )
    return hashes, files


def _corpus_label(spec, existing):
    if os.path.isdir(spec):
        base = os.path.basename(os.path.normpath(spec))
    elif glob.has_magic(spec):
        base = os.path.basename(os.path.dirname(spec)) or spec
    else:
        base = os.path.basename(os.path.dirname(spec)) or os.path.basename(spec) or spec
    if base not in existing:
        return base
    alt = os.path.basename(spec)
    if alt and alt not in existing:
        return alt
    idx = 2
    while "%s#%d" % (base, idx) in existing:
        idx += 1
    return "%s#%d" % (base, idx)


def cmd_battery_scan(args):
    corpora = args.corpus or DEF_CORPORA
    corpora = [os.path.expanduser(c) for c in corpora]
    if not corpora:
        print("ERROR: no corpus paths specified", file=sys.stderr)
        print("VERDICT: INVALID_CORPUS — no training corpora configured for scan.")
        sys.exit(2)

    # build per-corpus user-message hash sets (fail closed on any missing/empty corpus)
    corp_hashes = {}
    corp_files = {}
    corpus_errors = []
    for c in corpora:
        name = _corpus_label(c, corp_hashes)
        try:
            h, files = _corpus_user_hashes(c)
        except (FileNotFoundError, ValueError, OSError) as exc:
            corpus_errors.append(str(exc))
            print("ERROR: %s" % exc, file=sys.stderr)
            continue
        corp_hashes[name] = h
        corp_files[name] = files
        print("corpus %-24s %6d unique user-msg hashes  (%d file[s])" % (name, len(h), len(files)))

    total_corpus_files = sum(len(fs) for fs in corp_files.values())
    any_hashes = set().union(*corp_hashes.values()) if corp_hashes else set()
    if corpus_errors or total_corpus_files == 0 or not any_hashes:
        print(
            "VERDICT: INVALID_CORPUS — contamination gate failed closed "
            "(%d corpus error(s), %d file[s], %d unique user-msg hash(es))."
            % (len(corpus_errors), total_corpus_files, len(any_hashes))
        )
        sys.exit(2)

    battery_files = list(_iter_battery_files())
    if not battery_files:
        print("ERROR: no battery files found under %s" % GG_ROOT, file=sys.stderr)
        print("VERDICT: INVALID_BATTERY — contamination gate failed closed (0 battery files found).")
        sys.exit(2)

    contaminated = {}        # battery_rel -> [ids]
    counts = {}              # battery_rel -> {corpus: n, "any": n}
    contam_phashes = set()   # normalized-prompt sha1 of every contaminated prompt
    battery_errors = []
    total_prompts = 0
    total_contam = 0
    for bf in battery_files:
        rel = os.path.relpath(bf, GG_ROOT)
        ids_hit = []
        per_corp = {name: 0 for name in corp_hashes}
        try:
            prompts = list(_battery_prompts(bf))
        except Exception as exc:
            msg = "failed to parse battery file %s: %s" % (bf, exc)
            battery_errors.append(msg)
            print("ERROR: %s" % msg, file=sys.stderr)
            continue
        for pid, prompt in prompts:
            total_prompts += 1
            ph = _phash(prompt)
            hit_any = False
            for name, hs in corp_hashes.items():
                if ph in hs:
                    per_corp[name] += 1
                    hit_any = True
            if hit_any:
                ids_hit.append(pid)
                contam_phashes.add(ph)
        if ids_hit:
            contaminated[rel] = ids_hit
            counts[rel] = dict(per_corp, **{"any": len(ids_hit)})
            total_contam += len(ids_hit)
            print("  CONTAMINATED %-46s %3d prompt(s)  %s" % (rel, len(ids_hit), counts[rel]))

    if battery_errors or total_prompts == 0:
        print(
            "ERROR: battery scan invalid (%d prompt(s) scanned, %d parse error(s))"
            % (total_prompts, len(battery_errors)),
            file=sys.stderr,
        )
        print(
            "VERDICT: INVALID_BATTERY — contamination gate failed closed "
            "(%d prompt(s), %d error(s))." % (total_prompts, len(battery_errors))
        )
        sys.exit(2)

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "CONTAMINATED" if total_contam else "CLEAN",
        "match_rule": "normalized-whitespace exact prompt vs any training user message (sha1)",
        "scanned_corpus_files": total_corpus_files,
        "scanned_corpus_hashes": len(any_hashes),
        "scanned_prompts": total_prompts,
        "corpora": {name: corp_files[name] for name in corp_files},
        "counts": counts,
        "contaminated": contaminated,
        "contaminated_prompt_sha1": sorted(contam_phashes),
    }
    outpath = args.out or os.path.join(GG_ROOT, "results", "contaminated_ids.json")
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nscanned %d battery prompts across %d file[s]; %d contaminated -> %s"
          % (total_prompts, len(battery_files), total_contam, outpath))
    # reproduce the headline 314-battery numbers explicitly
    for rel, c in sorted(counts.items()):
        if "trading_gauntlet_314" in rel:
            print("  [314 battery] " + "  ".join("%s=%d" % (k, v) for k, v in c.items()))
    if total_contam:
        print("VERDICT: CONTAMINATED — %d battery prompt(s) appear verbatim in training data. "
              "Scoring should split clean vs contaminated (report.py reads %s)." % (total_contam, outpath))
        sys.exit(1)
    print("VERDICT: CLEAN — no battery prompt appears verbatim in the scanned corpora.")
    sys.exit(0)


def _msg(row, role):
    return next((m.get("content") or "" for m in row.get("messages", [])
                 if m.get("role") == role), "")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default=None,
                    help="dir containing train.jsonl (+ valid.jsonl) [holdout-key mode]")
    ap.add_argument("--holdout-keys", default=DEF_HOLDKEYS)
    ap.add_argument("--heldout", default=DEF_HELDOUT)
    ap.add_argument("--battery-scan", action="store_true",
                    help="fix 10: hash every battery prompt under gordon-gauntlet and fail on "
                         "exact match in the training corpus; write results/contaminated_ids.json")
    ap.add_argument("--corpus", action="append", default=None,
                    help="training corpus dir or glob to scan (repeatable). "
                         "Default: ~/dataset-v1-curated + ~/distill-v5/*.jsonl")
    ap.add_argument("--out", default=None, help="contaminated_ids.json output path")
    args = ap.parse_args()

    if args.battery_scan:
        cmd_battery_scan(args)
        return

    if not args.train_dir:
        ap.error("--train-dir is required (holdout-key mode), or use --battery-scan")

    tdir = os.path.expanduser(args.train_dir)
    if not os.path.isdir(tdir):
        print("ERROR: training dir does not exist: %s" % tdir, file=sys.stderr)
        print("VERDICT: INVALID_CORPUS — training directory missing (%s)." % tdir)
        sys.exit(2)
    files = [os.path.join(tdir, n) for n in ("train.jsonl", "valid.jsonl")
             if os.path.isfile(os.path.join(tdir, n))]
    if not files:
        print("ERROR: no train.jsonl/valid.jsonl in %s" % tdir, file=sys.stderr)
        print("VERDICT: INVALID_CORPUS — no train.jsonl or valid.jsonl in %s." % tdir)
        sys.exit(2)

    train_rows = []
    for p in files:
        rs = load_jsonl(p)
        print("loaded %-60s %6d rows (mtime %s)" % (
            p, len(rs), time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))))
        train_rows += rs
    if not train_rows:
        print("ERROR: training corpus has 0 rows across %d file[s] in %s" % (len(files), tdir),
              file=sys.stderr)
        print("VERDICT: INVALID_CORPUS — training corpus is empty (%s)." % tdir)
        sys.exit(2)

    holdkeys_path = os.path.expanduser(args.holdout_keys)
    if not os.path.isfile(holdkeys_path):
        print("ERROR: holdout keys file does not exist: %s" % holdkeys_path, file=sys.stderr)
        print("VERDICT: INVALID_INPUT — holdout keys file missing (%s)." % holdkeys_path)
        sys.exit(2)
    with open(holdkeys_path, encoding="utf-8") as f:
        holdkeys = set(l.strip() for l in f if l.strip())
    if not holdkeys:
        print("ERROR: holdout keys file is empty: %s" % holdkeys_path, file=sys.stderr)
        print("VERDICT: INVALID_INPUT — holdout keys file is empty (%s)." % holdkeys_path)
        sys.exit(2)

    heldout_path = os.path.expanduser(args.heldout)
    if not os.path.isfile(heldout_path):
        print("ERROR: held-out file does not exist: %s" % heldout_path, file=sys.stderr)
        print("VERDICT: INVALID_INPUT — held-out set missing (%s)." % heldout_path)
        sys.exit(2)
    held = load_jsonl(heldout_path)
    if not held:
        print("ERROR: held-out file has 0 rows: %s" % heldout_path, file=sys.stderr)
        print("VERDICT: INVALID_INPUT — held-out set is empty (%s)." % heldout_path)
        sys.exit(2)

    if "trade_key" not in globals():
        from eval_v4 import trade_key  # the SAME key fn the builders excluded by (lazy: heavy import)
        globals()["trade_key"] = trade_key

    bad = 0

    # ---- CHECK 1: TESTHOLDOUT trade-keys absent from training rows ----------
    hits = []
    for r in train_rows:
        try:
            if trade_key(r) in holdkeys:
                hits.append((r.get("meta", {}) or {}).get("ticker"))
        except Exception:
            continue  # rows without the real-trade shape can't collide
    print("\nCHECK 1  real-P&L holdout keys: %d keys vs %d training rows -> %d hits"
          % (len(holdkeys), len(train_rows), len(hits)))
    if hits:
        bad = 1
        print("  CONTAMINATED — held-back trades present in training: %s"
              % hits[:10])
    else:
        print("  DISJOINT")

    # ---- CHECK 2: 584 held-out rows absent from training rows ---------------
    train_user = {_h(_msg(r, "user")) for r in train_rows if _msg(r, "user").strip()}
    train_pair = {_h(_msg(r, "user") + "\x00" + _msg(r, "assistant"))
                  for r in train_rows if _msg(r, "user").strip()}
    if not train_user:
        print("ERROR: training corpus has 0 non-empty user messages in %s" % tdir, file=sys.stderr)
        print("VERDICT: INVALID_CORPUS — training corpus has no user messages.")
        sys.exit(2)
    held_with_user = [r for r in held if _msg(r, "user").strip()]
    if not held_with_user:
        print("ERROR: held-out file has 0 non-empty user messages: %s" % heldout_path,
              file=sys.stderr)
        print("VERDICT: INVALID_INPUT — held-out set has no user messages.")
        sys.exit(2)
    u_hits = sum(1 for r in held_with_user if _h(_msg(r, "user")) in train_user)
    p_hits = sum(1 for r in held_with_user if _h(_msg(r, "user") + "\x00" +
                                                 _msg(r, "assistant")) in train_pair)
    print("\nCHECK 2  held-out forecasting set: %d rows vs %d training rows"
          % (len(held), len(train_rows)))
    print("  exact user-message overlaps:            %d" % u_hits)
    print("  exact (user+assistant) pair overlaps:   %d" % p_hits)
    if u_hits or p_hits:
        bad = 1
        print("  CONTAMINATED — held-out setups appear in training data")
    else:
        print("  DISJOINT")

    print("\nVERDICT: %s" % ("CONTAMINATED — DO NOT TRUST THE EVAL" if bad
                             else "DISJOINT — eval sets are clean vs %s" % tdir))
    sys.exit(bad)


if __name__ == "__main__":
    main()
