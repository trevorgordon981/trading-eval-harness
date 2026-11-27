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
import re
import sys
import time
import unicodedata

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


NEAR_DUP_JACCARD = 0.85
NEAR_DUP_MIN_TOKENS = 6


def _h(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


def _norm(s):
    """Canonical whitespace + Unicode NFKC normalization."""
    s = unicodedata.normalize("NFKC", s or "")
    return " ".join(s.split()).strip()


def _canon(s):
    """Canonical case-folded, whitespace-collapsed normalization."""
    return _norm(s).casefold()


def _phash(s):
    return hashlib.sha1(_norm(s).encode("utf-8", "replace")).hexdigest()


def _chash(s):
    return hashlib.sha1(_canon(s).encode("utf-8", "replace")).hexdigest()


def _token_shingles(s, k=3):
    """Normalized alphanumeric token k-shingles for near-duplicate detection."""
    toks = re.findall(r"[a-z0-9]+", _canon(s))
    if len(toks) < NEAR_DUP_MIN_TOKENS:
        return frozenset()
    if len(toks) <= k:
        return frozenset([tuple(toks)])
    return frozenset(tuple(toks[i:i + k]) for i in range(len(toks) - k + 1))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _scenario_id(row, fallback_idx):
    """Return a stable scenario identifier for a held-out JSONL row."""
    if isinstance(row, dict) and row.get("id") is not None:
        return str(row["id"])
    u = _msg(row, "user")
    return "row_%04d_%s" % (fallback_idx, _phash(u)[:10])


# Canonical real-trade setup pattern & key function (identical to eval_v4.trade_key,
# self-contained here so verify_holdout_disjoint.py works in clean installs with zero
# external/unpublished module dependencies).
_RA = re.compile(
    r"Setup:\s*(?P<ticker>[A-Z][A-Z0-9.]*)\s+(?P<otype>CALL|PUT)\s+\$?(?P<strike>[0-9.]+)"
    r".*?~?(?P<dte>[0-9]+)\s*DTE"
    r".*?~?(?P<contracts>[0-9]+)\s*at\s*\$(?P<entry>[0-9.]+)"
    r".*?closed at\s*\$(?P<exit>[0-9.]+)\s*after\s*(?P<hold>[0-9]+)\s*day",
    re.S,
)


def trade_key(row):
    """Stable key for a real trade — deterministic across runs & orderings."""
    m = row.get("meta") or {}
    u = row["messages"][1]["content"]
    setup = _RA.search(u)
    tail = setup.group(0)[:80] if setup else u[:80]
    raw = "|".join(str(x) for x in (
        m.get("ticker"),
        m.get("year"),
        m.get("structure_type"),
        m.get("outcome_pnl"),
        tail,
    ))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


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
    """Return (prompts_list, unscannable_count) for every item/scenario in a battery file."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    items = d
    if isinstance(d, dict):
        items = d.get("scenarios") or d.get("items") or []
    if not isinstance(items, list):
        return [], 1
    prompts = []
    unscannable = 0
    for it in items:
        if not isinstance(it, dict):
            unscannable += 1
            continue
        pid = it.get("id")
        prompt = it.get("prompt")
        if not prompt:
            turns = it.get("turns") or []
            prompt = " ".join(x for x in turns if isinstance(x, str))
        if pid is not None and prompt and _norm(prompt):
            prompts.append((str(pid), prompt))
        else:
            unscannable += 1
    return prompts, unscannable


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


def _corpus_user_index(spec):
    """spec = a dir (uses train.jsonl+valid.jsonl), file, or glob.

    Returns dict with normalized exact hashes, canonical casefold hashes,
    token-shingle sets (for near-duplicate detection), files, and unscannable_lines.
    Fails closed by raising FileNotFoundError or ValueError if the corpus path is
    missing, matches zero files, or yields zero non-empty user-message entries.
    """
    files = _resolve_corpus_files(spec)
    hashes = set()
    canon_hashes = set()
    shingles = []
    seen_shingles = set()
    records = 0
    unscannable = 0
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    unscannable += 1
                    continue
                if not isinstance(r, dict):
                    unscannable += 1
                    continue
                records += 1
                found_user = False
                for m in r.get("messages", []):
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = _norm(m.get("content") or "")
                        if content:
                            found_user = True
                            hashes.add(_phash(content))
                            ch = _chash(content)
                            canon_hashes.add(ch)
                            if ch not in seen_shingles:
                                seen_shingles.add(ch)
                                sh = _token_shingles(content)
                                if sh:
                                    shingles.append(sh)
                if not found_user:
                    unscannable += 1
    if records == 0 or not hashes:
        raise ValueError(
            "corpus yielded 0 user-message entries (%d JSON record(s) across %d file[s]): %s"
            % (records, len(files), spec)
        )
    return {
        "hashes": hashes,
        "canon_hashes": canon_hashes,
        "shingles": shingles,
        "files": files,
        "unscannable": unscannable,
    }


def _corpus_user_hashes(spec):
    """Backward-compatible wrapper returning (hashes, files)."""
    idx = _corpus_user_index(spec)
    return idx["hashes"], idx["files"]


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

    # build per-corpus user-message indexes (fail closed on any missing/empty corpus)
    corp_idx = {}
    corp_files = {}
    corpus_errors = []
    unscannable_corpus_rows = 0
    for c in corpora:
        name = _corpus_label(c, corp_idx)
        try:
            idx = _corpus_user_index(c)
        except (FileNotFoundError, ValueError, OSError) as exc:
            corpus_errors.append(str(exc))
            print("ERROR: %s" % exc, file=sys.stderr)
            continue
        corp_idx[name] = idx
        corp_files[name] = idx["files"]
        unscannable_corpus_rows += idx["unscannable"]
        print("corpus %-24s %6d unique user-msg hashes  (%d file[s], %d unscannable)" % (
            name, len(idx["hashes"]), len(idx["files"]), idx["unscannable"]))

    total_corpus_files = sum(len(fs) for fs in corp_files.values())
    any_hashes = set().union(*(idx["hashes"] for idx in corp_idx.values())) if corp_idx else set()
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
    near_duplicates = {}     # battery_rel -> [ids]
    counts = {}              # battery_rel -> {corpus: n, "exact": n, "near_duplicate": n, "any": n}
    contam_phashes = set()   # normalized-prompt sha1 of every contaminated prompt
    battery_errors = []
    total_prompts = 0
    total_exact = 0
    total_near_dup = 0
    unscannable_battery_items = 0
    for bf in battery_files:
        rel = os.path.relpath(bf, GG_ROOT)
        ids_hit = []
        near_ids_hit = []
        per_corp = {name: 0 for name in corp_idx}
        exact_n = 0
        near_n = 0
        try:
            prompts, bad_items = _battery_prompts(bf)
        except Exception as exc:
            msg = "failed to parse battery file %s: %s" % (bf, exc)
            battery_errors.append(msg)
            print("ERROR: %s" % msg, file=sys.stderr)
            continue
        unscannable_battery_items += bad_items
        for pid, prompt in prompts:
            total_prompts += 1
            ph = _phash(prompt)
            ch = _chash(prompt)
            sh = _token_shingles(prompt)
            hit_exact = False
            hit_near = False
            for name, idx in corp_idx.items():
                if ph in idx["hashes"] or ch in idx["canon_hashes"]:
                    per_corp[name] += 1
                    hit_exact = True
                elif sh and any(_jaccard(sh, csh) >= NEAR_DUP_JACCARD for csh in idx["shingles"]):
                    per_corp[name] += 1
                    hit_near = True
            if hit_exact:
                exact_n += 1
                ids_hit.append(pid)
                contam_phashes.add(ph)
            elif hit_near:
                near_n += 1
                ids_hit.append(pid)
                near_ids_hit.append(pid)
                contam_phashes.add(ph)
        if ids_hit:
            contaminated[rel] = ids_hit
            if near_ids_hit:
                near_duplicates[rel] = near_ids_hit
            counts[rel] = dict(per_corp, **{"exact": exact_n, "near_duplicate": near_n, "any": len(ids_hit)})
            total_exact += exact_n
            total_near_dup += near_n
            print("  CONTAMINATED %-46s %3d prompt(s)  %s" % (rel, len(ids_hit), counts[rel]))

    total_contam = total_exact + total_near_dup
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
        "match_rule": (
            "canonical whitespace+NFKC+casefold exact match + "
            "3-token shingle Jaccard>=%.2f near-duplicate check" % NEAR_DUP_JACCARD
        ),
        "scanned_corpus_files": total_corpus_files,
        "scanned_corpus_hashes": len(any_hashes),
        "scanned_prompts": total_prompts,
        "exact_matches": total_exact,
        "near_duplicate_matches": total_near_dup,
        "unscannable_corpus_rows": unscannable_corpus_rows,
        "unscannable_battery_items": unscannable_battery_items,
        "corpora": {name: corp_files[name] for name in corp_files},
        "counts": counts,
        "contaminated": contaminated,
        "near_duplicates": near_duplicates,
        "contaminated_prompt_sha1": sorted(contam_phashes),
    }
    outpath = args.out or os.path.join(GG_ROOT, "results", "contaminated_ids.json")
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(
        "\nscanned %d battery prompts across %d file[s]; exact=%d near_dup=%d "
        "unscannable_corpus=%d unscannable_battery=%d -> %s"
        % (total_prompts, len(battery_files), total_exact, total_near_dup,
           unscannable_corpus_rows, unscannable_battery_items, outpath)
    )
    # reproduce the headline 314-battery numbers explicitly
    for rel, c in sorted(counts.items()):
        if "trading_gauntlet_314" in rel:
            print("  [314 battery] " + "  ".join("%s=%d" % (k, v) for k, v in c.items()))
    if total_contam:
        print("VERDICT: CONTAMINATED — %d battery prompt(s) matched training data "
              "(exact=%d, near_dup=%d). Scoring should split clean vs contaminated (%s)."
              % (total_contam, total_exact, total_near_dup, outpath))
        sys.exit(1)
    print("VERDICT: CLEAN — no battery prompt matched the scanned corpora.")
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


    bad = 0

    # ---- CHECK 1: TESTHOLDOUT trade-keys absent from training rows ----------
    hits = []
    unscannable_trade_rows = 0
    for r in train_rows:
        try:
            if trade_key(r) in holdkeys:
                hits.append((r.get("meta", {}) or {}).get("ticker"))
        except Exception:
            unscannable_trade_rows += 1  # rows without the real-trade shape can't collide
    print("\nCHECK 1  real-P&L holdout keys: %d keys vs %d training rows -> %d hits (%d non-trade/unscannable)"
          % (len(holdkeys), len(train_rows), len(hits), unscannable_trade_rows))
    if hits:
        bad = 1
        print("  CONTAMINATED — held-back trades present in training: %s"
              % hits[:10])
    else:
        print("  DISJOINT")

    # ---- CHECK 2: held-out rows absent from training rows (canonical + near-dup) ----
    train_user_raw = set()
    train_user_norm = set()
    train_user_canon = set()
    train_pair_raw = set()
    train_pair_norm = set()
    train_shingles = []
    seen_train_canon = set()
    unscannable_train = 0
    for r in train_rows:
        u = _msg(r, "user")
        if not _norm(u):
            unscannable_train += 1
            continue
        a = _msg(r, "assistant")
        train_user_raw.add(_h(u))
        train_user_norm.add(_phash(u))
        ch = _chash(u)
        train_user_canon.add(ch)
        train_pair_raw.add(_h(u + "\x00" + a))
        train_pair_norm.add(_h(_canon(u) + "\x00" + _canon(a)))
        if ch not in seen_train_canon:
            seen_train_canon.add(ch)
            sh = _token_shingles(u)
            if sh:
                train_shingles.append(sh)

    if not train_user_norm:
        print("ERROR: training corpus has 0 non-empty user messages in %s" % tdir, file=sys.stderr)
        print("VERDICT: INVALID_CORPUS — training corpus has no user messages.")
        sys.exit(2)

    held_with_user = [(idx, r) for idx, r in enumerate(held) if _norm(_msg(r, "user"))]
    unscannable_held = len(held) - len(held_with_user)
    if not held_with_user:
        print("ERROR: held-out file has 0 non-empty user messages: %s" % heldout_path,
              file=sys.stderr)
        print("VERDICT: INVALID_INPUT — held-out set has no user messages.")
        sys.exit(2)

    raw_u_hits = 0
    norm_u_hits = 0
    near_u_hits = 0
    p_hits = 0
    hit_scenario_ids = []
    for idx, r in held_with_user:
        u = _msg(r, "user")
        a = _msg(r, "assistant")
        sid = _scenario_id(r, idx)
        is_raw = _h(u) in train_user_raw
        is_norm = _phash(u) in train_user_norm or _chash(u) in train_user_canon
        is_pair = (_h(u + "\x00" + a) in train_pair_raw) or (_h(_canon(u) + "\x00" + _canon(a)) in train_pair_norm)
        is_near = False
        if not is_norm:
            sh = _token_shingles(u)
            if sh and any(_jaccard(sh, tsh) >= NEAR_DUP_JACCARD for tsh in train_shingles):
                is_near = True
        if is_raw:
            raw_u_hits += 1
        if is_norm:
            norm_u_hits += 1
        if is_near:
            near_u_hits += 1
        if is_pair:
            p_hits += 1
        if is_norm or is_near or is_pair:
            hit_scenario_ids.append(sid)

    print("\nCHECK 2  held-out forecasting set: %d rows vs %d training rows"
          % (len(held), len(train_rows)))
    print("  exact user-message overlaps:            %d" % raw_u_hits)
    print("  normalized user-message overlaps:       %d" % norm_u_hits)
    print("  near-duplicate user-message overlaps:   %d" % near_u_hits)
    print("  exact (user+assistant) pair overlaps:   %d" % p_hits)
    print("  unscannable rows (train / held-out):    %d / %d" % (unscannable_train, unscannable_held))
    if norm_u_hits or near_u_hits or p_hits:
        bad = 1
        print("  CONTAMINATED — held-out setups appear in training data (scenario_ids: %s)"
              % hit_scenario_ids[:10])
    else:
        print("  DISJOINT")

    print("\nVERDICT: %s" % ("CONTAMINATED — DO NOT TRUST THE EVAL" if bad
                             else "DISJOINT — eval sets are clean vs %s" % tdir))
    sys.exit(bad)


if __name__ == "__main__":
    main()
