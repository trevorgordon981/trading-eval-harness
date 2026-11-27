#!/usr/bin/env python3
"""
eval_finetuned.py — Is a fine-tuned trading LLM a BETTER TRADER than its base?

Runs BASE vs FINE-TUNED head-to-head (or BASE-only when --ft-url is omitted,
--base-only is set, or --ft-url == --base-url) across three targets and prints
a single scorecard plus a reliability-gated verdict:

  1. held-out      Held-out trading set (batteries/heldout_example.jsonl by default).
                   Objective: all-case move-category accuracy, direction accuracy,
                   vol_change / vs_iv accuracy, exp_move_pct MAE (with coverage),
                   parse/failure coverage census, conditional parsed-only accuracy
                   (reported separately, never used for ranking), and paired
                   bootstrap 95% CIs on immutable case IDs. Optional token-level
                   loss/perplexity requires --hf-base/--hf-ft.
  2. trading-gauntlet  The trading batteries (manifest profile `trading`:
                   tickers / trademath / trader). Objective checks are scored
                   locally (fail-closed on HTTP errors, timeouts, empty replies,
                   refusals, and wrong signs). llm_judge items are flagged
                   "NEEDS JUDGE" and optionally exported blind via --blind-judge-dir.
  3. gordon        General-ability regression over disjoint non-trading batteries
                   (manifest profile `general` + any extra non-trading batteries in
                   batteries/). Never overlaps with trading-gauntlet; withholds
                   verdicts if valid coverage is insufficient.

Optional protocol integrity hooks:
  --contract-dir DIR     Create and verify hash-bound (SHA-256) gauntlet run
                         contracts and gordon-result.v2 files via gauntlet_contract.py.
  --blind-judge-dir DIR  Export pending llm_judge / ib_grader items through
                         judge_blind.py (identity stripped + shuffled).

Both BASE and FT are queried at OpenAI-compatible /v1/chat/completions endpoints:

  python3 eval_finetuned.py \
      --base-url http://127.0.0.1:8082/v1/chat/completions \
      --ft-url   http://127.0.0.1:8090/v1/chat/completions \
      --mode all --n 300

Stdlib-only for the endpoint path. Perplexity path lazily imports torch/peft/
transformers and degrades gracefully if they're missing.
"""
import argparse
from collections import Counter
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------- paths
HOME = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Held-out trading set: override with --heldout or the HELDOUT_PATH env var.
# Defaults to a synthetic example shipped in batteries/ so the harness runs OOTB.
DEFAULT_HELDOUT = os.environ.get(
    "HELDOUT_PATH",
    os.path.join(SCRIPT_DIR, "batteries", "heldout_example.jsonl"),
)
MIN_RELIABLE_COVERAGE = 0.80


def _find_gg_root():
    for cand in [
        os.environ.get("GG_ROOT", ""),
        SCRIPT_DIR,
        os.path.join(HOME, "trading-eval-harness"),
    ]:
        if cand and os.path.isdir(os.path.join(cand, "batteries")):
            return cand
    return SCRIPT_DIR


GG_ROOT = _find_gg_root()
BATT_DIR = os.path.join(GG_ROOT, "batteries")
DEFAULT_TRADING_BATTERIES = ["tickers", "trademath", "trader"]
FAILURE_REASONS = (
    "http_error",
    "timeout",
    "empty",
    "refusal",
    "truncated",
    "parse_error",
    "schema_error",
)
ALLOWED_JSON_FINISH_REASONS = ("stop", None)
ALLOWED_TEXT_FINISH_REASONS = ("stop", "tool_calls", None)


def load_manifest_profiles(root=None):
    """Load battery profiles from battery_manifest.json if present."""
    base = root or GG_ROOT
    mpath = os.path.join(base, "battery_manifest.json")
    if not os.path.isfile(mpath):
        return {}
    try:
        with open(mpath, encoding="utf-8") as f:
            data = json.load(f)
        profiles = data.get("profiles") if isinstance(data, dict) else None
        if isinstance(profiles, dict):
            return {k: list(v) for k, v in profiles.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}


def get_trading_batteries(root=None):
    profiles = load_manifest_profiles(root)
    if profiles.get("trading"):
        return list(profiles["trading"])
    return list(DEFAULT_TRADING_BATTERIES)


def get_general_batteries(root=None, trading_batteries=None):
    """Return strictly non-trading batteries for general-ability regression."""
    base = root or GG_ROOT
    bdir = os.path.join(base, "batteries")
    trading_set = set(trading_batteries or get_trading_batteries(base))
    profiles = load_manifest_profiles(base)

    # Start with manifest 'general' profile (strictly excluding trading batteries)
    general = [
        b for b in profiles.get("general", [])
        if b not in trading_set and b != "ib_essay"
        and os.path.isfile(os.path.join(bdir, b + ".json"))
    ]
    seen = set(general)

    # Include any extra non-trading battery files dropped in batteries/
    for p in sorted(glob.glob(os.path.join(bdir, "*.json"))):
        bn = os.path.basename(p)
        if bn.endswith(".bak") or ".bak-" in bn:
            continue
        name = bn[:-5]
        if name in trading_set or name == "ib_essay" or name in seen:
            continue
        general.append(name)
        seen.add(name)
    return sorted(general)


TRADING_BATTERIES = get_trading_batteries(GG_ROOT)


# ----------------------------------------------------------------------------- failure census & classification
def new_failure_census():
    return {
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "coverage_rate": 0.0,
        "by_reason": {k: 0 for k in FAILURE_REASONS},
    }


def finalize_failure_census(c):
    att = c["attempted"]
    c["coverage_rate"] = (c["succeeded"] / att) if att else 0.0
    return c


def merge_failure_censuses(censuses):
    out = new_failure_census()
    for c in censuses:
        if not c:
            continue
        out["attempted"] += c.get("attempted", 0)
        out["succeeded"] += c.get("succeeded", 0)
        out["failed"] += c.get("failed", 0)
        for k in FAILURE_REASONS:
            out["by_reason"][k] += (c.get("by_reason") or {}).get(k, 0)
    return finalize_failure_census(out)


def _err_kind(err):
    if not err:
        return None
    low = str(err).lower()
    if "timed out" in low or "timeout" in low:
        return "timeout"
    return "http_error"


_REFUSAL_PATTERNS = [
    re.compile(
        r"\b(?:i\s+am|i'm)\s+(?:sorry|afraid|unable)\b[^.\n]{0,80}\b"
        r"(?:can't|cannot|unable|refuse|assist|help|fulfill|comply|invoke|call|access|provide)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:i\s+)?(?:cannot|can't|won't|will\s+not|refuse\s+to|am\s+unable\s+to)\s+"
        r"(?:assist|help|fulfill|comply|answer|provide|invoke|call|use|execute|run)\b",
        re.I,
    ),
    re.compile(
        r"\bas\s+an\s+ai\b[^.\n]{0,80}\b(?:cannot|can't|unable|don't\s+have|do\s+not\s+have)\b",
        re.I,
    ),
    re.compile(
        r"\bi\s+(?:do\s+not|don't)\s+have\s+(?:access\s+to|the\s+ability\s+to)\s+"
        r"(?:tools?|functions?|external|real[- ]time)\b",
        re.I,
    ),
    re.compile(
        r"\brefus(?:e|ing|al)\s+to\s+(?:invoke|call|execute|answer|comply|assist)\b",
        re.I,
    ),
]


def is_refusal(text):
    """Detect explicit AI/capability refusal replies (not legitimate domain trade declines)."""
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _REFUSAL_PATTERNS)


def classify_failure(text, tool_calls, err, finish_reason, *, parsed=True,
                     schema_valid=True, require_json=False, allow_refusal=False):
    """Return None if response succeeded, else a canonical failure reason string."""
    if err:
        return _err_kind(err)
    if finish_reason in ("error", "content_filter"):
        return "http_error"
    if finish_reason == "length":
        return "truncated"
    allowed_fr = ALLOWED_JSON_FINISH_REASONS if require_json else ALLOWED_TEXT_FINISH_REASONS
    if finish_reason not in allowed_fr:
        return "parse_error" if require_json else "http_error"
    has_text = bool((text or "").strip())
    has_tools = bool(_extract_tool_names(tool_calls))
    if not has_text and not has_tools:
        return "empty"
    if not allow_refusal and has_text and not has_tools and is_refusal(text):
        return "refusal"
    if require_json and not parsed:
        return "parse_error"
    if require_json and not schema_valid:
        return "schema_error"
    return None


# ----------------------------------------------------------------------------- http
def detect_model(endpoint, timeout=10):
    base = endpoint.rsplit("/chat/completions", 1)[0]
    try:
        d = json.loads(urllib.request.urlopen(base + "/models", timeout=timeout).read().decode())
        return d["data"][0]["id"]
    except Exception:
        return None


def chat(endpoint, messages, max_tokens=512, temperature=0.0, thinking=None,
         tools=None, model=None, key=None, timeout=240, retries=4, backoff=3):
    """One OpenAI-compatible /chat/completions call. Retries on 503 (single-gen
    backpressure on a single-gen server), honoring Retry-After. Returns the dict."""
    body = {"model": model or "x", "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    if thinking:
        body["thinking"] = thinking
    if tools:
        body["tools"] = tools
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = "Bearer " + key
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=hdr)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode(), strict=False)
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries:
                wait = backoff
                try:
                    wait = max(backoff, int(e.headers.get("Retry-After", backoff)))
                except Exception:
                    pass
                time.sleep(wait)
                continue
            raise


def _normalize_tool_calls(raw_tool_calls):
    """Normalize OpenAI-style tool_calls into a list of dicts (or empty list)."""
    if not raw_tool_calls:
        return []
    if isinstance(raw_tool_calls, list):
        out = []
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                out.append(tc)
            elif isinstance(tc, str) and tc.strip():
                out.append({"function": {"name": tc.strip()}})
        return out
    return []


def chat_text(endpoint, messages, **kw):
    """Convenience: returns (text, completion_tokens, finish_reason, tool_calls, err)."""
    try:
        d = chat(endpoint, messages, **kw)
        choice = d["choices"][0]
        m = choice["message"]
        tcalls = _normalize_tool_calls(m.get("tool_calls"))
        return (
            m.get("content") or "",
            int((d.get("usage") or {}).get("completion_tokens", 0) or 0),
            choice.get("finish_reason"),
            tcalls,
            None,
        )
    except Exception as e:
        return ("", 0, None, [], str(e)[:200])


# ----------------------------------------------------------------------------- JSON parsing helpers (held-out)
def extract_json(text):
    """Pull the last {...} JSON object out of a model response (handles ```json fences,
    leading prose, trailing think-tags). Returns a dict or None."""
    if not text:
        return None
    # strip any leaked CoT block
    text = re.sub(r"<mm:think>.*?</mm:think>", "", text, flags=re.S)
    text = text.replace("```json", "```")
    # try fenced first
    fences = re.findall(r"```(.*?)```", text, flags=re.S)
    cands = fences + [text]
    for c in cands:
        # find balanced-ish json objects, prefer the last
        objs = re.findall(r"\{[^{}]*\}", c, flags=re.S)
        for o in reversed(objs):
            try:
                parsed = json.loads(o)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        # fallback: greedy
        i, j = c.find("{"), c.rfind("}")
        if 0 <= i < j:
            try:
                parsed = json.loads(c[i:j + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return None


MOVE_CATS = {"QUIET", "NORMAL", "ELEVATED", "EXPLOSIVE"}
DIR_CATS = {"UP", "DOWN", "NEUTRAL"}
VOL_CHANGE_CATS = {"EXPANDING", "STABLE", "CONTRACTING"}
VS_IV_CATS = {"RICH", "FAIR", "CHEAP"}


def norm_move(v):
    if v is None:
        return None
    return str(v).strip().upper()


def norm_dir(v):
    """Map assorted direction encodings to UP/DOWN/NEUTRAL."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("up", "bull", "bullish", "long", "+", "1", "higher"):
        return "UP"
    if s in ("down", "bear", "bearish", "short", "-", "-1", "lower"):
        return "DOWN"
    if s in ("neutral", "flat", "none", "sideways", "0", "mixed"):
        return "NEUTRAL"
    return s.upper()


def norm_cat(v):
    """Normalize a categorical (vol_change / vs_iv) for matching."""
    if v is None:
        return None
    s = str(v).strip().upper()
    if s.startswith("EXPAND") or s in ("RISING", "UP", "INCREASING"):
        return "EXPANDING"
    if s.startswith("CONTRACT") or s in ("FALLING", "DOWN", "DECREASING"):
        return "CONTRACTING"
    if s.startswith("STABLE") or s in ("FLAT", "STEADY", "UNCHANGED"):
        return "STABLE"
    if s.startswith("RICH") or s in ("EXPENSIVE", "OVERPRICED", "HIGH"):
        return "RICH"
    if s.startswith("CHEAP") or s in ("UNDERPRICED", "LOW"):
        return "CHEAP"
    if s.startswith("FAIR") or s in ("NEUTRAL",):
        return "FAIR"
    return s.split()[0] if s else s


def to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        val = float(re.sub(r"[^0-9.\-]", "", str(v)))
        return val if math.isfinite(val) else None
    except Exception:
        return None


def validate_heldout_forecast(pred):
    """Strictly validate the 6-field held-out trading forecast JSON schema.

    Returns (is_valid: bool, errors: list[str]).
    Empty objects, missing required fields, wrong JSON types (e.g. bool/list/dict),
    non-finite values, or out-of-domain categories fail closed.
    """
    if not isinstance(pred, dict):
        return False, ["response is not a JSON object"]
    errors = []

    raw_move = pred.get("move")
    if not isinstance(raw_move, str) or not raw_move.strip():
        errors.append("missing or non-string field: move")
    elif norm_move(raw_move) not in MOVE_CATS:
        errors.append("invalid move category: %r" % (raw_move,))

    raw_call = pred.get("call")
    if not isinstance(raw_call, str) or not raw_call.strip():
        errors.append("missing or non-string field: call")
    elif norm_dir(raw_call) not in DIR_CATS:
        errors.append("invalid call direction: %r" % (raw_call,))

    raw_vol = pred.get("vol_change")
    if not isinstance(raw_vol, str) or not raw_vol.strip():
        errors.append("missing or non-string field: vol_change")
    elif norm_cat(raw_vol) not in VOL_CHANGE_CATS:
        errors.append("invalid vol_change category: %r" % (raw_vol,))

    raw_vsiv = pred.get("vs_iv")
    if not isinstance(raw_vsiv, str) or not raw_vsiv.strip():
        errors.append("missing or non-string field: vs_iv")
    elif norm_cat(raw_vsiv) not in VS_IV_CATS:
        errors.append("invalid vs_iv category: %r" % (raw_vsiv,))

    raw_emp = pred.get("exp_move_pct")
    if isinstance(raw_emp, bool) or not isinstance(raw_emp, (int, float)):
        errors.append("missing or non-numeric field: exp_move_pct")
    elif not math.isfinite(float(raw_emp)) or float(raw_emp) < 0.0:
        errors.append("invalid exp_move_pct value: %r" % (raw_emp,))

    raw_conv = pred.get("conviction")
    if isinstance(raw_conv, bool) or not isinstance(raw_conv, (int, float)):
        errors.append("missing or non-numeric field: conviction")
    elif not math.isfinite(float(raw_conv)) or not (0.0 <= float(raw_conv) <= 10.0):
        errors.append("invalid conviction value: %r" % (raw_conv,))

    return (len(errors) == 0), errors


def _is_base_only(base_url, ft_url, args=None):
    if args is not None and getattr(args, "base_only", False):
        return True
    return (not ft_url) or (ft_url == base_url)


def _heldout_case_id(row, idx, user_text):
    if not isinstance(row, dict):
        raise ValueError("held-out row %d is not a JSON object" % idx)
    if "id" in row and row["id"] is not None:
        raw_id = row["id"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError("held-out row %d has invalid explicit id: %r" % (idx, raw_id))
        cid = str(raw_id)
        if not cid.strip():
            raise ValueError("held-out row %d has blank explicit id" % idx)
        return cid
    digest = hashlib.sha1((user_text or "").encode("utf-8", "replace")).hexdigest()[:8]
    return "case_%04d_%s" % (idx, digest)


def _canonical_sha256(obj):
    payload = (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compute_case_ids_sha256(case_ids):
    """Return deterministic SHA-256 hex digest of an ordered sequence of case IDs."""
    return _canonical_sha256([str(cid) for cid in case_ids])


def validate_heldout_case_ids(rows, source_label="held-out"):
    """Validate that every held-out row resolves to a strictly unique case_id.

    Fails closed with ValueError on invalid, blank, or duplicate explicit/resolved
    case IDs before any LLM request is issued or paired statistic is computed.
    Returns (case_ids: list[str], case_ids_sha256: str).
    """
    if not isinstance(rows, list):
        raise ValueError("%s rows must be a list" % source_label)
    case_ids = []
    seen = {}
    duplicates = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("%s row %d is not a JSON object" % (source_label, idx))
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            raise ValueError("%s row %d is missing a valid 'messages' list" % (source_label, idx))
        user = next(
            (m.get("content") for m in msgs if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        cid = _heldout_case_id(row, idx, user)
        if cid in seen:
            duplicates.setdefault(cid, [seen[cid]]).append(idx)
        else:
            seen[cid] = idx
        case_ids.append(cid)
    if duplicates:
        dup_desc = ", ".join(
            "%r (rows %s)" % (cid, ",".join(str(i) for i in idxs))
            for cid, idxs in sorted(duplicates.items())
        )
        raise ValueError(
            "Duplicate case_id(s) in %s (%d unique across %d rows): %s"
            % (source_label, len(seen), len(rows), dup_desc)
        )
    return case_ids, compute_case_ids_sha256(case_ids)


def _load_and_validate_heldout(path, n=None):
    """Read a held-out JSONL file once, validate unique case IDs, and apply stride sampling.

    Validates ALL loaded rows before stride sampling AND the selected rows after
    sampling so duplicate explicit or resolved case IDs always fail closed with
    ValueError prior to any LLM requests.
    Returns (real_path, pre_sha256, rows, case_ids, case_ids_sha256).
    """
    real_path = os.path.realpath(path)
    with open(real_path, "rb") as f:
        raw_bytes = f.read()
    pre_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    rows = []
    for line in raw_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        return real_path, pre_sha256, [], [], None
    validate_heldout_case_ids(rows, source_label=path)
    if n and n < len(rows):
        # deterministic stride sample for reproducibility
        step = max(1, len(rows) // n)
        rows = rows[::step][:n]
    case_ids, case_ids_sha256 = validate_heldout_case_ids(rows, source_label=path)
    return real_path, pre_sha256, rows, case_ids, case_ids_sha256


def validate_battery_items(items, battery_name="battery"):
    """Validate that every battery item has a valid, strictly unique id.

    Fails closed with ValueError on missing, blank, or duplicate item IDs
    before any LLM request is issued.
    Returns (item_ids: list[str], case_ids_sha256: str).
    """
    if not isinstance(items, list):
        raise ValueError("battery %s must be a JSON list" % battery_name)
    item_ids = []
    seen = {}
    duplicates = {}
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError("battery %s item %d is not a JSON object" % (battery_name, idx))
        raw_id = it.get("id")
        if raw_id is None or isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError(
                "battery %s item %d has missing or invalid id: %r" % (battery_name, idx, raw_id)
            )
        item_id = str(raw_id)
        if not item_id.strip():
            raise ValueError("battery %s item %d has blank id" % (battery_name, idx))
        if item_id in seen:
            duplicates.setdefault(item_id, [seen[item_id]]).append(idx)
        else:
            seen[item_id] = idx
        item_ids.append(item_id)
    if duplicates:
        dup_desc = ", ".join(
            "%r (items %s)" % (cid, ",".join(str(i) for i in idxs))
            for cid, idxs in sorted(duplicates.items())
        )
        raise ValueError(
            "Duplicate item id(s) in battery %s (%d unique across %d items): %s"
            % (battery_name, len(seen), len(items), dup_desc)
        )
    return item_ids, compute_case_ids_sha256(item_ids)


def _validate_paired_row_ids(rows, side_label):
    """Validate every result row carries a valid, strictly unique case_id."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("%s rows must be a sequence" % side_label)
    ordered_ids = []
    by_id = {}
    duplicates = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("%s row %d is not a dict" % (side_label, idx))
        if "case_id" not in row or row["case_id"] is None:
            raise ValueError("%s row %d is missing required 'case_id'" % (side_label, idx))
        raw_id = row["case_id"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError("%s row %d has invalid 'case_id': %r" % (side_label, idx, raw_id))
        cid = str(raw_id)
        if not cid.strip():
            raise ValueError("%s row %d has blank 'case_id'" % (side_label, idx))
        if cid in by_id:
            duplicates.setdefault(cid, []).append(idx)
        else:
            by_id[cid] = row
        ordered_ids.append(cid)
    if duplicates or len(rows) != len(by_id):
        dup_list = ", ".join("%r" % k for k in sorted(duplicates))
        raise ValueError(
            "Duplicate case_id(s) in %s rows (%d unique across %d rows): %s"
            % (side_label, len(by_id), len(rows), dup_list)
        )
    return ordered_ids, by_id


def _compute_paired_heldout_ci(base_rows, ft_rows):
    """Compute paired bootstrap 95% CIs on validated unique case_ids for all-case accuracy.

    Fails closed with ValueError if either side has missing/blank/duplicate case_ids,
    if len(rows) != len(unique_ids), if compared membership differs between BASE and FT
    (set(base_ids) != set(ft_ids)), or if per-field eligibility disagrees on a case_id.
    """
    base_ids, by_id_base = _validate_paired_row_ids(base_rows, "BASE")
    ft_ids, by_id_ft = _validate_paired_row_ids(ft_rows, "FT")

    base_set = set(base_ids)
    ft_set = set(ft_ids)
    if len(base_rows) != len(base_set) or len(ft_rows) != len(ft_set):
        raise ValueError(
            "Paired row count does not match unique case_id count (BASE=%d/%d, FT=%d/%d)"
            % (len(base_rows), len(base_set), len(ft_rows), len(ft_set))
        )
    if base_set != ft_set:
        only_base = sorted(base_set - ft_set)
        only_ft = sorted(ft_set - base_set)
        raise ValueError(
            "Paired membership mismatch between BASE and FT "
            "(only_in_base=%r, only_in_ft=%r)" % (only_base[:10], only_ft[:10])
        )

    shared_ids = list(base_ids)
    case_ids_sha256 = compute_case_ids_sha256(shared_ids)
    if len(shared_ids) < 2:
        return {}

    try:
        import variance as _var
    except Exception:
        return {}

    paired = {
        "shared_n": len(shared_ids),
        "case_ids_sha256": case_ids_sha256,
    }
    for field in ("move", "call", "vol_change", "vs_iv"):
        cand = []
        base = []
        for cid in shared_ids:
            bc = (by_id_base[cid].get("correct") or {}).get(field)
            fc = (by_id_ft[cid].get("correct") or {}).get(field)
            if (bc is None) != (fc is None):
                raise ValueError(
                    "Asymmetric field eligibility for case_id %r on field %r (BASE=%r, FT=%r)"
                    % (cid, field, bc, fc)
                )
            if bc is None:
                continue
            base.append(bool(bc))
            cand.append(bool(fc))
        if len(cand) >= 2:
            delta, lo, hi, mde = _var.paired_bootstrap_delta_ci(
                cand, base, iters=2000, alpha=0.05, seed=1234
            )
            paired[field] = {
                "n": len(cand),
                "delta": delta,
                "ci_lo": lo,
                "ci_hi": hi,
                "mde": mde,
            }
    return paired


# ----------------------------------------------------------------------------- TARGET 1: held-out
def run_heldout(base_url, ft_url, n, args, system_override=None, label_tag="held-out",
                model_by_side=None):
    """Score the held-out 5-field vol set.

    Every attempted scenario is persisted as a terminal row. Headline accuracy
    metrics use the FULL eligible attempted denominator (unparsed/schema-failed/
    truncated/failed/refused rows count as incorrect). Strict valid-forecast
    coverage requires an allowed terminal status and all typed schema fields;
    loose JSON object parse count is reported separately as a diagnostic.
    """
    path = args.heldout
    if not os.path.exists(path):
        print("  [%s] SKIP — %s not found (training not finished yet?)" % (label_tag, path))
        return None
    real_path, pre_sha256, rows, selected_case_ids, case_ids_sha256 = _load_and_validate_heldout(path, n)
    if not rows:
        print("  [%s] SKIP — file empty" % label_tag)
        return None
    print("  [%s] %d rows from %s%s (case_ids_sha256=%s)" % (
        label_tag, len(rows), path,
        " (production system prompt)" if system_override else "",
        case_ids_sha256[:16]))

    base_only = _is_base_only(base_url, ft_url, args)
    sides = [("BASE", base_url)] + ([] if base_only else [("FT", ft_url)])

    out = {}
    for label, url in sides:
        model = (model_by_side or {}).get(label) or detect_model(url) or label
        moves_ok = dir_ok = vol_ok = vsiv_ok = 0
        move_elig = dir_elig = vol_elig = vsiv_elig = mae_elig = 0
        parsed_move_elig = parsed_dir_elig = parsed_vol_elig = parsed_vsiv_elig = 0
        parsed = json_parsed_count = total = 0
        census = new_failure_census()
        calib = []
        preds = []
        maes = []
        t0 = time.time()
        for i, row in enumerate(rows):
            msgs = row["messages"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), None)
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            gt_raw = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            gt = extract_json(gt_raw) or {}
            case_id = selected_case_ids[i]

            if system_override is not None:
                system = system_override
            mm = ([{"role": "system", "content": system}] if system else []) + \
                 [{"role": "user", "content": user}]
            prompt_sha256 = _canonical_sha256({"case_id": case_id, "messages": mm})
            txt, ctok, finish_reason, tcalls, err = chat_text(
                url, mm, max_tokens=args.move_max_tokens,
                temperature=0.0, model=model, timeout=args.timeout,
            )
            total += 1
            census["attempted"] += 1

            # Loose JSON object parse diagnostic (tracked separately from strict valid coverage)
            raw_pred = (
                extract_json(txt)
                if (not err and finish_reason not in ("error", "content_filter"))
                else None
            )
            is_json_parsed = isinstance(raw_pred, dict)
            if is_json_parsed:
                json_parsed_count += 1
            is_schema_valid, schema_errors = (
                validate_heldout_forecast(raw_pred)
                if is_json_parsed
                else (False, ["no JSON object parsed"])
            )

            # Fail closed on transport/HTTP error, content_filter, truncation (length),
            # disallowed finish_reason, empty, refusal, JSON parse failure, or schema error.
            fail_reason = classify_failure(
                txt, tcalls, err, finish_reason,
                parsed=is_json_parsed,
                schema_valid=is_schema_valid,
                require_json=True,
            )
            is_valid = (fail_reason is None)
            pred = raw_pred if is_valid else None
            status = "ok" if is_valid else fail_reason
            if is_valid:
                census["succeeded"] += 1
                parsed += 1
            else:
                census["failed"] += 1
                census["by_reason"][fail_reason] = census["by_reason"].get(fail_reason, 0) + 1

            gt_move = norm_move(gt.get("move"))
            gt_call = gt.get("call")
            gt_vol = gt.get("vol_change")
            gt_vsiv = gt.get("vs_iv")
            gv = to_float(gt.get("exp_move_pct"))

            is_move_elig = gt_move in MOVE_CATS
            is_dir_elig = gt_call is not None
            is_vol_elig = bool(gt_vol)
            is_vsiv_elig = bool(gt_vsiv)
            if is_move_elig:
                move_elig += 1
            if is_dir_elig:
                dir_elig += 1
            if is_vol_elig:
                vol_elig += 1
            if is_vsiv_elig:
                vsiv_elig += 1
            if gv is not None:
                mae_elig += 1

            move_hit = False
            call_hit = False
            vol_hit = False
            vsiv_hit = False
            if is_valid:
                if is_move_elig:
                    parsed_move_elig += 1
                    if norm_move(pred.get("move")) == gt_move:
                        moves_ok += 1
                        move_hit = True
                if is_dir_elig:
                    parsed_dir_elig += 1
                    if norm_dir(pred.get("call")) == norm_dir(gt_call):
                        dir_ok += 1
                        call_hit = True
                if is_vol_elig:
                    parsed_vol_elig += 1
                    if norm_cat(pred.get("vol_change")) == norm_cat(gt_vol):
                        vol_ok += 1
                        vol_hit = True
                if is_vsiv_elig:
                    parsed_vsiv_elig += 1
                    if norm_cat(pred.get("vs_iv")) == norm_cat(gt_vsiv):
                        vsiv_ok += 1
                        vsiv_hit = True
                pv = to_float(pred.get("exp_move_pct"))
                if pv is not None and gv is not None:
                    maes.append(abs(pv - gv))
                cv = to_float(pred.get("conviction"))
                if cv is not None and is_dir_elig:
                    calib.append((cv, bool(call_hit)))

            preds.append({
                "case_id": case_id,
                "status": status,
                "parsed": is_valid,
                "schema_valid": is_schema_valid,
                "json_parsed": is_json_parsed,
                "schema_errors": schema_errors,
                "model": model,
                "prompt_sha256": prompt_sha256,
                "err": err,
                "finish_reason": finish_reason,
                "completion_tokens": ctok,
                "pred": ({k: pred.get(k) for k in ("move", "call", "vol_change", "vs_iv", "exp_move_pct", "conviction")}
                         if is_valid else None),
                "gt": {k: gt.get(k) for k in ("move", "call", "vol_change", "vs_iv", "exp_move_pct", "conviction")},
                "correct": {
                    "move": move_hit if is_move_elig else None,
                    "call": call_hit if is_dir_elig else None,
                    "vol_change": vol_hit if is_vol_elig else None,
                    "vs_iv": vsiv_hit if is_vsiv_elig else None,
                },
            })
            if (i + 1) % 50 == 0:
                print("    [%s] %d/%d  valid=%.0f%% move(all)=%.0f%%" % (
                    label, i + 1, len(rows), 100 * parsed / total,
                    100 * moves_ok / max(1, move_elig)))

        secs = time.time() - t0
        finalize_failure_census(census)

        def _bucket(lo, hi):
            sub = [ok for c, ok in calib if lo <= c <= hi]
            return {"n": len(sub), "acc": (sum(sub) / len(sub)) if sub else None}

        calibration = {
            "low_1_4": _bucket(1, 4),
            "mid_5_7": _bucket(5, 7),
            "high_8_10": _bucket(8, 10),
        }

        # All-case headline metrics (full attempted eligible denominator)
        move_acc_all = (moves_ok / move_elig) if move_elig else 0.0
        dir_acc_all = (dir_ok / dir_elig) if dir_elig else 0.0
        vol_acc_all = (vol_ok / vol_elig) if vol_elig else 0.0
        vsiv_acc_all = (vsiv_ok / vsiv_elig) if vsiv_elig else 0.0

        # Conditional valid-only metrics (separately labeled; never used for ranking)
        move_acc_parsed = (moves_ok / parsed_move_elig) if parsed_move_elig else None
        dir_acc_parsed = (dir_ok / parsed_dir_elig) if parsed_dir_elig else None
        vol_acc_parsed = (vol_ok / parsed_vol_elig) if parsed_vol_elig else None
        vsiv_acc_parsed = (vsiv_ok / parsed_vsiv_elig) if parsed_vsiv_elig else None

        valid_cov = parsed / total if total else 0.0
        loose_json_rate = json_parsed_count / total if total else 0.0

        out[label] = {
            "n": total,
            "parsed": parsed,
            "parse_rate": valid_cov,
            "valid_count": parsed,
            "valid_coverage": valid_cov,
            "json_parsed": json_parsed_count,
            "json_parse_rate": loose_json_rate,
            "move_acc": move_acc_all,
            "dir_acc": dir_acc_all,
            "vol_change_acc": vol_acc_all,
            "vs_iv_acc": vsiv_acc_all,
            "move_acc_all": move_acc_all,
            "dir_acc_all": dir_acc_all,
            "vol_change_acc_all": vol_acc_all,
            "vs_iv_acc_all": vsiv_acc_all,
            "move_acc_parsed": move_acc_parsed,
            "dir_acc_parsed": dir_acc_parsed,
            "vol_change_acc_parsed": vol_acc_parsed,
            "vs_iv_acc_parsed": vsiv_acc_parsed,
            "exp_move_mae": statistics.mean(maes) if maes else None,
            "exp_move_mae_n": len(maes),
            "exp_move_mae_coverage": (len(maes) / mae_elig) if mae_elig else 0.0,
            "failure_census": census,
            "calibration": calibration,
            "rows": preds,
            "secs": round(secs, 1),
            "model": model,
            "heldout_sha256": pre_sha256,
            "case_ids_sha256": case_ids_sha256,
        }

        print(
            "  [%s:%s] valid_cov=%d/%d (%.1f%%) [json_parsed=%d/%d] move_acc(all)=%.1f%% [valid=%s] "
            "dir_acc(all)=%.1f%% mae=%s (%.0fs)" % (
                label_tag, label, parsed, total, 100 * valid_cov,
                json_parsed_count, total,
                100 * move_acc_all,
                ("%.1f%%" % (100 * move_acc_parsed)) if move_acc_parsed is not None else "n/a",
                100 * dir_acc_all,
                ("%.3f" % out[label]["exp_move_mae"]) if out[label]["exp_move_mae"] is not None else "n/a",
                secs,
            )
        )
        calib_str = " ".join(
            "%s=%s" % (k, ("%.0f%%" % (100 * calibration[k]["acc"]) if calibration[k]["acc"] is not None else "n/a"))
            for k in ("low_1_4", "mid_5_7", "high_8_10")
        )
        print("  [%s:%s] vol_change_acc(all)=%.1f%% vs_iv_acc(all)=%.1f%% | call-acc by conviction: %s" % (
            label_tag, label, 100 * vol_acc_all, 100 * vsiv_acc_all, calib_str))

        # majority-baseline + balanced-accuracy (all-case, unparsed counted wrong)
        def _field_metrics(field, normfn):
            pairs = [
                (normfn(pr["gt"].get(field)),
                 normfn(pr["pred"].get(field)) if isinstance(pr.get("pred"), dict) else None,
                 bool(pr.get("parsed")))
                for pr in preds
            ]
            pairs = [(g, p, ok) for g, p, ok in pairs if g]
            if not pairs:
                return None
            gtc = Counter(g for g, _, _ in pairs)
            nn = len(pairs)
            raw = sum(1 for g, p, ok in pairs if ok and g == p) / nn
            maj = max(gtc.values()) / nn
            bal = sum(
                sum(1 for g, p, ok in pairs if g == c and ok and p == c) / gtc[c]
                for c in gtc
            ) / len(gtc)
            parsed_pairs = [(g, p) for g, p, ok in pairs if ok]
            parsed_raw = (
                sum(1 for g, p in parsed_pairs if g == p) / len(parsed_pairs)
                if parsed_pairs else None
            )
            return {
                "raw": raw,
                "parsed_raw": parsed_raw,
                "parsed_n": len(parsed_pairs),
                "total_n": nn,
                "majority": maj,
                "balanced": bal,
                "lift": raw - maj,
            }

        fm = {
            f: _field_metrics(f, nf)
            for f, nf in (("move", norm_move), ("vol_change", norm_cat), ("vs_iv", norm_cat), ("call", norm_dir))
        }
        out[label]["field_metrics"] = fm
        print("  [%s:%s]  field        all  / major / balncd / lift  (valid-only)" % (label_tag, label))
        for _f, _m in fm.items():
            if _m:
                pr_str = ("%.1f%%" % (100 * _m["parsed_raw"])) if _m["parsed_raw"] is not None else "n/a"
                print("                 %-11s %5.1f%% /%5.1f%% /%5.1f%% / %+5.1fpp  (%s)" % (
                    _f, 100 * _m["raw"], 100 * _m["majority"], 100 * _m["balanced"],
                    100 * _m["lift"], pr_str))

    post_sha256 = None
    try:
        with open(real_path, "rb") as f:
            post_sha256 = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        post_sha256 = None
    out["_provenance"] = {
        "path": real_path,
        "sha256": pre_sha256,
        "post_sha256": post_sha256,
        "intact": (post_sha256 == pre_sha256),
        "case_count": len(selected_case_ids),
        "case_ids_sha256": case_ids_sha256,
    }
    if getattr(args, "contract_dir", None) and post_sha256 != pre_sha256:
        import gauntlet_contract as gc
        raise gc.ContractError(
            "held-out dataset %s changed during evaluation run (pre=%s post=%s)"
            % (real_path, pre_sha256[:16], (post_sha256 or "missing")[:16])
        )

    if "BASE" in out and "FT" in out:
        out["_paired"] = _compute_paired_heldout_ci(out["BASE"]["rows"], out["FT"]["rows"])

    # --- optional: token-level loss / perplexity on the assistant answer ---
    ppl = None
    if system_override is not None:
        pass  # skew slice: skip perplexity (GT answer is conditioned on the bare prompt)
    elif args.hf_base and args.hf_ft:
        ppl = heldout_perplexity(rows, args)
    else:
        print("  [held-out:ppl] SKIP token-loss/perplexity — pass --hf-base and --hf-ft to enable")
    if ppl:
        if "BASE" in out and "BASE" in ppl:
            out["BASE"]["ppl"] = ppl["BASE"]
        if "FT" in out and "FT" in ppl:
            out["FT"]["ppl"] = ppl["FT"]
    return out


# ----------------------------------------------------------------------------- TARGET 1b: train/serve skew
DEFAULT_EXITMGR_DIR = os.path.join(HOME, "exitmgr-app")

SKEW_FORMAT_REDIRECT = (
    'For THIS request only, ignore the trade-idea output format above. You are assessing '
    'volatility for the following instrument. Answer ONLY as JSON: '
    '{"move":"QUIET|NORMAL|ELEVATED|EXPLOSIVE","exp_move_pct":number,'
    '"vol_change":"EXPANDING|STABLE|CONTRACTING","vs_iv":"CHEAP|FAIR|RICH",'
    '"call":"BULLISH|BEARISH|NEUTRAL","conviction":1-10}.'
)


def load_production_system_prompt(args):
    """Pull the LIVE production strategist SYSTEM_PROMPT so the skew slice tracks the
    real prompt if you edit strategist.py. Tries an import first; falls back to
    statically extracting the constant. Returns the prompt string, or None."""
    exitmgr_dir = (getattr(args, "exitmgr_dir", None)
                   or os.environ.get("EXITMGR_DIR")
                   or DEFAULT_EXITMGR_DIR)
    strat_path = os.path.join(exitmgr_dir, "exitmgr", "strategist.py")
    try:
        if exitmgr_dir not in sys.path:
            sys.path.insert(0, exitmgr_dir)
        from exitmgr.strategist import SYSTEM_PROMPT as _SP  # type: ignore
        if isinstance(_SP, str) and _SP.strip():
            return _SP
    except Exception as e:
        print("  [skew] import of exitmgr.strategist failed (%s) — trying static read" % str(e)[:120])
    try:
        if not os.path.exists(strat_path):
            print("  [skew] SKIP — strategist.py not found at %s" % strat_path)
            return None
        import ast
        with open(strat_path, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("_UNIVERSE", "_SCORING", "_CONTRACT", "SYSTEM_PROMPT"):
                    try:
                        ns[name] = eval(compile(ast.Expression(node.value), strat_path, "eval"), {}, ns)
                    except Exception:
                        pass
                    if name == "SYSTEM_PROMPT" and isinstance(ns.get("SYSTEM_PROMPT"), str):
                        return ns["SYSTEM_PROMPT"]
        print("  [skew] SKIP — could not extract SYSTEM_PROMPT from %s" % strat_path)
        return None
    except Exception as e:
        print("  [skew] SKIP — failed to load production prompt (%s)" % str(e)[:160])
        return None


def build_skew_system_prompt(args):
    """Production SYSTEM_PROMPT + blank line + the 5-field format redirect. None if unloadable."""
    sp = load_production_system_prompt(args)
    if not sp:
        return None
    return sp + "\n\n" + SKEW_FORMAT_REDIRECT


def run_skew(base_url, ft_url, n, args, model_by_side=None):
    """Train/serve-skew slice: identical to run_heldout() under production framing."""
    skew_sys = build_skew_system_prompt(args)
    if not skew_sys:
        print("  [skew] SKIP — production system prompt unavailable (see warnings above)")
        return None
    return run_heldout(base_url, ft_url, n, args,
                       system_override=skew_sys, label_tag="skew",
                       model_by_side=model_by_side)


def heldout_perplexity(rows, args):
    """Mean token-level loss + perplexity on the GROUND-TRUTH assistant answer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        print("  [held-out:ppl] SKIP — torch/transformers/peft not importable (%s)" % str(e)[:80])
        return None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.hf_base, trust_remote_code=True)

    def load(base_path, adapter=None):
        m = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto")
        if adapter:
            m = PeftModel.from_pretrained(m, adapter)
            m = m.merge_and_unload()
        m.eval()
        return m

    def loss_over(model, sample):
        losses = []
        with torch.no_grad():
            for row in sample:
                msgs = row["messages"]
                prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
                answer = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                try:
                    prompt_ids = tok.apply_chat_template(
                        prompt_msgs, add_generation_prompt=True, return_tensors="pt")
                except Exception:
                    text = "".join(m["content"] for m in prompt_msgs)
                    prompt_ids = tok(text, return_tensors="pt").input_ids
                ans_ids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids
                input_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(model.device)
                labels = input_ids.clone()
                labels[:, :prompt_ids.shape[1]] = -100
                out = model(input_ids=input_ids, labels=labels)
                if out.loss is not None and not math.isnan(float(out.loss)):
                    losses.append(float(out.loss))
        return statistics.mean(losses) if losses else None

    pn = min(len(rows), args.ppl_n)
    sample = rows[:pn]
    print("  [held-out:ppl] forward pass on %d rows (dev=%s)" % (pn, dev))
    res = {}
    for label, base_path, adapter in [
        ("BASE", args.hf_base, None),
        ("FT", args.hf_base, args.hf_ft),
    ]:
        model = load(base_path, adapter)
        nll = loss_over(model, sample)
        del model
        if dev == "cuda":
            import torch as _t
            _t.cuda.empty_cache()
        res[label] = {"loss": nll, "ppl": math.exp(nll) if nll is not None else None}
        print("    [%s] loss=%s ppl=%s" % (
            label, ("%.4f" % nll) if nll is not None else "n/a",
            ("%.3f" % res[label]["ppl"]) if res[label]["ppl"] is not None else "n/a"))
    return res


# ----------------------------------------------------------------------------- objective check engine
_SIGNED_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.+-])"
    r"(?:"
    r"(?P<neg_word>\b(?:negative|minus)\s+)\$?"
    r"|(?P<sign_pre>[+-])\s*\$?"
    r"|\$(?P<sign_post>[+-])"
    r"|\$"
    r")?"
    r"(?P<num>\d+(?:\.\d+)?)"
    r"(?![A-Za-z0-9_]|\.\d)",
    re.IGNORECASE,
)


_NEGATION_PRE_RE = re.compile(
    r"(?:!=|!==|\u2260|<>)"
    r"|\b(?:not|cannot|never|neither|nor|wrong|incorrect|false|"
    r"rather\s+than|instead\s+of|unlike|except|excluding)\b"
    r"|\b[a-z]+n't\b",
    re.IGNORECASE,
)

_NEGATION_POST_RE = re.compile(
    r"^\s*(?:"
    r"(?:is|are|was|were)\s+(?:wrong|incorrect|false|not\b)"
    r"|\(\s*(?:wrong|incorrect|false|not\s+(?:correct|right|valid|true|the\s+answer))\s*\)"
    r")",
    re.IGNORECASE,
)

_FINAL_ANSWER_CUE_RE = re.compile(
    r"(?:^|[\n.?;]|!(?!=))\s*[^\n.?;]{0,40}?"
    r"\b(?:final\s+answer|(?:the\s+)?(?:correct\s+)?answer|result|"
    r"breakeven(?:\s+at\s+expiry)?|maximum\s+(?:dollar\s+)?(?:position\s+)?size|"
    r"max\s+size|contract\s+cost|total(?:\s+cost)?)\s*"
    r"(?:(?:is|equals)\b|(?<![!<>])=(?!=)|:)\s*([^\n?;]+?)(?=(?:\.(?!\d))|!(?!=)|[\n?;]|$)",
    re.IGNORECASE,
)

_STRUCTURED_NUM_KEYS = ("final_answer", "answer", "result", "value", "number")


def _normalize_numeric_text(text):
    s = re.sub(r"<mm:think>.*?</mm:think>", "", text or "", flags=re.S)
    s = s.replace("\u2212", "-")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return s


def _signed_number_matches(norm_text):
    """Return list of (float_val, start, end) for signed numbers in normalized text."""
    out = []
    for m in _SIGNED_NUM_RE.finditer(norm_text):
        num = float(m.group("num"))
        sign = -1.0 if (
            m.group("neg_word")
            or m.group("sign_pre") == "-"
            or m.group("sign_post") == "-"
        ) else 1.0
        out.append((sign * num, m.start(), m.end()))
    return out


def _asserted_numbers_in_span(span):
    """Extract non-negated signed numbers from a span, excluding explicitly negated mentions."""
    matches = _signed_number_matches(span)
    asserted = []
    prev_end = 0
    for val, start, end in matches:
        # Local clause prefix: since the last sentence/clause boundary or previous number
        seg = span[prev_end:start]
        # Reset at clause delimiters (., !, ?, ;, \n, ,) when not inside a decimal or '!='
        delim = None
        for dm in re.finditer(r"(?:\.(?!\d)|!(?!=)|[?;\n,])", seg):
            delim = dm
        local_prefix = seg[delim.end():] if delim is not None else seg
        local_suffix = span[end:]
        if _NEGATION_PRE_RE.search(local_prefix):
            prev_end = end
            continue
        if _NEGATION_POST_RE.search(local_suffix):
            prev_end = end
            continue
        asserted.append(val)
        prev_end = end
    return asserted


def _unique_scalar(vals, tol=1e-9):
    """Return the unique float if all values agree within tol, else None."""
    if not vals:
        return None
    first = vals[0]
    if all(abs(v - first) <= tol for v in vals[1:]):
        return first
    return None


def extract_signed_numbers(text):
    """Extract exact signed numeric quantities from text without stripping signs or decimals."""
    if not text:
        return []
    s = _normalize_numeric_text(text)
    return [val for val, _st, _en in _signed_number_matches(s)]


def extract_final_number(text):
    """Extract ONE unambiguous, non-negated final numeric answer from structured JSON or text.

    Rejects:
      - Explicitly negated numbers (e.g. 'not +3', 'is not 298.29', '+3 is wrong', '!= 3')
      - Ambiguous multi-number prose without a single final scalar or answer cue
      - Non-numeric or boolean JSON answer fields
    """
    if not text or not text.strip():
        return None
    cleaned = re.sub(r"<mm:think>.*?</mm:think>", "", text, flags=re.S).strip()
    if not cleaned:
        return None

    # 1. Structured JSON result object (e.g. {"answer": -3} or {"final_answer": 298.29})
    jobj = extract_json(cleaned)
    if isinstance(jobj, dict):
        matched_keys = [k for k in _STRUCTURED_NUM_KEYS if k in jobj]
        if matched_keys:
            vals = []
            for k in matched_keys:
                raw_v = jobj[k]
                if isinstance(raw_v, bool) or raw_v is None:
                    return None
                if isinstance(raw_v, (int, float)):
                    if not math.isfinite(float(raw_v)):
                        return None
                    vals.append(float(raw_v))
                elif isinstance(raw_v, str):
                    sub = _unique_scalar(_asserted_numbers_in_span(_normalize_numeric_text(raw_v)))
                    if sub is None:
                        return None
                    vals.append(sub)
                else:
                    return None
            return _unique_scalar(vals)
        # Pure JSON object without any recognized numeric answer field fails closed
        stripped_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S).strip()
        if stripped_json.startswith("{") and stripped_json.endswith("}"):
            return None

    norm = _normalize_numeric_text(cleaned)

    # 2. Explicit final-answer statement cue (use the LAST cue in the response)
    cue_matches = list(_FINAL_ANSWER_CUE_RE.finditer(norm))
    if cue_matches:
        last_span = cue_matches[-1].group(1)
        return _unique_scalar(_asserted_numbers_in_span(last_span))

    # 3. Explicit positive equation ('=' not part of !=, <=, >=, ==)
    eq_matches = list(re.finditer(r"(?<![!<>=])=(?!=)", norm))
    if eq_matches:
        rhs = norm[eq_matches[-1].end():]
        # Stop at next sentence boundary if any
        s_end = re.search(r"(?:\.(?!\d)|[!?;\n])", rhs)
        rhs_clause = rhs[:s_end.start()] if s_end else rhs
        rhs_vals = _asserted_numbers_in_span(rhs_clause)
        if rhs_vals or _NEGATION_PRE_RE.search(rhs_clause):
            return _unique_scalar(rhs_vals)

    # 4. Whole-text single unambiguous non-negated scalar
    return _unique_scalar(_asserted_numbers_in_span(norm))


def _extract_tool_names(tool_calls):
    """Extract called tool names strictly from structured tool_calls."""
    if not tool_calls:
        return []
    if tool_calls is True:
        return ["__any__"]
    names = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"].strip():
                    names.append(fn["name"].strip())
                elif isinstance(tc.get("name"), str) and tc["name"].strip():
                    names.append(tc["name"].strip())
            elif isinstance(tc, str) and tc.strip():
                names.append(tc.strip())
    return names


def _check_label(c):
    typ = c["type"]
    if typ == "non_empty":
        return "non_empty"
    if typ == "contains_none":
        return "no-blocklist"
    if typ == "contains_all":
        return "recall"
    if typ == "min_items":
        return "items>=%d" % c["n"]
    if typ == "equals_number":
        return "==%g" % c["value"]
    if typ == "max_sentences":
        return "<=%dsent" % c["n"]
    if typ == "regex_any":
        return c.get("label", "regex_any")
    if typ == "regex_none":
        return c.get("label", "regex_none")
    if typ == "tool_intent":
        return "tool:" + c.get("name", "")
    return typ


def run_check(c, text, tool_calls=None, *, err=None, finish_reason=None):
    """Returns (label, passed_bool_or_None). None => needs a judge (not auto-scorable).

    Fails closed on transport/HTTP errors, content_filter finish state, empty
    responses, explicit refusals, wrong numeric signs, and missing structured
    tool calls.
    """
    typ = c["type"]
    lbl = _check_label(c)
    if typ in ("llm_judge", "ib_grader"):
        return (lbl, None)

    # Hard fail-closed gate on API/transport errors, content filter, or truncation
    if err or finish_reason in ("error", "content_filter", "length"):
        return (lbl, False)

    t = text or ""
    tl = t.lower()
    has_text = bool(t.strip())
    called_tools = _extract_tool_names(tool_calls)

    # Tool intent is scored strictly from structured tool_calls (never prose markers)
    if typ == "tool_intent":
        nm = c.get("name", "")
        if not called_tools:
            return (lbl, False)
        if nm:
            nmn = nm.lower().replace("-", "").replace("_", "")
            hit = any(
                tn != "__any__" and tn.lower().replace("-", "").replace("_", "") == nmn
                for tn in called_tools
            )
            return (lbl, hit)
        return (lbl, True)

    # All remaining objective content checks require non-empty, non-refused text
    if not has_text:
        return (lbl, False)
    if not c.get("allow_refusal", False) and is_refusal(t):
        return (lbl, False)

    if typ == "non_empty":
        return (lbl, True)
    if typ == "contains_none":
        hits = [v for v in c["values"] if v.lower() in tl]
        return (lbl, len(hits) == 0)
    if typ == "contains_all":
        miss = [v for v in c["values"] if v.lower() not in tl]
        return (lbl, len(miss) == 0)
    if typ == "min_items":
        n = len([ln for ln in t.splitlines() if ln.strip() and re.search(r"[A-Z]{1,5}", ln)])
        return (lbl, n >= c["n"])
    if typ == "equals_number":
        target = float(c["value"])
        tol = float(c.get("tolerance", 1e-6))
        val = extract_final_number(t)
        return (lbl, val is not None and abs(val - target) <= tol)
    if typ == "max_sentences":
        n = len([s for s in re.split(r"[.!?]+", t.strip()) if s.strip()])
        return (lbl, n <= c["n"])
    if typ == "regex_any":
        return (lbl, any(re.search(p, t, re.I) for p in c["patterns"]))
    if typ == "regex_none":
        return (lbl, not any(re.search(p, t, re.I) for p in c["patterns"]))
    return (lbl, None)


def score_battery(label, url, battery, system_default, args, *,
                  model=None, battery_sha256=None, battery_name=None):
    """Run one battery's prompts/turns against a URL, score objective checks.
    Returns per-battery aggregate including full failure_census and per-row telemetry."""
    _item_ids, case_ids_sha256 = validate_battery_items(
        battery, battery_name=battery_name or label
    )
    eff_model = model or detect_model(url) or label
    hp = ht = judge_needed = empty = 0
    secs = 0.0
    census = new_failure_census()
    rows = []
    for it in battery:
        s = it.get("system", system_default)
        base_msgs = [{"role": "system", "content": s}] if s else []
        mt = it.get("max_tokens", 500)
        thinking = it.get("thinking")
        tools = it.get("tools")
        text = ""
        tcalls = []
        ctok_total = 0
        finish_reason = None
        err = None
        transcript = None
        sent_turns = []
        t0 = time.time()
        if it.get("turns"):
            msgs = list(base_msgs)
            transcript = []
            for u in it["turns"]:
                sent_turns.append(u)
                msgs.append({"role": "user", "content": u})
                txt, ctok, fr, tc, e = chat_text(
                    url, msgs, max_tokens=mt, temperature=0.0,
                    thinking=thinking, tools=tools, model=eff_model,
                    timeout=args.timeout,
                )
                ctok_total += ctok
                finish_reason = fr
                msgs.append({"role": "assistant", "content": txt})
                transcript.append({"user": u, "assistant": txt, "err": e, "finish": fr})
                text = txt
                tcalls = tcalls + tc
                if e:
                    err = e
                    break
            prompt_payload = {
                "id": str(it["id"]),
                "system": base_msgs[0]["content"] if base_msgs else None,
                "tools": tools or None,
                "turns": sent_turns,
            }
        else:
            msgs = base_msgs + [{"role": "user", "content": it["prompt"]}]
            prompt_payload = {
                "id": str(it["id"]),
                "system": base_msgs[0]["content"] if base_msgs else None,
                "tools": tools or None,
                "prompt": msgs[-1]["content"],
            }
            text, ctok_total, finish_reason, tcalls, err = chat_text(
                url, msgs, max_tokens=mt, temperature=0.0,
                thinking=thinking, tools=tools, model=eff_model,
                timeout=args.timeout,
            )
        prompt_sha256 = _canonical_sha256(prompt_payload)
        dt = time.time() - t0
        secs += dt
        if not (text or "").strip() and not tcalls:
            empty += 1

        allow_refusal = any(c.get("allow_refusal", False) for c in it.get("checks", []))
        fail_reason = classify_failure(
            text, tcalls, err, finish_reason,
            parsed=True, require_json=False, allow_refusal=allow_refusal,
        )
        status = "ok" if fail_reason is None else fail_reason
        census["attempted"] += 1
        if fail_reason is None:
            census["succeeded"] += 1
        else:
            census["failed"] += 1
            census["by_reason"][fail_reason] = census["by_reason"].get(fail_reason, 0) + 1

        marks = []
        check_details = []
        for c in it.get("checks", []):
            nm, ok = run_check(c, text, tcalls, err=err, finish_reason=finish_reason)
            if ok is None:
                judge_needed += 1
                marks.append("%s:JUDGE" % nm)
                check_details.append({"name": nm, "type": c["type"], "passed": None,
                                      "criteria": c.get("criteria")})
            else:
                ht += 1
                hp += 1 if ok else 0
                marks.append("%s:%s" % (nm, "OK" if ok else "X"))
                check_details.append({"name": nm, "type": c["type"], "passed": bool(ok)})

        row = {
            "id": it["id"],
            "secs": round(dt, 3),
            "status": status,
            "err": err,
            "finish_reason": finish_reason,
            "completion_tokens": ctok_total,
            "tool_calls": tcalls,
            "text": text,
            "marks": marks,
            "checks": check_details,
            "model": eff_model,
            "prompt_sha256": prompt_sha256,
        }
        if battery_sha256 is not None:
            row["battery_sha256"] = battery_sha256
        if transcript is not None:
            row["transcript"] = transcript
        rows.append(row)
        if args.verbose:
            print("    [%s] %-22s %5.1fs  [%s]  %s" % (label, it["id"], dt, status, "  ".join(marks)))

    finalize_failure_census(census)
    res = {
        "hp": hp,
        "ht": ht,
        "judge": judge_needed,
        "empty": empty,
        "secs": round(secs, 1),
        "n": len(battery),
        "case_ids_sha256": case_ids_sha256,
        "failure_census": census,
        "rows": rows,
        "model": eff_model,
    }
    if battery_sha256 is not None:
        res["battery_sha256"] = battery_sha256
    return res


def load_battery_frozen(name, root=None):
    """Read a battery JSON file once, validate unique item IDs, and return (items, sha256_hex)."""
    bdir = os.path.join(root or GG_ROOT, "batteries")
    path = os.path.join(bdir, name + ".json")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        raw = f.read()
    items = json.loads(raw.decode("utf-8"))
    validate_battery_items(items, battery_name=name)
    return items, hashlib.sha256(raw).hexdigest()


def load_battery(name, root=None):
    items, _sha = load_battery_frozen(name, root=root)
    return items


def run_gauntlet(base_url, ft_url, battery_names, args, system_default=None, *,
                 frozen_batteries=None, model_by_side=None):
    """Generic: run a list of batteries base-vs-FT (or base-only), return per-target aggregate."""
    base_only = _is_base_only(base_url, ft_url, args)
    out = {"BASE": {}, "FT": {}, "_batteries": list(battery_names), "_base_only": base_only}
    if not battery_names:
        print("  [%s] SKIP — no batteries configured for this target" % args.mode)
        return out

    # Validate every battery's item IDs BEFORE issuing any LLM requests
    loaded_batteries = []
    battery_case_digests = {}
    for bn in battery_names:
        if frozen_batteries and bn in frozen_batteries:
            b = frozen_batteries[bn]["items"]
            b_sha = frozen_batteries[bn]["sha256"]
        else:
            b, b_sha = load_battery_frozen(bn)
        if b is None:
            print("  [%s] SKIP battery '%s' — not found in %s" % (args.mode, bn, BATT_DIR))
            continue
        _ids, b_case_ids_sha = validate_battery_items(b, battery_name=bn)
        loaded_batteries.append((bn, b, b_sha, b_case_ids_sha))
        battery_case_digests[bn] = b_case_ids_sha

    if battery_case_digests:
        out["_battery_case_ids_sha256"] = battery_case_digests
        out["_case_ids_sha256"] = _canonical_sha256(battery_case_digests)

    sides = [("BASE", base_url)] + ([] if base_only else [("FT", ft_url)])
    for bn, b, b_sha, _b_case_ids_sha in loaded_batteries:
        print("  -- battery %s (%d items) --" % (bn, len(b)))
        for label, url in sides:
            eff_model = (model_by_side or {}).get(label)
            out[label][bn] = score_battery(
                label, url, b, system_default, args,
                model=eff_model, battery_sha256=b_sha, battery_name=bn,
            )
            r = out[label][bn]
            fc = r["failure_census"]
            print("     [%s] %s: hard %d/%d  cov=%d/%d (%.0f%%)  judge=%d  empty=%d  %.0fs" % (
                label, bn, r["hp"], r["ht"], fc["succeeded"], fc["attempted"],
                100 * fc["coverage_rate"], r["judge"], r["empty"], r["secs"]))
    return out


# ----------------------------------------------------------------------------- contract & blind-judge integration
def _resolve_manifest_profile(run_bats, root_str):
    import gauntlet_contract as gc

    run_bats_sorted = sorted(set(run_bats))
    profiles = load_manifest_profiles(root_str)
    for cand in ("trading", "general", "standard", "heavy", "smoke"):
        if sorted(profiles.get(cand, [])) == run_bats_sorted:
            return cand
    for pname, pval in profiles.items():
        if sorted(pval) == run_bats_sorted:
            return pname
    raise gc.ContractError(
        "run batteries %r do not match any profile in battery_manifest.json" % (run_bats_sorted,)
    )


def _prepare_frozen_contracts(planned_bats, model_by_side, args, sysdefault_path=None):
    """Freeze battery bytes, system prompt, and hash-bound run manifests BEFORE any model calls."""
    import gauntlet_contract as gc

    if not planned_bats:
        return None
    contract_dir = Path(os.path.realpath(os.path.expanduser(args.contract_dir)))
    contract_dir.mkdir(parents=True, exist_ok=True)
    root = Path(os.path.realpath(GG_ROOT))
    profile = _resolve_manifest_profile(planned_bats, str(root))

    base_only = _is_base_only(args.base_url, args.ft_url, args)
    sides = [("BASE", args.base_url)] + ([] if base_only else [("FT", args.ft_url)])

    runs_by_side = {}
    frozen_batteries = None
    frozen_system_text = None
    for side, url in sides:
        safe_model = gc.sanitize_model_id(model_by_side.get(side) or side, fallback=side)
        run_path = contract_dir / ("run.%s.json" % side)
        comp_path = contract_dir / ("completion.%s.json" % side)
        run, side_frozen, side_sys_text = gc.freeze_run_manifest(
            root,
            profile=profile,
            label=side,
            endpoint=url,
            model_id=safe_model,
            out_path=run_path,
            system=sysdefault_path or "none",
            timeout=args.timeout,
        )
        if frozen_batteries is None:
            frozen_batteries = side_frozen
            frozen_system_text = side_sys_text
        else:
            # Verify both sides froze identical battery bytes
            for bn, info in side_frozen.items():
                if frozen_batteries[bn]["sha256"] != info["sha256"]:
                    raise gc.ContractError(
                        "battery %s changed while freezing multi-side run contracts" % bn
                    )
        runs_by_side[side] = {
            "run": run,
            "run_path": run_path,
            "comp_path": comp_path,
        }

    return {
        "root": root,
        "contract_dir": contract_dir,
        "profile": profile,
        "sides": runs_by_side,
        "frozen_batteries": frozen_batteries or {},
        "frozen_system_text": frozen_system_text,
    }


def _finalize_verified_contracts(prepared, results):
    """Write gordon-result.v2 files and verify completion receipts against pre-call frozen manifests."""
    import gauntlet_contract as gc

    if not prepared:
        return None
    root = prepared["root"]
    contract_dir = prepared["contract_dir"]
    profile = prepared["profile"]

    combined_by_side = {"BASE": {}, "FT": {}}
    for tag in ("trading-gauntlet", "gordon"):
        g = results.get(tag)
        if not g:
            continue
        for side in ("BASE", "FT"):
            for bn, bres in (g.get(side) or {}).items():
                combined_by_side[side][bn] = bres

    receipts = {}
    for side, info in prepared["sides"].items():
        side_bats = combined_by_side.get(side) or {}
        if not side_bats:
            continue
        run = info["run"]
        run_path = info["run_path"]
        comp_path = info["comp_path"]
        for bn, bres in sorted(side_bats.items()):
            res_path = contract_dir / ("%s.%s.json" % (bn, side))
            gc.write_battery_result_v2(
                run,
                bn,
                bres["rows"],
                res_path,
                expected_battery_sha256=bres.get("battery_sha256"),
            )
        comp = gc.complete_run_manifest(root, run_path, contract_dir, comp_path)
        verified_run, _ = gc.verify_complete(root, run_path, comp_path, contract_dir)
        receipts[side] = {
            "run_id": verified_run["run_id"],
            "profile": profile,
            "manifest_sha256": verified_run["manifest_sha256"],
            "battery_set_sha256": verified_run["battery_set_sha256"],
            "case_ids_sha256": _canonical_sha256(
                {e["name"]: e["case_ids_sha256"] for e in verified_run["batteries"]}
            ),
            "run_manifest": str(run_path),
            "completion_receipt": str(comp_path),
            "verified": True,
            "result_count": len(comp["results"]),
        }
    return receipts


def _write_blind_judge_packets(results, args, frozen_batteries=None):
    """Export all llm_judge / ib_grader items via judge_blind.write_pending()."""
    import judge_blind as jb

    jdir = Path(args.blind_judge_dir).expanduser().absolute()
    jdir.mkdir(parents=True, exist_ok=True)
    records = []
    for tag in ("trading-gauntlet", "gordon"):
        g = results.get(tag)
        if not g:
            continue
        for side in ("BASE", "FT"):
            for bn, bres in (g.get(side) or {}).items():
                if frozen_batteries and bn in frozen_batteries:
                    batt_list = frozen_batteries[bn]["items"]
                else:
                    batt_list = load_battery(bn) or []
                validate_battery_items(batt_list, battery_name=bn)
                batt_items = {str(it["id"]): it for it in batt_list}
                model = bres.get("model") or side
                for row in bres.get("rows", []):
                    it = batt_items.get(row["id"], {})
                    prompt = it.get("prompt") or " ".join(it.get("turns") or [])
                    for chk in row.get("checks", []):
                        if chk.get("passed") is None:
                            records.append({
                                "key": "%s/%s/%s/%s/%s" % (tag, bn, row["id"], side, chk["type"]),
                                "model": model,
                                "label": side,
                                "domain": tag,
                                "battery": bn,
                                "id": row["id"],
                                "criteria": chk.get("criteria") or "",
                                "prompt": prompt,
                                "response": row.get("text") or "",
                                "status": row.get("status") or "ok",
                                "protocol": "eval_finetuned.v2",
                            })
    pending_path = str(jdir / "pending_judge.jsonl")
    map_path = str(jdir / "judge_uid_map.json")
    count = jb.write_pending(records, pending_path, map_path) if records else 0
    return {
        "count": count,
        "pending_path": pending_path if records else None,
        "map_path": map_path if records else None,
    }


# ----------------------------------------------------------------------------- reporting
def pct(x):
    return "%.1f%%" % (100 * x) if x is not None else "n/a"


def _fmt_census_reasons(census):
    br = (census or {}).get("by_reason") or {}
    parts = ["%s=%d" % (k, br[k]) for k in FAILURE_REASONS if br.get(k, 0) > 0]
    return ", ".join(parts) if parts else "none"


def agg_gauntlet(side):
    hp = sum(b["hp"] for b in side.values())
    ht = sum(b["ht"] for b in side.values())
    judge = sum(b["judge"] for b in side.values())
    empty = sum(b["empty"] for b in side.values())
    secs = sum(b["secs"] for b in side.values())
    n = sum(b["n"] for b in side.values())
    census = merge_failure_censuses(
        [b.get("failure_census") for b in side.values()]
    )
    return {
        "hp": hp,
        "ht": ht,
        "judge": judge,
        "empty": empty,
        "secs": secs,
        "n": n,
        "failure_census": census,
    }


def _detect_report_base_only(results, args):
    if getattr(args, "base_only", False):
        return True
    ft_url = getattr(args, "ft_url", None)
    base_url = getattr(args, "base_url", None)
    if base_url and (not ft_url or ft_url == base_url):
        return True
    ho = results.get("held-out")
    if ho and "FT" not in ho:
        return True
    for tag in ("trading-gauntlet", "gordon"):
        g = results.get(tag)
        if g and g.get("_base_only"):
            return True
        if g and g.get("BASE") and not g.get("FT"):
            return True
    return False


def report(results, args):
    min_cov = getattr(args, "min_coverage", MIN_RELIABLE_COVERAGE)
    base_only = _detect_report_base_only(results, args)

    print("\n" + "=" * 78)
    print("# BASE-ONLY — SCORECARD" if base_only else "# BASE vs FINE-TUNED — SCORECARD")
    print("=" * 78)
    verdict_lines = []

    # --- TARGET 1: held-out ---
    ho = results.get("held-out")
    if ho and "BASE" in ho:
        if base_only or "FT" not in ho:
            b = ho["BASE"]
            bc = b.get("failure_census") or new_failure_census()
            print("\n[1] HELD-OUT TRADING SET  (BASE-ONLY; all-case denominator)")
            print("  %-22s %12s" % ("metric", "BASE"))
            print("  %-22s %12s" % ("move_acc (all-case)", pct(b.get("move_acc"))))
            print("  %-22s %12s" % ("dir_acc (all-case)", pct(b.get("dir_acc"))))
            print("  %-22s %12s" % ("vol_change_acc (all)", pct(b.get("vol_change_acc"))))
            print("  %-22s %12s" % ("vs_iv_acc (all)", pct(b.get("vs_iv_acc"))))
            print("  %-22s %12s  (%d/%d)" % (
                "valid_coverage", pct(b.get("parse_rate")), bc["succeeded"], bc["attempted"]))
            if "json_parse_rate" in b:
                print("  %-22s %12s  (%d/%d, loose diag)" % (
                    "json_parsed (loose)", pct(b.get("json_parse_rate")),
                    b.get("json_parsed", 0), b.get("n", bc["attempted"])))
            print("  %-22s %12s  (cov %s)" % (
                "exp_move_mae",
                ("%.3f" % b["exp_move_mae"]) if b.get("exp_move_mae") is not None else "n/a",
                pct(b.get("exp_move_mae_coverage")),
            ))
            print("  conditional (parsed-only, NOT for ranking):")
            print("  %-22s %12s" % ("move_acc_parsed", pct(b.get("move_acc_parsed"))))
            print("  %-22s %12s" % ("dir_acc_parsed", pct(b.get("dir_acc_parsed"))))
            print("  failure census: attempted=%d succeeded=%d failed=%d (%s)" % (
                bc["attempted"], bc["succeeded"], bc["failed"], _fmt_census_reasons(bc)))
            if b.get("case_ids_sha256"):
                print("  case identity: n=%d case_ids_sha256=%s" % (
                    b.get("n", bc["attempted"]), b["case_ids_sha256"][:16]))
            if "ppl" in b and b["ppl"].get("loss") is not None:
                print("  %-22s %12.4f" % ("answer_loss", b["ppl"]["loss"]))
                print("  %-22s %12.3f" % ("answer_ppl", b["ppl"]["ppl"]))
            if bc["attempted"] == 0 or bc["succeeded"] == 0:
                verdict_lines.append("held-out (base-only): VERDICT WITHHELD — 0 valid parsed responses (%s)." % _fmt_census_reasons(bc))
            else:
                verdict_lines.append(
                    "held-out (base-only): move_acc=%s, dir_acc=%s, coverage=%d/%d (%s)."
                    % (pct(b.get("move_acc")), pct(b.get("dir_acc")),
                       bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"]))
                )
        else:
            b, f = ho["BASE"], ho["FT"]
            bc = b.get("failure_census") or new_failure_census()
            fc = f.get("failure_census") or new_failure_census()
            paired = ho.get("_paired") or {}
            print("\n[1] HELD-OUT TRADING SET  (headline = ALL-CASE full denominator; higher=better, lower MAE/PPL=better)")
            print("  %-22s %12s %12s %11s" % ("metric", "BASE", "FT", "Δ (FT-BASE)"))

            def line(name, key, fmt, better_high=True, scale=1.0, allow_arrow=True, ci_field=None):
                bv, fv = b.get(key), f.get(key)
                if bv is None or fv is None:
                    print("  %-22s %12s %12s %11s" % (
                        name,
                        (fmt % (bv * scale)) if bv is not None else "n/a",
                        (fmt % (fv * scale)) if fv is not None else "n/a",
                        "—",
                    ))
                    return None
                d = fv - bv
                arrow = ""
                if allow_arrow:
                    if (better_high and d > 0) or (not better_high and d < 0):
                        arrow = " ✓FT"
                    elif d != 0:
                        arrow = " ✗FT"
                ci_note = ""
                if ci_field and ci_field in paired:
                    pi = paired[ci_field]
                    ci_note = "  [95%%CI %+.1f..%+.1fpp]" % (100 * pi["ci_lo"], 100 * pi["ci_hi"])
                print("  %-22s %12s %12s %+10s%s%s" % (
                    name, fmt % (bv * scale), fmt % (fv * scale), fmt % (d * scale), arrow, ci_note))
                return d

            both_zero = (bc["succeeded"] == 0 and fc["succeeded"] == 0)
            d_move = line("move_acc (all-case)", "move_acc", "%.1f%%", True, 100, allow_arrow=not both_zero, ci_field="move")
            d_dir = line("dir_acc (all-case)", "dir_acc", "%.1f%%", True, 100, allow_arrow=not both_zero, ci_field="call")
            line("vol_change_acc (all)", "vol_change_acc", "%.1f%%", True, 100, allow_arrow=not both_zero, ci_field="vol_change")
            line("vs_iv_acc (all)", "vs_iv_acc", "%.1f%%", True, 100, allow_arrow=not both_zero, ci_field="vs_iv")
            line("valid_coverage", "parse_rate", "%.1f%%", True, 100, allow_arrow=not both_zero)
            if "json_parse_rate" in b or "json_parse_rate" in f:
                print("  %-22s %12s %12s  (loose diag — NOT for gating)" % (
                    "json_parsed (loose)", pct(b.get("json_parse_rate")), pct(f.get("json_parse_rate"))))

            # MAE arrow gated on comparable coverage so 1/10 lucky parse can't claim ✓FT
            mae_cov_ok = (
                (b.get("exp_move_mae_coverage") or 0.0) >= min_cov
                and (f.get("exp_move_mae_coverage") or 0.0) >= min_cov
            )
            d_mae = line("exp_move_mae", "exp_move_mae", "%.3f", False, 1.0, allow_arrow=mae_cov_ok)
            print("  %-22s %12s %12s" % (
                "exp_move_mae_cov", pct(b.get("exp_move_mae_coverage")), pct(f.get("exp_move_mae_coverage"))))

            # Conditional parsed-only (explicitly separated, NO ranking arrows)
            print("  conditional (parsed-only — NOT used for ranking):")
            print("  %-22s %12s %12s" % ("move_acc_parsed", pct(b.get("move_acc_parsed")), pct(f.get("move_acc_parsed"))))
            print("  %-22s %12s %12s" % ("dir_acc_parsed", pct(b.get("dir_acc_parsed")), pct(f.get("dir_acc_parsed"))))
            print("  failure census BASE: %d/%d succeeded (%s), failed=%d (%s)" % (
                bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"]), bc["failed"], _fmt_census_reasons(bc)))
            print("  failure census FT  : %d/%d succeeded (%s), failed=%d (%s)" % (
                fc["succeeded"], fc["attempted"], pct(fc["coverage_rate"]), fc["failed"], _fmt_census_reasons(fc)))
            paired_digest = paired.get("case_ids_sha256") or b.get("case_ids_sha256")
            if paired_digest:
                print("  case identity: n=%d case_ids_sha256=%s" % (
                    paired.get("shared_n", b.get("n", bc["attempted"])), paired_digest[:16]))

            # perplexity
            if "ppl" in b and "ppl" in f:
                bp, fp = b["ppl"], f["ppl"]
                bl, fl = bp.get("loss"), fp.get("loss")
                if bl is not None and fl is not None:
                    d = fl - bl
                    arrow = " ✓FT" if d < 0 else (" ✗FT" if d > 0 else "")
                    print("  %-22s %12.4f %12.4f %+10.4f%s" % ("answer_loss", bl, fl, d, arrow))
                    print("  %-22s %12.3f %12.3f %+10.3f%s" % (
                        "answer_ppl", bp["ppl"], fp["ppl"], fp["ppl"] - bp["ppl"], arrow))
                    if d < 0:
                        verdict_lines.append("held-out: FT assigns LOWER loss to the true answer (better-fit).")

            if both_zero:
                verdict_lines.append(
                    "⚠ held-out: VERDICT WITHHELD — 0 valid parsed responses on both BASE and FT."
                )
            else:
                if fc["coverage_rate"] < min_cov:
                    verdict_lines.append(
                        "⚠ held-out RELIABILITY: FT valid coverage is %d/%d (%s < %s gate; %s)."
                        % (fc["succeeded"], fc["attempted"], pct(fc["coverage_rate"]),
                           pct(min_cov), _fmt_census_reasons(fc))
                    )
                if bc["coverage_rate"] < min_cov:
                    verdict_lines.append(
                        "⚠ held-out RELIABILITY: BASE valid coverage is %d/%d (%s < %s gate; %s)."
                        % (bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"]),
                           pct(min_cov), _fmt_census_reasons(bc))
                    )
                if d_move is not None:
                    if d_move > 0 and fc["coverage_rate"] >= min_cov and bc["coverage_rate"] >= min_cov:
                        verdict_lines.append("held-out: FT move-category all-case accuracy +%.1fpp." % (d_move * 100))
                    elif d_move < 0:
                        verdict_lines.append("⚠ held-out REGRESSION: FT move-category all-case accuracy %.1fpp." % (d_move * 100))
                if d_dir is not None:
                    if d_dir > 0 and fc["coverage_rate"] >= min_cov and bc["coverage_rate"] >= min_cov:
                        verdict_lines.append("held-out: FT directional all-case accuracy +%.1fpp." % (d_dir * 100))
                    elif d_dir < 0:
                        verdict_lines.append("⚠ held-out REGRESSION: FT directional all-case accuracy %.1fpp." % (d_dir * 100))
                if d_mae is not None and d_mae < 0 and mae_cov_ok:
                    verdict_lines.append("held-out: FT exp_move_pct MAE improved by %.3f." % (-d_mae))

    # --- TARGET 1b: train/serve skew ---
    sk = results.get("skew")
    if ho and sk:
        side = "FT" if (ho.get("FT") and sk.get("FT")) else "BASE"
        b, s = ho.get(side), sk.get(side)
        if b and s:
            print("\n[1b] TRAIN/SERVE SKEW  (%s model: bare vol prompt vs PRODUCTION prompt)" % side)
            print("  metric scored on the SAME held-out rows (all-case); Δ<0 = degradation under prod context")
            print("  %-22s %12s %12s %11s" % ("metric", "baseline", "skew(prod)", "Δ (skew-base)"))

            def skline(name, key, fmt="%.1f%%", better_high=True, scale=100.0):
                bv, sv = b.get(key), s.get(key)
                if bv is None or sv is None:
                    print("  %-22s %12s %12s %11s" % (name, "n/a", "n/a", "—"))
                    return None
                d = sv - bv
                arrow = ""
                if (better_high and d > 0) or (not better_high and d < 0):
                    arrow = " ✓prod"
                elif d != 0:
                    arrow = " ⚠skew"
                print("  %-22s %12s %12s %+10s%s" % (
                    name, fmt % (bv * scale), fmt % (sv * scale), fmt % (d * scale), arrow))
                return d

            d_pm = skline("move_acc (all-case)", "move_acc")
            skline("dir_acc (all-case)", "dir_acc")
            skline("vol_change_acc (all)", "vol_change_acc")
            skline("vs_iv_acc (all)", "vs_iv_acc")
            skline("parse_rate (coverage)", "parse_rate")
            d_pmae = skline("exp_move_mae", "exp_move_mae", "%.3f", better_high=False, scale=1.0)
            bc = b.get("failure_census") or new_failure_census()
            sc = s.get("failure_census") or new_failure_census()
            if bc["coverage_rate"] < min_cov or sc["coverage_rate"] < min_cov:
                verdict_lines.append(
                    "⚠ skew (%s): VERDICT WITHHELD — insufficient valid coverage (base=%s, prod=%s)."
                    % (side, pct(bc["coverage_rate"]), pct(sc["coverage_rate"]))
                )
            else:
                worst = []
                if d_pm is not None and d_pm < -0.02:
                    worst.append("move_acc %.1fpp" % (d_pm * 100))
                if d_pmae is not None and d_pmae > 0.0:
                    worst.append("exp_move MAE +%.3f" % d_pmae)
                if worst:
                    verdict_lines.append("⚠ skew (%s): vol skill degrades under production prompt — %s." % (
                        side, ", ".join(worst)))
                else:
                    verdict_lines.append("skew (%s): vol skill holds under the production prompt (no major drop)." % side)
    elif sk and not ho:
        print("\n[1b] TRAIN/SERVE SKEW  (production prompt; no bare-prompt baseline in this run)")
        for sd in ("BASE", "FT"):
            r = sk.get(sd)
            if r:
                print("  %-4s move_acc(all)=%.1f%% dir_acc(all)=%.1f%% vol=%.1f%% vs_iv=%.1f%% parse=%.1f%%" % (
                    sd, 100 * r["move_acc"], 100 * r["dir_acc"], 100 * r["vol_change_acc"],
                    100 * r["vs_iv_acc"], 100 * r["parse_rate"]))

    # --- TARGET 2 & 3: gauntlets ---
    for tag, title in [("trading-gauntlet", "[2] TRADING GAUNTLET (trading profile)"),
                       ("gordon", "[3] GORDON GAUNTLET (general-ability regression)")]:
        g = results.get(tag)
        if g is None:
            continue
        print("\n%s" % title)
        if not g.get("_batteries") or not g.get("BASE"):
            print("  SKIP — no batteries evaluated for this target.")
            if tag == "gordon":
                verdict_lines.append("⚠ gordon: VERDICT WITHHELD — no general-ability batteries evaluated.")
            continue

        ab = agg_gauntlet(g["BASE"])
        bc = ab["failure_census"]
        target_base_only = base_only or g.get("_base_only") or not g.get("FT")

        if target_base_only:
            print("  %-20s %16s" % ("", "BASE"))
            print("  %-20s %16s" % (
                "objective-checks",
                "%d/%d (%s)" % (ab["hp"], ab["ht"], pct(ab["hp"] / ab["ht"]) if ab["ht"] else "n/a"),
            ))
            print("  %-20s %16s" % (
                "valid coverage",
                "%d/%d (%s)" % (bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"])),
            ))
            print("  %-20s %16d" % ("empty responses", ab["empty"]))
            print("  %-20s %16d" % ("judge-needed items", ab["judge"]))
            print("  %-20s %15.0fs" % ("total latency", ab["secs"]))
            print("  failure census BASE: failed=%d (%s)" % (bc["failed"], _fmt_census_reasons(bc)))
            print("  per-battery objective (BASE):")
            for bn in g["_batteries"]:
                if bn in g["BASE"]:
                    rb = g["BASE"][bn]
                    rfc = rb["failure_census"]
                    print("    %-18s %d/%d  cov=%d/%d (%s)  (judge %d)  [base-only]" % (
                        bn, rb["hp"], rb["ht"], rfc["succeeded"], rfc["attempted"],
                        pct(rfc["coverage_rate"]), rb["judge"]))
            if bc["attempted"] == 0 or bc["succeeded"] == 0 or bc["coverage_rate"] < min_cov:
                verdict_lines.append(
                    "⚠ %s (base-only): VERDICT WITHHELD — insufficient valid coverage (%d/%d = %s; %s)."
                    % (tag, bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"]), _fmt_census_reasons(bc))
                )
            elif ab["ht"]:
                verdict_lines.append(
                    "%s (base-only): objective %d/%d (%s), coverage %d/%d (%s)."
                    % (tag, ab["hp"], ab["ht"], pct(ab["hp"] / ab["ht"]),
                       bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"]))
                )
            if ab["judge"]:
                verdict_lines.append("%s: %d items need a judge — grade IN-SESSION or with a LOCAL judge model." % (tag, ab["judge"]))
            continue

        af = agg_gauntlet(g["FT"])
        fc = af["failure_census"]
        print("  %-20s %16s %16s" % ("", "BASE", "FT"))
        print("  %-20s %16s %16s" % (
            "objective-checks",
            "%d/%d (%s)" % (ab["hp"], ab["ht"], pct(ab["hp"] / ab["ht"]) if ab["ht"] else "n/a"),
            "%d/%d (%s)" % (af["hp"], af["ht"], pct(af["hp"] / af["ht"]) if af["ht"] else "n/a")))
        print("  %-20s %16s %16s" % (
            "valid coverage",
            "%d/%d (%s)" % (bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"])),
            "%d/%d (%s)" % (fc["succeeded"], fc["attempted"], pct(fc["coverage_rate"]))))
        print("  %-20s %16d %16d" % ("empty responses", ab["empty"], af["empty"]))
        print("  %-20s %16d %16d" % ("judge-needed items", ab["judge"], af["judge"]))
        print("  %-20s %15.0fs %15.0fs" % ("total latency", ab["secs"], af["secs"]))
        print("  failure census BASE: failed=%d (%s)" % (bc["failed"], _fmt_census_reasons(bc)))
        print("  failure census FT  : failed=%d (%s)" % (fc["failed"], _fmt_census_reasons(fc)))
        print("  per-battery objective (BASE -> FT):")
        for bn in g["_batteries"]:
            if bn in g["BASE"]:
                rb = g["BASE"][bn]
                rf = g["FT"].get(bn)
                if rf:
                    print("    %-18s %d/%d -> %d/%d   (judge %d)" % (
                        bn, rb["hp"], rb["ht"], rf["hp"], rf["ht"], rf["judge"]))
                else:
                    print("    %-18s %d/%d   (judge %d)  [base-only]" % (
                        bn, rb["hp"], rb["ht"], rb["judge"]))

        # Reliability-gated verdict signal
        cov_ok = (
            bc["attempted"] > 0 and fc["attempted"] > 0
            and bc["succeeded"] > 0 and fc["succeeded"] > 0
            and bc["coverage_rate"] >= min_cov and fc["coverage_rate"] >= min_cov
        )
        if not cov_ok:
            verdict_lines.append(
                "⚠ %s: VERDICT WITHHELD — insufficient valid coverage "
                "(BASE %d/%d=%s [%s], FT %d/%d=%s [%s]; gate >=%s)."
                % (tag, bc["succeeded"], bc["attempted"], pct(bc["coverage_rate"]), _fmt_census_reasons(bc),
                   fc["succeeded"], fc["attempted"], pct(fc["coverage_rate"]), _fmt_census_reasons(fc),
                   pct(min_cov))
            )
        elif ab["ht"] and af["ht"]:
            bb, ff = ab["hp"] / ab["ht"], af["hp"] / af["ht"]
            if tag == "trading-gauntlet":
                if ff > bb:
                    verdict_lines.append("trading-gauntlet: FT objective pass-rate %s -> %s." % (pct(bb), pct(ff)))
                elif ff < bb:
                    verdict_lines.append("⚠ trading-gauntlet REGRESSION: FT objective pass-rate dropped %s -> %s." % (pct(bb), pct(ff)))
                else:
                    verdict_lines.append("trading-gauntlet: objective pass-rate tied at %s." % pct(ff))
            if tag == "gordon":
                drop = bb - ff
                if drop > 0.05:
                    verdict_lines.append(
                        "⚠ gordon REGRESSION: general pass-rate dropped %s -> %s (%.1fpp). Inspect."
                        % (pct(bb), pct(ff), drop * 100)
                    )
                else:
                    verdict_lines.append(
                        "gordon: general ability held (%s -> %s) — no major regression." % (pct(bb), pct(ff))
                    )
        if af["judge"]:
            verdict_lines.append("%s: %d items need a judge — grade IN-SESSION or with a LOCAL judge model." % (tag, af["judge"]))

    # --- CONTRACT & BLIND JUDGE RECEIPTS ---
    if results.get("_contracts"):
        print("\n[CONTRACT] Hash-bound (SHA-256) run verification:")
        for side, rec in results["_contracts"].items():
            print("  %-4s run_id=%s profile=%s verified=%s manifest_sha256=%s" % (
                side, rec["run_id"], rec["profile"], rec["verified"], rec["manifest_sha256"][:16]))
    if results.get("_blind_judge"):
        bj = results["_blind_judge"]
        print("\n[BLIND JUDGE] Exported %d judge item(s) -> %s" % (bj["count"], bj["pending_path"] or "none"))

    # --- VERDICT ---
    print("\n" + "=" * 78)
    print("# VERDICT")
    print("=" * 78)
    if not verdict_lines:
        print("  Inconclusive — no comparable metrics produced (check that URLs served and files exist).")
    for ln in verdict_lines:
        print("  • " + ln)
    print("\n  NOTE: llm_judge / ib_grader items are NOT auto-scored. To complete the")
    print("  subjective trading-quality call (trader.json is judge-heavy), export")
    print("  blind packets with --blind-judge-dir and grade in-session or with a")
    print("  LOCAL judge model. Saved answers: %s" % args.out)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Base-vs-fine-tuned trading-LLM eval")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible /v1/chat/completions of the BASE model")
    ap.add_argument("--ft-url", default=None,
                    help="OpenAI-compatible /v1/chat/completions of the FINE-TUNED model "
                         "(omit or pass same URL as --base-url for base-only mode)")
    ap.add_argument("--base-only", action="store_true",
                    help="Explicitly run BASE-only evaluation and skip FT queries")
    ap.add_argument("--mode", default="all", choices=["held-out", "trading-gauntlet", "gordon", "all"])
    ap.add_argument("--n", type=int, default=300, help="held-out sample size (rows)")
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT, help="path to heldout_trading.jsonl")
    ap.add_argument("--skew", action="store_true",
                    help="ALSO run the train/serve-skew slice: same held-out rows under the REAL "
                         "production strategist system prompt (default OFF; additive)")
    ap.add_argument("--exitmgr-dir", default=None, dest="exitmgr_dir",
                    help="path to exitmgr-app (sources the live SYSTEM_PROMPT for --skew); "
                         "defaults to $EXITMGR_DIR or ~/exitmgr-app")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--move-max-tokens", type=int, default=160, dest="move_max_tokens",
                    help="max_tokens for held-out JSON answers")
    ap.add_argument("--min-coverage", type=float, default=MIN_RELIABLE_COVERAGE, dest="min_coverage",
                    help="minimum valid response coverage (0..1) required to issue parity/improvement verdicts")
    ap.add_argument("--system", default=None, help="optional system-prompt file for the gauntlet batteries")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "eval_finetuned_results.json"))
    ap.add_argument("--contract-dir", default=None, dest="contract_dir",
                    help="optional dir to write and verify hash-bound gauntlet_contract run/result/completion JSONs")
    ap.add_argument("--blind-judge-dir", default=None, dest="blind_judge_dir",
                    help="optional dir to export blind pending_judge.jsonl + judge_uid_map.json via judge_blind.py")
    ap.add_argument("--verbose", action="store_true")
    # perplexity (held-out, optional local forward pass)
    ap.add_argument("--hf-base", default=None, help="path/name of base model weights (HF id or local dir) to enable token-loss/ppl")
    ap.add_argument("--hf-ft", default=None, help="path to a LoRA/PEFT adapter dir for the fine-tuned model's ppl")
    ap.add_argument("--ppl-n", type=int, default=100, dest="ppl_n", help="rows for perplexity forward pass")
    args = ap.parse_args(argv)

    base_only = _is_base_only(args.base_url, args.ft_url, args)
    ft_url = args.base_url if base_only else args.ft_url

    sysdefault = None
    sysdefault_path = None
    if args.system and os.path.exists(args.system):
        sysdefault_path = os.path.realpath(args.system)
        with open(sysdefault_path, encoding="utf-8") as f:
            sysdefault = f.read()

    trading_bats = get_trading_batteries(GG_ROOT)
    general_bats = get_general_batteries(GG_ROOT, trading_batteries=trading_bats)

    planned_gauntlet_bats = []
    if args.mode in ("trading-gauntlet", "all"):
        planned_gauntlet_bats.extend(trading_bats)
    if args.mode in ("gordon", "all"):
        planned_gauntlet_bats.extend(general_bats)
    planned_gauntlet_bats = sorted(set(planned_gauntlet_bats))

    # Pre-flight validation BEFORE any network / model calls:
    # duplicate explicit case IDs or battery item IDs must fail closed immediately.
    if args.mode in ("held-out", "all") and os.path.exists(args.heldout):
        _load_and_validate_heldout(args.heldout, args.n)
    preloaded_batteries = {}
    for bn in planned_gauntlet_bats:
        items, b_sha = load_battery_frozen(bn)
        if items is not None:
            preloaded_batteries[bn] = {"items": items, "sha256": b_sha}

    detected_base = detect_model(args.base_url)
    detected_ft = None if base_only else detect_model(ft_url)
    model_by_side = {"BASE": detected_base or "BASE"}
    if not base_only:
        model_by_side["FT"] = detected_ft or "FT"

    print("BASE url:", args.base_url, "| model:", detected_base)
    if base_only:
        print("FT   url: (base-only mode — skipped)")
    else:
        print("FT   url:", ft_url, "| model:", detected_ft)
    print("Gauntlet root:", GG_ROOT)

    # Freeze battery bytes (and run contracts if --contract-dir) BEFORE any model calls
    prepared_contracts = None
    frozen_batteries = {}
    if args.contract_dir and planned_gauntlet_bats:
        prepared_contracts = _prepare_frozen_contracts(
            planned_gauntlet_bats, model_by_side, args, sysdefault_path=sysdefault_path
        )
        frozen_batteries = prepared_contracts["frozen_batteries"]
        if prepared_contracts["frozen_system_text"] is not None:
            sysdefault = prepared_contracts["frozen_system_text"]
    else:
        frozen_batteries = preloaded_batteries

    results = {}
    if args.mode in ("held-out", "all"):
        print("\n### TARGET 1: HELD-OUT TRADING SET")
        results["held-out"] = run_heldout(
            args.base_url, ft_url, args.n, args, model_by_side=model_by_side
        )
        if args.skew:
            print("\n### TARGET 1b: TRAIN/SERVE SKEW (held-out rows under PRODUCTION prompt)")
            results["skew"] = run_skew(
                args.base_url, ft_url, args.n, args, model_by_side=model_by_side
            )
    if args.mode in ("trading-gauntlet", "all"):
        print("\n### TARGET 2: TRADING GAUNTLET")
        results["trading-gauntlet"] = run_gauntlet(
            args.base_url, ft_url, trading_bats, args, sysdefault,
            frozen_batteries=frozen_batteries, model_by_side=model_by_side,
        )
    if args.mode in ("gordon", "all"):
        print("\n### TARGET 3: GORDON GAUNTLET (general regression — disjoint from trading)")
        results["gordon"] = run_gauntlet(
            args.base_url, ft_url, general_bats, args, sysdefault,
            frozen_batteries=frozen_batteries, model_by_side=model_by_side,
        )

    if prepared_contracts is not None:
        receipts = _finalize_verified_contracts(prepared_contracts, results)
        if receipts:
            results["_contracts"] = receipts

    if args.blind_judge_dir:
        results["_blind_judge"] = _write_blind_judge_packets(
            results, args, frozen_batteries=frozen_batteries
        )

    case_identity = {}
    for target in ("held-out", "skew"):
        t_res = results.get(target)
        if isinstance(t_res, dict) and isinstance(t_res.get("_provenance"), dict):
            prov = t_res["_provenance"]
            if prov.get("case_ids_sha256"):
                case_identity[target] = {
                    "case_count": prov.get("case_count"),
                    "case_ids_sha256": prov["case_ids_sha256"],
                }
    for target in ("trading-gauntlet", "gordon"):
        t_res = results.get(target)
        if isinstance(t_res, dict) and t_res.get("_battery_case_ids_sha256"):
            case_identity[target] = {
                "case_ids_sha256": t_res["_case_ids_sha256"],
                "batteries": dict(t_res["_battery_case_ids_sha256"]),
            }
    results["_manifest"] = {
        "mode": args.mode,
        "base_only": base_only,
        "case_ids_sha256": _canonical_sha256(case_identity) if case_identity else None,
        "targets": case_identity,
    }

    # save raw answers for in-session / local-judge grading of llm_judge items
    out_parent = os.path.dirname(os.path.abspath(args.out))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nsaved raw answers ->", args.out)

    report(results, args)
    return results


if __name__ == "__main__":
    main()
