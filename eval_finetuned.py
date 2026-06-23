#!/usr/bin/env python3
"""
eval_finetuned.py — Is a fine-tuned trading LLM a BETTER TRADER than its base?

Runs BASE vs FINE-TUNED head-to-head across three targets and prints a single
comparison table + a verdict:

  1. held-out      Held-out trading set (batteries/heldout_example.jsonl by default).
                   Objective: move-category accuracy, exp_move_pct MAE, direction
                   accuracy, % parseable JSON. Plus token-level loss/perplexity on
                   the ground-truth assistant answer (cleaner base-vs-FT signal that
                   doesn't depend on JSON parsing) — requires --hf-base/--hf-ft.
  2. trading-gauntlet  The trading batteries (trader/trademath/tickers).
                   Objective checks scored locally (equals_number, contains_none,
                   min_items, non_empty, ...). llm_judge items are run but flagged
                   "NEEDS JUDGE" — graded in-session or by a LOCAL judge, NEVER the
                   a paid hosted judge (grade those locally / in-session).
  3. gordon        Full Gordon Gauntlet regression — confirm the fine-tune did NOT
                   degrade general ability (reasoning/tools/refusal/persona/...).
                   Same objective-first scoring; judge items flagged.

Both BASE and FT must be queryable at OpenAI-compatible /v1/chat/completions
endpoints. The eval just points at two URLs.

  python eval_finetuned.py \
      --base-url http://127.0.0.1:8082/v1/chat/completions \
      --ft-url   http://127.0.0.1:8090/v1/chat/completions \
      --mode all --n 300

Modes: held-out | trading-gauntlet | gordon | all
The perplexity sub-metric (held-out) is optional and only runs if you pass
--hf-base / --hf-ft (paths to base weights + LoRA adapter for a local forward
pass). Without them the held-out target still runs the generative metrics.

Stdlib-only for the endpoint path. Perplexity path lazily imports torch/peft/
transformers and degrades gracefully if they're missing.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------- paths
HOME = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Held-out trading set: override with --heldout or the HELDOUT_PATH env var.
# Defaults to a synthetic example shipped in batteries/ so the harness runs OOTB.
DEFAULT_HELDOUT = os.environ.get(
    "HELDOUT_PATH",
    os.path.join(SCRIPT_DIR, "batteries", "heldout_example.jsonl"))
# Harness root: env override first (GG_ROOT), else the directory this script
# lives in (it ships its own batteries/ subdir).
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
TRADING_BATTERIES = ["trader", "trademath", "tickers"]


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


def chat_text(endpoint, messages, **kw):
    """Convenience: returns (text, completion_tokens, finish_reason, tool_calls, err)."""
    try:
        d = chat(endpoint, messages, **kw)
        m = d["choices"][0]["message"]
        return (m.get("content") or "",
                (d.get("usage") or {}).get("completion_tokens", 0),
                d["choices"][0].get("finish_reason"),
                bool(m.get("tool_calls")), None)
    except Exception as e:
        return ("", 0, None, False, str(e)[:200])


# ----------------------------------------------------------------------------- JSON parsing helpers (held-out)
def extract_json(text):
    """Pull the last {...} object out of a model response (handles ```json fences,
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
                return json.loads(o)
            except Exception:
                continue
        # fallback: greedy
        i, j = c.find("{"), c.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(c[i:j + 1])
            except Exception:
                pass
    return None


MOVE_CATS = {"QUIET", "NORMAL", "ELEVATED", "EXPLOSIVE"}


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
    if s.startswith("EXPAND") or s in ("RISING", "UP", "INCREASING"): return "EXPANDING"
    if s.startswith("CONTRACT") or s in ("FALLING", "DOWN", "DECREASING"): return "CONTRACTING"
    if s.startswith("STABLE") or s in ("FLAT", "STEADY", "UNCHANGED"): return "STABLE"
    if s.startswith("RICH") or s in ("EXPENSIVE", "OVERPRICED", "HIGH"): return "RICH"
    if s.startswith("CHEAP") or s in ("UNDERPRICED", "LOW"): return "CHEAP"
    if s.startswith("FAIR") or s in ("NEUTRAL",): return "FAIR"
    return s.split()[0] if s else s


def to_float(v):
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(v)))
    except Exception:
        return None


# ----------------------------------------------------------------------------- TARGET 1: held-out
def run_heldout(base_url, ft_url, n, args):
    path = args.heldout
    if not os.path.exists(path):
        print("  [held-out] SKIP — %s not found (training not finished yet?)" % path)
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("  [held-out] SKIP — file empty")
        return None
    if n and n < len(rows):
        # deterministic stride sample for reproducibility
        step = max(1, len(rows) // n)
        rows = rows[::step][:n]
    print("  [held-out] %d rows from %s" % (len(rows), path))

    out = {}
    for label, url in ([("BASE", base_url)] + ([("FT", ft_url)] if ft_url != base_url else [])):
        model = detect_model(url) or label
        moves_ok = dir_ok = parsed = total = 0
        vol_ok = vsiv_ok = 0
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
            mm = ([{"role": "system", "content": system}] if system else []) + \
                 [{"role": "user", "content": user}]
            txt, _, _, _, err = chat_text(url, mm, max_tokens=args.move_max_tokens,
                                          temperature=0.0, model=model,
                                          timeout=args.timeout)
            total += 1
            pred = extract_json(txt)
            if pred is None:
                continue
            parsed += 1
            # move category
            if norm_move(pred.get("move")) == norm_move(gt.get("move")) and \
               norm_move(gt.get("move")) in MOVE_CATS:
                moves_ok += 1
            # exp_move_pct MAE
            pv, gv = to_float(pred.get("exp_move_pct")), to_float(gt.get("exp_move_pct"))
            if pv is not None and gv is not None:
                maes.append(abs(pv - gv))
            # direction (call)
            call_ok = (norm_dir(pred.get("call")) == norm_dir(gt.get("call")) and gt.get("call") is not None)
            if call_ok:
                dir_ok += 1
            # vol_change regime (expanding/stable/contracting)
            if norm_cat(pred.get("vol_change")) == norm_cat(gt.get("vol_change")) and gt.get("vol_change"):
                vol_ok += 1
            # vs_iv (rich/fair/cheap)
            if norm_cat(pred.get("vs_iv")) == norm_cat(gt.get("vs_iv")) and gt.get("vs_iv"):
                vsiv_ok += 1
            # conviction calibration (model conviction vs call correctness)
            cv = to_float(pred.get("conviction"))
            if cv is not None:
                calib.append((cv, bool(call_ok)))
            preds.append({"pred": {k: pred.get(k) for k in ("move","call","vol_change","vs_iv","exp_move_pct","conviction")},
                          "gt": {k: gt.get(k) for k in ("move","call","vol_change","vs_iv","exp_move_pct","conviction")}})
            if (i + 1) % 50 == 0:
                print("    [%s] %d/%d  parse=%.0f%% move=%.0f%%" % (
                    label, i + 1, len(rows), 100 * parsed / total,
                    100 * moves_ok / max(1, parsed)))
        secs = time.time() - t0
        def _bucket(lo, hi):
            sub = [ok for c, ok in calib if lo <= c <= hi]
            return {"n": len(sub), "acc": (sum(sub) / len(sub)) if sub else None}
        calibration = {"low_1_4": _bucket(1, 4), "mid_5_7": _bucket(5, 7), "high_8_10": _bucket(8, 10)}
        out[label] = {
            "n": total,
            "parse_rate": parsed / total if total else 0.0,
            "move_acc": moves_ok / parsed if parsed else 0.0,
            "dir_acc": dir_ok / parsed if parsed else 0.0,
            "exp_move_mae": statistics.mean(maes) if maes else None,
            "vol_change_acc": vol_ok / parsed if parsed else 0.0,
            "vs_iv_acc": vsiv_ok / parsed if parsed else 0.0,
            "calibration": calibration,
            "rows": preds,
            "secs": round(secs, 1),
            "model": model,
        }
        print("  [held-out:%s] parse=%.1f%% move_acc=%.1f%% dir_acc=%.1f%% mae=%s (%.0fs)" % (
            label, 100 * out[label]["parse_rate"], 100 * out[label]["move_acc"],
            100 * out[label]["dir_acc"],
            ("%.3f" % out[label]["exp_move_mae"]) if out[label]["exp_move_mae"] is not None else "n/a",
            secs))
        calib_str = " ".join("%s=%s" % (k, ("%.0f%%" % (100 * calibration[k]["acc"]) if calibration[k]["acc"] is not None else "n/a")) for k in ("low_1_4", "mid_5_7", "high_8_10"))
        print("  [held-out:%s] vol_change_acc=%.1f%% vs_iv_acc=%.1f%% | call-acc by conviction: %s" % (
            label, 100 * out[label]["vol_change_acc"], 100 * out[label]["vs_iv_acc"], calib_str))
        # majority-baseline + balanced-accuracy (skew-robust) per categorical field
        def _field_metrics(field, normfn):
            pairs = [(normfn(pr["gt"].get(field)), normfn(pr["pred"].get(field))) for pr in preds]
            pairs = [(g, p) for g, p in pairs if g]
            if not pairs:
                return None
            from collections import Counter as _C
            gtc = _C(g for g, p in pairs); nn = len(pairs)
            raw = sum(1 for g, p in pairs if g == p) / nn
            maj = max(gtc.values()) / nn
            bal = sum(sum(1 for g, p in pairs if g == c and p == c) / gtc[c] for c in gtc) / len(gtc)
            return {"raw": raw, "majority": maj, "balanced": bal, "lift": raw - maj}
        fm = {f: _field_metrics(f, nf) for f, nf in
              (("move", norm_move), ("vol_change", norm_cat), ("vs_iv", norm_cat), ("call", norm_dir))}
        out[label]["field_metrics"] = fm
        print("  [held-out:%s]  field        raw  / major / balncd / lift" % label)
        for _f, _m in fm.items():
            if _m:
                print("                 %-11s %5.1f%% /%5.1f%% /%5.1f%% / %+5.1fpp" % (
                    _f, 100 * _m["raw"], 100 * _m["majority"], 100 * _m["balanced"], 100 * _m["lift"]))

    # --- optional: token-level loss / perplexity on the assistant answer ---
    ppl = None
    if args.hf_base and args.hf_ft:
        ppl = heldout_perplexity(rows, args)
    else:
        print("  [held-out:ppl] SKIP token-loss/perplexity — pass --hf-base and --hf-ft to enable")
    if ppl:
        out["BASE"]["ppl"] = ppl["BASE"]
        out["FT"]["ppl"] = ppl["FT"]
    return out


def heldout_perplexity(rows, args):
    """Mean token-level loss + perplexity on the GROUND-TRUTH assistant answer,
    computed by a local forward pass. Base = plain HF model; FT = base + LoRA
    adapter (peft). This is the cleanest base-vs-FT signal (no JSON parsing,
    directly measures whether the FT model assigns higher probability to the
    desired answer). Lazily imported; degrades to None on missing deps."""
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
        """Mean NLL over only the assistant-answer tokens, using the chat template
        to build the prompt and masking the prompt portion out of the labels."""
        losses = []
        with torch.no_grad():
            for row in sample:
                msgs = row["messages"]
                # prompt = everything up to (not including) the assistant turn
                prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
                answer = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                try:
                    prompt_ids = tok.apply_chat_template(
                        prompt_msgs, add_generation_prompt=True, return_tensors="pt")
                except Exception:
                    # fallback: simple concatenation if no chat template
                    text = "".join(m["content"] for m in prompt_msgs)
                    prompt_ids = tok(text, return_tensors="pt").input_ids
                ans_ids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids
                input_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(model.device)
                labels = input_ids.clone()
                labels[:, :prompt_ids.shape[1]] = -100  # mask the prompt
                out = model(input_ids=input_ids, labels=labels)
                if out.loss is not None and not math.isnan(float(out.loss)):
                    losses.append(float(out.loss))
        return statistics.mean(losses) if losses else None

    # cap perplexity sample (forward passes are expensive)
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
# Self-contained objective check engine: reproduces the harness's free
# auto-checks exactly. llm_judge / ib_grader
# are NOT auto-scored here (flagged for in-session / local-judge grading).
def run_check(c, text, tool_calls=False):
    """Returns (label, passed_bool_or_None). None => needs a judge (not auto-scorable)."""
    t = text or ""
    tl = t.lower()
    typ = c["type"]
    if typ == "non_empty":
        return ("non_empty", len(t.strip()) > 0)
    if typ == "contains_none":
        hits = [v for v in c["values"] if v.lower() in tl]
        return ("no-blocklist", len(hits) == 0)
    if typ == "contains_all":
        miss = [v for v in c["values"] if v.lower() not in tl]
        return ("recall", len(miss) == 0)
    if typ == "min_items":
        n = len([ln for ln in t.splitlines() if ln.strip() and re.search(r"[A-Z]{1,5}", ln)])
        return ("items>=%d" % c["n"], n >= c["n"])
    if typ == "equals_number":
        _n = re.sub(r"(?<=\d)[,\s](?=\d)", "", t)
        return ("==%d" % c["value"], str(c["value"]) in set(re.findall(r"\d+", _n)))
    if typ == "max_sentences":
        n = len([s for s in re.split(r"[.!?]+", t.strip()) if s.strip()])
        return ("<=%dsent" % c["n"], n <= c["n"])
    if typ == "regex_any":
        return (c.get("label", "regex_any"), any(re.search(p, t, re.I) for p in c["patterns"]))
    if typ == "regex_none":
        return (c.get("label", "regex_none"), not any(re.search(p, t, re.I) for p in c["patterns"]))
    if typ == "tool_intent":
        nm = c.get("name", "")
        nmn = nm.lower().replace("-", "").replace("_", "")
        norm = tl.replace("-", "").replace("_", "")
        markers = ["invoke name", "tool_call", "<tool", "function"]
        hit = bool(tool_calls) or (nmn and nmn in norm) or any(re.search(pp, t, re.I) for pp in markers)
        return ("tool:" + nm, hit)
    if typ in ("llm_judge", "ib_grader"):
        return (typ, None)  # needs a judge — flagged, not auto-scored
    return (typ, None)


def score_battery(label, url, battery, system_default, args):
    """Run one battery's prompts/turns against a URL, score objective checks.
    Returns per-battery aggregate: passed, total (auto), judge_needed, empty, secs."""
    model = detect_model(url) or label
    hp = ht = judge_needed = empty = 0
    secs = 0.0
    rows = []
    for it in battery:
        s = it.get("system", system_default)
        base_msgs = [{"role": "system", "content": s}] if s else []
        mt = it.get("max_tokens", 500)
        thinking = it.get("thinking")
        tools = it.get("tools")
        text = ""
        tcalls = False
        t0 = time.time()
        if it.get("turns"):
            msgs = list(base_msgs)
            for u in it["turns"]:
                msgs.append({"role": "user", "content": u})
                txt, _, _, tc, err = chat_text(url, msgs, max_tokens=mt, temperature=0.0,
                                               thinking=thinking, tools=tools, model=model,
                                               timeout=args.timeout)
                msgs.append({"role": "assistant", "content": txt})
                text, tcalls = txt, tcalls or tc
        else:
            msgs = base_msgs + [{"role": "user", "content": it["prompt"]}]
            text, _, _, tcalls, err = chat_text(url, msgs, max_tokens=mt, temperature=0.0,
                                                thinking=thinking, tools=tools, model=model,
                                                timeout=args.timeout)
        dt = time.time() - t0
        secs += dt
        if not (text or "").strip():
            empty += 1
        marks = []
        for c in it.get("checks", []):
            nm, ok = run_check(c, text, tcalls)
            if ok is None:
                judge_needed += 1
                marks.append("%s:JUDGE" % nm)
            else:
                ht += 1
                hp += 1 if ok else 0
                marks.append("%s:%s" % (nm, "OK" if ok else "X"))
        rows.append({"id": it["id"], "secs": round(dt, 1), "text": text, "marks": marks})
        if args.verbose:
            print("    [%s] %-22s %5.1fs  %s" % (label, it["id"], dt, "  ".join(marks)))
    return {"hp": hp, "ht": ht, "judge": judge_needed, "empty": empty,
            "secs": round(secs, 1), "n": len(battery), "rows": rows, "model": model}


def load_battery(name):
    path = os.path.join(BATT_DIR, name + ".json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def run_gauntlet(base_url, ft_url, battery_names, args, system_default=None):
    """Generic: run a list of batteries base-vs-FT, return per-target aggregate."""
    out = {"BASE": {}, "FT": {}, "_batteries": battery_names}
    for bn in battery_names:
        b = load_battery(bn)
        if b is None:
            print("  [%s] SKIP battery '%s' — not found in %s" % (args.mode, bn, BATT_DIR))
            continue
        print("  -- battery %s (%d items) --" % (bn, len(b)))
        for label, url in ([("BASE", base_url)] + ([("FT", ft_url)] if ft_url != base_url else [])):
            out[label][bn] = score_battery(label, url, b, system_default, args)
            r = out[label][bn]
            print("     [%s] %s: hard %d/%d  judge-needed=%d  empty=%d  %.0fs" % (
                label, bn, r["hp"], r["ht"], r["judge"], r["empty"], r["secs"]))
    return out


# ----------------------------------------------------------------------------- reporting
def pct(x):
    return "%.1f%%" % (100 * x) if x is not None else "n/a"


def agg_gauntlet(side):
    hp = sum(b["hp"] for b in side.values())
    ht = sum(b["ht"] for b in side.values())
    judge = sum(b["judge"] for b in side.values())
    empty = sum(b["empty"] for b in side.values())
    secs = sum(b["secs"] for b in side.values())
    n = sum(b["n"] for b in side.values())
    return {"hp": hp, "ht": ht, "judge": judge, "empty": empty, "secs": secs, "n": n}


def report(results, args):
    print("\n" + "=" * 78)
    print("# BASE vs FINE-TUNED — SCORECARD")
    print("=" * 78)
    verdict_lines = []

    # --- TARGET 1: held-out ---
    ho = results.get("held-out")
    if ho:
        print("\n[1] HELD-OUT TRADING SET  (objective; higher=better, lower MAE/PPL=better)")
        print("  %-16s %12s %12s %10s" % ("metric", "BASE", "FT", "Δ (FT-BASE)"))
        def line(name, key, fmt, better_high=True, scale=1.0):
            bv, fv = ho["BASE"].get(key), ho["FT"].get(key)
            if bv is None or fv is None:
                print("  %-16s %12s %12s %10s" % (name, "n/a", "n/a", "—"))
                return None
            d = fv - bv
            arrow = ""
            if (better_high and d > 0) or (not better_high and d < 0):
                arrow = " ✓FT"
            elif d != 0:
                arrow = " ✗FT"
            print("  %-16s %12s %12s %+9s%s" % (
                name, fmt % (bv * scale), fmt % (fv * scale), fmt % (d * scale), arrow))
            return d
        d_move = line("move_acc", "move_acc", "%.1f%%", True, 100)
        d_dir = line("dir_acc", "dir_acc", "%.1f%%", True, 100)
        line("parse_rate", "parse_rate", "%.1f%%", True, 100)
        d_mae = line("exp_move_mae", "exp_move_mae", "%.3f", False)
        # perplexity
        if "ppl" in ho["BASE"] and "ppl" in ho["FT"]:
            bp, fp = ho["BASE"]["ppl"], ho["FT"]["ppl"]
            bl, fl = bp.get("loss"), fp.get("loss")
            if bl is not None and fl is not None:
                d = fl - bl
                arrow = " ✓FT" if d < 0 else (" ✗FT" if d > 0 else "")
                print("  %-16s %12.4f %12.4f %+9.4f%s" % ("answer_loss", bl, fl, d, arrow))
                print("  %-16s %12.3f %12.3f %+9.3f%s" % (
                    "answer_ppl", bp["ppl"], fp["ppl"], fp["ppl"] - bp["ppl"], arrow))
                if d < 0:
                    verdict_lines.append("held-out: FT assigns LOWER loss to the true answer (better-fit).")
        if d_move is not None and d_move > 0:
            verdict_lines.append("held-out: FT move-category accuracy +%.1fpp." % (d_move * 100))
        if d_mae is not None and d_mae < 0:
            verdict_lines.append("held-out: FT exp_move_pct MAE improved by %.3f." % (-d_mae))

    # --- TARGET 2 & 3: gauntlets ---
    for tag, title in [("trading-gauntlet", "[2] TRADING GAUNTLET (trader/trademath/tickers)"),
                       ("gordon", "[3] GORDON GAUNTLET (general-ability regression)")]:
        g = results.get(tag)
        if not g:
            continue
        ab, af = agg_gauntlet(g["BASE"]), agg_gauntlet(g["FT"])
        print("\n%s" % title)
        print("  %-18s %14s %14s" % ("", "BASE", "FT"))
        print("  %-18s %14s %14s" % (
            "objective-checks",
            "%d/%d (%s)" % (ab["hp"], ab["ht"], pct(ab["hp"] / ab["ht"]) if ab["ht"] else "n/a"),
            "%d/%d (%s)" % (af["hp"], af["ht"], pct(af["hp"] / af["ht"]) if af["ht"] else "n/a")))
        print("  %-18s %14d %14d" % ("empty responses", ab["empty"], af["empty"]))
        print("  %-18s %14d %14d" % ("judge-needed items", ab["judge"], af["judge"]))
        print("  %-18s %13.0fs %13.0fs" % ("total latency", ab["secs"], af["secs"]))
        # per-battery breakdown
        print("  per-battery objective (BASE -> FT):")
        for bn in g["_batteries"]:
            if bn in g["BASE"]:
                rb = g["BASE"][bn]; rf = g["FT"].get(bn)
                if rf:
                    print("    %-12s %d/%d -> %d/%d   (judge %d)" % (
                        bn, rb["hp"], rb["ht"], rf["hp"], rf["ht"], rf["judge"]))
                else:
                    print("    %-12s %d/%d   (judge %d)  [base-only]" % (bn, rb["hp"], rb["ht"], rb["judge"]))
        # verdict signal
        if ab["ht"] and af["ht"]:
            bb, ff = ab["hp"] / ab["ht"], af["hp"] / af["ht"]
            if tag == "trading-gauntlet" and ff > bb:
                verdict_lines.append("trading-gauntlet: FT objective pass-rate %s -> %s." % (pct(bb), pct(ff)))
            if tag == "gordon":
                drop = bb - ff
                if drop > 0.05:
                    verdict_lines.append("⚠ gordon REGRESSION: general pass-rate dropped %s -> %s (%.1fpp). Inspect." % (pct(bb), pct(ff), drop * 100))
                else:
                    verdict_lines.append("gordon: general ability held (%s -> %s) — no major regression." % (pct(bb), pct(ff)))
        if af["judge"]:
            verdict_lines.append("%s: %d items need a judge — grade IN-SESSION or with a LOCAL judge model." % (tag, af["judge"]))

    # --- VERDICT ---
    print("\n" + "=" * 78)
    print("# VERDICT")
    print("=" * 78)
    if not verdict_lines:
        print("  Inconclusive — no comparable metrics produced (check that both URLs served and files exist).")
    for ln in verdict_lines:
        print("  • " + ln)
    print("\n  NOTE: llm_judge / ib_grader items are NOT auto-scored. To complete the")
    print("  subjective trading-quality call (trader.json is judge-heavy), pipe the")
    print("  saved answers to an in-session grade or a LOCAL judge — do NOT use the")
    print("  a paid hosted judge. Saved answers: %s" % args.out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Base-vs-fine-tuned trading-LLM eval")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible /v1/chat/completions of the BASE model")
    ap.add_argument("--ft-url", required=True, help="OpenAI-compatible /v1/chat/completions of the FINE-TUNED model")
    ap.add_argument("--mode", default="all", choices=["held-out", "trading-gauntlet", "gordon", "all"])
    ap.add_argument("--n", type=int, default=300, help="held-out sample size (rows)")
    ap.add_argument("--heldout", default=DEFAULT_HELDOUT, help="path to heldout_trading.jsonl")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--move-max-tokens", type=int, default=160, dest="move_max_tokens",
                    help="max_tokens for held-out JSON answers")
    ap.add_argument("--system", default=None, help="optional system-prompt file for the gauntlet batteries")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "eval_finetuned_results.json"))
    ap.add_argument("--verbose", action="store_true")
    # perplexity (held-out, optional local forward pass)
    ap.add_argument("--hf-base", default=None, help="path/name of base model weights (HF id or local dir) to enable token-loss/ppl")
    ap.add_argument("--hf-ft", default=None, help="path to a LoRA/PEFT adapter dir for the fine-tuned model\'s ppl")
    ap.add_argument("--ppl-n", type=int, default=100, dest="ppl_n", help="rows for perplexity forward pass")
    args = ap.parse_args()

    sysdefault = open(args.system).read() if args.system and os.path.exists(args.system) else None

    print("BASE url:", args.base_url, "| model:", detect_model(args.base_url))
    print("FT   url:", args.ft_url, "| model:", detect_model(args.ft_url))
    print("Gauntlet root:", GG_ROOT)

    results = {}
    if args.mode in ("held-out", "all"):
        print("\n### TARGET 1: HELD-OUT TRADING SET")
        results["held-out"] = run_heldout(args.base_url, args.ft_url, args.n, args)
    if args.mode in ("trading-gauntlet", "all"):
        print("\n### TARGET 2: TRADING GAUNTLET")
        results["trading-gauntlet"] = run_gauntlet(args.base_url, args.ft_url, TRADING_BATTERIES, args, sysdefault)
    if args.mode in ("gordon", "all"):
        print("\n### TARGET 3: GORDON GAUNTLET (general regression)")
        all_bats = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(BATT_DIR, "*.json"))
                          if not p.endswith(".bak") and ".bak-" not in os.path.basename(p))
        # exclude ib_essay by default (heavy 15-min grader) unless --verbose flagged heavy
        all_bats = [b for b in all_bats if b != "ib_essay"]
        results["gordon"] = run_gauntlet(args.base_url, args.ft_url, all_bats, args, sysdefault)

    # save raw answers for in-session / local-judge grading of llm_judge items
    json.dump(results, open(args.out, "w"), indent=1, default=str)
    print("\nsaved raw answers ->", args.out)

    report(results, args)


if __name__ == "__main__":
    main()
