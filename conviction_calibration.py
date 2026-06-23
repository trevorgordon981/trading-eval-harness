#!/usr/bin/env python3
"""conviction_calibration.py  --  Does conviction earn its risk?

The ONLY question this answers: do higher-conviction options trades actually
WIN MORE and MAKE MORE MONEY in realized P&L?  That is the single gate on
whether the exitmgr bot is allowed a steep conviction -> position-size curve
("double down when really sure").

This is a 100% READ-ONLY, stdlib-only analysis tool.  It places no orders,
modifies no journal, and does NOT contact the live model, the gateway, or
(by default) IBKR.  It reasons purely from on-disk data:

  * the trade journal           ~/exitmgr-app/trades.log    (JSONL, one entry/trade)
  * the audit log               ~/exitmgr-app/audit.jsonl   (for conviction, since
                                the journal itself does NOT persist conviction --
                                it is only carried on the live TradeIdea object and
                                logged to the `daily_rec_posted` audit events)

Realized P&L:
  The exitmgr journal records ENTRIES (cost basis, targets, legs).  Realized P&L
  on close is NOT written to any on-disk file in this app -- it lives only inside
  IBKR.  So a trade is "closed" here only if we are given a fills/realized-P&L
  source.  Two optional ways to feed that in WITHOUT this script touching IBKR:

    --fills FILE   JSON or JSONL mapping closing P&L to a trade.  Accepts a list
                   of objects or {conId: pnl}.  Recognized per-object keys:
                     contract_id / conId / conid   -> matches journal contract_id
                     realized_pnl / realizedPNL / pnl / realized  -> $ P&L
                     (optional) close_ts / closed_at / exit_ts   -> ISO close time
                   This is the clean hand-off: run `conviction_report.py --no-...`
                   or any IBKR dump elsewhere, drop the JSON here, stay read-only.

  With no fills source, every journaled trade is reported OPEN and the script
  states plainly that there is no realized data to calibrate on yet.

Usage:
  python3 conviction_calibration.py
  python3 conviction_calibration.py --journal ~/exitmgr-app/trades.log \
        --audit ~/exitmgr-app/audit.jsonl [--fills closed.json]

Sufficiency rule (deliberately conservative -- this gates REAL money sizing):
  A conviction->size curve is only justified when there are enough *closed*
  trades spread across the conviction range.  Defaults: >=30 closed trades AND
  >=8 closed in BOTH the high (8-10) and the low/mid (<8) groups.  Anything less
  is reported as INSUFFICIENT, with exactly how short we are.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_JOURNAL = os.path.expanduser("~/exitmgr-app/trades.log")
DEFAULT_AUDIT = os.path.expanduser("~/exitmgr-app/audit.jsonl")

# sufficiency thresholds (see module docstring)
MIN_CLOSED_TOTAL = 30
MIN_CLOSED_PER_GROUP = 8


# --------------------------------------------------------------------------- buckets
def bucket(conv):
    """System's sizing buckets.  None/<0 -> 'unknown' (conviction not recoverable)."""
    if conv is None:
        return "unknown"
    try:
        c = float(conv)
    except (TypeError, ValueError):
        return "unknown"
    if c < 0:
        return "unknown"
    if c >= 8:
        return "high (8-10)"
    if c >= 5:
        return "mid (5-7)"
    return "low (1-4)"


BUCKET_ORDER = ["high (8-10)", "mid (5-7)", "low (1-4)", "unknown"]


# --------------------------------------------------------------------------- loaders
def _iter_jsonl(path, label):
    if not os.path.exists(path):
        print(f"[WARN] {label} not found: {path}")
        return
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skipping unparseable {label} line {i}")


def load_journal(path):
    return list(_iter_jsonl(path, "journal"))


def _parse_ts(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def load_conviction_index(audit_path):
    """Build a lookup of conviction from `daily_rec_posted` audit events.

    Each such event has: underlying, conviction, order (a string that contains the
    strikes, e.g. 'BUY 1x MU 20260626 1185/1190C debit spread @ ...').  We index by
    (underlying, frozenset_of_strike_tokens) plus keep the timestamp, so a journal
    entry can be matched on symbol + its strike(s).  Multiple posts for the same
    symbol/strike keep the closest-in-time one (handled at match time).
    """
    posts = []
    for d in _iter_jsonl(audit_path, "audit"):
        if d.get("event") != "daily_rec_posted":
            continue
        conv = d.get("conviction")
        if conv is None:
            continue
        sym = d.get("underlying")
        order = d.get("order") or ""
        strikes = _strike_tokens(order)
        posts.append({
            "ts": _parse_ts(d.get("ts")),
            "symbol": sym,
            "strikes": strikes,
            "conviction": conv,
        })
    return posts


def _strike_tokens(s):
    """Extract numeric strike-like tokens from an order string.

    'BUY 1x MU 20260626 1185/1190C debit spread' -> {1185.0, 1190.0}
    Skips the expiry (8-digit date) and quantity tokens.
    """
    out = set()
    cleaned = s.replace("/", " ").replace("C", " ").replace("P", " ")
    for tok in cleaned.split():
        t = tok.strip("$x@()~,")
        try:
            v = float(t)
        except ValueError:
            continue
        # skip expiries (8-digit yyyymmdd) and obvious quantities
        if 1e7 <= v <= 9.9e7:
            continue
        out.add(v)
    return out


def match_conviction(entry, posts):
    """Recover an entry's conviction by matching symbol + long strike against posts.

    Returns (conviction_or_None, how_str).  Prefers an exact long-strike match;
    falls back to symbol-only if a single unambiguous post exists for that symbol.
    """
    sym = entry.get("symbol")
    long_strike = entry.get("strike")
    cands = [p for p in posts if p["symbol"] == sym]
    if not cands:
        return None, "no-audit-match"

    # exact: post strikes contain this entry's long strike
    strike_hits = [p for p in cands if long_strike in p["strikes"]]
    if strike_hits:
        # if several, take the post closest in time to the entry ts
        ent_ts = _parse_ts(entry.get("ts"))
        if ent_ts and any(p["ts"] for p in strike_hits):
            strike_hits.sort(key=lambda p: abs((p["ts"] - ent_ts).total_seconds())
                             if p["ts"] else 1e18)
        return strike_hits[0]["conviction"], "strike-match"

    # fallback: only one conviction value seen for this symbol -> use it
    convs = {p["conviction"] for p in cands}
    if len(convs) == 1:
        return next(iter(convs)), "symbol-only"
    return None, "ambiguous-symbol"


def load_fills(path):
    """Return {contract_id: {'pnl': float, 'close_ts': dt_or_None}} from a fills file.

    Tolerant of: a list of objects, or a flat {conId: pnl} dict, JSON or JSONL.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[WARN] fills file not found: {path}")
        return {}
    raw = open(path).read().strip()
    objs = []
    # try whole-file JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # flat {conId: pnl} ?
            flat = {}
            for k, v in data.items():
                try:
                    flat[int(k)] = {"pnl": float(v), "close_ts": None}
                except (ValueError, TypeError):
                    flat = None
                    break
            if flat is not None:
                return flat
            data = [data]
        if isinstance(data, list):
            objs = data
    except json.JSONDecodeError:
        # JSONL
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    out = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        cid = o.get("contract_id") or o.get("conId") or o.get("conid")
        pnl = (o.get("realized_pnl") if o.get("realized_pnl") is not None else
               o.get("realizedPNL") if o.get("realizedPNL") is not None else
               o.get("pnl") if o.get("pnl") is not None else o.get("realized"))
        if cid is None or pnl is None:
            continue
        try:
            cid = int(cid)
            pnl = float(pnl)
        except (ValueError, TypeError):
            continue
        ts = _parse_ts(o.get("close_ts") or o.get("closed_at") or o.get("exit_ts"))
        out[cid] = {"pnl": pnl, "close_ts": ts}
    return out


# --------------------------------------------------------------------------- core
def build_rows(entries, posts, fills):
    rows = []
    for e in entries:
        cid = e.get("contract_id")
        debit = e.get("debit")
        try:
            debit = float(debit) if debit is not None else None
        except (ValueError, TypeError):
            debit = None
        conv, how = match_conviction(e, posts)

        status, pnl_usd, pnl_pct, hold_days = "OPEN", None, None, None
        f = fills.get(cid)
        if f is not None:
            pnl_usd = f["pnl"]
            pnl_pct = (pnl_usd / debit * 100.0) if debit else None
            status = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "FLAT")
            ent_ts = _parse_ts(e.get("ts"))
            if ent_ts and f["close_ts"]:
                hold_days = (f["close_ts"] - ent_ts).total_seconds() / 86400.0

        rows.append({
            "ts": (e.get("ts") or "")[:16],
            "symbol": e.get("symbol", "?"),
            "right": e.get("right", "?"),
            "strike": e.get("strike", ""),
            "spread": "Y" if e.get("spread") else "",
            "conv": conv,
            "conv_src": how,
            "debit": debit,
            "status": status,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "hold_days": hold_days,
        })
    return rows


def aggregate(rows, key):
    agg = defaultdict(lambda: {"n": 0, "closed": 0, "wins": 0, "open": 0,
                               "pnl_pcts": [], "pnl_usd": 0.0, "holds": []})
    for r in rows:
        k = key(r)
        a = agg[k]
        a["n"] += 1
        if r["status"] == "OPEN":
            a["open"] += 1
        else:
            a["closed"] += 1
            if r["status"] == "WIN":
                a["wins"] += 1
            if r["pnl_pct"] is not None:
                a["pnl_pcts"].append(r["pnl_pct"])
            if r["pnl_usd"] is not None:
                a["pnl_usd"] += r["pnl_usd"]
            if r["hold_days"] is not None:
                a["holds"].append(r["hold_days"])
    return agg


def _fmt_pct(xs, fn=lambda xs: sum(xs) / len(xs)):
    return f"{fn(xs):+.0f}%" if xs else "-"


def print_table(title, agg, order):
    print(f"\n{title}\n")
    hdr = (f"{'group':16} {'N':>4} {'closed':>6} {'open':>5} {'win%':>6} "
           f"{'avgP&L%':>8} {'medP&L%':>8} {'total$':>9} {'avgHold':>8}")
    print(hdr)
    print("-" * len(hdr))
    for k in order:
        if k not in agg:
            continue
        a = agg[k]
        win = f"{a['wins'] / a['closed'] * 100:.0f}%" if a["closed"] else "-"
        avg = _fmt_pct(a["pnl_pcts"])
        med = _fmt_pct(a["pnl_pcts"], statistics.median)
        tot = f"{a['pnl_usd']:+,.0f}" if a["closed"] else "-"
        hold = f"{statistics.mean(a['holds']):.1f}d" if a["holds"] else "-"
        print(f"{str(k):16} {a['n']:>4} {a['closed']:>6} {a['open']:>5} "
              f"{win:>6} {avg:>8} {med:>8} {tot:>9} {hold:>8}")


def monotonic_read(agg, order):
    """Do win% and avg-P&L% rise as conviction rises?  Only over groups with closes."""
    seq = [(k, agg[k]) for k in order if k in agg and agg[k]["closed"] > 0]
    if len(seq) < 2:
        return "n/a (need closed trades in >=2 conviction groups)"
    # order is best-first; reverse to low->high for an 'increasing' read
    seq = list(reversed(seq))
    wins = [a["wins"] / a["closed"] * 100 for _, a in seq]
    avgs = [statistics.mean(a["pnl_pcts"]) if a["pnl_pcts"] else float("nan")
            for _, a in seq]
    win_up = all(b >= a for a, b in zip(wins, wins[1:]))
    avg_up = all((b >= a) for a, b in zip(avgs, avgs[1:])
                 if a == a and b == b)  # skip NaN
    parts = []
    parts.append("win-rate rises with conviction" if win_up
                 else "win-rate does NOT rise monotonically")
    parts.append("avg-P&L rises with conviction" if avg_up
                 else "avg-P&L does NOT rise monotonically")
    return "; ".join(parts)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", default=DEFAULT_JOURNAL)
    ap.add_argument("--audit", default=DEFAULT_AUDIT)
    ap.add_argument("--fills", default=None,
                    help="optional JSON/JSONL of realized P&L per contract_id")
    args = ap.parse_args()

    print("=" * 78)
    print("CONVICTION CALIBRATION  --  does high conviction earn its risk?")
    print("=" * 78)

    entries = load_journal(args.journal)
    posts = load_conviction_index(args.audit)
    fills = load_fills(args.fills)

    print(f"\nSources:")
    print(f"  journal : {args.journal}  ({len(entries)} entries)")
    print(f"  audit   : {args.audit}  ({len(posts)} daily_rec_posted convictions)")
    print(f"  fills   : {args.fills or '(none -- no realized P&L source on disk)'}"
          f"  ({len(fills)} closed)")

    if not entries:
        print("\nNo journal entries. Nothing to calibrate.")
        return 0

    rows = build_rows(entries, posts, fills)

    # ---- data state up front (the most important deliverable) -------------
    n_total = len(rows)
    n_closed = sum(1 for r in rows if r["status"] != "OPEN")
    n_open = n_total - n_closed
    conv_dist = defaultdict(int)
    conv_known = 0
    for r in rows:
        conv_dist[r["conv"] if r["conv"] is not None else "unknown"] += 1
        if r["conv"] is not None:
            conv_known += 1

    print("\n" + "-" * 78)
    print("RAW DATA STATE")
    print("-" * 78)
    print(f"  trades journaled (entries) : {n_total}")
    print(f"  CLOSED (have realized P&L) : {n_closed}")
    print(f"  still OPEN                  : {n_open}")
    print(f"  conviction recovered for   : {conv_known}/{n_total} "
          f"(from audit daily_rec_posted; journal itself stores no conviction)")
    print(f"  conviction distribution (all journaled entries):")
    for c in sorted(conv_dist, key=lambda x: (x == "unknown", -float(x) if x != "unknown" else 0)):
        print(f"      conviction {str(c):>8} : {conv_dist[c]}")

    # ---- per-trade table --------------------------------------------------
    print("\n" + "-" * 78)
    print("PER-TRADE")
    print("-" * 78)
    hdr = (f"{'date':16} {'sym':5} {'r':1} {'strike':>8} {'sp':2} {'conv':>4} "
           f"{'src':12} {'debit':>7} {'status':6} {'P&L$':>8} {'P&L%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pnl_usd = f"{r['pnl_usd']:+,.0f}" if r["pnl_usd"] is not None else "-"
        pnl_pct = f"{r['pnl_pct']:+.0f}%" if r["pnl_pct"] is not None else "-"
        debit = f"{r['debit']:,.0f}" if r["debit"] else "-"
        strike = (f"{r['strike']:g}" if isinstance(r["strike"], (int, float))
                  else str(r["strike"]))
        conv = str(r["conv"]) if r["conv"] is not None else "?"
        print(f"{r['ts']:16} {r['symbol']:5} {r['right']:1} {strike:>8} "
              f"{r['spread']:2} {conv:>4} {r['conv_src']:12} {debit:>7} "
              f"{r['status']:6} {pnl_usd:>8} {pnl_pct:>6}")

    # ---- calibration: per-conviction + by-bucket --------------------------
    by_conv = aggregate(rows, key=lambda r: (r["conv"] if r["conv"] is not None
                                             else "unknown"))
    conv_order = sorted([k for k in by_conv if k != "unknown"],
                        key=lambda x: -float(x)) + (["unknown"] if "unknown" in by_conv else [])
    print_table("CALIBRATION BY EXACT CONVICTION", by_conv, conv_order)

    by_bucket = aggregate(rows, key=lambda r: bucket(r["conv"]))
    print_table("CALIBRATION BY SIZING BUCKET (low 1-4 / mid 5-7 / high 8-10)",
                by_bucket, BUCKET_ORDER)

    # ---- monotonicity -----------------------------------------------------
    print("\n" + "-" * 78)
    print("MONOTONICITY (the property that would justify a steep size curve)")
    print("-" * 78)
    print("  " + monotonic_read(by_bucket, BUCKET_ORDER))

    # ---- verdict ----------------------------------------------------------
    hi = by_bucket.get("high (8-10)", {"closed": 0})
    rest_closed = sum(by_bucket.get(k, {"closed": 0})["closed"]
                      for k in ("mid (5-7)", "low (1-4)", "unknown"))
    sufficient = (n_closed >= MIN_CLOSED_TOTAL
                  and hi["closed"] >= MIN_CLOSED_PER_GROUP
                  and rest_closed >= MIN_CLOSED_PER_GROUP)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if n_closed == 0:
        print("  INSUFFICIENT DATA -- 0 closed trades with realized P&L on disk.")
        print("  Conviction-vs-outcome CANNOT be measured yet: nothing has been")
        print("  recorded as closed.  Realized P&L for this app lives only inside")
        print("  IBKR; it is not written to any journal/audit file.  Do NOT enable a")
        print("  steep conviction->size curve on zero realized outcomes.")
    elif not sufficient:
        need_total = max(0, MIN_CLOSED_TOTAL - n_closed)
        need_hi = max(0, MIN_CLOSED_PER_GROUP - hi["closed"])
        need_rest = max(0, MIN_CLOSED_PER_GROUP - rest_closed)
        print(f"  INSUFFICIENT SAMPLE -- N={n_closed} closed (need >={MIN_CLOSED_TOTAL}).")
        print(f"  high(8-10) closed={hi['closed']} (need >={MIN_CLOSED_PER_GROUP}); "
              f"rest closed={rest_closed} (need >={MIN_CLOSED_PER_GROUP}).")
        print(f"  Short by: {need_total} total, {need_hi} high-conviction, "
              f"{need_rest} lower-conviction closed trades.")
        print("  High conviction is NOT YET shown to earn its risk -- not enough")
        print("  realized outcomes.  Keep sizing conservative; re-run as trades close.")
    else:
        hi_win = hi["wins"] / hi["closed"] * 100
        hi_avg = (statistics.mean(by_bucket["high (8-10)"]["pnl_pcts"])
                  if by_bucket["high (8-10)"]["pnl_pcts"] else float("nan"))
        mono = monotonic_read(by_bucket, BUCKET_ORDER)
        earns = ("rises with conviction" in mono and hi_win >= 50)
        word = "IS" if earns else "IS NOT"
        print(f"  High conviction {word} earning its risk.  N={n_closed} closed; "
              f"high(8-10) win {hi_win:.0f}%, avg {hi_avg:+.0f}%.  {mono}.")

    print("\nRe-run this harness as trades close.  It is read-only and safe to run anytime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
