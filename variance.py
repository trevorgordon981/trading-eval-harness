#!/usr/bin/env python3
"""Reliability/variance + promotion NOISE BAND.

Two modes:
  1. reliability (default): run a battery N times for a model, report blank-rate +
     error-rate per prompt. Quantifies M3's occasional-blank weakness. Single-turn only.
     Usage: variance.py --endpoint URL --battery F --label L --n 3 [--system P --model ID ...]

  2. band (fix 4): bootstrap a 95% CI on the candidate-vs-baseline pass-rate DELTA so the
     promotion pipeline (eval_and_score.sh) only PROMOTEs when the improvement clears
     sampling noise. Prints the minimum detectable effect (MDE).
     Usage: variance.py band --cand PASS/N --base PASS/N [--alpha 0.05]
     Emits a machine-readable line:  BAND delta=.. lo=.. hi=.. mde=.. verdict=IMPROVE|TIE|REGRESS"""
import json, os, sys, time, random, argparse, urllib.request


def bootstrap_delta_ci(cand_pass, cand_n, base_pass, base_n, iters=10000, alpha=0.05, seed=1234):
    """Bootstrap CI on (cand_rate - base_rate), each item a Bernoulli pass/fail.
    Returns (delta, lo, hi, mde). mde = half-width of the delta CI = the smallest
    |delta| that clears the noise band at this N."""
    if cand_n <= 0 or base_n <= 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = random.Random(seed)
    cr = cand_pass / cand_n
    br = base_pass / base_n
    use_binom = hasattr(rng, "binomialvariate")
    def draw(n, p):
        if use_binom:
            return rng.binomialvariate(n, min(1.0, max(0.0, p)))
        return sum(1 for _ in range(n) if rng.random() < p)
    deltas = sorted((draw(cand_n, cr) / cand_n) - (draw(base_n, br) / base_n) for _ in range(iters))
    lo = deltas[int(alpha / 2 * iters)]
    hi = deltas[min(iters - 1, int((1 - alpha / 2) * iters))]
    return cr - br, lo, hi, (hi - lo) / 2.0


def paired_bootstrap_delta_ci(candidate, baseline, iters=10000, alpha=0.05,
                              seed=1234):
    """Bootstrap paired per-item pass deltas; preserves shared prompt difficulty."""
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired outcomes must be non-empty and equal length")
    if any(type(value) is not bool for value in list(candidate) + list(baseline)):
        raise ValueError("paired outcomes must be booleans")
    diffs = [int(c) - int(b) for c, b in zip(candidate, baseline)]
    n = len(diffs); rng = random.Random(seed)
    draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(iters))
    lo = draws[int(alpha / 2 * iters)]
    hi = draws[min(iters - 1, int((1 - alpha / 2) * iters))]
    delta = sum(diffs) / n
    return delta, lo, hi, (hi - lo) / 2.0


def paired_cluster_bootstrap_delta_ci(candidate, baseline, clusters,
                                      iters=10000, alpha=0.05, seed=1234):
    """Bootstrap prompt/item clusters while retaining every check in a cluster."""
    if len(candidate) != len(baseline) or len(candidate) != len(clusters) \
            or not candidate:
        raise ValueError("clustered outcomes must be non-empty and equal length")
    if any(type(value) is not bool for value in list(candidate) + list(baseline)):
        raise ValueError("paired outcomes must be booleans")
    if any(type(cluster) is not str or not cluster for cluster in clusters):
        raise ValueError("cluster IDs must be non-empty strings")
    grouped = {}
    for index, cluster in enumerate(clusters):
        grouped.setdefault(cluster, []).append(index)
    names = sorted(grouped)
    if len(names) < 2:
        raise ValueError("cluster bootstrap requires at least two independent items")

    observed = sum(int(c) - int(b) for c, b in zip(candidate, baseline)) / len(candidate)
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        chosen = [names[rng.randrange(len(names))] for _ in names]
        indices = [index for name in chosen for index in grouped[name]]
        draws.append(sum(int(candidate[index]) - int(baseline[index])
                         for index in indices) / len(indices))
    draws.sort()
    lo = draws[int(alpha / 2 * iters)]
    hi = draws[min(iters - 1, int((1 - alpha / 2) * iters))]
    return observed, lo, hi, (hi - lo) / 2.0


def _pn(s):
    p, n = s.split("/"); return int(p), int(n)


def cmd_band(argv):
    ap = argparse.ArgumentParser(prog="variance.py band")
    ap.add_argument("--cand", required=True, help="candidate PASS/N (non-trivial checks)")
    ap.add_argument("--base", required=True, help="baseline PASS/N")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=10000)
    a = ap.parse_args(argv)
    cp, cn = _pn(a.cand); bp, bn = _pn(a.base)
    delta, lo, hi, mde = bootstrap_delta_ci(cp, cn, bp, bn, iters=a.iters, alpha=a.alpha)
    # IMPROVE only if the whole CI is above 0; REGRESS if whole CI below 0; else TIE (within band).
    verdict = "IMPROVE" if lo > 0 else ("REGRESS" if hi < 0 else "TIE")
    print("[noise-band] candidate %d/%d (%.3f) vs baseline %d/%d (%.3f)  delta=%+.3f  %d%%CI=[%+.3f,%+.3f]"
          % (cp, cn, cp / cn if cn else 0, bp, bn, bp / bn if bn else 0, delta,
             int((1 - a.alpha) * 100), lo, hi))
    print("[noise-band] minimum detectable effect (MDE, half-width) = %.3f (%.1f%% of the %d-item scale)"
          % (mde, 100 * mde, cn))
    print("BAND delta=%.4f lo=%.4f hi=%.4f mde=%.4f verdict=%s" % (delta, lo, hi, mde, verdict))

def detect_model(ep):
    base = ep.rsplit("/chat/completions", 1)[0]
    try: return json.loads(urllib.request.urlopen(base + "/models", timeout=10).read().decode())["data"][0]["id"]
    except Exception: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--battery", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--system", default=None); ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default=None); ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()
    model = a.model or detect_model(a.endpoint) or "x"
    key = os.getenv(a.api_key_env) if a.api_key_env else None
    sysp = open(a.system).read() if a.system else None
    battery = [it for it in json.load(open(a.battery)) if "turns" not in it]
    print("[variance] %s x%d on %s (%d prompts)" % (a.label, a.n, a.battery, len(battery)), flush=True)
    tot_runs = 0; tot_blank = 0; tot_err = 0
    for it in battery:
        blank = 0; err = 0
        for _ in range(a.n):
            msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": it["prompt"]}]
            body = {"model": it.get("model", model), "messages": msgs, "max_tokens": it.get("max_tokens", 500), "temperature": 0}
            if it.get("tools"): body["tools"] = it["tools"]
            hdr = {"Content-Type": "application/json"}
            if key: hdr["Authorization"] = "Bearer " + key
            try:
                with urllib.request.urlopen(urllib.request.Request(a.endpoint, data=json.dumps(body).encode(), headers=hdr), timeout=a.timeout) as r:
                    d = json.loads(r.read().decode(), strict=False)
                txt = (d["choices"][0]["message"].get("content") or "")
                if not txt.strip(): blank += 1
            except Exception:
                err += 1
            tot_runs += 1
        tot_blank += blank; tot_err += err
        flag = "  <-- BLANKS" if blank else ""
        print("  %-22s blank %d/%d  err %d/%d%s" % (it["id"], blank, a.n, err, a.n, flag), flush=True)
    print("[variance] %s: blank-rate %d/%d (%.0f%%), error-rate %d/%d" % (
        a.label, tot_blank, tot_runs, 100.0 * tot_blank / tot_runs if tot_runs else 0, tot_err, tot_runs), flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "band":
        cmd_band(sys.argv[2:])
    else:
        main()
