#!/usr/bin/env python3
"""Re-run the rag_grounding battery WITH a RAG pipeline: for each prompt, retrieve
from a document corpus (an HTTP /search endpoint) and inject it as context before the
model answers. Compares grounded answers to the bare-model baseline.

Env / CLI:
  --rag-url   document-search endpoint (default $RAG_URL or http://localhost:9000/search)
  --model-url OpenAI-compatible /v1/chat/completions (default $MODEL_URL or localhost:8000)
  --battery   path to the rag_grounding battery JSON
"""
import argparse, json, os, re, time, urllib.request

def search(rag_url, q, top_k=6):
    body = json.dumps({"query": q, "top_k": top_k}).encode()
    req = urllib.request.Request(rag_url, data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=25).read().decode())
    return r.get("results", [])

def chat(model_url, messages, max_tokens=600):
    body = json.dumps({"model": "local", "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(model_url, data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
    return r["choices"][0]["message"]["content"]

def run_check(c, text):
    t = c.get("type")
    if t == "non_empty":
        return bool((text or "").strip())
    if t == "regex_any":
        return any(re.search(p, text, re.I | re.S) for p in c.get("patterns", []))
    if t == "regex_none":
        return not any(re.search(p, text, re.I | re.S) for p in c.get("patterns", []))
    if t == "contains_all":
        subs = c.get("substrings") or c.get("patterns") or c.get("values") or []
        return all(str(s).lower() in (text or "").lower() for s in subs)
    return None  # llm_judge / ib_grader -> graded in-session

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-url", default=os.environ.get("RAG_URL", "http://localhost:9000/search"))
    ap.add_argument("--model-url", default=os.environ.get("MODEL_URL", "http://localhost:8000/v1/chat/completions"))
    ap.add_argument("--battery", default=os.path.join(here, "batteries", "rag_grounding.json"))
    args = ap.parse_args()

    batt = json.load(open(args.battery))
    items = batt if isinstance(batt, list) else batt.get("items", batt.get("prompts", []))
    hp = ht = judge = 0
    print("=== rag_grounding WITH RAG pipeline (/search grounding) ===\n")
    for it in items:
        prompt = it.get("prompt") or " ".join(it.get("turns", []))
        hits = search(args.rag_url, prompt, 6)
        ctx = "\n\n".join("[%s] %s" % (h.get("filename", "?"), h.get("text", "")) for h in hits)
        sys = ("You are answering using your personal knowledge base. Below is retrieved context "
               "from it — use it to answer specifically and accurately. If a fact is in the context, "
               "state it precisely.\n\n=== RETRIEVED CONTEXT ===\n" + ctx + "\n=== END CONTEXT ===")
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            ans = chat(args.model_url, msgs, it.get("max_tokens", 600))
        except Exception as e:
            ans = "<error: %s>" % e
        marks = []
        for c in it.get("checks", []):
            ok = run_check(c, ans)
            lbl = c.get("label", c.get("type"))
            if ok is None:
                judge += 1; marks.append("%s:JUDGE" % lbl)
            else:
                ht += 1; hp += 1 if ok else 0; marks.append("%s:%s" % (lbl, "OK" if ok else "X"))
        print("[%-22s] %4.0fs  hits=%d  %s" % (it.get("id", "?"), time.time() - t0, len(hits), "  ".join(marks)))
        print("    A: %s\n" % (ans or "").replace("\n", " ")[:240])
    print("=" * 70)
    print("WITH-RAG objective: %d/%d   judge-needed=%d" % (hp, ht, judge))

if __name__ == "__main__":
    main()
